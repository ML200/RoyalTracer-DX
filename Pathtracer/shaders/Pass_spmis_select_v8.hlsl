#define COMPUTE_PASS
#define SPMIS_GRID_NONCOHERENT
#include "Includes_v8.hlsli"

#define SP_SEARCH_R0    spmis_searchR0
#define SP_SEARCH_GROW  spmis_searchGrow
#define SP_SEARCH_ITERS spmis_searchIters

[numthreads(16, 16, 1)]
// Select spatial partners and canonical reservoirs for reuse.
void main(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= IMG_W || tid.y >= IMG_H) return;
    gDispatchIdx = tid;

    const float2 dims     = float2(IMG_W, IMG_H);
    const uint   pixelIdx = MapPixelID(dims, tid.xy);

    [unroll]
    for (uint dOff = 0u; dOff < SPMIS_TOTAL_ROLES; ++dOff)
        g_pathStateBuffer.Store(SPM_slotZ(pixelIdx, dOff), SP_UNDEF);

    if (!SPMIS_SPATIAL_MODE) return;

    uint hdrFlags, hdrMatID; float cpr, cpm;
    load_SD_header(g_sample_current, pixelIdx, hdrFlags, hdrMatID, cpr, cpm);

    if ((hdrFlags & SD_FLAG_EMITTER) != 0u)
    {
        g_pathStateBuffer.Store(SPM_w0(pixelIdx), SPM_packHdr(0u, SPM_STATUS_SKIP));
        return;
    }

    const uint4 recA = g_spmisBuffer.Load4(SP_SRCH(pixelIdx));
    const uint4 recB = g_spmisBuffer.Load4(SP_SRCH(pixelIdx) + 16u);
    const uint  cellCenter = recA.x;
    const uint  centerM    = load_M(g_Reservoirs_current, pixelIdx);

    if (centerM == 0u || cellCenter == SP_UNDEF)
    {
        g_pathStateBuffer.Store(SPM_w0(pixelIdx), SPM_packHdr(0u, SPM_STATUS_PASS));
        return;
    }

    if (!HYBRID_SHIFT_ON &&
        cpr < SPMIS_GLASS_ROUGHNESS_MIN &&
        LoadKd_w(hdrMatID) < 1.0f - EPSILON)
    {
        g_pathStateBuffer.Store(SPM_w0(pixelIdx), SPM_packHdr(0u, SPM_STATUS_PASS));
        return;
    }

    const float3 myPos  = asfloat(recA.yzw);
    const float3 myN    = asfloat(recB.yzw);
    const float3 camPos = InitOrigin();

    uint2 seed  = GetSeed(pixelIdx, time, 6);
    uint  seedA = Hash32(seed.x);
    uint  seedO = Hash32(seed.y);

    const uint Ntn = min(max(spmis_reuseN, 1u), SPMIS_SPLIT_MAXDRAWS);

    float weight_sum   = asfloat(recB.x);
    uint  selectedCell = cellCenter;
    float radius       = SP_SEARCH_R0;
    const float planeThresh = spmis_planeDist * length(myPos - camPos);

    [loop]
    for (uint b = 0u; b < SP_SEARCH_ITERS; b += 4u)
    {

        uint4 rA[4];
        uint4 rB[4];
        [unroll]
        for (uint j = 0u; j < 4u; ++j)
        {
            rA[j] = uint4(SP_UNDEF, 0u, 0u, 0u);
            rB[j] = uint4(0u, 0u, 0u, 0u);
            if (b + j >= SP_SEARCH_ITERS) continue;

            int2 off = int2(round(radius * (RandomFloatPCG(seedO) * 2.0f - 1.0f)),
                            round(radius * (RandomFloatPCG(seedO) * 2.0f - 1.0f)));
            radius *= SP_SEARCH_GROW;
            int2 nc  = int2(tid.xy) + off;
            if (nc.x < 0) nc.x = -nc.x; else if (nc.x >= (int)IMG_W) nc.x = 2 * (int)IMG_W - nc.x - 1;
            if (nc.y < 0) nc.y = -nc.y; else if (nc.y >= (int)IMG_H) nc.y = 2 * (int)IMG_H - nc.y - 1;

            const uint npx = MapPixelID(dims, nc);
            if (npx != 0xFFFFFFFFu)
            {
                rA[j] = g_spmisBuffer.Load4(SP_SRCH(npx));
                rB[j] = g_spmisBuffer.Load4(SP_SRCH(npx) + 16u);
            }
        }

        [unroll]
        for (uint j2 = 0u; j2 < 4u; ++j2)
        {
            const uint ncell = rA[j2].x;
            if (ncell == SP_UNDEF || ncell == cellCenter) continue;

            if (spmis_normalSimCos > -1.0f)
            {
                const float3 nN = asfloat(rB[j2].yzw);
                if (dot(myN, nN) <= spmis_normalSimCos) continue;
            }

            const float3 nPos = asfloat(rA[j2].yzw);
            if (abs(dot(nPos - myPos, myN)) > planeThresh) continue;

            const float w = asfloat(rB[j2].x);
            weight_sum += w;
            if (RandomFloatPCG(seedA) < w / weight_sum) selectedCell = ncell;
        }
    }
    const uint reuseCell = selectedCell;

    const uint4 agg       = g_spmisBuffer.Load4(SP_AGG(reuseCell));
    const uint  cellBase  = agg.w;
    const float pixCount  = (float)agg.x;
    const uint  nzCount   = agg.y;
    const float scaling   = (SPMIS_CONF_ADJUST && pixCount > 0.0f) ? min(1.0f, (float)Ntn / pixCount) : 1.0f;
    const float neighbors_conf_sum = (float)agg.z * scaling;

    uint partnerPx = SP_UNDEF;
    if (neighbors_conf_sum > EPSILON && pixCount > 0.0f)
    {
        uint k = (uint)(pixCount * RandomFloatPCG(seedO));
        if (k >= (uint)pixCount) k = (uint)pixCount - 1u;
        partnerPx = g_spmisBuffer.Load(SP_A(SP_SORTED, cellBase + k));
    }

    uint canonRes = SP_UNDEF;
    const bool partnerOk = (partnerPx != SP_UNDEF) && neighbors_conf_sum > EPSILON && pixCount > 0.0f;
    if (partnerOk)
    {
        const Reservoir rdi = loadReservoir(g_Reservoirs_current, pixelIdx);
        const bool  canonReusable = !HYBRID_SHIFT_ON || RcReusable(rdi.rcInfo);
        const float p_c = GetPHat(rdi.F) * ((rdi.W > 0.0f) ? 1.0f : 0.0f);
        if (canonReusable && p_c > EPSILON)
        {
            canonRes = pixelIdx;

            if (!HYBRID_SHIFT_ON &&
                rdi.matID != MATID_LIGHT_TRI && rdi.matID != MATID_ENV_MISS &&
                !IsVolumeVertex(rdi.matID) && rdi.Pr < rs_reconnectRoughnessMin)
                canonRes |= SPM_KILLPH_BIT;
        }
    }

    g_pathStateBuffer.Store(SPM_w0(pixelIdx), SPM_packHdr(reuseCell, SPM_STATUS_NORM));

    g_pathStateBuffer.Store(SPM_slotS(pixelIdx, 0u), partnerPx & 0x7FFFFFFFu);
    g_pathStateBuffer.Store(SPM_slotZ(pixelIdx, 0u), canonRes);

    [loop]
    for (uint d = 0u; d < Ntn; ++d)
    {
        uint  outZ = SP_UNDEF;
        float outP = 0.0f;
        if (nzCount > 0u)
        {
            float ris_wsum = 0.0f, ris_selTF = 0.0f;
            uint  ris_selK = SP_UNDEF;
            [loop]
            for (uint r0 = 0u; r0 < spmis_risN; r0 += 4u)
            {
                uint  kArr[4];
                float tfArr[4];
                [unroll]
                for (uint j = 0u; j < 4u; ++j)
                {
                    kArr[j] = 0u; tfArr[j] = 0.0f;
                    if (r0 + j >= spmis_risN) continue;
                    uint k = (uint)((float)nzCount * RandomFloatPCG(seedO));
                    if (k >= nzCount) k = nzCount - 1u;
                    kArr[j]  = k;
                    tfArr[j] = asfloat(g_spmisBuffer.Load(SP_A(SP_SORTEDW, cellBase + k)));
                }
                [unroll]
                for (uint j2 = 0u; j2 < 4u; ++j2)
                {
                    if (r0 + j2 >= spmis_risN) continue;
                    const float w = tfArr[j2] * (float)nzCount / (float)spmis_risN;
                    ris_wsum += w;
                    if (ris_wsum > 0.0f && RandomFloatPCG(seedA) < w / ris_wsum)
                    { ris_selK = kArr[j2]; ris_selTF = tfArr[j2]; }
                }
            }
            if (ris_selK != SP_UNDEF && ris_selTF > 0.0f)
            {
                const uint zPx = g_spmisBuffer.Load(SP_A(SP_SORTED, cellBase + ris_selK));
                if (zPx != SP_UNDEF) { outZ = zPx; outP = ris_selTF / ris_wsum; }
            }
        }

        uint outRes = SP_UNDEF;
        if (outZ != SP_UNDEF)
        {
            const uint  zRcInfo = load_rcInfo(g_Reservoirs_current, outZ);
            const float zW      = load_W(g_Reservoirs_current, outZ);
            bool dead = (zW <= 0.0f);
            if (!dead)
            {
                if (HYBRID_SHIFT_ON)
                {
                    if (!RcReusable(zRcInfo))
                        dead = true;
                }
                else if (RcReplayLen(zRcInfo) > 0u)
                {
                    dead = true;
                }
                else
                {

                    const uint zMat = load_matID_res(g_Reservoirs_current, outZ);
                    if (zMat != MATID_LIGHT_TRI && zMat != MATID_ENV_MISS &&
                        !IsVolumeVertex(zMat))
                    {
                        float zEta, zPr, zPm;
                        UnpackEtaPrPm(g_Reservoirs_current.Load(addr_pay(outZ) + 8u),
                                      zEta, zPr, zPm);
                        if (zPr < rs_reconnectRoughnessMin)
                            dead = true;
                    }
                }
            }
            if (!dead)
                outRes = outZ;
        }

        g_pathStateBuffer.Store(SPM_slotS(pixelIdx, d + 1u), pixelIdx);
        g_pathStateBuffer.Store(SPM_slotZ(pixelIdx, d + 1u), outRes);
        g_pathStateBuffer.Store(SPM_slotP(pixelIdx, d + 1u), asuint(outP));
    }
}
