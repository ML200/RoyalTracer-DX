#define COMPUTE_PASS
#define SPMIS_GRID_NONCOHERENT
#include "Includes_v8.hlsli"

[numthreads(16, 16, 1)]
// Merge canonical and partner contributions with reuse MIS.
void main(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= IMG_W || tid.y >= IMG_H) return;
    gDispatchIdx = tid;

    const float2 dims     = float2(IMG_W, IMG_H);
    const uint   pixelIdx = MapPixelID(dims, tid.xy);

    if (!SPMIS_SPATIAL_MODE)
    {
        Reservoir rp = loadReservoir(g_Reservoirs_current, pixelIdx);
        if (!load_isEmitter(g_sample_current, pixelIdx))
        {
            const float Wp = (rp.W > 0.0f) ? rp.W : 0.0f;
            gScratchPing[uint3(tid.xy, 2)] = float4(rp.F * Wp, 0);
        }
        storeReservoir(g_Reservoirs_last, pixelIdx, rp);
        return;
    }

    const uint w0     = g_pathStateBuffer.Load(SPM_w0(pixelIdx));
    const uint status = SPM_hdrStatus(w0);

    if (status == SPM_STATUS_SKIP)
    {
        g_Reservoirs_last.Store(addr_v2(pixelIdx), PROBE_DI_NORMAL_ZERO_CODE);
        return;
    }

    Reservoir rdi = loadReservoir(g_Reservoirs_current, pixelIdx);

    if (status == SPM_STATUS_PASS)
    {
        const float  W    = (rdi.W > 0.0f) ? rdi.W : 0.0f;
        const float  vis  = asfloat(g_pathStateBuffer.Load(SPM_w1(pixelIdx)));
        const float3 outC = rdi.F * W * vis;
        gScratchPing[uint3(tid.xy, 2)] = float4(outC, 0);
        storeReservoir(g_Reservoirs_last, pixelIdx, rdi);
        return;
    }

    const uint  reuseCell = SPM_hdrCell(w0);
    const uint4 agg       = g_spmisBuffer.Load4(SP_AGG(reuseCell));
    const float pixCount  = (float)agg.x;
    const uint  Ntn       = min(max(spmis_reuseN, 1u), SPMIS_SPLIT_MAXDRAWS);
    const float scaling   = (SPMIS_CONF_ADJUST && pixCount > 0.0f) ? min(1.0f, (float)Ntn / pixCount) : 1.0f;
    const float neighbors_conf_sum = (float)agg.z * scaling;

    const float visReuse_c = (rdi.W > 0.0f) ? 1.0f : 0.0f;
    const float p_c        = GetPHat(rdi.F) * visReuse_c;
    const float centerConf = (float)rdi.M;

    SetSkyObserver(load_x1(g_sample_current, pixelIdx) + sceneOriginWorld);

    float mis_c = 1.0f;
    {
        const uint canonRes = g_pathStateBuffer.Load(SPM_slotZ(pixelIdx, 0u));
        if (canonRes != SP_UNDEF)
        {
            const uint  partnerPx = g_pathStateBuffer.Load(SPM_slotS(pixelIdx, 0u)) & 0x7FFFFFFFu;
            const uint2 j0        = g_pathStateBuffer.Load2(SPM_slotJ(pixelIdx, 0u));
            const float3 c0raw    = asfloat(g_pathStateBuffer.Load3(SPM_slotC(pixelIdx, 0u)));

            float  Jn_c;
            float3 c_canon;
            if (RcEnvReplay(rdi.rcInfo))
            {
                Jn_c    = 1.0f;
                c_canon = (GetPHat(c0raw) > 0.0f)
                    ? EnvTailFinish(c0raw, UnpackNormal(j0.x), asfloat(j0.y)) : (float3)0.0f;
            }
            else
            {
                Jn_c    = abs(asfloat(j0.y));
                c_canon = c0raw;
            }
            const float partnerConf = (float)load_M(g_Reservoirs_current, partnerPx) * scaling;

            float revT = 0.0f;
            if (!RcGeomReject(Jn_c, rdi.gBase, spmis_jacThreshold) && rdi.cachedJac > 0.0f)
                revT = GetPHat(c_canon) * Jn_c / rdi.cachedJac;

            mis_c = SpmisCanonicalMis(revT, p_c, centerConf, neighbors_conf_sum, partnerConf, pixCount);
        }
    }

    rdi.w_sum = mis_c * p_c * rdi.W;
    float3 contrib_final = rdi.F * visReuse_c;
    uint   effDraws      = 0u;
    uint   winnerZPx     = SP_UNDEF;
    uint   winnerD       = 0u;

    const float lumC0 = GetPHat(contrib_final);
    float3 rgbWsum = (lumC0 > 0.0f) ? rdi.w_sum * (contrib_final / lumC0) : (float3)0.0f;

    uint2 seed = GetSeed(pixelIdx, time, 9);
    seed.x = Hash32(seed.x);

    [loop]
    for (uint d = 0u; d < Ntn; ++d)
    {

        const uint slot = d + 1u;
        const uint resRaw = g_pathStateBuffer.Load(SPM_slotZ(pixelIdx, slot));
        if (resRaw == SP_UNDEF)
            continue;

        const uint zPx = resRaw & SPM_RESPX_MASK;

        Reservoir pr = (Reservoir)0;
        loadReservoirState(g_Reservoirs_current, zPx, pr);
        const uint prRcInfo = load_rcInfo(g_Reservoirs_current, zPx);
        const bool isEnv    = RcEnvReplay(prRcInfo);
        const bool needWalk = HYBRID_SHIFT_ON && RcReplayLen(prRcInfo) > 0u;

        const uint2  j8   = g_pathStateBuffer.Load2(SPM_slotJ(pixelIdx, slot));
        const float3 craw = asfloat(g_pathStateBuffer.Load3(SPM_slotC(pixelIdx, slot)));

        float  cachedNew, Jn;
        bool   preVisDead;
        float3 c;
        if (isEnv)
        {
            cachedNew = 1.0f; Jn = 1.0f; preVisDead = false;
            c = (GetPHat(craw) > 0.0f)
                ? EnvTailFinish(craw, UnpackNormal(j8.x), asfloat(j8.y)) : (float3)0.0f;
        }
        else
        {
            cachedNew            = asfloat(j8.x);
            const float JnSigned = asfloat(j8.y);
            Jn                   = abs(JnSigned);
            preVisDead           = JnSigned < 0.0f;
            c                    = craw;
        }

        const bool dead = needWalk
            ? (GetPHat(c) <= 0.0f || cachedNew <= 0.0f || RcGeomReject(Jn, pr.gBase, spmis_jacThreshold))
            : (preVisDead || cachedNew <= 0.0f || RcGeomReject(Jn, pr.gBase, spmis_jacThreshold));
        if (dead)
            continue;

        const float selProb    = asfloat(g_pathStateBuffer.Load(SPM_slotP(pixelIdx, slot)));
        const float c_i_scaled = (float)pr.M * scaling;
        const float w_draw = SpmisDrawWeight(GetPHat(pr.F), pr.cachedJac,
                                             GetPHat(c), Jn, selProb, Ntn,
                                             neighbors_conf_sum, centerConf, c_i_scaled) * pr.W;
        const float3 Fshift = c * Jn / cachedNew;

        rdi.w_sum += w_draw;
        effDraws++;
        const float lumCd = GetPHat(Fshift);
        if (w_draw > 0.0f && lumCd > 0.0f)
            rgbWsum += w_draw * (Fshift / lumCd);
        if (rdi.w_sum > 0.0f && RandomFloatPCG(seed.x) < w_draw / rdi.w_sum)
        {
            winnerZPx     = zPx;
            winnerD       = slot;
            contrib_final = Fshift;
        }
    }

    if (winnerZPx != SP_UNDEF)
    {
        loadReservoirPayload(g_Reservoirs_current, winnerZPx, rdi);
        if (RcEnvReplay(rdi.rcInfo))
        {

            rdi.cachedJac = 1.0f;
            rdi.gBase     = 1.0f;
        }
        else
        {
            const uint2 j8 = g_pathStateBuffer.Load2(SPM_slotJ(pixelIdx, winnerD));
            rdi.cachedJac = asfloat(j8.x);
            rdi.gBase     = abs(asfloat(j8.y));
        }
    }

    rdi.F = contrib_final;
    rdi.M = (uint)centerConf + effDraws;
    const float F_mag = GetPHat(rdi.F);

    rdi.W = FinalizeUCW(rdi.w_sum, F_mag, ucw_clampMax);

    if (spmis_mcap > 0u) rdi.M = min(rdi.M, spmis_mcap);

    float3 outC = rdi.F * rdi.W;
    if (RGB_SHADE_ON && rdi.w_sum > 0.0f && any(rgbWsum > 0.0f))
        outC = rgbWsum * (rdi.W * F_mag / rdi.w_sum);

    if (RcK(rdi.rcInfo) == 2u && !RcEnvReplay(rdi.rcInfo))
        outC *= ResolveReuseVis(pixelIdx, rdi, outC);
    gScratchPing[uint3(tid.xy, 2)] = float4(outC, 0);
    storeReservoir(g_Reservoirs_last, pixelIdx, rdi);
}
