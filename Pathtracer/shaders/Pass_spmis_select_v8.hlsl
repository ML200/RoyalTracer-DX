#define COMPUTE_PASS
#define SPMIS_GRID_NONCOHERENT   // read-only grid/scratch -> L1-cached (see Includes_v8.hlsli)
#include "Includes_v8.hlsli"

//====================================
//SPMIS SPATIAL REUSE - SELECT  (pass 1/3: cell search + candidate selection, NO rays)
//====================================
//First stage of the split SPMIS reuse (select -> shift -> merge), the structural
//analogue of the texture path's Pass_spat_gi_select. Owns the memory-heavy selection -
//hash-cell WRS search, the canonical partner pick, and the Ntilde inner-RIS draws - and
//writes the chosen candidate indices into the per-pixel scratch record
//(g_pathStateBuffer, free in SPMIS mode). Casts NO rays and stores NO reservoir; the
//reconnections + visibility happen in shift, the WRS combine + output in merge.
//Splitting keeps each stage's live state small so the ray stage (shift) runs at high
//occupancy. Layout + status codes: see HashGridHash_v8.hlsli (SPM_*). The selection
//math is identical to the monolithic Pass_spmis_reuse.

static const float SP_SEARCH_R0    = 20.0f;
static const float SP_SEARCH_GROW  = 1.25f;
static const uint  SP_SEARCH_ITERS = 12u;

//==== TEMP DIAGNOSTIC: isolate the cell-search L2 cost. Flip the value, reload, revert. ====
//  0 = normal full search (baseline, ~232px envelope)
//  1 = local search: SAME 12 probes but a tiny radius -> cache-local (isolates scatter)
//  2 = center cell only: NO neighbour probing at all (isolates the whole search)
//Mode 2 also skips the search's RNG draws, so the image shifts slightly - fine for a perf
//measurement. Delete this block + restore the plain search once the investigation is done.
#ifndef SPMIS_SELECT_DIAG
#define SPMIS_SELECT_DIAG 0
#endif

[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    if (!SPMIS_SPATIAL_MODE) return;
    if (tid.x >= IMG_W || tid.y >= IMG_H) return;
    gDispatchIdx = tid;

    const float2 dims     = float2(IMG_W, IMG_H);
    const uint   pixelIdx = MapPixelID(dims, tid.xy);

    //emitter -> mark skip; merge writes the V2 sentinel, shift does nothing.
    if (load_isEmitter(g_sample_current, pixelIdx))
    {
        g_pathStateBuffer.Store(SPM_w0(pixelIdx), SPM_packHdr(0u, SPM_STATUS_SKIP));
        return;
    }

    const uint centerM    = load_M(g_Reservoirs_current, pixelIdx);
    const uint cellCenter = g_spmisBuffer.Load(SP_A(SP_HASH, pixelIdx));

    //no center sample or no resolved cell -> passthrough (shift casts the one deferred
    //visibility ray, merge writes the shadowed canonical).
    if (centerM == 0u || cellCenter == SP_UNDEF)
    {
        g_pathStateBuffer.Store(SPM_w0(pixelIdx), SPM_packHdr(0u, SPM_STATUS_PASS));
        return;
    }

    const uint   myInst = load_instID(g_sample_current, pixelIdx);
    const float3 myN    = load_n1_s_with_instID(g_sample_current, pixelIdx, myInst);
    const float3 camPos = InitOrigin();
    const float3 myPos  = load_x1_with_instID(g_sample_current, pixelIdx, myInst);

    uint2 seed = GetSeed(pixelIdx, time, 6);
    seed.x = Hash32(seed.x);

    const uint Ntn = min(max(spmis_reuseN, 1u), SPMIS_SPLIT_MAXDRAWS);

    //==== CELL SEARCH - WRS, center pre-seeded ====
#if SPMIS_SELECT_DIAG == 2
    //DIAG mode 2: no neighbour search at all - reuse the centre cell.
    const uint reuseCell = cellCenter;
#else
  #if SPMIS_SELECT_DIAG == 1
    const float diagR0   = 4.0f;   // tiny envelope -> cache-local probes
    const float diagGrow = 1.0f;
  #else
    const float diagR0   = SP_SEARCH_R0;
    const float diagGrow = SP_SEARCH_GROW;
  #endif
    float weight_sum   = (float)g_spmisBuffer.Load(SP_CONF_A(cellCenter));
    uint  selectedCell = cellCenter;
    float radius       = diagR0;
    const float planeThresh = spmis_planeDist * length(myPos - camPos);
    [loop]
    for (uint i = 0u; i < SP_SEARCH_ITERS; ++i, radius *= diagGrow)
    {
        int2 off = int2(round(radius * (RandomFloatPCG(seed.x) * 2.0f - 1.0f)),
                        round(radius * (RandomFloatPCG(seed.x) * 2.0f - 1.0f)));
        int2 nc  = int2(tid.xy) + off;
        if (nc.x < 0) nc.x = -nc.x; else if (nc.x >= (int)IMG_W) nc.x = 2 * (int)IMG_W - nc.x - 1;
        if (nc.y < 0) nc.y = -nc.y; else if (nc.y >= (int)IMG_H) nc.y = 2 * (int)IMG_H - nc.y - 1;
        if (nc.x < 0 || nc.y < 0 || nc.x >= (int)IMG_W || nc.y >= (int)IMG_H) continue;

        const uint npx   = MapPixelID(dims, (uint2)nc);
        const uint ncell = g_spmisBuffer.Load(SP_A(SP_HASH, npx));
        if (ncell == SP_UNDEF || ncell == cellCenter) continue;   // cheap 4B reject key

        //bundled search record: {cellConf, worldPos} in ONE Load4 (npx-indexed, framebuffer
        //-local) - replaces the scattered SP_CONF[ncell] + neighbour-G-buffer instID/x1
        //fan-out. (No-hit neighbours have SP_HASH==UNDEF, already rejected above, so the old
        //instID!=0xFFFFFFFF guard is subsumed.)
        const uint4 rec = g_spmisBuffer.Load4(SP_SRCH(npx));
        //normal cone (uniform branch; default spmis_normalSimCos == -1 skips it, so the
        //scattered neighbour-normal fetch is only paid when the cone is actually enabled).
        if (spmis_normalSimCos > -1.0f)
        {
            const uint   ninst = load_instID(g_sample_current, npx);
            const float3 nN    = load_n1_s_with_instID(g_sample_current, npx, ninst);
            if (dot(myN, nN) <= spmis_normalSimCos) continue;
        }
        //plane-distance reject (bit-identical per-pixel position the G-buffer path used).
        const float3 nPos = asfloat(rec.yzw);
        if (abs(dot(nPos - myPos, myN)) > planeThresh) continue;

        const float w = asfloat(rec.x);   // cell confidence sum (== SP_CONF[ncell])
        weight_sum += w;
        if (RandomFloatPCG(seed.x) < w / weight_sum) selectedCell = ncell;
    }
    const uint reuseCell = selectedCell;
#endif
    const uint4 agg       = g_spmisBuffer.Load4(SP_AGG(reuseCell));   // PIXCNT|NZ|CONF|OFF (one fetch)
    const uint  cellBase  = agg.w;          // OFF
    const float pixCount  = (float)agg.x;   // PIXCNT
    const uint  nzCount   = agg.y;          // NZ
    const float scaling   = (SPMIS_CONF_ADJUST && pixCount > 0.0f) ? min(1.0f, (float)Ntn / pixCount) : 1.0f;
    const float neighbors_conf_sum = (float)agg.z * scaling;   // CONF

    //==== canonical partner pick (Nc=1 uniform over the cell); shift reconnects + rays ====
    uint partnerPx = SP_UNDEF;
    if (neighbors_conf_sum > EPSILON && pixCount > 0.0f)
    {
        uint k = (uint)(pixCount * RandomFloatPCG(seed.x));
        if (k >= (uint)pixCount) k = (uint)pixCount - 1u;
        partnerPx = g_spmisBuffer.Load(SP_A(SP_SORTED, cellBase + k));
    }

    g_pathStateBuffer.Store(SPM_w0(pixelIdx), SPM_packHdr(reuseCell, SPM_STATUS_NORM));
    g_pathStateBuffer.Store(SPM_w1(pixelIdx), partnerPx);

    //==== Ntilde inner-RIS selections (each prop. UCW*target*M, P=1/UCW) ====
    //Every slot in [0,Ntn) is written (valid zPx or SP_UNDEF) so merge can skip
    //unmaterialized draws without a separate clear; merge recomputes Ntn identically.
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
            for (uint ri = 0u; ri < spmis_risN; ++ri)
            {
                uint k = (uint)((float)nzCount * RandomFloatPCG(seed.x));
                if (k >= nzCount) k = nzCount - 1u;
                const float tf = asfloat(g_spmisBuffer.Load(SP_A(SP_SORTEDW, cellBase + k)));
                const float w  = tf * (float)nzCount / (float)spmis_risN;
                ris_wsum += w;
                if (ris_wsum > 0.0f && RandomFloatPCG(seed.x) < w / ris_wsum) { ris_selK = k; ris_selTF = tf; }
            }
            if (ris_selK != SP_UNDEF && ris_selTF > 0.0f)
            {
                const uint zPx = g_spmisBuffer.Load(SP_A(SP_SORTED, cellBase + ris_selK));
                if (zPx != SP_UNDEF) { outZ = zPx; outP = ris_selTF / ris_wsum; }
            }
        }
        const uint sa = SPM_slot(pixelIdx, d);
        g_pathStateBuffer.Store(sa,      outZ);
        g_pathStateBuffer.Store(sa + 4u, asuint(outP));
    }
}
