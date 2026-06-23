#define COMPUTE_PASS
#define SPMIS_GRID_NONCOHERENT   // read-only grid/scratch -> L1-cached (see Includes_v8.hlsli)
#include "Includes_v8.hlsli"

//====================================
//SPMIS SPATIAL REUSE - MERGE  (pass 3/3: WRS combine + finalize, NO rays)
//====================================
//Final stage of the split SPMIS reuse (analogue of the texture path's Pass_spat_gi_v8_1).
//Pure data: reads the cached canonical MIS weight + per-draw {w_draw, c} that shift
//produced, runs the outer WRS combine into the center reservoir, lazy-loads only the
//winning candidate's payload, finalizes (M, UCW, M-cap) and writes F*W -> scratch slot 2
//and the resampled reservoir -> g_Reservoirs_last. No reconnections, no rays - all of
//that was done in shift, so this stage runs lean / high-occupancy. Layout: see
//HashGridHash_v8.hlsli (SPM_*).
//
//Note: the outer WRS uses its own RNG stream (dim 9) since select consumed the
//cell-search / inner-RIS draws (dim 6). The estimate stays unbiased; the noise pattern
//differs slightly from the monolithic interleaved stream.

[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    if (!SPMIS_SPATIAL_MODE) return;
    if (tid.x >= IMG_W || tid.y >= IMG_H) return;
    gDispatchIdx = tid;

    const float2 dims     = float2(IMG_W, IMG_H);
    const uint   pixelIdx = MapPixelID(dims, tid.xy);

    const uint w0     = g_pathStateBuffer.Load(SPM_w0(pixelIdx));
    const uint status = SPM_hdrStatus(w0);

    //emitter: mirror the texture path's V2 sentinel, write nothing else.
    if (status == SPM_STATUS_SKIP)
    {
        g_Reservoirs_last.Store(addr_v2(pixelIdx), PROBE_DI_NORMAL_ZERO_CODE);
        return;
    }

    Reservoir rdi = loadReservoir(g_Reservoirs_current, pixelIdx);

    //passthrough: shadowed canonical (visibility was cached by shift).
    if (status == SPM_STATUS_PASS)
    {
        const float  W    = (rdi.W > 0.0f) ? rdi.W : 0.0f;
        const float  vis  = asfloat(g_pathStateBuffer.Load(SPM_w1(pixelIdx)));
        const float3 outC = rdi.F * W * vis;
        gScratchPing[uint3(tid.xy, 2)] = float4(outC, 0);
        storeReservoir(g_Reservoirs_last, pixelIdx, rdi);
        return;
    }

    //==== normal path: canonical seed + outer WRS over the cached draws ====
    const float visReuse_c = (rdi.W > 0.0f) ? 1.0f : 0.0f;
    const float p_c        = GetPHat(rdi.F) * visReuse_c;
    const float centerConf = (float)rdi.M;
    const float mis_c      = asfloat(g_pathStateBuffer.Load(SPM_w1(pixelIdx)));

    rdi.w_sum = mis_c * p_c * rdi.W;
    float3 contrib_final = rdi.F * visReuse_c;
    uint   effDraws      = 0u;
    uint   winnerZPx     = SP_UNDEF;

    const uint Ntn = min(max(spmis_reuseN, 1u), SPMIS_SPLIT_MAXDRAWS);
    uint2 seed = GetSeed(pixelIdx, time, 9);
    seed.x = Hash32(seed.x);

    [loop]
    for (uint d = 0u; d < Ntn; ++d)
    {
        const uint sa  = SPM_slot(pixelIdx, d);
        const uint zPx = g_pathStateBuffer.Load(sa);
        if (zPx == SP_UNDEF) continue;   // unmaterialized or gate-rejected in shift

        const float  w_draw = asfloat(g_pathStateBuffer.Load(sa + 4u));
        const float3 c      = asfloat(g_pathStateBuffer.Load3(sa + 8u));

        rdi.w_sum += w_draw;
        effDraws++;
        if (rdi.w_sum > 0.0f && RandomFloatPCG(seed.x) < w_draw / rdi.w_sum)
        {
            winnerZPx     = zPx;
            contrib_final = c;
        }
    }

    //lazy payload load: only the winning candidate's reconnection vertex is needed
    //(x2/n2/objID/matID/eta/Kd/Pr/Pm/L2/V2); F/W/M/w_sum are managed here.
    if (winnerZPx != SP_UNDEF)
        loadReservoirPayload(g_Reservoirs_current, winnerZPx, rdi);

    //==== FINALIZE (M = center.M + materialized draws; UCW = wsum / target; M-cap last) ====
    rdi.F = contrib_final;
    rdi.M = (uint)centerConf + effDraws;
    const float F_mag = GetPHat(rdi.F);

    if (F_mag > EPSILON && rdi.w_sum > 0.0f && rdi.w_sum < 1e10f)
    {
        float W = rdi.w_sum / F_mag;
        if (isnan(W) || isinf(W) || (W < 0.0f)) W = 0.0f;
        rdi.W = W;
    }
    else
    {
        rdi.W = 0.0f;
    }

    if (spmis_mcap > 0u) rdi.M = min(rdi.M, spmis_mcap);

    float3 outC = rdi.F * rdi.W;
    gScratchPing[uint3(tid.xy, 2)] = float4(outC, 0);
    storeReservoir(g_Reservoirs_last, pixelIdx, rdi);
}
