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
    if (tid.x >= IMG_W || tid.y >= IMG_H) return;
    gDispatchIdx = tid;

    const float2 dims     = float2(IMG_W, IMG_H);
    const uint   pixelIdx = MapPixelID(dims, tid.xy);

    //SPATIAL-OFF FALLBACK: Pass_shading reads scratch slot 2 unconditionally and
    //this pass owns the reservoir ping-pong, so with spatial reuse disabled the
    //(temporal) result must still resolve here — otherwise slot 2 goes stale and
    //next frame's temporal pass reads a dead history buffer.
    if (!SPMIS_SPATIAL_MODE)
    {
        if (load_isEmitter(g_sample_current, pixelIdx))
        {
            g_Reservoirs_last.Store(addr_v2(pixelIdx), PROBE_DI_NORMAL_ZERO_CODE);
            return;
        }
        Reservoir rdiP = loadReservoir(g_Reservoirs_current, pixelIdx);
        const float  Wp   = (rdiP.W > 0.0f) ? rdiP.W : 0.0f;
        float3 outCP = rdiP.F * Wp;
        if (RcK(rdiP.rcInfo) == 2u && !RcEnvReplay(rdiP.rcInfo))   //direct k==2 only, see below
            outCP *= ResolveReuseVis(pixelIdx, rdiP, outCP);       //deferred vis (REUSE_VIS_OFF)
        gScratchPing[uint3(tid.xy, 2)] = float4(outCP, 0);
        storeReservoir(g_Reservoirs_last, pixelIdx, rdiP);
        return;
    }

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
    uint   winnerD       = 0u;

    //§6.3 RGB shading weights: accumulate the VECTORIZED resampling weight
    //(w_i * chroma of each contributor) alongside the scalar WRS — free, the
    //contributions are already loaded. The resolve blends contributor chroma
    //at the luminance the scalar chain determines (color-noise reduction).
    const float lumC0 = GetPHat(contrib_final);
    float3 rgbWsum = (lumC0 > 0.0f) ? rdi.w_sum * (contrib_final / lumC0) : (float3)0.0f;

    const uint Ntn = min(max(spmis_reuseN, 1u), SPMIS_SPLIT_MAXDRAWS);
    uint2 seed = GetSeed(pixelIdx, time, 9);
    seed.x = Hash32(seed.x);

    [loop]
    for (uint d = 0u; d < Ntn; ++d)
    {
        const uint zPx = g_pathStateBuffer.Load(SPM_slotZ(pixelIdx, d));
        if (zPx == SP_UNDEF || (zPx & SPM_Z_PENDING_BIT) != 0u)
            continue;   // unmaterialized, gate-rejected, or an unresolved replay slot
                        // (pending only possible if the replay pass was skipped)

        const float  w_draw = asfloat(g_pathStateBuffer.Load(SPM_slotP(pixelIdx, d)));
        const float3 c      = asfloat(g_pathStateBuffer.Load3(SPM_slotC(pixelIdx, d)));

        rdi.w_sum += w_draw;
        effDraws++;
        const float lumCd = GetPHat(c);
        if (w_draw > 0.0f && lumCd > 0.0f)
            rgbWsum += w_draw * (c / lumCd);
        if (rdi.w_sum > 0.0f && RandomFloatPCG(seed.x) < w_draw / rdi.w_sum)
        {
            winnerZPx     = zPx;
            winnerD       = d;
            contrib_final = c;
        }
    }

    //lazy payload load: only the winning candidate's reconnection vertex is needed
    //(x2/n2/objID/matID/eta/Kd/Pr/Pm/L2/V2 + rcInfo/seed); F/W/M/w_sum are managed
    //here. The accept RE-ANCHORS the jacobian cache to this pixel: cachedJac takes
    //the shift's new-side bundle, gBase its geometric factor (SPM_slotJ), so the
    //next reuse of this reservoir measures against THIS pixel's prefix. slotC
    //already holds the re-anchored F = c*vis*Jn/cachedNew.
    if (winnerZPx != SP_UNDEF)
    {
        loadReservoirPayload(g_Reservoirs_current, winnerZPx, rdi);
        const uint2 j8 = g_pathStateBuffer.Load2(SPM_slotJ(pixelIdx, winnerD));
        rdi.cachedJac = asfloat(j8.x);
        rdi.gBase     = asfloat(j8.y);
    }

    //==== FINALIZE (M = center.M + materialized draws; UCW = wsum / target; M-cap last) ====
    rdi.F = contrib_final;
    rdi.M = (uint)centerConf + effDraws;
    const float F_mag = GetPHat(rdi.F);

    //bounded UCW (ucw_clampMax): unclamped w_sum/p_hat spikes near grazing/occluded
    //surfaces and feeds back through reuse into a diverging firefly. See FinalizeUCW.
    rdi.W = FinalizeUCW(rdi.w_sum, F_mag, ucw_clampMax);

    if (spmis_mcap > 0u) rdi.M = min(rdi.M, spmis_mcap);

    //§6.3 resolve: scale the vectorized weight sum so its luminance equals the
    //scalar chain's F_mag*W — identical brightness (the bounded-UCW clamp
    //semantics carry over exactly), blended chroma. Falls back to the winner's
    //color when the flag is off or the vector sum is degenerate.
    float3 outC = rdi.F * rdi.W;
    if (RGB_SHADE_ON && rdi.w_sum > 0.0f && any(rgbWsum > 0.0f))
        outC = rgbWsum * (rdi.W * F_mag / rdi.w_sum);
    //deferred reuse visibility (RS_FLAG_NO_REUSE_VIS): the one reconnection
    //shadow ray the reuse passes skipped lands here, once, on the resolved
    //winner. Direct k==2 reconnections ONLY: replay shifts fold visibility
    //inline at the replayed vertex (x1 -> x_k is a non-segment through
    //geometry for k>2; env-replay's must-miss trace IS its visibility), and
    //pin-less canonicals were generated fully shadowed. Returns 1.0 when the
    //flag is off — inert on the default path.
    if (RcK(rdi.rcInfo) == 2u && !RcEnvReplay(rdi.rcInfo))
        outC *= ResolveReuseVis(pixelIdx, rdi, outC);
    gScratchPing[uint3(tid.xy, 2)] = float4(outC, 0);
    storeReservoir(g_Reservoirs_last, pixelIdx, rdi);
}
