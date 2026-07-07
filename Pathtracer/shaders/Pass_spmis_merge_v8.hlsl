#define COMPUTE_PASS
#define SPMIS_GRID_NONCOHERENT   // read-only grid/scratch -> L1-cached (see Includes_v8.hlsli)
#include "Includes_v8.hlsli"

//====================================
//SPMIS SPATIAL REUSE - MERGE  (pass 3/3: weights + WRS combine + finalize, NO rays)
//====================================
//Final stage of the split SPMIS reuse. Reads the RAW shifted {c, Jn, cachedNew}
//the unified Pass_shift_v8 produced for the canonical (job slot d=0) and each
//draw (d=1..Ntn) — see the "UNIFIED SHIFT JOB" note in HashGridHash_v8.hlsli —
//computes the SpmisCanonicalMis / SpmisDrawWeight algebra itself (round 8:
//this used to be shift's job; moving it here is what let Pass_shift_v8 shrink
//to a single, role-agnostic code path with no per-role weight tail — see its
//header), does the outer WRS combine into the centre reservoir, lazy-loads
//only the winning candidate's payload, finalizes (M, UCW, M-cap) and writes
//F*W -> scratch slot 2 and the resampled reservoir -> g_Reservoirs_last. No
//reconnections, no rays - all of that was done in the shift loop, so this
//stage runs lean / high-occupancy. Layout: see HashGridHash_v8.hlsli (SPM_*).
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
        if (RcK(rdiP.rcInfo) == 2u && !RcEnvReplay(rdiP.rcInfo))
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

    //passthrough: shadowed canonical (visibility was cached by Pass_spmis_passthrough).
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
    const uint  reuseCell = SPM_hdrCell(w0);
    const uint4 agg       = g_spmisBuffer.Load4(SP_AGG(reuseCell));   // PIXCNT|NZ|CONF|OFF
    const float pixCount  = (float)agg.x;
    const uint  Ntn       = min(max(spmis_reuseN, 1u), SPMIS_SPLIT_MAXDRAWS);
    const float scaling   = (SPMIS_CONF_ADJUST && pixCount > 0.0f) ? min(1.0f, (float)Ntn / pixCount) : 1.0f;
    const float neighbors_conf_sum = (float)agg.z * scaling;

    const float visReuse_c = (rdi.W > 0.0f) ? 1.0f : 0.0f;
    const float p_c        = GetPHat(rdi.F) * visReuse_c;
    const float centerConf = (float)rdi.M;

    //Per-pixel sky observer for any DEFERRED env-replay finish (canonical or draws)
    //below: Pass_shift_v8 kept the atmosphere stack out of its RT binary and only
    //persisted {partial, dir, misPdf}; EnvTailFinish reads this observer. Approx:
    //the offset bounce vertex is replaced by this pixel's primary hit (sky observer
    //is very low-frequency over a pixel neighbourhood).
    SetSkyObserver(load_x1(g_sample_current, pixelIdx) + sceneOriginWorld);

    //==== canonical (job slot d=0): re-derive SpmisCanonicalMis from the raw shift output ====
    float mis_c = 1.0f;
    {
        const uint canonRes = g_pathStateBuffer.Load(SPM_slotZ(pixelIdx, 0u));
        if (canonRes != SP_UNDEF)
        {
            const uint  partnerPx = g_pathStateBuffer.Load(SPM_slotS(pixelIdx, 0u)) & 0x7FFFFFFFu;
            const uint2 j0        = g_pathStateBuffer.Load2(SPM_slotJ(pixelIdx, 0u));
            const float3 c0raw    = asfloat(g_pathStateBuffer.Load3(SPM_slotC(pixelIdx, 0u)));
            //DEFERRED env-replay canonical: slotJ = {PackNormal(dir), misPdf}, c0raw
            //is the partial → finish the sky+sun eval here (observer set once above);
            //cachedNew/Jn are the env const 1, so Jn_c := 1. Reconnect/direct
            //canonicals keep the old {Jn} (its sign = preVisDead, unused here).
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

            //revT naturally collapses to 0 on a killed/failed job (c_canon == 0
            //from Pass_shift_v8's early-out or a dead env finish), so no explicit
            //dead check is needed here — unlike draws, canonical never had one.
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

    //§6.3 RGB shading weights: accumulate the VECTORIZED resampling weight
    //(w_i * chroma of each contributor) alongside the scalar WRS — free, the
    //contributions are already loaded. The resolve blends contributor chroma
    //at the luminance the scalar chain determines (color-noise reduction).
    const float lumC0 = GetPHat(contrib_final);
    float3 rgbWsum = (lumC0 > 0.0f) ? rdi.w_sum * (contrib_final / lumC0) : (float3)0.0f;

    uint2 seed = GetSeed(pixelIdx, time, 9);
    seed.x = Hash32(seed.x);

    [loop]
    for (uint d = 0u; d < Ntn; ++d)
    {
        //job slot d+1 (canonical owns d=0 — see HashGridHash_v8.hlsli)
        const uint slot = d + 1u;
        const uint resRaw = g_pathStateBuffer.Load(SPM_slotZ(pixelIdx, slot));
        if (resRaw == SP_UNDEF)
            continue;   //select-gated: no job ever existed for this draw

        const uint zPx = resRaw & SPM_RESPX_MASK;
        //loadReservoirState (not loadReservoir): the 48B payload is only needed for
        //the eventual WINNER, loaded lazily below. prRcInfo also tells us how to
        //read this draw's slotJ/slotC (env-replay vs reconnect/direct).
        Reservoir pr = (Reservoir)0;
        loadReservoirState(g_Reservoirs_current, zPx, pr);
        const uint prRcInfo = load_rcInfo(g_Reservoirs_current, zPx);
        const bool isEnv    = RcEnvReplay(prRcInfo);
        const bool needWalk = HYBRID_SHIFT_ON && RcReplayLen(prRcInfo) > 0u;

        const uint2  j8   = g_pathStateBuffer.Load2(SPM_slotJ(pixelIdx, slot));
        const float3 craw = asfloat(g_pathStateBuffer.Load3(SPM_slotC(pixelIdx, slot)));
        //DEFERRED env-replay: slotJ = {PackNormal(offset dir), misPdf}, craw = the
        //partial → finish the sky+sun eval here (observer set once above). cachedNew/
        //Jn are the env const 1. Reconnect/direct draws keep {cachedNew, Jn} and
        //preVisDead in Jn's sign.
        float  cachedNew, Jn;
        bool   preVisDead;
        float3 c;
        if (isEnv)
        {
            cachedNew = 1.0f; Jn = 1.0f; preVisDead = false;   //preVisDead unused (env is a walk job)
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

        //dead condition: walked branches re-derive the post-fence geometric band
        //via RcGeomReject(Jn, pr.gBase) (env-replay has Jn=gBase=1 so it never
        //rejects there). The direct branch's reject was folded OUT of the shift
        //(useHint, ~30-VGPR spatial recovery) and re-derived here — same Jn/gBase/
        //threshold. Walked jobs reject on their SHADOWED target; a direct job's
        //preVisDead is PRE-visibility, so a materialized draw whose shadow ray
        //lands fully occluded still counts into M (M tracks sample validity, not
        //this frame's momentary visibility). needWalk re-derived from resPx's own
        //rcInfo (not stored in the descriptor) — cheap, same L1-cached word.
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

    //lazy payload load: only the winning candidate's reconnection vertex is needed
    //(x2/n2/objID/matID/eta/Kd/Pr/Pm/L2/V2 + rcInfo/seed); F/W/M/w_sum are managed
    //here. The accept RE-ANCHORS the jacobian cache to this pixel: cachedJac takes
    //the shift's new-side bundle, gBase its geometric factor (SPM_slotJ), so the
    //next reuse of this reservoir measures against THIS pixel's prefix.
    if (winnerZPx != SP_UNDEF)
    {
        loadReservoirPayload(g_Reservoirs_current, winnerZPx, rdi);
        if (RcEnvReplay(rdi.rcInfo))
        {
            //env-replay winner: slotJ held {PackNormal(dir), misPdf}, NOT
            //{cachedNew, Jn} — the env re-anchor constants are cachedJac = gBase = 1
            //(exactly what the old {1,1} slotJ used to decode to here).
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
