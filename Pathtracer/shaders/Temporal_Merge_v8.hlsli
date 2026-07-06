//====================================
//TEMPORAL MERGE BODY  (shared: Pass_temp_gi + Pass_temp_replay)
//====================================
//The complete temporal resampling for one pixel given the already-chosen
//reprojected candidate coordinate. Compiled in two variants:
//  TEMPORAL_CAN_REPLAY == 0  (Pass_temp_gi): direct (no-replay) shifts + RayQuery
//    visibility. Any direction needing a prefix/env replay parks the resolved
//    candidate (coord + dual flag) in SPM_w1 and appends the pixel to the
//    temporal replay queue (g_raygenQueue counter at byte 4) — no TraceRay ever
//    links into this pass.
//  TEMPORAL_CAN_REPLAY == 1  (Pass_temp_replay): identical body, but replay
//    shifts evaluate through Hybrid_Replay's HybridShiftEval. Runs as a
//    compacted 1D indirect dispatch over exactly the queued pixels.
//Both variants execute the same loads, gates, MIS and accept algebra, so a
//pixel produces the same merge no matter which pass finishes it.
//
//§6.4 DUAL MOTION VECTORS: when the reprojected candidate fails the geometry
//reject (a moving occluder disoccluded this pixel), retry once at
//launchIndex - (occluder screen motion), where the occluder is the prev-frame
//surface at the reprojected pixel projected with the CURRENT camera (static-
//occluder approximation). Dual candidates carry far-field geometry, so they
//use the geometric REJECT band (spmis_jacThreshold) instead of the force-to-1
//clamp — clamping their Jgeo is a systematic energy bias at disocclusions.
//
//PSS reuse algebra (see Reservoir_v8.hlsli / ReconnectPSS):
//  weight     w_n = mis * lum(c*vis) * Jn * geomScale / cachedJac_src * W_src
//  accept     F   = c*vis * Jn / cachedNew ; cachedJac = cachedNew ; gBase = Jn
//  band       geometric ratio Jn/gBase only (clamp for base candidates,
//             reject for dual candidates)
//There are NO reuse-time roughness/distance gates: shifts are the fixed maps
//T_k of their samples, glossy receivers self-gate via the BSDF magnitude.

#ifndef TEMPORAL_MERGE_V8_HLSLI
#define TEMPORAL_MERGE_V8_HLSLI

#ifndef TEMPORAL_CAN_REPLAY
#define TEMPORAL_CAN_REPLAY 0
#endif

//candidate resolution status
#define TM_CAND_OK      0u
#define TM_CAND_DEAD    1u   //off-screen / emitter / invalid / unreusable
#define TM_CAND_GEOMREJ 2u   //geometry mismatch (dual-MV eligible)

//Resolve one candidate coordinate: bounds/emitter test, reservoir validity +
//reusability, geometry rejects. On TM_CAND_GEOMREJ, rPos still holds the
//rejected surface (the dual-MV occluder estimate).
inline uint TemporalResolveCand(
    int2 cand, float2 dims_f,
    float3 myPos, float3 myN1s, float3 cameraPos,
    out uint tempPixelIdx, out Reservoir rdi_r, out float3 rPos)
{
    tempPixelIdx = 0xFFFFFFFFu;
    rdi_r = (Reservoir)0;
    rPos  = (float3)0.0f;

    uint tpx = 0xFFFFFFFFu;
    if (!TestTemporalCandidate(cand, dims_f, g_sample_last, tpx))
        return TM_CAND_DEAD;

    Reservoir rr = loadReservoir(g_Reservoirs_last, tpx);
    if (!IsValidReservoir(rr))
        return TM_CAND_DEAD;

    //reusability: hybrid = pin-less samples define no shift; legacy = replay
    //candidates from a pre-toggle frame can't be evaluated + the near-specular
    //reconnection-vertex reject (only REJECTION is unbiased there).
    if (HYBRID_SHIFT_ON)
    {
        if (!RcReusable(rr.rcInfo))
            return TM_CAND_DEAD;
    }
    else
    {
        if (RcReplayLen(rr.rcInfo) > 0u)
            return TM_CAND_DEAD;
        if (!IsSentinelMatID(rr.matID) && !IsVolumeVertex(rr.matID) &&
            rr.Pr < rs_reconnectRoughnessMin)
            return TM_CAND_DEAD;
    }

    //geometry rejects: a reprojection onto a different face / across a depth
    //discontinuity is what drives the reconnection cos to ~0.
    rPos = load_x1(g_sample_last, tpx);
    {
        const float3 rN = load_n1_s(g_sample_last, tpx);
        if (temp_normalSimCos > -1.0f && dot(myN1s, rN) <= temp_normalSimCos)
            return TM_CAND_GEOMREJ;
        const float planeThresh = temp_planeDist * length(myPos - cameraPos);
        if (abs(dot(rPos - myPos, myN1s)) > planeThresh)
            return TM_CAND_GEOMREJ;
    }

    tempPixelIdx = tpx;
    rdi_r = rr;
    return TM_CAND_OK;
}

void TemporalMergeBody(uint pixelIdx, uint2 launchIndex, int2 candCoord, bool dualIn)
{
    const float2 dims_f = float2(IMG_W, IMG_H);

    //own-pixel loads up front (the geometry gate + reverse shift need them)
    const uint   myFlags  = load_flagsWord(g_sample_current, pixelIdx);
    const uint   myInstID = load_instID(g_sample_current, pixelIdx);
    const uint   myMatID  = load_matID(g_sample_current, pixelIdx);
    const float3 myPos    = load_x1_with_instID(g_sample_current, pixelIdx, myInstID);
    const float3 myN1s    = load_n1_s_with_instID(g_sample_current, pixelIdx, myInstID);
    float3 myKd = load_kd(g_sample_current, pixelIdx);
    float  myPr, myPm;
    load_prpm(g_sample_current, pixelIdx, myPr, myPm);
    const float3 cameraPos = InitOrigin();

    //====================================
    //CANDIDATE RESOLUTION (+ §6.4 dual-MV retry on geometry reject)
    //====================================
    int2      cand = candCoord;
    bool      dual = dualIn;
    uint      tempPixelIdx;
    Reservoir rdi_r;
    float3    rPos;
    uint st = TemporalResolveCand(cand, dims_f, myPos, myN1s, cameraPos,
                                  tempPixelIdx, rdi_r, rPos);
    if (st == TM_CAND_GEOMREJ && DUAL_MV_ON && !dual)
    {
        //occluder screen motion under the current camera; deterministic, no RNG
        const float2 occNow = GetCurrentFramePixelCoordinates_World(rPos, view, projection, dims_f);
        if (occNow.x > -1e8f)
        {
            const int2 dc = int2(round(float2(launchIndex) - (occNow - float2(cand))));
            if (dc.x >= 0 && dc.y >= 0 && dc.x < (int)IMG_W && dc.y < (int)IMG_H &&
                any(dc != cand))
            {
                st = TemporalResolveCand(dc, dims_f, myPos, myN1s, cameraPos,
                                         tempPixelIdx, rdi_r, rPos);
                if (st == TM_CAND_OK) { cand = dc; dual = true; }
            }
        }
    }
    if (st != TM_CAND_OK)
        return;

    //canonical reservoir
    Reservoir rdi = loadReservoir(g_Reservoirs_current, pixelIdx);

    //====================================
    //SHIFT STRUCTURE (replay need per direction) + REPLAY ROUTING
    //====================================
    const bool fwdReplay = RcReplayLen(rdi_r.rcInfo) > 0u;     //neighbour sample -> me
    const bool revValid  = IsValidReservoir(rdi) && rdi.W > 0.0f &&
                           (!HYBRID_SHIFT_ON || RcReusable(rdi.rcInfo));
    const bool revReplay = revValid && RcReplayLen(rdi.rcInfo) > 0u;   //my sample -> neighbour

#if !TEMPORAL_CAN_REPLAY
    if (fwdReplay || revReplay)
    {
        //park the resolved candidate (coord + dual band policy) for the replay pass
        g_pathStateBuffer.Store(SPM_w1(pixelIdx),
                                (uint(cand.x) & 0xFFFFu) | ((uint(cand.y) & 0x7FFFu) << 16) |
                                (dual ? 0x80000000u : 0u));
        uint slot;
        g_raygenQueue.InterlockedAdd(4u, 1u, slot);
        g_raygenQueue.Store(16u + slot * 4u,
                            (launchIndex.x & 0xFFFFu) | (launchIndex.y << 16));
        return;
    }
#endif

    uint2 seed = GetSeed(pixelIdx, time, 12);
    seed.x = Hash32(seed.x);

    //====================================
    //FORWARD: neighbour sample evaluated at me   (target + accept payload)
    //====================================
    float  Jn_f = 0.0f, cachedNew_f = 0.0f;
    float3 c_f  = (float3)0.0f;                //shifted numerator * vis
    if (!fwdReplay)
    {
        const SurfaceVertex sv_c = MakeVertex(myPos, myN1s, cameraPos, myMatID,
                                              myKd, myPr, myPm,
                                              (myFlags & SD_FLAG_BACKFACE) != 0u);
        float3 c = ReconnectPSS_sv(sv_c, rdi_r, Jn_f, cachedNew_f);
        if (GetPHat(c) > 0.0f)
        {
            //RS_FLAG_NO_REUSE_VIS: skip the reuse shadow ray (deferred to resolve).
            //Volume vertices are in-medium (entry surface self-occludes) -> visible.
            float3 vis = 1.0.xxx;
            if (!REUSE_VIS_OFF && !IsVolumeVertex(rdi_r.matID))
            {
                if (rdi_r.matID == MATID_ENV_MISS)
                {
                    const float3 md = normalize(rdi_r.x2);
                    vis = VisibilityTransmittance(myPos, myN1s, myPos + md * RAY_TMAX_PLANET, -md);
                }
                else
                    vis = VisibilityTransmittance(myPos, myN1s, rdi_r.x2, rdi_r.n2_s);
            }
            c_f = c * vis;
        }
    }
#if TEMPORAL_CAN_REPLAY
    else
    {
        //replay shifts fold visibility inline at the replayed vertex — the
        //deferred resolve can never re-derive y_{k-1}, so REUSE_VIS_OFF does
        //not apply here.
        c_f = HybridShiftEval(g_sample_current, pixelIdx, rdi_r,
                              true, Jn_f, cachedNew_f);
    }
#endif

    //====================================
    //REVERSE: my sample evaluated at the neighbour   (canonical MIS denominator)
    //====================================
    float  Jn_r = 0.0f, cachedNew_r = 0.0f;
    float3 c_r  = (float3)0.0f;
    if (revValid)
    {
        if (!revReplay)
        {
            const SurfaceVertex sv_r = BuildVertex(g_sample_last, tempPixelIdx, rPos, cameraPos);
            float3 c = ReconnectPSS_sv(sv_r, rdi, Jn_r, cachedNew_r);
            if (GetPHat(c) > 0.0f)
            {
                float3 vis = 1.0.xxx;
                if (!REUSE_VIS_OFF && !IsVolumeVertex(rdi.matID))
                {
                    if (rdi.matID == MATID_ENV_MISS)
                    {
                        const float3 md = normalize(rdi.x2);
                        vis = VisibilityTransmittance(rPos, sv_r.n_s, rPos + md * RAY_TMAX_PLANET, -md);
                    }
                    else
                        vis = VisibilityTransmittance(rPos, sv_r.n_s, rdi.x2, rdi.n2_s);
                }
                c_r = c * vis;
            }
        }
#if TEMPORAL_CAN_REPLAY
        else
        {
            c_r = HybridShiftEval(g_sample_last, tempPixelIdx, rdi,
                                  true, Jn_r, cachedNew_r);
        }
#endif
    }

    //====================================
    //MIS + RESERVOIR MERGE  (PSS weights)
    //====================================
    const float visReuse_c = (rdi.W > 0.0f) ? 1.0f : 0.0f;
    const float p_c = GetPHat(rdi.F) * visReuse_c;

    //geometric band (Jn/gBase only, pdf factors stay exact): clamp-scale for
    //base candidates, REJECT band for dual candidates (see header).
    float geomScale_f, geomScale_r;
    if (dual)
    {
        geomScale_f = RcGeomReject(Jn_f, rdi_r.gBase, spmis_jacThreshold) ? 0.0f : 1.0f;
        geomScale_r = RcGeomReject(Jn_r, rdi.gBase,   spmis_jacThreshold) ? 0.0f : 1.0f;
    }
    else
    {
        geomScale_f = RcGeomClampScale(Jn_f, rdi_r.gBase, temp_jacClamp);
        geomScale_r = RcGeomClampScale(Jn_r, rdi.gBase,   temp_jacClamp);
    }

    //shifted target densities (lum(c*vis) * Jn / cachedJac_src), band-scaled
    const float fwdT = (rdi_r.cachedJac > 0.0f)
        ? GetPHat(c_f) * Jn_f * geomScale_f / rdi_r.cachedJac : 0.0f;   //neighbour sample at me
    const float revT = (rdi.cachedJac > 0.0f)
        ? GetPHat(c_r) * Jn_r * geomScale_r / rdi.cachedJac   : 0.0f;   //my sample at neighbour
    const float n_n  = GetPHat(rdi_r.F) * ((rdi_r.W > 0.0f) ? 1.0f : 0.0f);

    //correlation reduction cCap, dup count D refreshes the chain (2026 paper).
    const float D        = saturate(gScratchPing[uint3(uint2(cand), 6)].x);
    const float effMcapF = (CORR_REDUCTION_OFF || rdi_r.matID == MATID_ENV_MISS)
                           ? (float)rs_tempMcap
                           : lerp((float)rs_tempMcap, 1.0f, pow(D, 0.1f));

    const uint mcapU   = max(1u, rs_tempMcap);
    const uint effMcap = (uint)clamp(round(effMcapF), 1.0f, (float)mcapU);

    //Roughness-dependent neighbour mcap: LEGACY ONLY — it papered over the old
    //shift's specular temporal lag; the hybrid shift replays exactly that
    //transport, so the full mcap applies to every roughness.
    uint dynTempMcap = effMcap;
    if (!HYBRID_SHIFT_ON)
    {
        const float roughScale    = smoothstep(rs_reuseRoughnessMin, rs_reuseRoughnessMax, myPr);
        const float tempMcapScale = lerp(1.0f, roughScale, myPm);   // dielectrics exempt
        dynTempMcap = (uint)clamp(round(effMcap * tempMcapScale), 1.0f, (float)mcapU);
    }

    const uint M_c = clamp(min(effMcap,     rdi.M),   1u, mcapU);
    const uint M_n = clamp(min(dynTempMcap, rdi_r.M), 1u, mcapU);

    //  canonical: m_c = M_c p_c / (M_c p_c + M_n revT)   [canonical at center vs shifted to neighbour]
    //  neighbour: m_n = M_n n_n / (M_n n_n + M_c fwdT)   [neighbour in its domain vs shifted to center]
    const float denom_c = M_c * p_c + M_n * revT;
    const float denom_n = M_n * n_n + M_c * fwdT;
    const float mis_c = (denom_c > EPSILON) ? (M_c * p_c / denom_c) : 1.0f;
    const float mis_n = (denom_n > EPSILON) ? (M_n * n_n / denom_n) : 0.0f;

    //resampling weights: canonical keeps its own-domain form; the neighbour's is
    //its shifted density at this pixel times its unbiased contribution weight.
    const float w_c = mis_c * p_c * rdi.W;
    const float w_n = mis_n * fwdT * rdi_r.W;

    //accept payload: re-anchored PSS contribution + new-side jacobian cache
    const float3 F_shifted = (cachedNew_f > 0.0f) ? (c_f * Jn_f / cachedNew_f) : (float3)0.0f;

    rdi.w_sum = w_c;

    float p_hat_final = p_c;
    bool  accepted    = false;
    if (UpdateReservoir(rdi, w_n, M_n, rdi_r, F_shifted, cachedNew_f, Jn_f, seed))
    {
        p_hat_final = GetPHat(F_shifted);
        accepted    = true;
    }

    //bounded UCW: an unclamped w_sum/p_hat spikes + feeds back every frame near
    //grazing/occluded surfaces (see FinalizeUCW). ucw_clampMax breaks it.
    rdi.W = FinalizeUCW(rdi.w_sum, p_hat_final, ucw_clampMax);

    if (accepted)
    {
        storeReservoir(g_Reservoirs_current, pixelIdx, rdi);
    }
    else
    {
        //payload bits in memory are still raygen's - persist only W and M
        store_W(g_Reservoirs_current, pixelIdx, rdi.W);
        store_M(g_Reservoirs_current, pixelIdx, rdi.M);
    }
}

#endif // TEMPORAL_MERGE_V8_HLSLI
