//====================================
//TEMPORAL MERGE — RESOLVE + ROUTE  (Pass_temp_gi only)
//====================================
//The candidate resolution for one pixel, reduced (round 8) to JUST resolving
//+ writing the two job descriptors Pass_shift_v8 needs — see the "UNIFIED
//SHIFT JOB" note in HashGridHash_v8.hlsli. Mirrors spatial's select/
//shift/merge exactly: temp_gi (this file) resolves + routes, Pass_shift_v8
//(dispatched right after as a single Depth=2 DispatchRays — the SAME one
//binary spatial's Depth=Ntn+1 dispatch also uses, see its header for why one
//binary/one dispatch beats specialized-and-looped alternatives here) performs
//both directions' reconnection-or-replay mapping, Pass_temp_merge is pure
//data that reads the raw {c, Jn, cachedNew} outputs and does the MIS combine
//+ reservoir update.
//
//ROUND 8 UNIFICATION: previously this file ALSO evaluated whichever direction
//didn't need a prefix/env replay walk directly (ReconnectPSS_sv + a
//visibility ray, right here), parking a PENDING sentinel for the walk-needing
//direction so a separate Pass_temp_replay could fill it in later. That's
//gone: EVERY direction is now a job descriptor Pass_shift_v8 resolves
//uniformly (walk or direct, it doesn't matter which — see its header for why
//unifying the *mapping* itself, not just where it runs, is what let the
//register-pressure fix stick). This file no longer touches
//ReconnectPSS/VisibilityTransmittance/MakeVertex at all, and no longer needs
//myKd/myPr/myPm/myMatID/myFlags — only the geometry-gate inputs survive.
//
//§6.4 DUAL MOTION VECTORS: when the reprojected candidate fails the geometry
//reject (a moving occluder disoccluded this pixel), retry once at
//launchIndex - (occluder screen motion), where the occluder is the prev-frame
//surface at the reprojected pixel projected with the CURRENT camera (static-
//occluder approximation). Dual candidates carry far-field geometry, so
//Pass_temp_merge uses the geometric REJECT band (spmis_jacThreshold) instead
//of the force-to-1 clamp for them — clamping their Jgeo is a systematic
//energy bias at disocclusions.
//
//PSS reuse algebra (see Reservoir_v8.hlsli / ReconnectPSS; Pass_temp_merge
//applies the weight/accept/band forms — this file only produces their inputs):
//  weight     w_n = mis * lum(c*vis) * Jn * geomScale / cachedJac_src * W_src
//  accept     F   = c*vis * Jn / cachedNew ; cachedJac = cachedNew ; gBase = Jn
//  band       geometric ratio Jn/gBase only (clamp for base candidates,
//             reject for dual candidates)
//There are NO reuse-time roughness/distance gates: shifts are the fixed maps
//T_k of their samples, glossy receivers self-gate via the BSDF magnitude.

#ifndef TEMPORAL_MERGE_V8_HLSLI
#define TEMPORAL_MERGE_V8_HLSLI

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

//Resolve the candidate and write both direction job descriptors. Caller
//(Pass_temp_gi) has already written TEMP_STATUS_DEAD to SPM_w0 before calling
//this — every return here (including TemporalResolveCand failing) leaves
//that sentinel in place, so Pass_shift_v8 and Pass_temp_merge always
//see a coherent per-pixel status regardless of which path was taken.
void TemporalMergeBody(uint pixelIdx, uint2 launchIndex, int2 candCoord, bool dualIn)
{
    const float2 dims_f = float2(IMG_W, IMG_H);

    //geometry-gate inputs only — the direct-eval loads (Kd/Pr/Pm/matID/flags)
    //are gone along with the direct eval itself (see header).
    const uint   myInstID = load_instID(g_sample_current, pixelIdx);
    const float3 myPos    = load_x1_with_instID(g_sample_current, pixelIdx, myInstID);
    const float3 myN1s    = load_n1_s_with_instID(g_sample_current, pixelIdx, myInstID);
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
        return;   //w0 stays TEMP_STATUS_DEAD (caller's pre-write)

    //reverse job existence: my own reservoir has to be valid + reusable for
    //"my sample evaluated at the neighbour" to mean anything.
    const Reservoir rdi = loadReservoir(g_Reservoirs_current, pixelIdx);
    const bool revValid = IsValidReservoir(rdi) && rdi.W > 0.0f &&
                          (!HYBRID_SHIFT_ON || RcReusable(rdi.rcInfo));

    //====================================
    //JOB DESCRIPTORS (d=0 forward, d=1 reverse — see HashGridHash_v8.hlsli)
    //====================================
    //forward ALWAYS exists: TM_CAND_OK already guarantees a valid, reusable
    //neighbour reservoir at tempPixelIdx. Receiver is me (current); source is
    //the neighbour's LAST-frame reservoir. No pre-ray hint (gBaseHint is
    //compile-time -1 in Pass_shift_v8 for every job, both domains) —
    //temporal bands post-fence in Pass_temp_merge, dual-aware, regardless of
    //walk vs. direct.
    g_pathStateBuffer.Store(SPM_slotS(pixelIdx, 0u), pixelIdx);
    g_pathStateBuffer.Store(SPM_slotZ(pixelIdx, 0u), tempPixelIdx | SPM_BUF_LAST_BIT);

    //reverse only if my sample is valid: receiver is the neighbour (last
    //frame's G-buffer); source is my own CURRENT reservoir.
    g_pathStateBuffer.Store(SPM_slotS(pixelIdx, 1u), tempPixelIdx | SPM_BUF_LAST_BIT);
    g_pathStateBuffer.Store(SPM_slotZ(pixelIdx, 1u), revValid ? pixelIdx : SP_UNDEF);

    g_pathStateBuffer.Store(SPM_w1(pixelIdx),
                            (uint(cand.x) & 0xFFFFu) | ((uint(cand.y) & 0x7FFFu) << 16) |
                            (dual ? 0x80000000u : 0u));
    g_pathStateBuffer.Store(SPM_w0(pixelIdx), TEMP_STATUS_OK);
}

#endif // TEMPORAL_MERGE_V8_HLSLI
