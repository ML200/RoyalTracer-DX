#define SPMIS_GRID_NONCOHERENT   // read-only grid/scratch -> L1-cached (see Includes_v8.hlsli)
#include "Includes_v8.hlsli"
#include "Raygen_Common_v8.hlsli"   // HitContext (replayed bounce-loop ctx)
#include "Hybrid_Replay_v8.hlsli"

//====================================
//UNIFIED SHIFT — SPATIAL + TEMPORAL, ONE BINARY, ONE DISPATCH PER DOMAIN
//====================================
//Performs exactly ONE reconnection-or-replay mapping per thread, entirely
//parameterized by a job descriptor loaded from g_pathStateBuffer at
//(pixelIdx, role) — see the "UNIFIED SHIFT JOB" layout note in
//HashGridHash_v8.hlsli. ONE compiled binary, dispatched TWICE in the pass
//list — once for loop:temporal_shift's old slot (now a plain
//`Pass_shift_v8.hlsl|rg:temporal_shift` entry, Depth=2) and once for
//spatial's (`|rg:spatial_shift`, Depth=Ntn+1) — Renderer_Pipeline.cpp's
//PassIndexByFile dedup (see its header comment) compiles+exports this file
//once and reuses the same SBT record for both. Role is `DispatchRaysIndex().z`,
//i.e. the Depth-dimension index, NOT a root constant: each of the two
//pass-list entries is its OWN single DispatchRays call with a Depth Stage::
//RayGen picks from the entry's "rg:<tag>" annotation (PassDesc::dispatchTag).
//
//WAS SPLIT into two compile-time-specialized binaries (round 8b — one
//folding resBufLast/startBufLast for spatial, one folding useHint/killPh for
//temporal), REVERTED (round 12): Malte's profiling showed one binary
//dispatches FASTER in practice than the two-binary split, despite its higher
//static register count. useHint (the actual register hog the split was built
//to isolate, round 9) is unconditionally false regardless of domain, and
//EnvReplayTail's atmosphere eval (the BIGGER hog, round 10) has been
//deferred out of this shader entirely into the merges — so resBufLast/
//startBufLast/killPh are the only remaining per-domain differences, and
//round 9's own measurement showed folding THOSE saves ~0 registers (buffer
//selection is near-free); reading them at runtime costs effectively nothing.
//
//THEN role dispatch itself was ALSO reverted (round 13, same finding
//generalized): the pass-list `loop:temporal_shift`/`loop:spatial_shift`
//mechanism (round 8) redispatched this SAME shader once per role via a HOST
//loop, with a `barrier` after every iteration — meaning 2 (temporal) or
//Ntn+1 (spatial) fully-serialized DispatchRays calls, each one waiting for
//every previous role's UAV writes before it could even start. A single
//DispatchRays with Depth=roleCount has no such barrier between "slices" —
//the GPU is free to schedule/overlap every role's waves however it likes,
//limited only by real data dependencies (there are none between roles; each
//owns disjoint job-slot planes). This is the same lesson as the round-12
//binary-split reversal one level up: a mechanism that looks like it should
//help (specialization / serialized-but-simpler dispatches) can cost more in
//scheduling/overhead than it saves, and that cost doesn't show up in a
//register or instruction-count measurement — only in wall-clock.
//
//WEIGHTS LIVE IN MERGE, NOT HERE: this pass produces the raw shifted
//{c, Jn, cachedNew} PSS numerator/jacobian and nothing else — no MIS weight,
//no reservoir accept, no domain-specific banding. Pass_spmis_merge and
//Pass_temp_merge each read these same three outputs and apply their OWN
//(genuinely different) weight algebra.

[shader("raygeneration")]
void Pass_shift_v8()
{
    const uint2  launchIndex = DispatchRaysIndex().xy;
    const float2 dims        = float2(IMG_W, IMG_H);
    const uint   pixelIdx    = MapPixelID(dims, launchIndex);
    const uint   d           = DispatchRaysIndex().z;

    //no job routed to this (pixel, role) this frame — select/temp_gi write
    //SP_UNDEF for every slot with nothing to do, so this is the ONLY gate.
    const uint resRaw = g_pathStateBuffer.Load(SPM_slotZ(pixelIdx, d));
    if (resRaw == SP_UNDEF) return;

    const uint  resPx      = resRaw & SPM_RESPX_MASK;
    const bool  resBufLast = (resRaw & SPM_BUF_LAST_BIT) != 0u;
    const bool  killPh     = (resRaw & SPM_KILLPH_BIT) != 0u;   //spatial-only legacy bit (select, hybrid-off only) — temporal never sets it, reads false there
    //useHint's pre-visibility geometric reject is COMPILE-TIME FALSE (round 9 —
    //held gBaseHint live across the ReconnectPSS_sv high-water mark plus a
    //divergent early-out -> measured ~+30 VGPRs). The reject is re-derived
    //post-fence in the merges instead (Pass_spmis_merge direct-draw dead
    //check / Pass_temp_merge), same Jn/gBase/threshold -> render-identical.
    const bool  useHint    = false;

    const uint startRaw     = g_pathStateBuffer.Load(SPM_slotS(pixelIdx, d));
    const uint startPx      = startRaw & 0x7FFFFFFFu;
    const bool startBufLast = (startRaw & SPM_BUF_LAST_BIT) != 0u;

    //needWalk is derived, not stored: the shifted reservoir's OWN rcInfo says
    //whether its pin sits beyond a direct (k==2) reconnection.
    const uint rcInfo = resBufLast ? load_rcInfo(g_Reservoirs_last, resPx)
                                   : load_rcInfo(g_Reservoirs_current, resPx);
    const bool needWalk = RcReplayLen(rcInfo) > 0u;

    ReplayResult rr;
    uint pathSeed = 0u;
    if (needWalk)
    {
        pathSeed = resBufLast ? load_seed(g_Reservoirs_last, resPx) : load_seed(g_Reservoirs_current, resPx);
        const uint pinMat = resBufLast ? load_matID_res(g_Reservoirs_last, resPx)
                                       : load_matID_res(g_Reservoirs_current, resPx);
        HitContext ctx;
        float3     camToX1;
        if (startBufLast)
            Replay_PrimaryCtx(g_sample_last, startPx, ctx, camToX1);
        else
            Replay_PrimaryCtx(g_sample_current, startPx, ctx, camToX1);
        rr = ReplayWalk(ctx, camToX1, pathSeed, rcInfo, pinMat);
        Replay_AnchorLoads();   //everything below stays post-trace
    }
    else
    {
        //HLSL's ternary can't return a struct, so if/else (SurfaceVertex, not
        //a numeric type — see BSDF_term_sel's identical constraint elsewhere).
        const float3 camPos = InitOrigin();
        SurfaceVertex sv;
        if (startBufLast) sv = BuildVertex(g_sample_last,    startPx, load_x1(g_sample_last,    startPx), camPos);
        else               sv = BuildVertex(g_sample_current, startPx, load_x1(g_sample_current, startPx), camPos);
        rr = DirectShiftResult(sv);
    }

    //pre-visibility-ray geometric reject: useHint is compile-time false (above),
    //so this ternary folds to gBaseHint = -1.0f and the reject branch in
    //HybridShiftEval_post_r dies (load_gBase deleted with it). The reject is
    //re-derived post-fence in the merges instead (Pass_spmis_merge direct-draw
    //dead check / Pass_temp_merge) -- that relocation is the round-9 VGPR win.
    const float gBaseHint = useHint
        ? (resBufLast ? load_gBase(g_Reservoirs_last, resPx) : load_gBase(g_Reservoirs_current, resPx))
        : -1.0f;

    Reservoir r = (Reservoir)0;
    if (!RcEnvReplay(rcInfo) && rr.ok && !killPh)
    {
        if (resBufLast) loadReservoirPayload(g_Reservoirs_last,    resPx, r);
        else            loadReservoirPayload(g_Reservoirs_current, resPx, r);
    }

    float Jn, cachedNew;
    bool  preVisDead;
    float3 envDir; float envMisPdf;
    const float3 c = HybridShiftEval_post_r(rr, r, pathSeed, rcInfo, true,
                                            gBaseHint, spmis_jacThreshold, killPh,
                                            Jn, cachedNew, preVisDead, envDir, envMisPdf);

    //slotJ carries {cachedNew, Jn} for reconnect/direct jobs — preVisDead rides
    //Jn's sign (Jn is always > 0 in every defined path, walk or direct) so merge
    //recovers the direct-vs-replay dead/materialize asymmetry without a fourth
    //plane. For DEFERRED env-replay it instead carries {PackNormal(offset dir),
    //misPdf}: cachedNew/Jn are the env constant 1 (merge hardcodes them) and the
    //merge finishes the sky+sun eval from these two words + slotC's partial
    //contribution (EnvTailFinish) — keeping the atmosphere stack out of this RT
    //binary. slotC holds cPartial for env-replay, the full c otherwise.
    uint2 jWord;
    if (RcEnvReplay(rcInfo))
        jWord = uint2(PackNormal(envDir), asuint(envMisPdf));
    else
        jWord = uint2(asuint(cachedNew), asuint(preVisDead ? -Jn : Jn));
    g_pathStateBuffer.Store2(SPM_slotJ(pixelIdx, d), jWord);
    g_pathStateBuffer.Store3(SPM_slotC(pixelIdx, d), asuint(c));
}
