//====================================
//HYBRID SHIFT — RANDOM-REPLAY PREFIX
//====================================
//Re-traces the BSDF prefix x_1..x_{k-1} of a reservoir sample at an OFFSET pixel
//using the per-bounce RNG streams (RcBounceSeed) recorded in the reservoir seed.
//Random replay is an identity map in primary sample space (jacobian 1); only the
//final reconnection segment contributes the jacobian bundle, which ReconnectPSS
//evaluates at the replayed vertex y_{k-1}.
//
//Included ONLY by Pass_shift_v8.hlsl (ONE binary handling both spatial and
//temporal shift/replay mapping — round 8 unification, round 8b split into two
//compile-time specializations, round 12 reverted back to one binary after
//profiling showed the split dispatched slower despite its lower register
//count): the sole reuse-side TraceRay site for either domain — select/temp_gi
//and the compute merges stay ray-free.
//
//LIVE-STATE CONTRACT: the walk carries ONLY {HitContext, thr, rayDir} plus the
//three ident words {pathSeed, rcInfo, pinMatID} across its SER reorder points.
//The 48B reservoir payload the reconnection needs is loaded AFTER the last
//trace (loadReservoirPayloadSel inside HybridShiftEval_post) — callers must NOT
//hold a full Reservoir across the walk: pass the ident words from the routing
//loads and re-load merge state (F/W/M/cachedJac/gBase, or the full record)
//post-call. Nothing writes the reservoir buffers while the walk passes run, so
//every post-walk re-load is bit-identical to a pre-walk load.
//
//ENFORCEMENT (Replay_AnchorLoads): the deferral above is only real if the
//compiler cannot undo it. The reservoir buffers are NOT globallycoherent (and
//the reuse passes drop it from the scratch buffers for L1 reads), so plain
//"reload after the trace" code is legal to fold back: DXC store-to-load
//forwards / CSEs a post-walk reload into the pre-walk load of the same address,
//and the driver may hoist fresh post-walk loads ABOVE the traces to hide their
//latency — either way the full records ride every reorder point again and the
//contract silently dies (measured ~600B live state vs raygen's ~230B). Callers
//MUST place Replay_AnchorLoads() between the last trace of a walk and the
//first deferred load; loads sequenced after the fence cannot move above it.
//
//CALL-SITE BUDGET: ReplayWalk and HybridShiftEval_post inline the full
//material stack (4-lobe eval + forced-lobe variants, ReconnectPSS). Each pass
//keeps exactly ONE call site of each — route every direction/role through the
//same call with selected arguments (a direct k==2 reconnection is the
//{ok, thr=1, sv=receiver} degenerate of the same map). Two call sites double
//the shader; the old split passes carried 2x walks + 2-4x reconnects each,
//which is where the compile time and the I$-thrash (noinstr) stalls lived.
//
//Every shift is the FIXED map T_k of its sample (k rides rcInfo), so NO pin
//criteria are re-derived on the offset side — full reuse across all
//roughnesses; glossy receivers self-gate through the shifted BSDF magnitude
//and the geometric band bounds the rest. A shift is UNDEFINED (returns fail)
//only on STRUCTURAL divergence, where the replay cannot produce y_{k-1}:
//  * the SSS enter-roll at a prefix vertex succeeds    -> walk divergence
//  * the enter-roll at y_{k-1} disagrees with the pin type (volume/exit pins
//    require ENTER, all others require REFLECT)
//  * a prefix trace misses, dies, or lands on an emitter (raygen would have
//    terminated the path there)
//  * an RC_F_LOBES sample's recorded lobe is ABSENT at the offset vertex
//    (strategy probability ~ 0) — checked before sampling/tracing
//
//LOBE-INDEXED samples (RC_F_LOBES, supp §1) replay DETERMINISTICALLY in lobe
//space: the recorded lobe is FORCED per bounce (the selection draw is burned
//for stream alignment, the direction dims replay within the lobe), and the
//throughput takes the lobe-clean rho_l*cos/p(w|l) matching the lobe-clean
//stored F. No strategy re-roll can flip lobes at offset pixels — the shift
//map is continuous across the multilobe seam — and the per-bounce eval drops
//from the full 4-lobe marginal walk to one lobe + the upper-layer
//transmittances.

#ifndef HYBRID_REPLAY_V8_HLSLI
#define HYBRID_REPLAY_V8_HLSLI

//hard bound on replayable prefix bounces (k-2 <= this); rs_rcMaxK is host-clamped
//to RC_REPLAY_MAX_BOUNCES+2 so raygen can never pin deeper than replay can walk.
#define RC_REPLAY_MAX_BOUNCES 8u

//SER hint for the walk traces. Raygen's TraceRay_Custom sorts on
//(hit | instID&0x3F) — right for its population, but replay threads ALSO
//diverge on how many loop trips REMAIN: a lane that exits the bounce loop
//idles at the wave join, and the reorder cannot retire it. REPLAY_HINT_DEPTH=1
//re-partitions the same 7 hint bits depth-major — (hit | remaining<<3 |
//instID&7) — so lanes with equal walk futures group together; instID low bits
//keep some surface-fetch coherence within a depth class. Execution-order only,
//output is bit-identical either way; set 0 to A/B raygen's original hint.
#ifndef REPLAY_HINT_DEPTH
#define REPLAY_HINT_DEPTH 1
#endif

inline dx::HitObject Replay_Trace(RayDesc ray, uint remaining)
{
    TracePayload payload = (TracePayload)0;
    //RayContribution=0, MultiplierForGeometry=1, MissIndex=0 (mirrors TraceRay_Custom)
    dx::HitObject hit = dx::HitObject::TraceRay(SceneBVH, RAY_FLAG_FORCE_OMM_2_STATE,
                                                0xFF, 0, 1, 0, ray, payload);
#if REPLAY_HINT_DEPTH
    const uint hint = hit.IsHit()
        ? (0x40u | (min(remaining, 7u) << 3) | (hit.GetInstanceID() & 0x7u)) : 0u;
#else
    const uint hint = hit.IsHit() ? (0x40u | (hit.GetInstanceID() & 0x3Fu)) : 0u;
#endif
    dx::MaybeReorderThread(hit, hint, 7);
    return hit;
}

struct ReplayResult
{
    bool          ok;
    float3        thr;      //pdf-divided prefix throughput  prod(f*cos/pdf), RR-free
    SurfaceVertex sv;       //y_{k-1}, ready for ReconnectPSS (o points at y_{k-2})
};

//Direct-shift wrapper: a k==2 reconnection is the walk-free degenerate of the
//replay map. Funnelling it through the SAME HybridShiftEval_post call site as
//the replayed shifts keeps ReconnectPSS inlined once per pass.
inline ReplayResult DirectShiftResult(in SurfaceVertex sv)
{
    ReplayResult rr;
    rr.ok  = true;
    rr.thr = float3(1, 1, 1);
    rr.sv  = sv;
    return rr;
}

//Acquire fence between a walk's LAST trace and the first deferred merge-state
//load (see the header's ENFORCEMENT note). Loads sequenced after this cannot
//be hoisted above it and cannot be satisfied by forwarding a pre-walk load, so
//the post-walk placement is enforced rather than advisory. Cheap: the walk has
//no pending UAV stores for the fence to drain, and the reloads still come from
//L1 (the fence orders, it does not decohere).
inline void Replay_AnchorLoads() { DeviceMemoryBarrier(); }

//Bounce-loop context from already-decoded primary-hit fields. The temporal merge
//body's own-pixel entry loads use the exact same decode helpers as load_SD
//(documented bit-identical in Sample_Data_v8.hlsli), so feeding them here skips
//the redundant SD-record fetch. Mirrors Pass_camera's derivation; the camera->x1
//absorption is reconstructed from the primary distance since the HOT2 extras
//only exist for the current frame's own pixel.
inline void Replay_CtxFromLocals(uint instID, float3 x1, float3 n1s, uint matID,
                                 bool backface, float3 kd, float pr, float pm,
                                 out HitContext ctx, out float3 camToX1)
{
    ctx = (HitContext)0;
    ctx.instID     = instID;
    ctx.hitPos     = x1;
    ctx.hitNormal  = n1s;
    ctx.matID      = matID;
    ctx.backface   = backface;
    ctx.hitLocalKd = (half3)kd;
    ctx.hitLocalPr = (half)pr;
    ctx.hitLocalPm = (half)pm;

    const float  matNi        = LoadNi(matID);
    const bool   transmissive = LoadKd_w(matID) < 1.0f - EPSILON;
    const bool   flipIOR      = backface && transmissive && !LoadIsThinGlass(matID);
    ctx.iors        = (half2)(flipIOR ? float2(matNi, 1.0f) : float2(1.0f, matNi));
    ctx.mediumMatID = flipIOR ? matID : MEDIUM_INVALID;

    camToX1 = x1 - InitOrigin();
    ctx.absorptionTint = (half3)((ctx.mediumMatID != MEDIUM_INVALID)
        ? CalculateAbsorptionThroughput(LoadTf(matID), length(camToX1))
        : float3(1, 1, 1));
}

//Rebuild the bounce-loop context from a baked G-buffer record (the offset
//pixel's primary hit).
inline void Replay_PrimaryCtx(RWByteAddressBuffer sampleBuf, uint px,
                              out HitContext ctx, out float3 camToX1)
{
    const SDRecord sd = load_SD(sampleBuf, px);
    Replay_CtxFromLocals(sd.instID, sd.x1, sd.n1_s, sd.matID,
                         (sd.flags & SD_FLAG_BACKFACE) != 0u,
                         sd.Kd, sd.Pr, sd.Pm,
                         ctx, camToX1);
}

//SSS enter-roll at a replayed vertex: reproduces raygen's stream draw exactly.
//Returns true when the offset path would ENTER the medium at this vertex.
inline bool Replay_SssEnters(in HitContext ctx, float3 rayDir, uint pathSeed, uint bounce)
{
    if (!LoadIsSSS(ctx.matID)) return false;
    uint sSss = RcBounceSeed(pathSeed, bounce, RC_STREAM_SSS);
    const float fT     = 1.0f - FresnelDielectric(-rayDir, ctx.hitNormal, ctx.iors.x, ctx.iors.y).x;
    const float pEnter = saturate(LoadSSSWeight(ctx.matID) * fT);
    return RandomFloatSingle(sSss) < pEnter;
}

//Replay bounces 1..rcK-2 from the prebuilt primary context. `pinMatID` is the
//reservoir's pin matID (the enter-roll agreement check at y_{k-1}) — with
//pathSeed/rcInfo these three words are the ONLY reservoir state in the walk.
inline ReplayResult ReplayWalk(HitContext ctx, float3 camToX1,
                               uint pathSeed, uint rcInfo, uint pinMatID)
{
    ReplayResult rr;
    rr.ok  = false;
    rr.thr = float3(1, 1, 1);
    rr.sv  = (SurfaceVertex)0;

    //prefix walk = bounces 1..k-2 ONLY. RcReplayLen is the ROUTING metric — it
    //adds the env-replay tail dim so k==2 env candidates still queue, but that
    //dim belongs to EnvReplayTail, never this loop (walking it here traced one
    //bounce past y_{k-1} and double-consumed the tail's stream).
    const uint rcK      = RcK(rcInfo);
    if (rcK < 2u) return rr;
    const uint nBounces = rcK - 2u;
    if (nBounces > RC_REPLAY_MAX_BOUNCES) return rr;

    //supp §1: forced-lobe replay for lobe-indexed samples
    const bool forceLobes = RcHasLobes(rcInfo);

    float3 rayDir = normalize(camToX1);            //incoming at y_1 (view direction)

    [loop]
    for (uint b = 1u; b <= nBounces; ++b)
    {
        //walk divergence: the base prefix never entered an SSS medium (it would
        //have pinned there); the offset must not either.
        if (Replay_SssEnters(ctx, rayDir, pathSeed, b))
            return rr;

        uint sBsdf = RcBounceSeed(pathSeed, b, RC_STREAM_BSDF);

        const SamplingP sp = CalculateStrategyProbabilities(ctx.matID, -rayDir, ctx.hitNormal,
                                                            ctx.iors.x, ctx.iors.y,
                                                            ctx.hitLocalKd, ctx.hitLocalPm);
        //FUSED forced/free bounce (code-size: the sampling and eval stacks
        //inline ONCE, not per branch — the old if/else pair doubled them):
        //  forced lobe (RC_F_LOBES): availability check BEFORE any sampling or
        //  tracing — a missing lobe kills the whole walk early — then the
        //  selection draw is burned for stream alignment and the RECORDED lobe
        //  sampled; free mode consumes the same draw to select. Identical draw
        //  order either way. The _L walk's latched pair IS the old
        //  EvaluateLobePdf_COMBINED result; its marginal IS the old
        //  EvaluateAndPdf_COMBINED — the select below is value-identical to
        //  the old branch pair.
        uint strat;
        if (forceLobes)
        {
            const uint lb = RcLobeAt(rcInfo, b);
            if (StrategyP(sp, lb) < EPSILON)
                return rr;
            RandomFloatSingle(sBsdf);   //burn the SelectSamplingStrategy draw
            strat = lb;
        }
        else
        {
            strat = SelectSamplingStrategy(sp, sBsdf);
        }
        const float3 s = SampleBRDF_WithStrategy(strat, ctx.matID, -rayDir, ctx.hitNormal, ctx.hitNormal,
                                                 ctx.hitLocalKd, ctx.hitLocalPr, ctx.hitLocalPm,
                                                 sBsdf, ctx.iors.x, ctx.iors.y, false);
        float3 lobeVal; float lobePdf;
        const BrdfData mbd = EvaluateAndPdf_COMBINED_L(sp, forceLobes ? strat : 0xFFFFFFFFu,
                                                       ctx.matID, ctx.hitNormal, ctx.hitNormal,
                                                       s, -rayDir,
                                                       ctx.hitLocalKd, ctx.hitLocalPr, ctx.hitLocalPm,
                                                       ctx.iors.x, ctx.iors.y, false,
                                                       lobeVal, lobePdf);
        BrdfData bdata;
        bdata.val = forceLobes ? lobeVal : mbd.val;
        bdata.pdf = forceLobes ? lobePdf : mbd.pdf;

        const float  cosTheta = abs(dot(ctx.hitNormal, s));
        const float3 w        = (bdata.pdf > 1e-6f)
            ? (bdata.val * ctx.absorptionTint * cosTheta) / bdata.pdf
            : float3(0, 0, 0);
        if (dot(s, s) < 1e-12f || bdata.pdf <= 1e-6f || any(isnan(w)) || any(isinf(w)) || all(w <= 0.0f))
            return rr;                                 //dead BSDF sample
        rr.thr *= w;

        const float3 offsetN   = (dot(s, ctx.hitNormal) >= 0.0f) ? ctx.hitNormal : -ctx.hitNormal;
        const float3 rayOrigin = offset_ray(ctx.hitPos, offsetN);
        if (!IsRayValid(rayOrigin, s, 10000.0f))
            return rr;

        RayDesc ray;
        ray.Origin    = rayOrigin;
        ray.Direction = s;
        ray.TMin      = 0.00001f;
        ray.TMax      = RAY_TMAX_PLANET;
        dx::HitObject hit = Replay_Trace(ray, nBounces - b);
        if (!hit.IsHit())
            return rr;                                 //prefix escaped -> no y_{b+1}

        const float  hitT   = hit.GetRayTCurrent();
        const uint   instID = hit.GetInstanceID();
        const uint   primID = FlatPrimID(instID, hit.GetGeometryIndex(), hit.GetPrimitiveIndex());
        const uint   matID  = GetMatIDFast(instID, primID);
        BuiltInTriangleIntersectionAttributes attr;
        hit.GetAttributes(attr);
        HitInfo hinfo = EvalSurfaceState(instID, primID, attr.barycentrics, rayOrigin, b);

        //raygen terminates on emissive hits — a base prefix vertex is never an
        //emitter, so an offset prefix landing on one is a structural divergence.
        if (hinfo.lightID != 0xFFFFFFFFu &&
            any(g_EmissiveTriangles[hinfo.lightID].emission > 0.0f))
            return rr;

        float3 kd; float pr, pm;
        RefetchMaterial(matID, hinfo.uv, kd, pr, pm, b);

        const float  matNi        = LoadNi(matID);
        const bool   transmissive = LoadKd_w(matID) < 1.0f - EPSILON;
        const bool   flipIOR      = hinfo.backface && transmissive && !LoadIsThinGlass(matID);

        ctx.hitPos         = rayOrigin + s * hitT;
        ctx.hitNormal      = hinfo.hitNormal;
        ctx.matID          = matID;
        ctx.instID         = instID;
        ctx.backface       = hinfo.backface;
        ctx.hitLocalKd     = (half3)kd;
        ctx.hitLocalPr     = (half)pr;
        ctx.hitLocalPm     = (half)pm;
        ctx.iors           = (half2)(flipIOR ? float2(matNi, 1.0f) : float2(1.0f, matNi));
        ctx.mediumMatID    = flipIOR ? matID : MEDIUM_INVALID;
        ctx.absorptionTint = (half3)((ctx.mediumMatID != MEDIUM_INVALID)
            ? CalculateAbsorptionThroughput(LoadTf(matID), hitT)
            : float3(1, 1, 1));
        rayDir = s;
    }

    //fold the FINAL traced segment's Beer-Lambert arrival tint: raygen folds the
    //tint arriving at x_{k-1} into bounce k-1's updateWeight, whose f*cos/pdf part
    //the reconnection re-evaluates — the tint itself has no other carrier. Leaving
    //it pending makes every k>2 shift through absorbing media too bright vs the
    //anchor target.
    rr.thr *= (float3)ctx.absorptionTint;

    //enter-roll agreement at y_{k-1}: volume/exit pins entered the medium there,
    //every other pin type reflected. (Entry-surface pins store a plain surface
    //matID and reconnect INTO the frozen entry vertex, so they take the reflect
    //branch here — the walk beyond the pin is suffix.)
    {
        const bool mustEnter = IsVolumeVertex(pinMatID) || IsSSSExitVertex(pinMatID);
        const bool enters    = Replay_SssEnters(ctx, rayDir, pathSeed, rcK - 1u);
        if (enters != mustEnter)
            return rr;
    }

    //package y_{k-1} for ReconnectPSS: o points back along the arriving segment
    rr.sv.x     = ctx.hitPos;
    rr.sv.n_s   = ctx.hitNormal;
    rr.sv.o     = -rayDir;
    rr.sv.matID = ctx.matID;
    rr.sv.Kd    = (float3)ctx.hitLocalKd;
    rr.sv.Pr    = (float)ctx.hitLocalPr;
    rr.sv.Pm    = (float)ctx.hitLocalPm;
    rr.sv.uv    = float2(0, 0);
    rr.sv.etai  = (float)ctx.iors.x;
    rr.sv.etat  = (float)ctx.iors.y;
    rr.ok = true;
    return rr;
}

//RC_F_ENV_REPLAY tail — SHIFT-SIDE HALF (the atmosphere eval is DEFERRED to the
//merges; see EnvTailFinish in SunSampler_v8.hlsli). Consumes a COMPLETED prefix
//walk, RE-DERIVES the FINAL BSDF dim from the stream at y_{k-1} (bounce k-1),
//and traces the must-miss (which doubles as the env visibility test). Returns
//the PARTIAL contribution cPartial = thr·dimVal·cosTheta/dimPdf and hands back
//the offset direction `sOut` + its MARGINAL BSDF pdf `misPdfOut`; the merge
//finishes c = cPartial·envL(sOut) with sky+sun re-evaluated at sOut (per-pixel
//observer). Splitting it this way keeps the atmosphere LUT/march stack
//(SetSkyObserver/EvaluateSky/EvaluateSun/GetSunPdf) OUT of the register-tight RT
//shift binary — inlined there it set the occupancy ceiling for ALL shift lanes,
//not just env-replay ones. Pure PSS: identity replay of every dim, jacobian 1,
//cachedNew = 1 (the caller stamps those). `ok` is false on any structural
//failure (dead tail dim / trace hit); sOut/misPdfOut are then meaningless and
//the caller writes cPartial = 0 as the merge's dead sentinel. Touches NO payload.
inline float3 EnvReplayTailPartial(
    in ReplayResult rr,
    uint pathSeed, uint rcInfo,
    out float3 sOut, out float misPdfOut, out bool ok)
{
    sOut = float3(0, 0, 1); misPdfOut = 0.0f; ok = false;

    if (!rr.ok) return (float3)0.0f;

    const uint kk = RcK(rcInfo);
    uint sBsdf = RcBounceSeed(pathSeed, kk - 1u, RC_STREAM_BSDF);

    const SamplingP sp = CalculateStrategyProbabilities(rr.sv.matID, rr.sv.o, rr.sv.n_s,
                                                        (half)rr.sv.etai, (half)rr.sv.etat,
                                                        rr.sv.Kd, (half)rr.sv.Pm);
    //final dim, FUSED forced/free exactly like the walk bounce above (one
    //sampling + one _L eval inline): forced recorded lobe for RC_F_LOBES
    //samples (throughput-side value/pdf go lobe-clean), but the sun MIS weight
    //MUST stay on the MARGINAL pdf — generation folded prev_pdf (marginal)
    //into its MIS, and the _L marginal is the SAME accumulation raygen's miss
    //MIS folded at generation. misPdf (=that marginal) is handed to the merge.
    const bool tailLobe = RcHasLobes(rcInfo);
    uint strat;
    if (tailLobe)
    {
        const uint lb = RcLobeAt(rcInfo, kk - 1u);
        if (StrategyP(sp, lb) < EPSILON)
            return (float3)0.0f;
        RandomFloatSingle(sBsdf);   //burn the SelectSamplingStrategy draw
        strat = lb;
    }
    else
    {
        strat = SelectSamplingStrategy(sp, sBsdf);
    }
    const float3 s = SampleBRDF_WithStrategy(strat, rr.sv.matID, rr.sv.o, rr.sv.n_s, rr.sv.n_s,
                                             rr.sv.Kd, (half)rr.sv.Pr, (half)rr.sv.Pm,
                                             sBsdf, (half)rr.sv.etai, (half)rr.sv.etat, false);
    float3 lobeVal; float lobePdf;
    const BrdfData mbd = EvaluateAndPdf_COMBINED_L(sp, tailLobe ? strat : 0xFFFFFFFFu,
                                                   rr.sv.matID, rr.sv.n_s, rr.sv.n_s,
                                                   s, rr.sv.o,
                                                   rr.sv.Kd, (half)rr.sv.Pr, (half)rr.sv.Pm,
                                                   (half)rr.sv.etai, (half)rr.sv.etat, false,
                                                   lobeVal, lobePdf);
    const float3 dimVal = tailLobe ? lobeVal : mbd.val;   //throughput-side (conditional under lobes)
    const float  dimPdf = tailLobe ? lobePdf : mbd.pdf;
    const float  misPdf = mbd.pdf;                        //sun-MIS pdf (always marginal)
    const float cosTheta = abs(dot(rr.sv.n_s, s));
    if (dot(s, s) < 1e-12f || dimPdf <= 1e-6f)
        return (float3)0.0f;

    //the re-derived ray must ESCAPE — a hit is a structural divergence (and the
    //miss requirement doubles as the visibility test)
    const float3 offsetN   = (dot(s, rr.sv.n_s) >= 0.0f) ? rr.sv.n_s : -rr.sv.n_s;
    const float3 rayOrigin = offset_ray(rr.sv.x, offsetN);
    if (!IsRayValid(rayOrigin, s, 10000.0f))
        return (float3)0.0f;
    RayDesc ray;
    ray.Origin    = rayOrigin;
    ray.Direction = s;
    ray.TMin      = 0.00001f;
    ray.TMax      = RAY_TMAX_PLANET;
    dx::HitObject hit = Replay_Trace(ray, 0u);
    if (hit.IsHit())
        return (float3)0.0f;

    //DEFERRED: the merge evaluates sky+sun at sOut (per-pixel observer) and
    //multiplies. We only hand back the partial throughput + the direction + pdf.
    sOut      = s;
    misPdfOut = misPdf;
    ok        = true;
    return rr.thr * (dimVal * cosTheta / dimPdf);
}

//Post half of the shift eval, GIVEN an already-selected payload Reservoir
//(garbage/zero is fine when RcEnvReplay(rcInfo) — the env tail never reads
//it). This is the split-out reconnect MATH: env tail OR reconnection + reuse
//visibility, ONE call site regardless of how many buffers a caller's payload
//could have come from — see HybridShiftEval_post below for why that split
//matters. Same contract as before: caller calls Replay_AnchorLoads() between
//the walk and loading `r`; gBaseHint/killPh/preVisDead semantics unchanged.
inline float3 HybridShiftEval_post_r(
    in ReplayResult rr,
    in Reservoir r,
    uint  pathSeed, uint rcInfo,
    bool  traceVis,
    float gBaseHint, float jacThreshold, bool killPh,
    out float Jn, out float cachedNew, out bool preVisDead,
    out float3 envDir, out float envMisPdf)
{
    Jn = 1.0f; cachedNew = 0.0f; preVisDead = true;
    envDir = (float3)0.0f; envMisPdf = 0.0f;   //only the env-replay branch fills these

    if (RcEnvReplay(rcInfo))
    {
        //DEFERRED env tail: the shift produces the PARTIAL contribution + the
        //offset direction, the MERGE finishes sky+sun (EnvTailFinish). cachedNew/
        //Jn are the env constant 1; final dead-ness is decided in the merge from
        //the finished c. preVisDead just tracks the structural-failure sentinel
        //(!ok -> cPartial 0). The must-miss trace inside IS the visibility.
        bool ok;
        const float3 cPartial = EnvReplayTailPartial(rr, pathSeed, rcInfo, envDir, envMisPdf, ok);
        cachedNew  = 1.0f;   //Jn already 1
        preVisDead = !ok;
        return ok ? cPartial : (float3)0.0f;
    }

    if (!rr.ok || killPh) return (float3)0.0f;

    float3 c = ReconnectPSS_sv(rr.sv, r, Jn, cachedNew);
    //hinted geometric reject (direct paths, gBaseHint >= 0): identical to
    //rejecting after the ray, minus the ray — including the gBase<=0 reject
    //inside RcGeomReject. Replay paths pass -1 and band post-fence instead.
    if (gBaseHint >= 0.0f && RcGeomReject(Jn, gBaseHint, jacThreshold))
        return (float3)0.0f;
    if (GetPHat(c) <= 0.0f) return (float3)0.0f;
    preVisDead = false;
    c *= rr.thr;

    if (traceVis && !IsVolumeVertex(r.matID))
    {
        //thin glass attenuates (1-F)*Tf -> RGB transmittance, matching the k==2 sites
        float3 vis;
        if (r.matID == MATID_ENV_MISS)
        {
            const float3 md = normalize(r.x2);
            vis = VisibilityTransmittance(rr.sv.x, rr.sv.n_s, rr.sv.x + md * RAY_TMAX_PLANET, -md);
        }
        else
        {
            vis = VisibilityTransmittance(rr.sv.x, rr.sv.n_s, r.x2, r.n2_s);
        }
        c *= vis;
    }
    return c;
}

//Convenience wrapper: loads the payload itself, buffer picked by `resLast`.
//  * REPLAY (rr from ReplayWalk): the caller MUST call Replay_AnchorLoads()
//    between the walk and this call (payload load is a live-state-contract
//    deferral, not just a convenience — see the header note).
//  * DIRECT k==2 (rr = DirectShiftResult(receiver sv)): no walk, no fence.
//  gBaseHint/killPh/rcInfo/pathSeed semantics: see HybridShiftEval_post_r.
//
//resLast MUST be a call-site LITERAL, not a value threaded through from a
//runtime condition (e.g. a per-thread dispatch index) — this is what lets
//loadReservoirPayloadSel's dead branch fold away entirely. A caller whose
//buffer choice IS runtime-only (temporal's forward-vs-reverse) must NOT
//route it through this bool: branch on its own condition with literal `true`/
//`false` arguments to TWO calls of the small loadReservoirPayload load, then
//call HybridShiftEval_post_r ONCE with the resulting value — that keeps the
//expensive reconnect math single-instanced and only duplicates the tiny load
//(see Pass_shift_v8.hlsl's resBufLast branch — runtime for both domains now,
//literal true/false at each of its two call sites — for the pattern). Threading a runtime bool through
//THIS wrapper instead would still compile (HLSL doesn't require the literal),
//but silently reintroduces the exact per-call-site doubling the split above
//exists to avoid — the compiler cannot fold the dead half without the literal,
//and inlining then duplicates everything downstream of it, not just the load.
inline float3 HybridShiftEval_post(
    in ReplayResult rr,
    bool  resLast, uint resPx,
    uint  pathSeed, uint rcInfo,
    bool  traceVis,
    float gBaseHint, float jacThreshold, bool killPh,
    out float Jn, out float cachedNew, out bool preVisDead,
    out float3 envDir, out float envMisPdf)
{
    //payload load DEFERRED past the walk (live-state contract; fence at caller)
    Reservoir r = (Reservoir)0;
    if (!RcEnvReplay(rcInfo) && rr.ok && !killPh)
        loadReservoirPayloadSel(resLast, resPx, r);
    return HybridShiftEval_post_r(rr, r, pathSeed, rcInfo, traceVis, gBaseHint, jacThreshold, killPh,
                                  Jn, cachedNew, preVisDead, envDir, envMisPdf);
}

#endif // HYBRID_REPLAY_V8_HLSLI
