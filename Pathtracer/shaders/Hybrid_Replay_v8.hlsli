//====================================
//HYBRID SHIFT — RANDOM-REPLAY PREFIX
//====================================
//Re-traces the BSDF prefix x_1..x_{k-1} of a reservoir sample at an OFFSET pixel
//using the per-bounce RNG streams (RcBounceSeed) recorded in the reservoir seed.
//Random replay is an identity map in primary sample space (jacobian 1); only the
//final reconnection segment contributes the jacobian bundle, which ReconnectPSS
//evaluates at the replayed vertex y_{k-1}.
//
//Included ONLY by Pass_temp_replay_v8 / Pass_spmis_replay_v8: this is the sole
//reuse-side TraceRay site, so the select/shift passes stay RayQuery-only and the
//replay cost is compacted into the two indirect dispatches.
//
//LIVE-STATE CONTRACT: the walk carries ONLY {HitContext, thr, rayDir} plus the
//three ident words {pathSeed, rcInfo, pinMatID} across its SER reorder points.
//The 48B reservoir payload the reconnection needs is loaded AFTER the last
//trace (loadReservoirPayload inside HybridShiftEval_post) — callers must NOT
//hold a full Reservoir across a HybridShiftEval call: pass the ident words from
//the routing loads and re-load merge state (F/W/M/cachedJac/gBase, or the full
//record) post-call. Nothing writes the reservoir buffers while the replay
//passes run, so every post-walk re-load is bit-identical to a pre-walk load.
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
        //forced lobe (RC_F_LOBES): availability check BEFORE any sampling or
        //tracing — a missing lobe kills the whole walk early. The lobe-clean
        //bdata makes the identical w formula below produce rho_l*cos/p(w|l).
        float3   s;
        BrdfData bdata;
        if (forceLobes)
        {
            const uint lb = RcLobeAt(rcInfo, b);
            if (StrategyP(sp, lb) < EPSILON)
                return rr;
            s     = SampleBRDF_Forced(lb, ctx.matID, -rayDir, ctx.hitNormal, ctx.hitNormal,
                                      ctx.hitLocalKd, ctx.hitLocalPr, ctx.hitLocalPm,
                                      sBsdf, ctx.iors.x, ctx.iors.y, false);
            bdata = EvaluateLobePdf_COMBINED(sp, lb, ctx.matID, ctx.hitNormal, ctx.hitNormal,
                                             s, -rayDir,
                                             ctx.hitLocalKd, ctx.hitLocalPr, ctx.hitLocalPm,
                                             ctx.iors.x, ctx.iors.y, false);
        }
        else
        {
            s     = SampleBRDF(sp, ctx.matID, -rayDir, ctx.hitNormal, ctx.hitNormal,
                               ctx.hitLocalKd, ctx.hitLocalPr, ctx.hitLocalPm,
                               sBsdf, ctx.iors.x, ctx.iors.y, false);
            bdata = EvaluateAndPdf_COMBINED(sp, ctx.matID, ctx.hitNormal, ctx.hitNormal,
                                            s, -rayDir,
                                            ctx.hitLocalKd, ctx.hitLocalPr, ctx.hitLocalPm,
                                            ctx.iors.x, ctx.iors.y, false);
        }

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

//RC_F_ENV_REPLAY tail: consumes a COMPLETED prefix walk, then the FINAL BSDF
//dim is RE-DERIVED from the stream at y_{k-1} (bounce k-1); the trace must MISS
//(doubling as visibility) and sky+sun are re-evaluated along the new direction
//with raygen's exact MIS. Pure PSS: identity replay of every dim, jacobian 1,
//cachedNew = 1 — the returned c IS the shifted F' directly. Touches NO
//reservoir payload at all.
inline float3 EnvReplayTail(
    in ReplayResult rr,
    uint pathSeed, uint rcInfo,
    out float Jn, out float cachedNew)
{
    Jn = 1.0f; cachedNew = 0.0f;

    if (!rr.ok) return (float3)0.0f;

    const uint kk = RcK(rcInfo);
    uint sBsdf = RcBounceSeed(pathSeed, kk - 1u, RC_STREAM_BSDF);

    const SamplingP sp = CalculateStrategyProbabilities(rr.sv.matID, rr.sv.o, rr.sv.n_s,
                                                        (half)rr.sv.etai, (half)rr.sv.etat,
                                                        rr.sv.Kd, (half)rr.sv.Pm);
    //final dim: forced recorded lobe for RC_F_LOBES samples (throughput-side
    //value/pdf go lobe-clean), but the sun MIS weight MUST stay on the
    //MARGINAL pdf — generation folded prev_pdf (marginal) into its MIS.
    float3 s;
    float3 dimVal;
    float  dimPdf;    //throughput-side pdf (conditional under lobes)
    float  misPdf;    //sun-MIS pdf (always marginal)
    if (RcHasLobes(rcInfo))
    {
        const uint lb = RcLobeAt(rcInfo, kk - 1u);
        if (StrategyP(sp, lb) < EPSILON)
            return (float3)0.0f;
        s = SampleBRDF_Forced(lb, rr.sv.matID, rr.sv.o, rr.sv.n_s, rr.sv.n_s,
                              rr.sv.Kd, (half)rr.sv.Pr, (half)rr.sv.Pm,
                              sBsdf, (half)rr.sv.etai, (half)rr.sv.etat, false);
        //ONE fused walk (was EvaluateLobePdf_COMBINED + a second BRDF_PDF_COMBINED
        //pass): the latched pair matches EvaluateLobePdf_COMBINED exactly, and the
        //marginal is the SAME accumulation raygen's miss MIS folded at generation
        //(EvaluateAndPdf_COMBINED_L) — the separate pdf-only walk was the
        //approximate reconstruction, not this.
        float3 lobeVal; float lobePdf;
        const BrdfData mbd = EvaluateAndPdf_COMBINED_L(sp, lb, rr.sv.matID, rr.sv.n_s, rr.sv.n_s,
                                                       s, rr.sv.o,
                                                       rr.sv.Kd, (half)rr.sv.Pr, (half)rr.sv.Pm,
                                                       (half)rr.sv.etai, (half)rr.sv.etat, false,
                                                       lobeVal, lobePdf);
        dimVal = lobeVal;
        dimPdf = lobePdf;
        misPdf = mbd.pdf;
    }
    else
    {
        s = SampleBRDF(sp, rr.sv.matID, rr.sv.o, rr.sv.n_s, rr.sv.n_s,
                       rr.sv.Kd, (half)rr.sv.Pr, (half)rr.sv.Pm,
                       sBsdf, (half)rr.sv.etai, (half)rr.sv.etat, false);
        const BrdfData bdata = EvaluateAndPdf_COMBINED(sp, rr.sv.matID, rr.sv.n_s, rr.sv.n_s,
                                                       s, rr.sv.o,
                                                       rr.sv.Kd, (half)rr.sv.Pr, (half)rr.sv.Pm,
                                                       (half)rr.sv.etai, (half)rr.sv.etat, false);
        dimVal = bdata.val;
        dimPdf = bdata.pdf;
        misPdf = bdata.pdf;
    }
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

    //sky/sun from the replayed bounce origin, raygen's exact MIS
    SetSkyObserver(rayOrigin + sceneOriginWorld);
    const float  sunSAPdf   = GetSunPdf(s);
    const float3 sunRad     = (sunSAPdf > 0.0f) ? EvaluateSun(s) : float3(0, 0, 0);
    const float  sunMisBsdf = (sunSAPdf > 0.0f)
        ? misPdf / max(misPdf + sunSAPdf, EPSILON) : 0.0f;
    float3 cloudTr;
    const float3 sky  = EvaluateSky(s, cloudTr);
    const float3 envL = sky + sunRad * sunMisBsdf * cloudTr;

    float3 c = rr.thr * (dimVal * cosTheta / dimPdf) * envL;
    if (any(isnan(c)) || any(isinf(c)))
        return (float3)0.0f;
    cachedNew = 1.0f;
    return max(c, 0.0f);
}

//Post-walk half of the shift eval: env tail OR payload load + reconnection +
//reuse visibility. The FIRST reservoir-payload touch sits here, after the last
//prefix trace. Returns the SHIFTED PSS numerator c (prefix folded in), Jn,
//cachedNew; c==0 on any failure. `vis` is the reconnection-segment
//transmittance (folded by callers into the shifted target exactly like the
//k==2 inline paths do).
inline float3 HybridShiftEval_post(
    in ReplayResult rr,
    RWByteAddressBuffer resBuf, uint resPx,
    uint pathSeed, uint rcInfo,
    bool  traceVis,
    out float Jn, out float cachedNew)
{
    Jn = 1.0f; cachedNew = 0.0f;

    if (RcEnvReplay(rcInfo))
        return EnvReplayTail(rr, pathSeed, rcInfo, Jn, cachedNew);

    if (!rr.ok) return (float3)0.0f;

    //payload load DEFERRED past the walk (live-state contract)
    Reservoir r = (Reservoir)0;
    loadReservoirPayload(resBuf, resPx, r);

    float3 c = ReconnectPSS_sv(rr.sv, r, Jn, cachedNew);
    if (GetPHat(c) <= 0.0f) return (float3)0.0f;
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

//Full replay-shift evaluation, G-buffer primary context: prefix replay +
//reconnection (or the env-replay tail) + reuse visibility. (resBuf, resPx)
//locate the reservoir whose payload the reconnection consumes — loaded
//post-walk; the caller supplies only the three ident words.
inline float3 HybridShiftEval(
    RWByteAddressBuffer sampleBuf, uint startPx,
    RWByteAddressBuffer resBuf, uint resPx,
    uint pathSeed, uint rcInfo, uint pinMatID,
    bool  traceVis,
    out float Jn, out float cachedNew)
{
    HitContext ctx;
    float3 camToX1;
    Replay_PrimaryCtx(sampleBuf, startPx, ctx, camToX1);
    ReplayResult rr = ReplayWalk(ctx, camToX1, pathSeed, rcInfo, pinMatID);
    return HybridShiftEval_post(rr, resBuf, resPx, pathSeed, rcInfo, traceVis, Jn, cachedNew);
}

//Ctx-hoisted variant: the caller already decoded the receiver's primary hit
//(the temporal merge body's own-pixel entry loads) — skips the SD-record fetch.
inline float3 HybridShiftEvalCtx(
    in HitContext ctx0, float3 camToX1,
    RWByteAddressBuffer resBuf, uint resPx,
    uint pathSeed, uint rcInfo, uint pinMatID,
    bool  traceVis,
    out float Jn, out float cachedNew)
{
    ReplayResult rr = ReplayWalk(ctx0, camToX1, pathSeed, rcInfo, pinMatID);
    return HybridShiftEval_post(rr, resBuf, resPx, pathSeed, rcInfo, traceVis, Jn, cachedNew);
}

#endif // HYBRID_REPLAY_V8_HLSLI
