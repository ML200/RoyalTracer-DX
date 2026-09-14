#ifndef HYBRID_REPLAY_V8_HLSLI
#define HYBRID_REPLAY_V8_HLSLI

#define RC_REPLAY_MAX_BOUNCES 8u

#ifndef REPLAY_HINT_DEPTH
#define REPLAY_HINT_DEPTH 1
#endif

inline dx::HitObject Replay_Trace(RayDesc ray, uint remaining)
{
    TracePayload payload = (TracePayload)0;

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
    float3        thr;
    SurfaceVertex sv;
};

inline ReplayResult DirectShiftResult(in SurfaceVertex sv)
{
    ReplayResult rr;
    rr.ok  = true;
    rr.thr = float3(1, 1, 1);
    rr.sv  = sv;
    return rr;
}

inline void Replay_AnchorLoads() { DeviceMemoryBarrier(); }

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

inline void Replay_PrimaryCtx(RWByteAddressBuffer sampleBuf, uint px,
                              out HitContext ctx, out float3 camToX1)
{
    const SDRecord sd = load_SD(sampleBuf, px);
    Replay_CtxFromLocals(sd.instID, sd.x1, sd.n1_s, sd.matID,
                         (sd.flags & SD_FLAG_BACKFACE) != 0u,
                         sd.Kd, sd.Pr, sd.Pm,
                         ctx, camToX1);
}

// Decide whether the replayed bounce enters subsurface transport.
inline bool Replay_SssEnters(in HitContext ctx, float3 rayDir, uint pathSeed, uint bounce)
{
    if (!LoadIsSSS(ctx.matID)) return false;
    uint sSss = RcBounceSeed(pathSeed, bounce, RC_STREAM_SSS);
    const float fT     = 1.0f - FresnelDielectric(-rayDir, ctx.hitNormal, ctx.iors.x, ctx.iors.y).x;
    const float pEnter = saturate(LoadSSSWeight(ctx.matID) * fT);
    return RandomFloatSingle(sSss) < pEnter;
}

// Rebuild a stored suffix while checking each sampled surface.
inline ReplayResult ReplayWalk(HitContext ctx, float3 camToX1,
                               uint pathSeed, uint rcInfo, uint pinMatID)
{
    ReplayResult rr;
    rr.ok  = false;
    rr.thr = float3(1, 1, 1);
    rr.sv  = (SurfaceVertex)0;

    const uint rcK      = RcK(rcInfo);
    if (rcK < 2u) return rr;
    const uint nBounces = rcK - 2u;
    if (nBounces > RC_REPLAY_MAX_BOUNCES) return rr;

    const bool forceLobes = RcHasLobes(rcInfo);

    float3 rayDir = normalize(camToX1);

    [loop]
    for (uint b = 1u; b <= nBounces; ++b)
    {

        if (Replay_SssEnters(ctx, rayDir, pathSeed, b))
            return rr;

        uint sBsdf = RcBounceSeed(pathSeed, b, RC_STREAM_BSDF);

        const SamplingP sp = CalculateStrategyProbabilities(ctx.matID, -rayDir, ctx.hitNormal,
                                                            ctx.iors.x, ctx.iors.y,
                                                            ctx.hitLocalKd, ctx.hitLocalPr, ctx.hitLocalPm);

        uint strat;
        if (forceLobes)
        {
            const uint lb = RcLobeAt(rcInfo, b);
            if (StrategyP(sp, lb) < EPSILON)
                return rr;
            RandomFloatSingle(sBsdf);
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
            return rr;
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
            return rr;

        const float  hitT   = hit.GetRayTCurrent();
        const uint   instID = hit.GetInstanceID();
        const uint   primID = FlatPrimID(instID, hit.GetGeometryIndex(), hit.GetPrimitiveIndex());
        const uint   matID  = GetMatIDFast(instID, primID);
        BuiltInTriangleIntersectionAttributes attr;
        hit.GetAttributes(attr);
        HitInfo hinfo = EvalSurfaceState(instID, primID, attr.barycentrics, rayOrigin, b);

        if (hinfo.lightID != 0xFFFFFFFFu &&
            any(g_EmissiveTriangles[hinfo.lightID].emission > 0.0f))
            return rr;

        float3 kd; float pr, pm;
        RefetchMaterial(matID, hinfo.uv, kd, pr, pm, b);

        const float  matNi        = LoadNi(matID);
        const bool   transmissive = LoadKd_w(matID) < 1.0f - EPSILON;
        const bool   flipIOR      = hinfo.backface && transmissive && !LoadIsThinGlass(matID);

        ctx.hitPos         = hinfo.hitPos;
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

    rr.thr *= (float3)ctx.absorptionTint;

    {
        const bool mustEnter = IsVolumeVertex(pinMatID) || IsSSSExitVertex(pinMatID);
        const bool enters    = Replay_SssEnters(ctx, rayDir, pathSeed, rcK - 1u);
        if (enters != mustEnter)
            return rr;
    }

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

// Evaluate the final BSDF tail for an environment replay.
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
                                                        rr.sv.Kd, (half)rr.sv.Pr, (half)rr.sv.Pm);

    const bool tailLobe = RcHasLobes(rcInfo);
    uint strat;
    if (tailLobe)
    {
        const uint lb = RcLobeAt(rcInfo, kk - 1u);
        if (StrategyP(sp, lb) < EPSILON)
            return (float3)0.0f;
        RandomFloatSingle(sBsdf);
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
    const float3 dimVal = tailLobe ? lobeVal : mbd.val;
    const float  dimPdf = tailLobe ? lobePdf : mbd.pdf;
    const float  misPdf = mbd.pdf;
    const float cosTheta = abs(dot(rr.sv.n_s, s));
    if (dot(s, s) < 1e-12f || dimPdf <= 1e-6f)
        return (float3)0.0f;

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

    sOut      = s;
    misPdfOut = misPdf;
    ok        = true;
    return rr.thr * (dimVal * cosTheta / dimPdf);
}

// Reconnect replay output and apply optional visibility tracing.
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
    envDir = (float3)0.0f; envMisPdf = 0.0f;

    if (RcEnvReplay(rcInfo))
    {

        bool ok;
        const float3 cPartial = EnvReplayTailPartial(rr, pathSeed, rcInfo, envDir, envMisPdf, ok);
        cachedNew  = 1.0f;
        preVisDead = !ok;
        return ok ? cPartial : (float3)0.0f;
    }

    if (!rr.ok || killPh) return (float3)0.0f;

    float3 c = ReconnectPSS_sv(rr.sv, r, Jn, cachedNew);

    if (gBaseHint >= 0.0f && RcGeomReject(Jn, gBaseHint, jacThreshold))
        return (float3)0.0f;
    if (GetPHat(c) <= 0.0f) return (float3)0.0f;
    preVisDead = false;
    c *= rr.thr;

    if (traceVis && !IsVolumeVertex(r.matID))
    {

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

inline float3 HybridShiftEval_post(
    in ReplayResult rr,
    bool  resLast, uint resPx,
    uint  pathSeed, uint rcInfo,
    bool  traceVis,
    float gBaseHint, float jacThreshold, bool killPh,
    out float Jn, out float cachedNew, out bool preVisDead,
    out float3 envDir, out float envMisPdf)
{

    Reservoir r = (Reservoir)0;
    if (!RcEnvReplay(rcInfo) && rr.ok && !killPh)
        loadReservoirPayloadSel(resLast, resPx, r);
    return HybridShiftEval_post_r(rr, r, pathSeed, rcInfo, traceVis, gBaseHint, jacThreshold, killPh,
                                  Jn, cachedNew, preVisDead, envDir, envMisPdf);
}

#endif
