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

#ifndef HYBRID_REPLAY_V8_HLSLI
#define HYBRID_REPLAY_V8_HLSLI

//hard bound on replayable prefix bounces (k-2 <= this); rs_rcMaxK is host-clamped
//to RC_REPLAY_MAX_BOUNCES+2 so raygen can never pin deeper than replay can walk.
#define RC_REPLAY_MAX_BOUNCES 8u

struct ReplayResult
{
    bool          ok;
    float3        thr;      //pdf-divided prefix throughput  prod(f*cos/pdf), RR-free
    SurfaceVertex sv;       //y_{k-1}, ready for ReconnectPSS (o points at y_{k-2})
};

//Rebuild the bounce-loop context from a baked G-buffer record (the offset
//pixel's primary hit). Mirrors Pass_camera's derivation; the camera->x1
//absorption is reconstructed from the primary distance since the HOT2 extras
//only exist for the current frame's own pixel.
inline void Replay_PrimaryCtx(RWByteAddressBuffer sampleBuf, uint px,
                              out HitContext ctx, out float3 camToX1)
{
    const SDRecord sd = load_SD(sampleBuf, px);
    ctx = (HitContext)0;
    ctx.instID     = sd.instID;
    ctx.hitPos     = sd.x1;
    ctx.hitNormal  = sd.n1_s;
    ctx.matID      = sd.matID;
    ctx.backface   = (sd.flags & SD_FLAG_BACKFACE) != 0u;
    ctx.hitLocalKd = (half3)sd.Kd;
    ctx.hitLocalPr = (half)sd.Pr;
    ctx.hitLocalPm = (half)sd.Pm;

    const float  matNi        = LoadNi(sd.matID);
    const bool   transmissive = LoadKd_w(sd.matID) < 1.0f - EPSILON;
    const bool   flipIOR      = ctx.backface && transmissive && !LoadIsThinGlass(sd.matID);
    ctx.iors        = (half2)(flipIOR ? float2(matNi, 1.0f) : float2(1.0f, matNi));
    ctx.mediumMatID = flipIOR ? sd.matID : MEDIUM_INVALID;

    camToX1 = sd.x1 - InitOrigin();
    ctx.absorptionTint = (half3)((ctx.mediumMatID != MEDIUM_INVALID)
        ? CalculateAbsorptionThroughput(LoadTf(sd.matID), length(camToX1))
        : float3(1, 1, 1));
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

//Replay bounces 1..rcK-2 of `r` starting at the primary surface of `startPx`
//(sampleBuf selects the current or the prior frame's G-buffer).
inline ReplayResult ReplayPrefix(
    RWByteAddressBuffer sampleBuf, uint startPx,
    in Reservoir r)
{
    ReplayResult rr;
    rr.ok  = false;
    rr.thr = float3(1, 1, 1);
    rr.sv  = (SurfaceVertex)0;

    const uint rcK      = RcK(r.rcInfo);
    const uint nBounces = RcReplayLen(r.rcInfo);
    if (rcK < 2u || nBounces > RC_REPLAY_MAX_BOUNCES) return rr;

    HitContext ctx;
    float3 camToX1;
    Replay_PrimaryCtx(sampleBuf, startPx, ctx, camToX1);

    float3 rayDir = normalize(camToX1);            //incoming at y_1 (view direction)

    [loop]
    for (uint b = 1u; b <= nBounces; ++b)
    {
        //walk divergence: the base prefix never entered an SSS medium (it would
        //have pinned there); the offset must not either.
        if (Replay_SssEnters(ctx, rayDir, r.seed, b))
            return rr;

        uint sBsdf = RcBounceSeed(r.seed, b, RC_STREAM_BSDF);

        const SamplingP sp = CalculateStrategyProbabilities(ctx.matID, -rayDir, ctx.hitNormal,
                                                            ctx.iors.x, ctx.iors.y,
                                                            ctx.hitLocalKd, ctx.hitLocalPm);
        const float3 s       = SampleBRDF(sp, ctx.matID, -rayDir, ctx.hitNormal, ctx.hitNormal,
                                          ctx.hitLocalKd, ctx.hitLocalPr, ctx.hitLocalPm,
                                          sBsdf, ctx.iors.x, ctx.iors.y, false);
        const BrdfData bdata = EvaluateAndPdf_COMBINED(sp, ctx.matID, ctx.hitNormal, ctx.hitNormal,
                                                       s, -rayDir,
                                                       ctx.hitLocalKd, ctx.hitLocalPr, ctx.hitLocalPm,
                                                       ctx.iors.x, ctx.iors.y, false);

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
        dx::HitObject hit = TraceRay_Custom(SceneBVH, ray, RAY_FLAG_FORCE_OMM_2_STATE, 0xFF);
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
        const bool mustEnter = IsVolumeVertex(r.matID) || IsSSSExitVertex(r.matID);
        const bool enters    = Replay_SssEnters(ctx, rayDir, r.seed, rcK - 1u);
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

//RC_F_ENV_REPLAY evaluation: prefix replay, then the FINAL BSDF dim is
//RE-DERIVED from the stream at y_{k-1} (bounce k-1); the trace must MISS
//(doubling as visibility) and sky+sun are re-evaluated along the new direction
//with raygen's exact MIS. Pure PSS: identity replay of every dim, jacobian 1,
//cachedNew = 1 — the returned c IS the shifted F' directly.
inline float3 EnvReplayEval(
    RWByteAddressBuffer sampleBuf, uint startPx,
    in Reservoir r,
    out float Jn, out float cachedNew)
{
    Jn = 1.0f; cachedNew = 0.0f;

    ReplayResult rr = ReplayPrefix(sampleBuf, startPx, r);
    if (!rr.ok) return (float3)0.0f;

    const uint  kk     = RcK(r.rcInfo);
    const float3 rayDir = -rr.sv.o;                       //incoming at y_{k-1}
    uint sBsdf = RcBounceSeed(r.seed, kk - 1u, RC_STREAM_BSDF);

    const SamplingP sp = CalculateStrategyProbabilities(rr.sv.matID, rr.sv.o, rr.sv.n_s,
                                                        (half)rr.sv.etai, (half)rr.sv.etat,
                                                        rr.sv.Kd, (half)rr.sv.Pm);
    const float3 s       = SampleBRDF(sp, rr.sv.matID, rr.sv.o, rr.sv.n_s, rr.sv.n_s,
                                      rr.sv.Kd, (half)rr.sv.Pr, (half)rr.sv.Pm,
                                      sBsdf, (half)rr.sv.etai, (half)rr.sv.etat, false);
    const BrdfData bdata = EvaluateAndPdf_COMBINED(sp, rr.sv.matID, rr.sv.n_s, rr.sv.n_s,
                                                   s, rr.sv.o,
                                                   rr.sv.Kd, (half)rr.sv.Pr, (half)rr.sv.Pm,
                                                   (half)rr.sv.etai, (half)rr.sv.etat, false);
    const float cosTheta = abs(dot(rr.sv.n_s, s));
    if (dot(s, s) < 1e-12f || bdata.pdf <= 1e-6f)
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
    dx::HitObject hit = TraceRay_Custom(SceneBVH, ray, RAY_FLAG_FORCE_OMM_2_STATE, 0xFF);
    if (hit.IsHit())
        return (float3)0.0f;

    //sky/sun from the replayed bounce origin, raygen's exact MIS
    SetSkyObserver(rayOrigin + sceneOriginWorld);
    const float  sunSAPdf   = GetSunPdf(s);
    const float3 sunRad     = (sunSAPdf > 0.0f) ? EvaluateSun(s) : float3(0, 0, 0);
    const float  sunMisBsdf = (sunSAPdf > 0.0f)
        ? bdata.pdf / max(bdata.pdf + sunSAPdf, EPSILON) : 0.0f;
    float3 cloudTr;
    const float3 sky  = EvaluateSky(s, cloudTr);
    const float3 envL = sky + sunRad * sunMisBsdf * cloudTr;

    float3 c = rr.thr * (bdata.val * cosTheta / bdata.pdf) * envL;
    if (any(isnan(c)) || any(isinf(c)))
        return (float3)0.0f;
    cachedNew = 1.0f;
    return max(c, 0.0f);
}

//Full replay-shift evaluation: prefix replay + reconnection (or the env-replay
//tail) + reuse visibility. Returns the SHIFTED PSS numerator c (prefix folded
//in), Jn, cachedNew; c==0 on any failure. `vis` is the reconnection-segment
//transmittance (folded by callers into the shifted target exactly like the
//k==2 inline paths do).
inline float3 HybridShiftEval(
    RWByteAddressBuffer sampleBuf, uint startPx,
    in Reservoir r,
    bool  traceVis,
    out float Jn, out float cachedNew)
{
    Jn = 1.0f; cachedNew = 0.0f;

    if (RcEnvReplay(r.rcInfo))
        return EnvReplayEval(sampleBuf, startPx, r, Jn, cachedNew);

    ReplayResult rr = ReplayPrefix(sampleBuf, startPx, r);
    if (!rr.ok) return (float3)0.0f;

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

#endif // HYBRID_REPLAY_V8_HLSLI
