#pragma once

// Primary surface replacement for the DLSS-RR guides.

// Lobe share one probe ray represents.
inline float PsrRoughnessFade(float roughness)
{
    return 1.0f - smoothstep(DLSS_PSR_ROUGH_FULL, DLSS_PSR_ROUGH_END, roughness);
}

// Near-delta reflectors incl. glass; SSS keeps the legacy guides.
inline bool PsrCandidateMaterial(uint matID, float pr)
{
    if (LoadIsSSS(matID)) return false;
    return pr < DLSS_PSR_ROUGH_END || (LoadPc(matID) > 0.0f && LoadPcr(matID) < DLSS_PSR_ROUGH_END);
}

// Thin or clear glass; any diffuse share stops the chain.
inline bool PsrSeeThroughMaterial(uint matID)
{
    return !LoadIsSSS(matID) && (LoadIsThinGlass(matID) || LoadKd_w(matID) < EPSILON);
}

inline float3x3 PsrIdentity()
{
    return float3x3(1.0f, 0.0f, 0.0f,
                    0.0f, 1.0f, 0.0f,
                    0.0f, 0.0f, 1.0f);
}

// Householder reflection.
inline float3x3 PsrReflectionMatrix(float3 n)
{
    return float3x3(1.0f - 2.0f * n.x * n.x, -2.0f * n.x * n.y, -2.0f * n.x * n.z,
                    -2.0f * n.y * n.x, 1.0f - 2.0f * n.y * n.y, -2.0f * n.y * n.z,
                    -2.0f * n.z * n.x, -2.0f * n.z * n.y, 1.0f - 2.0f * n.z * n.z);
}

// Delta-chain end as seen along the primary ray.
struct PsrChainEnd
{
    float3   xVirtual;    // end as seen along the primary ray
    float3   xHit;        // actual end (xVirtual for the sky)
    float3   xFirst;      // first hit along the primary ray (plain probe)
    uint     instFirst;   // 0xFFFFFFFF for the sky
    float3   nVirtual;    // end normal through the mirror chain
    float3   Kd;
    float    Pr;
    float    Pm;
    float3   throughput;  // product of the lobe factors
    uint     instID;      // 0xFFFFFFFF for the sky
    uint     flags;       // DLSS_PSR_FLAG_*
    uint     bounces;     // segments traced
    float    paneF;       // Fresnel of a pane ending the chain
    float3x3 M;           // mirror chain transform
};

inline PsrChainEnd PsrChainSurface(float3 x, float3 n, float3 kd, float pr, float pm, uint instID)
{
    PsrChainEnd e;
    e.xVirtual   = x;
    e.xHit       = x;
    e.xFirst     = x;
    e.instFirst  = instID;
    e.nVirtual   = n;
    e.Kd         = kd;
    e.Pr         = pr;
    e.Pm         = pm;
    e.throughput = float3(1.0f, 1.0f, 1.0f);
    e.instID     = instID;
    e.flags      = DLSS_PSR_FLAG_VALID | DLSS_PSR_FLAG_DELTA;
    e.bounces    = 0u;
    e.paneF      = 0.0f;
    e.M          = PsrIdentity();
    return e;
}

inline PsrChainEnd PsrChainSky(float3 primaryHit, float3 primaryDir, float pathLength)
{
    PsrChainEnd e = PsrChainSurface(primaryHit + primaryDir * (pathLength + cameraFar), -primaryDir,
                                    float3(1.0f, 1.0f, 1.0f), 1.0f, 0.0f, 0xFFFFFFFFu);
    e.flags |= DLSS_PSR_FLAG_SKY;
    return e;
}

// Without `evaluate`, only the unshaded first hit (plain probe).
inline PsrChainEnd PsrWalkDeltaChain(float3 origin, float3 dir, uint mediumMatID, float3x3 M,
                                     float3 primaryHit, float3 primaryDir, bool evaluate, uint maxBounces,
                                     bool stopAtThinGlass)
{
    float  pathLength = 0.0f;
    float3 throughput = float3(1.0f, 1.0f, 1.0f);
    bool   allDelta   = true;
    PsrChainEnd e = PsrChainSky(primaryHit, primaryDir, 0.0f);
    float3 xFirst    = e.xFirst;
    uint   instFirst = e.instFirst;

    [loop]
    for (uint bounce = 0u; bounce < maxBounces; ++bounce)
    {
        RayDesc r;
        r.Origin    = origin;
        r.Direction = dir;
        r.TMin      = 0.00001f;
        r.TMax      = RAY_TMAX_PLANET;
        if (!IsRayDescValid(r)) break;
        RayQuery<RAY_FLAG_NONE, RAYQUERY_FLAG_ALLOW_OPACITY_MICROMAPS> q;
        q.TraceRayInline(SceneBVH, RAY_FLAG_NONE, 0xFF, r);
        [loop]
        for (uint it = 0u; q.Proceed() && it < 128u; ++it)
        {
            if (q.CandidateType() != CANDIDATE_NON_OPAQUE_TRIANGLE) continue;
            const uint ci = q.CandidateInstanceID();
            const uint cp = FlatPrimID(ci, q.CandidateGeometryIndex(), q.CandidatePrimitiveIndex());
            const uint cm = GetMatIDFast(ci, cp);
            // Chain: path-tracer glass/cutout rule; probe: cheap alpha rule.
            const bool commit = evaluate
                ? (LoadIsThinGlass(cm) || LoadKd_w(cm) < 1.0f - EPSILON ||
                   AlphaCandidateOccludes(ci, cp, q.CandidateTriangleBarycentrics()))
                : LoadAlphaThreshold(cm) < 1.0f;
            if (commit) q.CommitNonOpaqueTriangleHit();
        }

        if (q.CommittedStatus() != COMMITTED_TRIANGLE_HIT)
        {
            e = PsrChainSky(primaryHit, primaryDir, pathLength);
            e.bounces = bounce + 1u;
            break;
        }

        const float  t      = q.CommittedRayT();
        const uint   inst   = q.CommittedInstanceID();
        const uint   prim   = FlatPrimID(inst, q.CommittedGeometryIndex(), q.CommittedPrimitiveIndex());
        uint matID = GetMatIDFast(inst, prim);
        const float3 hitPos = origin + dir * t;
        pathLength += t;
        if (mediumMatID != MEDIUM_INVALID)
            throughput *= CalculateAbsorptionThroughput(LoadTf(mediumMatID), t);

        e.xVirtual = primaryHit + primaryDir * pathLength;
        e.xHit     = hitPos;
        e.instID   = inst;
        e.bounces  = bounce + 1u;
        e.flags    = evaluate ? DLSS_PSR_FLAG_VALID : 0u;
        if (bounce == 0u) { xFirst = e.xVirtual; instFirst = inst; }
        if (!evaluate) break;

        const float   beam = PixelConeAngle() * (length(primaryHit - mul(viewI, float4(0, 0, 0, 1)).xyz) + pathLength);
        const HitInfo h    = EvalSurfaceStateDir(inst, prim, q.CommittedTriangleBarycentrics(), dir, beam);
        matID = ResolveSurfaceMaterial(matID, h);
        float3 kd; float pr, pm;
        RefetchMaterial(matID, h, kd, pr, pm);
        e.nVirtual = mul(M, h.hitNormal);
        e.Kd = kd; e.Pr = pr; e.Pm = pm;

        const float3 emission = (h.lightID != 0xFFFFFFFFu)
            ? g_EmissiveTriangles[h.lightID].emission * GLOBAL_EMISSION_STRENGTH : 0.0f;
        if (any(emission > 0.0f))
        {
            e.Kd = saturate(emission); e.Pr = 1.0f; e.Pm = 0.0f;
            e.flags |= DLSS_PSR_FLAG_EMITTER;
            break;
        }

        // A pane ends the chain; shading derives its reflection motion from paneF.
        const bool thin = LoadIsThinGlass(matID);
        if (thin && stopAtThinGlass)
        {
            const float F = FresnelDielectric(-dir, h.hitNormal, 1.0f, LoadNi(matID)).x;
            e.Kd    = (1.0f - F) * LoadTf(matID) + F;
            e.Pm    = 0.0f;
            e.paneF = F;
            e.flags |= DLSS_PSR_FLAG_GLASS;
            break;
        }

        // Metal reflects, glass transmits; a per-pixel choice would tear the guides.
        const bool clear = !thin && LoadKd_w(matID) < EPSILON;
        const bool metal = pm >= 0.5f;
        if (pr >= DLSS_PSR_ROUGH_END || LoadIsSSS(matID) || !(thin || clear || metal)) break;
        allDelta = allDelta && pr < SMOOTH_SPECULAR_THRESHOLD;

        const float3 V = -dir;
        const float3 N = h.hitNormal;
        bool   reflectLobe = true;
        float3 lobeT       = float3(1.0f, 1.0f, 1.0f);
        float3 nextDir     = reflect(dir, N);
        if (metal)
        {
            lobeT = FresnelConductor(kd, V, N);
        }
        else if (thin)
        {
            const float F = FresnelDielectric(V, N, 1.0f, LoadNi(matID)).x;
            reflectLobe = false;
            lobeT   = (1.0f - F) * LoadTf(matID);
            nextDir = dir;
        }
        else
        {
            const float ni   = LoadNi(matID);
            const float etai = h.backface ? ni : 1.0f;
            const float etat = h.backface ? 1.0f : ni;
            const float F    = FresnelDielectricTIR(V, N, etai, etat).x;
            float3 refracted;
            reflectLobe = !RefractVector(V, N, etai / etat, refracted);
            lobeT = (reflectLobe ? F : 1.0f - F).xxx;
            if (!reflectLobe)
            {
                nextDir     = refracted;
                mediumMatID = h.backface ? MEDIUM_INVALID : matID;
            }
        }
        throughput *= lobeT;
        if (reflectLobe) M = mul(M, PsrReflectionMatrix(N));
        origin = offset_ray(hitPos, reflectLobe ? N : -N);
        dir    = nextDir;
    }
    e.throughput = throughput;
    e.M = M;
    e.xFirst     = xFirst;
    e.instFirst  = instFirst;
    if (allDelta) e.flags |= DLSS_PSR_FLAG_DELTA;
    return e;
}

// Chain end as packed into the probe slice.
struct PsrProbe { float3 nVirtual; float3 Kd; float3 throughput; float Pr; float Pm; uint flags; uint bounces; float paneF; };

inline float4 PsrProbePack(PsrChainEnd e)
{
    const uint pr = (uint)round(saturate(e.Pr) * 255.0f);
    const uint pm = (uint)round(saturate(e.Pm) * 255.0f);
    const uint fp = (uint)round(saturate(e.paneF) * 255.0f);
    const uint packed = pr | (pm << 8u) | ((e.flags & 31u) << 16u) | (min(e.bounces, 7u) << 21u) | (fp << 24u);
    return float4(asfloat(PackNormal(e.nVirtual)), asfloat(PackRGB9E5(e.Kd)),
                  asfloat(PackRGB9E5(saturate(e.throughput))), asfloat(packed));
}

inline PsrProbe PsrProbeUnpack(float4 v)
{
    PsrProbe p;
    p.nVirtual   = UnpackNormal(asuint(v.x));
    p.Kd         = UnpackRGB9E5(asuint(v.y));
    p.throughput = UnpackRGB9E5(asuint(v.z));
    const uint packed = asuint(v.w);
    p.Pr      = (float)(packed & 0xFFu) * (1.0f / 255.0f);
    p.Pm      = (float)((packed >> 8u) & 0xFFu) * (1.0f / 255.0f);
    p.flags   = (packed >> 16u) & 31u;
    p.bounces = (packed >> 21u) & 7u;
    p.paneF   = (float)(packed >> 24u) * (1.0f / 255.0f);
    return p;
}

// Current-to-previous, in pixels.
inline float2 SkyMotionVector(uint2 px, float2 dims)
{
    const float2 d           = ((float2(px) + 0.5f) / dims) * 2.0f - 1.0f;
    const float4 target      = mul(projectionI, float4(d.x, -d.y, 1, 1));
    const float3 worldDir    = normalize(mul(viewI, float4(target.xyz, 0)).xyz);
    const float3 prevViewDir = mul(prevView, float4(worldDir, 0)).xyz;
    float2 mv = float2(0.0f, 0.0f);
    if (prevViewDir.z < 0.0f)
    {
        const float4 prevClip = mul(prevProjection, float4(prevViewDir * cameraFar, 1.0f));
        if (prevClip.w > 0.0f)
        {
            const float2 prevNdc = prevClip.xy / prevClip.w;
            const float2 prevUV  = float2(prevNdc.x * 0.5f + 0.5f, 0.5f - prevNdc.y * 0.5f);
            mv = prevUV * dims - 0.5f - float2(px);
        }
    }
    return mv;
}

inline float2 SurfaceMotionVector(uint2 px, float2 dims, float3 x, uint instID)
{
    if (instID == 0xFFFFFFFFu) return SkyMotionVector(px, dims);
    float2 prevPix = GetLastFramePixelCoordinates_Unclamped(x, prevView, prevProjection, dims, instID);
    if (IS_OCEAN_INSTANCE(instID)) {
        const float4 correspondence = gScratchPing[uint3(px, OCEAN_PREVIOUS_POSITION_SLOT)];
        if (OceanParams().historyValid <= 0.0f || asuint(correspondence.w) != instID) return 0.0f;
        const float3 previous = OceanPreviousHit(instID, asuint(correspondence.z), correspondence.xy, x);
        prevPix = GetLastFramePixelCoordinates_World(previous, prevView, prevProjection, dims);
    }
    const float2 curPix  = GetCurrentFramePixelCoordinates_Unclamped(x, view, projection, dims, instID);
    return (prevPix.x > -1e8f && curPix.x > -1e8f) ? (prevPix - curPix) : float2(0.0f, 0.0f);
}

// Mirrors along the chain are assumed static.
inline float2 PsrChainMotionVector(uint2 px, float2 dims, PsrChainEnd e)
{
    if (e.instID == 0xFFFFFFFFu) return SkyMotionVector(px, dims);
    if (IS_OCEAN_INSTANCE(e.instID)) return SurfaceMotionVector(px, dims, e.xHit, e.instID);
    const float3 localHit    = mul(instanceProps[e.instID].objectToWorldInverse, float4(e.xHit, 1.0f));
    const float3 prevHit     = mul(instanceProps[e.instID].prevObjectToWorld, float4(localHit, 1.0f));
    const float3 prevVirtual = e.xVirtual + mul(e.M, prevHit - e.xHit);
    const float2 prevPix = GetLastFramePixelCoordinates_World(prevVirtual, prevView, prevProjection, dims);
    const float2 curPix  = GetCurrentFramePixelCoordinates_Unclamped(e.xVirtual, view, projection, dims, e.instID);
    return (prevPix.x > -1e8f && curPix.x > -1e8f) ? (prevPix - curPix) : float2(0.0f, 0.0f);
}

// Covers both the hit's and the mirror's motion.
inline float2 PsrVirtualPrevPixel(float3 xVirtual, uint instHit, float3 xMirror, float3 nMirror,
                                  uint instMirror, float2 dims)
{
    const float3 xHit     = xVirtual - 2.0f * dot(xVirtual - xMirror, nMirror) * nMirror;
    const float3 localHit = mul(instanceProps[instHit].objectToWorldInverse, float4(xHit, 1.0f));
    const float3 prevHit  = mul(instanceProps[instHit].prevObjectToWorld, float4(localHit, 1.0f));

    const float3 localM = mul(instanceProps[instMirror].objectToWorldInverse, float4(xMirror, 1.0f));
    const float3 prevM  = mul(instanceProps[instMirror].prevObjectToWorld, float4(localM, 1.0f));

    // Via transformed tangents: survives non-uniform scale.
    const float3   localN = WorldToObjectNrm(instMirror, nMirror);
    const float3   localT = normalize(cross(localN, abs(localN.y) < 0.9f ? float3(0, 1, 0) : float3(1, 0, 0)));
    const float3   localB = cross(localN, localT);
    const float3x3 prevR  = (float3x3)instanceProps[instMirror].prevObjectToWorld;
    const float3   prevX  = cross(mul(prevR, localT), mul(prevR, localB));
    const float3   prevN  = dot(prevX, prevX) > 1e-10f ? normalize(prevX) : nMirror;

    const float3 prevVirtual = prevHit - 2.0f * dot(prevHit - prevM, prevN) * prevN;
    return GetLastFramePixelCoordinates_World(prevVirtual, prevView, prevProjection, dims);
}

inline float2 PsrVirtualMotionVector(float3 xVirtual, uint instHit, float3 xMirror, float3 nMirror,
                                     uint instMirror, float2 dims)
{
    const float2 prevPix = PsrVirtualPrevPixel(xVirtual, instHit, xMirror, nMirror, instMirror, dims);
    const float2 curPix  = GetCurrentFramePixelCoordinates_Unclamped(xVirtual, view, projection, dims, instHit);
    return (prevPix.x > -1e8f && curPix.x > -1e8f) ? (prevPix - curPix) : float2(0.0f, 0.0f);
}

