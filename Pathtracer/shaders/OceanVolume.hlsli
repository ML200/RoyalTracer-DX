#pragma once
// Snapshot the trace result into ordinary values immediately. In particular, do
// not carry/reassign the driver's opaque HitObject through a volume-retrace loop
// or across a later reorder point. Traversal and alpha-test behavior are unchanged.
struct OceanPathHit {
    bool hit;
    float distance;
    uint instance;
    uint geometry;
    uint primitive;
    float2 barycentrics;
};
OceanPathHit OceanTracePathHit(RayDesc ray) {
    OceanPathHit result = (OceanPathHit)0;
    result.distance = ray.TMax;
    dx::HitObject hit = TraceRayChecked(SceneBVH,RAY_FLAG_FORCE_OMM_2_STATE,0xFF,ray);
    if (hit.IsHit()) {
        BuiltInTriangleIntersectionAttributes attr;
        hit.GetAttributes(attr);
        result.hit = true;
        result.distance = hit.GetRayTCurrent();
        result.instance = hit.GetInstanceID();
        result.geometry = hit.GetGeometryIndex();
        result.primitive = hit.GetPrimitiveIndex();
        result.barycentrics = attr.barycentrics;
    }
    return result;
}

// The path's next hit, keeping the water surface consistent with the medium the path is in. A
// ray in air cannot meet the water from below, nor one in the water meet it from above - but
// under a full-resolution wave normal on a coarser mesh they do: a ripple steeper than the
// triangles it sits on bends its reflection past the next facet, which the ray then crosses
// from the wrong side. That was read as the ray leaving the water and the reflected sky came
// out as black blobs on the surface. The ray is let through such a facet instead, as if it were
// part of the ripple it reflected off.
OceanPathHit OceanTracePathHitInMedium(RayDesc ray, bool inWater) {
    OceanPathHit h = OceanTracePathHit(ray);
    [loop] for (uint i = 0u; i < 4u && h.hit && IS_OCEAN_INSTANCE(h.instance); ++i) {
        const float3 n = CandidateGeoNormalW(h.instance, FlatPrimID(h.instance, h.geometry, h.primitive));
        if ((dot(ray.Direction, n) > 0.0f) == inWater || h.distance + 1e-3f >= ray.TMax)
            break;
        ray.TMin = h.distance + 1e-3f;
        h = OceanTracePathHit(ray);
    }
    return h;
}

// One actual path-scattering event between object bounces. No lighting/shadow
// queries here: the scattered ray continues to real geometry and the environment.
OceanFlight OceanSampleWaterSegment(float3 origin, float3 dir, float limit, bool used, inout uint seed) {
    OceanFlight f;
    f.distance = limit; f.weight = 1.0f; f.scattered = false;
    if (!OceanMediumEnabled()) return f;
    limit = OceanBoundaryDistance(origin,dir,limit);
    float3 sigmaA,sigmaS; float g;
    OceanMediumCoefficients(sigmaA,sigmaS,g);
    const float2 u = !used && any(sigmaS > 0.0f) ?
        float2(RandomFloatSingle(seed),RandomFloatSingle(seed)) : 0.0f;
    return OceanFreeFlight(sigmaA,sigmaS,limit,used,u);
}
// Direct sunlight at a scattering point in the water: the light the point sends back along the
// ray it was reached on, per unit of the throughput the free flight left (which already carries
// the scattering coefficient).
//
// Without it the sun reached the water only when a phase-sampled ray happened to leave through the
// surface straight into the half-degree disk, which is why the lit water was a sparse field of
// fireflies no denoiser could hold. The connection through a refracting, moving surface cannot be
// solved exactly, so it is aimed as if the surface were flat - sunlight enters along the refracted
// direction t and arrives here from -t - and one ray up that direction finds the actual surface:
// whatever blocks it (a hull) shadows the point, its foam blocks it too, and the transmission
// through the real wave is its Fresnel. The wave's normal also decides whether the sunlight it
// passes actually heads this way: a lobe around the flat direction, normalised by the surface's
// slope spread so it averages to one, focuses the light into the shafts and caustic ripples
// sunlit water shows, and widens with depth, where those blur out.
float3 OceanVolumeSunNee(float3 pos, float3 rayDir, inout uint seed) {
    if (!OceanMediumEnabled()) return 0.0f;
    const OceanParamsGPU P = OceanParams();
    // The sun is judged from the surface above: the point itself can lie below the atmosphere's
    // ground plane, which the sun sampler would black out.
    float3 receiver = pos + sceneOriginWorld;
    receiver.y = max(receiver.y, P.surfaceY + sceneOriginWorld.y);
    const SunSampleResult sun = SampleSun(float2(RandomFloatSingle(seed), RandomFloatSingle(seed)), receiver);
    if (sun.pdf <= 0.0f || sun.direction.y <= 1e-3f) return 0.0f;

    const float eta = max(LoadNi(P.materialBase), 1.0f);
    const float3 t = refract(-sun.direction, float3(0.0f, 1.0f, 0.0f), 1.0f / eta);
    const float3 toSurface = -t;

    RayDesc r;
    r.Origin = pos;
    r.Direction = toSurface;
    r.TMin = 1e-4f;
    r.TMax = RAY_TMAX_PLANET;
    RayQuery<RAY_FLAG_NONE, RAYQUERY_FLAG_ALLOW_OPACITY_MICROMAPS> q;
    q.TraceRayInline(SceneBVH, RAY_FLAG_NONE, 0xFF, r);
    [loop]
    for (uint it = 0u; q.Proceed() && it < 32u; ++it) {
        if (q.CandidateType() != CANDIDATE_NON_OPAQUE_TRIANGLE) continue;
        const uint ci = q.CandidateInstanceID();
        const uint cp = FlatPrimID(ci, q.CandidateGeometryIndex(), q.CandidatePrimitiveIndex());
        const uint cm = GetMatIDFast(ci, cp);
        if (LoadIsOceanMaterial(cm) || LoadIsThinGlass(cm) || LoadKd_w(cm) < 1.0f - EPSILON ||
            AlphaCandidateOccludes(ci, cp, q.CandidateTriangleBarycentrics()))
            q.CommitNonOpaqueTriangleHit();
    }
    // Only the sea's own surface lets the sun through; anything else in between shadows the point.
    if (q.CommittedStatus() != COMMITTED_TRIANGLE_HIT || !IS_OCEAN_INSTANCE(q.CommittedInstanceID())) return 0.0f;
    const float d = q.CommittedRayT();
    const uint hi = q.CommittedInstanceID();
    const uint hp = FlatPrimID(hi, q.CommittedGeometryIndex(), q.CommittedPrimitiveIndex());
    // The foam is read at a quarter-metre footprint: what shades the water below is how much of the
    // surface is white there, not where each strand of the lace runs - and drawing the lace at every
    // scattering point cost more than the rest of the connection.
    const HitInfo h = EvalSurfaceState(hi, hp, q.CommittedTriangleBarycentrics(), pos, 0.25f);
    const float3 n = h.hitNormal.y >= 0.0f ? h.hitNormal : -h.hitNormal;
    if (dot(toSurface, n) <= 0.0f) return 0.0f;
    // What the foam and the bubble cloud under it catch of the light the surface lets through: the
    // step's diffuse weight over the water's own (WriteMaterialSlot).
    float foam = 0.0f;
    if (h.isOcean && h.oceanMaterialOffset < OCEAN_MATERIAL_LEVELS) {
        const float water = LoadKd_w(P.materialBase);
        foam = saturate((LoadKd_w(P.materialBase + h.oceanMaterialOffset) - water) / max(1.0f - water, 1e-3f));
    }
    const float transmit = 1.0f - FresnelDielectricTIR(-toSurface, n, eta, 1.0f).x;

    float caustic = 1.0f;
    const float3 tn = refract(-sun.direction, n, 1.0f / eta);
    if (dot(tn, tn) > 0.0f) {
        // Spread of the refracted direction over the sea, from its total slope variance (the
        // widest entry of the footprint table), and the lobe that picks out the focused part.
        const float spread2 = 0.0625f * max(P.residualSlope[OCEAN_ROUGHNESS_ENTRIES / 4 - 1].w, 1e-4f);
        const float blur = 0.01f + 0.004f * d;
        const float lobe2 = 0.25f * spread2 + blur * blur;
        caustic = exp(-dot(tn - t, tn - t) / lobe2) * (lobe2 + spread2) / lobe2;
    }

    float3 sigmaA, sigmaS; float g;
    OceanMediumCoefficients(sigmaA, sigmaS, g);
    // Irradiance of the beam in the water, normal to it: the horizontal irradiance the surface
    // passes, spread over the refracted beam's cross-section.
    const float3 beam = sun.radiance / sun.pdf * sun.direction.y / max(t.y * -1.0f, 1e-3f);
    return beam * transmit * (1.0f - foam) * caustic * OceanMediumTransmittance(sigmaA + sigmaS, d) *
           OceanPhase(dot(rayDir, toSurface), g);
}

float3 OceanSampleScatterDirection(float3 incoming, inout uint seed) {
    const float g = LoadPhaseG(OceanParams().materialBase);
    const float mu = OceanSamplePhaseCosine(g,RandomFloatSingle(seed));
    const float phi = 6.28318530718f*RandomFloatSingle(seed);
    const float r = sqrt(max(1.0f-mu*mu,0.0f));
    float3 tangent,bitangent;
    GetOrthoBasis(incoming,tangent,bitangent);
    // Phase value / phase PDF is one. Positive g continues along the incoming ray.
    return normalize(mu*incoming+r*(cos(phi)*tangent+sin(phi)*bitangent));
}
