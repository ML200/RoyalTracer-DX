#pragma once
// Plain copy; never carry a HitObject through loops or reorder points.
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

// Next hit, skipping water facets met from the wrong side for the medium.
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

// One scatter event between object bounces; no lighting queries.
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
// Sun at pos relative to flat-surface transmission along t; phase not included.
float3 OceanSunReach(float3 pos, float3 sunDir, float3 t, float eta, float causticShare, OceanParamsGPU P,
                     out bool covered)
{
    covered = false;
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
    // Must reach the sea surface; anything else shadows.
    if (q.CommittedStatus() != COMMITTED_TRIANGLE_HIT) return 0.0f;
    if (!IS_OCEAN_INSTANCE(q.CommittedInstanceID())) {
        covered = true;
        return 0.0f;
    }
    const float d = q.CommittedRayT();
    const uint hi = q.CommittedInstanceID();
    const uint hp = FlatPrimID(hi, q.CommittedGeometryIndex(), q.CommittedPrimitiveIndex());
    // Foam at a 0.25 m footprint, from the step's Kd_w (WriteMaterialSlot).
    float3 n = CandidateGeoNormalW(hi, hp);
    float foam = 0.0f;
    if (causticShare > 0.0f) {
        const HitInfo h = EvalSurfaceState(hi, hp, q.CommittedTriangleBarycentrics(), pos, 0.25f);
        n = h.hitNormal;
        if (h.isOcean && h.oceanMaterialOffset < OCEAN_MATERIAL_LEVELS) {
            const float water = LoadKd_w(P.materialBase);
            foam = saturate((LoadKd_w(P.materialBase + h.oceanMaterialOffset) - water) / max(1.0f - water, 1e-3f));
        }
    }
    n = n.y >= 0.0f ? n : -n;
    if (dot(toSurface, n) <= 0.0f) return 0.0f;
    const float3 surfacePos = pos + toSurface * d;
    const float3 above = VisibilityTransmittance(surfacePos, n, surfacePos + sunDir * RAY_TMAX_PLANET, -sunDir, true, false);
    if (!any(above > 0.0f)) return 0.0f;
    // Fresnel on the air side; avoids false TIR at tilted facets.
    const float3 up = float3(0.0f, 1.0f, 0.0f);
    if (dot(sunDir, n) <= 0.0f) return 0.0f;
    const float transmit = 1.0f - FresnelDielectricTIR(sunDir, n, 1.0f, eta).x;
    const float flat = 1.0f - FresnelDielectricTIR(sunDir, up, 1.0f, eta).x;

    float caustic = 1.0f;
    const float3 tn = refract(-sunDir, n, 1.0f / eta);
    if (causticShare > 0.0f && dot(tn, tn) > 0.0f) {
        // Refracted spread from total slope variance (widest table entry).
        const float spread2 = 0.0625f * max(P.residualSlope[OCEAN_ROUGHNESS_ENTRIES / 4 - 1].w, 1e-4f);
        const float blur = 0.01f + 0.004f * d;
        const float lobe2 = 0.25f * spread2 + blur * blur;
        caustic = lerp(1.0f, exp(-dot(tn - t, tn - t) / lobe2) * (lobe2 + spread2) / lobe2, causticShare);
    }
    float3 sigmaA, sigmaS; float g;
    OceanMediumCoefficients(sigmaA, sigmaS, g);
    return above * (transmit / max(flat, 1e-3f)) * (1.0f - foam) * caustic * OceanMediumTransmittance(sigmaA + sigmaS, d);
}

// Sun light sample for submerged surfaces, refracted through a flat sea.
struct OceanWaterSun {
    float3 L;
    float3 radiance;
    float  pdf;
    float3 sunDir;
    float3 t;
};
bool OceanSampleWaterSun(float3 pos, float2 u, out OceanWaterSun s)
{
    s = (OceanWaterSun)0;
    const OceanParamsGPU P = OceanParams();
    float3 receiver = pos + sceneOriginWorld;
    receiver.y = max(receiver.y, P.surfaceY + sceneOriginWorld.y);
    const SunSampleResult sun = SampleSun(u, receiver);
    if (!(sun.pdf > 1e-20f) || sun.direction.y <= 1e-3f) return false;
    const float eta = max(LoadNi(P.materialBase), 1.0f);
    const float3 up = float3(0.0f, 1.0f, 0.0f);
    s.t = refract(-sun.direction, up, 1.0f / eta);
    s.L = -s.t;
    s.sunDir = sun.direction;
    s.radiance = sun.radiance * sun.direction.y / max(-s.t.y, 1e-3f) *
                 (1.0f - FresnelDielectricTIR(sun.direction, up, 1.0f, eta).x);
    s.pdf = sun.pdf;
    return true;
}
float3 OceanWaterSunReach(float3 pos, float3 n, OceanWaterSun s)
{
    const OceanParamsGPU P = OceanParams();
    bool covered;
    return OceanSunReach(offset_ray(pos, dot(s.L, n) >= 0.0f ? n : -n), s.sunDir, s.t,
                         max(LoadNi(P.materialBase), 1.0f), 0.0f, P, covered);
}

// Henyey-Greenstein mass in the forward hemisphere.
float OceanForwardShare(float g)
{
    if (abs(g) < 1e-3f) return 0.5f;
    return (1.0f + g) / (2.0f * g) - (1.0f - g * g) / (2.0f * g * sqrt(1.0f + g * g));
}
// (1 - e^(-k s)) / k, series for small k s; k may be negative.
float3 OceanExpIntegral(float3 k, float s)
{
    const float3 x = k * s;
    const float3 series = s * (1.0f - 0.5f * x + x * x / 6.0f);
    return select(abs(x) < 1e-3f, series, (1.0f - exp(-x)) / select(abs(k) > 1e-30f, k, (float3)1e-30f));
}
// Share of uniform light in Snell's cone scattered into wo.
float OceanConeShare(float3 wo, float g, float cosCone)
{
    float3 tangent, bitangent;
    GetOrthoBasis(wo, tangent, bitangent);
    float inside = 0.0f;
    [unroll] for (uint i = 0u; i < 24u; ++i) {
        const float mu = OceanSamplePhaseCosine(g, (float(i) + 0.5f) / 24.0f);
        const float sn = sqrt(max(1.0f - mu * mu, 0.0f));
        const float phi = 2.39996323f * float(i);
        const float3 wi = mu * wo + sn * (cos(phi) * tangent + sin(phi) * bitangent);
        inside += smoothstep(cosCone - 0.08f, cosCone + 0.08f, -wi.y);
    }
    return max(inside / 24.0f, 0.5f * (1.0f - OceanForwardShare(g)));
}
// Depth below mean sea level, Earth's curve included; 0 in crests.
float OceanMeanDepth(float3 p, OceanParamsGPU P)
{
    const float2 xz = p.xz + P.curveOrigin;
    return max(P.surfaceY - 0.5f * dot(xz, xz) * P.invRadius - p.y, 0.0f);
}

// Segment in-scatter: analytic diffuse field, plus sun and lights if sampled.
static const float OCEAN_SKY_TRANSMIT = 0.934f;   // 1 - 0.066 diffuse reflectance
static const float OCEAN_DIFFUSE_COSINE = 0.8f;   // mean cosine just under the surface
static const float OCEAN_COVERED_AMBIENT = 0.3f;  // ambient share under a hull
float3 OceanSegmentInScatter(float3 origin, float3 dir, float s, bool sampled, inout uint seed)
{
    if (!OceanMediumEnabled() || !(s > 0.0f)) return 0.0f;
    const OceanParamsGPU P = OceanParams();
    float3 sigmaA, sigmaS; float g;
    OceanMediumCoefficients(sigmaA, sigmaS, g);
    if (!any(sigmaS > 0.0f)) return 0.0f;
    const float3 sigmaT = sigmaA + sigmaS;
    const float3 up = float3(0.0f, 1.0f, 0.0f);
    const float3 lum = float3(0.2126f, 0.7152f, 0.0722f);
    const float eta = max(LoadNi(P.materialBase), 1.0f);
    const float y0 = OceanMeanDepth(origin, P);
    const float w = -dir.y;

    // Horizontal irradiance just under a flat surface.
    const SunState S = ComputeSunState();
    float3 sunE = 0.0f;
    float mu = OCEAN_DIFFUSE_COSINE;
    if (S.pdf > 0.0f && S.dirWS.y > 1e-3f) {
        const float3 t = refract(-S.dirWS, up, 1.0f / eta);
        mu = max(-t.y, 1e-3f);
        sunE = S.radiance / S.pdf * S.dirWS.y * (1.0f - FresnelDielectricTIR(t, up, eta, 1.0f).x);
    }
    const float3 skyE = EvaluateSkyIrradiance() * OCEAN_SKY_TRANSMIT;
    const float sunShare = dot(sunE, lum) / max(dot(sunE + skyE, lum), 1e-20f);
    // Kd = 1.04 (a + bb) / mu0 (Gordon 1989)
    const float3 Kd = 1.04f * (sigmaA + (1.0f - OceanForwardShare(g)) * sigmaS) /
                      lerp(OCEAN_DIFFUSE_COSINE, mu, sunShare);
    // Uniform radiance over Snell's cone gives E = pi L / eta^2.
    const float share = OceanConeShare(-dir, g, sqrt(max(1.0f - 1.0f / (eta * eta), 0.0f)));
    const float3 field = sigmaS * (share * eta * eta / PI) *
        ((sunE + skyE) * exp(-Kd * y0) * OceanExpIntegral(sigmaT + Kd * w, s) -
         sunE * exp(-sigmaT * y0 / mu) * OceanExpIntegral(sigmaT * (1.0f + w / mu), s));
    const float3 ambient = max(field, 0.0f);
    if (!sampled) return ambient;

    // One point, pdf ~ unshadowed single scattering.
    const float k = dot(sigmaT * (1.0f + w / mu), lum);
    const float u = RandomFloatSingle(seed);
    float t = u * s, pdf = 1.0f / s;
    if (abs(k * s) >= 1e-3f) {
        const float norm = 1.0f - exp(-k * s);
        t = clamp(-log(1.0f - u * norm) / k, 0.0f, s);
        pdf = k * exp(-k * t) / norm;
    }
    const float3 x = origin + dir * t;
    const float3 view = OceanMediumTransmittance(sigmaT, t) * sigmaS / max(pdf, 1e-20f);
    float3 result = ambient;

    bool covered = false;
    float3 receiver = x + sceneOriginWorld;
    receiver.y = max(receiver.y, P.surfaceY + sceneOriginWorld.y);
    const SunSampleResult sun = SampleSun(float2(RandomFloatSingle(seed), RandomFloatSingle(seed)), receiver);
    if (sun.pdf > 0.0f && sun.direction.y > 1e-3f) {
        const float3 tSun = refract(-sun.direction, up, 1.0f / eta);
        const float3 reach = OceanSunReach(x, sun.direction, tSun, eta, 1.0f, P, covered);
        // Irradiance normal to the refracted beam.
        const float3 beam = sun.radiance / sun.pdf * sun.direction.y / max(-tSun.y, 1e-3f) *
                            (1.0f - FresnelDielectricTIR(tSun, up, eta, 1.0f).x);
        result += view * beam * reach * OceanPhase(dot(dir, -tSun), g);
    }
    if (covered) result -= ambient * (1.0f - OCEAN_COVERED_AMBIENT);

    if ((rs_flags & RS_FLAG_NO_MESH_LIGHTS) == 0u) {
        const LT_Sample pick = LT_SampleLight(x, 0.0f, seed, false);
        const LT_LightSampleResult light = LT_SamplePointOnLightTree(x, pick, seed);
        if (light.pdfSolidAngle > 1e-20f) {
            const float3 L = normalize(light.position - x);
            const float3 vis = VisibilityTransmittance(x, 0.0f, light.position, light.normal, false, true);
            if (any(vis > 0.0f))
                result += view * light.emission * vis * OceanPhase(dot(dir, L), g) / light.pdfSolidAngle;
        }
    }
    return result;
}

float3 OceanSampleScatterDirection(float3 incoming, inout uint seed) {
    const float g = LoadPhaseG(OceanParams().materialBase);
    const float mu = OceanSamplePhaseCosine(g,RandomFloatSingle(seed));
    const float phi = 6.28318530718f*RandomFloatSingle(seed);
    const float r = sqrt(max(1.0f-mu*mu,0.0f));
    float3 tangent,bitangent;
    GetOrthoBasis(incoming,tangent,bitangent);
    // Phase / pdf = 1; g > 0 scatters forward.
    return normalize(mu*incoming+r*(cos(phi)*tangent+sin(phi)*bitangent));
}
