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
// What of the sun's beam reaches `pos` in the water, as a share of the beam a flat surface would
// let through along the refracted direction `t` (the phase left out).
//
// The connection through a refracting, moving surface cannot be solved exactly, so it is aimed as
// if the surface were flat - sunlight enters along t and arrives here from -t - and one ray up that
// direction finds the actual surface: whatever blocks it (a hull) shadows the point, its foam
// blocks it too, and the transmission through the real wave is its Fresnel; a second ray on from
// the surface to the sun finds whatever stands above the water. The wave's normal also decides
// whether the sunlight it passes actually heads this way: a lobe around the flat direction,
// normalised by the surface's slope spread so it averages to one, focuses the light into the
// shafts and caustic ripples sunlit water shows, and widens with depth, where those blur out.
// `causticShare` is how much of that lobe is applied: all of it for the water's own scattering,
// none for a surface, which is lit by the mean the lobe averages to. `covered` reports that
// something in the water stands over the point.
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
    // Only the sea's own surface lets the sun through; anything else in between shadows the point.
    if (q.CommittedStatus() != COMMITTED_TRIANGLE_HIT) return 0.0f;
    if (!IS_OCEAN_INSTANCE(q.CommittedInstanceID())) {
        covered = true;
        return 0.0f;
    }
    const float d = q.CommittedRayT();
    const uint hi = q.CommittedInstanceID();
    const uint hp = FlatPrimID(hi, q.CommittedGeometryIndex(), q.CommittedPrimitiveIndex());
    // The foam is read at a quarter-metre footprint: what shades the water below is how much of the
    // surface is white there, not where each strand of the lace runs - and drawing the lace at every
    // scattering point cost more than the rest of the connection. What the foam and the bubble cloud
    // under it catch of the light the surface lets through is the step's diffuse weight over the
    // water's own (WriteMaterialSlot). A surface without the lobe takes the facet the ray crossed and
    // no foam: the wave's full normal and its foam made no visible difference there, and evaluating
    // them was half of what the connection added over a straight shadow ray (looking down at a
    // submerged deck: pt_trace 3.83 -> 3.58 ms against 3.3 ms for the straight ray).
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
    // The sunlight the wave lets in, taken on the air side where it arrives: the flat direction
    // leaves the water past the critical angle at a facet tilted away from the sun, which light
    // coming down through that facet never does.
    const float3 up = float3(0.0f, 1.0f, 0.0f);
    if (dot(sunDir, n) <= 0.0f) return 0.0f;
    const float transmit = 1.0f - FresnelDielectricTIR(sunDir, n, 1.0f, eta).x;
    const float flat = 1.0f - FresnelDielectricTIR(sunDir, up, 1.0f, eta).x;

    float caustic = 1.0f;
    const float3 tn = refract(-sunDir, n, 1.0f / eta);
    if (causticShare > 0.0f && dot(tn, tn) > 0.0f) {
        // Spread of the refracted direction over the sea, from its total slope variance (the
        // widest entry of the footprint table), and the lobe that picks out the focused part.
        const float spread2 = 0.0625f * max(P.residualSlope[OCEAN_ROUGHNESS_ENTRIES / 4 - 1].w, 1e-4f);
        const float blur = 0.01f + 0.004f * d;
        const float lobe2 = 0.25f * spread2 + blur * blur;
        caustic = lerp(1.0f, exp(-dot(tn - t, tn - t) / lobe2) * (lobe2 + spread2) / lobe2, causticShare);
    }
    float3 sigmaA, sigmaS; float g;
    OceanMediumCoefficients(sigmaA, sigmaS, g);
    return above * (transmit / max(flat, 1e-3f)) * (1.0f - foam) * caustic * OceanMediumTransmittance(sigmaA + sigmaS, d);
}

// The sun as the light sample of a surface in the water. Sunlight reaches it refracted, so it is
// aimed as if through a flat surface: arriving from L = -t, t the beam's direction under the
// water, as the irradiance a flat surface passes spread over the refracted beam's cross-section
// (radiance / pdf is that irradiance, as for the scattering point in OceanSegmentInScatter), and
// what actually reaches the surface is OceanSunReach's, without the caustic lobe. A straight
// connection to the sun left the water at the sun's own angle, past the critical angle whenever
// the sun stood below about 41 degrees: a flat surface turned it back entirely, and only the
// facets tilted towards the sun let any through, so a submerged hull was lit in dark blotches that
// moved with every wave, which the reconstruction could only smear. The receiver is lifted to the
// surface for the sun's own radiance: the beam is the one that reaches the sea.
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

// Share of a Henyey-Greenstein phase function's mass in the hemisphere ahead: 1/2 for isotropic
// scattering, 0.977 for the ocean's g = 0.9. One minus it is the backscattered share.
float OceanForwardShare(float g)
{
    if (abs(g) < 1e-3f) return 0.5f;
    return (1.0f + g) / (2.0f * g) - (1.0f - g * g) / (2.0f * g * sqrt(1.0f + g * g));
}
// (1 - e^(-k s)) / k, the integral of e^(-k t) over [0, s], kept accurate where k s is small (k may
// be negative).
float3 OceanExpIntegral(float3 k, float s)
{
    const float3 x = k * s;
    const float3 series = s * (1.0f - 0.5f * x + x * x / 6.0f);
    return select(abs(x) < 1e-3f, series, (1.0f - exp(-x)) / select(abs(k) > 1e-30f, k, (float3)1e-30f));
}
// Share of the light arriving inside Snell's cone - every direction a flat surface refracts the sky
// into, down to a half-angle of asin(1/eta) from straight down - that the water scatters on into
// the outgoing direction `wo`, for radiance uniform over the cone: the phase function's mass inside
// it, found by sending a fixed set of phase-sampled directions from wo and counting those that land
// inside, softly. Nearly all of it looking straight up into the cone (0.92 for g = 0.9), a few per
// cent looking sideways, a trace looking down. The ocean's phase function is so peaked that light
// spread evenly over the upper hemisphere instead lit a sideways view seven times too bright.
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
// Depth below the sea's flat mean level at scene-relative p, the Earth's curve included; a point up
// in a crest counts as at the surface.
float OceanMeanDepth(float3 p, OceanParamsGPU P)
{
    const float2 xz = p.xz + P.curveOrigin;
    return max(P.surfaceY - 0.5f * dot(xz, xz) * P.invRadius - p.y, 0.0f);
}

// Light the water scatters towards the viewer along a segment of it, as radiance arriving at
// `origin`: the segment runs from origin along unit `dir` for s metres. The path itself goes on
// past it to whatever the segment ends at, weighted by the segment's transmittance; the light
// scattered along the way is added here and goes nowhere further. Choosing per path between the
// two, as a free flight does, sent only part of the paths on to a hull or seabed behind the water
// and turned the rest aside, which the reconstruction could only blur into mush.
//
// Two parts:
//
// The diffuse light field of the water - skylight let through the surface and sunlight scattered
// more than once - is modelled, not traced: its downwelling irradiance falls off with depth as
// e^(-Kd y), with Gordon's Kd = 1.04 (a + bb) / mu0 (absorption, backscattering, the refracted
// sun's cosine), and the sun's direct beam, which is sampled below, is taken out of it. Treated as
// arriving evenly over Snell's cone, what it scatters towards the viewer is the phase function's
// mass inside that cone (OceanConeShare), high looking up into it and nearly nothing looking
// sideways or down; integrated along the segment in closed form, it costs no ray and carries no noise.
//
// Where `sampled`, the first water segment of a camera path, the sun's direct beam is added at one
// point, drawn along the segment in proportion to e^(-k t) with k = sigmaT (1 + w / mu): the view's
// own attenuation and the beam's, which grows with depth as the segment runs down at w. That is
// the shape of the unshadowed single scattering, so what is left to chance is only what the point
// actually receives (OceanSunReach): shadows, foam, caustics. The same point takes one light-tree
// sample for the lights in or above the water. A point with something standing over it in the
// water takes a share of the modelled field too.
static const float OCEAN_SKY_TRANSMIT = 0.934f;   // diffuse skylight through the surface: 1 - 0.066
static const float OCEAN_DIFFUSE_COSINE = 0.8f;   // mean cosine of the diffuse field just under the surface
static const float OCEAN_COVERED_AMBIENT = 0.3f;  // share of it under a hull
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

    // The sun's beam and the sky just under a flat surface, as irradiance on a horizontal plane.
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
    const float3 Kd = 1.04f * (sigmaA + (1.0f - OceanForwardShare(g)) * sigmaS) /
                      lerp(OCEAN_DIFFUSE_COSINE, mu, sunShare);
    // Radiance uniform over Snell's cone lays irradiance pi sin^2 of its half-angle on a horizontal
    // plane, with sin = 1/eta.
    const float share = OceanConeShare(-dir, g, sqrt(max(1.0f - 1.0f / (eta * eta), 0.0f)));
    const float3 field = sigmaS * (share * eta * eta / PI) *
        ((sunE + skyE) * exp(-Kd * y0) * OceanExpIntegral(sigmaT + Kd * w, s) -
         sunE * exp(-sigmaT * y0 / mu) * OceanExpIntegral(sigmaT * (1.0f + w / mu), s));
    const float3 ambient = max(field, 0.0f);
    if (!sampled) return ambient;

    // One point along the segment, in proportion to the unshadowed single scattering.
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
        // Irradiance of the beam in the water, normal to it: the horizontal irradiance a flat
        // surface passes, spread over the refracted beam's cross-section.
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
    // Phase value / phase PDF is one. Positive g continues along the incoming ray.
    return normalize(mu*incoming+r*(cos(phi)*tangent+sin(phi)*bitangent));
}
