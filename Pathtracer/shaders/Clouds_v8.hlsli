#ifndef CLOUDS_V8_HLSLI
#define CLOUDS_V8_HLSLI

#ifndef ENABLE_CLOUDS
#define ENABLE_CLOUDS 1
#endif

#ifndef CLOUD_STBN_AVAILABLE
#define CLOUD_STBN_AVAILABLE 1
#endif

#if CLOUD_STBN_AVAILABLE
Texture2DArray<float4> g_cloudSTBN : register(t41);
#endif

// Hard caps that size unrolled constant arrays; the rest of the
// CLOUD_* knobs are cbuffer-bound via Includes_v8.hlsli.
#define CLOUD_SHADOW_CONE_SAMPLES_MAX 5
#define CLOUD_AMBIENT_STEPS_MAX       6

inline float CloudLodT(float distKm)
{
    return saturate((distKm - CLOUD_LOD_NEAR_KM)
                  / max(1e-4f, CLOUD_LOD_FAR_KM - CLOUD_LOD_NEAR_KM));
}

inline float4 CloudRand4(uint2 px, uint frame, uint tap)
{
#if CLOUD_STBN_AVAILABLE
    uint w, h, slices, mips;
    g_cloudSTBN.GetDimensions(0, w, h, slices, mips);
    uint sliceIdx = (frame + tap * 17u) % max(slices, 1u);
    uint2 puv = uint2((px.x + tap * 113u) & (w - 1u),
                      (px.y + tap *  71u) & (h - 1u));
    return g_cloudSTBN.Load(int4(int(puv.x), int(puv.y), int(sliceIdx), 0));
#else
    uint s = initRandomData(px, uint2(0, 0), frame, 71u + tap * 13u);
    return float4(RandomFloatSingle(s),
                  RandomFloatSingle(s),
                  RandomFloatSingle(s),
                  RandomFloatSingle(s));
#endif
}

// g_cloudNoise: 256³ RGBA8 baked by Pass_cloudnoise_bake_v8.hlsl. Each
// channel uses its own bake period so WRAP sampling tiles cleanly.
#define CLOUD_NOISE_R_PERIOD   32.0f   // Perlin-Worley FBM
#define CLOUD_NOISE_G_PERIOD   16.0f   // Worley FBM (2 octaves)
#define CLOUD_NOISE_B_PERIOD   48.0f   // value noise (high-freq erosion)
#define CLOUD_NOISE_A_PERIOD   32.0f   // single-octave Worley (raw)

inline float CloudValueNoise(float3 p)
{
    return g_cloudNoise.SampleLevel(g_sampler,
        p * (1.0f / CLOUD_NOISE_B_PERIOD), 0).b;
}

inline float CloudWorley(float3 p)
{
    return g_cloudNoise.SampleLevel(g_sampler,
        p * (1.0f / CLOUD_NOISE_A_PERIOD), 0).a;
}

// quality kept on the signature for call-site symmetry but ignored —
// the bake stores a single 2-octave FBM.
inline float CloudWorleyFBM(float3 p, uint quality)
{
    return g_cloudNoise.SampleLevel(g_sampler,
        p * (1.0f / CLOUD_NOISE_G_PERIOD), 0).g;
}

// Runtime 2-octave FBM on the baked Perlin-Worley channel; the offset
// breaks visible octave alignment, weights keep the result in [0,1].
inline float CloudPerlinWorley(float3 p)
{
    float3 uvw0 = p * (1.0f / CLOUD_NOISE_R_PERIOD);
    float3 uvw1 = uvw0 * 2.0f + float3(0.117f, 0.053f, 0.239f);
    float  n0   = g_cloudNoise.SampleLevel(g_sampler, uvw0, 0).r;
    float  n1   = g_cloudNoise.SampleLevel(g_sampler, uvw1, 0).r;
    return saturate(n0 * 0.65f + n1 * 0.35f);
}

// Sphere-aware noise coord: radial walk = altitude along local up at each
// planet point, so cumulus columns stay vertical at home, limb, and poles
// (world-Y stamping stretched them horizontally at the limb). Wind applied
// in noise space so drift stays globally consistent.
inline float3 CloudNoiseCoord(float3 P, float timeSec)
{
    float  r   = length(P);
    float3 dir = (r > 1e-6f) ? (P / r) : float3(0, 1, 0);
    float  midAlt = 0.5f * (CLOUD_LAYER_BOT_KM + CLOUD_LAYER_TOP_KM);
    float3 wind   = float3(CLOUD_WIND_X, 0.0f, CLOUD_WIND_Z) * timeSec;
    return P - dir * midAlt + wind;
}

// Wind-free horizontal coord — wind here would migrate per-cloud top
// altitudes across the planet, which isn't the intended visual.
inline float3 CloudSphereHorizCoord(float3 P)
{
    float  r   = length(P);
    float3 dir = (r > 1e-6f) ? (P / r) : float3(0, 1, 0);
    return dir * ATMOS_BOTTOM_RADIUS;
}

// Per-cloud top altitude (towering vs flat). Uses spherical horizontal
// coord; the (X,0,Z) form had a pole singularity that pinched top noise.
inline float CloudEffectiveTopKm(float3 P)
{
    float3 horizP = CloudSphereHorizCoord(P);
    float  n = CloudValueNoise(horizP * CLOUD_TOP_FREQ);
    return CLOUD_LAYER_TOP_KM + n * CLOUD_TOP_VARIATION_KM;
}

// Heart-shape vertical profile (Nubis). FromTop variant skips the
// CloudEffectiveTopKm tap when the caller already has effTop.
inline float CloudAltitudeProfileFromTop(float altKm, float effTop)
{
    float h = (altKm - CLOUD_LAYER_BOT_KM)
            / max(1e-4f, effTop - CLOUD_LAYER_BOT_KM);
    if (h <= 0.0f || h >= 1.0f) return 0.0f;
    float bottomRamp = smoothstep(0.00f, 0.18f, h);
    float topCap     = 1.0f - smoothstep(0.55f, 1.00f, h);
    return bottomRamp * topCap;
}

inline float CloudAltitudeProfile(float altKm, float3 P)
{
    return CloudAltitudeProfileFromTop(altKm, CloudEffectiveTopKm(P));
}

inline float coverageModulation(float coverage, float detail, float filterWidth)
{
    float f = 1.0f - coverage;
    float modDetail = detail * (1.0f - filterWidth) + filterWidth;
    return saturate((modDetail - f) / max(filterWidth, 1e-3f));
}

// Planet-scale coverage = editor base × NASA Blue Marble map, scaled so
// map=0.5 reproduces the base. The map is keyed in absolute planet lat/lon,
// so the world ENU direction must rotate into planet-absolute coords before
// the equirect math — otherwise the polar singularity sits at world +Y
// (overhead) and pinches the zenith every render.
//
// The ENU basis is uniform per invocation; cached in thread-local statics
// (init once at every cloud entry point) so per-sample CloudGlobalCoverage
// pays a 9-mul rotation instead of 4 trig + matrix build. Identity defaults
// keep a pre-init read sane.
static float3 g_cloudEnuEast  = float3(1, 0, 0);
static float3 g_cloudEnuUp    = float3(0, 1, 0);
static float3 g_cloudEnuNorth = float3(0, 0, 1);

inline void InitCloudEnuBasis()
{
    float L   = SUN_LATITUDE_DEG  * DEG2RAD;
    float lon = SUN_LONGITUDE_DEG * DEG2RAD;
    float sL = sin(L), cL = cos(L);
    float sLon = sin(lon), cLon = cos(lon);
    g_cloudEnuEast  = float3(-sLon,       0.0f, cLon);
    g_cloudEnuUp    = float3( cL * cLon,  sL,   cL * sLon);
    g_cloudEnuNorth = float3(-sL * cLon,  cL,  -sL * sLon);
}

inline float3 EnuToPlanetDir(float3 dirEnu)
{
    return dirEnu.x * g_cloudEnuEast
         + dirEnu.y * g_cloudEnuUp
         + dirEnu.z * g_cloudEnuNorth;
}

// Plain bilinear+wrap sampler avoids the aniso seam across the ±180° meridian
// (atan2 jumps ~1.0 between adjacent pixels, aniso reads that as huge mag).
inline float CloudSampleCoverageMap(float3 P)
{
    float3 dirPlanet = EnuToPlanetDir(SafeNormalize(P));
    float u = atan2(dirPlanet.z, dirPlanet.x) * (0.5f / PI) + 0.5f;
    float v = acos(clamp(dirPlanet.y, -1.0f, 1.0f)) * (1.0f / PI);
    return g_cloudCoverage.SampleLevel(g_samplerLinearWrap, float2(u, v), 0);
}

inline float CloudGlobalCoverage(float3 P)
{
    float base = saturate(CLOUD_COVERAGE_BASE);
    float map  = CloudSampleCoverageMap(P);
    return saturate(base * map * 2.0f);
}

// Cheap "is there cloud here?" upper-bound. Ex variant exposes the
// components so a following CloudSampleMaterialFromHull doesn't recompute
// them; the bare wrapper is for the ambient column probe.
float CloudProbeHullEx(float3 P, float timeSec,
                       out float coverage, out float profile, out float effTop)
{
    coverage = 0.0f;
    profile  = 0.0f;
    effTop   = CLOUD_LAYER_TOP_KM;

    float r   = length(P);
    float alt = r - ATMOS_BOTTOM_RADIUS;

    effTop  = CloudEffectiveTopKm(P);
    profile = CloudAltitudeProfileFromTop(alt, effTop);
    if (profile <= 0.0f) return 0.0f;

    coverage = CloudGlobalCoverage(P);
    return coverage * profile;
}

float CloudProbeHull(float3 P, float timeSec)
{
    float coverage, profile, effTop;
    return CloudProbeHullEx(P, timeSec, coverage, profile, effTop);
}

struct CloudMaterial
{
    float density;     // final density after coverage + erosion [0..1]
    float profile;     // macro shape before erosion [0..1]
    float heightFrac;  // position within the layer [0..1]
};

// FromHull: caller passes the hull's (coverage, profile, effTop) so this
// skips the redundant equirect fetch + EffectiveTop tap. Requires hull > 0.
CloudMaterial CloudSampleMaterialFromHull(
    float3 P, float timeSec, float lodT, uint quality,
    float coverage, float profile, float effTop)
{
    CloudMaterial m;
    m.density    = 0.0f;
    m.profile    = 0.0f;
    m.heightFrac = 0.0f;

    float alt = length(P) - ATMOS_BOTTOM_RADIUS;
    // heightFrac uses per-cloud effTop so MS height-bias reads correctly
    // for tall vs short cumulus.
    float h   = (alt - CLOUD_LAYER_BOT_KM)
              / max(1e-4f, effTop - CLOUD_LAYER_BOT_KM);
    m.heightFrac = saturate(h);

    float3 q = CloudNoiseCoord(P, timeSec);

    // Low-frequency domain warp breaks the stamped-grid look; skipped at
    // distance where it can't resolve.
    float warpBlend = saturate(1.0f - lodT * 2.5f);
    float3 shapeQ = q;
    if (warpBlend > 0.001f)
    {
        float3 wp = q * 0.7f;
        float3 warp = float3(
            CloudValueNoise(wp + float3(  7.7f, 0.0f, 0.0f)),
            CloudValueNoise(wp + float3( 13.1f, 0.0f, 0.0f)),
            CloudValueNoise(wp + float3( 23.5f, 0.0f, 0.0f))
        );
        shapeQ += (warp - 0.5f) * CLOUD_WARP_AMP_KM * warpBlend;
    }

    float base = (quality == 0u)
               ? CloudPerlinWorley(shapeQ * CLOUD_BASE_FREQ)
               : CloudWorleyFBM   (shapeQ * CLOUD_BASE_FREQ, quality);

    float coverageHull = coverage * profile;
    float d = coverageModulation(coverageHull, base, CLOUD_COVMOD_FILTER_WIDTH);
    if (d <= 0.001f) return m;

    m.profile = saturate(d);

    // HF erosion: silhouette wisps. Edge-masked + LOD-tapered.
    float hfAmount = CLOUD_HF_AMOUNT * lerp(1.0f, 0.6f, lodT);
    if (hfAmount > 0.001f)
    {
        float hf = CloudValueNoise(q * CLOUD_HF_FREQ);
        float edgeMask = (1.0f - d);
        edgeMask = edgeMask * edgeMask * edgeMask;
        d = saturate(d - hfAmount * hf * edgeMask);
    }

    float hf2Amount = hfAmount * lerp(0.45f, 0.0f, lodT);
    if (hf2Amount > 0.001f)
    {
        float hf2 = CloudValueNoise(q * CLOUD_HF_FREQ * 2.5f
                                    + float3(13.1f, 5.7f, 19.3f));
        float midMask = (1.0f - d);
        midMask = midMask * midMask;
        d = saturate(d - hf2Amount * hf2 * midMask);
    }

    // Near-field cauliflower bumps. Mid-mask keeps it off silhouettes.
    if (quality == 0u)
    {
        float midModAmount = 0.15f * lerp(1.0f, 0.3f, lodT);
        if (midModAmount > 0.001f)
        {
            float midW = CloudWorley(shapeQ * CLOUD_BASE_FREQ * 5.0f
                                      + float3(31.7f, 11.9f, 23.1f));
            float modulation = 2.0f * ((1.0f - midW) - 0.5f);
            float midMask    = 1.0f - abs(2.0f * d - 1.0f);
            d = saturate(d + modulation * midModAmount * midMask);
        }
    }

    // Soft density curve lifts mid-densities so the body fills out.
    d = pow(saturate(d), lerp(0.85f, 1.0f, d));

    m.density = d;
    return m;
}

// Wrapper for callers without a precomputed hull.
CloudMaterial CloudSampleMaterial(float3 P, float timeSec, float lodT, uint quality)
{
    float coverage, profile, effTop;
    float hull = CloudProbeHullEx(P, timeSec, coverage, profile, effTop);
    if (hull <= 0.0f)
    {
        CloudMaterial m = (CloudMaterial)0;
        return m;
    }
    return CloudSampleMaterialFromHull(P, timeSec, lodT, quality,
                                       coverage, profile, effTop);
}


bool RayCloudShell(float3 ro, float3 rd, out float tNear, out float tFar)
{
    tNear = 0.0f;
    tFar  = 0.0f;

    float Rb = ATMOS_BOTTOM_RADIUS;
    // Rt uses the MAX possible cloud top so the shell captures tall cumulus
    // variation; out-of-cloud shell samples cost nothing via the profile gate.
    float Rt = ATMOS_BOTTOM_RADIUS + CLOUD_LAYER_TOP_KM + CLOUD_TOP_VARIATION_KM;
    float Rm = ATMOS_BOTTOM_RADIUS + CLOUD_LAYER_BOT_KM;

    float tT0, tT1, tM0, tM1;
    bool  hitTop = RaySphereIntersect(ro, rd, Rt, tT0, tT1);
    if (!hitTop || tT1 <= 0.0f) return false;
    bool  hitMid = RaySphereIntersect(ro, rd, Rm, tM0, tM1);

    float r = length(ro);

    if (r >= Rm && r <= Rt)
    {
        tNear = 0.0f;
        tFar  = tT1;
        if (hitMid && tM0 > 0.0f && tM0 < tFar) tFar = tM0;
    }
    else if (r > Rt)
    {
        if (tT0 <= 0.0f) return false;
        tNear = tT0;
        tFar  = tT1;
        if (hitMid && tM0 > 0.0f && tM0 < tFar) tFar = tM0;
    }
    else
    {
        if (!hitMid || tM1 <= 0.0f) return false;
        tNear = max(tM1, 0.0f);
        tFar  = tT1;
    }

    float tG0, tG1;
    if (RaySphereIntersect(ro, rd, Rb, tG0, tG1) && tG0 >= 0.0f && tG0 < tFar)
        tFar = tG0;

    return tFar > tNear;
}

inline float3 SampleConeAroundDir(float3 dir, float cosThetaMax, float2 u)
{
    float z = 1.0f - u.y * (1.0f - cosThetaMax);
    float sinT = sqrt(max(0.0f, 1.0f - z * z));
    float phi = TAU * u.x;
    float3 local = float3(cos(phi) * sinT, sin(phi) * sinT, z);
    float3 T, B;
    GetOrthoBasis(dir, T, B);
    return SafeNormalize(local.x * T + local.y * B + local.z * dir);
}

inline float CloudEarthShadowFactor(float3 P, float3 L)
{
    float3 oc = P;
    float  b  = dot(oc, L);
    if (b >= 0.0f) return 1.0f;

    float Rb   = ATMOS_BOTTOM_RADIUS;
    float c    = dot(oc, oc) - Rb * Rb;
    float disc = b * b - c;

    float penumbra = Rb * Rb * 2e-2f;
    return saturate(0.5f - disc / penumbra);
}

inline float CloudHG(float cosT, float g)
{
    float g2 = g * g;
    float denom = 1.0f + g2 - 2.0f * g * cosT;
    return (1.0f / (4.0f * PI)) * (1.0f - g2)
         / max(1e-4f, denom * sqrt(max(1e-4f, denom)));
}

// Jendersie & d'Eon 2023 Mie approximation: HG + Draine lobe, parameters
// fit to droplet diameter. Sharper silver lining than HG; lifted backward
// shoulder. CLOUD_SILVER_INTENSITY drives diameter (5..30 um).
inline float JDDraineLobe(float cosT, float g, float alpha)
{
    float g2 = g * g;
    float t  = 1.0f + g2 - 2.0f * g * cosT;
    float hgPart = (1.0f - g2) / (4.0f * PI * t * sqrt(max(1e-6f, t)));
    float boost  = (1.0f + alpha * cosT * cosT)
                 / (1.0f + alpha * (1.0f + 2.0f * g2) / 3.0f);
    return hgPart * boost;
}

// JD parameters depend only on droplet diameter (uniform); cache per-thread
// so CloudPhaseDirect skips the 4 exp/divides per sample. Defaults match
// d ~ 17.5 um so a pre-init read stays sane.
static float g_jdGHG   = 0.954f;
static float g_jdGD    = 0.491f;
static float g_jdAlpha = 23.8f;
static float g_jdWD    = 0.392f;

inline void InitCloudJDPhase()
{
    // Slider 0..1 maps to droplet diameter 5..30 um (typical cumulus 10..15).
    // Floor at 2 um — the rational fits assume d > ~1.7.
    float d = lerp(5.0f, 30.0f, saturate(CLOUD_SILVER_INTENSITY));
    d        = max(d, 2.0f);
    g_jdGHG  = exp(-0.0990567f / (d - 1.67154f));
    g_jdGD   = exp(-2.20679f  / (d + 3.91029f)) - 0.428934f;
    g_jdAlpha= exp( 3.62489f  - 8.29288f / (d + 5.52825f));
    g_jdWD   = saturate(exp(-0.599085f / (d - 0.641583f)) - 0.665888f);
}

inline float CloudPhaseDirect(float cosT)
{
    return (1.0f - g_jdWD) * CloudHG(cosT, g_jdGHG)
         + g_jdWD          * JDDraineLobe(cosT, g_jdGD, g_jdAlpha);
}

// Multi-scatter phase: mostly isotropic with a soft forward bias.
inline float CloudPhaseMS(float cosT)
{
    const float kIso = 0.07957747f;  // 1 / (4 * pi)
    return lerp(kIso, CloudHG(cosT, CLOUD_SECONDARY_G), 0.4f);
}

// Shadow-density variant. MUST use the same Perlin-Worley base + coverage
// source as the view march or shadows desync from the visible cloud
// silhouette. Skips the cauliflower and density curve (sub-100m detail,
// invisible at shadow-march resolution).
inline float CloudDensityForShadow(float3 P, float timeSec)
{
    float alt     = length(P) - ATMOS_BOTTOM_RADIUS;
    float profile = CloudAltitudeProfile(alt, P);
    if (profile <= 0.0f) return 0.0f;

    float coverage = CloudGlobalCoverage(P);
    if (coverage <= 0.0f) return 0.0f;

    float3 q    = CloudNoiseCoord(P, timeSec);
    float  base = CloudPerlinWorley(q * CLOUD_BASE_FREQ);

    float coverageHull = coverage * profile;
    float d = coverageModulation(coverageHull, base, CLOUD_COVMOD_FILTER_WIDTH);
    if (d <= 0.001f) return 0.0f;

    // HF erosion at the view march's lodT=1 strength.
    float hfAmount = CLOUD_HF_AMOUNT * 0.6f;
    if (hfAmount > 0.001f)
    {
        float hf       = CloudValueNoise(q * CLOUD_HF_FREQ);
        float edgeMask = 1.0f - d;
        edgeMask       = edgeMask * edgeMask * edgeMask;
        d              = saturate(d - hfAmount * hf * edgeMask);
    }

    return d;
}

// Macro-shape only (no HF / cauliflower) — DLSS RR normal needs to be
// stable at the ~1km feature scale, not jittered by 100m detail noise.
inline float CloudShapeDensity(float3 P, float timeSec)
{
    float alt     = length(P) - ATMOS_BOTTOM_RADIUS;
    float profile = CloudAltitudeProfile(alt, P);
    if (profile <= 0.0f) return 0.0f;

    float coverage = CloudGlobalCoverage(P);
    if (coverage <= 0.0f) return 0.0f;

    float3 q    = CloudNoiseCoord(P, timeSec);
    float  base = CloudPerlinWorley(q * CLOUD_BASE_FREQ);
    return coverageModulation(coverage * profile, base,
                              CLOUD_COVMOD_FILTER_WIDTH);
}

// 0.15 km ≈ 5 % of base-noise wavelength: directional gradient without
// sub-step noise.
#define CLOUD_NORMAL_GRAD_STEP_KM 0.15f

inline float3 ExtractCloudNormal(float3 P, float timeSec)
{
    const float eps = CLOUD_NORMAL_GRAD_STEP_KM;
    float dxp = CloudShapeDensity(P + float3(eps, 0, 0), timeSec);
    float dxm = CloudShapeDensity(P - float3(eps, 0, 0), timeSec);
    float dyp = CloudShapeDensity(P + float3(0, eps, 0), timeSec);
    float dym = CloudShapeDensity(P - float3(0, eps, 0), timeSec);
    float dzp = CloudShapeDensity(P + float3(0, 0, eps), timeSec);
    float dzm = CloudShapeDensity(P - float3(0, 0, eps), timeSec);

    // Negate: density rises INTO the cloud, normal points outward.
    float3 grad = float3(dxp - dxm, dyp - dym, dzp - dzm);
    float  gradLen2 = dot(grad, grad);
    if (gradLen2 < 1e-12f) return float3(0, 0, 0);
    return -grad * rsqrt(gradLen2);
}

// Frame-stable (depth, normal) for DLSS RR g-buffer. Uses macro-shape
// density at deterministic stratified positions so the result doesn't
// jitter frame-to-frame (the Monte Carlo radiance march would, and DLSS
// RR reads depth jitter as motion → cloud boiling).
//
// maxDistKm > 0 = mesh terminates the ray at that distance; the march
// clips to min(shell exit, mesh) so a mesh inside the cloud shell
// doesn't push the Hillaire mean depth past it.
// cloudAlphaOut is cloud-only opacity (atmosphere excluded); shading pass
// uses it to pick cloud-vs-mesh DLSS dominance independent of in-scatter.
void EvaluateCloudGBuffer(float3 V,
                          float  maxDistKm,
                          out float  depthKmOut,
                          out float3 normalWSOut,
                          out float  cloudAlphaOut)
{
    depthKmOut    = 0.0f;
    normalWSOut   = float3(0, 0, 0);
    cloudAlphaOut = 0.0f;

#if !ENABLE_CLOUDS
    return;
#else
    if (cloud_enabled < 0.5f) return;

    InitCloudEnuBasis();
    InitCloudJDPhase();

    float3 O  = g_skyObserverPlanet;
    float3 Vn = SafeNormalize(V);

    float tNear, tFar;
    if (!RayCloudShell(O, Vn, tNear, tFar)) return;

    // Mesh clip: cloud beyond a mesh inside the shell can't contribute.
    if (maxDistKm > 0.0f) tFar = min(tFar, maxDistKm);

    // Past ~60 km the first visible cloud dominates the mean depth.
    const float maxLenKm   = 60.0f;
    const int   N_GBUF_MAX = 32;
    const float stepGoalKm = 0.6f;

    float tEnd = min(tFar, tNear + maxLenKm);
    if (tEnd <= tNear) return;

    int   N  = clamp((int)((tEnd - tNear) / stepGoalKm + 0.5f), 8, N_GBUF_MAX);
    float ds = (tEnd - tNear) / (float)N;

    float trShape  = 1.0f;
    float sumDepth = 0.0f;
    float sumW     = 0.0f;

    [loop]
    for (int i = 0; i < N_GBUF_MAX; ++i)
    {
        if (i >= N) break;
        float t  = tNear + ((float)i + 0.5f) * ds;
        float3 P = O + Vn * t;

        float d     = CloudShapeDensity(P, walltime);
        float segTr = exp(-d * CLOUD_EXTINCTION * ds);
        float alpha = 1.0f - segTr;
        float w     = trShape * alpha;
        sumDepth   += w * t;
        sumW       += w;
        trShape    *= segTr;

        if (trShape < 0.01f) break;
    }

    cloudAlphaOut = saturate(1.0f - trShape);

    if (sumW > 1e-5f)
    {
        depthKmOut = sumDepth / sumW;
        float3 surfaceP = O + Vn * depthKmOut;
        normalWSOut    = ExtractCloudNormal(surfaceP, walltime);
    }
#endif
}

// Per-sample sun shadow march. Geometric ramp out to ~380 km so low-sun
// rays reach distant cloud banks; TAU_EARLYOUT keeps noon rays cheap.
float CloudOpticalDepthToSun(float3 P, float3 L, float timeSec)
{
    const float SAMPLE_DIST_KM[12] = { 0.2f, 0.6f, 1.5f, 3.5f, 7.0f, 12.0f, 22.0f, 40.0f, 70.0f, 125.0f, 220.0f, 380.0f };
    const float SAMPLE_SEG_KM[12]  = { 0.2f, 0.4f, 1.1f, 2.0f, 3.5f,  5.0f, 10.0f, 18.0f, 30.0f,  50.0f,  90.0f, 150.0f };
    const float TAU_EARLYOUT = 64.0f;

    float tau = 0.0f;
    [loop]
    for (int i = 0; i < 12; ++i)
    {
        float3 Q = P + L * SAMPLE_DIST_KM[i];
        float  d = CloudDensityForShadow(Q, timeSec);
        tau += d * CLOUD_EXTINCTION * SAMPLE_SEG_KM[i];
        if (tau >= TAU_EARLYOUT) break;
    }
    return tau * CLOUD_SUN_TAU_MULT;
}

// Surface shadow march (ground NEE + aerial perspective). MUST use the
// real noise density — the 2D hull is uniform across the shell and
// over-darkens surfaces where no cloud actually sits overhead.
// CLOUD_SHADOW_STEPS=1 takes a single-midpoint fast path (~4x cheaper,
// visually fine for overhead cumulus). N≥2 walks the shell.
float CloudOpticalDepthAlongRay(float3 P, float3 D, float maxLenKm, float timeSec)
{
    int N = max(CLOUD_SHADOW_STEPS, 1);

    if (N == 1)
    {
        // One midpoint sample × oblique-path correction (1/cos sun zenith).
        float midAlt = 0.5f * (CLOUD_LAYER_BOT_KM + CLOUD_LAYER_TOP_KM);
        float Rmid   = ATMOS_BOTTOM_RADIUS + midAlt;
        float r      = length(P);
        float dotPD  = dot(P, D);
        float disc   = dotPD * dotPD - (r * r - Rmid * Rmid);
        if (disc < 0.0f) return 0.0f;
        float tMid   = -dotPD + sqrt(disc);
        if (tMid <= 0.0f || tMid > maxLenKm) return 0.0f;

        float3 Q = P + D * tMid;
        float  d = CloudDensityForShadow(Q, timeSec);
        float shellTh = CLOUD_LAYER_TOP_KM - CLOUD_LAYER_BOT_KM;
        float cosZ    = saturate(dot(D, SafeNormalize(Q)));
        float pathLen = shellTh / max(cosZ, 0.15f);
        return d * CLOUD_EXTINCTION * pathLen;
    }

    // Multi-sample shell march (N >= 2).
    float tNear, tFar;
    if (!RayCloudShell(P, D, tNear, tFar)) return 0.0f;
    tFar = min(tFar, tNear + maxLenKm);
    if (tFar <= tNear) return 0.0f;

    const float ds = (tFar - tNear) / (float)N;
    float tau = 0.0f;
    [loop]
    for (int i = 0; i < N; ++i)
    {
        float  ti = tNear + ((float)i + 0.5f) * ds;
        float3 Q  = P + D * ti;
        tau += CloudDensityForShadow(Q, timeSec) * CLOUD_EXTINCTION * ds;
    }
    return tau;
}

float CloudSunVisibility(float3 surfacePosWorld, float3 sunDirWS)
{
#if !ENABLE_CLOUDS
    return 1.0f;
#else
    if (cloud_enabled < 0.5f) return 1.0f;
    InitCloudEnuBasis();

    float3 P = WorldToPlanet(surfacePosWorld);
    float  r = length(P);
    // MAX top so tall cumulus still cast when observer is above baseline.
    if (r - ATMOS_BOTTOM_RADIUS > CLOUD_LAYER_TOP_KM + CLOUD_TOP_VARIATION_KM + 1e-3f) return 1.0f;

    float tau = CloudOpticalDepthAlongRay(P, SafeNormalize(sunDirWS), 50.0f, walltime);
    return exp(-tau);
#endif
}

// Planet-space variant — skips WorldToPlanet's flat-earth reprojection,
// which doesn't round-trip spherical-projection inputs (cloud foot on
// planet surface).
float CloudSunVisibilityPlanet(float3 Pplanet, float3 sunDirWS)
{
#if !ENABLE_CLOUDS
    return 1.0f;
#else
    if (cloud_enabled < 0.5f) return 1.0f;
    InitCloudEnuBasis();

    float r = length(Pplanet);
    if (r - ATMOS_BOTTOM_RADIUS > CLOUD_LAYER_TOP_KM + CLOUD_TOP_VARIATION_KM + 1e-3f) return 1.0f;

    float tau = CloudOpticalDepthAlongRay(Pplanet, SafeNormalize(sunDirWS), 50.0f, walltime);
    return exp(-tau);
#endif
}

// Trapezoidal column density above P at exponentially-spaced altitudes.
inline float CloudColumnDensityAbove(float3 P, float3 upDir, float startDensity)
{
    const int Namb = clamp((int)CLOUD_AMBIENT_STEPS, 2, CLOUD_AMBIENT_STEPS_MAX);
    const float kZ[CLOUD_AMBIENT_STEPS_MAX] =
        { 0.3f, 0.8f, 1.6f, 3.0f, 5.0f, 8.0f };

    float columnKm = 0.0f;
    float dPrev    = startDensity;
    float zPrev    = 0.0f;
    [loop]
    for (int ip = 0; ip < CLOUD_AMBIENT_STEPS_MAX; ++ip)
    {
        if (ip >= Namb) break;
        float zCur = kZ[ip];
        float dCur = CloudProbeHull(P + upDir * zCur, walltime);
        columnKm  += 0.5f * (dPrev + dCur) * (zCur - zPrev);
        zPrev      = zCur;
        dPrev      = dCur;
    }
    return columnKm;
}

// Per-sample in-scattered radiance. Caller folds it as sigma_t * L * segIn.
float3 CloudComputeLighting(float3 P, float3 V, float3 L, CloudMaterial m,
                            float3 sunRad, float3 sunAtmos,
                            float3 skyAmbientTop, float3 skyAmbientHorizon,
                            float3 groundIrrad,
                            float3 localUp,
                            float cosTheta,
                            float earthShadow,
                            uint2 pixel, uint frame, uint tap)
{
    // Cone-jittered shadow taps. Min 1.5° even at Kshadow=1: without jitter,
    // neighbouring view samples hit the same six SAMPLE_DIST_KM points and
    // the heart-profile bands wallpaper into horizontal stripes.
    int Kshadow = clamp((int)CLOUD_SHADOW_CONE_SAMPLES, 1, CLOUD_SHADOW_CONE_SAMPLES_MAX);
    const float kMinShadowConeDeg = 1.5f;
    float coneDeg    = max(CLOUD_SHADOW_CONE_DEG, kMinShadowConeDeg);
    float cosConeMax = cos(coneDeg * DEG2RAD);

    float sunOD = 0.0f;
    [loop]
    for (int k = 0; k < CLOUD_SHADOW_CONE_SAMPLES_MAX; ++k)
    {
        if (k >= Kshadow) break;
        float4 rk = CloudRand4(pixel, frame, tap + (uint)k);
        float3 Lk = SampleConeAroundDir(L, cosConeMax, rk.xy);
        sunOD += CloudOpticalDepthToSun(P, Lk, walltime);
    }
    sunOD /= (float)Kshadow;

    float h = m.heightFrac;
    float3 K = sunRad * sunAtmos * earthShadow * CLOUD_ALBEDO;

    // Direct = Wrenninge octave 0.
    float  Tdir   = exp(-sunOD);
    float3 direct = K * Tdir * CloudPhaseDirect(cosTheta);

    // Floor at FLOOR so bases stay grey-lit, not black.
    float heightBias  = lerp(CLOUD_MS_HEIGHT_FLOOR, 1.0f,
                             pow(saturate(h), 0.5f));

    // MS_MODE 0 = Nubis sqrt(Tdir) shortcut (cheap, default).
    // MS_MODE 1,2 = N extra Wrenninge octaves (isotropic, more fill).
    float3 ms = float3(0, 0, 0);
    if (CLOUD_MS_MODE == 0)
    {
        float depthFactor = pow(max(Tdir, 1e-6f), 0.5f);
        float msAmount    = depthFactor * heightBias
                          * (0.5f + CLOUD_SECONDARY_STRENGTH)
                          * CLOUD_MS_STRENGTH;
        ms = K * msAmount * CloudPhaseMS(cosTheta);
    }
    else
    {
        // Hillaire 2016 §5.8 octaves: extinction × a^n, weight × b^n,
        // phase collapsed to isotropic 1/(4π).
        const float a = 0.5f;
        const float b = 0.5f;
        const float kIso = 1.0f / (4.0f * PI);
        const int extraOctaves = clamp(CLOUD_MS_MODE, 1, 2);
        [unroll]
        for (int n = 1; n <= 2; ++n)
        {
            if (n > extraOctaves) break;
            float an = pow(a, (float)n);
            float bn = pow(b, (float)n);
            float Tn = exp(-sunOD * an);
            ms += K * bn * Tn * kIso;
        }
        ms *= heightBias * CLOUD_MS_STRENGTH;
    }

    float3 inscatter = direct + ms;

    // Sky + ground bounce ambient, gated by column density above sample.
    float3 ambient = float3(0, 0, 0);
    if (CLOUD_SKY_AMBIENT > 0.5f || CLOUD_GROUND_BOUNCE > 0.5f)
    {
        float columnKm = CloudColumnDensityAbove(P, localUp, m.profile);
        float ambOD    = min(columnKm * CLOUD_EXTINCTION * CLOUD_AMBIENT_AO_SCALE,
                             CLOUD_AMBIENT_OD_MAX);
        float skyOcc   = exp(-ambOD);
        float ambExp   = pow(saturate(1.0f - m.profile), 0.5f);

        if (CLOUD_SKY_AMBIENT > 0.5f)
        {
            float3 skyColor = lerp(skyAmbientHorizon, skyAmbientTop, h);
            ambient += skyColor * ambExp * skyOcc * CLOUD_AMBIENT_INTENSITY;
        }
        if (CLOUD_GROUND_BOUNCE > 0.5f)
        {
            // Tdir gates bounce — ground beneath thick cloud isn't sun-lit
            // either, so the bounce reaching this sample must die with it.
            ambient += groundIrrad * ambExp * (1.0f - h) * Tdir;
        }
    }

    return inscatter + ambient;
}

// Bounce-ray cheap variant. Mirrors the primary's lighting model
// (CloudComputeLighting equivalent — distFade, MS mode dispatch,
// disk-fraction earth shadow) but trims the expensive parts: single
// shadow tap (no cone), midpoint sunAtmos (one TransmittanceToSun),
// no sky ambient / no ground bounce probe, no atmosphere coupling.
//
// Step sizing is adaptive (small step in cloud, coarse jump in empty)
// ported from the primary path — the old fixed ds = (tFar-tNear)/N gave
// 30..50 km steps at grazing angles, and one cloudy sample inside such
// a step produced tau = density * 4 * 50 ≈ 20 → segTr ≈ 0, hard-zeroing
// trCloud and killing the sky behind every reflection. The cheap empty
// step is capped looser than the primary's (kCheapEmptyStepCap ≈ 8 km)
// since this loop only has CLOUD_CHEAP_STEPS iterations (≈10) vs the
// primary's CLOUD_VIEW_STEPS_MAX (≈256).
float3 EvaluateCloudsCheap(float3 V, float3 sunDir, float3 sunIrradiance,
                           out float3 cloudTrOut)
{
    cloudTrOut = float3(1, 1, 1);

#if !ENABLE_CLOUDS
    return float3(0, 0, 0);
#else
    if (cloud_enabled < 0.5f) return float3(0, 0, 0);

    InitCloudEnuBasis();
    InitCloudJDPhase();

    float3 O = g_skyObserverPlanet;
    float3 L = SafeNormalize(sunDir);

    float tNear, tFar;
    if (!RayCloudShell(O, V, tNear, tFar)) return float3(0, 0, 0);

    tFar = min(tFar, tNear + CLOUD_CHEAP_MAX_LEN_KM);
    if (tFar <= tNear) return float3(0, 0, 0);

    const float cosTheta = dot(V, L);
    const float phaseDir = CloudPhaseDirect(cosTheta);
    const float kIso     = 1.0f / (4.0f * PI);
    const int   msMode   = CLOUD_MS_MODE;

    // One sunAtmos tap at the shell midpoint — per-sample TransmittanceToSun
    // is the most expensive thing in the primary path and the largest perf
    // saving we get here. Bias shows up at low sun for samples far from PMid.
    float3 PMid     = O + V * (0.5f * (tNear + tFar));
    float3 sunAtmos = TransmittanceToSun(PMid, L,
                                         ATMOS_BOTTOM_RADIUS,
                                         ATMOS_TOP_RADIUS);

    uint2 px   = DispatchRaysIndex().xy;
    uint  seed = initRandomData(px, uint2(0, 0), (uint)time, 73u);

    float3 L_acc   = float3(0, 0, 0);
    float3 trCloud = float3(1, 1, 1);

    // 8 km empty cap × ~10 iters ≈ 80 km clear-sky coverage. Lerped against
    // fineStep by lodT so near-camera samples stay tight and don't skip over
    // visible cumulus, while distant empty space gets stridden through fast.
    const float kCheapEmptyStepCap = 8.0f;

    float t        = tNear;
    float fineStep = CLOUD_TARGET_STEP_KM
                   * lerp(0.55f, 1.0f, CloudLodT(tNear));

    [loop]
    for (int i = 0; i < CLOUD_CHEAP_STEPS; ++i)
    {
        if (t >= tFar) break;
        if (max(trCloud.r, max(trCloud.g, trCloud.b)) < CLOUD_TR_EPS) break;

        float distFade = 1.0f - smoothstep(CLOUD_FADE_DISTANCE_KM,
                                            CLOUD_RENDER_DISTANCE_KM, t);
        float lodT     = CloudLodT(t);
        float rJit     = RandomFloatSingle(seed);
        float tSample  = t + rJit * fineStep;
        float3 P       = O + V * tSample;

        float hCoverage = 0.0f, hProfile = 0.0f, hEffTop = CLOUD_LAYER_TOP_KM;
        float hullRaw   = (distFade > 0.0f)
                        ? CloudProbeHullEx(P, walltime, hCoverage, hProfile, hEffTop)
                        : 0.0f;
        float hull      = hullRaw * distFade;
        bool  hullEmpty = (hull <= CLOUD_EFFECTIVE_ZERO_DENSITY);

        float thisStep;
        if (hullEmpty)
        {
            float maxStepAdaptive = min(
                kCheapEmptyStepCap * (1.0f + t * CLOUD_EMPTY_STEP_GROWTH_PER_KM),
                CLOUD_MAX_EMPTY_STEP_KM);
            thisStep = lerp(fineStep, maxStepAdaptive, saturate(lodT));
        }
        else
        {
            thisStep = fineStep;
        }
        thisStep = min(thisStep, tFar - t);

        if (!hullEmpty)
        {
            CloudMaterial m = CloudSampleMaterialFromHull(
                P, walltime, lodT, 0u, hCoverage, hProfile, hEffTop);
            float cloudDensity = m.density * distFade;
            if (cloudDensity > CLOUD_EFFECTIVE_ZERO_DENSITY)
            {
                float sigma_t = cloudDensity * CLOUD_EXTINCTION;
                float sunOD   = CloudOpticalDepthToSun(P, L, walltime);
                float Tdir    = exp(-sunOD);

                // Match primary: physical disk-fraction earth shadow
                // (smoothstep width = sun angular radius). The old
                // CloudEarthShadowFactor used a fixed penumbra and didn't
                // match the terminator transition the rest of the engine
                // uses.
                float3 Pnorm      = SafeNormalize(P);
                float  sunCosZ    = dot(Pnorm, L);
                float  cosHorizon = -sqrt(max(0.0f, 1.0f -
                                              (ATMOS_BOTTOM_RADIUS * ATMOS_BOTTOM_RADIUS)
                                              / dot(P, P)));
                float  earthShadow = SunDiskFractionAboveHorizon(sunCosZ, cosHorizon);

                float3 K = sunIrradiance * sunAtmos * earthShadow * CLOUD_ALBEDO;

                float h          = m.heightFrac;
                float heightBias = lerp(CLOUD_MS_HEIGHT_FLOOR, 1.0f,
                                        pow(saturate(h), 0.5f));

                // Same MS dispatch as CloudComputeLighting — without this
                // the cheap path was locked to Mode 0 (Nubis sqrt(Tdir)),
                // so reflections of MS Mode 2 scenes (the default) showed
                // much darker cloud cores than the primary view.
                float3 direct = K * Tdir * phaseDir;
                float3 ms     = float3(0, 0, 0);
                if (msMode == 0)
                {
                    float depthFactor = pow(max(Tdir, 1e-6f), 0.5f);
                    float msAmount    = depthFactor * heightBias
                                      * (0.5f + CLOUD_SECONDARY_STRENGTH)
                                      * CLOUD_MS_STRENGTH;
                    ms = K * msAmount * CloudPhaseMS(cosTheta);
                }
                else
                {
                    const float a = 0.5f;
                    const float b = 0.5f;
                    int extraOctaves = clamp(msMode, 1, 2);
                    [unroll]
                    for (int n = 1; n <= 2; ++n)
                    {
                        if (n > extraOctaves) break;
                        float an = pow(a, (float)n);
                        float bn = pow(b, (float)n);
                        float Tn = exp(-sunOD * an);
                        ms += K * bn * Tn * kIso;
                    }
                    ms *= heightBias * CLOUD_MS_STRENGTH;
                }

                float3 inscatter = direct + ms;

                float segTr = exp(-sigma_t * thisStep);
                L_acc   += trCloud * inscatter * (1.0f - segTr);
                trCloud *= segTr;
            }
        }

        if (!hullEmpty)
            fineStep = min(fineStep * CLOUD_STEP_GROWTH, CLOUD_MAX_FINE_STEP_KM);
        t += thisStep;
    }

    cloudTrOut = trCloud;
    return L_acc;
#endif
}

// Unified atmosphere+cloud march. Splits the ray into three phases:
//   [tStart, tC0]    atmosphere only
//   [tC0, tC1]       combined march inside cloud shell (hull-skipped)
//   [tC1, tEnd]      atmosphere only
// Each phase shares (inscatter, transmittance) so the total stays
// consistent. Replaces the old "atmosphere first then composite cloud"
// pipeline which double-attenuated and never let atmosphere shadow the
// cloud or vice versa. Caller adds planet body / stars / nightBase
// behind, attenuated by combined_T.

// Atmosphere-only segment with running state. Per-sample cloud shadow tap
// modulates sunIllum — without it the haze in front of clouds reads as
// fully sunlit and a bright Mie halo lands on every distant silhouette
// at sunset. STBN jitter spreads the sharp shadow edge for DLSS RR.
inline void IntegrateAtmosphereSegment(
    float3 O, float3 V, float3 L,
    float tStart, float tEnd, int N,
    float cosTheta, float phR, float phM,
    float3 sunIrradiance,
    uint2  pixel,
    inout float3 inscatter,
    inout float3 transmittance)
{
    if (tEnd <= tStart || N <= 0) return;
    float ds = (tEnd - tStart) / (float)N;

    const float kAtmosShadowConeCos = cos(ATMOS_CLOUD_SHADOW_CONE_DEG * DEG2RAD);
    const uint  frameU = (uint)time;

    [loop]
    for (int i = 0; i < N; ++i)
    {
        // STBN: r.x = step jitter, r.yz = cone jitter. Salt 500u+i keeps
        // it distinct from CloudComputeLighting's cone taps (200u+).
        float4 r  = CloudRand4(pixel, frameU, (uint)i + 500u);
        float  t  = tStart + ((float)i + r.x) * ds;
        float3 P  = O + V * t;
        float alt = max(0.0f, length(P) - ATMOS_BOTTOM_RADIUS);

        MediumSample med  = SampleMedium(alt);
        float3       segTr = exp(-med.extinction * ds);
        float3       sunTr = TransmittanceToSun(P, L,
                                                ATMOS_BOTTOM_RADIUS,
                                                ATMOS_TOP_RADIUS);

        // Disk-fraction earth shadow — smoothstep width = sun radius.
        float3 Pnorm      = SafeNormalize(P);
        float  sunCosZ    = dot(Pnorm, L);
        float  cosHorizon = -sqrt(max(0.0f, 1.0f -
                                      (ATMOS_BOTTOM_RADIUS * ATMOS_BOTTOM_RADIUS)
                                      / dot(P, P)));
        float  earthShadow = SunDiskFractionAboveHorizon(sunCosZ, cosHorizon);

        // Cone-jittered cloud shadow on the air column. Without jitter,
        // the fast-path single-tap correlates across pixels and the
        // Perlin-Worley pattern wallpapers onto haze as horizontal bands.
        // Floor keeps near-haze contributing so the far reddened samples
        // don't dominate and tint the integral sunset-orange at noon.
        float cloudVis = 1.0f;
        if (cloud_cloudShadowOnSurfaces > 0.5f)
        {
            float3 Lj = SampleConeAroundDir(L, kAtmosShadowConeCos,
                                            float2(r.y, r.z));
            cloudVis = CloudSunVisibilityPlanet(P, Lj);
            cloudVis = max(cloudVis, ATMOS_CLOUD_SHADOW_FLOOR);
        }

        float3 scatterPhase = med.scatterR * phR + med.scatterM * phM;
        float3 sunIllum     = sunIrradiance * earthShadow * sunTr * cloudVis
                            * ATMOS_MULTI_SCATTER_FACTOR;
        float3 rate         = scatterPhase * sunIllum;

        float3 scatterInteg;
        scatterInteg.x = (med.extinction.x > 1e-10f)
            ? rate.x * (1.0f - segTr.x) / med.extinction.x : rate.x * ds;
        scatterInteg.y = (med.extinction.y > 1e-10f)
            ? rate.y * (1.0f - segTr.y) / med.extinction.y : rate.y * ds;
        scatterInteg.z = (med.extinction.z > 1e-10f)
            ? rate.z * (1.0f - segTr.z) / med.extinction.z : rate.z * ds;

        inscatter     += transmittance * scatterInteg;
        transmittance *= segTr;
    }
}

// Returns combined in-scatter + transmittance + planet-hit flag. DLSS RR
// g-buffer comes from EvaluateCloudGBuffer separately (deterministic).
// maxDistKm > 0 = ray terminates on a mesh at that distance (skips the
// planet clip; hitPlanetOut stays false so caller doesn't add ground albedo).
float3 EvaluateAtmosphereAndClouds(
    float3 V, float3 sunDir, float3 sunIrradiance,
    float  maxDistKm,
    out float3 transmittanceOut,
    out bool   hitPlanetOut)
{
    transmittanceOut  = float3(1, 1, 1);
    hitPlanetOut      = false;
#if ATM_DEBUG_RING == 3
    return float3(0, 0, 0);            //sky/atmosphere disabled — ATM_DEBUG_RING
#endif

    InitCloudEnuBasis();
    InitCloudJDPhase();

    float3 O  = g_skyObserverPlanet;
    float3 Vn = SafeNormalize(V);
    float3 L  = SafeNormalize(sunDir);

    float tA0, tA1;
    if (!RaySphereIntersect(O, Vn, ATMOS_TOP_RADIUS, tA0, tA1) || tA1 <= 0.0f)
        return float3(0, 0, 0);

    float tStart = max(0.0f, tA0);
    float tEnd   = tA1;

    if (maxDistKm > 0.0f)
    {
        tEnd = min(tEnd, maxDistKm);
        // Robust below ground clamp. A coarse mesh planet (Milestone 1 uses
        // a UV sphere with chord drop on the order of km below ATMOS_BOTTOM_
        // RADIUS) produces hit points at altitude < 0. Without this clip the
        // march would step from camera through the analytical surface and
        // into "underground" along the ray; the density saturation
        // (max(0, alt) in the integrators) then makes every below ground
        // step contribute at ground density and inscatter / transmittance
        // both go to garbage. Clamping tEnd to the analytical ground entry
        // makes the atmosphere see the mesh as if it terminated cleanly at
        // ATMOS_BOTTOM_RADIUS; the residual transmittance error from the
        // missing few km between Rb and the actual mesh hit is ~2% at most
        // and disappears entirely once the cube sphere quadtree (M2+) is
        // tessellated finely enough to keep mesh hits above the surface.
        float tG0, tG1;
        if (RaySphereIntersect(O, Vn, ATMOS_BOTTOM_RADIUS, tG0, tG1)
            && tG0 > 0.0f && tG0 < tEnd)
        {
            tEnd = tG0;
        }
    }
    else
    {
        // Sky path: planet hit clips the ray.
        float tG0, tG1;
        if (RaySphereIntersect(O, Vn, ATMOS_BOTTOM_RADIUS, tG0, tG1)
            && tG0 > 0.0f && tG0 < tEnd)
        {
            tEnd         = tG0;
            hitPlanetOut = true;
        }
    }
    if (tEnd <= tStart) return float3(0, 0, 0);

    // Cloud shell clipped to [tStart, tEnd].
    float tC0 = 0.0f, tC1 = 0.0f;
    bool hasCloud = (cloud_enabled >= 0.5f)
                  && RayCloudShell(O, Vn, tC0, tC1);
    if (hasCloud)
    {
        tC0 = max(tC0, tStart);
        tC1 = min(tC1, tEnd);
        if (tC0 >= tC1) hasCloud = false;
    }

    float cosTheta = dot(Vn, L);
    float phR      = PhaseRayleigh(cosTheta);
    float phM      = PhaseMieTwoLobe(cosTheta);

    float3 inscatter     = float3(0, 0, 0);
    float3 transmittance = float3(1, 1, 1);

    if (!hasCloud)
    {
        IntegrateAtmosphereSegment(O, Vn, L, tStart, tEnd, ATMOS_VIEW_STEPS,
                                   cosTheta, phR, phM, sunIrradiance,
                                   DispatchRaysIndex().xy,
                                   inscatter, transmittance);
        transmittanceOut = transmittance;
        return inscatter;
    }

    // Split atmosphere budget between phases 1 and 3 proportionally to length.
    float p1Len         = tC0 - tStart;
    float p3Len         = tEnd - tC1;
    float totalAtmosLen = max(1e-6f, p1Len + p3Len);
    int   N1            = max(2, (int)((float)ATMOS_VIEW_STEPS * p1Len / totalAtmosLen + 0.5f));
    int   N3            = max(2, (int)((float)ATMOS_VIEW_STEPS * p3Len / totalAtmosLen + 0.5f));

    // Phase 1 — observer to cloud entry, atmosphere only.
    IntegrateAtmosphereSegment(O, Vn, L, tStart, tC0, N1,
                               cosTheta, phR, phM, sunIrradiance,
                               DispatchRaysIndex().xy,
                               inscatter, transmittance);

    // Sky+ground probes anchored at cloud midpoint (not camera): a dayside
    // observer looking at nightside clouds otherwise lit them with dayside
    // sky/bounce. probeUp tracks the cloud's local geography.
    float3 cloudMid       = O + Vn * (0.5f * (tC0 + tC1));
    float3 probeUp        = SafeNormalize(cloudMid);
    float3 cloudProbePos  = probeUp * (ATMOS_BOTTOM_RADIUS
                                     + CLOUD_LAYER_TOP_KM
                                     + CLOUD_TOP_VARIATION_KM + 0.5f);
    float3 savedObserver  = g_skyObserverPlanet;
    g_skyObserverPlanet   = cloudProbePos;

    float3 skyAmbientTop     = float3(0, 0, 0);
    float3 skyAmbientHorizon = float3(0, 0, 0);
    if (CLOUD_SKY_AMBIENT > 0.5f)
    {
        float3 horizonDir;
        {
            float3 sunHoriz = L - probeUp * dot(L, probeUp);
            float  shLen2   = dot(sunHoriz, sunHoriz);
            float3 perp;
            if (shLen2 > 1e-4f)
                perp = SafeNormalize(cross(probeUp, sunHoriz));
            else
            {
                float3 fwd = Vn - probeUp * dot(Vn, probeUp);
                perp = (dot(fwd, fwd) > 1e-4f) ? SafeNormalize(fwd) : float3(1, 0, 0);
            }
            horizonDir = SafeNormalize(perp * 0.95f + probeUp * 0.05f);
        }
        float3 trZ; bool hitPZ;
        float3 scatterTop     = IntegrateScattering(probeUp,    L, trZ, hitPZ);
        float3 scatterHorizon = IntegrateScattering(horizonDir, L, trZ, hitPZ);
        skyAmbientTop     = scatterTop     * SKY_INTENSITY * CLOUD_SKY_AMBIENT_SCALE;
        skyAmbientHorizon = scatterHorizon * SKY_INTENSITY * CLOUD_SKY_AMBIENT_SCALE;

        float topLum     = max(Luma(skyAmbientTop), 1e-4f);
        float horizonLum = Luma(skyAmbientHorizon);
        if (horizonLum > topLum) skyAmbientHorizon *= topLum / horizonLum;
    }

    float3 groundIrrad = float3(0, 0, 0);
    if (CLOUD_GROUND_BOUNCE > 0.5f)
    {
        float  sunCosUp   = saturate(dot(L, probeUp));
        float3 sunGroundT = TransmittanceToSun(cloudProbePos, L,
                                               ATMOS_BOTTOM_RADIUS,
                                               ATMOS_TOP_RADIUS);
        groundIrrad = sunIrradiance * sunGroundT * sunCosUp
                    * CLOUD_GROUND_ALBEDO * CLOUD_GROUND_SCALE
                    * (1.0f / PI);

        // Global cover proxy — tracking per-pixel cloud shadow on the
        // ground caused DLSS RR to read the parallax between ground-depth
        // shadow and cloud-depth surface as a separate layer.
        if (cloud_cloudShadowOnSurfaces > 0.5f)
        {
            float coverage = saturate(CLOUD_COVERAGE_BASE);
            groundIrrad   *= (1.0f - 0.85f * coverage);
        }
    }

    g_skyObserverPlanet = savedObserver;

    // Phase 2 — combined atmosphere + cloud march through cloud shell.
    {
        uint2 pixel    = DispatchRaysIndex().xy;
        uint  frame    = (uint)time;
        // seed = cloud body step jitter + RR coin. STAYS ON WHITE NOISE —
        // routing through STBN disrupts the body integration; only the
        // atmospheric cone tap below uses STBN safely.
        uint  seed     = initRandomData(pixel, uint2(0, 0), frame, 71u);

        const float kAtmosShadowConeCosP2 = cos(ATMOS_CLOUD_SHADOW_CONE_DEG * DEG2RAD);

        int   acceptedI = 0;

        float t        = tC0;
        float stepSize = CLOUD_TARGET_STEP_KM
                       * lerp(0.55f, 1.0f, CloudLodT(tC0));

        [loop]
        for (int ts = 0; ts < CLOUD_VIEW_STEPS_MAX; ++ts)
        {
            if (t >= tC1) break;
            if (max(transmittance.r, max(transmittance.g, transmittance.b)) < CLOUD_TR_EPS) break;

            float distFade = 1.0f - smoothstep(CLOUD_FADE_DISTANCE_KM,
                                                CLOUD_RENDER_DISTANCE_KM, t);
            float lodT     = CloudLodT(t);
            float rJit     = RandomFloatSingle(seed);
            float tSample  = t + rJit * stepSize;
            float3 P       = O + Vn * tSample;

            float alt           = max(0.0f, length(P) - ATMOS_BOTTOM_RADIUS);
            MediumSample atmosMed = SampleMedium(alt);

            // Hull-Ex passes (coverage, profile, effTop) to the material call.
            float hCoverage = 0.0f, hProfile = 0.0f, hEffTop = CLOUD_LAYER_TOP_KM;
            float hullRaw   = (distFade > 0.0f)
                            ? CloudProbeHullEx(P, walltime, hCoverage, hProfile, hEffTop)
                            : 0.0f;
            float hull      = hullRaw * distFade;
            bool  hullEmpty = (hull <= CLOUD_EFFECTIVE_ZERO_DENSITY);

            CloudMaterial m = (CloudMaterial)0;
            m.heightFrac    = saturate((alt - CLOUD_LAYER_BOT_KM)
                                       / max(1e-4f, CLOUD_LAYER_TOP_KM - CLOUD_LAYER_BOT_KM));
            float cloudDensity = 0.0f;
            if (!hullEmpty)
            {
                m            = CloudSampleMaterialFromHull(P, walltime, lodT, 0u,
                                                           hCoverage, hProfile, hEffTop);
                cloudDensity = m.density * distFade;
            }
            float cloudSigmaT = cloudDensity * CLOUD_EXTINCTION;

            // Combined extinction: atmos (per-channel) + cloud (scalar).
            float3 sigma_t_total = atmosMed.extinction
                                 + float3(cloudSigmaT, cloudSigmaT, cloudSigmaT);

            // Fine step inside density, coarse when empty.
            float thisStep;
            if (hullEmpty)
            {
                float maxStepAdaptive = min(
                    CLOUD_MAX_STEP_KM * (1.0f + t * CLOUD_EMPTY_STEP_GROWTH_PER_KM),
                    CLOUD_MAX_EMPTY_STEP_KM);
                thisStep = lerp(stepSize, maxStepAdaptive, saturate(lodT));
            }
            else
            {
                thisStep = stepSize;
            }
            thisStep = min(thisStep, tC1 - t);

            float3 segTr = exp(-sigma_t_total * thisStep);

            float3 sunAtmosT = TransmittanceToSun(P, L,
                                                  ATMOS_BOTTOM_RADIUS,
                                                  ATMOS_TOP_RADIUS);

            float3 Pnorm      = SafeNormalize(P);
            float  sunCosZ    = dot(Pnorm, L);
            float  cosHorizon = -sqrt(max(0.0f, 1.0f -
                                          (ATMOS_BOTTOM_RADIUS * ATMOS_BOTTOM_RADIUS)
                                          / dot(P, P)));
            float  earthShadow = SunDiskFractionAboveHorizon(sunCosZ, cosHorizon);

            // Cloud shadow on the air column inside the shell — without
            // it the haze brightens visibly as the ray crosses from Phase
            // 1 into Phase 2. Salt 700u keeps this STBN tap independent.
            float cloudVisAtmos = 1.0f;
            if (cloud_cloudShadowOnSurfaces > 0.5f)
            {
                float4 rCone = CloudRand4(pixel, frame, (uint)ts + 700u);
                float3 LjAtmos = SampleConeAroundDir(L, kAtmosShadowConeCosP2,
                                                     float2(rCone.x, rCone.y));
                cloudVisAtmos = CloudSunVisibilityPlanet(P, LjAtmos);
                cloudVisAtmos = max(cloudVisAtmos, ATMOS_CLOUD_SHADOW_FLOOR);
            }

            float3 atmosScatterPhase = atmosMed.scatterR * phR + atmosMed.scatterM * phM;
            float3 atmosSunIllum     = sunIrradiance * earthShadow * sunAtmosT
                                     * cloudVisAtmos * ATMOS_MULTI_SCATTER_FACTOR;
            float3 atmosRate         = atmosScatterPhase * atmosSunIllum;

            // Cloud rate = radiance × cloud sigma_s (≈ sigma_t for albedo≈1).
            float3 cloudRate = float3(0, 0, 0);
            if (cloudDensity > 0.0f)
            {
                float3 cloudRadiance = CloudComputeLighting(
                    P, Vn, L, m,
                    sunIrradiance, sunAtmosT,
                    skyAmbientTop, skyAmbientHorizon, groundIrrad,
                    Pnorm, cosTheta, earthShadow,
                    pixel, frame, 200u + (uint)acceptedI * 11u);
                cloudRate = cloudRadiance * cloudSigmaT;
                ++acceptedI;
            }

            float3 totalRate = atmosRate + cloudRate;

            float3 scatterInteg;
            scatterInteg.x = (sigma_t_total.x > 1e-10f)
                ? totalRate.x * (1.0f - segTr.x) / sigma_t_total.x : totalRate.x * thisStep;
            scatterInteg.y = (sigma_t_total.y > 1e-10f)
                ? totalRate.y * (1.0f - segTr.y) / sigma_t_total.y : totalRate.y * thisStep;
            scatterInteg.z = (sigma_t_total.z > 1e-10f)
                ? totalRate.z * (1.0f - segTr.z) / sigma_t_total.z : totalRate.z * thisStep;

            inscatter += transmittance * scatterInteg;

            transmittance *= segTr;

            // RR lets distant clouds render through dense foreground; without
            // it tangent rays hard-stop on the first opaque cumulus.
            {
                float trMaxRR = max(transmittance.r,
                                    max(transmittance.g, transmittance.b));
                if (trMaxRR < CLOUD_RR_THRESHOLD)
                {
                    float pRR = trMaxRR * (1.0f / CLOUD_RR_THRESHOLD);
                    if (RandomFloatSingle(seed) > pRR)
                    {
                        transmittance = float3(0, 0, 0);
                        break;
                    }
                    transmittance *= 1.0f / max(pRR, 1e-4f);
                }
            }

            // Grow fine-step stride only when in density.
            if (!hullEmpty)
                stepSize = min(stepSize * CLOUD_STEP_GROWTH, CLOUD_MAX_FINE_STEP_KM);
            t += thisStep;
        }
    }

    // Phase 3 — cloud exit to atmosphere top / planet, atmosphere only.
    if (tC1 < tEnd)
    {
        IntegrateAtmosphereSegment(O, Vn, L, tC1, tEnd, N3,
                                   cosTheta, phR, phM, sunIrradiance,
                                   DispatchRaysIndex().xy,
                                   inscatter, transmittance);
    }

    transmittanceOut = transmittance;
    return inscatter;
}

#endif
