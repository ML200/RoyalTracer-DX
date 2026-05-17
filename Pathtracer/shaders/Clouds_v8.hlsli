#ifndef CLOUDS_V8_HLSLI
#define CLOUDS_V8_HLSLI

/* ============================================================================
   VOLUMETRIC CLOUDS — clean rewrite (Nubis Cubed / Bouthors inspired)
   ============================================================================

   DENSITY
     Perlin-Worley base shape (Schneider 2015 / HZD)
     Value-noise coverage modulation (weather map proxy)
     Worley FBM erosion at edges + mid-frequency body detail
     Heart-shaped altitude profile within the cloud layer

   LIGHTING (Nubis Evolved 2022 — Schneider)
     Sun ray Beer-Lambert transmittance (no fake multipliers)
     Direct: dual-lobe Henyey-Greenstein matching Mie shape
     Multi-scatter: single Nubis-style term
         MS = K * sqrt(Tdir) * heightBias * SECONDARY_STRENGTH * STRENGTH * phaseMS
       MS amplitude FOLLOWS direct illumination via sqrt(Tdir), so cloud
       tops (high Tdir) glow white even when the direct back lobe is
       small (e.g., overhead-sun + downward view). This is critical for
       orbital views — without it, cloud tops end up darker than bases
       because the only thing they have is sky ambient. The earlier
       Bouthors envelope (MS growing with sunOD) inverted this and gave
       a "sun-from-below" appearance.
     Sky ambient: occluded by column-density above sample
       Probe origin relocated to cloud altitude — orbital cameras would
       otherwise integrate through near-zero atmosphere upward, producing
       skyAmbientTop ≈ 0 and the same inversion.
     Ground bounce: attenuated by sample height in layer

   PIPELINE
     EvaluateClouds:       full quality adaptive march, primary rays
     EvaluateCloudsCheap:  fixed 10-step march, bounce rays
     CloudSunVisibility:   surface shadow lookup (cloud shadows on ground)
   ============================================================================ */

#ifndef ENABLE_CLOUDS
#define ENABLE_CLOUDS 1
#endif

#ifndef CLOUD_STBN_AVAILABLE
#define CLOUD_STBN_AVAILABLE 0
#endif

#if CLOUD_STBN_AVAILABLE
Texture2DArray<float4> g_cloudSTBN : register(t41);
#endif

//------------------------------------------------------------------------------
// CONSTANTS
//------------------------------------------------------------------------------

// Layer geometry
#ifndef CLOUD_LAYER_BOT_KM
#define CLOUD_LAYER_BOT_KM      1.5f
#endif
#ifndef CLOUD_LAYER_TOP_KM
#define CLOUD_LAYER_TOP_KM      3.5f
#endif
#ifndef CLOUD_HORIZON_FADE_KM
#define CLOUD_HORIZON_FADE_KM   2.0f
#endif

// Shape / coverage
#ifndef CLOUD_COVERAGE_BASE
#define CLOUD_COVERAGE_BASE     0.45f
#endif
#ifndef CLOUD_COVERAGE_VAR
#define CLOUD_COVERAGE_VAR      0.30f
#endif
#ifndef CLOUD_COVERAGE_FREQ
#define CLOUD_COVERAGE_FREQ     0.025f
#endif
#ifndef CLOUD_BASE_FREQ
#define CLOUD_BASE_FREQ         0.45f
#endif
#ifndef CLOUD_HF_FREQ
#define CLOUD_HF_FREQ           3.5f
#endif
#ifndef CLOUD_HF_AMOUNT
#define CLOUD_HF_AMOUNT         0.70f
#endif
#ifndef CLOUD_COVMOD_FILTER_WIDTH
#define CLOUD_COVMOD_FILTER_WIDTH 0.3f
#endif
#ifndef CLOUD_WARP_AMP_KM
#define CLOUD_WARP_AMP_KM       0.4f
#endif

// Wind / weather offset
#ifndef CLOUD_WIND_X
#define CLOUD_WIND_X            0.04f
#endif
#ifndef CLOUD_WIND_Z
#define CLOUD_WIND_Z            0.015f
#endif
#ifndef CLOUD_WEATHER_OFFSET_X
#define CLOUD_WEATHER_OFFSET_X  0.0f
#endif
#ifndef CLOUD_WEATHER_OFFSET_Z
#define CLOUD_WEATHER_OFFSET_Z  0.0f
#endif

// Extinction / albedo
#ifndef CLOUD_EXTINCTION
#define CLOUD_EXTINCTION        9.0f
#endif
#ifndef CLOUD_ALBEDO
#define CLOUD_ALBEDO            float3(0.995f, 0.995f, 0.995f)
#endif

// Phase function — silver-lining artist controls
#ifndef CLOUD_SILVER_INTENSITY
#define CLOUD_SILVER_INTENSITY  0.35f
#endif
#ifndef CLOUD_SILVER_SPREAD
#define CLOUD_SILVER_SPREAD     0.08f
#endif

// Multi-scatter
#ifndef CLOUD_SECONDARY_STRENGTH
#define CLOUD_SECONDARY_STRENGTH 0.45f
#endif
#ifndef CLOUD_SECONDARY_G
#define CLOUD_SECONDARY_G       0.20f
#endif

// MS strength — global multiplier on the Nubis-style MS term. 1.5 gives
// the soft fluffy cumulus look. Lower for thin/wispy clouds, higher for
// hyper-stylized marshmallow puffs.
#ifndef CLOUD_MS_STRENGTH
#define CLOUD_MS_STRENGTH       1.5f
#endif

// MS height floor — minimum MS contribution at cloud base (h=0). Without
// a floor bases turn black; 0.40 maps to "shaded white" which is what
// real cumulus bottoms look like.
#ifndef CLOUD_MS_HEIGHT_FLOOR
#define CLOUD_MS_HEIGHT_FLOOR   0.40f
#endif

// Sky ambient
#ifndef CLOUD_AMBIENT_INTENSITY
#define CLOUD_AMBIENT_INTENSITY 0.85f
#endif
#ifndef CLOUD_AMBIENT_AO_SCALE
#define CLOUD_AMBIENT_AO_SCALE  0.35f
#endif
#ifndef CLOUD_AMBIENT_OD_MAX
#define CLOUD_AMBIENT_OD_MAX    2.0f
#endif

// Sun shadow march (soft sun via cone tracing)
#ifndef CLOUD_SUN_TAU_MULT
#define CLOUD_SUN_TAU_MULT      1.0f
#endif
#ifndef CLOUD_SHADOW_CONE_DEG
#define CLOUD_SHADOW_CONE_DEG    0.0f
#endif
#ifndef CLOUD_SHADOW_CONE_SAMPLES
#define CLOUD_SHADOW_CONE_SAMPLES 1.0f
#endif
#define CLOUD_SHADOW_CONE_SAMPLES_MAX 5

// Ambient probe
#ifndef CLOUD_AMBIENT_STEPS
#define CLOUD_AMBIENT_STEPS      3
#endif
#define CLOUD_AMBIENT_STEPS_MAX  6

// Sky / ground toggles
#ifndef CLOUD_SKY_AMBIENT
#define CLOUD_SKY_AMBIENT       1.0f
#endif
#ifndef CLOUD_GROUND_BOUNCE
#define CLOUD_GROUND_BOUNCE     1.0f
#endif
#ifndef CLOUD_GROUND_ALBEDO
#define CLOUD_GROUND_ALBEDO     0.20f
#endif
#ifndef CLOUD_SKY_AMBIENT_SCALE
#define CLOUD_SKY_AMBIENT_SCALE 1.0f
#endif
#ifndef CLOUD_GROUND_SCALE
#define CLOUD_GROUND_SCALE      1.0f
#endif

// Integration
#ifndef CLOUD_VIEW_STEPS
#define CLOUD_VIEW_STEPS        32
#endif
#ifndef CLOUD_SHADOW_STEPS
#define CLOUD_SHADOW_STEPS      4
#endif
#ifndef CLOUD_TR_EPS
#define CLOUD_TR_EPS            0.005f
#endif
#ifndef CLOUD_RR_THRESHOLD
#define CLOUD_RR_THRESHOLD      0.10f
#endif
#ifndef CLOUD_LOD_NEAR_KM
#define CLOUD_LOD_NEAR_KM       8.0f
#endif
#ifndef CLOUD_LOD_FAR_KM
#define CLOUD_LOD_FAR_KM        30.0f
#endif
#ifndef CLOUD_FADE_DISTANCE_KM
#define CLOUD_FADE_DISTANCE_KM   2500.0f
#endif
#ifndef CLOUD_RENDER_DISTANCE_KM
#define CLOUD_RENDER_DISTANCE_KM 3000.0f
#endif
#ifndef CLOUD_MAX_STEP_KM
#define CLOUD_MAX_STEP_KM        0.5f
#endif
#ifndef CLOUD_STEP_GROWTH
#define CLOUD_STEP_GROWTH        1.02f
#endif
#ifndef CLOUD_EFFECTIVE_ZERO_DENSITY
#define CLOUD_EFFECTIVE_ZERO_DENSITY 1e-3f
#endif
#ifndef CLOUD_TARGET_STEP_KM
#define CLOUD_TARGET_STEP_KM     0.6f
#endif
#ifndef CLOUD_VIEW_STEPS_MAX
#define CLOUD_VIEW_STEPS_MAX     256
#endif
#ifndef CLOUD_MAX_EMPTY_STEP_KM
#define CLOUD_MAX_EMPTY_STEP_KM  50.0f
#endif
#ifndef CLOUD_EMPTY_STEP_GROWTH_PER_KM
#define CLOUD_EMPTY_STEP_GROWTH_PER_KM   0.1f
#endif
#ifndef CLOUD_MAX_FINE_STEP_KM
#define CLOUD_MAX_FINE_STEP_KM   2.0f
#endif
#ifndef CLOUD_HAZE_STRENGTH
#define CLOUD_HAZE_STRENGTH      1.0f
#endif

// Cheap variant (bounce rays)
#ifndef CLOUD_CHEAP_STEPS
#define CLOUD_CHEAP_STEPS 10
#endif
#ifndef CLOUD_CHEAP_MAX_LEN_KM
#define CLOUD_CHEAP_MAX_LEN_KM 60.0f
#endif

//------------------------------------------------------------------------------
// NOISE PRIMITIVES (hash, Worley, Perlin, Perlin-Worley)
//------------------------------------------------------------------------------

inline float CloudLodT(float distKm)
{
    return saturate((distKm - CLOUD_LOD_NEAR_KM)
                  / max(1e-4f, CLOUD_LOD_FAR_KM - CLOUD_LOD_NEAR_KM));
}

inline uint CloudHash3D(int3 p)
{
    uint h = uint(p.x) * 73856093u
           ^ uint(p.y) * 19349663u
           ^ uint(p.z) * 83492791u;
    h ^= h >> 16; h *= 0x7feb352du;
    h ^= h >> 15; h *= 0x846ca68bu;
    h ^= h >> 16;
    return h;
}

inline float CloudHashFloat(int3 p)
{
    return float(CloudHash3D(p)) * (1.0f / 4294967296.0f);
}

inline float3 CloudHashVec3(int3 p)
{
    uint h = CloudHash3D(p);
    return float3((h        & 1023u),
                  ((h >> 10) & 1023u),
                  ((h >> 20) & 1023u)) * (1.0f / 1023.0f);
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

float CloudValueNoise(float3 p)
{
    int3   i = int3(floor(p));
    float3 f = frac(p);
    float3 u = f * f * f * (f * (f * 6.0f - 15.0f) + 10.0f);

    float c000 = CloudHashFloat(i + int3(0, 0, 0));
    float c100 = CloudHashFloat(i + int3(1, 0, 0));
    float c010 = CloudHashFloat(i + int3(0, 1, 0));
    float c110 = CloudHashFloat(i + int3(1, 1, 0));
    float c001 = CloudHashFloat(i + int3(0, 0, 1));
    float c101 = CloudHashFloat(i + int3(1, 0, 1));
    float c011 = CloudHashFloat(i + int3(0, 1, 1));
    float c111 = CloudHashFloat(i + int3(1, 1, 1));

    float x00 = lerp(c000, c100, u.x);
    float x10 = lerp(c010, c110, u.x);
    float x01 = lerp(c001, c101, u.x);
    float x11 = lerp(c011, c111, u.x);
    float y0  = lerp(x00,  x10,  u.y);
    float y1  = lerp(x01,  x11,  u.y);
    return lerp(y0, y1, u.z);
}

float CloudWorley(float3 p)
{
    int3   i = int3(floor(p));
    float3 f = frac(p);
    float  minD2 = 1.0e10f;

    [unroll] for (int x = -1; x <= 1; ++x)
    [unroll] for (int y = -1; y <= 1; ++y)
    [unroll] for (int z = -1; z <= 1; ++z)
    {
        int3   cell    = i + int3(x, y, z);
        float3 feature = float3(x, y, z) + CloudHashVec3(cell) - f;
        float  d2      = dot(feature, feature);
        minD2          = min(minD2, d2);
    }
    return saturate(sqrt(minD2) * 1.15f);
}

float CloudWorleyFBM(float3 p, uint quality)
{
    float wInv0 = 1.0f - CloudWorley(p);
    if (quality >= 2u) return wInv0;

    float wInv1 = 1.0f - CloudWorley(p * 2.7f + float3(13.31f, -7.13f, 19.77f));
    if (quality >= 1u) return wInv0 * 0.62f + wInv1 * 0.38f;

    float wInv2 = 1.0f - CloudWorley(p * 7.0f + float3(-3.47f, 21.97f,  5.13f));
    return wInv0 * 0.50f + wInv1 * 0.32f + wInv2 * 0.18f;
}

inline float3 CloudPerlinGrad(int3 p)
{
    uint h = CloudHash3D(p) & 15u;
    float3 g;
    g.x = (h <  8u) ? 1.0f : -1.0f;
    g.y = (h <  4u || h == 12u || h == 14u) ? 1.0f : -1.0f;
    g.z = (h <  2u || h == 12u || h == 13u) ?  0.0f : 1.0f;
    return g;
}

float CloudPerlin(float3 p)
{
    int3   i = int3(floor(p));
    float3 f = frac(p);
    float3 u = f * f * f * (f * (f * 6.0f - 15.0f) + 10.0f);

    float3 g000 = CloudPerlinGrad(i + int3(0, 0, 0));
    float3 g100 = CloudPerlinGrad(i + int3(1, 0, 0));
    float3 g010 = CloudPerlinGrad(i + int3(0, 1, 0));
    float3 g110 = CloudPerlinGrad(i + int3(1, 1, 0));
    float3 g001 = CloudPerlinGrad(i + int3(0, 0, 1));
    float3 g101 = CloudPerlinGrad(i + int3(1, 0, 1));
    float3 g011 = CloudPerlinGrad(i + int3(0, 1, 1));
    float3 g111 = CloudPerlinGrad(i + int3(1, 1, 1));

    float c000 = dot(g000, f - float3(0, 0, 0));
    float c100 = dot(g100, f - float3(1, 0, 0));
    float c010 = dot(g010, f - float3(0, 1, 0));
    float c110 = dot(g110, f - float3(1, 1, 0));
    float c001 = dot(g001, f - float3(0, 0, 1));
    float c101 = dot(g101, f - float3(1, 0, 1));
    float c011 = dot(g011, f - float3(0, 1, 1));
    float c111 = dot(g111, f - float3(1, 1, 1));

    float x00 = lerp(c000, c100, u.x);
    float x10 = lerp(c010, c110, u.x);
    float x01 = lerp(c001, c101, u.x);
    float x11 = lerp(c011, c111, u.x);
    float y0  = lerp(x00,  x10,  u.y);
    float y1  = lerp(x01,  x11,  u.y);
    return lerp(y0, y1, u.z);
}

float CloudPerlinFBM(float3 p)
{
    float v = CloudPerlin(p) * 0.5f
            + CloudPerlin(p * 2.13f + float3(5.7f, -2.3f, 9.1f)) * 0.25f;
    return saturate(v * (1.0f / 1.5f) + 0.5f);
}

float CloudPerlinWorley(float3 p)
{
    float perlin = CloudPerlinFBM(p);
    float worleyInv = CloudWorleyFBM(p, 0u);
    return saturate((perlin - (1.0f - worleyInv)) / max(worleyInv, 1e-3f));
}

//------------------------------------------------------------------------------
// DENSITY MODEL
//------------------------------------------------------------------------------
// Heart-shaped altitude profile within the cloud layer. Bottom-up ramp for
// flat cumulus bases, taper to zero at the top for anvil-like crowns.

inline float CloudAltitudeProfile(float altKm)
{
    float h = (altKm - CLOUD_LAYER_BOT_KM)
            / max(1e-4f, CLOUD_LAYER_TOP_KM - CLOUD_LAYER_BOT_KM);
    if (h <= 0.0f || h >= 1.0f) return 0.0f;

    float bottomRamp = smoothstep(0.00f, 0.18f, h);
    float topCap     = 1.0f - smoothstep(0.55f, 1.00f, h);
    return bottomRamp * topCap;
}

inline float coverageModulation(float coverage, float detail, float filterWidth)
{
    float f = 1.0f - coverage;
    float modDetail = detail * (1.0f - filterWidth) + filterWidth;
    return saturate((modDetail - f) / max(filterWidth, 1e-3f));
}

// Cheap upper-bound estimator. Used by empty-space skipping and by the
// ambient column probe — both want a fast "is there cloud here?" check
// without paying for the full noise pyramid.
float CloudProbeHull(float3 P, float timeSec)
{
    float r = length(P);
    float alt = r - ATMOS_BOTTOM_RADIUS;
    float profile = CloudAltitudeProfile(alt);
    if (profile <= 0.0f) return 0.0f;

    float3 wind = float3(CLOUD_WIND_X, 0.0f, CLOUD_WIND_Z) * timeSec;
    float3 q    = P + wind;
    float3 covQ = float3(q.x + CLOUD_WEATHER_OFFSET_X,
                         0.0f,
                         q.z + CLOUD_WEATHER_OFFSET_Z) * CLOUD_COVERAGE_FREQ;
    float  covNoise = CloudValueNoise(covQ);
    float  coverage = saturate(CLOUD_COVERAGE_BASE
                              + (covNoise - 0.5f) * CLOUD_COVERAGE_VAR);
    return coverage * profile;
}

struct CloudMaterial
{
    float density;     // final density after coverage + erosion [0..1]
    float profile;     // macro shape before erosion [0..1]
    float heightFrac;  // position within the layer [0..1]
};

CloudMaterial CloudSampleMaterial(float3 P, float timeSec, float lodT, uint quality)
{
    CloudMaterial m;
    m.density    = 0.0f;
    m.profile    = 0.0f;
    m.heightFrac = 0.0f;

    float r   = length(P);
    float alt = r - ATMOS_BOTTOM_RADIUS;
    float h   = (alt - CLOUD_LAYER_BOT_KM)
              / max(1e-4f, CLOUD_LAYER_TOP_KM - CLOUD_LAYER_BOT_KM);
    m.heightFrac = saturate(h);

    if (h <= 0.0f || h >= 1.0f) return m;

    float profile = CloudAltitudeProfile(alt);
    if (profile <= 0.0f) return m;

    float3 wind = float3(CLOUD_WIND_X, 0.0f, CLOUD_WIND_Z) * timeSec;
    float3 q    = P + wind;

    float3 covQ      = float3(q.x + CLOUD_WEATHER_OFFSET_X,
                              0.0f,
                              q.z + CLOUD_WEATHER_OFFSET_Z) * CLOUD_COVERAGE_FREQ;
    float  covNoise  = CloudValueNoise(covQ);
    float  coverage  = saturate(CLOUD_COVERAGE_BASE
                               + (covNoise - 0.5f) * CLOUD_COVERAGE_VAR);
    if (coverage <= 0.0f) return m;

    // Low-frequency domain warp pushes the base shape around so cumulus
    // don't look like a stamped grid pattern. Skipped at distance.
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

    // High-frequency erosion — eats the cloud at edges so silhouettes look
    // wispy rather than chunky. Tapered with LOD because the detail can't
    // resolve at long range anyway.
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

    // Near-field detail: subtle mid-frequency Worley modulation at close
    // range adds cauliflower bumps without changing the bulk density.
    // Previously this block also ran an aggressive wisp-erosion (`hhf`)
    // and a high-magnitude `closeAmount` modulation that subtracted
    // density as the camera approached — visible as clouds "losing
    // substance" up close. Toned way down; the base Perlin-Worley plus
    // HF erosion already supplies enough silhouette character.
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

    // Soft density curve — lifts mid-densities slightly so the body fills
    // out rather than feathering everywhere.
    d = pow(saturate(d), lerp(0.85f, 1.0f, d));

    m.density = d;
    return m;
}

float CloudDensity(float3 P, float timeSec, float lodT, uint quality)
{
    return CloudSampleMaterial(P, timeSec, lodT, quality).density;
}

//------------------------------------------------------------------------------
// SHELL INTERSECTION
//------------------------------------------------------------------------------

bool RayCloudShell(float3 ro, float3 rd, out float tNear, out float tFar)
{
    tNear = 0.0f;
    tFar  = 0.0f;

    float Rb = ATMOS_BOTTOM_RADIUS;
    float Rt = ATMOS_BOTTOM_RADIUS + CLOUD_LAYER_TOP_KM;
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

inline float CloudHorizonFade(float3 ro, float3 rd)
{
    float Rt = ATMOS_BOTTOM_RADIUS + CLOUD_LAYER_TOP_KM;
    float r  = length(ro);
    if (r <= Rt) return 1.0f;

    float d2 = dot(ro, ro) - dot(ro, rd) * dot(ro, rd);
    float d  = sqrt(max(d2, 0.0f));
    float Rm = ATMOS_BOTTOM_RADIUS + CLOUD_LAYER_BOT_KM;
    return 1.0f - smoothstep(Rm, Rm + CLOUD_HORIZON_FADE_KM, d);
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

//------------------------------------------------------------------------------
// PHASE FUNCTIONS
//------------------------------------------------------------------------------

inline float CloudHG(float cosT, float g)
{
    float g2 = g * g;
    float denom = 1.0f + g2 - 2.0f * g * cosT;
    return (1.0f / (4.0f * PI)) * (1.0f - g2)
         / max(1e-4f, denom * sqrt(max(1e-4f, denom)));
}

// Direct scattering: dual-lobe HG matching Mie shape for water droplets.
// Forward 0.85 lobe = bright sun-side halo; back -0.5 lobe = soft glow
// opposite the sun. CLOUD_SILVER_INTENSITY shifts more weight into a
// sharper forward lobe for tighter silver linings.
inline float CloudPhaseDirect(float cosT)
{
    float sIntens = saturate(CLOUD_SILVER_INTENSITY);
    float gFwd = lerp(0.85f, 0.99f - CLOUD_SILVER_SPREAD, sIntens);
    float wFwd = lerp(0.85f, 0.95f, sIntens);

    float fwd = CloudHG(cosT, gFwd);
    float bwd = CloudHG(cosT, -0.50f);
    return wFwd * fwd + (1.0f - wFwd) * bwd;
}

// Multi-scatter phase: nearly isotropic with a slight forward bias.
// After many bounces the light has effectively lost its initial direction
// so the phase tends toward isotropic. The slight forward bias keeps the
// sun-facing side a touch brighter than the back, which reads as a soft
// directional fill.
inline float CloudPhaseMS(float cosT)
{
    const float kIso = 0.07957747f;  // 1 / (4 * pi)
    return lerp(kIso, CloudHG(cosT, CLOUD_SECONDARY_G), 0.4f);
}

//------------------------------------------------------------------------------
// OPTICAL DEPTH (sun ray + arbitrary direction)
//------------------------------------------------------------------------------

// Quality=0 here is required: the view march uses CloudPerlinWorley
// (quality=0) for its base shape. If the shadow march uses quality=1
// (CloudWorleyFBM, a different noise function entirely) it computes
// shadows against a *different shaped cloud*, which surfaces as phantom
// dark patches that don't correspond to anything visible. lodT=1 keeps
// the shadow detail simplified — same macro shape, no high-frequency
// near-field modulation, fast.
float CloudOpticalDepthToSun(float3 P, float3 L, float timeSec)
{
    const float SAMPLE_DIST_KM[6] = { 0.2f, 0.6f, 1.5f, 3.5f, 7.0f, 12.0f };
    const float SAMPLE_SEG_KM[6]  = { 0.2f, 0.4f, 1.1f, 2.0f, 3.5f,  5.0f };
    const float TAU_EARLYOUT = 64.0f;

    float tau = 0.0f;
    [loop]
    for (int i = 0; i < 6; ++i)
    {
        float3 Q = P + L * SAMPLE_DIST_KM[i];
        float  d = CloudDensity(Q, timeSec, 1.0f, 0u);
        tau += d * CLOUD_EXTINCTION * SAMPLE_SEG_KM[i];
        if (tau >= TAU_EARLYOUT) break;
    }
    return tau * CLOUD_SUN_TAU_MULT;
}

float CloudOpticalDepthAlongRay(float3 P, float3 D, float maxLenKm, float timeSec)
{
    float tNear, tFar;
    if (!RayCloudShell(P, D, tNear, tFar)) return 0.0f;
    tFar = min(tFar, tNear + maxLenKm);
    if (tFar <= tNear) return 0.0f;

    const int   N  = CLOUD_SHADOW_STEPS;
    const float ds = (tFar - tNear) / (float)N;
    float tau = 0.0f;
    [loop]
    for (int i = 0; i < N; ++i)
    {
        float  ti = tNear + ((float)i + 0.5f) * ds;
        float3 Q  = P + D * ti;
        tau += CloudDensity(Q, timeSec, 1.0f, 0u) * CLOUD_EXTINCTION * ds;
    }
    return tau;
}

float CloudSunVisibility(float3 surfacePosWorld, float3 sunDirWS)
{
#if !ENABLE_CLOUDS
    return 1.0f;
#else
    if (cloud_enabled < 0.5f) return 1.0f;

    float3 P = WorldToPlanet(surfacePosWorld);
    float  r = length(P);
    if (r - ATMOS_BOTTOM_RADIUS > CLOUD_LAYER_TOP_KM + 1e-3f) return 1.0f;

    float tau = CloudOpticalDepthAlongRay(P, SafeNormalize(sunDirWS), 50.0f, walltime);
    return exp(-tau);
#endif
}

//------------------------------------------------------------------------------
// AMBIENT COLUMN PROBE
//------------------------------------------------------------------------------
// Trapezoidal integration of CloudProbeHull along `upDir` at exponentially-
// spaced altitudes. Approximates Nubis Cubed's baked 32^3 column-density
// voxel grid. Cheap because ProbeHull skips the noise FBM.

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

//------------------------------------------------------------------------------
// CLOUD LIGHTING — Direct + Bouthors MS + Sky ambient + Ground bounce
//------------------------------------------------------------------------------
// Returns the in-scattered radiance at a single sample, ready to be folded
// into the volume integration by the caller as sigma_t * L * segIn.

float3 CloudComputeLighting(float3 P, float3 V, float3 L, CloudMaterial m,
                            float3 sunRad, float3 sunAtmos,
                            float3 skyAmbientTop, float3 skyAmbientHorizon,
                            float3 groundIrrad,
                            float3 localUp,
                            float cosTheta,
                            float earthShadow,
                            uint2 pixel, uint frame, uint tap)
{
    //-- Sun optical depth (optionally cone-traced for soft shadows) --
    int Kshadow = clamp((int)CLOUD_SHADOW_CONE_SAMPLES, 1, CLOUD_SHADOW_CONE_SAMPLES_MAX);

    float sunOD;
    if (Kshadow == 1 && CLOUD_SHADOW_CONE_DEG <= 0.0f)
    {
        sunOD = CloudOpticalDepthToSun(P, L, walltime);
    }
    else
    {
        float cosConeMax = cos(CLOUD_SHADOW_CONE_DEG * DEG2RAD);
        sunOD = 0.0f;
        [loop]
        for (int k = 0; k < CLOUD_SHADOW_CONE_SAMPLES_MAX; ++k)
        {
            if (k >= Kshadow) break;
            float4 rk = CloudRand4(pixel, frame, tap + (uint)k);
            float3 Lk = SampleConeAroundDir(L, cosConeMax, rk.xy);
            sunOD += CloudOpticalDepthToSun(P, Lk, walltime);
        }
        sunOD /= (float)Kshadow;
    }

    float h = m.heightFrac;
    float3 K = sunRad * sunAtmos * earthShadow * CLOUD_ALBEDO;

    //-- Direct scattering --
    // Beer-Lambert sun transmittance times dual-lobe HG phase. Dominates
    // at cloud surfaces and on the sun-side rim; dies off in cloud cores
    // where multi-scatter takes over.
    float Tdir   = exp(-sunOD);
    float3 direct = K * Tdir * CloudPhaseDirect(cosTheta);

    //-- Multi-scatter (Nubis Evolved formulation) --
    // MS amplitude follows the DIRECT illumination, not opposes it.
    // sqrt(Tdir) is a softer-than-Beer attenuation: MS reaches further
    // into the cloud than direct (its effective extinction is halved),
    // but still dies off in the deep core where no light reaches at all.
    //
    // For our orbital view (looking down at cloud tops with sun above)
    // this is what produces the bright white cumulus caps: the back lobe
    // of the phase function gives weak direct, but MS — proportional to
    // sqrt(Tdir) which is near 1 at the top — fills in dramatically.
    //
    // heightBias floors at FLOOR so bases retain some MS even at h=0.
    // SECONDARY_STRENGTH is the artist amplitude (0..1 maps to 0.5..1.5x).
    // CLOUD_MS_STRENGTH is the per-scene tuning constant.
    float depthFactor = pow(max(Tdir, 1e-6f), 0.5f);
    float heightBias  = lerp(CLOUD_MS_HEIGHT_FLOOR, 1.0f,
                             pow(saturate(h), 0.5f));
    float msAmount    = depthFactor * heightBias
                      * (0.5f + CLOUD_SECONDARY_STRENGTH)
                      * CLOUD_MS_STRENGTH;
    float3 ms = K * msAmount * CloudPhaseMS(cosTheta);

    //-- Sky ambient --
    // Column density above the sample, attenuated through the cloud
    // material between sky and sample. Bounded by AMBIENT_OD_MAX so
    // pathologically dense overcast bodies still retain a visible floor.
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
            ambient += groundIrrad * ambExp * (1.0f - h);
        }
    }

    return direct + ms + ambient;
}

//------------------------------------------------------------------------------
// CHEAP VARIANT — bounce rays
//------------------------------------------------------------------------------
// Same lighting model as the full path, simplified for bounce-ray cost.
// Fixed step count, no shadow cone, no ambient probe. Used by indirect
// rays that already have lots of variance from other sources.

float3 EvaluateCloudsCheap(float3 V, float3 sunDir, float3 sunIrradiance,
                           out float3 cloudTrOut)
{
    cloudTrOut = float3(1, 1, 1);

#if !ENABLE_CLOUDS
    return float3(0, 0, 0);
#else
    if (cloud_enabled < 0.5f) return float3(0, 0, 0);

    float3 O = g_skyObserverPlanet;
    float3 L = SafeNormalize(sunDir);

    float tNear, tFar;
    if (!RayCloudShell(O, V, tNear, tFar)) return float3(0, 0, 0);

    float limbFade = CloudHorizonFade(O, V);
    if (limbFade <= 0.0f) return float3(0, 0, 0);

    tFar = min(tFar, tNear + CLOUD_CHEAP_MAX_LEN_KM);
    if (tFar <= tNear) return float3(0, 0, 0);

    const int   N        = CLOUD_CHEAP_STEPS;
    const float ds       = (tFar - tNear) / (float)N;
    const float cosTheta = dot(V, L);
    const float phaseDir = CloudPhaseDirect(cosTheta);
    const float phaseMS  = CloudPhaseMS   (cosTheta);

    float3 PMid = O + V * (0.5f * (tNear + tFar));
    float3 sunAtmos = TransmittanceToSun(PMid, L,
                                         ATMOS_BOTTOM_RADIUS,
                                         ATMOS_TOP_RADIUS);

    uint2 px   = DispatchRaysIndex().xy;
    uint  seed = initRandomData(px, uint2(0, 0), (uint)time, 73u);

    float3 L_acc   = float3(0, 0, 0);
    float3 trCloud = float3(1, 1, 1);

    [loop]
    for (int i = 0; i < N; ++i)
    {
        float  rJit = RandomFloatSingle(seed);
        float  ti   = tNear + ((float)i + rJit) * ds;
        float3 P    = O + V * ti;
        float  lodT = CloudLodT(ti);

        CloudMaterial m = CloudSampleMaterial(P, walltime, lodT, 0u);
        if (m.density <= CLOUD_EFFECTIVE_ZERO_DENSITY) continue;

        float sigma_t     = m.density * CLOUD_EXTINCTION;
        float sunOD       = CloudOpticalDepthToSun(P, L, walltime);
        float Tdir        = exp(-sunOD);
        float earthShadow = CloudEarthShadowFactor(P, L);

        float3 K = sunIrradiance * sunAtmos * earthShadow * CLOUD_ALBEDO;

        // Direct
        float3 inscatter = K * Tdir * phaseDir;

        // MS — Nubis Evolved: amplitude follows direct via sqrt(Tdir),
        // so cloud tops (high Tdir) glow even when the back lobe of the
        // direct phase is small. Same formula as the full path.
        float depthFactor = pow(max(Tdir, 1e-6f), 0.5f);
        float heightBias  = lerp(CLOUD_MS_HEIGHT_FLOOR, 1.0f,
                                 pow(saturate(m.heightFrac), 0.5f));
        float msAmount    = depthFactor * heightBias
                          * (0.5f + CLOUD_SECONDARY_STRENGTH)
                          * CLOUD_MS_STRENGTH;
        inscatter += K * msAmount * phaseMS;

        float segTr = exp(-sigma_t * ds);
        float segIn = (1.0f - segTr) / max(sigma_t, 1e-6f);
        L_acc   += trCloud * (sigma_t * inscatter) * segIn;
        trCloud *= segTr;

        if (max(trCloud.r, max(trCloud.g, trCloud.b)) < CLOUD_TR_EPS)
        {
            trCloud = float3(0, 0, 0);
            break;
        }
    }

    L_acc   *= limbFade;
    trCloud  = lerp(float3(1, 1, 1), trCloud, limbFade);

    cloudTrOut = trCloud;
    return L_acc;
#endif
}

//------------------------------------------------------------------------------
// FULL PRIMARY MARCH
//------------------------------------------------------------------------------

float3 EvaluateClouds(float3 V, float3 sunDir, float3 sunIrradiance,
                      out float3 cloudTrOut)
{
    cloudTrOut = float3(1, 1, 1);

#if !ENABLE_CLOUDS
    return float3(0, 0, 0);
#else
    if (cloud_enabled < 0.5f) return float3(0, 0, 0);

    float3 O = g_skyObserverPlanet;
    float3 L = SafeNormalize(sunDir);

    float tNear, tFar;
    if (!RayCloudShell(O, V, tNear, tFar)) return float3(0, 0, 0);

    float limbFade = CloudHorizonFade(O, V);
    if (limbFade <= 0.0f) return float3(0, 0, 0);

    tFar = min(tFar, tNear + CLOUD_RENDER_DISTANCE_KM);
    if (tFar <= tNear) return float3(0, 0, 0);

    //-- Aerial perspective in front of cloud --
    float3 aerialIn;
    float3 aerialTr;
    {
        aerialIn = ComputeAerialPerspective(V, L, tNear, aerialTr);
        aerialTr = lerp(float3(1, 1, 1), aerialTr, CLOUD_HAZE_STRENGTH);
    }

    float3 obsUp = SafeNormalize(O);

    //-- Cloud-altitude probe for sky / atmosphere lookups --
    //IntegrateScattering and TransmittanceToSun use g_skyObserverPlanet as
    //their integration start point. From an orbital observer that point
    //is in space — looking straight up integrates through *no* remaining
    //atmosphere, so skyAmbientTop returns near zero. Meanwhile the
    //horizon ray still grazes through 80+ km of air and returns bright.
    //The result inverts cloud lighting: tops (which lerp toward
    //skyAmbientTop) go dark, bases (which lerp toward skyAmbientHorizon)
    //go bright. Temporarily relocate the sky-probe origin to a point at
    //the cloud-shell top in the observer's radial direction so the sky
    //lookups see the atmosphere *the cloud sees*, not the atmosphere the
    //orbital camera sees.
    float3 cloudProbePos =
        obsUp * (ATMOS_BOTTOM_RADIUS + CLOUD_LAYER_TOP_KM + 0.5f);
    float3 savedObserverPlanet = g_skyObserverPlanet;
    g_skyObserverPlanet = cloudProbePos;

    //-- Per-pass sky ambient: zenith and a sun-perpendicular horizon ray --
    float3 skyAmbientTop = float3(0, 0, 0);
    float3 skyAmbientHorizon = float3(0, 0, 0);
    if (CLOUD_SKY_AMBIENT > 0.5f)
    {
        float3 horizonDir;
        {
            float3 sunHoriz = L - obsUp * dot(L, obsUp);
            float  shLen2   = dot(sunHoriz, sunHoriz);
            float3 perp;
            if (shLen2 > 1e-4f)
            {
                perp = SafeNormalize(cross(obsUp, sunHoriz));
            }
            else
            {
                float3 fwd = V - obsUp * dot(V, obsUp);
                perp = (dot(fwd, fwd) > 1e-4f) ? SafeNormalize(fwd)
                                               : float3(1, 0, 0);
            }
            horizonDir = SafeNormalize(perp * 0.95f + obsUp * 0.05f);
        }
        float3 trZ; bool hitPZ;
        float3 scatterTop     = IntegrateScattering(obsUp,      L, trZ, hitPZ);
        float3 scatterHorizon = IntegrateScattering(horizonDir, L, trZ, hitPZ);
        skyAmbientTop     = scatterTop     * SKY_INTENSITY * CLOUD_SKY_AMBIENT_SCALE;
        skyAmbientHorizon = scatterHorizon * SKY_INTENSITY * CLOUD_SKY_AMBIENT_SCALE;

        //Hard cap horizon ambient at the zenith brightness. Without this
        //cap the horizon limb scatter (much brighter than zenith blue at
        //low sun) flows into cloud BASES via the lerp(horizon, top, h),
        //which inverts the lighting: bases get more ambient than tops,
        //as if the sun were illuminating from below. The zenith direction
        //dominates a cloud sample's visible sky hemisphere anyway, so
        //capping horizon to <= top is also a reasonable physical bound.
        float topLum     = max(Luma(skyAmbientTop), 1e-4f);
        float horizonLum = Luma(skyAmbientHorizon);
        if (horizonLum > topLum)
            skyAmbientHorizon *= topLum / horizonLum;
    }

    //-- Ground bounce envelope --
    //sunGroundT also uses g_skyObserverPlanet through TransmittanceToSun's
    //internal lookups; using the cloud-altitude probe gives a sensible
    //atmospheric attenuation factor (cloud-to-sun column) rather than
    //orbital-to-sun (which is ~1.0, over-brightening the ground bounce).
    float3 groundIrrad = float3(0, 0, 0);
    if (CLOUD_GROUND_BOUNCE > 0.5f)
    {
        float  sunCosUp   = saturate(dot(L, obsUp));
        float3 sunGroundT = TransmittanceToSun(cloudProbePos, L,
                                               ATMOS_BOTTOM_RADIUS,
                                               ATMOS_TOP_RADIUS);
        groundIrrad = sunIrradiance * sunGroundT * sunCosUp
                    * CLOUD_GROUND_ALBEDO * CLOUD_GROUND_SCALE
                    * (1.0f / PI);
    }

    //Restore the real observer for the rest of the function (sunAtmosShared
    //below uses a per-sample position, but anything else that might query
    //g_skyObserverPlanet during the march should see the camera).
    g_skyObserverPlanet = savedObserverPlanet;

    //-- Shared atmosphere transmittance at slab midpoint --
    float3 sunAtmosShared;
    {
        float tMid     = 0.5f * (tNear + tFar);
        float3 PMid    = O + V * tMid;
        sunAtmosShared = TransmittanceToSun(PMid, L,
                                            ATMOS_BOTTOM_RADIUS,
                                            ATMOS_TOP_RADIUS);
    }

    uint2 pixel = DispatchRaysIndex().xy;
    uint  frame = (uint)time;
    uint  cloudSeed = initRandomData(pixel, uint2(0, 0), frame, 71u);

    float cosTheta = dot(V, L);

    float3 trCloud = float3(1, 1, 1);
    float3 L_acc   = float3(0, 0, 0);

    float t        = tNear;
    float stepSize = CLOUD_TARGET_STEP_KM
                   * lerp(0.55f, 1.0f, CloudLodT(tNear));
    bool  lastStep  = false;
    int   acceptedI = 0;

    [loop]
    for (int totalSteps = 0; totalSteps < CLOUD_VIEW_STEPS_MAX; ++totalSteps)
    {
        if (t >= tFar) break;

        float distFade = 1.0f - smoothstep(CLOUD_FADE_DISTANCE_KM,
                                           CLOUD_RENDER_DISTANCE_KM, t);
        if (distFade <= 0.0f) break;

        float lodT    = CloudLodT(t);
        float rJit    = RandomFloatSingle(cloudSeed);
        float tSample = t + rJit * stepSize;
        float3 P      = O + V * tSample;

        float hull = CloudProbeHull(P, walltime) * distFade;
        if (hull <= CLOUD_EFFECTIVE_ZERO_DENSITY)
        {
            float maxStepAdaptive = min(
                CLOUD_MAX_STEP_KM * (1.0f + t * CLOUD_EMPTY_STEP_GROWTH_PER_KM),
                CLOUD_MAX_EMPTY_STEP_KM);
            float bigStep = lerp(stepSize, maxStepAdaptive, saturate(lodT));
            t += bigStep;
            continue;
        }

        CloudMaterial m = CloudSampleMaterial(P, walltime, lodT, 0u);
        float density = m.density * distFade;

        if (density > CLOUD_EFFECTIVE_ZERO_DENSITY)
        {
            float  sigma_t     = density * CLOUD_EXTINCTION;
            float3 sampleUp    = SafeNormalize(P);
            float  earthShadow = CloudEarthShadowFactor(P, L);

            float3 inscatter = CloudComputeLighting(
                P, V, L, m,
                sunIrradiance, sunAtmosShared,
                skyAmbientTop, skyAmbientHorizon, groundIrrad,
                sampleUp, cosTheta, earthShadow,
                pixel, frame, 200u + (uint)acceptedI * 11u);

            float segTr = exp(-sigma_t * stepSize);
            float segIn = (1.0f - segTr) / max(sigma_t, 1e-6f);
            L_acc   += trCloud * (sigma_t * inscatter) * segIn;
            trCloud *= segTr;
            acceptedI++;

            float trMax = max(trCloud.r, max(trCloud.g, trCloud.b));
            if (trMax < CLOUD_RR_THRESHOLD)
            {
                float p = trMax * (1.0f / CLOUD_RR_THRESHOLD);
                if (RandomFloatSingle(cloudSeed) > p)
                {
                    trCloud = float3(0, 0, 0);
                    break;
                }
                trCloud *= 1.0f / max(p, 1e-4f);
            }
        }

        stepSize = min(stepSize * CLOUD_STEP_GROWTH, CLOUD_MAX_FINE_STEP_KM);
        t += stepSize;

        if (lastStep) break;
        if (t > tFar) { t = tFar; lastStep = true; }
    }

    L_acc   *= limbFade;
    trCloud  = lerp(float3(1, 1, 1), trCloud, limbFade);

    L_acc *= aerialTr;
    float cloudOpacity = saturate(1.0f
                       - dot(trCloud, float3(0.2126f, 0.7152f, 0.0722f)));
    L_acc += aerialIn * SKY_INTENSITY * cloudOpacity * CLOUD_HAZE_STRENGTH;

    cloudTrOut = trCloud;
    return L_acc;
#endif
}

#endif
