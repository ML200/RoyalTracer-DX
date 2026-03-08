#ifndef CLOUDS_HLSLI
#define CLOUDS_HLSLI

// ============================================================================
// Procedural Volumetric Clouds
// Single-pass, fully procedural clouds with a Nubis-3-style lighting model.
//
// Notes:
//   - Intended for miss shaders / ray traced sky.
//   - Uses white-noise stochastic sampling for DLSS RR.
//   - Does NOT use half-res / checkerboard / downsampling.
//   - Long-distance cost is controlled by distance fade + reduced light steps.
//   - Far clouds also use lower-bandwidth density LOD so RR can denoise them.
//
// Public API:
//   CloudResult EvaluateClouds(float3 rayDir, float3 sunDir, float3 sunColor,
//                              float3 skyColorTop, float3 skyColorBot)
//
// CloudResult contains:
//   .color     — premultiplied cloud radiance (add to background)
//   .transmit  — remaining transmittance [0..1] (multiply background by this)
//
// Usage in path tracer miss shader:
//   float3 sky = EvaluateSky(rayDir);
//   CloudResult clouds = EvaluateClouds(rayDir, sunDir, sunColor, skyTop, skyBot);
//   float3 finalColor = clouds.color + sky * clouds.transmit;
//
// All customization via #defines before including this file.
// ============================================================================

#ifndef PI
#define PI 3.14159265359
#endif

// ============================================================================
// DLSS Ray Reconstruction support
// Pass pixel coordinate and frame index for white noise.
// RR expects uncorrelated per-pixel-per-frame noise.
// ============================================================================
#ifndef CLOUD_PIXEL_COORD
#define CLOUD_PIXEL_COORD       float2(0, 0)
#endif

#ifndef CLOUD_FRAME_INDEX
#define CLOUD_FRAME_INDEX       0u
#endif

#ifndef CLOUD_USE_STOCHASTIC
#define CLOUD_USE_STOCHASTIC    0
#endif

#ifndef CLOUD_STOCHASTIC_STEPS
#define CLOUD_STOCHASTIC_STEPS  24
#endif

// ============================================================================
// Cloud layer geometry (meters)
// ============================================================================
#ifndef CLOUD_PLANET_RADIUS
#define CLOUD_PLANET_RADIUS 6360000.0f
#endif

#ifndef CLOUD_ALTITUDE_MIN
#define CLOUD_ALTITUDE_MIN  1500.0f
#endif

#ifndef CLOUD_ALTITUDE_MAX
#define CLOUD_ALTITUDE_MAX  3500.0f
#endif

// ============================================================================
// Ray march settings
// ============================================================================
#ifndef CLOUD_MARCH_STEPS
#define CLOUD_MARCH_STEPS       64
#endif

#ifndef CLOUD_LIGHT_STEPS
#define CLOUD_LIGHT_STEPS       4
#endif

#ifndef CLOUD_TRANSMIT_CUTOFF
#define CLOUD_TRANSMIT_CUTOFF   0.01f
#endif

// ============================================================================
// Cloud density modeling
// ============================================================================
#ifndef CLOUD_COVERAGE
#define CLOUD_COVERAGE          0.45f
#endif

#ifndef CLOUD_DENSITY
#define CLOUD_DENSITY           0.009f
#endif

#ifndef CLOUD_SHAPE_FREQ
#define CLOUD_SHAPE_FREQ        0.00035f
#endif

#ifndef CLOUD_DETAIL_FREQ
#define CLOUD_DETAIL_FREQ       0.003f
#endif

#ifndef CLOUD_DETAIL_STRENGTH
#define CLOUD_DETAIL_STRENGTH   0.45f
#endif

#ifndef CLOUD_SHAPE_OCTAVES
#define CLOUD_SHAPE_OCTAVES     5
#endif

#ifndef CLOUD_DETAIL_OCTAVES
#define CLOUD_DETAIL_OCTAVES    4
#endif

#ifndef CLOUD_WIND_DIR
#define CLOUD_WIND_DIR          float3(0.1f, 0.0f, 0.05f)
#endif

#ifndef CLOUD_TIME
#define CLOUD_TIME              0.0f
#endif

// ============================================================================
// Distance-based cloud field LOD
// These reduce far-cloud bandwidth so RR can reconstruct them better.
// ============================================================================
#ifndef CLOUD_LOD_START_DISTANCE
#define CLOUD_LOD_START_DISTANCE          30000.0f
#endif

#ifndef CLOUD_LOD_END_DISTANCE
#define CLOUD_LOD_END_DISTANCE            45000.0f
#endif

#ifndef CLOUD_FAR_SHAPE_OCTAVES
#define CLOUD_FAR_SHAPE_OCTAVES           3
#endif

#ifndef CLOUD_FAR_DETAIL_OCTAVES
#define CLOUD_FAR_DETAIL_OCTAVES          2
#endif

#ifndef CLOUD_FAR_DETAIL_STRENGTH
#define CLOUD_FAR_DETAIL_STRENGTH         0.12f
#endif

#ifndef CLOUD_FAR_DETAIL_FREQ_SCALE
#define CLOUD_FAR_DETAIL_FREQ_SCALE       0.35f
#endif

#ifndef CLOUD_FAR_LIGHT_STEPS
#define CLOUD_FAR_LIGHT_STEPS             1
#endif

// ============================================================================
// Cloud lighting
// ============================================================================
#ifndef CLOUD_SCATTER_ALBEDO
#define CLOUD_SCATTER_ALBEDO    0.99f
#endif

#ifndef CLOUD_HG_FORWARD
#define CLOUD_HG_FORWARD        0.7f
#endif

// Kept for compatibility with prior overrides, but unused in this version.
#ifndef CLOUD_HG_BACK
#define CLOUD_HG_BACK           -0.4f
#endif

#ifndef CLOUD_HG_BLEND
#define CLOUD_HG_BLEND          0.6f
#endif

#ifndef CLOUD_POWDER_STRENGTH
#define CLOUD_POWDER_STRENGTH   0.2f
#endif

#ifndef CLOUD_AMBIENT_INTENSITY
#define CLOUD_AMBIENT_INTENSITY 0.62f
#endif

#ifndef CLOUD_AMBIENT_SUN_SIDE
#define CLOUD_AMBIENT_SUN_SIDE  0.6f
#endif

#ifndef CLOUD_SUN_INTENSITY
#define CLOUD_SUN_INTENSITY     1.0f
#endif

// ============================================================================
// Nubis-3-style single-pass lighting controls
// ============================================================================
#ifndef CLOUD_NUBIS_PRIMARY_G
#define CLOUD_NUBIS_PRIMARY_G              CLOUD_HG_FORWARD
#endif

#ifndef CLOUD_NUBIS_SILVER_INTENSITY
#define CLOUD_NUBIS_SILVER_INTENSITY       0.35f
#endif

#ifndef CLOUD_NUBIS_SILVER_SPREAD
#define CLOUD_NUBIS_SILVER_SPREAD          0.08f
#endif

#ifndef CLOUD_NUBIS_SECONDARY_G
#define CLOUD_NUBIS_SECONDARY_G            0.18f
#endif

#ifndef CLOUD_NUBIS_SECONDARY_STRENGTH
#define CLOUD_NUBIS_SECONDARY_STRENGTH     0.45f
#endif

#ifndef CLOUD_NUBIS_MS_BASE_ATTEN
#define CLOUD_NUBIS_MS_BASE_ATTEN          0.25f
#endif

#ifndef CLOUD_NUBIS_MS_CORE_ATTEN
#define CLOUD_NUBIS_MS_CORE_ATTEN          0.05f
#endif

#ifndef CLOUD_NUBIS_MS_BOTTOM_START
#define CLOUD_NUBIS_MS_BOTTOM_START        0.08f
#endif

#ifndef CLOUD_NUBIS_MS_BOTTOM_END
#define CLOUD_NUBIS_MS_BOTTOM_END          0.35f
#endif

#ifndef CLOUD_NUBIS_AMBIENT_STEPS
#define CLOUD_NUBIS_AMBIENT_STEPS          2
#endif

#ifndef CLOUD_NUBIS_AMBIENT_DISTANCE
#define CLOUD_NUBIS_AMBIENT_DISTANCE       6000.0f
#endif

#ifndef CLOUD_NUBIS_LIGHT_GROWTH
#define CLOUD_NUBIS_LIGHT_GROWTH           1.6f
#endif

#ifndef CLOUD_NUBIS_NORMAL_EPS
#define CLOUD_NUBIS_NORMAL_EPS             140.0f
#endif

#ifndef CLOUD_NUBIS_DIFFUSE_STRENGTH
#define CLOUD_NUBIS_DIFFUSE_STRENGTH       0.45f
#endif

#ifndef CLOUD_NUBIS_DIFFUSE_ATTEN
#define CLOUD_NUBIS_DIFFUSE_ATTEN          0.45f
#endif

#ifndef CLOUD_NUBIS_AMBIENT_FLOOR
#define CLOUD_NUBIS_AMBIENT_FLOOR          0.5f
#endif

#ifndef CLOUD_NUBIS_AMBIENT_OCCLUSION
#define CLOUD_NUBIS_AMBIENT_OCCLUSION      0.12f
#endif

#ifndef CLOUD_NUBIS_AMBIENT_HORIZON_BOOST
#define CLOUD_NUBIS_AMBIENT_HORIZON_BOOST  0.30f
#endif

#ifndef CLOUD_NUBIS_GROUND_BOUNCE
#define CLOUD_NUBIS_GROUND_BOUNCE          0.11f
#endif

// ============================================================================
// Render distance
// ============================================================================
#ifndef CLOUD_RENDER_DISTANCE
#define CLOUD_RENDER_DISTANCE   60000.0f
#endif

#ifndef CLOUD_FADE_DISTANCE
#define CLOUD_FADE_DISTANCE     43000.0f
#endif

// ============================================================================
// Step budget [0..1]
// Controls ONLY step count, never base/detail octave quality.
// Lower = fewer steps, more white noise.
// ============================================================================
#ifndef CLOUD_STEP_BUDGET
#define CLOUD_STEP_BUDGET       0.3f
#endif

// ============================================================================
// Atmospheric haze
// ============================================================================
#ifndef CLOUD_HAZE_RAYLEIGH
#define CLOUD_HAZE_RAYLEIGH     float3(0.005802f, 0.013558f, 0.033100f)
#endif

#ifndef CLOUD_HAZE_DENSITY
#define CLOUD_HAZE_DENSITY      0.003f
#endif

// ============================================================================
// Structs
// ============================================================================
struct CloudResult
{
    float3 color;
    float  transmit;
};

struct CloudMaterialSample
{
    float density;
    float profile;
    float heightFrac;
};

// ============================================================================
// Hash-based 3D noise
// ============================================================================
inline float Hash31(float3 p)
{
    uint3 q = uint3(int3(p)) * uint3(1597334677u, 3812015801u, 2798796415u);
    uint n = (q.x ^ q.y ^ q.z) * 1597334677u;
    return float(n) * (1.0f / float(0xFFFFFFFFu));
}

struct CloudRNG
{
    uint state;

    static CloudRNG Create(float2 pixelCoord, uint frameIndex)
    {
        CloudRNG rng;
        uint2 px = uint2(pixelCoord);
        rng.state = px.x * 1597334677u + px.y * 3812015801u + frameIndex * 2798796415u + 1u;
        rng.Next();
        return rng;
    }

    float Next()
    {
        state = state * 747796405u + 2891336453u;
        uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
        word = (word >> 22u) ^ word;
        return float(word) * (1.0f / float(0xFFFFFFFFu));
    }
};

inline float ValueNoise3D(float3 p)
{
    float3 i = floor(p);
    float3 f = frac(p);
    float3 u = f * f * f * (f * (f * 6.0f - 15.0f) + 10.0f);

    float n000 = Hash31(i + float3(0, 0, 0));
    float n100 = Hash31(i + float3(1, 0, 0));
    float n010 = Hash31(i + float3(0, 1, 0));
    float n110 = Hash31(i + float3(1, 1, 0));
    float n001 = Hash31(i + float3(0, 0, 1));
    float n101 = Hash31(i + float3(1, 0, 1));
    float n011 = Hash31(i + float3(0, 1, 1));
    float n111 = Hash31(i + float3(1, 1, 1));

    return lerp(
        lerp(lerp(n000, n100, u.x), lerp(n010, n110, u.x), u.y),
        lerp(lerp(n001, n101, u.x), lerp(n011, n111, u.x), u.y),
        u.z
    );
}

inline float FBM(float3 p, int octaves, float freqMul, float ampMul)
{
    float value = 0.0f;
    float amplitude = 0.5f;
    float totalAmp = 0.0f;

    for (int i = 0; i < octaves; i++)
    {
        value += amplitude * ValueNoise3D(p);
        totalAmp += amplitude;
        amplitude *= ampMul;
        p *= freqMul;

        p = float3(
            p.x * 0.866f - p.z * 0.5f,
            p.y * 0.866f - p.x * 0.25f - p.z * 0.433f,
            p.x * 0.433f + p.y * 0.5f + p.z * 0.75f
        );
    }

    return value / totalAmp;
}

// ============================================================================
// Planet geometry
// ============================================================================
static const float3 CLOUD_PLANET_CENTER = float3(0, -CLOUD_PLANET_RADIUS, 0);

inline float CloudAltitude(float3 worldPos)
{
    return length(worldPos - CLOUD_PLANET_CENTER) - CLOUD_PLANET_RADIUS;
}

inline float3 CloudLocalUp(float3 worldPos)
{
    return normalize(worldPos - CLOUD_PLANET_CENTER);
}

inline bool CloudRaySphere(float3 ro, float3 rd, float radius, out float t0, out float t1)
{
    float3 oc = ro - CLOUD_PLANET_CENTER;
    float b = dot(oc, rd);
    float c = dot(oc, oc) - radius * radius;
    float disc = b * b - c;
    if (disc < 0.0f) { t0 = 0.0f; t1 = 0.0f; return false; }
    disc = sqrt(disc);
    t0 = -b - disc;
    t1 = -b + disc;
    return true;
}

// ============================================================================
// Density shaping
// ============================================================================
inline float HeightGradient(float heightFraction)
{
    float bottom = saturate(smoothstep(0.0f, 0.2f, heightFraction));
    float top    = saturate(smoothstep(1.0f, 0.7f, heightFraction));
    return bottom * top;
}

inline float CloudDistanceLOD(float distanceToSample)
{
    return saturate((distanceToSample - CLOUD_LOD_START_DISTANCE) /
                    max(1.0f, (CLOUD_LOD_END_DISTANCE - CLOUD_LOD_START_DISTANCE)));
}

CloudMaterialSample SampleCloudMaterialInternal(
    float3 worldPos,
    int shapeOctaves,
    int detailOctaves,
    float detailStrength,
    float detailFreq,
    float warpBlend)
{
    CloudMaterialSample s;
    s.density    = 0.0f;
    s.profile    = 0.0f;
    s.heightFrac = 0.0f;

    if (shapeOctaves < 0)  shapeOctaves  = CLOUD_SHAPE_OCTAVES;
    if (detailOctaves < 0) detailOctaves = CLOUD_DETAIL_OCTAVES;

    float altitude = CloudAltitude(worldPos);
    float layerThickness = CLOUD_ALTITUDE_MAX - CLOUD_ALTITUDE_MIN;
    float hFrac = (altitude - CLOUD_ALTITUDE_MIN) / layerThickness;
    s.heightFrac = saturate(hFrac);

    if (hFrac < 0.0f || hFrac > 1.0f) return s;

    float hGrad = HeightGradient(hFrac);
    float3 windOffset = CLOUD_WIND_DIR * CLOUD_TIME;

    float3 shapePos = worldPos * CLOUD_SHAPE_FREQ + windOffset * CLOUD_SHAPE_FREQ;

    if (warpBlend > 0.001f)
    {
        float3 warp = float3(
            ValueNoise3D(shapePos * 0.7f + 7.7f),
            ValueNoise3D(shapePos * 0.7f + 13.1f),
            ValueNoise3D(shapePos * 0.7f + 23.5f)
        );
        shapePos += (warp - 0.5f) * 400.0f * CLOUD_SHAPE_FREQ * warpBlend;
    }

    float baseNoise = FBM(shapePos, shapeOctaves, 2.0f, 0.5f);

    float baseShape = saturate(baseNoise - (1.0f - CLOUD_COVERAGE));
    baseShape = baseShape / max(CLOUD_COVERAGE, 0.01f);
    baseShape *= hGrad;

    if (baseShape <= 0.001f) return s;

    s.profile = saturate(baseShape);

    float finalShape = baseShape;

    if (detailOctaves > 0 && detailStrength > 0.001f)
    {
        float3 detailPos = worldPos * detailFreq + windOffset * detailFreq * 1.5f;
        float detailNoise = FBM(detailPos, detailOctaves, 2.5f, 0.5f);

        float edgeFactor = 1.0f - finalShape;
        float erosion = detailStrength * detailNoise * (0.5f + 0.5f * edgeFactor);

        finalShape = saturate(finalShape - erosion);
    }

    s.density = finalShape * CLOUD_DENSITY;
    return s;
}

CloudMaterialSample SampleCloudMaterial(float3 worldPos, int shapeOctaves, int detailOctaves)
{
    return SampleCloudMaterialInternal(
        worldPos,
        shapeOctaves,
        detailOctaves,
        CLOUD_DETAIL_STRENGTH,
        CLOUD_DETAIL_FREQ,
        (shapeOctaves >= CLOUD_SHAPE_OCTAVES) ? 1.0f : 0.0f
    );
}

CloudMaterialSample SampleCloudMaterial(float3 worldPos)
{
    return SampleCloudMaterial(worldPos, -1, -1);
}

CloudMaterialSample SampleCloudMaterialLOD(float3 worldPos)
{
    return SampleCloudMaterialInternal(
        worldPos,
        3,
        0,
        0.0f,
        CLOUD_DETAIL_FREQ,
        0.0f
    );
}

CloudMaterialSample SampleCloudMaterialDistanceLOD(float3 worldPos, float distanceToSample)
{
    float lod = CloudDistanceLOD(distanceToSample);

    int shapeOctaves =
        (lod < 0.5f) ? CLOUD_SHAPE_OCTAVES : CLOUD_FAR_SHAPE_OCTAVES;

    int detailOctaves =
        (lod < 0.35f) ? CLOUD_DETAIL_OCTAVES :
        (lod < 0.75f) ? 2 :
                        CLOUD_FAR_DETAIL_OCTAVES;

    float detailFreq = lerp(CLOUD_DETAIL_FREQ,
                            CLOUD_DETAIL_FREQ * CLOUD_FAR_DETAIL_FREQ_SCALE,
                            lod);

    float detailStrength = lerp(CLOUD_DETAIL_STRENGTH,
                                CLOUD_FAR_DETAIL_STRENGTH,
                                lod);

    float warpBlend = (lod < 0.35f) ? (1.0f - lod / 0.35f) : 0.0f;

    return SampleCloudMaterialInternal(
        worldPos,
        shapeOctaves,
        detailOctaves,
        detailStrength,
        detailFreq,
        warpBlend
    );
}

float SampleCloudDensity(float3 worldPos, int shapeOctaves, int detailOctaves)
{
    return SampleCloudMaterial(worldPos, shapeOctaves, detailOctaves).density;
}

float SampleCloudDensity(float3 worldPos)
{
    return SampleCloudMaterial(worldPos).density;
}

float SampleCloudDensityLOD(float3 worldPos)
{
    return SampleCloudMaterialLOD(worldPos).density;
}

// ============================================================================
// Lighting helpers
// ============================================================================
inline float HG(float cosTheta, float g)
{
    float g2 = g * g;
    float denom = 1.0f + g2 - 2.0f * g * cosTheta;
    return (1.0f / (4.0f * PI)) * (1.0f - g2) / (denom * sqrt(max(1e-4f, denom)));
}

inline float CloudPrimaryPhase(float cosTheta)
{
    float basePhase   = HG(cosTheta, CLOUD_NUBIS_PRIMARY_G);
    float silverPhase = CLOUD_NUBIS_SILVER_INTENSITY * HG(cosTheta, 0.99f - CLOUD_NUBIS_SILVER_SPREAD);
    return max(basePhase, silverPhase);
}

inline float CloudSecondaryPhase(float cosTheta)
{
    return HG(cosTheta, CLOUD_NUBIS_SECONDARY_G);
}

inline float CloudSafeRcpLength(float3 v)
{
    return rsqrt(max(dot(v, v), 1e-8f));
}

inline float3 CloudSafeNormalize(float3 v)
{
    return v * CloudSafeRcpLength(v);
}

// 4-tap tetrahedral gradient on low-LOD density.
inline float3 EstimateCloudNormalLOD(float3 p)
{
    float e = CLOUD_NUBIS_NORMAL_EPS;

    float3 k1 = float3( 1.0f, -1.0f, -1.0f);
    float3 k2 = float3(-1.0f, -1.0f,  1.0f);
    float3 k3 = float3(-1.0f,  1.0f, -1.0f);
    float3 k4 = float3( 1.0f,  1.0f,  1.0f);

    float d1 = SampleCloudDensityLOD(p + k1 * e);
    float d2 = SampleCloudDensityLOD(p + k2 * e);
    float d3 = SampleCloudDensityLOD(p + k3 * e);
    float d4 = SampleCloudDensityLOD(p + k4 * e);

    float3 grad = k1 * d1 + k2 * d2 + k3 * d3 + k4 * d4;

    return CloudSafeNormalize(-grad);
}

inline float EarthShadowFactor(float3 worldPos, float3 sunDir)
{
    float3 oc = worldPos - CLOUD_PLANET_CENTER;
    float b = dot(oc, sunDir);

    if (b >= 0.0f) return 1.0f;

    float c = dot(oc, oc) - CLOUD_PLANET_RADIUS * CLOUD_PLANET_RADIUS;
    float disc = b * b - c;

    float penumbra = CLOUD_PLANET_RADIUS * CLOUD_PLANET_RADIUS * 2e-2f;
    return saturate(0.5f - disc / penumbra);
}

float MarchSunOpticalDepth(float3 startPos, float3 sunDir, int steps, inout CloudRNG rng)
{
    float maxDist = CLOUD_ALTITUDE_MAX - CLOUD_ALTITUDE_MIN;
    steps = max(1, steps);

    float weightSum = 0.0f;
    for (int i = 0; i < steps; i++)
    {
        weightSum += pow(CLOUD_NUBIS_LIGHT_GROWTH, (float)i);
    }

    float baseLen = maxDist / max(weightSum, 1e-4f);

    float tau = 0.0f;
    float accum = 0.0f;

    for (int i = 0; i < steps; i++)
    {
        float segLen = baseLen * pow(CLOUD_NUBIS_LIGHT_GROWTH, (float)i);
        float t = accum + rng.Next() * segLen;

        tau += SampleCloudMaterialLOD(startPos + sunDir * t).density * segLen;
        accum += segLen;
    }

    return tau;
}

float MarchSkyOpticalDepth(float3 startPos, float3 dir, float distance, int steps, inout CloudRNG rng)
{
    steps = max(1, steps);
    float stepLen = distance / (float)steps;
    float tau = 0.0f;

    for (int i = 0; i < steps; i++)
    {
        float t = ((float)i + rng.Next()) * stepLen;
        tau += SampleCloudMaterialLOD(startPos + dir * t).density * stepLen;
    }

    return tau;
}

// ============================================================================
// Nubis-3-style single-pass lighting
// ============================================================================
//
// direct = forward phase + broad secondary term + sun-facing diffuse shell
// ambient = sky hemisphere + sky occlusion + subtle ground bounce
//
float3 CloudLightEnergyNubis3(float3 samplePos,
                              CloudMaterialSample material,
                              float cosAngle,
                              float3 sunDir, float3 sunColor,
                              float3 ambientTop, float3 ambientBot,
                              int lightSteps,
                              inout CloudRNG rng)
{
    float sunOD = MarchSunOpticalDepth(samplePos, sunDir, lightSteps, rng);
    float earthShadow = EarthShadowFactor(samplePos, sunDir);

    float transmittance = exp(-sunOD);
    float primary = transmittance * CloudPrimaryPhase(cosAngle);

    float coreAtten = lerp(CLOUD_NUBIS_MS_BASE_ATTEN,
                           CLOUD_NUBIS_MS_CORE_ATTEN,
                           material.profile);

    float msVertical = smoothstep(CLOUD_NUBIS_MS_BOTTOM_START,
                                  CLOUD_NUBIS_MS_BOTTOM_END,
                                  material.heightFrac);

    float msVolume = material.profile;
    msVolume *= msVertical;
    msVolume *= exp(-sunOD * coreAtten);

    float secondary = CLOUD_NUBIS_SECONDARY_STRENGTH *
                      msVolume *
                      CloudSecondaryPhase(cosAngle);

    float3 normalWS = EstimateCloudNormalLOD(samplePos);
    float NdotL = saturate(dot(normalWS, sunDir));

    float shell = sqrt(saturate(1.0f - material.profile));
    float shellMask = shell * shell;

    float diffuseShell = CLOUD_NUBIS_DIFFUSE_STRENGTH *
                         NdotL *
                         shellMask *
                         exp(-sunOD * CLOUD_NUBIS_DIFFUSE_ATTEN);

    float3 directLight = sunColor * CLOUD_SUN_INTENSITY *
                         (primary + secondary + diffuseShell) *
                         earthShadow;

    float3 localUp = CloudLocalUp(samplePos);
    float ambientOD = MarchSkyOpticalDepth(samplePos,
                                           localUp,
                                           CLOUD_NUBIS_AMBIENT_DISTANCE,
                                           CLOUD_NUBIS_AMBIENT_STEPS,
                                           rng);

    float ambientVisibility = exp(-ambientOD * CLOUD_NUBIS_AMBIENT_OCCLUSION);

    float h = material.heightFrac;
    float horizonBand = 4.0f * h * (1.0f - h);

    float3 horizonColor = lerp(ambientBot, ambientTop, 0.5f);

    float3 ambientBase =
        lerp(ambientBot * 0.35f,
             ambientTop,
             h);

    ambientBase += horizonColor * (CLOUD_NUBIS_AMBIENT_HORIZON_BOOST * horizonBand);

    float ambientPotential = lerp(CLOUD_NUBIS_AMBIENT_FLOOR, 1.0f, shell);

    float3 ambient = ambientBase *
                     ambientPotential *
                     ambientVisibility *
                     CLOUD_AMBIENT_INTENSITY;

    ambient += ambientBot *
               ambientPotential *
               (1.0f - h) *
               CLOUD_NUBIS_GROUND_BOUNCE;

    return (directLight + ambient) * CLOUD_SCATTER_ALBEDO;
}

// ============================================================================
// Public API
// ============================================================================
CloudResult EvaluateClouds(float3 rayDir, float3 sunDir, float3 sunColor,
                           float3 skyColorTop, float3 skyColorBot)
{
    CloudResult result;
    result.color    = float3(0, 0, 0);
    result.transmit = 1.0f;

    float Rb = CLOUD_PLANET_RADIUS;
    float cloudInner = Rb + CLOUD_ALTITUDE_MIN;
    float cloudOuter = Rb + CLOUD_ALTITUDE_MAX;

    float3 O = float3(0, 0.01f, 0);
    float3 V = normalize(rayDir);

    float tInner0, tInner1, tOuter0, tOuter1;
    bool hitInner = CloudRaySphere(O, V, cloudInner, tInner0, tInner1);
    bool hitOuter = CloudRaySphere(O, V, cloudOuter, tOuter0, tOuter1);

    if (!hitOuter) return result;

    float tEntry, tExit;

    if (hitInner && tInner1 > 0.0f)
    {
        tEntry = max(0.0f, tInner1);
        tExit  = tOuter1;
    }
    else
    {
        tEntry = max(0.0f, tOuter0);
        tExit  = tOuter1;
    }

    float tGround0, tGround1;
    bool hitGround = CloudRaySphere(O, V, Rb, tGround0, tGround1);
    if (hitGround && tGround0 > 0.0f)
    {
        tExit = min(tExit, tGround0);
    }

    float upDot = dot(V, float3(0, 1, 0));
    float horizonFade = saturate(upDot / 0.009f);
    if (horizonFade <= 0.0f) return result;

    if (tExit <= tEntry || tExit <= 0.0f) return result;

    float maxDist  = CLOUD_RENDER_DISTANCE;
    float fadeDist = CLOUD_FADE_DISTANCE;
    tExit = min(tExit, tEntry + maxDist);

    float totalDist = tExit - tEntry;
    float cosAngle  = dot(V, sunDir);

    float transmittance = 1.0f;
    float3 scatteredLight = float3(0, 0, 0);

    float hazeWeightedDist = 0.0f;
    float hazeWeightSum    = 0.0f;

    float q = saturate(CLOUD_STEP_BUDGET);

    CloudRNG rng = CloudRNG::Create(CLOUD_PIXEL_COORD, CLOUD_FRAME_INDEX);
    rng.state ^= (uint)(frac(dot(abs(V), float3(0.7548777f, 0.5698403f, 0.4382891f))) * 16777215.0f);
    rng.Next();

#if CLOUD_USE_STOCHASTIC
    int marchStepsBase = max(8,  (int)lerp(12.0f, (float)CLOUD_STOCHASTIC_STEPS, q));
    int lightStepsBase = max(2,  (int)lerp(2.0f,  4.0f, q));
#else
    int marchStepsBase = max(12, (int)lerp(20.0f, (float)CLOUD_MARCH_STEPS, q));
    int lightStepsBase = max(2,  (int)lerp(2.0f,  (float)CLOUD_LIGHT_STEPS, q));
#endif

    float stepSize = totalDist / (float)max(1, marchStepsBase);
    float jitter   = rng.Next();

    for (int i = 0; i < marchStepsBase; i++)
    {
        if (transmittance < CLOUD_TRANSMIT_CUTOFF) break;

        float t = tEntry + ((float)i + jitter) * stepSize;
        if (t >= tExit) break;

        float3 pos = O + V * t;

        float distFromEntry = t - tEntry;
        float distFade = 1.0f - saturate((distFromEntry - fadeDist) / max(1.0f, (maxDist - fadeDist)));

        CloudMaterialSample material = SampleCloudMaterialDistanceLOD(pos, t);
        float density = material.density * distFade;

        if (density <= 0.001f) continue;

        float sampleWeight = density * stepSize;
        hazeWeightedDist += t * sampleWeight;
        hazeWeightSum    += sampleWeight;

        float extinction = density * stepSize;

        float farLerp = CloudDistanceLOD(t);
        int lightStepsThisSample = max(1, (int)lerp((float)lightStepsBase,
                                                    (float)CLOUD_FAR_LIGHT_STEPS,
                                                    farLerp));

        float3 lightEnergy = CloudLightEnergyNubis3(pos,
                                                    material,
                                                    cosAngle,
                                                    sunDir, sunColor,
                                                    skyColorTop, skyColorBot,
                                                    lightStepsThisSample,
                                                    rng);

        float stepTransmittance = exp(-extinction);
        float3 integScatter = lightEnergy * (1.0f - stepTransmittance);

        scatteredLight += transmittance * integScatter;
        transmittance *= stepTransmittance;
    }

    float hazeDist = (hazeWeightSum > 0.001f)
        ? (hazeWeightedDist / hazeWeightSum)
        : tEntry;

    float hazeOD = hazeDist * CLOUD_HAZE_DENSITY;
    float3 hazeTransmit = exp(-CLOUD_HAZE_RAYLEIGH * hazeOD);

    float skyViewT = saturate(pow(max(V.y, 0.0f), 0.35f));
    float3 skyAlongRay = lerp(skyColorBot, skyColorTop, skyViewT);
    float3 hazeColor   = skyAlongRay * (1.0f - hazeTransmit);

    float cloudOpacity = 1.0f - transmittance;
    result.color    = (scatteredLight * hazeTransmit + hazeColor * cloudOpacity) * horizonFade;
    result.transmit = lerp(1.0f, transmittance, horizonFade);
    return result;
}

#endif // CLOUDS_HLSLI