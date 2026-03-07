#ifndef CLOUDS_HLSLI
#define CLOUDS_HLSLI

// ============================================================================
// Procedural Volumetric Clouds
// Based on: Schneider/Nubis lighting model (Guerrilla Games, 2015-2024)
//           Beer-Powder transmittance, dual-lobe Henyey-Greenstein,
//           ambient scattering, height-gradient density shaping
//
// Fully procedural (no textures). All noise is hash-based 3D value noise FBM.
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
// Cloud layer geometry (km, matching SunLight.hlsli planet scale)
// ============================================================================

// Planet radius (must match your atmosphere)
#ifndef CLOUD_PLANET_RADIUS
#define CLOUD_PLANET_RADIUS     6360.0f
#endif

// Cloud layer bottom and top altitude above sea level (km)
#ifndef CLOUD_ALTITUDE_MIN
#define CLOUD_ALTITUDE_MIN      1.5f
#endif

#ifndef CLOUD_ALTITUDE_MAX
#define CLOUD_ALTITUDE_MAX      3.5f
#endif

// ============================================================================
// Ray march settings
// ============================================================================

// Max ray march steps through the cloud volume
#ifndef CLOUD_MARCH_STEPS
#define CLOUD_MARCH_STEPS       64
#endif

// Steps for light sampling (from sample point toward sun)
#ifndef CLOUD_LIGHT_STEPS
#define CLOUD_LIGHT_STEPS       6
#endif

// Early-out transmittance threshold
#ifndef CLOUD_TRANSMIT_CUTOFF
#define CLOUD_TRANSMIT_CUTOFF   0.01f
#endif

// ============================================================================
// Cloud density modeling
// ============================================================================

// Overall cloud coverage [0..1]. 0 = clear, 1 = overcast.
#ifndef CLOUD_COVERAGE
#define CLOUD_COVERAGE          0.45f
#endif

// Base density multiplier (extinction coefficient scale)
#ifndef CLOUD_DENSITY
#define CLOUD_DENSITY           0.06f
#endif

// Noise frequency for base shape (lower = larger clouds)
#ifndef CLOUD_SHAPE_FREQ
#define CLOUD_SHAPE_FREQ        0.0008f
#endif

// Noise frequency for detail erosion (higher = finer wispy detail)
#ifndef CLOUD_DETAIL_FREQ
#define CLOUD_DETAIL_FREQ       0.004f
#endif

// Detail erosion strength [0..1]. Higher = more eroded edges.
#ifndef CLOUD_DETAIL_STRENGTH
#define CLOUD_DETAIL_STRENGTH   0.35f
#endif

// FBM octaves for base shape noise
#ifndef CLOUD_SHAPE_OCTAVES
#define CLOUD_SHAPE_OCTAVES     5
#endif

// FBM octaves for detail noise
#ifndef CLOUD_DETAIL_OCTAVES
#define CLOUD_DETAIL_OCTAVES    3
#endif

// Wind direction and speed (world units per... frame? second? Your choice.)
// This offsets the noise sampling position for animation.
#ifndef CLOUD_WIND_DIR
#define CLOUD_WIND_DIR          float3(0.1f, 0.0f, 0.05f)
#endif

// Time variable for animation (override if your global time symbol differs)
#ifndef CLOUD_TIME
#define CLOUD_TIME              0.0f
#endif

// ============================================================================
// Cloud lighting (Nubis model)
// ============================================================================

// Scattering albedo of cloud droplets [0..1]. Water clouds ~0.99.
#ifndef CLOUD_SCATTER_ALBEDO
#define CLOUD_SCATTER_ALBEDO    0.99f
#endif

// Dual-lobe Henyey-Greenstein parameters (Schneider model):
// Forward lobe asymmetry (strong forward scatter, silver lining)
#ifndef CLOUD_HG_FORWARD
#define CLOUD_HG_FORWARD        0.8f
#endif

// Backward lobe asymmetry (subtle backscatter)
#ifndef CLOUD_HG_BACK
#define CLOUD_HG_BACK           -0.5f
#endif

// Blend weight between forward and backward lobes [0..1]
// Higher = more forward scattering dominant
#ifndef CLOUD_HG_BLEND
#define CLOUD_HG_BLEND          0.7f
#endif

// Beer-Powder blending weight [0..1]
// Controls the "dark edges + inner glow" effect.
// 0 = pure Beer (no powder), 1 = full powder effect
#ifndef CLOUD_POWDER_STRENGTH
#define CLOUD_POWDER_STRENGTH   0.5f
#endif

// Ambient scattering intensity (zenith sky + ground bounce)
#ifndef CLOUD_AMBIENT_INTENSITY
#define CLOUD_AMBIENT_INTENSITY 0.25f
#endif

// Sun-side ambient: how much the sky hemisphere facing the sun
// contributes extra brightness to the sun-facing side of clouds.
// This simulates the bright horizon glow on the lit side vs darker
// ambient on the shadow side.  [0..1], 0 = no directional ambient.
#ifndef CLOUD_AMBIENT_SUN_SIDE
#define CLOUD_AMBIENT_SUN_SIDE  0.6f
#endif

// Sun intensity multiplier for cloud lighting
#ifndef CLOUD_SUN_INTENSITY
#define CLOUD_SUN_INTENSITY     1.0f
#endif

// ============================================================================
// Structs
// ============================================================================
struct CloudResult
{
    float3 color;       // premultiplied cloud radiance
    float  transmit;    // remaining transmittance (multiply background)
};

// ============================================================================
// Hash-based 3D noise (no textures)
// ============================================================================

// Integer hash for noise (fast, decorrelates well)
inline float Hash31(float3 p)
{
    p = frac(p * float3(0.1031f, 0.1030f, 0.0973f));
    p += dot(p, p.yzx + 33.33f);
    return frac((p.x + p.y) * p.z);
}

// Smooth 3D value noise
inline float ValueNoise3D(float3 p)
{
    float3 i = floor(p);
    float3 f = frac(p);

    // Quintic Hermite interpolation (smoother than cubic, fewer artifacts)
    float3 u = f * f * f * (f * (f * 6.0f - 15.0f) + 10.0f);

    // 8 corner hashes
    float n000 = Hash31(i + float3(0, 0, 0));
    float n100 = Hash31(i + float3(1, 0, 0));
    float n010 = Hash31(i + float3(0, 1, 0));
    float n110 = Hash31(i + float3(1, 1, 0));
    float n001 = Hash31(i + float3(0, 0, 1));
    float n101 = Hash31(i + float3(1, 0, 1));
    float n011 = Hash31(i + float3(0, 1, 1));
    float n111 = Hash31(i + float3(1, 1, 1));

    // Trilinear interpolation
    return lerp(
        lerp(lerp(n000, n100, u.x), lerp(n010, n110, u.x), u.y),
        lerp(lerp(n001, n101, u.x), lerp(n011, n111, u.x), u.y),
        u.z
    );
}

// FBM (fractal Brownian motion) for cloud shapes
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

        // Rotate slightly per octave to break axis alignment
        p = float3(
            p.x * 0.866f - p.z * 0.5f,
            p.y,
            p.x * 0.5f + p.z * 0.866f
        );
    }

    return value / totalAmp;
}

// ============================================================================
// Planet center: surface is at world origin (0,0,0), planet center below
// ============================================================================
static const float3 CLOUD_PLANET_CENTER = float3(0, -CLOUD_PLANET_RADIUS, 0);

// Altitude above planet surface from a world-space position
inline float CloudAltitude(float3 worldPos)
{
    return length(worldPos - CLOUD_PLANET_CENTER) - CLOUD_PLANET_RADIUS;
}

// Ray-sphere intersection (sphere centered at CLOUD_PLANET_CENTER)
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
// Cloud density
// ============================================================================

// Height gradient: controls vertical density profile of the cloud layer.
// Based on Schneider's gradient for cumulus clouds:
//   - Rounded bottom (density ramps up from base)
//   - Rounded top (density tapers off at top)
//   - Thickest in the lower-middle
inline float HeightGradient(float heightFraction)
{
    // heightFraction: 0 at cloud bottom, 1 at cloud top
    // Cumulus-like: ramp up quickly, dome off at top
    float bottom = saturate(smoothstep(0.0f, 0.2f, heightFraction));
    float top    = saturate(smoothstep(1.0f, 0.7f, heightFraction));
    return bottom * top;
}

// Sample cloud density at a world-space point (surface at origin)
float SampleCloudDensity(float3 worldPos)
{
    float altitude = CloudAltitude(worldPos);

    // Height fraction within cloud layer
    float layerThickness = CLOUD_ALTITUDE_MAX - CLOUD_ALTITUDE_MIN;
    float hFrac = (altitude - CLOUD_ALTITUDE_MIN) / layerThickness;

    // Outside cloud layer
    if (hFrac < 0.0f || hFrac > 1.0f) return 0.0f;

    // Height-based vertical density profile
    float hGrad = HeightGradient(hFrac);

    // Wind animation offset
    float3 windOffset = CLOUD_WIND_DIR * CLOUD_TIME;

    // -- Base shape (low frequency FBM) --
    float3 shapePos = worldPos * CLOUD_SHAPE_FREQ + windOffset * CLOUD_SHAPE_FREQ;
    float baseNoise = FBM(shapePos, CLOUD_SHAPE_OCTAVES, 2.0f, 0.5f);

    // Map noise to cloud coverage: remap so that CLOUD_COVERAGE controls fill
    // Higher coverage -> more of the noise range produces clouds
    float shapeDensity = saturate(baseNoise - (1.0f - CLOUD_COVERAGE));
    shapeDensity = shapeDensity / max(CLOUD_COVERAGE, 0.01f); // normalize

    // Apply height gradient
    shapeDensity *= hGrad;

    // Early out if no base density
    if (shapeDensity <= 0.001f) return 0.0f;

    // -- Detail erosion (high frequency FBM) --
    float3 detailPos = worldPos * CLOUD_DETAIL_FREQ + windOffset * CLOUD_DETAIL_FREQ * 1.5f;
    float detailNoise = FBM(detailPos, CLOUD_DETAIL_OCTAVES, 2.5f, 0.5f);

    // Erode edges: subtract detail noise, more erosion at cloud boundaries
    float edgeFactor = 1.0f - shapeDensity; // more erosion at edges
    float erosion = CLOUD_DETAIL_STRENGTH * detailNoise * (0.5f + 0.5f * edgeFactor);

    float finalDensity = saturate(shapeDensity - erosion);

    return finalDensity * CLOUD_DENSITY;
}

// Low-LOD density sample (fewer octaves, used for light march)
float SampleCloudDensityLOD(float3 worldPos)
{
    float altitude = CloudAltitude(worldPos);

    float layerThickness = CLOUD_ALTITUDE_MAX - CLOUD_ALTITUDE_MIN;
    float hFrac = (altitude - CLOUD_ALTITUDE_MIN) / layerThickness;

    if (hFrac < 0.0f || hFrac > 1.0f) return 0.0f;

    float hGrad = HeightGradient(hFrac);

    float3 windOffset = CLOUD_WIND_DIR * CLOUD_TIME;
    float3 shapePos = worldPos * CLOUD_SHAPE_FREQ + windOffset * CLOUD_SHAPE_FREQ;

    // Only 3 octaves for light sampling
    float baseNoise = FBM(shapePos, 3, 2.0f, 0.5f);

    float shapeDensity = saturate(baseNoise - (1.0f - CLOUD_COVERAGE));
    shapeDensity = shapeDensity / max(CLOUD_COVERAGE, 0.01f);
    shapeDensity *= hGrad;

    return max(0.0f, shapeDensity) * CLOUD_DENSITY;
}

// ============================================================================
// Cloud lighting (Schneider/Nubis model)
// ============================================================================

// Henyey-Greenstein phase function
inline float HG(float cosTheta, float g)
{
    float g2 = g * g;
    float denom = 1.0f + g2 - 2.0f * g * cosTheta;
    return (1.0f / (4.0f * PI)) * (1.0f - g2) / (denom * sqrt(max(1e-4f, denom)));
}

// Dual-lobe HG: combines forward and backward lobes for silver lining
inline float DualLobeHG(float cosTheta)
{
    float forward  = HG(cosTheta, CLOUD_HG_FORWARD);
    float backward = HG(cosTheta, CLOUD_HG_BACK);
    return lerp(backward, forward, CLOUD_HG_BLEND);
}

// Beer-Powder transmittance (Schneider 2015)
// Beer's Law alone makes clouds too dark in the interior.
// The "powder" term simulates in-scattering that brightens thin regions.
//   Beer:   T_beer   = exp(-density)
//   Powder: T_powder = 1 - exp(-2 * density)
//   Combined: lerp(T_beer, T_beer * T_powder, powder_strength)
inline float BeerPowder(float opticalDepth)
{
    float beer   = exp(-opticalDepth);
    float powder = 1.0f - exp(-2.0f * opticalDepth);
    return lerp(beer, beer * powder, CLOUD_POWDER_STRENGTH);
}

// Compute light energy at a sample point (Nubis model)
float3 CloudLightEnergy(float3 samplePos, float localDensity, float cosAngle,
                        float3 sunDir, float3 sunColor, float3 ambientTop, float3 ambientBot)
{
    // -- Direct scattering (sun) --
    // March toward sun to accumulate optical depth
    float lightOpticalDepth = 0.0f;
    float lightStepSize = (CLOUD_ALTITUDE_MAX - CLOUD_ALTITUDE_MIN) / (float)CLOUD_LIGHT_STEPS;

    for (int i = 0; i < CLOUD_LIGHT_STEPS; i++)
    {
        float3 lightSamplePos = samplePos + sunDir * ((float)(i + 1) * lightStepSize);
        lightOpticalDepth += SampleCloudDensityLOD(lightSamplePos) * lightStepSize;
    }

    // Beer-Powder transmittance toward sun
    float sunTransmittance = BeerPowder(lightOpticalDepth);

    // Phase function (dual-lobe HG)
    float phase = DualLobeHG(cosAngle);

    // Multi-scattering approximation: at high optical depth, scattered light
    // bounces around isotropically.  Lerp phase toward isotropic (1/4pi)
    // as optical depth increases (Schneider's "in-scattering" approximation).
    float msPhase = lerp(phase, 1.0f / (4.0f * PI), saturate(lightOpticalDepth * 3.0f));

    float3 directLight = sunColor * CLOUD_SUN_INTENSITY * sunTransmittance * msPhase * CLOUD_SCATTER_ALBEDO;

    // -- Ambient scattering (3-term directional model) --
    //
    // Term 1: Sky zenith — blue light from above, strongest at cloud tops.
    // Term 2: Ground bounce — warm reflected light from below, strongest at cloud bases.
    // Term 3: Sun-side horizon — the sky hemisphere toward the sun is significantly
    //         brighter than the opposite side.  We approximate the "pseudo-normal"
    //         of the cloud at this point using the height gradient: upper parts face
    //         up, lower parts face down, and we add a horizontal component toward/away
    //         from the sun to capture the directional asymmetry.
    //
    // This avoids expensive multi-direction sky sampling while producing the key
    // visual: sun-facing cloud flanks are brighter, shadow sides are darker/bluer.

    float altitude = CloudAltitude(samplePos);
    float hFrac = saturate((altitude - CLOUD_ALTITUDE_MIN) / (CLOUD_ALTITUDE_MAX - CLOUD_ALTITUDE_MIN));

    // Vertical blend: sky above, ground below
    float3 verticalAmbient = lerp(ambientBot, ambientTop, hFrac);

    // Sun-side horizon term: the sky hemisphere toward the sun is much brighter
    // than the opposite side.  We weight by view-sun angle (cosAngle) as a proxy
    // for how much of the bright hemisphere this cloud sample "sees", and by
    // sun elevation (more pronounced at sunrise/sunset when horizon is brightest).

    // At high sun, the horizon glow is less pronounced;
    // at low sun (sunrise/sunset), the horizon hemisphere is much brighter.
    float sunElevFactor = 1.0f - saturate(sunDir.y);

    // The sun-side brightness: a warm blend of sun color (attenuated) and sky top
    // representing the bright atmospheric glow on the sun-facing hemisphere.
    float3 sunSideColor = lerp(ambientTop, sunColor * 0.3f + ambientTop * 0.7f, sunElevFactor);

    // Weight by cosAngle: positive = facing sun, negative = facing away.
    // Remap from [-1,1] to [0,1] with a bias toward the lit side.
    float sunSideWeight = saturate(cosAngle * 0.5f + 0.5f);
    sunSideWeight *= sunSideWeight; // sharpen the transition slightly

    // Flank emphasis: strongest at mid-height, weakest at very top/bottom
    float flankFactor = 4.0f * hFrac * (1.0f - hFrac); // peaks at hFrac=0.5

    float3 sunSideAmbient = sunSideColor * sunSideWeight * flankFactor * CLOUD_AMBIENT_SUN_SIDE;

    float3 ambient = (verticalAmbient + sunSideAmbient) * CLOUD_AMBIENT_INTENSITY * CLOUD_SCATTER_ALBEDO;

    return directLight + ambient;
}

// ============================================================================
// ============================================================================
// Public API: Evaluate clouds along a ray direction
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

    // Observer at world origin (planet surface)
    float3 O = float3(0, 0.001f, 0);
    float3 V = normalize(rayDir);

    // Intersect with inner and outer cloud spheres
    float tInner0, tInner1, tOuter0, tOuter1;
    bool hitInner = CloudRaySphere(O, V, cloudInner, tInner0, tInner1);
    bool hitOuter = CloudRaySphere(O, V, cloudOuter, tOuter0, tOuter1);

    if (!hitOuter) return result; // ray misses atmosphere entirely

    // Determine entry/exit distances through the cloud shell
    // Observer is below clouds (at sea level):
    //   Entry = intersection with inner sphere (going up)
    //   Exit  = intersection with outer sphere (going up)
    float tEntry, tExit;

    if (hitInner && tInner1 > 0.0f)
    {
        // Observer below cloud base
        tEntry = max(0.0f, tInner1); // exit of inner sphere = entry into cloud shell
        tExit  = tOuter1;            // exit of outer sphere
    }
    else
    {
        // Observer inside or above cloud layer, or ray doesn't hit inner sphere
        tEntry = max(0.0f, tOuter0);
        tExit  = tOuter1;
    }

    // Skip if behind camera or zero-length
    if (tExit <= tEntry || tExit <= 0.0f) return result;

    // Limit max march distance to avoid excessive cost on nearly-horizontal rays
    float maxDist = (CLOUD_ALTITUDE_MAX - CLOUD_ALTITUDE_MIN) * 20.0f;
    tExit = min(tExit, tEntry + maxDist);

    float stepSize = (tExit - tEntry) / (float)CLOUD_MARCH_STEPS;
    float cosAngle = dot(V, sunDir);

    float transmittance = 1.0f;
    float3 scatteredLight = float3(0, 0, 0);

    for (int i = 0; i < CLOUD_MARCH_STEPS; i++)
    {
        if (transmittance < CLOUD_TRANSMIT_CUTOFF) break;

        float t = tEntry + ((float)i + 0.5f) * stepSize;
        float3 pos = O + V * t;

        float density = SampleCloudDensity(pos);

        if (density > 0.001f)
        {
            // Extinction for this step
            float extinction = density * stepSize;

            // Light energy at this point
            float3 lightEnergy = CloudLightEnergy(pos, density, cosAngle,
                                                  sunDir, sunColor,
                                                  skyColorTop, skyColorBot);

            // Accumulate: standard front-to-back compositing
            // In-scattered radiance = lightEnergy * density * stepSize
            // Attenuated by current transmittance
            float stepTransmittance = exp(-extinction);
            float3 integScatter = lightEnergy * (1.0f - stepTransmittance);

            scatteredLight += transmittance * integScatter;
            transmittance *= stepTransmittance;
        }
    }

    result.color    = scatteredLight;
    result.transmit = transmittance;
    return result;
}

#endif // CLOUDS_HLSLI