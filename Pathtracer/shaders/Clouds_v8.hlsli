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

   LIGHTING (Wrenninge octaves, Frostbite-style)
     Sun ray Beer-Lambert transmittance (no fake multipliers)
     Direct: dual-lobe Henyey-Greenstein matching Mie shape (octave 0)
     Multi-scatter: Wrenninge octave sum (Hillaire 2016 §5.8). Each
       octave n in [1..N-1] is an additional single-scatter term with
       sun-ray extinction raised to b^n, contribution scaled by a^n,
       and phase eccentricity scaled by c^n. With N=3, a=b=c=0.5 the
       cloud cores receive light from the n=2 octave (Tdir^0.25, 0.25x
       amplitude) which the old single-term approximation couldn't
       reach. heightBias floor + STRENGTH artist amp still modulate
       octaves >=1 so cumulus bases retain their lit floor.
     Sky ambient: occluded by column-density above sample
       Probe origin relocated to cloud altitude — orbital cameras would
       otherwise integrate through near-zero atmosphere upward, producing
       skyAmbientTop ≈ 0 and the same inversion.
     Ground bounce: attenuated by sample height in layer
     Integration: energy-conserving analytical (Hillaire 2016 §5.6.3):
       L_acc += T * inscatter * (1 - segTr); transmittance-weighted mean
       depth fed back to caller for DLSS motion-vector reprojection
       (§5.9.1 Eq. 21).

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

// Shape / coverage. CLOUD_COVERAGE_BASE is the single global coverage
// value. The per-location weathermap variation (low-freq value noise in
// xz) was removed because its grid-cell boundaries produced visible
// rectangular patches in the sky.
#ifndef CLOUD_COVERAGE_BASE
#define CLOUD_COVERAGE_BASE     0.45f
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

// Wind animation — translates the noise sample space so clouds drift.
// Unrelated to the removed weathermap variation.
#ifndef CLOUD_WIND_X
#define CLOUD_WIND_X            0.04f
#endif
#ifndef CLOUD_WIND_Z
#define CLOUD_WIND_Z            0.015f
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

// MS strength — global multiplier on the Nubis single-term MS. The
// single-term formulation only captures one effective scattering bounce;
// for an albedo-≈1 cloud, real multi-bounce contributes ~2-3x more
// energy on top of that. 4.0 compensates so overcast tops viewed from
// above reach the physical Lambertian-reflective brightness (~1.9 per
// channel at K=6) instead of reading as mid-grey (~0.6). Lower for
// dramatic cumulus shadows, higher for marshmallow-puffy character.
#ifndef CLOUD_MS_STRENGTH
#define CLOUD_MS_STRENGTH       4.0f
#endif

// MS height floor — minimum MS contribution at cloud base (h=0).
// Without a floor bases turn black; 0.40 maps to "shaded white" which
// matches real cumulus bottoms.
#ifndef CLOUD_MS_HEIGHT_FLOOR
#define CLOUD_MS_HEIGHT_FLOOR   0.40f
#endif

// Wrenninge multi-scatter octaves (Hillaire 2016 §5.8, after Wrenninge
// "Oz: The Great and Volumetric" 2013). Approximates higher orders of
// scattering as a sum of single-scatter terms, each with reduced
// scattering amplitude (a^n), reduced extinction (b^n), and reduced
// phase eccentricity (c^n). Octave 0 is the un-modified direct term;
// octaves 1..N-1 stand in for the multi-scatter contribution.
//
// Constraint for energy conservation: a <= b. With a=b=0.5 (the
// Frostbite-default values), each octave halves its amplitude AND
// halves its effective extinction — so the sun ray reaches deeper into
// the cloud at each successive octave, brightening the core. c=0.5
// broadens the phase function per octave so later octaves contribute
// near-isotropic fill rather than a directional halo.
//
// N=3 is the sweet spot: octave 0 = direct, octave 1 = "first bounce"
// (matches the old sqrt(Tdir) approximation), octave 2 = "deep core
// fill" (brightens the back of thick cumulus that single-term MS
// leaves dark). N=2 falls back to the previous behaviour roughly.
#ifndef CLOUD_MS_OCTAVES_N
#define CLOUD_MS_OCTAVES_N      3
#endif
#ifndef CLOUD_MS_A
#define CLOUD_MS_A              0.50f
#endif
#ifndef CLOUD_MS_B
#define CLOUD_MS_B              0.50f
#endif
#ifndef CLOUD_MS_C
#define CLOUD_MS_C              0.50f
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
#define CLOUD_VIEW_STEPS_MAX     128
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

// CloudHash3D / CloudHashFloat / CloudHashVec3 were used by the old analytical
// CloudValueNoise / CloudWorley / CloudPerlin. Removed — the runtime path now
// samples a baked 3D texture (g_cloudNoise) and the tileable hashes live in
// Pass_cloudnoise_bake_v8.hlsl (BakeHash3D etc.) where they belong.

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

//------------------------------------------------------------------------------
// NOISE — texture-baked lookups (drop-in replacements for the old analytical
// CloudPerlinWorley / CloudWorleyFBM / CloudValueNoise / CloudWorley).
//------------------------------------------------------------------------------
// g_cloudNoise (Includes_v8.hlsli) is a 256³ RGBA8 3D texture filled once
// at startup by Pass_cloudnoise_bake_v8.hlsl. Each channel was baked at a
// specific period so the texture tiles seamlessly when sampled with WRAP
// addressing; the wrappers below divide the caller's `p` by that period to
// produce the correct UVW coordinate.
//
// Per-sample cost goes from ~80-200 ALU ops (analytical FBM with 27-neighbour
// Worley + Perlin gradient FBM) to a single texture fetch + trilinear
// interpolation. NSight identified the analytical path as the dominant cost
// of the cloud march, so this is the single biggest perf lever in the
// pipeline. Visuals are preserved because the bake evaluated the same
// noise functions on a fine voxel grid.
//
// Bake-side periods (must match Pass_cloudnoise_bake_v8.hlsl):
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

// quality argument retained for call-site compatibility but ignored —
// the bake stores the 2-octave FBM; per-call quality variation isn't
// possible with a single texture, and the FBM is the only level that
// matters for the cloud body anyway.
inline float CloudWorleyFBM(float3 p, uint quality)
{
    return g_cloudNoise.SampleLevel(g_sampler,
        p * (1.0f / CLOUD_NOISE_G_PERIOD), 0).g;
}

// CloudPerlinWorley drives the primary cloud body silhouette — it's the
// noise that gives cumulus their characteristic clustered-billows shape.
// Two-octave runtime FBM: combine a base tap with a 2× scaled & offset
// tap of the same R channel. The combined result has the frequency
// richness of a real FBM (which is what the analytical version produced)
// while only paying for one extra Texture3D fetch per call — the second
// tap reuses the same texture so there's no extra storage cost.
//
// Per-octave weights mirror the analytical CloudPerlinFBM's 0.5/0.25
// after the +0.5 bias was removed and the weights renormalised so the
// result stays in [0,1]: w0=0.65, w1=0.35. The 2× scale + non-axis-
// aligned offset (11.7, 5.3, 23.9) breaks any visible alignment between
// the two octaves so the combined noise doesn't read as a repeating
// motif. The texture WRAPs so both taps tile cleanly.
inline float CloudPerlinWorley(float3 p)
{
    float3 uvw0 = p * (1.0f / CLOUD_NOISE_R_PERIOD);
    float3 uvw1 = uvw0 * 2.0f + float3(0.117f, 0.053f, 0.239f);
    float  n0   = g_cloudNoise.SampleLevel(g_sampler, uvw0, 0).r;
    float  n1   = g_cloudNoise.SampleLevel(g_sampler, uvw1, 0).r;
    return saturate(n0 * 0.65f + n1 * 0.35f);
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

//------------------------------------------------------------------------------
// PLANET-SCALE COVERAGE MAP — NASA Blue Marble climatology
//------------------------------------------------------------------------------
// Samples g_cloudCoverage (8192×4096 R8 equirectangular luminance map) via the
// planet-radial direction of the sample position. Hardware bilinear filtering
// produces smooth gradients between source texels — no blocky coverage edges.
//
// Combination with the editor's CLOUD_COVERAGE_BASE: the map value (0..1) is
// rescaled around 0.5 so a "typical cloudy" map texel reproduces the user's
// base coverage exactly, and very cloudy texels (storm regions, mapValue→1)
// can double up to fully overcast. Formula:
//     coverage = saturate(base * map * 2.0)
// This keeps the editor slider behaving as "global average target", while the
// map adds per-location continental-scale variation. When the map texture is
// missing the loader binds a 1×1 grey fallback (mapValue = 0.5) so the
// product collapses to `base` and pre-map behaviour is preserved.

// World is the observer's local ENU frame (X=east, Y=up, Z=north at the
// observer's lat/lon on the planet — see ENU_ToWorld in SunSampler_v8.hlsli).
// The equirectangular cloud coverage map is keyed by absolute planet lat/lon,
// not by world-frame direction, so we have to rotate world directions into
// planet-absolute coordinates before doing the equirect math.
//
// Why this matters: without the rotation, the equirect's polar axis IS
// world +Y, and world +Y IS the observer's overhead direction (local up
// at the observer's lat/lon). Looking up therefore hits the equirect
// singularity at every render — manifests as a vertical "ice cream cone"
// pinch at the zenith, visible in the cloud body wherever rays cluster
// near world +Y. With the rotation applied, looking up samples the map
// at the observer's actual lat/lon (a normal map region) and the
// singularity moves to the planet's true geographic poles, which sit
// sideways/below for any non-pole observer.
//
// The ENU→planet rotation uses the observer's lat/lon from the SUN_*
// cbuffer fields. Those values are semantically OBSERVER lat/lon (the
// "Sun" in their name is legacy — the sun direction is *derived* from
// them via standard astronomical formulas), so reusing them here is
// consistent with the rest of the atmosphere pipeline.
inline float3 EnuToPlanetDir(float3 dirEnu)
{
    float L   = SUN_LATITUDE_DEG  * DEG2RAD;
    float lon = SUN_LONGITUDE_DEG * DEG2RAD;
    float sL = sin(L), cL = cos(L);
    float sLon = sin(lon), cLon = cos(lon);

    // ENU basis expressed in planet frame:
    //   up    = radial direction at (lat, lon)
    //   east  = orthogonal to up, in the planet equatorial plane
    //   north = cross(up, east)
    float3 east  = float3(-sLon,       0.0f, cLon);
    float3 up    = float3( cL * cLon,  sL,   cL * sLon);
    float3 north = float3(-sL * cLon,  cL,  -sL * sLon);

    return dirEnu.x * east + dirEnu.y * up + dirEnu.z * north;
}

inline float CloudSampleCoverageMap(float3 P)
{
    // Rotate the cloud sample's world-frame direction into planet-absolute
    // coords, then take the equirectangular UV.
    //   u = longitude wrap: atan2 ∈ [-π, π] → [0, 1]
    //   v = colatitude:     acos(y)/π,  0 at planet north pole, 1 at south
    //
    // Sampler is g_samplerLinearWrap (plain bilinear + wrap, no aniso).
    // The default g_sampler is anisotropic and uses screen-space UV
    // derivatives to size its filter footprint; at the longitude=±180°
    // meridian atan2 makes u jump by ~1.0 between adjacent pixels, which
    // the aniso filter mistakes for an enormous magnification → coarse
    // mip / mis-oriented kernel → visible streak across the seam.
    // Bilinear sidesteps the problem entirely because it ignores
    // derivatives.
    float3 dirEnu    = SafeNormalize(P);
    float3 dirPlanet = EnuToPlanetDir(dirEnu);

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

// Cheap upper-bound estimator. Used by empty-space skipping and by the
// ambient column probe — both want a fast "is there cloud here?" check
// without paying for the full noise pyramid.
float CloudProbeHull(float3 P, float timeSec)
{
    float r = length(P);
    float alt = r - ATMOS_BOTTOM_RADIUS;
    float profile = CloudAltitudeProfile(alt);
    if (profile <= 0.0f) return 0.0f;

    //Coverage = editor base × NASA Blue Marble map at this lon/lat,
    //scaled so map=0.5 reproduces the base (see CloudGlobalCoverage).
    //Gives continental-scale weather variation without the rectangular-
    //grid artefacts of the old procedural weathermap.
    float coverage = CloudGlobalCoverage(P);
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

    //Coverage = editor base × NASA Blue Marble map at this lon/lat,
    //scaled so map=0.5 reproduces base. Gives continental-scale weather
    //variation without the rectangular-grid artefacts of the old
    //procedural weathermap. See CloudGlobalCoverage for the formula.
    float coverage = CloudGlobalCoverage(P);
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

// Eccentricity-scaled variant for the Wrenninge multi-scatter octaves
// (Hillaire 2016 §5.8, the c^n term). Each octave broadens the phase
// function — scale=1 returns the un-modified dual lobe (octave 0),
// scale=0.5 halves both lobe g's (octave 1), scale=0 collapses to
// isotropic. Same weight blend so the back/forward energy ratio stays
// consistent across octaves.
inline float CloudPhaseDirectScaled(float cosT, float eccentricity)
{
    float sIntens = saturate(CLOUD_SILVER_INTENSITY);
    float gFwd = lerp(0.85f, 0.99f - CLOUD_SILVER_SPREAD, sIntens) * eccentricity;
    float wFwd = lerp(0.85f, 0.95f, sIntens);

    float fwd = CloudHG(cosT, gFwd);
    float bwd = CloudHG(cosT, -0.50f * eccentricity);
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
// Dedicated shadow-density function. Mirrors what CloudSampleMaterial
// computes at (quality=0, lodT=1) — same Perlin-Worley base so the
// shadow ray sees the same cloud shape the view ray does, same HF
// erosion strength so silhouettes match — but skips the cauliflower
// additive bumps (Worley at 5x base, +0.045 strength at lodT=1) and
// the final soft density curve. Those passes affect only sub-100m
// surface detail that a 6-tap geometric shadow march can't resolve
// anyway. Saves the cauliflower's CloudWorley call (~30 ops) per
// sample, with no visible change in shadow appearance.
//
// CRITICAL: the base noise MUST stay Perlin-Worley (not a cheaper
// Worley-FBM-only or a hull-only approximation). The previous perf
// optimisation that used a different noise function for shadow rays
// produced phantom self-shadow where the two noise fields disagreed
// — the view said "clear" but the shadow said "blocked", and no
// lighting model could recover the lost energy.
inline float CloudDensityForShadow(float3 P, float timeSec)
{
    float r       = length(P);
    float alt     = r - ATMOS_BOTTOM_RADIUS;
    float profile = CloudAltitudeProfile(alt);
    if (profile <= 0.0f) return 0.0f;

    // MUST sample the same coverage source as the view march (see
    // CloudGlobalCoverage) so the surface shadow matches the visible
    // cloud silhouette. Using a different coverage source here would
    // make shadows appear in places with no overhead clouds and vice
    // versa.
    float coverage = CloudGlobalCoverage(P);
    if (coverage <= 0.0f) return 0.0f;

    float3 wind = float3(CLOUD_WIND_X, 0.0f, CLOUD_WIND_Z) * timeSec;
    float3 q    = P + wind;

    // Perlin-Worley base — same function the view march evaluates.
    // Domain warps are skipped here, matching what the view march does
    // at lodT=1 (warpBlend = saturate(1 - 2.5*lodT) collapses to zero).
    float base = CloudPerlinWorley(q * CLOUD_BASE_FREQ);

    float coverageHull = coverage * profile;
    float d = coverageModulation(coverageHull, base, CLOUD_COVMOD_FILTER_WIDTH);
    if (d <= 0.001f) return 0.0f;

    // HF erosion at the lodT=1 strength used by the view march
    // (hfAmount = CLOUD_HF_AMOUNT * lerp(1.0, 0.6, lodT=1.0) = 0.6 *
    // CLOUD_HF_AMOUNT). Edge-mask gate (1-d)^3 keeps erosion focused on
    // silhouettes; interiors stay solid.
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

// Sun shadow march sampled by every cloud-body sample. Uses
// CloudDensityForShadow so the base noise + macro shape match the
// view march exactly (no phantom self-shadow from divergent noise
// functions). Cheaper than CloudDensity(quality=0, lodT=1) by ~25%
// per tap (skips cauliflower + soft density curve, neither of which
// is resolvable in a 6-sample geometric march).
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
        float  d = CloudDensityForShadow(Q, timeSec);
        tau += d * CLOUD_EXTINCTION * SAMPLE_SEG_KM[i];
        if (tau >= TAU_EARLYOUT) break;
    }
    return tau * CLOUD_SUN_TAU_MULT;
}

// Surface shadow march (called from CloudSunVisibility for ground NEE
// and aerial-perspective per-pixel attenuation). MUST use the real
// noise-driven density, not the 2D hull. The hull reports uniform
// `coverage * profile` throughout the cloud shell — at low coverage
// (e.g. 0.15) that's still 0.15 density EVERYWHERE the ray crosses
// the shell, even where actual cloud cells don't exist. Integrated
// over a long sun-direction path that's `tau = 0.15 * 9 * shell` ≈
// many units, so exp(-tau) collapses to zero and surfaces register
// "fully shadowed" with no actual clouds nearby. CloudDensityForShadow
// uses the same Perlin-Worley noise the cloud body uses, so the
// surface shadow matches the visible cloud silhouette exactly.
//
// Two paths gated by CLOUD_SHADOW_STEPS (editor knob):
//   N == 1: fast path. One sphere intersect at the cloud-layer
//           midpoint + one noise tap, multiplied by an oblique-
//           corrected shell thickness. Skips RayCloudShell and the
//           loop entirely — ~4-5x cheaper than the multi-tap march
//           and visually indistinguishable for the dominant overhead-
//           cumulus shadow case (the integration along the sun ray
//           is dominated by the density at the cloud center; the
//           shell-thickness multiplier captures the rest).
//   N >= 2: multi-tap shell march. Better fidelity at cloud edges
//           and for low sun angles where the noise varies along the
//           path. Used when the user wants softer / more accurate
//           shadows at proportional cost.
float CloudOpticalDepthAlongRay(float3 P, float3 D, float maxLenKm, float timeSec)
{
    int N = max(CLOUD_SHADOW_STEPS, 1);

    if (N == 1)
    {
        // Fast path: single midpoint sample at the cloud-layer
        // mid-altitude along the sun ray. Sphere intersect picks
        // the outward hit of the mid-shell sphere; one noise tap
        // gates the shadow; oblique-path correction (1/cos(sun
        // zenith at Q)) scales the constant shell thickness so
        // low sun = longer (darker) shadow path.
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

        // Constant shell thickness scaled by 1/cos(sun zenith at
        // Q). Floor at 0.15 keeps grazing rays from blowing up.
        float shellTh = CLOUD_LAYER_TOP_KM - CLOUD_LAYER_BOT_KM;
        float cosZ    = saturate(dot(D, SafeNormalize(Q)));
        float pathLen = shellTh / max(cosZ, 0.15f);

        return d * CLOUD_EXTINCTION * pathLen;
    }

    // Multi-sample shell march (N >= 2). Same per-tap cost as
    // the cloud-body sun shadow.
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

    //-- Direct: Beer-Lambert × dual-lobe HG --
    // Dominates the sun-side rim and cloud surface; collapses in deep
    // cores where Tdir → 0.
    float  Tdir   = exp(-sunOD);
    float3 direct = K * Tdir * CloudPhaseDirect(cosTheta);

    //-- Multi-scatter: single Nubis-Evolved term --
    // MS amplitude FOLLOWS the direct illumination via sqrt(Tdir) — a
    // softer-than-Beer attenuation that lets MS reach roughly twice as
    // deep into the cloud as direct (effective extinction halved) while
    // still dying off in the unlit deep core. This is what produces
    // bright white cumulus tops on orbital downward views: the back
    // lobe of CloudPhaseDirect gives weak direct, but MS — proportional
    // to sqrt(Tdir) which is near 1 at the sunlit top — fills in big.
    //
    // heightBias floors at FLOOR so bases retain some MS at h=0 (real
    // cumulus bases are "shaded white" not black). SECONDARY_STRENGTH
    // is the per-shot artist amplitude (0..1 maps to 0.5..1.5x), and
    // CLOUD_MS_STRENGTH is the global multiplier.
    float depthFactor = pow(max(Tdir, 1e-6f), 0.5f);
    float heightBias  = lerp(CLOUD_MS_HEIGHT_FLOOR, 1.0f,
                             pow(saturate(h), 0.5f));
    float msAmount    = depthFactor * heightBias
                      * (0.5f + CLOUD_SECONDARY_STRENGTH)
                      * CLOUD_MS_STRENGTH;
    float3 ms = K * msAmount * CloudPhaseMS(cosTheta);

    float3 inscatter = direct + ms;

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

    return inscatter + ambient;
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

        // Hull-first skip: the cheap 2D coverage * altitude profile
        // gates the expensive 3D noise + sun-shadow march. Most bounce-
        // ray samples land in empty sky and this lets them bail at
        // ~10 ops instead of paying ~6000 ops for the full eval.
        float hull = CloudProbeHull(P, walltime);
        if (hull <= CLOUD_EFFECTIVE_ZERO_DENSITY) continue;

        CloudMaterial m = CloudSampleMaterial(P, walltime, lodT, 0u);
        if (m.density <= CLOUD_EFFECTIVE_ZERO_DENSITY) continue;

        float sigma_t     = m.density * CLOUD_EXTINCTION;
        float sunOD       = CloudOpticalDepthToSun(P, L, walltime);
        float Tdir        = exp(-sunOD);
        float earthShadow = CloudEarthShadowFactor(P, L);

        float3 K = sunIrradiance * sunAtmos * earthShadow * CLOUD_ALBEDO;

        // Direct + single-term Nubis-Evolved MS — same formulation as
        // CloudComputeLighting on the primary path. MS amplitude follows
        // sqrt(Tdir) so cloud tops glow even when the direct back lobe
        // is dim; heightBias floors at FLOOR for lit-grey cumulus bases.
        float depthFactor = pow(max(Tdir, 1e-6f), 0.5f);
        float heightBias  = lerp(CLOUD_MS_HEIGHT_FLOOR, 1.0f,
                                 pow(saturate(m.heightFrac), 0.5f));
        float msAmount    = depthFactor * heightBias
                          * (0.5f + CLOUD_SECONDARY_STRENGTH)
                          * CLOUD_MS_STRENGTH;

        float3 inscatter = K * Tdir * phaseDir
                         + K * msAmount * CloudPhaseMS(cosTheta);

        // Energy-conserving analytical integration (Hillaire 2016 §5.6.3
        // / Wrenninge / Frostbite): integrate the scattered radiance
        // over the segment as (L - L*Tr) / sigma_t. Equivalent to the
        // expanded form (sigma_t * inscatter) * (1-segTr)/sigma_t used
        // here — the inscatter * (1 - segTr) cancellation gives the
        // physically correct accumulation for any segment length.
        float segTr = exp(-sigma_t * ds);
        L_acc   += trCloud * inscatter * (1.0f - segTr);
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

// cloudHitDistKmOut returns the transmittance-weighted mean ray distance
// at which the cloud actually contributes radiance (Hillaire 2016 §5.9.1
// Eq. 21: mean front depth weighted by alpha contribution per sample).
// Zero when no sample contributed weight. Used by Pass_clouds_primary_v8
// to write a representative world-space cloud hit position into the DLSS
// depth / motion-vector inputs so cloud silhouettes reproject correctly
// under camera translation.
float3 EvaluateClouds(float3 V, float3 sunDir, float3 sunIrradiance,
                      out float3 cloudTrOut,
                      out float  cloudHitDistKmOut)
{
    cloudTrOut        = float3(1, 1, 1);
    cloudHitDistKmOut = 0.0f;

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

    // Transmittance-weighted distance accumulator (Hillaire 2016 §5.9.1
    // Eq. 21). Per-sample weight is the alpha contribution of the
    // segment — visibility of this sample (trCloud BEFORE multiplying by
    // segTr) times the absorbed fraction (1 - segTr). The final mean
    // distance is sumHitWT / sumHitW; falls back to 0 when no sample
    // contributed (clear-sky pixel).
    float sumHitW  = 0.0f;
    float sumHitWT = 0.0f;

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

            // Energy-conserving analytical integration: the closed-form
            // integral of inscatter * exp(-sigma_t * x) over [0, stepSize]
            // reduces to inscatter * (1 - segTr) when the inscatter is
            // expressed as the in-scattered radiance per unit length
            // (i.e., already scaled to the local extinction). Stays
            // numerically stable for any sigma_t including small values.
            float segTr = exp(-sigma_t * stepSize);
            L_acc      += trCloud * inscatter * (1.0f - segTr);

            // Accumulate transmittance-weighted distance for the cloud
            // mean depth. Must use trCloud BEFORE the segTr multiply so
            // the weight represents this sample's visibility (not the
            // visibility of the sample after it). Luma weight on trCloud
            // collapses the per-channel transmittance into a scalar.
            {
                float wK = dot(trCloud, float3(0.2126f, 0.7152f, 0.0722f))
                         * (1.0f - segTr);
                sumHitW  += wK;
                sumHitWT += wK * tSample;
            }

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

    cloudTrOut        = trCloud;
    cloudHitDistKmOut = (sumHitW > 1e-5f) ? (sumHitWT / sumHitW) : 0.0f;
    return L_acc;
#endif
}

//------------------------------------------------------------------------------
// UNIFIED ATMOSPHERE + CLOUD MARCH
//------------------------------------------------------------------------------
// Single integration loop that handles both atmospheric scattering and
// cloud scattering against the SAME combined extinction. Replaces the
// "atmosphere first, then composite cloud" pipeline (which double-attenuates
// in some configurations and never correctly attenuates atmosphere by cloud
// shadow, or vice versa).
//
// The ray is split into three legs based on cloud-shell intersection:
//
//   tStart  ─── Phase 1 ───►  tC0  ─── Phase 2 ───►  tC1  ─── Phase 3 ───► tEnd
//                                  (combined march          (atmosphere only)
//                                   with hull skip)
//   (atmosphere only)
//
// Each phase shares the running (inscatter, transmittance) state so the
// total integral is consistent. Inside the cloud shell every step samples
// BOTH atmosphere (Rayleigh + Mie + ozone) AND cloud, with combined
// extinction sigma_t_total = atmos_extinction + cloud_density * CLOUD_EXTINCTION
// and a per-channel analytic segment integration
//   delta_L = T * (atmos_rate + cloud_rate) * (1 - exp(-sigma_t * ds)) / sigma_t
// where atmos_rate = scatterPhase * sun_illum (per channel) and
// cloud_rate = cloudRadiance * cloud_sigma_t (cloud albedo ≈ 1 collapses
// sigma_s → sigma_t there). Cloud sample lighting reuses CloudComputeLighting
// with PER-SAMPLE atmospheric sun transmittance (not the shared midpoint
// approximation EvaluateClouds used), so clouds at low sun get correct
// reddish lighting from the wavelength-tinted atmospheric sun column.
//
// What this fixes naturally vs the old composite:
//   - aerial perspective accumulates THROUGH the cloud (not just before)
//   - atmospheric scatter behind a cloud is correctly attenuated by cloud T
//   - sun reaching cloud samples is attenuated by both atmosphere AND
//     cloud column, so low-sun clouds pick up the orange tint
//   - the "observer in tiny cloud shadow → entire sky goes dark" bug from
//     the old observer-shadow proxy doesn't exist here because every
//     per-ray attenuation is intrinsic to that ray's integral
//
// Returns the scattered radiance (atmosphere + cloud); the caller adds
// the planet body / stars / nightBase × combined_T separately. The combined
// transmittance is what attenuates the sun disc, planet, and any other
// background behind the ray.

// Atmospheric-only segment integration with running state. Body is a
// linear-stepped version of IntegrateScattering, parameterized by [tStart,
// tEnd] so it can be called for the pre-cloud and post-cloud phases of
// the unified march. ATMOS_MULTI_SCATTER_FACTOR is folded into the per-
// step rate so the final result is in absolute luminance (assuming the
// caller passed sunIrradiance = ATMOS_SOLAR_IRRADIANCE * SKY_INTENSITY).
inline void IntegrateAtmosphereSegment(
    float3 O, float3 V, float3 L,
    float tStart, float tEnd, int N,
    float cosTheta, float phR, float phM,
    float3 sunIrradiance,
    inout float3 inscatter,
    inout float3 transmittance)
{
    if (tEnd <= tStart || N <= 0) return;
    float ds = (tEnd - tStart) / (float)N;

    [loop]
    for (int i = 0; i < N; ++i)
    {
        float t   = tStart + ((float)i + 0.5f) * ds;
        float3 P  = O + V * t;
        float alt = max(0.0f, length(P) - ATMOS_BOTTOM_RADIUS);

        MediumSample med  = SampleMedium(alt);
        float3       segTr = exp(-med.extinction * ds);
        float3       sunTr = TransmittanceToSun(P, L,
                                                ATMOS_BOTTOM_RADIUS,
                                                ATMOS_TOP_RADIUS);

        float3 Pnorm      = SafeNormalize(P);
        float  sunCosZ    = dot(Pnorm, L);
        float  cosHorizon = -sqrt(max(0.0f, 1.0f -
                                      (ATMOS_BOTTOM_RADIUS * ATMOS_BOTTOM_RADIUS)
                                      / dot(P, P)));
        float  earthShadow = (sunCosZ > cosHorizon) ? 1.0f : 0.0f;

        float3 scatterPhase = med.scatterR * phR + med.scatterM * phM;
        float3 sunIllum     = sunIrradiance * earthShadow * sunTr
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

float3 EvaluateAtmosphereAndClouds(
    float3 V, float3 sunDir, float3 sunIrradiance,
    out float3 transmittanceOut,
    out bool   hitPlanetOut,
    out float  cloudHitDistKmOut)
{
    transmittanceOut  = float3(1, 1, 1);
    hitPlanetOut      = false;
    cloudHitDistKmOut = 0.0f;

    float3 O  = g_skyObserverPlanet;
    float3 Vn = SafeNormalize(V);
    float3 L  = SafeNormalize(sunDir);

    // Ray vs atmosphere top
    float tA0, tA1;
    if (!RaySphereIntersect(O, Vn, ATMOS_TOP_RADIUS, tA0, tA1) || tA1 <= 0.0f)
        return float3(0, 0, 0);

    float tStart = max(0.0f, tA0);
    float tEnd   = tA1;

    // Planet hit clips the end of the ray
    float tG0, tG1;
    if (RaySphereIntersect(O, Vn, ATMOS_BOTTOM_RADIUS, tG0, tG1)
        && tG0 > 0.0f && tG0 < tEnd)
    {
        tEnd         = tG0;
        hitPlanetOut = true;
    }
    if (tEnd <= tStart) return float3(0, 0, 0);

    // Cloud shell intersection — clipped to [tStart, tEnd]
    float tC0 = 0.0f, tC1 = 0.0f;
    bool hasCloud = (cloud_enabled >= 0.5f)
                  && RayCloudShell(O, Vn, tC0, tC1);
    float limbFade = 1.0f;
    if (hasCloud)
    {
        limbFade = CloudHorizonFade(O, Vn);
        if (limbFade <= 0.0f) { hasCloud = false; }
        else
        {
            tC0 = max(tC0, tStart);
            tC1 = min(tC1, tEnd);
            if (tC0 >= tC1) hasCloud = false;
        }
    }

    // Phase function cosines for the combined march
    float cosTheta = dot(Vn, L);
    float phR      = PhaseRayleigh(cosTheta);
    float phM      = PhaseMieTwoLobe(cosTheta);

    float3 inscatter     = float3(0, 0, 0);
    float3 transmittance = float3(1, 1, 1);

    if (!hasCloud)
    {
        // No cloud crossing — single atmospheric integration over the full
        // ray. Same step budget as the standalone IntegrateScattering call.
        IntegrateAtmosphereSegment(O, Vn, L, tStart, tEnd, ATMOS_VIEW_STEPS,
                                   cosTheta, phR, phM, sunIrradiance,
                                   inscatter, transmittance);
        transmittanceOut = transmittance;
        return inscatter;
    }

    // Distribute the atmospheric step budget between phase 1 and phase 3
    // proportionally to their lengths, with a floor of 2 per phase.
    float p1Len         = tC0 - tStart;
    float p3Len         = tEnd - tC1;
    float totalAtmosLen = max(1e-6f, p1Len + p3Len);
    int   N1            = max(2, (int)((float)ATMOS_VIEW_STEPS * p1Len / totalAtmosLen + 0.5f));
    int   N3            = max(2, (int)((float)ATMOS_VIEW_STEPS * p3Len / totalAtmosLen + 0.5f));

    // === Phase 1 — observer to cloud entry, atmosphere only ===
    IntegrateAtmosphereSegment(O, Vn, L, tStart, tC0, N1,
                               cosTheta, phR, phM, sunIrradiance,
                               inscatter, transmittance);

    // === Sky ambient + ground bounce probes for cloud lighting ===
    // Same relocation pattern as the standalone EvaluateClouds — sample
    // the atmospheric scatter from above the cloud layer so orbital
    // observers don't see "near-zero atmosphere" at the cloud top.
    float3 obsUp          = SafeNormalize(O);
    float3 cloudProbePos  = obsUp * (ATMOS_BOTTOM_RADIUS + CLOUD_LAYER_TOP_KM + 0.5f);
    float3 savedObserver  = g_skyObserverPlanet;
    g_skyObserverPlanet   = cloudProbePos;

    float3 skyAmbientTop     = float3(0, 0, 0);
    float3 skyAmbientHorizon = float3(0, 0, 0);
    if (CLOUD_SKY_AMBIENT > 0.5f)
    {
        float3 horizonDir;
        {
            float3 sunHoriz = L - obsUp * dot(L, obsUp);
            float  shLen2   = dot(sunHoriz, sunHoriz);
            float3 perp;
            if (shLen2 > 1e-4f)
                perp = SafeNormalize(cross(obsUp, sunHoriz));
            else
            {
                float3 fwd = Vn - obsUp * dot(Vn, obsUp);
                perp = (dot(fwd, fwd) > 1e-4f) ? SafeNormalize(fwd) : float3(1, 0, 0);
            }
            horizonDir = SafeNormalize(perp * 0.95f + obsUp * 0.05f);
        }
        float3 trZ; bool hitPZ;
        float3 scatterTop     = IntegrateScattering(obsUp,      L, trZ, hitPZ);
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
        float  sunCosUp   = saturate(dot(L, obsUp));
        float3 sunGroundT = TransmittanceToSun(cloudProbePos, L,
                                               ATMOS_BOTTOM_RADIUS,
                                               ATMOS_TOP_RADIUS);
        groundIrrad = sunIrradiance * sunGroundT * sunCosUp
                    * CLOUD_GROUND_ALBEDO * CLOUD_GROUND_SCALE
                    * (1.0f / PI);
    }

    g_skyObserverPlanet = savedObserver;

    // === Phase 2 — combined atmosphere + cloud march through cloud shell ===
    {
        uint2 pixel    = DispatchRaysIndex().xy;
        uint  frame    = (uint)time;
        uint  seed     = initRandomData(pixel, uint2(0, 0), frame, 71u);

        float sumHitW  = 0.0f;
        float sumHitWT = 0.0f;
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

            // Always sample atmospheric medium
            float alt           = max(0.0f, length(P) - ATMOS_BOTTOM_RADIUS);
            MediumSample atmosMed = SampleMedium(alt);

            // Hull check — cheap 2D cloud presence
            float hull       = (distFade > 0.0f) ? CloudProbeHull(P, walltime) * distFade : 0.0f;
            bool  hullEmpty  = (hull <= CLOUD_EFFECTIVE_ZERO_DENSITY);

            CloudMaterial m = (CloudMaterial)0;
            m.heightFrac    = saturate((alt - CLOUD_LAYER_BOT_KM)
                                       / max(1e-4f, CLOUD_LAYER_TOP_KM - CLOUD_LAYER_BOT_KM));
            float cloudDensity = 0.0f;
            if (!hullEmpty)
            {
                m            = CloudSampleMaterial(P, walltime, lodT, 0u);
                cloudDensity = m.density * distFade * limbFade;
            }
            float cloudSigmaT = cloudDensity * CLOUD_EXTINCTION;

            // Combined extinction (atmospheric is per-channel; cloud is scalar)
            float3 sigma_t_total = atmosMed.extinction
                                 + float3(cloudSigmaT, cloudSigmaT, cloudSigmaT);

            // Step length: fine inside density, coarse when hull is empty
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

            // Sun transmittance through atmosphere at this sample
            float3 sunAtmosT = TransmittanceToSun(P, L,
                                                  ATMOS_BOTTOM_RADIUS,
                                                  ATMOS_TOP_RADIUS);

            // Earth shadow (sun below local horizon → dim)
            float3 Pnorm      = SafeNormalize(P);
            float  sunCosZ    = dot(Pnorm, L);
            float  cosHorizon = -sqrt(max(0.0f, 1.0f -
                                          (ATMOS_BOTTOM_RADIUS * ATMOS_BOTTOM_RADIUS)
                                          / dot(P, P)));
            float  earthShadow = (sunCosZ > cosHorizon) ? 1.0f : 0.0f;

            // Atmospheric in-scatter rate (per unit length, per channel)
            float3 atmosScatterPhase = atmosMed.scatterR * phR + atmosMed.scatterM * phM;
            float3 atmosSunIllum     = sunIrradiance * earthShadow * sunAtmosT
                                     * ATMOS_MULTI_SCATTER_FACTOR;
            float3 atmosRate         = atmosScatterPhase * atmosSunIllum;

            // Cloud in-scatter rate — radiance × cloud sigma_s (≈ sigma_t for
            // cloud albedo ≈ 1). CloudComputeLighting handles its own
            // 6-tap sun shadow march + Wrenninge octaves + ambient floor,
            // using the per-sample atmospheric sun transmittance we computed.
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

            // Per-channel analytic integration with combined sigma_t
            float3 scatterInteg;
            scatterInteg.x = (sigma_t_total.x > 1e-10f)
                ? totalRate.x * (1.0f - segTr.x) / sigma_t_total.x : totalRate.x * thisStep;
            scatterInteg.y = (sigma_t_total.y > 1e-10f)
                ? totalRate.y * (1.0f - segTr.y) / sigma_t_total.y : totalRate.y * thisStep;
            scatterInteg.z = (sigma_t_total.z > 1e-10f)
                ? totalRate.z * (1.0f - segTr.z) / sigma_t_total.z : totalRate.z * thisStep;

            inscatter += transmittance * scatterInteg;

            // Cloud hit-distance accumulator (Frostbite §5.9.1) for DLSS depth.
            // Weight by luma transmittance × cloud-only alpha, so the mean
            // depth is biased toward the visible cloud surface.
            if (cloudDensity > 0.0f)
            {
                float trLuma     = dot(transmittance, float3(0.2126f, 0.7152f, 0.0722f));
                float cloudAlpha = 1.0f - exp(-cloudSigmaT * thisStep);
                float wK         = trLuma * cloudAlpha;
                sumHitW  += wK;
                sumHitWT += wK * tSample;
            }

            transmittance *= segTr;

            // Advance and grow fine-step stride only when we're in density
            if (!hullEmpty)
                stepSize = min(stepSize * CLOUD_STEP_GROWTH, CLOUD_MAX_FINE_STEP_KM);
            t += thisStep;
        }

        if (sumHitW > 1e-5f)
            cloudHitDistKmOut = sumHitWT / sumHitW;
    }

    // === Phase 3 — cloud exit to atmosphere top / planet, atmosphere only ===
    if (tC1 < tEnd)
    {
        IntegrateAtmosphereSegment(O, Vn, L, tC1, tEnd, N3,
                                   cosTheta, phR, phM, sunIrradiance,
                                   inscatter, transmittance);
    }

    transmittanceOut = transmittance;
    return inscatter;
}

#endif
