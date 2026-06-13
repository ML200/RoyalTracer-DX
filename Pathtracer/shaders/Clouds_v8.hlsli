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

//====================================
//ADAPTIVE MARCH DISTANCE SCALING
//====================================
//Distance-proportional step floors for the cloud marches. At distance t the
//world-space size of a pixel grows linearly, so the step can too without
//visible undersampling: FRAC_FINE = 0.004 gives 1 km steps at 250 km
//(~2 px at 1080p / 60 deg FOV) while leaving the near field at
//CLOUD_TARGET_STEP_KM untouched. FRAC_EMPTY strides empty air at 2% of
//distance. Combined with the half-budget reach floor in the march loops
//(spend the second half of the iteration budget evenly over the remaining
//segment), the march always reaches the shell exit — clouds no longer
//hard-cut at the horizon when the step budget runs out, they just get
//coarser with distance. Compile-time on purpose: these define the
//resolution model, not an artistic look.
#define CLOUD_STEP_DIST_FRAC_FINE  0.004f
#define CLOUD_STEP_DIST_FRAC_EMPTY 0.02f

//====================================
//NUBIS3 LIGHT ENERGY — exact model from A. Schneider (Guerrilla),
//"Nubis3: Methods (and madness) to model and render immersive real-time
//voxel-based clouds", SIGGRAPH 2023 Advances in Real-Time Rendering;
//direct-scattering structure from "Nubis, Evolved", SIGGRAPH 2022.
//====================================
//  Light Energy = Direct Scattering + Ambient Scattering
//                 (+ Secondary Scattering = in-cloud light sources,
//                    i.e. lightning — no such lights here, omitted)
//
//  Direct  = Transmittance * PrimaryPhase + ms_volume * SecondaryPhase
//  T       = exp(-DL)                            [Beer-Lambert, slide 46]
//  ms_volume (slide "Nubis3/Lighting", verbatim):
//      ms_volume  = dimensional_profile;
//      ms_volume *= exp(-DL * Remap(sun_dot, 0.0, 0.9, 0.25,
//                       ValueRemap(cloud_distance, -128.0, 0.0,
//                                  0.05, 0.25)));
//  Ambient (verbatim):
//      ambient_scattering = pow(1.0 - dimensional_profile, 0.5)
//                         * exp(-summed_ambient_density);
//
//Where the cumulus look comes from in this model:
//  DARK EDGES: the dimensional_profile GATE on ms_volume. Thin samples
//  (wisps, eroded rims, crease walls) get no multi-scatter — only the
//  forward-peaked single-scatter phase, which is tiny frontlit — while
//  dense faces keep full MS. Local-density keyed, so the sunlit face
//  never dims below the shadow side (the failure of every sun-OD-keyed
//  powder variant tried 2026-06-11).
//  INNER GLOW: deep backlit samples drop the MS extinction scale from
//  0.25 (the classic Wrenninge/Nubis multi-scatter octave) toward 0.05,
//  letting light flood through thick cores around the sun.
//
//Mapping onto this renderer:
//  DL            -> sunOD (cone-averaged optical depth toward sun; their
//                   summed density samples read radiometrically).
//                   ms_volume gets the PHYSICAL OD (CLOUD_SUN_TAU_MULT
//                   divided back out): the published exp(-DL·scale)
//                   assumes physical depth, and the x2 artistic shadow
//                   deepener squared the attenuation exactly in the
//                   DL 2-8 band where the backlit body glow lives —
//                   that killed the inner glow from outside. The DIRECT
//                   Beer term keeps the multiplier (tuned shadow depth).
//  dimensional_profile -> m.profile (post-covmod macro density INCLUDING
//                   the lobe octave — base composition, like the shape
//                   FBM inside their modeled NVDFs — but pre billow/HF;
//                   their ValueErosion detail never feeds lighting
//                   either. The lobe in the profile is what darkens the
//                   inter-lobe creases through the MS gate).
//  cloud_distance (SDF voxels, -128 = deep inside) -> accumulated
//                   in-cloud view-path length, remapped over
//                   GLOW_DEPTH_KM. Path length overestimates SDF depth
//                   near the far boundary; acceptable without an SDF.
//  sun_dot       -> cosTheta = dot(V, L) (+1 = looking at the sun).
//                   Remaps CLAMPED on both axes (Decima ValueRemap is
//                   clamped; unclamped extrapolation below sun_dot 0
//                   would re-darken frontlit MS — the exact artifact
//                   this model replaces).
//  PrimaryPhase  -> max(HG(g = 0.6), silver_i * HG(0.99 - silver_s)):
//                   the Nubis dual-lobe; rides the existing
//                   silverIntensity/silverSpread sliders (Common.h added
//                   them for this exact formula). SecondaryPhase ->
//                   HG(CLOUD_SECONDARY_G slider, default 0.18).
//  ms_volume     -> additionally scaled by msStrength/4 so the existing
//                   slider keeps working; its default 4.0 = exact Nubis
//                   (scale 1.0).
//
//Sliders made INERT by this model (kept in the cbuffer mirror):
//  msMode (single fixed model now), msHeightFloor (no height bias in
//  Nubis3 — the ambient column owns vertical shaping), secondaryStrength
//  (was the legacy mode-0 amplitude).
//
//Replaces (all REJECTED in 2026-06-11 A/B testing): Beer-powder in every
//scope/ramp/gain variant, Wrenninge 4-octave ladder + phase-eccentricity
//ladder, Nubis-2017 in-scatter depth probability. Nubis3 has NO powder
//term at all — these knobs are gone, not tunable back.
//MS_BRIGHTNESS: Decima's absolute light calibration is NOT in the
//slides (the published equations are structure; their engine runs them
//through a brightness/exposure rig — the 2015/2017 talks carried
//explicit brightness constants). Unity-scaled, the frontlit body sun
//term lands ~2.5-2.9x below this renderer's previously calibrated
//level and bodies read GREY (2026-06-12). 2.5 restores the calibrated
//white body while keeping every Nubis ratio (profile-gated dark edges,
//glow remap, phase shape). Applies to ms_volume only — the direct term
//carries the silver lining, which was already calibrated.
#define CLOUD_N3_PHASE_G        0.6f   // primary forward HG eccentricity
#define CLOUD_N3_MS_BASE        0.25f  // baseline MS extinction scale
#define CLOUD_N3_MS_GLOW        0.05f  // deep backlit MS extinction scale
#define CLOUD_N3_MS_BRIGHTNESS  2.5f   // engine calibration gain on ms
#define CLOUD_N3_GLOW_SUNDOT    0.9f   // sun_dot where glow fully engages
//-128 voxels of a 256^3 NVDF over a ~2-4 km formation is ~1 km, not the
//2 km first guessed — at 2 km the glow scale barely left 0.25 before
//view transmittance killed the samples, so the inner glow never showed.
#define CLOUD_N3_GLOW_DEPTH_KM  1.0f   // in-cloud depth for full glow
//
//Billow: mid-frequency convex bubble structure (the 0.4-1.5 km band real
//cumulus boils at). Bubbles are a union of convex cells: CARVE density at
//Worley cell seams (raw A-channel Worley is 0 at cell centers, 1 at seams
//— ready-made) and BULGE the centers. Height-shaped (flat shaded bases,
//boiling tops) and LOD-tapered like the domain warp.
#define CLOUD_BILLOW_FREQ       (CLOUD_BASE_FREQ * 3.0f)  // ~1 km cells
#define CLOUD_BILLOW_AMOUNT     0.35f  // seam carve depth
#define CLOUD_BILLOW_BULGE      0.12f  // center push-out
#define CLOUD_BILLOW_SHARP      1.5f   // carve falloff (higher = thinner seams)
#define CLOUD_BILLOW_H_LO       0.08f  // height ramp start (heightFrac)
#define CLOUD_BILLOW_H_HI       0.45f  // full strength above this
//
//Lobe: the missing middle octave between the billow cells (~1 km) and the
//HF wisps (~140 m) — without it the billow bubbles read as smooth ovals
//with fuzz on top. Uses the baked G-channel 3-octave inverted-Worley FBM
//(1 at cell centers, 0 at seams — the classic "billowy" cauliflower
//noise, previously only read by the quality!=0 base path), folded into
//the BASE noise BEFORE the coverage modulation as a signed term — the
//HZD/Nubis base composition (their shape FBM's Worley octaves live in
//base_cloud, pre-coverage). (fbm - 0.5) bulges lobe centers OUT past the
//nominal silhouette and carves the seams in; the covmod slope
//(1/filterWidth, ~3.3x at default) amplifies it, so the default amount
//displaces the isosurface by roughly ±200 m at the ~0.4 km lobe scale.
//WHY pre-coverage: the first cut ran post-covmod as an edge-masked carve
//(the billow/HF pattern). That operator is bounded by its own mask —
//d -= A·n·(1-d)² removes at most ~A/4 at the mid-edge — and post-covmod
//the density-to-distance slope is steep enough that 0.05 of d is tens of
//meters of silhouette: invisible. Only a pre-coverage term moves the
//covmod THRESHOLD itself, and it is the only formulation that can bulge
//OUTWARD at all.
//The lobe deliberately FEEDS m.profile (it is base shape, not erosion
//detail — matches the Nubis3 mapping note above): the MS gate then
//darkens the thin inter-lobe creases while bulge cores keep full MS.
//Without that the creases stay bright and the lobes read only in
//silhouette, not in lighting.
//LOD: zero by lodT 0.625 (~195 km at the default 20/300 band), where a
//0.4 km lobe is ~2 px.
#define CLOUD_LOBE_FREQ         (CLOUD_BASE_FREQ * 8.0f)  // ~0.4 km lobes
#define CLOUD_LOBE_AMOUNT       0.35f  // signed base amplitude (~±half this)
#define CLOUD_LOBE_H_LO         0.10f  // height ramp start (heightFrac)
#define CLOUD_LOBE_H_HI         0.50f  // full strength above this

inline float CloudLodT(float distKm)
{
    return saturate((distKm - CLOUD_LOD_NEAR_KM)
                  / max(1e-4f, CLOUD_LOD_FAR_KM - CLOUD_LOD_NEAR_KM));
}

// Ambient-probe LUT (g_cloudAmbientLUT, t50): 128 x 2 RGBA16F baked per
// frame by Pass_skylut_bake_v8.hlsl. Row 0 = zenith-probe sky scatter,
// row 1 = horizon-probe sky scatter, both parameterized by the sun-zenith
// cosine at the cloud-top probe radius. Width must match the bake dispatch.
#define CLOUD_AMBIENT_LUT_SIZE 128.0f

inline float CloudAmbientLutU(float sunCosZ)
{
    // Texel-center inset so cosZ = -1 / +1 land exactly on the edge texels.
    float x = saturate(sunCosZ * 0.5f + 0.5f);
    return (0.5f + x * (CLOUD_AMBIENT_LUT_SIZE - 1.0f)) / CLOUD_AMBIENT_LUT_SIZE;
}

// STBN dims are fixed at bake time (Renderer::BakeCloudSTBNTexture: 128 x 128
// x 64). Compile-time constants instead of a per-call GetDimensions — this
// runs per atmosphere step and per shadow cone tap, and the resinfo was pure
// overhead. The pow2 masking below requires CLOUD_STBN_SIZE_PX stays pow2.
#define CLOUD_STBN_SIZE_PX 128u
#define CLOUD_STBN_SLICES  64u

inline float4 CloudRand4(uint2 px, uint frame, uint tap)
{
#if CLOUD_STBN_AVAILABLE
    uint sliceIdx = (frame + tap * 17u) % CLOUD_STBN_SLICES;
    uint2 puv = uint2((px.x + tap * 113u) & (CLOUD_STBN_SIZE_PX - 1u),
                      (px.y + tap *  71u) & (CLOUD_STBN_SIZE_PX - 1u));
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
#define CLOUD_NOISE_G_PERIOD   16.0f   // inverted-Worley FBM (3 octaves)
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
// the bake stores a single 3-octave inverted-Worley FBM (1 at cell
// centers, 0 at seams).
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

// PLANET v8: smoothed-terrain offset (km) used to shift the cloud layer
// with general terrain elevation so individual mountain peaks poke through
// instead of being buried in cloud. The baker writes a 256^2 per-face
// cloud_offset map (block-averaged elevation in km); TerrainCloudBaseHeight
// returns the same value in METRES via the equiangular projection. Returns 0
// for null SRV (legacy bake), keeping the cloud layer at its constant
// spherical altitude in that case.
//
// LIFT ONLY: clamped to >= 0 so basins (negative bedrock) don't drag the
// cloud layer below CLOUD_LAYER_BOT_KM. Negative shifts would push the
// shell underground in deep-basin areas and visually leave clouds
// hugging the surface across the planet (mean bake elevation is ~+1 km,
// but the global range spans -5..+5 km; without the clamp the entire
// basin half of the planet had clouds clipped against the body).
// Matches the max(TerrainHeight, 0) pattern used by every other terrain
// shadow probe (Clouds_v8.hlsli, SunSampler_v8.hlsli).
inline float CloudLocalBaseShiftKm(float3 P)
{
    float r = length(P);
    if (r <= 1e-6f) return 0.0f;
    return max(TerrainCloudBaseHeight(P / r) * 0.001f, 0.0f);
}

// PLANET v8: expected magnitude of the cloud-base shift. The ray-cloud-shell
// intersection expands by this margin so the marching range still covers
// the per-direction cloud layer when terrain pushes it up. Since the shift
// is clamped to >= 0 (see CloudLocalBaseShiftKm) only the OUTER radius
// actually needs the expansion; the inner expansion is kept symmetric as a
// cheap safety net (a few km of extra empty shell costs nothing because the
// profile gate zeros density outside the layer). 6 km matches the bake's
// ~+5 km max smoothed elevation with a small margin; bump if you tune the
// baker to a planet with higher peaks.
#define CLOUD_TERRAIN_SHIFT_BOUND_KM 6.0f

// Heart-shape vertical profile (Nubis). FromTop variant skips the
// CloudEffectiveTopKm tap when the caller already has effTop.
// PLANET v8: takes the local cloud-base shift so the layer follows
// general terrain. baseShiftKm = 0 reproduces the legacy spherical-shell
// profile exactly.
inline float CloudAltitudeProfileFromTop(float altKm, float effTop, float baseShiftKm)
{
    float botKm = CLOUD_LAYER_BOT_KM + baseShiftKm;
    float topKm = effTop              + baseShiftKm;
    float h = (altKm - botKm) / max(1e-4f, topKm - botKm);
    if (h <= 0.0f || h >= 1.0f) return 0.0f;
    float bottomRamp = smoothstep(0.00f, 0.18f, h);
    float topCap     = 1.0f - smoothstep(0.55f, 1.00f, h);
    return bottomRamp * topCap;
}

inline float CloudAltitudeProfile(float altKm, float3 P)
{
    return CloudAltitudeProfileFromTop(altKm,
                                       CloudEffectiveTopKm(P),
                                       CloudLocalBaseShiftKm(P));
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
//
// Guarded init-once: CloudSunVisibility(Planet) calls this per invocation,
// and the atmosphere segment loops invoke those per STEP — without the
// guard each step re-paid the 4 sincos. The flag is a per-thread register;
// the basis is uniform within a thread's lifetime so once is correct.
static float3 g_cloudEnuEast  = float3(1, 0, 0);
static float3 g_cloudEnuUp    = float3(0, 1, 0);
static float3 g_cloudEnuNorth = float3(0, 0, 1);
static bool   g_cloudEnuInitDone = false;

inline void InitCloudEnuBasis()
{
    if (g_cloudEnuInitDone) return;
    g_cloudEnuInitDone = true;
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

// Conservative altitude band of the whole cloud layer, pure ALU. The local
// shift is clamped >= 0 (CloudLocalBaseShiftKm) so the true bottom is never
// below CLOUD_LAYER_BOT_KM, and the true top never exceeds
// TOP + VARIATION + SHIFT_BOUND. Checking this BEFORE any texture work lets
// the density probes reject out-of-band samples without paying the effTop
// noise tap + cloud-offset cubemap fetch the profile would need.
inline bool CloudAltitudeInBand(float altKm)
{
    return altKm > CLOUD_LAYER_BOT_KM
        && altKm < CLOUD_LAYER_TOP_KM + CLOUD_TOP_VARIATION_KM
                   + CLOUD_TERRAIN_SHIFT_BOUND_KM;
}

// Cheap "is there cloud here?" upper-bound. Exposes the components
// (including baseShiftKm) so a following CloudSampleMaterialFromHull doesn't
// recompute any of the direction-only terms.
// Gate order is cheapest-and-most-selective first: ALU altitude band, then
// coverage (1 fetch + equirect trig — zero exactly where the map is clear),
// then the profile (effTop noise tap + cloud-offset cubemap fetch).
float CloudProbeHullEx(float3 P, float timeSec,
                       out float coverage, out float profile, out float effTop,
                       out float baseShiftKm)
{
    coverage    = 0.0f;
    profile     = 0.0f;
    effTop      = CLOUD_LAYER_TOP_KM;
    baseShiftKm = 0.0f;

    float r   = length(P);
    float alt = r - ATMOS_BOTTOM_RADIUS;
    if (!CloudAltitudeInBand(alt)) return 0.0f;

    coverage = CloudGlobalCoverage(P);
    if (coverage <= 0.0f) return 0.0f;

    effTop      = CloudEffectiveTopKm(P);
    baseShiftKm = CloudLocalBaseShiftKm(P);
    profile     = CloudAltitudeProfileFromTop(alt, effTop, baseShiftKm);
    if (profile <= 0.0f) return 0.0f;

    return coverage * profile;
}

struct CloudMaterial
{
    float density;     // final density after coverage + erosion [0..1]
    float profile;     // macro shape before erosion [0..1]
    float heightFrac;  // position within the layer [0..1]
};

// FromHull: caller passes the hull's (coverage, profile, effTop,
// baseShiftKm) so this skips the redundant equirect fetch, EffectiveTop tap
// AND the cloud-offset cubemap fetch. Requires hull > 0.
CloudMaterial CloudSampleMaterialFromHull(
    float3 P, float timeSec, float lodT, uint quality,
    float coverage, float profile, float effTop, float baseShiftKm)
{
    CloudMaterial m;
    m.density    = 0.0f;
    m.profile    = 0.0f;
    m.heightFrac = 0.0f;

    float alt = length(P) - ATMOS_BOTTOM_RADIUS;
    // heightFrac uses per-cloud effTop so MS height-bias reads correctly
    // for tall vs short cumulus. PLANET v8: shift by the local cloud-base
    // offset so heightFrac stays correctly in [0,1] when the cloud layer
    // is riding above general terrain.
    float botKm = CLOUD_LAYER_BOT_KM + baseShiftKm;
    float topKm = effTop              + baseShiftKm;
    float h     = (alt - botKm) / max(1e-4f, topKm - botKm);
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

    // LOBE: mid-frequency octave folded into the base, PRE-coverage (see
    // the knob block for why the post-covmod carve variant was invisible).
    // Signed: bulges lobe centers out past the nominal silhouette, carves
    // the FBM seams in. Feeds covmod and therefore m.profile by design —
    // the MS gate darkens the creases, the bulge cores stay white.
    if (quality == 0u && CLOUD_LOBE_AMOUNT > 0.001f)
    {
        float lobeShape = saturate(1.0f - lodT * 1.6f)
                        * smoothstep(CLOUD_LOBE_H_LO, CLOUD_LOBE_H_HI,
                                     m.heightFrac);
        if (lobeShape > 0.01f)
        {
            float fbm = CloudWorleyFBM(shapeQ * CLOUD_LOBE_FREQ
                                       + float3(19.3f, 7.7f, 41.9f), 0u);
            base += CLOUD_LOBE_AMOUNT * lobeShape * (fbm - 0.5f);
        }
    }

    float coverageHull = coverage * profile;
    float d = coverageModulation(coverageHull, base, CLOUD_COVMOD_FILTER_WIDTH);
    if (d <= 0.001f) return m;

    m.profile = saturate(d);

    // BILLOW: convex bubble structure in the 0.4-1.5 km band (see the
    // knob block up top). Carve at Worley cell seams — edge-weighted so
    // cores survive and the silhouette scallops into arcs of spheres —
    // and bulge the cell centers, mid-masked. Runs BEFORE the HF erosion
    // so the wisps ride on the scalloped edges (coarse-to-fine order).
    // Replaces the old +/-0.15 symmetric "cauliflower" term, which only
    // undulated the isosurface — a union of convex cells needs the
    // asymmetric carve.
    if (quality == 0u && CLOUD_BILLOW_AMOUNT > 0.001f)
    {
        float bilShape = lerp(1.0f, 0.25f, lodT)
                       * smoothstep(CLOUD_BILLOW_H_LO, CLOUD_BILLOW_H_HI,
                                    m.heightFrac);
        if (bilShape > 0.01f)
        {
            float bil  = CloudWorley(shapeQ * CLOUD_BILLOW_FREQ
                                     + float3(31.7f, 11.9f, 23.1f));
            float bil2 = CloudWorley(shapeQ * CLOUD_BILLOW_FREQ * 2.13f
                                     + float3(7.3f, 27.1f, 13.9f));
            float carve = saturate(0.65f * bil + 0.35f * bil2); // 1 at seams

            float edgeMask = (1.0f - d) * (1.0f - d);
            float midMask  = 1.0f - abs(2.0f * d - 1.0f);
            d = saturate(d
                - CLOUD_BILLOW_AMOUNT * bilShape
                    * pow(carve, CLOUD_BILLOW_SHARP) * edgeMask
                + CLOUD_BILLOW_BULGE * bilShape
                    * (1.0f - carve) * (1.0f - carve) * midMask);
        }
    }

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

    // (Old near-field cauliflower term removed — subsumed by the BILLOW
    // stage above, which carves convex cells instead of wobbling the
    // isosurface symmetrically.)

    // Soft density curve lifts mid-densities so the body fills terrain.
    d = pow(saturate(d), lerp(0.85f, 1.0f, d));

    m.density = d;
    return m;
}

// Wrapper for callers without a precomputed hull.
CloudMaterial CloudSampleMaterial(float3 P, float timeSec, float lodT, uint quality)
{
    float coverage, profile, effTop, baseShiftKm;
    float hull = CloudProbeHullEx(P, timeSec, coverage, profile, effTop,
                                  baseShiftKm);
    if (hull <= 0.0f)
    {
        CloudMaterial m = (CloudMaterial)0;
        return m;
    }
    return CloudSampleMaterialFromHull(P, timeSec, lodT, quality,
                                       coverage, profile, effTop, baseShiftKm);
}


bool RayCloudShell(float3 ro, float3 rd, out float tNear, out float tFar)
{
    tNear = 0.0f;
    tFar  = 0.0f;

    float Rb = ATMOS_BOTTOM_RADIUS;
    // Rt uses the MAX possible cloud top so the shell captures tall cumulus
    // variation; terrain-of-cloud shell samples cost nothing via the profile gate.
    // PLANET v8: extra +/- CLOUD_TERRAIN_SHIFT_BOUND_KM so the shell still
    // wraps the cloud layer when it's been shifted by terrain.
    float Rt = ATMOS_BOTTOM_RADIUS + CLOUD_LAYER_TOP_KM + CLOUD_TOP_VARIATION_KM
                                    + CLOUD_TERRAIN_SHIFT_BOUND_KM;
    float Rm = ATMOS_BOTTOM_RADIUS + CLOUD_LAYER_BOT_KM
                                    - CLOUD_TERRAIN_SHIFT_BOUND_KM;

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

inline float CloudTerrainCosHorizon(float3 P)
{
    return -sqrt(max(0.0f, 1.0f - (ATMOS_BOTTOM_RADIUS * ATMOS_BOTTOM_RADIUS) / dot(P, P)));
}

inline float CloudHG(float cosT, float g)
{
    float g2 = g * g;
    float denom = 1.0f + g2 - 2.0f * g * cosT;
    return (1.0f / (4.0f * PI)) * (1.0f - g2)
         / max(1e-4f, denom * sqrt(max(1e-4f, denom)));
}

// NUBIS dual-lobe primary phase (Schneider 2017, carried through Nubis3):
// forward HG lobe max'd against a narrow "silver" lobe around the sun.
// The CloudHG above matches the HenyeyGreenstein on Nubis Evolved slide
// 48 verbatim (incl. the 1/4pi). silverIntensity/silverSpread sliders
// were added to CloudSettings for exactly this formula.
// (Replaces the Jendersie-d'Eon droplet fit — Nubis3 uses plain HG.)
inline float CloudPhaseNubisPrimary(float cosT)
{
    float fwd = CloudHG(cosT, CLOUD_N3_PHASE_G);
    float slv = CLOUD_SILVER_INTENSITY
              * CloudHG(cosT, 0.99f - CLOUD_SILVER_SPREAD);
    return max(fwd, slv);
}

// Secondary (multi-scattering) phase: broad HG on the secondaryG slider.
inline float CloudPhaseNubisSecondary(float cosT)
{
    return CloudHG(cosT, CLOUD_SECONDARY_G);
}

// Shadow-density variant. MUST use the same Perlin-Worley base + coverage
// source as the view march or shadows desync from the visible cloud
// silhouette. Skips the density curve (sub-100m detail, invisible at
// shadow-march resolution).
//
// withBillow: include the mid-frequency shape terms — the lobe base
// perturbation and the billow seam carve. The SELF-shadow march
// (CloudOpticalDepthToSun — what lights the cloud body) needs them so
// the lobe bulges cast OD and the inter-bubble creases self-shadow —
// Beer crease darkening complements the Nubis3 profile-gated MS dark
// edges.
// The haze / surface visibility taps (CloudSunVisibility*) skip it:
// they are the per-step hot path and 800 m fidelity is irrelevant there.
float CloudDensityForShadow(float3 P, float timeSec, bool withBillow)
{
    // Gate order: ALU band first (no fetches), then coverage (1 fetch),
    // then profile (2 fetches) — see CloudProbeHullEx.
    float alt = length(P) - ATMOS_BOTTOM_RADIUS;
    if (!CloudAltitudeInBand(alt)) return 0.0f;

    float coverage = CloudGlobalCoverage(P);
    if (coverage <= 0.0f) return 0.0f;

    float profile = CloudAltitudeProfile(alt, P);
    if (profile <= 0.0f) return 0.0f;

    float3 q    = CloudNoiseCoord(P, timeSec);
    float  base = CloudPerlinWorley(q * CLOUD_BASE_FREQ);

    // Plain-layer height ramp (no effTop/baseShift taps) shared by the
    // lobe perturbation and the billow carve below.
    float hApx = saturate((alt - CLOUD_LAYER_BOT_KM)
               / max(1e-3f, CLOUD_LAYER_TOP_KM - CLOUD_LAYER_BOT_KM));

    // Lobe base perturbation — MUST mirror the view march's pre-coverage
    // composition: it moves the cloud surface by ~±200 m, so a shadow
    // march without it would cast no OD from the bulges (over-lit) and
    // keep phantom OD in the creases (over-dark). Unwarped q, like the
    // shadow base.
    if (withBillow && CLOUD_LOBE_AMOUNT > 0.001f)
    {
        float lobeShape = smoothstep(CLOUD_LOBE_H_LO, CLOUD_LOBE_H_HI, hApx);
        if (lobeShape > 0.01f)
        {
            float fbm = CloudWorleyFBM(q * CLOUD_LOBE_FREQ
                                       + float3(19.3f, 7.7f, 41.9f), 0u);
            base += CLOUD_LOBE_AMOUNT * lobeShape * (fbm - 0.5f);
        }
    }

    float coverageHull = coverage * profile;
    float d = coverageModulation(coverageHull, base, CLOUD_COVMOD_FILTER_WIDTH);
    if (d <= 0.001f) return 0.0f;

    if (withBillow && CLOUD_BILLOW_AMOUNT > 0.001f)
    {
        // Mirrors the view-march carve at shadow resolution: unwarped q
        // (precedent: the shadow base is unwarped too), seams only — the
        // bulge doesn't move shadows at this resolution.
        float amt  = CLOUD_BILLOW_AMOUNT
                   * smoothstep(CLOUD_BILLOW_H_LO, CLOUD_BILLOW_H_HI, hApx);
        if (amt > 0.01f)
        {
            float bil      = CloudWorley(q * CLOUD_BILLOW_FREQ
                                         + float3(31.7f, 11.9f, 23.1f));
            float edgeMask = (1.0f - d) * (1.0f - d);
            d = saturate(d - amt * pow(bil, CLOUD_BILLOW_SHARP) * edgeMask);
        }
    }

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

// Billow-free overload for the haze / surface visibility taps.
float CloudDensityForShadow(float3 P, float timeSec)
{
    return CloudDensityForShadow(P, timeSec, false);
}

// Macro-shape only (no HF / cauliflower) — DLSS RR normal needs to be
// stable at the ~1km feature scale, not jittered by 100m detail noise.
inline float CloudShapeDensity(float3 P, float timeSec)
{
    // Same cheapest-first gate order as CloudDensityForShadow.
    float alt = length(P) - ATMOS_BOTTOM_RADIUS;
    if (!CloudAltitudeInBand(alt)) return 0.0f;

    float coverage = CloudGlobalCoverage(P);
    if (coverage <= 0.0f) return 0.0f;

    float profile = CloudAltitudeProfile(alt, P);
    if (profile <= 0.0f) return 0.0f;

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

    // ENU basis only — this march never evaluates a phase function.
    InitCloudEnuBasis();

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

// Per-sample sun shadow march. Geometric ramp terrain to ~380 km so low-sun
// rays reach distant cloud banks; TAU_EARLYOUT keeps noon rays cheap.
// After the local march, a geometric large-scale term estimates occlusion
// from the far-side cloud-shell passage (the part of the sun ray that
// re-enters the shell on the opposite side of the planet after dipping
// below the cloud layer). Catches planet-scale blocking at sunrise/sunset
// without extra density samples.
// Tallest terrain that can occlude a sun ray, conservative. Raw peaks exceed
// the smoothed CLOUD_TERRAIN_SHIFT_BOUND_KM map; 10 km clears any plausible
// bake. Lets the shadow march skip the heightmap fetch for samples that are
// geometrically above all terrain.
#define CLOUD_TERRAIN_OCCLUDER_MAX_KM 10.0f

float CloudOpticalDepthToSun(float3 P, float3 L, float timeSec)
{
    const float SAMPLE_DIST_KM[12] = { 0.2f, 0.6f, 1.5f, 3.5f, 7.0f, 12.0f, 22.0f, 40.0f, 70.0f, 125.0f, 220.0f, 380.0f };
    const float SAMPLE_SEG_KM[12]  = { 0.2f, 0.4f, 1.1f, 2.0f, 3.5f,  5.0f, 10.0f, 18.0f, 30.0f,  50.0f,  90.0f, 150.0f };
    const float TAU_EARLYOUT = 64.0f;

    // r(t) along the ray has derivative sign(dot(Q, L)) = sign(dotPL + t):
    // once a sample is outside the max shell top AND moving radially
    // outward, every later (farther) sample is higher still — the rest of
    // the table is guaranteed zero. For mid/high sun this cuts the 12
    // samples to the ~5 that can actually intersect the layer.
    const float shellTopR = ATMOS_BOTTOM_RADIUS + CLOUD_LAYER_TOP_KM
                          + CLOUD_TOP_VARIATION_KM + CLOUD_TERRAIN_SHIFT_BOUND_KM;
    const float terrainMaxR = ATMOS_BOTTOM_RADIUS + CLOUD_TERRAIN_OCCLUDER_MAX_KM;
    const float dotPL = dot(P, L);

    float tau = 0.0f;
    [loop]
    for (int i = 0; i < 12; ++i)
    {
        float  dist = SAMPLE_DIST_KM[i];
        float3 Q  = P + L * dist;
        float  rQ = length(Q);
        if (rQ >= shellTopR && dotPL + dist > 0.0f) break;

        // Terrain block only where terrain can exist — the heightmap fetch
        // (cubemap sample + equiangular projection) is pointless at 70+ km.
        if (rQ < terrainMaxR)
        {
            float terrainR = ATMOS_BOTTOM_RADIUS + max(TerrainHeight(Q / rQ) * 0.001f, 0.0f);
            if (rQ < terrainR) { tau = TAU_EARLYOUT; break; }
        }

        // withBillow: this march lights the cloud body itself — the
        // inter-bubble creases must self-shadow (see CloudDensityForShadow).
        float  d = CloudDensityForShadow(Q, timeSec, true);
        tau += d * CLOUD_EXTINCTION * SAMPLE_SEG_KM[i];
        if (tau >= TAU_EARLYOUT) break;
    }

    if (tau < TAU_EARLYOUT)
    {
        // PLANET v8: +/-CLOUD_TERRAIN_SHIFT_BOUND_KM so the long-distance
        // shadow probe stays inside the shifted cloud shell at any terrain.
        float Rt = ATMOS_BOTTOM_RADIUS + CLOUD_LAYER_TOP_KM + CLOUD_TOP_VARIATION_KM
                                       + CLOUD_TERRAIN_SHIFT_BOUND_KM;
        float Rm = ATMOS_BOTTOM_RADIUS + CLOUD_LAYER_BOT_KM
                                       - CLOUD_TERRAIN_SHIFT_BOUND_KM;

        float tT0, tT1, tM0, tM1;
        bool hitOuter = RaySphereIntersect(P, L, Rt, tT0, tT1);
        bool hitInner = RaySphereIntersect(P, L, Rm, tM0, tM1);

        if (hitOuter && tT1 > 0.0f && hitInner && tM0 > 0.0f)
        {
            float tG0, tG1;
            bool hitGround = RaySphereIntersect(P, L, ATMOS_BOTTOM_RADIUS, tG0, tG1);

            if (!hitGround || tG0 <= 0.0f)
            {
                float3 Pmid = P + L * (0.5f * (tM1 + tT1));
                float  cov  = CloudGlobalCoverage(Pmid);
                if (cov > CLOUD_COVERAGE_BASE)
                    tau += cov * TAU_EARLYOUT;
            }
        }
    }

    return tau * CLOUD_SUN_TAU_MULT;
}

// Distance along a sun ray to the outer cloud-shell exit, capped to the
// same ~380 km reach the table-based CloudOpticalDepthToSun uses. This is
// the maxLenKm the CloudSunVisibility* callers must pass to
// CloudOpticalDepthAlongRay: the old fixed 50 km cap silently dropped the
// deck once the sun fell below ~3° elevation (the oblique in-shell path
// from a below-deck sample exceeds 50 km there), popping under-deck haze
// and surfaces to fully sunlit at exactly the sun angles where the
// overcast horizon look matters most. Callers sit at or below the shell
// (they early-out above it), so the ray always exits through Rt and
// tT1 > 0 holds.
inline float CloudShadowReachKm(float3 P, float3 D)
{
    float Rt = ATMOS_BOTTOM_RADIUS + CLOUD_LAYER_TOP_KM + CLOUD_TOP_VARIATION_KM
                                   + CLOUD_TERRAIN_SHIFT_BOUND_KM;
    float tT0, tT1;
    if (!RaySphereIntersect(P, D, Rt, tT0, tT1) || tT1 <= 0.0f) return 0.0f;
    return min(tT1, 380.0f);
}

// Surface shadow march (ground NEE + aerial perspective). MUST use the
// real noise density — the 2D hull is uniform across the shell and
// over-darkens surfaces where no cloud actually sits overhead.
// CLOUD_SHADOW_STEPS=1 takes a single-sample fast path (~4x cheaper).
// N≥2 walks the shell. maxLenKm comes from CloudShadowReachKm —
// geometric, not a tuning knob.
//
// xiAlt picks WHICH altitude slice of the layer the fast path samples
// (0 = layer bottom, 1 = layer top; 0.5 = the old fixed midpoint).
// A DETERMINISTIC slice is a ghost surface: the sample always lies on one
// world-space sphere, so the Perlin-Worley pattern of that single slice
// gets painted onto every haze/fog sample's shadow and parallaxes as a
// sphere DECOUPLED from the visible clouds (dense cloud samples use the
// table march anchored to the sample, so the cloud bodies themselves look
// right — which is what made the sphere read as a separate layer).
// Jittering xiAlt per tap (STBN at the integrator call sites) turns the
// slice into an unbiased estimate of the full column: pathLen is the slab
// path, the slice average over xi is the slab's mean density, and DLSS
// resolves the per-pixel slice noise to the true column shadow.
float CloudOpticalDepthAlongRay(float3 P, float3 D, float maxLenKm, float timeSec,
                                float xiAlt)
{
    int N = max(CLOUD_SHADOW_STEPS, 1);

    if (N == 1)
    {
        // One slice sample × oblique-path correction (1/cos sun zenith).
        float sliceAlt = lerp(CLOUD_LAYER_BOT_KM, CLOUD_LAYER_TOP_KM,
                              saturate(xiAlt));
        float Rslice   = ATMOS_BOTTOM_RADIUS + sliceAlt;
        float r      = length(P);
        float dotPD  = dot(P, D);
        float disc   = dotPD * dotPD - (r * r - Rslice * Rslice);
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

// Stratified 3-slice column OD for SURFACE sun shadows. A single-slice
// estimator is unbiased in OD, but the surface lighting averages
// exp(-tau) across NEE samples, and E[exp(-tau_slice)] >> exp(-tau_column)
// when slices vary (Jensen, exp is convex): every jitter landing in the
// heart profile's transparent bottom ramp / top cap reported "fully lit",
// and those samples dominated the average — a thick cumulus converged to
// ~15% direct sun and object shadows survived under overcast. Three
// stratified slices (one per third of the layer, rotated by xiAlt per NEE
// sample) collapse the variance so the exp bias is negligible, sum in
// OPTICAL DEPTH before the single exp, and stay anchored to the actual
// cloud along the sun ray — shadow opacity now grows exponentially with
// thickness, which is exactly how fast real object shadows vanish under a
// thickening deck.
float CloudSunOpticalDepthColumn(float3 P, float3 L, float maxLenKm,
                                 float timeSec, float xiAlt)
{
    float r       = length(P);
    float dotPL   = dot(P, L);
    float shellTh = CLOUD_LAYER_TOP_KM - CLOUD_LAYER_BOT_KM;

    float tau = 0.0f;
    [unroll]
    for (int i = 0; i < 3; ++i)
    {
        float hFrac = ((float)i + saturate(xiAlt)) * (1.0f / 3.0f);
        float Rs    = ATMOS_BOTTOM_RADIUS
                    + lerp(CLOUD_LAYER_BOT_KM, CLOUD_LAYER_TOP_KM, hFrac);
        float disc  = dotPL * dotPL - (r * r - Rs * Rs);
        if (disc < 0.0f) continue;
        float tMid  = -dotPL + sqrt(disc);
        if (tMid <= 0.0f || tMid > maxLenKm) continue;

        float3 Q    = P + L * tMid;
        float  d    = CloudDensityForShadow(Q, timeSec);
        float  cosZ = saturate(dot(L, SafeNormalize(Q)));
        // Each slice stands in for its third of the slab path.
        tau += d * CLOUD_EXTINCTION
             * (shellTh / (3.0f * max(cosZ, 0.15f)));
    }
    return tau;
}

// xiAlt rotates the shadow strata (see CloudSunOpticalDepthColumn);
// surface NEE passes a fresh rand per call, the 2-arg overload uses the
// deterministic midpoints.
float CloudSunVisibility(float3 surfacePosWorld, float3 sunDirWS, float xiAlt)
{
#if !ENABLE_CLOUDS
    return 1.0f;
#else
    if (cloud_enabled < 0.5f) return 1.0f;
    InitCloudEnuBasis();

    float3 P = WorldToPlanet(surfacePosWorld);
    float  r = length(P);
    // MAX top so tall cumulus still cast when observer is above baseline.
    // PLANET v8: include terrain-shift margin in the "above cloud" early-out.
    if (r - ATMOS_BOTTOM_RADIUS > CLOUD_LAYER_TOP_KM + CLOUD_TOP_VARIATION_KM
                                  + CLOUD_TERRAIN_SHIFT_BOUND_KM + 1e-3f) return 1.0f;

    float3 L = SafeNormalize(sunDirWS);
    float tau = CloudSunOpticalDepthColumn(P, L, CloudShadowReachKm(P, L),
                                           walltime, xiAlt);
    return exp(-tau);
#endif
}

float CloudSunVisibility(float3 surfacePosWorld, float3 sunDirWS)
{
    return CloudSunVisibility(surfacePosWorld, sunDirWS, 0.5f);
}

// Planet-space variant — skips WorldToPlanet's flat-earth reprojection,
// which doesn't round-trip spherical-projection inputs (cloud foot on
// planet surface). xiAlt = altitude-slice jitter for the fast-path shadow
// tap (see CloudOpticalDepthAlongRay) — pass per-step STBN; the 2-arg
// overload keeps the deterministic mid-slice for callers without a rand.
float CloudSunVisibilityPlanet(float3 Pplanet, float3 sunDirWS, float xiAlt)
{
#if !ENABLE_CLOUDS
    return 1.0f;
#else
    if (cloud_enabled < 0.5f) return 1.0f;
    InitCloudEnuBasis();

    float3 L = SafeNormalize(sunDirWS);
    float  r = length(Pplanet);
    // PLANET v8: include terrain-shift margin in the "above cloud" early-out.
    if (r - ATMOS_BOTTOM_RADIUS > CLOUD_LAYER_TOP_KM + CLOUD_TOP_VARIATION_KM
                                  + CLOUD_TERRAIN_SHIFT_BOUND_KM + 1e-3f) return 1.0f;

    float tau = CloudOpticalDepthAlongRay(Pplanet, L, CloudShadowReachKm(Pplanet, L),
                                          walltime, xiAlt);

    // Far-side cloud shell blocking (planet-scale sunset occlusion).
    if (tau < 64.0f)
    {
        // PLANET v8: +/-CLOUD_TERRAIN_SHIFT_BOUND_KM as elsewhere so the
        // limb-crossing shell still wraps the terrain-shifted cloud layer.
        float Rt = ATMOS_BOTTOM_RADIUS + CLOUD_LAYER_TOP_KM + CLOUD_TOP_VARIATION_KM
                                       + CLOUD_TERRAIN_SHIFT_BOUND_KM;
        float Rm = ATMOS_BOTTOM_RADIUS + CLOUD_LAYER_BOT_KM
                                       - CLOUD_TERRAIN_SHIFT_BOUND_KM;
        float tT0, tT1, tM0, tM1;
        bool hitOuter = RaySphereIntersect(Pplanet, L, Rt, tT0, tT1);
        bool hitInner = RaySphereIntersect(Pplanet, L, Rm, tM0, tM1);
        if (hitOuter && tT1 > 0.0f && hitInner && tM0 > 0.0f)
        {
            float tG0, tG1;
            bool hitGround = RaySphereIntersect(Pplanet, L, ATMOS_BOTTOM_RADIUS, tG0, tG1);
            if (!hitGround || tG0 <= 0.0f)
            {
                float3 Pmid = Pplanet + L * (0.5f * (tM1 + tT1));
                float  cov  = CloudGlobalCoverage(Pmid);
                if (cov > CLOUD_COVERAGE_BASE)
                    tau += cov * 64.0f;
            }
        }
    }

    return exp(-tau);
#endif
}

// Deterministic mid-slice overload for callers without a per-step rand
// (bounce-miss sky in IntegrateScattering via the SunSampler forward decl).
float CloudSunVisibilityPlanet(float3 Pplanet, float3 sunDirWS)
{
    return CloudSunVisibilityPlanet(Pplanet, sunDirWS, 0.5f);
}

// NO analytic ground backstop (added and removed 2026-06-11): planet-
// clipped sky rays deliberately composite against black — the scene's own
// geometry provides any visible ground, and the dark below-horizon void
// is the intended look. (The "dark patches on clouds" that briefly
// motivated a backstop turned out to be the coverage-map-at-cloudMid
// modulation, fixed at the covGlobal proxy.)

// Trapezoidal column density above the sample at exponentially-spaced
// altitudes. The probe column is exactly radial (localUp = normalize(P), so
// P + up*z = (r+z) * dir with the SAME direction) — coverage, effTop and
// baseShiftKm are direction-only and therefore constant along it. The caller
// passes them from the hull it already evaluated and the whole column
// becomes pure ALU: the old version paid ~3 texture fetches + equirect trig
// per probe (~18 fetches per lit cloud sample) for bitwise-identical values.
inline float CloudColumnDensityAbove(float altKm, float coverage, float effTop,
                                     float baseShiftKm, float startDensity)
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
        float dCur = coverage * CloudAltitudeProfileFromTop(altKm + zCur,
                                                            effTop, baseShiftKm);
        columnKm  += 0.5f * (dPrev + dCur) * (zCur - zPrev);
        zPrev      = zCur;
        dPrev      = dCur;
    }
    return columnKm;
}

// Per-sample in-scattered radiance — NUBIS3 light energy (see knob
// block for the exact published formulas and the mapping).
// Caller folds it as sigma_t * L * segIn.
// hullCoverage/hullEffTop/baseShiftKm come from the caller's CloudProbeHullEx
// so the ambient column probe runs fetch-free. depthInCloudKm is the
// caller-tracked in-cloud view-path length standing in for Nubis3's SDF
// cloud_distance (inner glow driver). sunODOut returns the
// cone-averaged cloud optical depth toward the sun so the caller can reuse
// it for the atmosphere haze shadow at the same point instead of running a
// second shadow march.
float3 CloudComputeLighting(float3 P, float3 L, CloudMaterial m,
                            float hullCoverage, float hullEffTop, float baseShiftKm,
                            float depthInCloudKm,
                            float3 sunRad, float3 sunAtmos,
                            float3 skyAmbientTop, float3 skyAmbientHorizon,
                            float3 groundIrrad,
                            float cosTheta,
                            float earthShadow,
                            uint2 pixel, uint frame, uint tap,
                            out float sunODOut)
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
    sunODOut = sunOD;

    float h = m.heightFrac;
    float3 K = sunRad * sunAtmos * earthShadow * CLOUD_ALBEDO;

    // ===== NUBIS3 DIRECT SCATTERING (see knob block) =====
    //   Direct = T * PrimaryPhase + ms_volume * SecondaryPhase
    // ms_volume verbatim: dimensional_profile gate (-> the dark edges:
    // thin samples get no multi-scatter) times exp(-DL * scale) with the
    // scale remapped 0.25 -> 0.05 by sun_dot and in-cloud depth (-> the
    // inner glow through deep backlit cores). Both remaps clamped.
    // ms_volume rides the PHYSICAL OD; only the direct Beer term keeps
    // the sunTauMult artistic shadow deepener (see knob block).
    float Tdir = exp(-sunOD);

    float sunODphys = sunOD / max(CLOUD_SUN_TAU_MULT, 1e-2f);
    float glowScale = lerp(CLOUD_N3_MS_BASE, CLOUD_N3_MS_GLOW,
                           saturate(depthInCloudKm
                                    * (1.0f / CLOUD_N3_GLOW_DEPTH_KM)));
    float msScale   = lerp(CLOUD_N3_MS_BASE, glowScale,
                           saturate(cosTheta
                                    * (1.0f / CLOUD_N3_GLOW_SUNDOT)));
    // msStrength/4: slider default 4.0 == published scale 1; the
    // MS_BRIGHTNESS calibration gain sits on top (see knob block).
    float msVolume  = m.profile * exp(-sunODphys * msScale)
                    * (CLOUD_MS_STRENGTH * 0.25f * CLOUD_N3_MS_BRIGHTNESS);

    float3 inscatter = K * (Tdir     * CloudPhaseNubisPrimary(cosTheta)
                          + msVolume * CloudPhaseNubisSecondary(cosTheta));

    // ===== NUBIS3 AMBIENT SCATTERING (see knob block) =====
    // ambient_scattering = pow(1 - dimensional_profile, 0.5)
    //                    * exp(-summed_ambient_density)
    // The (1-profile)^0.5 gate is the published model: ambient enters
    // where material is thin (edges, boiling tops via the profile
    // taper); dense cores are lit by the MS term instead. The column
    // probe above the sample is our summed_ambient_density.
    float3 ambient = float3(0, 0, 0);
    if (CLOUD_SKY_AMBIENT > 0.5f || CLOUD_GROUND_BOUNCE > 0.5f)
    {
        float altKm    = length(P) - ATMOS_BOTTOM_RADIUS;
        float columnKm = CloudColumnDensityAbove(altKm, hullCoverage, hullEffTop,
                                                 baseShiftKm, m.profile);
        float ambOD    = min(columnKm * CLOUD_EXTINCTION * CLOUD_AMBIENT_AO_SCALE,
                             CLOUD_AMBIENT_OD_MAX);
        float skyOcc   = exp(-ambOD);

        if (CLOUD_SKY_AMBIENT > 0.5f)
        {
            float ambScatter = sqrt(saturate(1.0f - m.profile));
            float3 skyColor  = lerp(skyAmbientHorizon, skyAmbientTop, h);
            ambient += skyColor * ambScatter * skyOcc * CLOUD_AMBIENT_INTENSITY;
        }
        if (CLOUD_GROUND_BOUNCE > 0.5f)
        {
            // The ground a base bounces off is mostly this cloud's OWN
            // footprint: it sees direct sun attenuated by this column
            // (visG) plus the diffusely transmitted fraction (deckT) —
            // thick decks bounce only their diffuse transmission, thin
            // wisps bounce nearly full sun. The old un-gated version gave
            // every base a thickness-independent light floor. (The
            // earlier "no local Tdir gate" rationale predates deckT: a
            // pure exp(-sunOD) gate really did zero the term under thick
            // cloud — the diffuse path is what makes the gate physical.)
            // groundIrrad itself still carries the global-cover proxy for
            // neighbouring clouds shading the wider bounce area.
            float sunCosZb = saturate(dot(SafeNormalize(P), L));
            float visG     = exp(-sunOD);
            float bounceT  = visG + (1.0f - visG)
                           * CloudDeckDiffuseT(sunOD, sunCosZb);
            ambient += groundIrrad * bounceT * (1.0f - h);
        }
    }

    return inscatter + ambient;
}

// Bounce-ray cheap variant. Mirrors the primary's lighting model
// (CloudComputeLighting equivalent — Nubis3 direct scattering, distFade,
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

    float3 O = g_skyObserverPlanet;
    float3 L = SafeNormalize(sunDir);

    float tNear, tFar;
    if (!RayCloudShell(O, V, tNear, tFar)) return float3(0, 0, 0);

    tFar = min(tFar, tNear + CLOUD_CHEAP_MAX_LEN_KM);
    if (tFar <= tNear) return float3(0, 0, 0);

    // NUBIS3 per-ray constants (phases and the glow sun_dot weight are
    // view-constant for the whole bounce ray) — mirror of
    // CloudComputeLighting.
    const float cosTheta = dot(V, L);
    const float ph1      = CloudPhaseNubisPrimary(cosTheta);
    const float ph2      = CloudPhaseNubisSecondary(cosTheta);
    const float sunDotW  = saturate(cosTheta * (1.0f / CLOUD_N3_GLOW_SUNDOT));

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

    // Base empty-step cap; the in-loop distance floor + half-budget reach
    // floor (see CLOUD_STEP_DIST_FRAC_*) extend the march across the whole
    // shell segment, so the old "8 km × ~10 iters ≈ 80 km" reach limit that
    // cut reflected clouds off early no longer applies. Lerped against
    // fineStep by lodT so near-camera samples stay tight and don't skip over
    // visible cumulus.
    const float kCheapEmptyStepCap = 8.0f;

    float t          = tNear;
    float fineStep   = CLOUD_TARGET_STEP_KM
                     * lerp(0.55f, 1.0f, CloudLodT(tNear));
    // In-cloud view-path length — Nubis3 cloud_distance proxy for the
    // inner glow (see knob block). Resets across empty strides.
    float inCloudKm  = 0.0f;

    [loop]
    for (int i = 0; i < CLOUD_CHEAP_STEPS; ++i)
    {
        if (t >= tFar) break;
        if (max(trCloud.r, max(trCloud.g, trCloud.b)) < CLOUD_TR_EPS) break;

        float distFade = 1.0f - smoothstep(CLOUD_FADE_DISTANCE_KM,
                                            CLOUD_RENDER_DISTANCE_KM, t);
        float lodT     = CloudLodT(t);
        float rJit     = RandomFloatSingle(seed);

        // Jitter window = smallest possible stride (see the primary
        // phase-2 march for the rationale — the raw fineStep window
        // aliased the heart profile into layers on distant decks).
        float strideMin = max(fineStep, t * CLOUD_STEP_DIST_FRAC_FINE);
        if (i * 2 >= CLOUD_CHEAP_STEPS)
            strideMin = max(strideMin, (tFar - t) / (float)(CLOUD_CHEAP_STEPS - i));
        strideMin = min(strideMin, tFar - t);

        float tSample  = t + rJit * strideMin;
        float3 P       = O + V * tSample;

        float hCoverage = 0.0f, hProfile = 0.0f, hEffTop = CLOUD_LAYER_TOP_KM;
        float hBaseShift = 0.0f;
        float hullRaw   = (distFade > 0.0f)
                        ? CloudProbeHullEx(P, walltime, hCoverage, hProfile,
                                           hEffTop, hBaseShift)
                        : 0.0f;
        float hull      = hullRaw * distFade;
        bool  hullEmpty = (hull <= CLOUD_EFFECTIVE_ZERO_DENSITY);

        // Same adaptive scheme as the primary phase-2 march: distance-
        // proportional floors + half-budget reach floor. With ~10 steps the
        // floor engages after step 5 and spreads the rest over the whole
        // remaining shell — reflections keep their near clouds fine and get
        // coarse-but-present distant clouds instead of an ~80 km cutoff.
        float thisStep;
        if (hullEmpty)
        {
            float maxStepAdaptive = min(
                max(kCheapEmptyStepCap, t * CLOUD_STEP_DIST_FRAC_EMPTY)
                    * (1.0f + t * CLOUD_EMPTY_STEP_GROWTH_PER_KM),
                CLOUD_MAX_EMPTY_STEP_KM);
            thisStep = lerp(fineStep, maxStepAdaptive, saturate(lodT));
            thisStep = max(thisStep, strideMin);
        }
        else
        {
            thisStep = strideMin;
        }
        thisStep = min(thisStep, tFar - t);

        if (!hullEmpty)
        {
            CloudMaterial m = CloudSampleMaterialFromHull(
                P, walltime, lodT, 0u, hCoverage, hProfile, hEffTop, hBaseShift);
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
                float  cosHorizon = CloudTerrainCosHorizon(P);
                float  earthShadow = SunDiskFractionAboveHorizon(sunCosZ, cosHorizon);

                float3 K = sunIrradiance * sunAtmos * earthShadow * CLOUD_ALBEDO;

                // NUBIS3 direct scattering — mirror of CloudComputeLighting
                // (see knob block): T * ph1 + ms_volume * ph2, with the
                // profile-gated MS (dark edges) and the depth/sun_dot
                // extinction-scale remap (inner glow). ms_volume rides
                // the PHYSICAL OD; Tdir keeps sunTauMult.
                float sunODphys = sunOD / max(CLOUD_SUN_TAU_MULT, 1e-2f);
                float glowScale = lerp(CLOUD_N3_MS_BASE, CLOUD_N3_MS_GLOW,
                                       saturate(inCloudKm
                                                * (1.0f / CLOUD_N3_GLOW_DEPTH_KM)));
                float msScale   = lerp(CLOUD_N3_MS_BASE, glowScale, sunDotW);
                float msVolume  = m.profile * exp(-sunODphys * msScale)
                                * (CLOUD_MS_STRENGTH * 0.25f
                                   * CLOUD_N3_MS_BRIGHTNESS);

                float3 inscatter = K * (Tdir * ph1 + msVolume * ph2);

                float segTr = exp(-sigma_t * thisStep);
                L_acc   += trCloud * inscatter * (1.0f - segTr);
                trCloud *= segTr;
            }
        }

        // Nubis3 cloud_distance proxy: grow inside cloud hull, reset in
        // clear air (an SDF would go positive there).
        inCloudKm = hullEmpty ? 0.0f : (inCloudKm + thisStep);

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
        float  rP  = length(P);
        float  alt = max(0.0f, rP - ATMOS_BOTTOM_RADIUS);

        MediumSample med  = SampleMedium(alt);
        float3       segTr = exp(-med.extinction * ds);
        float3       sunTr = TransmittanceToSun(P, L,
                                                ATMOS_BOTTOM_RADIUS,
                                                ATMOS_TOP_RADIUS);

        // Disk-fraction earth shadow — smoothstep width = sun radius.
        float3 Pnorm      = SafeNormalize(P);
        float  sunCosZ    = dot(Pnorm, L);
        float  cosHorizon = CloudTerrainCosHorizon(P);
        float  earthShadow = SunDiskFractionAboveHorizon(sunCosZ, cosHorizon);

        // Cone-jittered (r.yz) + altitude-slice-jittered (r.w) cloud shadow
        // on the air column. Without BOTH jitters the fast-path single tap
        // correlates across pixels and the Perlin-Worley slice paints onto
        // the haze as a ghost sphere surface decoupled from the clouds.
        // The shadowed fraction re-emerges as isotropic through-deck
        // skylight (CloudShadowAmbientTerms) instead of the old
        // softness/floor brightening of the directional term.
        // earthShadow gate: everything the tap feeds is multiplied by
        // earthShadow, so night-side samples skip the fetches outright
        // (psiMS carries its own earth shadow and is ~0 there).
        float cloudVis = 1.0f;
        float cloudAmb = 0.0f;
        float msGate   = 1.0f;
        if (cloud_cloudShadowOnSurfaces > 0.5f && earthShadow > 1e-4f)
        {
            float3 Lj = SampleConeAroundDir(L, kAtmosShadowConeCos,
                                            float2(r.y, r.z));
            float visRaw = CloudSunVisibilityPlanet(P, Lj, r.w);
            // Coverage fetch + ambient terms only where actually shadowed
            // (clear sky skips both and msGate stays 1).
            if (visRaw < 0.999f)
            {
                float cov = CloudGlobalCoverage(P);
                cloudAmb  = CloudShadowAmbientTerms(visRaw, sunCosZ, cov,
                                                    msGate);
            }
            cloudVis = pow(max(visRaw, 1e-6f), ATMOS_CLOUD_SHADOW_SOFTNESS);
            cloudVis = max(cloudVis, ATMOS_CLOUD_SHADOW_FLOOR);
        }

        float3 scatterPhase = med.scatterR * phR + med.scatterM * phM;
        float3 scatterIso   = med.scatterR + med.scatterM;
        float3 sunIllum     = sunIrradiance * earthShadow * sunTr
                            * ATMOS_MULTI_SCATTER_FACTOR;
        // Hillaire MS: phase, sun transmittance and earth shadow live
        // inside the LUT — multiply by sigma_s and sun irradiance only;
        // msGate attenuates it under cloud decks.
        float3 psiMS        = MultiScatterPsi(rP, sunCosZ);
        float3 rate         = scatterPhase * sunIllum * cloudVis
                            + scatterIso * (sunIllum * cloudAmb
                                            + psiMS * sunIrradiance * msGate);

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
//
// maxCloudHullOut = max raw hull (pre-distance-fade) seen by the phase-2
// march. Zero means the hull was empty along the whole ray — the caller can
// then skip the deterministic EvaluateCloudGBuffer march entirely, since the
// same shell along the same ray can't produce cloud there either. Tracked
// pre-fade because the g-buffer march has no distance fade of its own.
float3 EvaluateAtmosphereAndClouds(
    float3 V, float3 sunDir, float3 sunIrradiance,
    float  maxDistKm,
    out float3 transmittanceOut,
    out bool   hitPlanetOut,
    out float  maxCloudHullOut)
{
    transmittanceOut  = float3(1, 1, 1);
    hitPlanetOut      = false;
    maxCloudHullOut   = 0.0f;
#if ATM_DEBUG_RING == 3
    return float3(0, 0, 0);            //sky/atmosphere disabled — ATM_DEBUG_RING
#endif

    InitCloudEnuBasis();

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
    //
    // LUT: the probe sits at a FIXED radius and the atmosphere is
    // spherically symmetric, so both probe integrals reduce to functions of
    // one scalar — the sun-zenith cosine at the probe. They used to be two
    // full IntegrateScattering marches PER PIXEL (each ATMOS_VIEW_STEPS ×
    // an inner sun-transmittance march); Pass_skylut_bake_v8.hlsl now bakes
    // them once per frame into g_cloudAmbientLUT (row 0 = zenith probe,
    // row 1 = horizon probe) and this is two bilinear fetches. The baked
    // values exclude cloud shadowing — the global-cover proxy below
    // (covGlobal) stands in for the old per-step CloudSunVisibilityPlanet
    // term.
    float3 cloudMid       = O + Vn * (0.5f * (tC0 + tC1));
    float3 probeUp        = SafeNormalize(cloudMid);
    float3 cloudProbePos  = probeUp * (ATMOS_BOTTOM_RADIUS
                                     + CLOUD_LAYER_TOP_KM
                                     + CLOUD_TOP_VARIATION_KM + 0.5f);
    float  probeSunCosZ   = dot(probeUp, L);

    // GLOBAL cover proxy — deliberately a per-frame CONSTANT, not a map
    // sample. Sampling the coverage map at the per-ray cloud midpoint
    // (introduced and reverted 2026-06-11) painted the Blue-Marble pattern
    // onto every cloud's ambient / ground bounce as up-to-85%-dark
    // patches: the remote sample point (mid-ray, tens of km from the lit
    // cloud) decouples the pattern from the visible clouds, so it slides
    // over them under camera motion — a ghost "sphere surface" visible
    // only on cloud bodies. A constant cannot pattern anything. The
    // per-SAMPLE coverage weights in the air integrators are a different
    // story and stay: they are anchored at the exact positions they light.
    float covGlobal = saturate(CLOUD_COVERAGE_BASE);

    float3 skyAmbientTop     = float3(0, 0, 0);
    float3 skyAmbientHorizon = float3(0, 0, 0);
    if (CLOUD_SKY_AMBIENT > 0.5f)
    {
        float u = CloudAmbientLutU(probeSunCosZ);
        float3 scatterTop     = g_cloudAmbientLUT.SampleLevel(
            g_sampler_LUT, float2(u, 0.25f), 0).rgb;
        float3 scatterHorizon = g_cloudAmbientLUT.SampleLevel(
            g_sampler_LUT, float2(u, 0.75f), 0).rgb;
        skyAmbientTop     = scatterTop     * SKY_INTENSITY * CLOUD_SKY_AMBIENT_SCALE;
        skyAmbientHorizon = scatterHorizon * SKY_INTENSITY * CLOUD_SKY_AMBIENT_SCALE;

        float topLum     = max(Luma(skyAmbientTop), 1e-4f);
        float horizonLum = Luma(skyAmbientHorizon);
        if (horizonLum > topLum) skyAmbientHorizon *= topLum / horizonLum;

        // Zenith row: NO cover darkening — the probe sits at cloud-top
        // radius and integrates the air ABOVE the deck, which the deck
        // does not shadow; self-occlusion within the cloud is skyOcc's
        // job (CloudComputeLighting).
        // Horizon row: the probe's ~390 km sunlit clear-air path is
        // exactly what a base under wide overcast does NOT see — looking
        // sideways it faces the dark under-deck cavity, an occlusion the
        // radial-column AO cannot capture. Attenuate by the GLOBAL cover
        // constant (same proxy as the ground bounce; see covGlobal note
        // above for why this must not be a map sample); clear skies keep
        // the full horizon ring, and the heightFrac lerp already retires
        // this row near cloud tops.
        skyAmbientHorizon *= (1.0f - 0.85f * covGlobal);
    }

    float3 groundIrrad = float3(0, 0, 0);
    if (CLOUD_GROUND_BOUNCE > 0.5f)
    {
        float  sunCosUp   = saturate(probeSunCosZ);
        float3 sunGroundT = TransmittanceToSun(cloudProbePos, L,
                                               ATMOS_BOTTOM_RADIUS,
                                               ATMOS_TOP_RADIUS);
        groundIrrad = sunIrradiance * sunGroundT * sunCosUp
                    * CLOUD_GROUND_ALBEDO * CLOUD_GROUND_SCALE
                    * (1.0f / PI);

        // GLOBAL cover proxy (covGlobal, per-frame constant — see note
        // above). Tracking per-pixel cloud shadow on the ground caused
        // DLSS RR to read the parallax between ground-depth shadow and
        // cloud-depth surface as a separate layer; the map-at-cloudMid
        // variant painted the coverage map onto cloud bases the same way.
        if (cloud_cloudShadowOnSurfaces > 0.5f)
            groundIrrad *= (1.0f - 0.85f * covGlobal);
    }

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
        // In-cloud view-path length — Nubis3 cloud_distance proxy for
        // the inner glow (see knob block). Resets across empty strides.
        float inCloudKm = 0.0f;

        [loop]
        for (int ts = 0; ts < CLOUD_VIEW_STEPS_MAX; ++ts)
        {
            if (t >= tC1) break;
            if (max(transmittance.r, max(transmittance.g, transmittance.b)) < CLOUD_TR_EPS) break;

            float distFade = 1.0f - smoothstep(CLOUD_FADE_DISTANCE_KM,
                                                CLOUD_RENDER_DISTANCE_KM, t);
            float lodT     = CloudLodT(t);
            float rJit     = RandomFloatSingle(seed);

            // Jitter window = the smallest stride this step can take: the
            // in-cloud stride incl. its distance floor and, in the second
            // half of the budget, the reach floor (both hull-independent,
            // so they're known BEFORE sampling). The old window was the
            // raw stepSize alone — at horizon distances the actual stride
            // is 2-10x larger, so samples clustered in the first fraction
            // of every segment and the heart profile's vertical bands
            // aliased into horizontal LAYERS on far decks. Empty strides
            // may still exceed the window (harmless — empty segments add
            // no cloud shading); in-cloud strides never do.
            float strideMin = max(stepSize, t * CLOUD_STEP_DIST_FRAC_FINE);
            if (ts * 2 >= CLOUD_VIEW_STEPS_MAX)
                strideMin = max(strideMin,
                                (tC1 - t) / (float)(CLOUD_VIEW_STEPS_MAX - ts));
            strideMin = min(strideMin, tC1 - t);

            float tSample  = t + rJit * strideMin;
            float3 P       = O + Vn * tSample;

            float rP            = length(P);
            float alt           = max(0.0f, rP - ATMOS_BOTTOM_RADIUS);
            MediumSample atmosMed = SampleMedium(alt);

            // Hull-Ex passes (coverage, profile, effTop, baseShift) to the
            // material call. The old fallback heightFrac block that ran here
            // for every step was dead work: hull-empty samples never feed
            // CloudComputeLighting, and dense samples get heightFrac from
            // CloudSampleMaterialFromHull anyway — yet it paid a cloud-offset
            // cubemap fetch + equiangular projection per stride.
            float hCoverage = 0.0f, hProfile = 0.0f, hEffTop = CLOUD_LAYER_TOP_KM;
            float hBaseShift = 0.0f;
            float hullRaw   = (distFade > 0.0f)
                            ? CloudProbeHullEx(P, walltime, hCoverage, hProfile,
                                               hEffTop, hBaseShift)
                            : 0.0f;
            float hull      = hullRaw * distFade;
            bool  hullEmpty = (hull <= CLOUD_EFFECTIVE_ZERO_DENSITY);
            maxCloudHullOut = max(maxCloudHullOut, hullRaw);

            CloudMaterial m = (CloudMaterial)0;
            float cloudDensity = 0.0f;
            if (!hullEmpty)
            {
                m            = CloudSampleMaterialFromHull(P, walltime, lodT, 0u,
                                                           hCoverage, hProfile,
                                                           hEffTop, hBaseShift);
                cloudDensity = m.density * distFade;
            }
            float cloudSigmaT = cloudDensity * CLOUD_EXTINCTION;

            // Combined extinction: atmos (per-channel) + cloud (scalar).
            float3 sigma_t_total = atmosMed.extinction
                                 + float3(cloudSigmaT, cloudSigmaT, cloudSigmaT);

            // Fine step inside density, coarse when empty. Both get a
            // distance-proportional floor (sub-pixel where it engages, see
            // CLOUD_STEP_DIST_FRAC_*), and the second half of the iteration
            // budget enforces a reach floor that spreads the remaining steps
            // evenly over the remaining segment — the march now ALWAYS
            // reaches tC1, degrading to coarse distant sampling instead of
            // hard-cutting clouds at the horizon when CLOUD_VIEW_STEPS_MAX
            // runs out. The first half of the budget is unconstrained so
            // near-field sampling stays at CLOUD_TARGET_STEP_KM quality.
            float thisStep;
            if (hullEmpty)
            {
                float maxStepAdaptive = min(
                    max(CLOUD_MAX_STEP_KM, t * CLOUD_STEP_DIST_FRAC_EMPTY)
                        * (1.0f + t * CLOUD_EMPTY_STEP_GROWTH_PER_KM),
                    CLOUD_MAX_EMPTY_STEP_KM);
                thisStep = lerp(stepSize, maxStepAdaptive, saturate(lodT));
                // Never below the jitter window so the sample always lies
                // inside the segment it stands for.
                thisStep = max(thisStep, strideMin);
            }
            else
            {
                // strideMin IS the in-cloud stride (fine + distance floor
                // + reach floor) — and exactly the window tSample was
                // jittered over.
                thisStep = strideMin;
            }
            thisStep = min(thisStep, tC1 - t);

            float3 segTr = exp(-sigma_t_total * thisStep);

            float3 sunAtmosT = TransmittanceToSun(P, L,
                                                  ATMOS_BOTTOM_RADIUS,
                                                  ATMOS_TOP_RADIUS);

            float3 Pnorm      = SafeNormalize(P);
            float  sunCosZ    = dot(Pnorm, L);
            float  cosHorizon = CloudTerrainCosHorizon(P);
            float  earthShadow = SunDiskFractionAboveHorizon(sunCosZ, cosHorizon);

            // Cloud rate = radiance × cloud sigma_s (≈ sigma_t for albedo≈1).
            // Evaluated before the haze shadow so dense samples can hand
            // their cone-averaged sun OD to the atmosphere term below.
            float3 cloudRate  = float3(0, 0, 0);
            float  cloudSunOD = -1.0f;   // < 0 = no cloud lighting this step
            if (cloudDensity > 0.0f)
            {
                float3 cloudRadiance = CloudComputeLighting(
                    P, L, m,
                    hCoverage, hEffTop, hBaseShift,
                    inCloudKm,
                    sunIrradiance, sunAtmosT,
                    skyAmbientTop, skyAmbientHorizon, groundIrrad,
                    cosTheta, earthShadow,
                    pixel, frame, 200u + (uint)acceptedI * 11u,
                    cloudSunOD);
                cloudRate = cloudRadiance * cloudSigmaT;
                ++acceptedI;
            }

            // Cloud shadow on the air column inside the shell — without
            // it the haze brightens visibly as the ray crosses from Phase
            // 1 into Phase 2. Salt 700u keeps this STBN tap independent.
            //
            // Dense samples reuse the cloud lighting's own sun OD (longer
            // reach + far-side term, already cone-averaged) instead of
            // paying a second CloudOpticalDepthAlongRay march for the same
            // occlusion at the same point.
            // earthShadow gate as in IntegrateAtmosphereSegment — skips
            // the cone tap + ambient fetches on the night side, where the
            // directional and deck terms are zeroed by earthShadow anyway.
            float cloudVisAtmos = 1.0f;
            float cloudAmbAtmos = 0.0f;
            float msGateAtmos   = 1.0f;
            if (cloud_cloudShadowOnSurfaces > 0.5f && earthShadow > 1e-4f)
            {
                float visRaw;
                float tauAmb;
                if (cloudSunOD >= 0.0f)
                {
                    // OD-direct — exp/log round-tripping the vis would cap
                    // thick-deck tau at the float clamp.
                    visRaw = exp(-cloudSunOD);
                    tauAmb = cloudSunOD;
                }
                else
                {
                    // rCone.xy = cone jitter, rCone.z = altitude-slice
                    // jitter — both needed or the fast path's single tap
                    // paints a ghost mid-shell sphere onto the fog (see
                    // CloudOpticalDepthAlongRay).
                    float4 rCone = CloudRand4(pixel, frame, (uint)ts + 700u);
                    float3 LjAtmos = SampleConeAroundDir(L, kAtmosShadowConeCosP2,
                                                         float2(rCone.x, rCone.y));
                    visRaw = CloudSunVisibilityPlanet(P, LjAtmos, rCone.z);
                    tauAmb = -log(max(visRaw, 1e-20f));
                }
                // Ambient terms only where actually shadowed. In-band
                // samples reuse the hull's coverage fetch; sub-deck strides
                // (hull empty on altitude) pay one equirect fetch.
                if (visRaw < 0.999f)
                {
                    float cov = (hullRaw > 0.0f) ? hCoverage
                                                 : CloudGlobalCoverage(P);
                    cloudAmbAtmos = CloudShadowAmbientTermsOD(tauAmb, sunCosZ,
                                                              cov, msGateAtmos);
                }
                cloudVisAtmos = pow(max(visRaw, 1e-6f), ATMOS_CLOUD_SHADOW_SOFTNESS);
                cloudVisAtmos = max(cloudVisAtmos, ATMOS_CLOUD_SHADOW_FLOOR);
            }

            float3 atmosScatterPhase = atmosMed.scatterR * phR + atmosMed.scatterM * phM;
            float3 atmosScatterIso   = atmosMed.scatterR + atmosMed.scatterM;
            float3 atmosSunIllum     = sunIrradiance * earthShadow * sunAtmosT
                                     * ATMOS_MULTI_SCATTER_FACTOR;
            // Hillaire MS: phase / sun transmittance / earth shadow live
            // inside the LUT — multiply by sigma_s and sun irradiance only;
            // msGateAtmos attenuates it under (and inside) cloud decks.
            float3 psiMS             = MultiScatterPsi(rP, sunCosZ);
            float3 atmosRate         = atmosScatterPhase * atmosSunIllum * cloudVisAtmos
                                     + atmosScatterIso * (atmosSunIllum * cloudAmbAtmos
                                                          + psiMS * sunIrradiance
                                                                  * msGateAtmos);

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

            // Nubis3 cloud_distance proxy: grow inside cloud hull, reset
            // in clear air (an SDF would go positive there).
            inCloudKm = hullEmpty ? 0.0f : (inCloudKm + thisStep);

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
