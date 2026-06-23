//====================================
//INCLUDES V8
//====================================
//compute shaders must define COMPUTE_PASS before include

#ifndef INCLUDES_V8_HLSLI
#define INCLUDES_V8_HLSLI

//====================================
//TEMP: ALPHA TEST KILL SWITCH
//====================================
//Set to 1 to TEMPORARILY disable per-texel alpha testing for both shadow/
//visibility rays (AlphaCandidateOccludes in Inline_RT_v8.hlsli) and regular
//path-tracing rays (AlphaTestAnyHit in AnyHit.hlsl). Alpha-tested geometry then
//behaves as fully opaque - foliage/fences/etc. stop punching holes. Restore by
//setting back to 0 (or deleting this block and the two #if guards).
#define DISABLE_ALPHA_TEST 0

//====================================
//PUSH CONSTANTS
//====================================
cbuffer Push : register(b1)
{
    uint2 gImageSize;
    uint  g_InputStackIdx;
    uint  g_OutputStackIdx;
    uint  rs_tempMcap;
    uint  rs_spatCountMax;
    uint  rs_spatCountMin;
    uint  rs_spatRadMax;
    uint  rs_spatRadMin;
    uint  rs_flags;
    float rs_reuseRoughnessMin;
    float rs_reuseRoughnessMax;
    uint  rs_spatTries;
    //paired reuse textures, flag bits 1=flipX 2=flipY 4=transpose
    uint  rs_reuseOffset0_x;
    uint  rs_reuseOffset0_y;
    uint  rs_reuseFlags0;
    uint  rs_reuseOffset1_x;
    uint  rs_reuseOffset1_y;
    uint  rs_reuseFlags1;
    uint  rs_reuseOffset2_x;
    uint  rs_reuseOffset2_y;
    uint  rs_reuseFlags2;
    //neighbor rejection thresholds for spatial select
    float rs_rejNormalDot;
    float rs_rejDistance;
    //NRC runtime, flag bits in Nrc_v8.hlsli
    uint  nrc_flags;
    float nrc_area_spread_c;
    float nrc_lr_scale;
    //dynamic per-frame cap on inference slots, sized from prior frame's actual
    //counter via async readback. Both raygen (NrcAppendInference cap) and the
    //CUDA inference dispatch (count) read this so the work matches demand
    //instead of paying for the buffer's static 2*W*H capacity every frame.
    //Slot 27 also doubles as the 16B alignment pad for nrc_scene_center below.
    uint  nrc_inference_capacity;
    //scene to [0,1]^3 for tcnn HashGrid, scale_inv = 0.5/halfExtent
    float3 nrc_scene_center;
    float  nrc_scene_scale_inv;
    //SPMIS spatial reuse. Slots 32-37, read by the Pass_spmis_* kernels. Selected by
    //RS_FLAG_SPMIS_SPATIAL (0x10).
    uint  spmis_reuseN;       // Ntilde: non-canonical reuse draws
    uint  spmis_risN;         // inner-RIS candidate count per draw
    uint  spmis_mcap;         // confidence M cap applied on the spatial pass
    uint  spmis_tileSize;     // screen-space cell tile size in pixels
    float spmis_jacThreshold; // reconnection-shift jacobian reject band [1/T, T]
    float spmis_normalSimCos; // neighbor-similarity normal cone (cos) used in the cell search
    //path-trace termination (slots 38-39, fills out register 9). Runtime editor
    //sliders, both capped at 32 host-side. Read by Pass_raygen_v8.
    uint  pt_maxBounces;      // raygen path loop bound: depth runs [1, pt_maxBounces)
    uint  pt_rrStartDepth;    // Russian roulette begins at depth >= this (set high to disable RR)
    //slot 40 (opens register 10). RIS-over-N initial samples per pixel, host-
    //clamped [1,8]. Read by Pass_raygen_v8 as the initial-sample loop count.
    uint  pt_initialSamples;
    //slot 41 (register 10.y). SPMIS cell-search plane-distance rejection, as a FRACTION
    //of the distance to camera (regular-ReSTIR geometry rejection). Read by
    //Pass_spmis_reuse_v8.
    float spmis_planeDist;
};

//====================================
//PIPELINE FLAGS (packed into rs_flags)
//====================================
//High bit of rs_flags, clear of the ReSTIR bits (0x2 tempGI, 0x8 spatGI,
//0x10 native-spatial) and the NRC/reuse fields. Set by the Renderer from the
//editor toggle, NOT by ReSTIRSettings::Flags(). The ReSTIR passes only test the
//low bits, so it is inert for them.
//
//RS_FLAG_CLAMP_EMITTERS — Pass_shading luminance-clamps emitter radiance before
//the reversible DlssReinhard pre-tonemap, pulling big bright emitters off the
//[0,1] rail so DLSS RR (and the postprocess inverse) don't amplify denoiser
//error into artefacts. Mirrors DLSSManager::clampEmitterSpikes.
#define RS_FLAG_CLAMP_EMITTERS  0x100u
#define CLAMP_EMITTERS_MODE  ((rs_flags & RS_FLAG_CLAMP_EMITTERS) != 0u)

//RS_FLAG_SPMIS_SPATIAL — selects the SPMIS global-hash-grid spatial-reuse path
//(Pass_spmis_* : reset/count/offsets/sort/reuse)
//instead of the texture-paired select/shift/_v8_1 passes. Sub-mode of spatial GI
//(0x8): when set, the texture passes no-op, raygen inserts each pixel's hash, and
//the SPMIS pipeline owns spatial reuse. Set by ReSTIRSettings::Flags().
#define RS_FLAG_SPMIS_SPATIAL  0x10u
#define SPMIS_SPATIAL_MODE  ((rs_flags & RS_FLAG_SPMIS_SPATIAL) != 0u)


//RS_FLAG_SPMIS_CONF_ADJUST — §4.3 non-canonical confidence scaling. Scales neighbour
//confidences by Ntilde/cell_pixel_count, which BOOSTS the canonical MIS weight
//(~1/(Ntilde+1)) and suppresses dark-pepper at the cost of WEAKER neighbour reuse /
//equalization. OFF (scaling = 1) gives the canonical ~no weight and maximal neighbour
//reuse; on = the scaled variant.
#define RS_FLAG_SPMIS_CONF_ADJUST  0x2000u
#define SPMIS_CONF_ADJUST  ((rs_flags & RS_FLAG_SPMIS_CONF_ADJUST) != 0u)


//RS_FLAG_DISABLE_CORR_REDUCTION — A/B: turn OFF the duplication-map correlation
//reduction. Pass_dup_gi counts how many of the 17x17 neighbours share this pixel's
//reconnection vertex (V2) and writes that fraction D to scratch slot 6; the temporal
//pass normally COLLAPSES the confidence cap toward 1 as D rises
//(effMcap = lerp(rs_tempMcap, 1, pow(D,0.1)) — very aggressive: D=0.1 -> cap~5).
//Because cell reuse deliberately SPREADS a good sample across a coherent cell, D goes
//up and this decorrelation then caps confidence back down, fighting the reuse. When
//this flag is set the temporal cap ignores D (effMcap = rs_tempMcap), letting us test
//whether the decorrelation is what starves cell-reuse effectiveness.
#define RS_FLAG_DISABLE_CORR_REDUCTION  0x40u
#define CORR_REDUCTION_OFF  ((rs_flags & RS_FLAG_DISABLE_CORR_REDUCTION) != 0u)

//RS_FLAG_DISABLE_X1_DIRECT — diagnostic A/B: zero scratch slot 3 (directAtX1 =
//the depth==1 NEE-sun + depth==1 BSDF-ray-miss ENV, the ONLY radiance that bypasses
//the ReSTIR reservoir). It is a single-sample 1/pdf estimate with NO spatio-temporal
//reuse, so even a dim env floor (nightBase/stars) shows up as fireflies immune to
//the temporal M-cap. If the "high-variance layer over clean ReSTIR" DISAPPEARS with
//this set, the culprit is this un-reused env/sun direct; if it PERSISTS, the noise is
//inside the reservoir (a fraction of pixels failing temporal/spatial reuse).
#define RS_FLAG_DISABLE_X1_DIRECT  0x80u
#define X1_DIRECT_OFF  ((rs_flags & RS_FLAG_DISABLE_X1_DIRECT) != 0u)

//RS_FLAG_NO_SPEC_REPROJ — force the temporal pass to use SURFACE (self) reprojection
//for the DI reservoir instead of the stochastic specular reprojection. The temporal
//pass otherwise flips, per pixel per frame, between self-reproject and the reflection's
//virtual-position reproject (useSpecReproj = rSpec < specularity). That coin flip pulls
//history from a DIFFERENT pixel (the reflection) with inconsistent frame-to-frame
//lineage, so those pixels' confidence M never accumulates -> a flat high-variance band
//layered over the cleanly-accumulating self-reproject pixels (concentrated at grazing
//angles / near lights where specularity is high). Forcing self-reprojection gives every
//pixel a stable history lineage so M accumulates uniformly.
#define RS_FLAG_NO_SPEC_REPROJ  0x200u
#define NO_SPEC_REPROJ  ((rs_flags & RS_FLAG_NO_SPEC_REPROJ) != 0u)

//RS_FLAG_NO_REUSE_VIS — A/B: take the reconnection shadow ray OUT of the temporal
//AND spatial reuse passes and apply it exactly once, at the spatial resolve write.
//Motivation: small/far lights behind high-frequency thin occluders (fences, poles)
//make the x1->x2 reconnection visibility a near-binary, high-frequency function of
//the SHADING POINT. Sub-pixel jitter (for DLSS RR) slides that point across the
//occluder, so vis flips 1->0 frame-to-frame, p_n=ph*vis collapses to 0 and the
//reservoir effectively resets - reuse stops working (the well-converged top vs the
//noisy side of the container). With this set the reuse passes treat vis==1 (so the
//target is UNSHADOWED and light SELECTION keeps accumulating M under jitter); the
//stored F is then unshadowed, so the winning sample's reconnection visibility is
//applied ONCE to F*W where the spatial pass writes scratch slot 2. The intrinsic
//NEE/sun visibility in raygen is untouched. Unbiased: W is taken w.r.t. the
//unshadowed target and the true visibility multiplies the final contribution.
#define RS_FLAG_NO_REUSE_VIS  0x800u
#define REUSE_VIS_OFF  ((rs_flags & RS_FLAG_NO_REUSE_VIS) != 0u)

//RS_FLAG_NO_FINAL_VIS — sub-toggle of RS_FLAG_NO_REUSE_VIS: also skip the ONE
//deferred reconnection shadow ray at the spatial resolve, so the GI output is fully
//UNSHADOWED. Diagnostic only (shows the unshadowed upper bound / what the unshadowed
//reservoir selection looks like). Only meaningful when REUSE_VIS_OFF is set - with
//the reuse rays already gone, this removes the last visibility too. The NRC gather
//follows it (trains on the same unshadowed GI it displays).
#define RS_FLAG_NO_FINAL_VIS  0x1000u
#define FINAL_VIS_OFF  ((rs_flags & RS_FLAG_NO_FINAL_VIS) != 0u)

//====================================
//IMAGE SIZE MACROS
//====================================
#define IMG_W (gImageSize.x)
#define IMG_H (gImageSize.y)

#ifdef COMPUTE_PASS
    //emulate DispatchRaysIndex/Dimensions
    #define gImageWidth  (gImageSize.x)
    #define gImageHeight (gImageSize.y)
    #define DispatchRaysDimensions() uint3(gImageWidth, gImageHeight, 1)
    static uint3 gDispatchIdx;
    #define DispatchRaysIndex()      gDispatchIdx
#endif

#define ENABLE_RAY_QUERY_INLINE

//====================================
//SAMPLERS AND LUTS
//====================================
SamplerState   g_sampler           : register(s0);
SamplerState   g_sampler_LUT       : register(s1);
//Plain bilinear + WRAP (no anisotropy). Used for samples whose UV
//derivatives become enormous at wrap boundaries — anisotropic filtering
//would otherwise see the huge derivative as a giant footprint and
//pick a coarse mip / mis-orient the aniso kernel, producing a visible
//streak across the seam. The equirectangular cloud coverage map
//(g_cloudCoverage) is the prototypical case: atan2(z,x) wraps ±π at
//the longitude=180° meridian, and any anisotropic sampler shows a
//banded artifact there. This sampler avoids that entirely.
SamplerState   g_samplerLinearWrap : register(s2);
Texture2DArray g_LUT         : register(t33);

//====================================
//STAR / MILKY WAY SKYBOX
//====================================
//Equirectangular sky texture in the celestial (RA/Dec) frame. Source: e.g.
//NASA SVS 4851 "Deep Star Maps 2020" Tycho-2 / Gaia based all-sky map.
//Sampled in EvaluateStars (SunSampler_v8.hlsli) using a celestial-frame ray
//direction so the texture rotates with sidereal time in lockstep with the sun.
//
//HOST WIRING (Renderer_Pipeline.cpp / Renderer.h):
//  1. Add `ComPtr<ID3D12Resource> m_skyStarsTexture;` to Renderer.h.
//  2. Load the equirectangular star map (BC6H DDS for HDR linear, or
//     R8G8B8A8_UNORM_SRGB for LDR) and upload to m_skyStarsTexture.
//  3. In CreateRayGenSignature, add:
//       ranges.emplace_back().Init(
//         D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 40, 0, STATIC,
//         D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
//  4. In WriteDescriptors, after the AE state UAV (current last slot 63),
//     CreateShaderResourceView for m_skyStarsTexture at the next heap slot.
//  5. If using LDR sRGB, set SKY_STAR_TEXTURE_SRGB=1 in SunSampler_v8.hlsli.
//  6. If stars appear mirrored east/west, set SKY_STAR_TEXTURE_FLIP_U=1.
Texture2D<float4> gSkyStars   : register(t40);

//Volumetric cloud noise — 256³ RGBA8 3D texture baked once at startup by
//Pass_cloudnoise_bake_v8.hlsl. The runtime cloud integrator
//(Clouds_v8.hlsli) samples this instead of evaluating Perlin/Worley
//analytically, dropping per-density-tap cost from ~80-200 ALU ops to one
//texture fetch. t41 is intentionally skipped to leave room for the
//optional STBN array used by CLOUD_STBN_AVAILABLE in Clouds_v8.hlsli.
//Channel layout: R = Perlin-Worley FBM (low-freq cloud body),
//G = Worley FBM (alt low-freq), B = value noise (HF erosion),
//A = single-octave Worley (mid-freq cauliflower).
Texture3D<float4> g_cloudNoise : register(t42);

//Planet-scale cloud coverage map (NASA Blue Marble, equirectangular 8192×4096
//R8 luminance). Loaded once by Renderer::InitCloudCoverageTexture. The cloud
//integrator (Clouds_v8.hlsli) samples this via planet-radial direction to gate
//the procedural noise body with real-world climatology — gives continental-
//scale weather patterns instead of uniform global coverage. Hardware bilinear
//filtering interpolates between source texels so coverage transitions read as
//smooth gradients (no blocky pixel boundaries).
Texture2D<float> g_cloudCoverage : register(t43);

//PLANET: baked surface elevation cubemap. 6 layers, one per cube face,
//R32F, values in KILOMETRES. Sampled via equiangular cubed-sphere
//projection (see TerrainHeight below); face order matches the baker
//(+X, -X, +Y, -Y, +Z, -Z). Uploaded once by
//Renderer::InitTerrainHeightmapTexture from the planet::HeightmapCubemap
//CPU mirror. Declared up here (rather than near g_terrainTable at t44)
//so it's visible to TerrainHeight() lower down - HLSL is single-pass.
Texture2DArray<float> g_terrainHeightmap : register(t45);

//PLANET: baker v8 companion layers. All 6-face Texture2DArrays, equiangular
//cubed-sphere projection identical to the heightmap so a single
//SphereToEquiangularFaceUV() lookup feeds all four samples.
//
//  surface_color (t46): Mars-style RGB tint baked by SurfaceColorPass.
//    Alpha channel forced to 1.0 by the baker; a 0 alpha indicates a null
//    SRV (the bake didn't produce the layer) so the shader should fall
//    back to the legacy TERRAIN_ALBEDO constant.
//  normal (t47): tangent-space normal map (-gx, -gy, +1 normalised, packed
//    [-1,1] -> [0,1]). Alpha = 1.0 sentinel as above. Same resolution as
//    the heightmap so a sample picks up bake-resolution gradient detail
//    finer than the runtime central-difference eps would resolve.
//  cloud_offset (t48): block-averaged surface elevation in KILOMETRES,
//    much smaller (typically 256^2 per face). Used by Clouds_v8.hlsli to
//    shift the cloud layer with general terrain so sharp peaks poke through
//    clouds while gentle plateaus do not.
Texture2DArray<float4> g_terrainSurfaceColor : register(t46);
Texture2DArray<float4> g_terrainNormalMap    : register(t47);
Texture2DArray<float>  g_terrainCloudOffset  : register(t48);

//Per-frame sky LUTs, baked by Pass_skylut_bake_v8.hlsl (dedicated dispatch
//recorded at the top of PopulateCommandList, before any consumer pass).
//
//  t49 g_skyTransmittanceLUT: 256x64 RGBA16F. rgb = atmosphere transmittance
//    to space, Bruneton (r, mu) parameterization — see
//    TransmittanceLutUvFromRMu in SunSampler_v8.hlsli. Replaces the old
//    per-call ATMOS_LIGHT_STEPS inner march in TransmittanceToSun.
//  t50 g_cloudAmbientLUT: 128x2 RGBA16F. Cloud ambient probe scatter over
//    sun-zenith cosine at the cloud-top probe radius; row 0 = zenith probe,
//    row 1 = horizon probe — see CloudAmbientLutU in Clouds_v8.hlsli.
//    Replaces the two per-pixel IntegrateScattering probe marches in
//    EvaluateAtmosphereAndClouds.
//  t51 g_skyMultiScatterLUT: 32x32 RGBA16F. Hillaire 2020 multiple-
//    scattering transfer Psi_ms over (sun-zenith cosine, normalized
//    altitude) — see MultiScatterPsi in SunSampler_v8.hlsli. Supplies the
//    isotropic 2nd+ order air scattering the flat multi-scatter factor
//    used to stand in for.
//All sampled with g_sampler_LUT (s1, bilinear clamp), SampleLevel 0.
Texture2D<float4> g_skyTransmittanceLUT : register(t49);
Texture2D<float4> g_cloudAmbientLUT     : register(t50);
Texture2D<float4> g_skyMultiScatterLUT  : register(t51);

//====================================
//CAMERA
//====================================
cbuffer CameraParams : register(b0)
{
    float4x4 view;
    float4x4 projection;
    float4x4 viewI;
    float4x4 projectionI;
    float4x4 prevView;
    float4x4 prevProjection;
    float  time;       //jitter frame index cast to float, drives random seeds and the sun
    float2 jitter;
    float  cameraFar;
    float  walltime;   //accumulated wall-clock seconds, drives dt-based auto-exposure
    //Floating origin: scene origin in absolute world coords. The view
    //matrix is built in shifted space (eye = absolute - sceneOriginWorld),
    //so InitOrigin() returns the shifted camera. For atmosphere math that
    //needs planet centered absolute coords, use InitOrigin() + sceneOriginWorld.
    float3 sceneOriginWorld;
    //sun settings
    float sunLatitude;
    float sunLongitude;
    float sunDayOfYear;
    float sunSimSpeed;
    float sunStartUTCHours;
    float sunNightSpeedup;
    float sunTurbidity;
    float sunSunIntensity;
    float sunSkyIntensity;
    float globalEmissionStrength;
    //thin-lens DoF, driven from the Camera class on the host side
    float dofApertureRadius;
    float dofFocusDistance;
    //star skybox runtime knobs, mirror SunSettings tail in Common.h
    float skyStarIntensity;
    float skyStarGamma;
    float skyStarLodBias;
    float skyStarThreshold;
    float skyNightBaseIntensity;
    //atmosphere quality + look (Bruneton march). Mirror SunSettings tail
    //in Common.h. Cast int fields at the use site because the cbuffer
    //exposes them as float for uniform scalar packing.
    float atmos_viewSteps;
    float atmos_lightSteps;
    float atmos_aerialViewSteps;
    float atmos_aerialLightSteps;
    float atmos_multiScatterFactor;
    float atmos_cloudShadowConeDeg;
    float atmos_cloudShadowFloor;
    float atmos_earthShadowSoftness;
    //volumetric cloud knobs, mirror CloudSettings in Common.h (same order!)
    //Scalar packing concatenates these into the existing cbuffer register
    //layout — no padding needed because every field is float.
    float cloud_enabled;
    float cloud_coverage;
    float cloud_layerBotKm;
    float cloud_layerTopKm;
    float cloud_horizonFadeKm;
    float cloud_extinction;
    float cloud_baseFrequency;
    float cloud_hfFrequency;
    float cloud_hfAmount;
    //Nubis-3 silver lining: amplitude + spread of the narrow forward
    //HG lobe max-blended into the primary phase. Replaces the old
    //J&D droplet diameter knobs.
    float cloud_silverIntensity;
    float cloud_silverSpread;
    //Defocus cone half angle for cone traced sun shadows (degrees).
    //0 = strict sun direction.
    float cloud_shadowConeDeg;
    //Nubis-3 secondary multi-scatter phase: strength of the broader
    //HG term and its eccentricity. Replaces the Wrenninge octave
    //triple (a, b, c) which is no longer used.
    float cloud_secondaryStrength;
    float cloud_secondaryG;
    float cloud_windX;
    float cloud_windZ;
    float cloud_trEps;
    //indirect lighting on cloud samples
    float cloud_skyAmbient;
    float cloud_groundBounce;
    float cloud_groundAlbedo;
    float cloud_skyAmbientScale;
    float cloud_groundScale;
    //surface shadowing and termination
    float cloud_cloudShadowOnSurfaces;
    float cloud_rrThreshold;
    //Sun shadow ray count per scattering event (1 = strict sun dir,
    //3..5 = soft self shadows when shadowConeDeg > 0) and the upward
    //density step count for the sky ambient occlusion estimate.
    float cloud_shadowConeSamples;
    float cloud_ambientSteps;
    //Adaptive march loop bound (main path) + base step size. The
    //big perf levers for the primary view march.
    float cloud_viewStepsMax;
    float cloud_targetStepKm;
    //Surface shadow march sample count. 1 = fast single-sample
    //sphere-intersect path, 2..6 = multi-tap shell march.
    float cloud_shadowSteps;
    //Bounce-ray cheap path: step count and max march length (km).
    float cloud_cheapSteps;
    float cloud_cheapMaxLenKm;
    //Cloud-eval distance window (fade start / hard clamp).
    float cloud_fadeDistanceKm;
    float cloud_renderDistanceKm;
    //Aerial-perspective haze multiplier in front of clouds.
    float cloud_hazeStrength;
    //Per-cloud top altitude jitter — noise amplitude (km) and
    //horizontal noise frequency (1/km). Controls towering cumulus.
    float cloud_topVariationKm;
    float cloud_topFrequency;
    //Coverage modulation soft edge width + low-frequency domain warp
    //amplitude (km). Filter width sharpens / softens silhouettes;
    //warp amp breaks up the noise grid pattern.
    float cloud_covModFilterWidth;
    float cloud_warpAmpKm;
    //Per-channel single-scattering albedo (white cloud ≈ 0.995).
    float cloud_albedoR;
    float cloud_albedoG;
    float cloud_albedoB;
    //Nubis Evolved multi-scatter: global amplitude + base floor that
    //keeps cumulus bottoms from going black at h=0.
    float cloud_msStrength;
    float cloud_msHeightFloor;
    //Multi-scatter model selector (cast to int at the use site).
    //0 = Nubis sqrt(Tdir) shortcut, 1 = 2 octave Wrenninge, 2 = 3 octave.
    float cloud_msMode;
    //Sky ambient probe: brightness, AO scale on the column density
    //above the sample, and max optical depth cap so dense overcast
    //columns can actually shut the sky term down.
    float cloud_ambientIntensity;
    float cloud_ambientAOScale;
    float cloud_ambientODMax;
    //Multiplier on the sun shadow optical depth (>1 darker self
    //shadow, <1 brighter).
    float cloud_sunTauMult;
    //Distance LOD blend band — full quality below near, simplified
    //above far.
    float cloud_lodNearKm;
    float cloud_lodFarKm;
    //Adaptive march bounds: small step cap, geometric growth factor,
    //zero-density floor, big empty-space step cap + distance growth,
    //fine step ceiling.
    float cloud_maxStepKm;
    float cloud_stepGrowth;
    float cloud_effectiveZeroDensity;
    float cloud_maxEmptyStepKm;
    float cloud_emptyStepGrowthPerKm;
    float cloud_maxFineStepKm;
    //Nubis-3 light-energy shape (exposed 2026-06-13; were CLOUD_N3_* defines).
    //Forward phase eccentricity, MS extinction scale (surface/glow), engine
    //MS brightness gain, and the inner-glow sun-dot + in-cloud-depth drivers.
    float cloud_n3PhaseG;
    float cloud_n3MsBase;
    float cloud_n3MsGlow;
    float cloud_n3MsBrightness;
    float cloud_n3GlowSunDot;
    float cloud_n3GlowDepthKm;
    //Cauliflower shape detail (exposed 2026-06-13; were CLOUD_LOBE_*/BILLOW_*).
    //Lobe signed amplitude + freq (×base), billow seam carve / centre bulge /
    //seam sharpness + freq (×base). Vertical height ramps stay compile-time.
    float cloud_lobeAmount;
    float cloud_lobeFreqMult;
    float cloud_billowAmount;
    float cloud_billowBulge;
    float cloud_billowSharp;
    float cloud_billowFreqMult;
    //Wisps: thin wind-sheared filaments beyond the body (pre-coverage additive
    //octave). amount = reach, freqMult = filament fineness (×base), stretch =
    //wind-shear elongation. Vertical band + additive bias stay compile-time.
    float cloud_wispAmount;
    float cloud_wispFreqMult;
    float cloud_wispStretch;
    //====================================
    //PLANET TERRAIN (Phase 5)
    //====================================
    //Procedural cube-sphere terrain parameters. planetCenter* is ABSOLUTE world
    //space - subtract sceneOriginWorld for camera-local (shifted) math. Six
    //scalar floats, matching the Camera::UploadGPUBuffer tail exactly (scalar
    //packing avoids the float3 16-byte-straddle padding the compiler would
    //otherwise insert and which the C++ side does not write).
    //planetCenter/Radius are read by the cloud + atmosphere code (TerrainHeight,
    //planet sphere intersection). amplitude/frequency are vestigial (the terrain
    //is now a baked heightmap + real mesh), kept as zero to preserve the tail
    //layout that Camera::UploadGPUBuffer writes.
    float planetCenterX;
    float planetCenterY;
    float planetCenterZ;
    float planetRadius;
    float terrainHeightAmplitude;
    float terrainHeightFrequency;
}

#define SUN_LATITUDE_DEG    sunLatitude
#define SUN_LONGITUDE_DEG   sunLongitude
#define SUN_DAY_OF_YEAR     sunDayOfYear
#define SUN_SIM_SPEED       sunSimSpeed
#define SUN_START_UTC_HOURS sunStartUTCHours
#define SUN_NIGHT_SPEEDUP   sunNightSpeedup
#define SUN_TURBIDITY       sunTurbidity
#define SUN_INTENSITY_VAL   sunSunIntensity
//Sky integrates atmospheric in-scatter in units of ATMOS_SOLAR_IRRADIANCE
//(=1 in the model). For the sky to read in the same scene units as the path-
//traced sun, the final scale must equal the sun's irradiance. sunSkyIntensity
//therefore acts as a multiplicative artistic boost on top of physical (1.0 =
//physically calibrated, >1 = sky pops more than ground, <1 = muted sky).
#define SKY_INTENSITY_VAL   (sunSunIntensity * sunSkyIntensity)
#define SKY_INTENSITY       (sunSunIntensity * sunSkyIntensity)
#define GLOBAL_EMISSION_STRENGTH globalEmissionStrength

//Atmosphere quality + look redirects. SunSampler_v8.hlsli wraps each ATMOS_*
//constant in #ifndef, so defining them here (which is included before that
//file at line 467) overrides the static fallbacks and binds the atmosphere
//integrator to the editor driven cbuffer fields. Loop bounds cast to int
//at the use site because the CB exposes them as float for uniform packing.
#define ATMOS_VIEW_STEPS              ((int)atmos_viewSteps)
#define ATMOS_LIGHT_STEPS             ((int)atmos_lightSteps)
#define ATMOS_AERIAL_VIEW_STEPS       ((int)atmos_aerialViewSteps)
#define ATMOS_AERIAL_LIGHT_STEPS      ((int)atmos_aerialLightSteps)
#define ATMOS_MULTI_SCATTER_FACTOR    atmos_multiScatterFactor
//Atmosphere look knobs not used inside SunSampler_v8.hlsli — read directly
//by Clouds_v8.hlsli (cloud shadow tap on atmospheric samples, earth shadow
//smoothstep band). The macros below give the cloud integrator a stable
//name in case the cbuffer field is renamed later.
#define ATMOS_CLOUD_SHADOW_CONE_DEG   atmos_cloudShadowConeDeg
#define ATMOS_CLOUD_SHADOW_FLOOR      atmos_cloudShadowFloor
// Exponent applied to cloud visibility when shadowing atmospheric in-scatter.
// 1.0 = the real single-scatter shadow. The old 0.3 softening stood in for
// multi-scattered fill light, but it brightened the DIRECTIONAL term — air
// under an opaque deck kept scattering 10-30% of full sun with the clear-sky
// phase/spectrum (blue band away from the sun, orange Mie glow toward it).
// That fill role moved to the explicit isotropic through-deck source
// (CloudShadowAmbientTerms) + the Hillaire MS LUT, so the directional term
// now takes the physical shadow.
#define ATMOS_CLOUD_SHADOW_SOFTNESS   1.0f
#define ATMOS_EARTH_SHADOW_SOFTNESS   atmos_earthShadowSoftness
// Surface cloud shadow: softness exponent and cone half-angle in degrees
// for spatial blur (0 = sharp point sample). Softness 1.0 = the physical
// single-scatter shadow: a tau-8 deck passes exp(-8) ~ 0.03% direct sun,
// not the 9% the old 0.3 exponent leaked (same band-aid class as the
// retired atmosphere softness — the diffuse light on ground under cloud
// comes from path-traced sky GI off the bright deck, not from softened
// direct sun).
#define SURFACE_CLOUD_SHADOW_SOFTNESS 1.0f
#define SURFACE_CLOUD_SHADOW_CONE_DEG 3.0f

//Volumetric cloud knob redirects. Clouds_v8.hlsli wraps each constant in
//#ifndef so defining them here overrides the static fallbacks and binds
//the cloud integrator to the editor-driven CB fields. Loop bounds are
//cast to int at the use site because the CB exposes them as float for
//uniform packing.
#define CLOUD_COVERAGE_BASE     cloud_coverage
#define CLOUD_LAYER_BOT_KM      cloud_layerBotKm
#define CLOUD_LAYER_TOP_KM      cloud_layerTopKm
// (cloud_horizonFadeKm has no consumer — INERT field kept for cbuffer
// layout, no macro so dead usage can't silently come back.)
#define CLOUD_EXTINCTION        cloud_extinction
#define CLOUD_BASE_FREQ              cloud_baseFrequency
#define CLOUD_HF_FREQ                cloud_hfFrequency
#define CLOUD_HF_AMOUNT              cloud_hfAmount
#define CLOUD_SILVER_INTENSITY       cloud_silverIntensity
#define CLOUD_SILVER_SPREAD          cloud_silverSpread
#define CLOUD_SHADOW_CONE_DEG        cloud_shadowConeDeg
#define CLOUD_SECONDARY_STRENGTH     cloud_secondaryStrength
#define CLOUD_SECONDARY_G            cloud_secondaryG
#define CLOUD_WIND_X                 cloud_windX
#define CLOUD_WIND_Z                 cloud_windZ
#define CLOUD_TR_EPS                 cloud_trEps
#define CLOUD_SKY_AMBIENT            cloud_skyAmbient
#define CLOUD_GROUND_BOUNCE          cloud_groundBounce
#define CLOUD_GROUND_ALBEDO          cloud_groundAlbedo
#define CLOUD_SKY_AMBIENT_SCALE      cloud_skyAmbientScale
#define CLOUD_GROUND_SCALE           cloud_groundScale
#define CLOUD_RR_THRESHOLD           cloud_rrThreshold
#define CLOUD_SHADOW_CONE_SAMPLES    cloud_shadowConeSamples
#define CLOUD_AMBIENT_STEPS          ((int)cloud_ambientSteps)
//Adaptive march bounds + step (override the static fallbacks in
//Clouds_v8.hlsli so the editor drives them at runtime).
#define CLOUD_VIEW_STEPS_MAX         ((int)cloud_viewStepsMax)
#define CLOUD_TARGET_STEP_KM         cloud_targetStepKm
//Surface shadow march sample count — drives CloudOpticalDepthAlongRay
//(surface NEE shadow). Set to 1 to take the fast sphere-intersect path.
#define CLOUD_SHADOW_STEPS           ((int)cloud_shadowSteps)
//Bounce-ray cheap path knobs — drive EvaluateCloudsCheap so bounce
//rays respect the editor's quality/perf trade-off.
#define CLOUD_CHEAP_STEPS            ((int)cloud_cheapSteps)
#define CLOUD_CHEAP_MAX_LEN_KM       cloud_cheapMaxLenKm
//Cloud-eval distance window (fade + hard clamp).
#define CLOUD_FADE_DISTANCE_KM       cloud_fadeDistanceKm
#define CLOUD_RENDER_DISTANCE_KM     cloud_renderDistanceKm
// (cloud_hazeStrength has no consumer since the unified march — INERT
// field kept for cbuffer layout, macro removed.)
//Top altitude variability (per-cloud tops).
#define CLOUD_TOP_VARIATION_KM       cloud_topVariationKm
#define CLOUD_TOP_FREQ               cloud_topFrequency
//Density shaping.
#define CLOUD_COVMOD_FILTER_WIDTH    cloud_covModFilterWidth
#define CLOUD_WARP_AMP_KM            cloud_warpAmpKm
//Cloud albedo as a packed float3 from three scalar cbuffer slots
//(scalar packing keeps the surrounding fields aligned without
//manual padding).
#define CLOUD_ALBEDO                 float3(cloud_albedoR, cloud_albedoG, cloud_albedoB)
//Multi-scatter.
#define CLOUD_MS_STRENGTH            cloud_msStrength
#define CLOUD_MS_HEIGHT_FLOOR        cloud_msHeightFloor
//Multi-scatter mode selector (0 = shortcut, 1 = 2-octave, 2 = 3-octave).
//Cast at the use site because the cbuffer exposes it as float for uniform
//packing — matches the same pattern as CLOUD_VIEW_STEPS_MAX et al.
#define CLOUD_MS_MODE                ((int)cloud_msMode)
//Sky ambient.
#define CLOUD_AMBIENT_INTENSITY      cloud_ambientIntensity
#define CLOUD_AMBIENT_AO_SCALE       cloud_ambientAOScale
#define CLOUD_AMBIENT_OD_MAX         cloud_ambientODMax
//Sun shadow tau multiplier.
#define CLOUD_SUN_TAU_MULT           cloud_sunTauMult
//Distance LOD.
#define CLOUD_LOD_NEAR_KM            cloud_lodNearKm
#define CLOUD_LOD_FAR_KM             cloud_lodFarKm
//Adaptive march step bounds.
#define CLOUD_MAX_STEP_KM            cloud_maxStepKm
#define CLOUD_STEP_GROWTH            cloud_stepGrowth
#define CLOUD_EFFECTIVE_ZERO_DENSITY cloud_effectiveZeroDensity
#define CLOUD_MAX_EMPTY_STEP_KM      cloud_maxEmptyStepKm
#define CLOUD_EMPTY_STEP_GROWTH_PER_KM cloud_emptyStepGrowthPerKm
#define CLOUD_MAX_FINE_STEP_KM       cloud_maxFineStepKm
//Nubis-3 light-energy shape (exposed 2026-06-13). Override the Clouds_v8.hlsli
//#ifndef fallbacks with the editor-driven CB fields.
#define CLOUD_N3_PHASE_G             cloud_n3PhaseG
#define CLOUD_N3_MS_BASE             cloud_n3MsBase
#define CLOUD_N3_MS_GLOW             cloud_n3MsGlow
#define CLOUD_N3_MS_BRIGHTNESS       cloud_n3MsBrightness
#define CLOUD_N3_GLOW_SUNDOT         cloud_n3GlowSunDot
#define CLOUD_N3_GLOW_DEPTH_KM       cloud_n3GlowDepthKm
//Cauliflower shape detail. The two FREQ macros are multiples of the base
//frequency, matching the Clouds_v8.hlsli fallback form.
#define CLOUD_LOBE_AMOUNT            cloud_lobeAmount
#define CLOUD_LOBE_FREQ              (CLOUD_BASE_FREQ * cloud_lobeFreqMult)
#define CLOUD_BILLOW_AMOUNT          cloud_billowAmount
#define CLOUD_BILLOW_BULGE           cloud_billowBulge
#define CLOUD_BILLOW_SHARP           cloud_billowSharp
#define CLOUD_BILLOW_FREQ            (CLOUD_BASE_FREQ * cloud_billowFreqMult)
//Wisps: thin wind-sheared filaments beyond the body (pre-coverage additive).
#define CLOUD_WISP_AMOUNT            cloud_wispAmount
#define CLOUD_WISP_FREQ              (CLOUD_BASE_FREQ * cloud_wispFreqMult)
#define CLOUD_WISP_STRETCH           cloud_wispStretch

//PLANET: equiangular cubed-sphere lookup. Used by every per-direction
//terrain sample (heightmap, surface_color, normal, cloud_offset). Mirror
//of tools/planetbaker/src/core/cubed_sphere.h::sphere_to_face_uv so the
//baker's pixel layout lines up with this lookup texel-for-texel.
inline void SphereToEquiangularFaceUV(float3 dir, out int face, out float2 uv)
{
    float3 a = abs(dir);
    float  ut, vt;
    if (a.x >= a.y && a.x >= a.z) {
        if (dir.x > 0.0f) { face = 0; ut = -dir.z / a.x; vt = -dir.y / a.x; }
        else              { face = 1; ut =  dir.z / a.x; vt = -dir.y / a.x; }
    } else if (a.y >= a.x && a.y >= a.z) {
        if (dir.y > 0.0f) { face = 2; ut =  dir.x / a.y; vt =  dir.z / a.y; }
        else              { face = 3; ut =  dir.x / a.y; vt = -dir.z / a.y; }
    } else {
        if (dir.z > 0.0f) { face = 4; ut =  dir.x / a.z; vt = -dir.y / a.z; }
        else              { face = 5; ut = -dir.x / a.z; vt = -dir.y / a.z; }
    }
    //Equiangular: u,v = atan(*) * 4/PI in [-1, +1]. Tangent-warped
    //cube projection -> face-local equal-area cells (the baker's pixel
    //layout; a linear cube projection would mis-align).
    const float kInv = 4.0f / 3.14159265358979f;
    uv = float2(atan(ut), atan(vt)) * kInv * 0.5f + 0.5f;
}

//PLANET: surface elevation (metres) along a unit direction. Samples the
//baker cubemap via equiangular cubed-sphere projection.
//Used by Inline_RT_v8.hlsli (normal finite-difference + ReSTIR reconnect),
//Clouds_v8.hlsli (terrain shadow occluder), SunSampler_v8.hlsli (terrain
//shadow at closest approach). Adding to planetRadius gives the surface
//point along `dir`.
inline float TerrainHeight(float3 dir)
{
    int    face;
    float2 uv;
    SphereToEquiangularFaceUV(dir, face, uv);
    //Bilinear + clamp via g_sampler_LUT (s1). Explicit LOD 0 - the texture
    //is mip-0 only. .r is the elevation in KILOMETRES; convert to m so
    //the cbuffer math stays in metres.
    float km = g_terrainHeightmap.SampleLevel(g_sampler_LUT, float3(uv, (float)face), 0.0f);
    return km * 1000.0f;
}

//PLANET: smoothed surface elevation (metres) used as the local cloud-base
//reference. Same projection as TerrainHeight but reads the much smaller
//cloud_offset cubemap (typically 256^2 per face) so individual mountain
//peaks don't perturb the cloud base. Clouds_v8.hlsli adds this to
//CLOUD_LAYER_BOT_KM to get the per-direction cloud bottom altitude.
inline float TerrainCloudBaseHeight(float3 dir)
{
    int    face;
    float2 uv;
    SphereToEquiangularFaceUV(dir, face, uv);
    float km = g_terrainCloudOffset.SampleLevel(g_sampler_LUT,
                                                 float3(uv, (float)face), 0.0f);
    return km * 1000.0f;
}

//PLANET: per-vertex-UV terrain tint. `faceMarker` is InstanceProperties._pad[0]
//(0 = scene mesh, 1..6 = terrain on cube face 0..5). `uv` is the equiangular
//face UV baked into the terrain vertices by the tessellator, already
//barycentric-interpolated by EvalSurfaceState - so this is just an array
//sample, no atan. Returns false (leaving Kd untouched) for scene meshes and
//when the surface_color layer is absent: a null SRV samples 0, and the baker
//forces alpha = 1 on real texels, so alpha <= 0 is the "fall back to the flat
//TERRAIN_ALBEDO material" sentinel.
inline bool TerrainTintFromUV(float2 uv, uint faceMarker, out float3 kd)
{
    kd = float3(0.0f, 0.0f, 0.0f);
    if (faceMarker == 0u) return false;
    const float face = (float)(faceMarker - 1u);
    const float4 c = g_terrainSurfaceColor.SampleLevel(g_sampler_LUT,
                                                       float3(uv, face), 0.0f);
    if (c.a <= 0.0f) return false;
    kd = c.rgb;
    return true;
}

//====================================
//CORE UTILITY HEADERS
//====================================
#include "Constants_v8.hlsli"
#include "Common_v8.hlsli"
#include "Data_v8.hlsli"
#include "Random_v8.hlsli"
#include "Compression_v8.hlsli"

//====================================
//OUTPUT SCRATCH RESERVOIR BUFFERS
//====================================
RWTexture2DArray<float4> gOutput             : register(u0);
RWTexture2D<float4>      gPermanentData      : register(u1);
RWTexture2DArray<float4> gScratchPing        : register(u8);

RWByteAddressBuffer g_sample_current         : register(u6);
RWByteAddressBuffer g_sample_last            : register(u7);
RWByteAddressBuffer g_Reservoirs_current     : register(u4);
RWByteAddressBuffer g_Reservoirs_last        : register(u5);
//SPMIS_GRID_NONCOHERENT: the SPMIS reuse passes (select/shift/merge) only READ a
//finished, barrier-fenced grid + scratch, so they drop globallycoherent to get
//L1-cached reads. The coherent path bypasses L1 and saturates L2 on the scattered
//cell-search / inner-RIS gathers (the Pass_spmis_select L2 bottleneck). Every WRITER
//and same-dispatch cross-group READER (raygen CAS hash insert, count/offsets atomics)
//keeps coherence by simply never defining the macro.
#ifdef SPMIS_GRID_NONCOHERENT
RWByteAddressBuffer g_pathStateBuffer        : register(u10);
RWByteAddressBuffer g_spmisBuffer            : register(u25);
#else
//globallycoherent so the raygen scratch+reload pattern (tpost, nrcA0)
//actually drops those values from live state across the bounce-loop
//TraceRay reorder boundary. Without it, DXC store-to-load forwards the
//scratch values and keeps them register-resident across the trace,
//defeating the spill entirely. Spat GI pass takes a small cache hit on
//its cross-pixel reads; revert if that shows up in profiling.
globallycoherent RWByteAddressBuffer g_pathStateBuffer        : register(u10);
//SPMIS global hash grid — single raw buffer (root UAV at u25) holding every per-pixel
//and per-cell SPMIS array (see HashGridHash_v8.hlsli for the sub-allocation layout).
//globallycoherent so the raygen hash-insertion CAS probes observe other threadgroups'
//writes within the same dispatch (the table is shared across all pixels).
globallycoherent RWByteAddressBuffer g_spmisBuffer            : register(u25);
#endif

//====================================
//AUTO EXPOSURE STATE (20 B persistent)
//====================================
//[0]  uint  AE_OFFS_SUM        — fixed-point log2-luminance sum (cleared each frame)
//[4]  float AE_OFFS_SMOOTHED   — temporally smoothed log2-luminance (read by postprocess)
//[8]  uint  AE_OFFS_INIT       — first-frame flag, 0 = uninitialized, 1 = ready
//[12] uint  AE_OFFS_TILE_COUNT — contributing tile count (cleared each frame)
//[16] float AE_OFFS_PREV_TIME  — last frame's CameraParams.time, drives dt-based smoothing
//fixed-point packing: store uint((log2Lum + AE_LOG_OFFSET) * AE_LOG_SCALE) per group.
//log2 units are clamped to [-AE_LOG_OFFSET, +AE_LOG_OFFSET] before packing so the sum
//stays inside uint32 at 4K: tilesAt4K(32400) * 2*OFFSET*SCALE = 32400*224 ≈ 7.3M, safe.
RWByteAddressBuffer gAutoExpose              : register(u24);
static const uint  AE_OFFS_SUM        = 0u;   // fixed-point log2-lum sum (cleared each frame)
static const uint  AE_OFFS_SMOOTHED   = 4u;   // float, persistent
static const uint  AE_OFFS_INIT       = 8u;   // uint flag, persistent
static const uint  AE_OFFS_TILE_COUNT = 12u;  // uint, contributing tile count (cleared each frame)
static const uint  AE_OFFS_PREV_TIME  = 16u;  // float, persistent
static const float AE_LOG_OFFSET      = 14.0f;
static const float AE_LOG_SCALE       =  8.0f;

//====================================
//SCENE DATA
//====================================
StructuredBuffer<STriVertex>         BTriVertex          : register(t2);
StructuredBuffer<int>                indices             : register(t1);
RaytracingAccelerationStructure      SceneBVH            : register(t0);
StructuredBuffer<InstanceProperties> instanceProps       : register(t3);
StructuredBuffer<uint>               materialIDs         : register(t4);

//40B compressed AoS, accessors in Material_Decoder_v8.hlsli
StructuredBuffer<MatPacked>          g_mat               : register(t5);

StructuredBuffer<LightTriangle>      g_EmissiveTriangles : register(t6);
StructuredBuffer<uint>               gTriToLightId       : register(t15);

//(g_terrainHeightmap at t45 is declared earlier, with the other SRVs, so
// the TerrainHeight() function - used by the cloud/atmosphere code - can see
// it. HLSL is single-pass.)

//====================================
//LIGHT TREE
//====================================
StructuredBuffer<LightTLASNodeGpu> gLT_TLAS         : register(t9);
StructuredBuffer<LightBLASNodeGpu> gLT_BLAS         : register(t10);
StructuredBuffer<BlasRangeGpu>     gLT_Range        : register(t11);
Buffer<uint>                       gLT_LeafTriIndex : register(t12);
//t16/t17/t18 used by LightTree_v8.hlsli lookup buffers

//====================================
//SHADING AND MATERIAL HEADERS
//====================================
#include "Material_Decoder_v8.hlsli"
#include "LightTree_v8.hlsli"
#include "Sample_Data_v8.hlsli"
#include "Fresnel_v8.hlsli"
#include "Material_Common_v8.hlsli"
#include "Material_GGX_v8.hlsli"
#include "Material_Lambertian_v8.hlsli"
#include "Material_Coat_v8.hlsli"
#include "Material_Sheen_v8.hlsli"
#include "BXDF_v8.hlsli"

//====================================
//SAMPLING AND RESERVOIRS
//====================================
#include "Path_Sampler_v8.hlsli"
#include "SunSampler_v8.hlsli"
#include "procedural_terrain_v8.hlsli"
#include "Inline_RT_v8.hlsli"
#include "Reservoir_v8.hlsli"
#include "Path_State_v8.hlsli"
#include "HashGridHash_v8.hlsli"

#include "Camera_ray_v8.hlsli"
#include "MIS_v8.hlsli"

//====================================
//DLSS RR RESOURCES
//====================================
#ifdef COMPUTE_PASS
RWTexture2D<float>  g_dlssDepth          : register(u11);
RWTexture2D<float2> g_dlssMVec           : register(u12);
RWTexture2D<float4> g_dlssNormals        : register(u13);
RWTexture2D<float4> g_dlssDiffuseAlbedo  : register(u14);
RWTexture2D<float4> g_dlssOutput         : register(u15);
RWTexture2D<float4> g_dlssSpecularAlbedo : register(u16);
RWTexture2D<float>  g_dlssRoughness      : register(u17);
RWTexture2D<float2> g_dlssSpecMVec       : register(u18);
RWTexture2D<float>  g_dlssSpecHitDist    : register(u19);
RWTexture2D<float4> g_dlssTransparency   : register(u20);
RWTexture2D<float4> g_dlssColorPreTrans  : register(u21);
RWTexture2D<float4> g_dlssInput          : register(u22);
RWTexture2D<float>  g_dlssBiasHint       : register(u23);
#endif

//====================================
//DEFERRED REUSE VISIBILITY  (RS_FLAG_NO_REUSE_VIS)
//====================================
//One reconnection shadow ray for the resolved reservoir at this pixel, called by
//the spatial passes where they write F*W to scratch slot 2. Returns 1.0 when the
//flag is off or the contribution carries no energy, so the default path (visibility
//baked inside the reuse passes) stays byte-identical. See RS_FLAG_NO_REUSE_VIS.
#ifdef COMPUTE_PASS
inline float ResolveReuseVis(uint pixelIdx, Reservoir r, float3 contrib)
{
    if (!REUSE_VIS_OFF || FINAL_VIS_OFF || GetPHat(contrib) <= 0.0f) return 1.0f;
    const float3        x1 = load_x1(g_sample_current, pixelIdx);
    const SurfaceVertex sv = BuildVertex(g_sample_current, pixelIdx, x1, InitOrigin());
    return ReconnectVis(sv.x, sv.n_s, r.matID, r.x2, r.n2_s);
}

#endif

#endif
