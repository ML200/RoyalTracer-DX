#pragma once
//====================================
//SHARED TYPES MACROS FORWARD DECLS
//====================================

#include <d3d12.h>
#include <dxgi1_4.h>
#include <DirectXMath.h>
#include <DirectXPackedVector.h>
#include <wrl/client.h>
#include <wrl/wrappers/corewrappers.h>

#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <chrono>
#include <algorithm>

#include "glm/gtc/matrix_transform.hpp"
#include "../src/Components/Vertex.h"
#include "d3dx12.h"

using Microsoft::WRL::ComPtr;
using namespace DirectX;

//====================================
//LOGGING
//====================================
#ifndef LT_ENABLE_LOGS
#define LT_ENABLE_LOGS 1
#endif

#if LT_ENABLE_LOGS
  #define LOG(expr)  do { std::wcout << L"[Engine] "      << expr << std::endl; } while(0)
  #define WARN(expr) do { std::wcout << L"[Engine][WARN] " << expr << std::endl; } while(0)
#else
  #define LOG(expr)  do {} while(0)
  #define WARN(expr) do {} while(0)
#endif

//====================================
//SCOPED CPU TIMER
//====================================
struct ScopedTimer {
    const char* name;
    std::chrono::high_resolution_clock::time_point t0;
    ScopedTimer(const char* n) : name(n), t0(std::chrono::high_resolution_clock::now()) {}
    ~ScopedTimer() {
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - t0).count();
        std::wcout << L"[CPU] " << name << L" took " << ms << L" ms" << std::endl;
    }
};
#define SCOPE_TIMER(label) ScopedTimer _scopedTimer_##__LINE__(label)

//====================================
//CONSTANTS
//====================================
static constexpr UINT  FRAME_COUNT          = 3;
static constexpr UINT  MAX_BACK_BUFFERS     = 6;
static constexpr UINT  MAX_STACKS           = 4;
static constexpr UINT  MAX_INDIRECT_COMMANDS = MAX_STACKS;
static constexpr UINT  SORT_BUCKETS         = 65536;
static constexpr int   NUM_LUTS             = 2;
static constexpr int   LUT_RESOLUTION       = 16;
static constexpr int   NUM_SAMPLES_LUT      = 32000;
//NRC reserves 5 UAVs at heap 58..62, autoexpose at heap 63, sky stars SRV at
//heap 64 (register t40), cloud noise 3D SRV at heap 65 (register t42), cloud
//coverage 2D SRV at heap 66 (register t43, NASA Blue Marble equirect map),
//spatiotemporal blue noise array SRV at heap 67 (register t41, baked once by
//BakeCloudSTBNTexture). Bindless starts at 68.
static constexpr UINT  AUTOEXPOSE_HEAP_SLOT     = 63;
static constexpr UINT  SKY_STARS_HEAP_SLOT      = 64;
static constexpr UINT  CLOUD_NOISE_HEAP_SLOT    = 65;
static constexpr UINT  CLOUD_COVERAGE_HEAP_SLOT = 66;
static constexpr UINT  CLOUD_STBN_HEAP_SLOT     = 67;
static constexpr UINT  BINDLESS_HEAP_START      = 68;

static constexpr D3D12_RESOURCE_STATES kSRV =
    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE |
    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;

//====================================
//GPU VERTEX LAYOUT
//====================================
struct BTriVertex {
    XMFLOAT3                       vertex;
    UINT                           packedNormal;
    PackedVector::XMHALF2         texCoord;
};

//====================================
//PER-INSTANCE GPU DATA
//====================================
struct InstanceProperties {
    XMMATRIX objectToWorld;
    XMMATRIX objectToWorldInverse;
    XMMATRIX prevObjectToWorld;
    XMMATRIX prevObjectToWorldInverse;
    XMMATRIX objectToWorldNormal;
    XMMATRIX prevObjectToWorldNormal;
    UINT     indexBase;
    UINT     vertexBase;
    UINT     materialBase;
    UINT     triToLightBase;
    UINT     opaqueTriCount;
    UINT     _pad[3];
};

//====================================
//GEOMETRY OFFSETS
//====================================
struct GeometryOffsets {
    UINT vertexBase;
    UINT indexBase;
    UINT materialBase;
};

//====================================
//RESTIR RUNTIME SETTINGS
//====================================
//DI+GI unified, DI knobs removed
struct ReSTIRSettings {
    int   tempMcapGI       = 8;
    int   spatCountMaxGI   = 2;
    int   spatCountMinGI   = 2;
    int   spatRadMaxGI     = 56;
    int   spatRadMinGI     = 8;
    int   spatTriesGI      = 8;
    bool  enableTempGI     = true;
    bool  enableSpatGI     = true;
    float reuseRoughnessMin = 0.1f;
    float reuseRoughnessMax = 0.3f;

    //neighbor rejection thresholds for Pass_spat_gi_select_v8
    float rejNormalDot     = 0.36f;
    float rejDistance      = 0.10f;

    UINT Flags() const {
        //bits 0 (tempDI) and 2 (spatDI) stay zero, DI pipeline gone
        return (enableTempGI ? 2u : 0u) | (enableSpatGI ? 8u : 0u);
    }
};

//====================================
//DLSS-G FRAME GEN SETTINGS
//====================================
struct DLSSGSettings {
    bool available        = false;
    bool enabled          = false;
    int  framesToGenerate = 1;
    int  maxFrames        = 1;
};

//====================================
//SUN TIME-OF-DAY SETTINGS
//====================================
//Naming note: `latitude` / `longitude` are the OBSERVER's lat/lon on the
//planet, not the sun's. The sun direction is derived from these (the
//standard astronomy formula in GetSunDirAndElev uses observer lat as `L`
//and observer lon to offset the local solar hour), and the cloud coverage
//map's equirectangular projection is rotated by these so the equirect
//pole follows the planet's actual north pole instead of the observer's
//overhead direction (without this, the cloud zenith pinches at the
//equirect singularity — see EnuToPlanetDir in Clouds_v8.hlsli).
//
//World coordinates are the observer's local ENU frame at (latitude,
//longitude): +X = east, +Y = up (planet radial at observer),
//+Z = north. All atmosphere math (transmittance, scattering, sun
//direction) operates in this local frame, which is correct for any
//lat/lon since the relevant quantities (altitude, zenith angles) are
//scalar. Only the cloud coverage map needs the ENU→planet rotation
//because it's the only thing keyed by absolute planet geography.
struct SunSettings {
    float latitude      = 48.52f;   // observer latitude  (degrees, -90..90)
    float longitude     = 11.405f;  // observer longitude (degrees, -180..180)
    float dayOfYear     = 172.0f;
    float simSpeed      = 10.0f;
    float startUTCHours = 6.0f;
    float nightSpeedup  = 2.0f;
    float turbidity     = 2.0f;
    float sunIntensity  = 5.0f;
    //multiplicative boost on the physically calibrated sky brightness (which is
    //internally tied to sunIntensity). 1.0 = real-world sun-to-sky ratio,
    //higher values make the sky pop more than ground (stylized).
    float skyIntensity  = 1.0f;
    float globalEmissionStrength = 1.0f;
    //thin-lens DoF, populated from Camera::apertureRadius / focusDistance
    //during UploadGPUBuffer, lives in this struct so the cbuffer tail stays 16-byte aligned
    float dofApertureRadius = 0.0f;
    float dofFocusDistance  = 10.0f;
    //star skybox tuning (EvaluateStars in SunSampler_v8.hlsli). Runtime
    //knobs so the artist can dial in sparkle vs bloom without recompiling.
    //intensity = final brightness multiplier (after gamma)
    //gamma     = power curve on luminance; >1 darkens mid tones (bilinear
    //            mip smear) and preserves peaks (star centres). 1.0 = off.
    //lodBias   = additional mip offset on the footprint based pick. Higher
    //            = blurrier and more stable under jitter, lower = sharper.
    //threshold = black level lift applied before the gamma curve. Cuts the
    //            soft halo bilinear filtering creates around each star
    //            (the "blob"), so visible star footprint shrinks to the
    //            bright centre. Subtraction is hard clamped at 0.
    //Tuned for the 8K NASA SVS EXR + mipmap chain: low LOD bias to grab
    //sharp detail (the gamma + threshold tricks are no longer needed once
    //the source resolution is high enough that single stars are 1 texel
    //even at mip 0), gamma left at the linear identity, threshold at zero.
    //Intensity dialed down so the bright Milky Way doesn't overpower the
    //rest of the night sky after AE.
    float skyStarIntensity = 0.047f;
    float skyStarGamma     = 1.57f;
    float skyStarLodBias   = -1.0f;
    float skyStarThreshold = 0.0f;
    //Scalar multiplier on the SKY_NIGHT_BASE airglow tint. 1.0 is the
    //literal SKY_NIGHT_BASE value (sun-independent, matches the chromatic
    //balance picked in SunSampler_v8.hlsli). AE compensates for the dim
    //absolute luminance via the AE_LOG_LUM_MIN floor in
    //Pass_autoexpose_finalize_v8.hlsl, so a clear night still reads as
    //"dim" rather than "black". Higher = stylized brighter night.
    float skyNightBaseIntensity = 0.57f;

    //====================================
    // atmosphere quality + look (Bruneton march)
    //====================================
    //Per ray sample count along the view ray inside the atmosphere shell.
    //Drives the dominant cost of the atmosphere integration (per pixel,
    //per cloud shell phase). 12 is the Bruneton baseline; raise for
    //smoother haze gradients, drop to 6..8 for cheap preview modes where
    //banding is acceptable.
    float atmosViewSteps              = 12.0f;
    //Sun ray transmittance step count. Drives the TransmittanceToSun
    //integral; bumping this only helps if you see the sun's spectral
    //tint stepping at sunset across long view rays. 8 is the baseline.
    float atmosLightSteps             = 8.0f;
    //Aerial perspective march step counts (view and light). Used by
    //ComputeAerialPerspective. 4 / 4 is the baseline; doubling smooths
    //the in front of cloud haze on long mesh rays.
    float atmosAerialViewSteps        = 4.0f;
    float atmosAerialLightSteps       = 4.0f;
    //Multi scatter boost folded into the per sample atmospheric in
    //scatter rate. 1.0 = pure single scatter (slightly dim), 1.1 =
    //Hillaire's tuned factor, 1.3..1.6 = stylized brighter sky.
    float atmosMultiScatterFactor     = 1.1f;
    //Half angle of the cloud shadow cone sampled on each atmospheric
    //sample inside and beyond the cloud shell. Wider = softer shafts of
    //light but more bleed across cloud edges; narrower = sharper shafts
    //but more visible stepping (the cone jitter is what breaks tap
    //correlation). 5 degrees matches the previous hardcoded 0.9962 cos.
    float atmosCloudShadowConeDeg     = 5.0f;
    //Diffuse multi scatter floor for the cloud shadow tap on atmospheric
    //samples. Without a floor, near atmospheric samples under thick
    //cloud die completely and the integral becomes dominated by reddened
    //far samples (reads as sunset coloured even at noon). 0.04 keeps
    //the near haze at ~4 percent of its unshadowed value.
    float atmosCloudShadowFloor       = 0.04f;
    //Half width (in cosine units) of the planet shadow penumbra used by
    //the smoothstep that softens the earth shadow boundary in the
    //atmosphere march. 0.005 cos ≈ 0.57 degrees angular, comparable to
    //the sun's apparent diameter. Larger = wider soft band, smaller =
    //sharper terminator on the horizon haze.
    float atmosEarthShadowSoftness    = 0.005f;
};

//====================================
//CLOUD SETTINGS
//====================================
//Runtime knobs for the volumetric cloud system in Clouds_v8.hlsli. Mirrored
//into the camera cbuffer tail after SunSettings so the same upload path
//feeds them to the shader. All fields are float so HLSL scalar packing
//interleaves cleanly across cbuffer register boundaries.
//
//Defaults match the static fallbacks in Clouds_v8.hlsli — the runtime
//knobs only take effect once the host has uploaded a non-zero buffer, so
//the visual baseline is identical to the pre-editor state.
struct CloudSettings {
    //master toggle: 0 = clouds OFF (early-out, cheap), 1 = clouds ON.
    //ENABLE_CLOUDS at compile time still wins over this (kill switch).
    float enabled            = 1.0f;
    //coverage controls how much of the sky is filled with cumulus. 0
    //gives clear skies, ~0.5 is "scattered", 1 is overcast.
    float coverage           = 0.60f;
    //shell geometry: layer occupies [bottomKm, topKm] above the planet
    //surface. Typical fair-weather cumulus base sits 1..2 km, top 3..6.
    float layerBotKm         = 1.5f;
    float layerTopKm         = 3.73f;
    //limb softening from orbit, in km of fade distance above cloud base
    //(prevents the layer reading as a hard ring at the planet horizon).
    float horizonFadeKm      = 2.0f;
    //sigma_t at unit density (1/km). 9 matches the Nubis tuned baseline
    //(thin stratocumulus); raise to 25..40 for thick / opaque cumulus.
    //Higher values make clouds read as solid walls instead of letting
    //the multi-scatter terms show through — for "fluffy" cumulus stay
    //in 6..15.
    float extinction         = 4.0f;
    //base shape Worley FBM frequency (1/km). The base octave runs at this
    //frequency, a second detail octave at ~2.7x sits on top. Lower =
    //larger cumulus clusters; 0.18 gives ~5 km cluster spacing which
    //reads as natural skies rather than the tiled "blobs on a grid"
    //look the single octave default produced.
    float baseFrequency      = 0.321f;
    //high-frequency value-noise erosion (1/km) and its amount [0,1].
    //Eats the edges of the base blobs, producing the wispy detail that
    //distinguishes cumulus from raw spheres.
    float hfFrequency        = 7.27f;
    float hfAmount           = 0.65f;
    //====================================
    // Nubis-3 lighting model (Schneider, SIGGRAPH)
    //====================================
    // The primary HG forward lobe is blended with a narrow "silver"
    // lobe via max(); together they reproduce the sharp halo around
    // the sun without paying for a separate Mie evaluation.
    // silverIntensity: amplitude of the silver lobe (0..1).
    // silverSpread:    angular spread of the silver lobe (0..0.3).
    //                  Smaller = tighter halo, larger = wider glow.
    float silverIntensity       =  0.35f;
    float silverSpread          =  0.088f;
    // Half angle of the cone traced sun shadow defocus cone, in degrees.
    // 0 = strict sun direction (cheapest). 2..6 = visibly softer self
    // shadows characteristic of real cumulus.
    float shadowConeDeg         =  0.0f;
    // Secondary multi-scatter phase term: a broader HG modulated by
    // cloud depth + extinction-attenuated sun term. Captures the soft
    // fill on the shadow side without paying for a Wrenninge octave
    // loop.
    // secondaryStrength: amplitude of the secondary term (0..1).
    // secondaryG:        HG eccentricity of the secondary lobe (0..1).
    //                    Smaller = more isotropic (more fill in core).
    float secondaryStrength  = 0.45f;
    float secondaryG         = 0.18f;
    //wind drift in km/s (horizontal only). Animates noise via walltime.
    float windX              = -0.026f;
    float windZ              = -0.037f;
    //view-transmittance cutoff for early-out — below this the cumulative
    //radiance contribution is below the sensor noise floor.
    float trEps              = 0.005f;

    //====================================
    // indirect lighting on cloud samples
    //====================================
    // skyAmbient and groundBounce are 0/1 runtime toggles. skyAmbient
    // adds the hemispherical sky-dome contribution at every cloud
    // scattering event, groundBounce adds a Lambertian terrain bounce
    // proxy underneath the cloud. groundAlbedo is the gray average
    // reflectance of the surface, biased proxy until a per-cell ground
    // irradiance map exists. skyAmbientScale and groundScale are
    // artistic multipliers, 1.0 keeps the values physically scaled.
    float skyAmbient         = 1.0f;
    float groundBounce       = 1.0f;
    float groundAlbedo       = 0.20f;
    float skyAmbientScale    = 0.37f;
    float groundScale        = 1.0f;

    //====================================
    // surface shadowing and termination
    //====================================
    // cloudShadowOnSurfaces enables the inline CloudSunVisibility lookup
    // at surface NEE — attenuates direct sun radiance by the cloud column
    // along the sun ray, so overcast scenes show no direct sun on the
    // ground. Cost is a ~4-sample cloud march per surface NEE call (so
    // per-pixel per-bounce). On by default; flip off if perf is tight in
    // a no-clouds scene. rrThreshold sets the Russian roulette boundary
    // on view-march throughput.
    float cloudShadowOnSurfaces = 1.0f;
    float rrThreshold        = 0.01f;

    //====================================
    // sampling counts (Monte Carlo)
    //====================================
    // shadowConeSamples: number of jittered shadow rays inside the sun
    // defocus cone per scattering event. 1 = strict sun direction
    // (cheapest), 3..5 = visibly softer self shadow if shadowConeDeg
    // is non zero.
    // ambientSteps: number of upward density samples used to estimate
    // sky ambient occlusion per scattering event. 2 default cuts the
    // single-sample variance in half — important for tall cloud
    // layers (3+ km) where individual cumulus can be tall enough
    // that a single random sample's outcome (in-cloud vs above-cloud)
    // varies wildly. Bump to 3-4 if dark patches still appear.
    float shadowConeSamples  = 1.0f;
    float ambientSteps       = 2.0f;

    //====================================
    // adaptive march loop bounds (the big perf levers)
    //====================================
    //Hard upper loop bound on the main view march. The adaptive
    //stepper exits early via t >= tFar in the common case; this is
    //the runaway guard. 128 = baseline (needed for grazing far-cloud
    //views — drop to 48/32 only for cheap preview modes where
    //banding is acceptable). The adaptive stepper means the perf hit
    //of a high ceiling is minimal in nearby/dense scenes.
    //Cast to int at the use site (CB exposes as float for uniform
    //packing).
    float viewStepsMax       = 256.0f;
    //Base step size (km) for the fine portion of the adaptive view
    //march. Smaller = denser sampling = better quality, worse perf.
    //0.6 tuned for stratocumulus shells; bump to 1.0..1.5 to drop
    //sample count by ~half on thick cumulus where banding is hidden
    //by HF noise anyway.
    float targetStepKm       = 0.5f;
    //Surface shadow march sample count (per cloud_cloudShadowOnSurfaces
    //NEE call). 1 = fast single-sample sphere-intersect path
    //(~4-5x cheaper than multi-sample), 2..6 = multi-tap shell
    //march for higher fidelity at cloud edges. Default 1 — visually
    //indistinguishable from 4 for the dominant overhead-cumulus
    //shadow case, and surface NEE is per-pixel per-bounce so this
    //is the single biggest knob for the "Shadow On Surfaces" cost.
    float shadowSteps        = 1.0f;
    //Bounce-ray cheap path: per-bounce volume march step count and
    //max march length along the bounce ray (km). The cheap path
    //runs for specular / transmission bounces and trades quality
    //for speed. 10 / 60 km is the Nubis baseline; drop to 6 / 30
    //if bounce-ray clouds are an indirect-illumination niche.
    float cheapSteps         = 10.0f;
    float cheapMaxLenKm      = 500.0f;
    //Cloud-eval distance limits. fadeDistanceKm starts the fade,
    //renderDistanceKm clamps the march. fadeDistanceKm must be <
    //renderDistanceKm. Lower both aggressively for ground-level
    //scenes where the horizon is < 50 km and far clouds aren't
    //visible anyway.
    float fadeDistanceKm     = 94.0f;
    float renderDistanceKm   = 10000.0f;
    //Atmospheric haze in front of clouds (artistic multiplier on
    //the aerial perspective). 1.0 = physically calibrated, 0.0 =
    //clouds pop without haze attenuation against the sky.
    float hazeStrength       = 0.96f;

    //====================================
    // shell geometry (top variability)
    //====================================
    //Per cloud top altitude jitter. The effective top of each
    //cumulus column is layerTopKm + topVariationKm * noise, so
    //topVariationKm = 0 collapses to a flat slab top, larger
    //values produce towering cumulus that reach much higher than
    //the baseline. topFrequency is the horizontal frequency of
    //the noise (1/km), lower = larger cloud groups share a top
    //altitude, higher = more chaotic top heights cloud to cloud.
    float topVariationKm     = 2.02f;
    float topFrequency       = 0.20f;

    //====================================
    // density field shaping
    //====================================
    //Coverage modulation filter width. Schneider's coverage threshold
    //remap uses this as the soft edge width: smaller = sharper cloud
    //silhouettes, larger = softer fade between cloud and clear sky.
    float covModFilterWidth  = 0.3f;
    //Low frequency domain warp amplitude (km). Pushes the base shape
    //around so cumulus don't look like a stamped grid. 0 disables warp
    //(faster, more obvious tiling). Skipped at distance via LOD blend.
    float warpAmpKm          = 0.30f;

    //====================================
    // cloud albedo (single scattering tint)
    //====================================
    //Per channel single scattering albedo. Default 0.995 across the
    //board is "white cloud, near unit albedo with a hair of absorption".
    //Drop all three for pollution / dust loaded clouds; tint asymmetric
    //for sunset rim experiments.
    float albedoR            = 0.995f;
    float albedoG            = 0.995f;
    float albedoB            = 0.995f;

    //====================================
    // multi scatter shaping (Nubis Evolved single term)
    //====================================
    //Global multiplier on the secondary multi scatter contribution.
    //4.0 is the Nubis Evolved baseline; 0 disables MS and clouds collapse
    //to pure single scatter (very dark cores).
    float msStrength         = 4.05f;
    //Minimum MS amplitude floor at cloud base (h=0). Without this real
    //cumulus bases read as black; 0.18 keeps them "shaded white".
    float msHeightFloor      = 0.18f;
    //Multi scatter model selector:
    //  0 = Nubis single MS term following sqrt(Tdir) (cheapest — one phase
    //      function eval per sample beyond direct)
    //  1 = Wrenninge 2 octave (Hillaire 2016 §5.8): direct + one extra
    //      octave with a^n attenuated extinction and isotropic fill
    //  2 = Wrenninge 3 octave (default): direct + two extra octaves, deepest
    //      cloud cores read as illuminated rather than dark. Combined with
    //      the JD direct phase gives the sharpest silver lining plus the
    //      most uniform frontlit body across modes.
    //Octaves 1+ are isotropic (the c^n -> 0 limit of the Wrenninge
    //prescription, see CloudPhaseDirectScaled rationale) and still scaled
    //by msHeightFloor and msStrength so the artist controls keep working
    //across all three modes.
    float msMode             = 2.0f;

    //====================================
    // sky ambient probe
    //====================================
    //Brightness multiplier on the sky dome ambient contribution.
    //ambientAOScale scales how much the column density above a sample
    //occludes the sky probe. ambientODMax caps the resulting optical
    //depth so dense overcast columns still shut the sky term down.
    float ambientIntensity   = 0.85f;
    float ambientAOScale     = 0.35f;
    float ambientODMax       = 8.0f;

    //====================================
    // sun shadow march
    //====================================
    //Multiplier on the optical depth accumulated along the sun shadow
    //ray. >1 deepens self shadows, <1 brightens them. 1.0 = physical.
    float sunTauMult         = 2.01f;

    //====================================
    // distance LOD blend (noise quality fall off)
    //====================================
    //Distance band over which cloud noise tapers from full quality
    //(<lodNearKm) to simplified (>lodFarKm). Larger band = smoother
    //quality transition, smaller = sharper LOD step.
    float lodNearKm          = 2.0f;
    float lodFarKm           = 30.0f;

    //====================================
    // adaptive march step bounds
    //====================================
    //Largest empty space step taken when the hull says "no cloud here".
    //maxStepKm is the base cap on a single empty step before adaptive
    //growth, maxEmptyStepKm is the absolute cap. emptyStepGrowthPerKm
    //grows the empty step with distance (1 + t * growth) so the march
    //takes huge strides through distant empty sky. maxFineStepKm caps
    //the in cloud step after geometric growth (stepGrowth per step).
    //effectiveZeroDensity is the density floor below which a sample
    //is treated as empty.
    float maxStepKm                 = 0.5f;
    float stepGrowth                = 1.02f;
    float effectiveZeroDensity      = 1e-3f;
    float maxEmptyStepKm            = 500.0f;
    float emptyStepGrowthPerKm      = 0.0f;
    float maxFineStepKm             = 10.0f;
};

//====================================
//PER-FRAME STATS
//====================================
struct FrameStats {
    float cpuFrameMs      = 0;
    float cpuUpdateMs     = 0;
    float cpuInstanceMs   = 0;
    float cpuPopulateMs   = 0;
    float tlasMs          = 0;
    float gpuMs           = 0;
    UINT  instanceCount   = 0;
    UINT  meshCount       = 0;
    bool  tlasWasRefit    = false;
    bool  tlasWasRebuilt  = false;
};

//====================================
//HALTON SEQUENCE FOR JITTER
//====================================
inline float Halton(uint32_t index, uint32_t base) {
    float f = 1.0f, r = 0.0f;
    while (index > 0) { f /= base; r += f * (index % base); index /= base; }
    return r;
}

//====================================
//OCTAHEDRAL NORMAL ENCODE
//====================================
inline UINT EncodeNormalOct(const XMVECTOR& n) {
    XMVECTOR p = n / (abs(XMVectorGetX(n)) + abs(XMVectorGetY(n)) + abs(XMVectorGetZ(n)));
    if (XMVectorGetZ(p) < 0.0f) {
        float oldX = XMVectorGetX(p), oldY = XMVectorGetY(p);
        p = XMVectorSetX(p, (1.0f - abs(oldY)) * (oldX >= 0.0f ? 1.0f : -1.0f));
        p = XMVectorSetY(p, (1.0f - abs(oldX)) * (oldY >= 0.0f ? 1.0f : -1.0f));
    }
    return (static_cast<uint16_t>(static_cast<int>(XMVectorGetY(p) * 32767.0f)) << 16)
         |  static_cast<uint16_t>(static_cast<int>(XMVectorGetX(p) * 32767.0f));
}

inline float Luminance(const XMFLOAT3& c) {
    return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}
