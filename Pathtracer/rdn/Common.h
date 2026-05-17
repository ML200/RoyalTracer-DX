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
//heap 64 (register t40 in Includes_v8.hlsli), bindless starts at 65
static constexpr UINT  AUTOEXPOSE_HEAP_SLOT = 63;
static constexpr UINT  SKY_STARS_HEAP_SLOT  = 64;
static constexpr UINT  BINDLESS_HEAP_START  = 65;

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
struct SunSettings {
    float latitude      = 48.52f;
    float longitude     = 11.405f;
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
    float coverage           = 0.55f;
    //horizontal variation around the coverage base — adds "weather front"
    //character (denser here, clearer there) rather than uniform fill.
    //0.5 default gives noticeable clusters and clearings across the
    //sky which is the dominant cue against the "artificial / repeated"
    //feel of uniform-coverage cloudscapes.
    float coverageVariation  = 0.50f;
    //horizontal frequency of the coverage modulation field (1/km). Lower
    //= bigger weather cells, higher = more local variation.
    float coverageFrequency  = 0.025f;
    //shell geometry: layer occupies [bottomKm, topKm] above the planet
    //surface. Typical fair-weather cumulus base sits 1..2 km, top 3..6.
    float layerBotKm         = 1.5f;
    float layerTopKm         = 3.5f;
    //limb softening from orbit, in km of fade distance above cloud base
    //(prevents the layer reading as a hard ring at the planet horizon).
    float horizonFadeKm      = 2.0f;
    //sigma_t at unit density (1/km). 9 matches the Nubis tuned baseline
    //(thin stratocumulus); raise to 25..40 for thick / opaque cumulus.
    //Higher values make clouds read as solid walls instead of letting
    //the multi-scatter terms show through — for "fluffy" cumulus stay
    //in 6..15.
    float extinction         = 9.0f;
    //base shape Worley FBM frequency (1/km). The base octave runs at this
    //frequency, a second detail octave at ~2.7x sits on top. Lower =
    //larger cumulus clusters; 0.18 gives ~5 km cluster spacing which
    //reads as natural skies rather than the tiled "blobs on a grid"
    //look the single octave default produced.
    float baseFrequency      = 0.18f;
    //high-frequency value-noise erosion (1/km) and its amount [0,1].
    //Eats the edges of the base blobs, producing the wispy detail that
    //distinguishes cumulus from raw spheres.
    float hfFrequency        = 3.5f;
    float hfAmount           = 0.55f;
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
    float silverSpread          =  0.08f;
    // Half angle of the cone traced sun shadow defocus cone, in degrees.
    // 0 = strict sun direction (cheapest). 2..6 = visibly softer self
    // shadows characteristic of real cumulus.
    float shadowConeDeg         =   0.0f;
    // Secondary multi-scatter phase term: a broader HG modulated by
    // cloud depth + extinction-attenuated sun term. Captures the soft
    // fill on the shadow side without paying for a Wrenninge octave
    // loop.
    // secondaryStrength: amplitude of the secondary term (0..1).
    // secondaryG:        HG eccentricity of the secondary lobe (0..1).
    //                    Smaller = more isotropic (more fill in core).
    float secondaryStrength  = 0.45f;
    float secondaryG         = 0.18f;
    // Sun-facing diffuse shell term: Lambertian NdotL on the cloud
    // normal estimated from a 4-tap tetrahedral gradient. Lives mostly
    // on cloud EDGES (shell mask = (1-density)^2) so cumulus get the
    // sun-lit silhouette without flooding the cores. 0.7 default gives
    // strong sunlit/shadowed contrast that separates cumulus shape
    // from the cloud field background.
    float diffuseShellStrength = 0.70f;
    //wind drift in km/s (horizontal only). Animates noise via walltime.
    float windX              = 0.04f;
    float windZ              = 0.015f;
    //View march step floor (adaptive step count grows with view path
    //length above this) and per-sample sun shadow march step count.
    //Both default to stochastic-friendly values — DLSS RR is expected
    //to denoise the per-step variance. Bump to 64/6 if banding shows
    //up behind dense cumulus that the denoiser can't track; drop to
    //16/2 for absolute max perf.
    float viewSteps          = 32.0f;
    float lightSteps         = 3.0f;
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
    float skyAmbientScale    = 1.0f;
    float groundScale        = 1.0f;

    //====================================
    // surface shadowing and termination
    //====================================
    // cloudShadowOnSurfaces enables the inline CloudSunVisibility lookup
    // at surface NEE. expensive when called per-pixel, off by default
    // until a precomputed shadow map is wired in. rrThreshold sets the
    // Russian roulette boundary on view-march throughput.
    float cloudShadowOnSurfaces = 0.0f;
    float rrThreshold        = 0.10f;

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

    //weather-map XZ offset in km. Scrolls the coverage/weather field
    //horizontally without re-seeding noise, so the user can browse
    //different cloud arrangements without changing the underlying
    //random pattern. Wind animation still adds to these at runtime;
    //these are a static artistic offset on top.
    float weatherOffsetX     = 0.0f;
    float weatherOffsetZ     = 0.0f;
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
