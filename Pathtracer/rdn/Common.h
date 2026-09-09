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
#include <tuple>

#include "glm/gtc/matrix_transform.hpp"
#include "../src/Components/Vertex.h"
#include "d3dx12.h"
#include "../shaders/SharcLayout.h"
#include "../shaders/RenderFlags.h"
#include "../shaders/SkyBakeLayout.h"

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
static constexpr UINT  SCRATCH_LAYER_COUNT = 15; // atmosphere 10/11, SHaRC 12, cloud RR guides 13/14
static constexpr int   NUM_LUTS             = 2;
static constexpr int   LUT_RESOLUTION       = 16;
static constexpr int   NUM_SAMPLES_LUT      = 32000;
//NRC reserves heap 58..62; auto-exposure and stars occupy 63..64.
//Terrain SRVs occupy 65..68; atmospheric LUTs occupy 69..70.
static constexpr UINT  AUTOEXPOSE_HEAP_SLOT             = 63;
static constexpr UINT  SKY_STARS_HEAP_SLOT              = 64;
static constexpr UINT  TERRAIN_TABLE_HEAP_SLOT          = 65;
static constexpr UINT  TERRAIN_HEIGHTMAP_HEAP_SLOT      = 66;
static constexpr UINT  TERRAIN_SURFACE_COLOR_HEAP_SLOT  = 67;
static constexpr UINT  TERRAIN_NORMAL_HEAP_SLOT         = 68;
static constexpr UINT  SKY_TRANSMITTANCE_LUT_HEAP_SLOT  = 69;
static constexpr UINT  SKY_MULTISCATTER_LUT_HEAP_SLOT   = 70;
// Cumulus SRVs 71..73, UAVs 74..76, ambient SRV/UAV 77/78, query UAV 79.
static constexpr UINT  CUMULUS_QUERY_HEAP_SLOT           = 79;
// Noise B/A plane SRV/UAV occupy 80/81; density/tag SRVs 82/83, UAVs 84/85.
static constexpr UINT  BINDLESS_HEAP_START              = 86;

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
//GPU-packed affine 3x4 matrix: 4 packed float3 rows (the w column is dropped —
//always 0,0,0,1 for these affine transforms). This byte layout equals HLSL's
//default column_major float3x4 in a StructuredBuffer, so mul(M, float4(p,1))
//computes exactly the same 4-term dot per component as the old float4x4 path
//(bit-identical transforms, 48B per matrix instead of 64B).
struct Float3x4 { float m[12]; };
inline Float3x4 MakeFloat3x4(DirectX::XMMATRIX M) {
    Float3x4 r;
    for (int row = 0; row < 4; ++row) {
        DirectX::XMFLOAT4 f;
        XMStoreFloat4(&f, M.r[row]);
        r.m[row * 3 + 0] = f.x;
        r.m[row * 3 + 1] = f.y;
        r.m[row * 3 + 2] = f.z;
    }
    return r;
}

//GPU per-instance record (224B, was 416B). Hot transforms as affine 3x4 (25%
//fewer bytes per fetch, 2 sectors -> 1-2), the hot index/material bases packed
//right after them, and only the ONE prev matrix any shader reads
//(prevObjectToWorld — Camera_Ray reprojection) kept, cold at the tail.
//prevObjectToWorldInverse/prevObjectToWorldNormal were uploaded every frame but
//never read by any pass, so they exist only on the CPU working struct now.
//MUST mirror InstanceProperties in Data_v8.hlsli field-for-field.
struct InstanceProperties {
    Float3x4 objectToWorld;
    Float3x4 objectToWorldInverse;
    Float3x4 objectToWorldNormal;
    UINT     indexBase;
    UINT     vertexBase;
    UINT     materialBase;
    UINT     triToLightBase;
    UINT     opaqueTriCount;
    UINT     _pad[3];
    Float3x4 prevObjectToWorld;
};
static_assert(sizeof(InstanceProperties) == 224, "GPU record must stay in sync with Data_v8.hlsli");

//CPU working state (full 4x4s): Scene::PrepareInstanceProperties runs inverse /
//transpose math and prev-frame caching on these, then packs to the GPU record at
//upload (Scene::UploadInstanceProperties). prevObjectToWorldInverse/Normal stay
//here because the snap-recompute logic writes them (GPU never reads them).
struct InstancePropertiesCpu {
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
//INTEGRATOR RUNTIME SETTINGS
//====================================
//DI+GI unified, DI knobs removed
struct IntegratorSettings {
    //====================================
    //INTEGRATOR SELECT
    //====================================
    //0 = Path tracer (Pass_pt_v8): the clean RIS-free unidirectional path
    //    tracer — per-vertex light-tree NEE + sun NEE with balance-heuristic
    //    MIS, same transport math as Pass_raygen but accumulating radiance
    //    directly. Every reservoir pass is skipped; doubles as the ground-
    //    truth reference for the ReSTIR target functions with SHaRC disabled.
    //1 = ReSTIR (DEPRECATED): the legacy reservoir pipeline, kept selectable
    //    for comparison. Everything below this block configures it.
    int   integratorMode   = 0;

    //SHaRC is exclusive to the regular path tracer. Disabling it restores the
    //uncached reference and its original bounce budgets. Grid size is a power
    //of two for floating-origin precision; cache memory is resolution-independent.
    bool  sharcEnabled = true;
    bool  sharcReset = false;
    int   sharcDebugMode = 0; // 0: off, 1: cells, 2: stored cell lighting, 3: guiding coverage
    bool  sharcDebugCoarse = false; // inspect the other (less sampled) of the two queried levels
    int   sharcCellSizeExponent = -3; // 0.125 m minimum spacing
    float sharcLodScale = 0.01f;      // spacing grows with camera distance
    int   sharcUpdateStride = 4;      // one rotating sample per 4x4 tile
    int   sharcMinSamples = 16;
    int   sharcHistoryFrames = 64;
    int   sharcMaxAge = 512;
    // Path spread required before a secondary vertex may terminate into the
    // cache, in coarser-cell widths (ramps to full at twice this). Cells are
    // ~1% of the camera distance, so 3 made distant first bounces miss almost
    // always and trace on instead.
    float sharcQueryFootprint = 0.5f;
    // Training paths end by cache resampling or by roulette on their own suffix
    // throughput; the cap bounds the lanes that do neither. A sparse dispatch
    // finishes when its longest lane does, so the cap is the tail: 48 with fixed
    // 0.9 survival was latency-bound, 16 with roulette from 6 still left a tail.
    int   sharcTrainBounces = 8;
    int   sharcTrainRrDepth = 5; // difficult multi-bounce areas need the suffix intact this deep
    // Cache-driven path guiding (SharcGuide_v8.hlsli). Diffuse-lobe samples are
    // drawn toward bright cached patches that training paths saw from the
    // receiver cell, as a MIS-weighted mixture with cosine sampling, so the
    // estimator stays unbiased. Off restores the plain cosine sampler.
    bool  sharcGuideEnabled = true;
    bool  sharcGuideTrain = true;    // training paths use the mixture too: faster discovery
    float sharcGuideMax = 0.6f;      // cap on the guided fraction of diffuse-lobe samples
    int   sharcGuideLevelOffset = 2; // receiver cell = cache cell x 2^offset (0.5 m near the camera)
    int   sharcGuideLifetime = 256;  // frames; an unseen patch fades by 2^(-age / lifetime)
    float sharcGuideRadius = 0.75f;  // patch bounding radius in cell widths
    int   sharcGuideDepth = 2;       // deepest guided/trained path vertex (1 = primary only)
    // ReSTIR lite (shaders/RestirLite_v8.hlsli), regular path tracer only:
    // the primary vertex's diffuse lobe is resampled instead of path traced.
    // Candidates are its NEE samples and whatever its scatter ray finds (an
    // emitter, the sky, or the secondary vertex with its cached or path-traced
    // outgoing radiance, ReSTIR GI style); paired spatial reuse follows,
    // with visibility in every target. Lite reservoirs last only this frame.
    bool  liteEnabled = true;
    bool  liteSpatial = true;
    bool  liteUnshadowedTargets = false; // off: every reuse target traces its own ray, exact (A/B on: winner only, ~0.2 ms cheaper, over-credits at shadow edges)
    bool  liteDebugView = false;        // display the resampled contribution only
    int   liteSpatMcap = 8;             // spatial confidence cap (canonical and partners)
    int   liteSpatSlots = 3;            // paired partners per pixel (0..3)
    float liteReuseSigma = 20.0f;       // pair distance of the reuse tables, pixels (std dev)

    int   tempMcapGI       = 8;
    int   spatCountMaxGI   = 2;
    int   spatCountMinGI   = 2;
    int   spatRadMaxGI     = 56;
    int   spatRadMinGI     = 8;
    int   spatTriesGI      = 8;
    //path-trace termination (uploaded as cbuffer slots 38-39, read by raygen). Not
    //ReSTIR per se, but they ride the same rs-consts block. Editor caps both at 32.
    int   maxBounces       = 32;  // raygen path loop bound: depth runs [1, maxBounces)
    int   rrStartDepth     = 2;   // Russian roulette starts at depth >= this; set to 32 to disable
    //Diffuse-bounce budget (cbuffer slot 29, read by raygen). A path may take at
    //most this many scattering events on materials with a diffuse component
    //(also metals and specular/coat layered over diffuse); glass and translucent
    //(SSS) materials bounce past it, up to the maxBounces hard cap. Caps the
    //expensive/noisy diffuse GI depth while letting refraction paths run deep.
    int   maxDiffuseBounces = 3;
    int   initialSamples   = 1;   // RIS-over-N initial samples per pixel (host-clamped [1,8])
    //Material-texture filtering (cbuffer slot 42, read by SampleMaterialTex). 0 =
    //hardware bilinear/aniso, 1 = nearest-texel point sampling for crisp pixel-art /
    //Minecraft assets. Global (all textures); editor toggle under Materials.
    int   texturePointFilter = 0;
    //DLSS-RR guide-buffer inspector (cbuffer slot 30, read by
    //Pass_postprocess_v8::DlssInputDebugView). 0 = off; otherwise the selected
    //DLSS input layer is rendered raw into gOutput slice 3 — the 4th 'C' stop —
    //for chasing denoiser artifacts. Layer list in the editor's "DLSS Inputs"
    //window. Not ReSTIR per se; rides the rs-consts block like the pt_* knobs.
    int   dlssDebugLayer = 0;
    //Depth / spec-hit-dist display window in metres (slot 31, packed f16 pair).
    //The viewer maps [near, far] LINEARLY onto the 8-bit ramp — a planet-scale
    //log map put ~1 gray step per metre and manufactured contour banding on a
    //perfectly smooth R32F buffer. Banding that survives a TIGHT window is real.
    float dlssDebugDepthNear = 0.0f;
    float dlssDebugDepthFar  = 50.0f;
    //Materials debug (flag 0x20000): every material decodes as opaque Lambertian
    //(albedo + emission kept; transmission/specular/coat/sheen/thin-glass/SSS
    //forced off inside Material_Decoder_v8.hlsli). Live-toggleable.
    bool  forceDiffuseMats = false;
    bool  enableTempGI     = false;
    bool  enableSpatGI     = false;
    bool  disableCorrReduction = false; // A/B: ignore dup-map D in the temporal confidence cap (flag 0x40)
    //Temporal correlation-reduction strength (cbuffer slot 21): the dup-map
    //exponent e in effMcap = lerp(tempMcap, 1, pow(D, e)). SMALLER = stronger
    //collapse at the same duplication; each halving doubles the strength in
    //log space (0.1 = original tuning, 0.025 = 4x).
    float corrReductionPow     = 0.025f;
    // (flag bit 0x80 retired — was disableX1Direct; §6.1 removed directAtX1 so the diagnostic had nothing to zero)
    bool  noSpecReproj         = false; // force surface (self) reprojection for the DI reservoir, off the stochastic specular MV (flag 0x200)
    bool  disableReuseVis      = false; // take the reconnection shadow ray OUT of temporal+spatial reuse and apply it once at the spatial resolve (flag 0x800)
    bool  disableFinalVis      = false; // sub-toggle of disableReuseVis: also skip the deferred resolve shadow ray -> fully unshadowed GI (diagnostic) (flag 0x1000)
    float reuseRoughnessMin = 0.1f;
    float reuseRoughnessMax = 0.3f;
    //Reconnection-vertex (x2) specularity reject. A temporal or spatial neighbour
    //whose GI reconnection vertex roughness is below this is dropped from reuse
    //outright (a near-delta BSDF at x2 makes the reconnection shift invalid; down-
    //weighting it would bias the estimator — only rejection stays unbiased).
    float reconnectRoughnessMin = 0.15f;

    //neighbor rejection thresholds (SPMIS cell search + reconnection rejection)
    float rejNormalDot     = 0.36f;
    float rejDistance      = 0.10f;

    //Temporal-reuse reprojected-pixel geometry rejection + Jacobian clamp (cbuffer
    //slots 13-15, read by Pass_temp_gi_v8). Rejecting a reprojection onto a
    //different surface (a corner face / depth discontinuity) is what prevents the
    //grazing-cos reconnection Jacobian blow-up; the clamp bounds whatever remains.
    float tempNormalSimCos = 0.5f;   // reprojected normal cone (cos); -1 = accept all
    float tempPlaneDist    = 0.10f;  // reprojected plane-distance reject, FRACTION of camera distance
    float tempJacClamp     = 1.0f;   // reconnection-Jacobian clamp band T -> ratio bounded to [1/T, T]
    //Reuse-output UCW (reservoir W) clamp (temporal + SPMIS). W = w_sum/p_hat is
    //unbounded; a tiny p_hat near a grazing/occluded surface spikes it and the spike
    //feeds back through reuse into a diverging firefly. Clamp bounds it. <=0 disables.
    float ucwClampMax      = 10000.0f;

    //Spatial reuse is the SPMIS global-hash-grid pipeline (Pass_spmis_*
    //reset/count/offsets/sort/select/passthrough/shift/merge; raygen inserts
    //each pixel's hash). The old texture-paired select/shift/_v8_1 variant was
    //removed — RS_FLAG_SPMIS_SPATIAL (0x10) is now raised whenever enableSpatGI.
    int   spmisReuseN       = 2;     // Ntilde: non-canonical reuse draws
    int   spmisRisN         = 8;     // inner-RIS candidate count per draw
    int   spmisMcap         = 20;    // output confidence M cap (0 disables)
    int   spmisTileSize     = 32;    // screen-space cell tile size in pixels
    //Hash-grid + cell-search tuning (cbuffer slots 24-28 + flag 0x40000). The
    //grid rebuilds every frame, so all of these are live-toggleable.
    bool  spmisCellJitter   = true;  // per-frame tile-origin jitter (flag 0x40000); off = static grid
    float spmisNormalFuzz   = 0.2f;  // slot 24: tangent-plane normal jitter in the cell hash (0 = hard buckets)
    int   spmisNormalBits   = 2;     // slot 25: normal quantization bits per component (1..4)
    float spmisSearchR0     = 20.0f; // slot 26: cell-search initial probe radius (px)
    float spmisSearchGrow   = 1.25f; // slot 27: cell-search radius growth per probe
    int   spmisSearchIters  = 12;    // slot 28: cell-search probe count (4..32)
    float spmisJacThreshold = 15.0f; // reconnection-shift jacobian reject band [1/T, T]
    float spmisNormalSimCos = -1.0f; // neighbor-similarity normal cone (cos); -1 = accept all normals
    float spmisPlaneDist    = 0.111f;// neighbor-similarity plane-distance reject, as a FRACTION of
                                     // the distance to camera (regular-ReSTIR geometry rejection)
    //§4.3 non-canonical confidence scaling. OFF (scaling=1, canonical weight ~0, max
    //neighbour reuse). ON boosts the canonical (less reuse, less dark-pepper).
    //Flag 0x2000.
    bool  spmisConfidenceAdjust = true;

    //ReSTIR PT hybrid shift (random replay + reconnection in primary sample
    //space; flag 0x4000). ON: raygen pins the reconnection vertex at the first
    //vertex pair passing the criteria and glossy prefixes are random-replayed
    //at reuse (temporal: compacted Pass_temp_replay indirect dispatch;
    //spatial: replay roles inside the unified Pass_spmis_shift). OFF: legacy
    //pin at x2, zero replay, legacy glossy reuse gates.
    bool  hybridShift = true;
    //Hybrid pin criteria: minimum reconnection-segment length as a FRACTION of
    //the primary camera distance (cbuffer slot 18). Short segments make the
    //area-measure geometric factor singular; the pin postpones via replay.
    float reconnectDistMin = 0.01f;
    //Hybrid pin cap: largest reconnection-vertex index k raygen may pin
    //(cbuffer slot 19; replay length = k-2, capped at RC_REPLAY_MAX_BOUNCES=8).
    //2 = first-vertex pins only (zero replay even with hybrid ON).
    int   rcMaxK = 8;
    //ReSTIR PT Enhanced §4 pin criteria (flag 0x100000): dual footprint
    //threshold + pdf-proxy glossiness guard replace the roughness+distance
    //pair test. rcFpKappa is the paper's c (cbuffer slot 20; Eq. 5 literal,
    //cross-scene optimum 0.02, ablation range 0.005-0.64).
    bool  rcFootprint = true;
    float rcFpKappa   = 0.02f;
    //ReSTIR PT Enhanced §6.4 dual motion vectors (flag 0x200000): on a
    //temporal geometry reject, retry at launchIndex - occluder screen motion.
    bool  dualMotionVectors = true;
    //ReSTIR PT Enhanced §6.3 RGB shading weights (flag 0x400000): the spatial
    //resolve blends contributor chroma via vectorized resampling weights.
    bool  rgbShadeWeights = true;
    //ReSTIR PT Enhanced supplemental §1 lobe-indexed PSS (flag 0x800000):
    //extension dims split per sampled BSDF lobe (throughput rho_l/(P(l)*p(w|l)),
    //pmf product peels out of the stored F like RR survival), the 2-bit lobe
    //ids ride rcInfo bits 16-31, jacobian bundles go conditional, and shifts
    //PRESERVE the lobe sequence — replay forces the recorded lobe per bounce
    //(the multilobe/plastic variance fix). Needs hybridShift; host caps rcMaxK
    //at 8 while set (the rcInfo mask covers 8 vertices).
    bool  lobeIndexedPss = true;

    // Image semantics that must not blend with the previous reconstruction.
    // Cache inspection/timing controls do not change the underlying image.
    auto ReconstructionKey() const {
        const bool pt = integratorMode == 0;
        const bool lite = pt && liteEnabled;
        return std::make_tuple(integratorMode, maxBounces, maxDiffuseBounces,
            texturePointFilter, forceDiffuseMats, pt && sharcEnabled, lite,
            lite && liteDebugView, lite && liteUnshadowedTargets, ucwClampMax,
            pt ? 0u : Flags());
    }

    UINT Flags() const {
        //bits 0 (tempDI) and 2 (spatDI) stay zero, DI pipeline gone
        return (enableTempGI ? 2u : 0u) | (enableSpatGI ? 8u : 0u)
             | (enableSpatGI ? 0x10u : 0u)   // spatial reuse is always SPMIS now
             | ((enableSpatGI && spmisConfidenceAdjust) ? 0x2000u : 0u)
             | (disableCorrReduction ? 0x40u : 0u)
             | (noSpecReproj ? 0x200u : 0u)
             | (disableReuseVis ? 0x800u : 0u)
             | (disableFinalVis ? 0x1000u : 0u)
             | (hybridShift ? 0x4000u : 0u)
             | (forceDiffuseMats ? 0x20000u : 0u)
             | (spmisCellJitter ? 0x40000u : 0u)
             | (rcFootprint ? 0x100000u : 0u)
             | (dualMotionVectors ? 0x200000u : 0u)
             | (rgbShadeWeights ? 0x400000u : 0u)
             | ((lobeIndexedPss && hybridShift) ? 0x800000u : 0u);
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

//Observer latitude/longitude drive local solar time and the ENU sky frame.
//Scalar float layout matches CameraParams in Includes_v8.hlsli.
// Scalar packing is shared with CameraParams. Distances in kilometres except wind (m/s).
struct CumulusSettings {
    float enabled = 1.0f;
    float coverage = 0.28f;
    float baseKm = 1.5f;
    float thicknessKm = 3.6f;
    float scale = 0.8f;
    float extinction = 14.0f;
    float detail = 1.0f;
    float windX = 0.0f;
    float windZ = 0.0f;
    float multipleScattering = 1.0f;
    float ambient = 1.0f;
    float viewSteps = 64.0f;
    float reflectionSteps = 12.0f;
    float seed = 17.0f;
    float debugView = 0.0f;
    float guideThreshold = 0.5f;
    float lightingSamples = 2.0f; // 0 shades every occupied step; 1-4 use RR lighting reservoirs.
    float fineDetail = 1.0f; // Wind deformation of existing billows and detailed local shadows.

    auto LightingKey() const {
        return std::make_tuple(enabled, coverage, baseKm, thicknessKm, scale,
            extinction, detail, windX, windZ, multipleScattering, ambient,
            viewSteps, reflectionSteps, seed, lightingSamples, fineDetail);
    }
};
static_assert(sizeof(CumulusSettings) == 72);

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

    float atmosViewSteps              = 12.0f;
    //Sun ray transmittance step count. Drives the TransmittanceToSun
    //integral; bumping this only helps if you see the sun's spectral
    //tint stepping at sunset across long view rays. 8 is the baseline.
    float atmosLightSteps             = 8.0f;
    float atmosAerialViewSteps        = 4.0f;
    float atmosAerialLightSteps       = 4.0f;
    //Artistic boost on the per sample DIRECTIONAL single scatter rate
    //only. Real 2nd+ order scattering now comes from the per frame
    //Hillaire Psi_ms LUT (Pass_skylut_bake_v8.hlsl::mainMultiScatter),
    //which this factor deliberately does NOT touch — the old 1.1 default
    //was the flat stand-in for that missing term. 1.0 = physical;
    //1.2..1.5 = stylized brighter sky.
    float atmosMultiScatterFactor     = 1.0f;
    //Half width (in cosine units) of the planet shadow penumbra used by
    //the smoothstep that softens the earth shadow boundary in the
    //atmosphere march. 0.005 cos ≈ 0.57 degrees angular, comparable to
    //the sun's apparent diameter. Larger = wider soft band, smaller =
    //sharper terminator on the horizon haze.
    float atmosEarthShadowSoftness    = 0.005f;
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
    float gpuWaitMs       = 0; // CPU fence wait, not a GPU timestamp
    float cachePassMs[9]   = {}; // prepare/train/resolve, PT + NEE, lite shift/merge, cloud caches, primary sky/air, secondary clouds
    UINT  cacheTimingMask  = 0;
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
