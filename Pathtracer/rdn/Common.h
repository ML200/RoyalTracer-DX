#pragma once

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

#ifndef LT_ENABLE_LOGS
#define LT_ENABLE_LOGS 1
#endif

#if LT_ENABLE_LOGS
#define LOG(expr)                                                                                                      \
    do {                                                                                                               \
        std::wcout << L"[Engine] " << expr << std::endl;                                                               \
    } while (0)
#define WARN(expr)                                                                                                     \
    do {                                                                                                               \
        std::wcout << L"[Engine][WARN] " << expr << std::endl;                                                         \
    } while (0)
#else
#define LOG(expr)                                                                                                      \
    do {                                                                                                               \
    } while (0)
#define WARN(expr)                                                                                                     \
    do {                                                                                                               \
    } while (0)
#endif

struct ScopedTimer {
    const char* name;
    std::chrono::high_resolution_clock::time_point t0;
    ScopedTimer(const char* n) : name(n), t0(std::chrono::high_resolution_clock::now()) {}
    ~ScopedTimer() {
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - t0)
                      .count();
        std::wcout << L"[CPU] " << name << L" took " << ms << L" ms" << std::endl;
    }
};
#define SCOPE_TIMER(label) ScopedTimer _scopedTimer_##__LINE__(label)

static constexpr UINT FRAME_COUNT = 3;
static constexpr UINT MAX_BACK_BUFFERS = 6;
static constexpr UINT MAX_STACKS = 4;
static constexpr UINT MAX_INDIRECT_COMMANDS = MAX_STACKS;
static constexpr UINT SORT_BUCKETS = 65536;
static constexpr UINT SCRATCH_LAYER_COUNT = 15;
static constexpr int NUM_LUTS = 2;
static constexpr int LUT_RESOLUTION = 16;
static constexpr int NUM_SAMPLES_LUT = 32000;

static constexpr UINT AUTOEXPOSE_HEAP_SLOT = 63;
static constexpr UINT SKY_STARS_HEAP_SLOT = 64;
static constexpr UINT TERRAIN_TABLE_HEAP_SLOT = 65;
static constexpr UINT TERRAIN_HEIGHTMAP_HEAP_SLOT = 66;
static constexpr UINT TERRAIN_SURFACE_COLOR_HEAP_SLOT = 67;
static constexpr UINT TERRAIN_NORMAL_HEAP_SLOT = 68;
static constexpr UINT SKY_TRANSMITTANCE_LUT_HEAP_SLOT = 69;
static constexpr UINT SKY_MULTISCATTER_LUT_HEAP_SLOT = 70;

static constexpr UINT CUMULUS_QUERY_HEAP_SLOT = 79;

static constexpr UINT BLUE_NOISE_HEAP_SLOT = 86;
static constexpr UINT BLUE_NOISE_MASK_SIZE = 128;
static constexpr UINT BINDLESS_HEAP_START = 87;

static constexpr D3D12_RESOURCE_STATES kSRV =
    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;

struct BTriVertex {
    XMFLOAT3 vertex;
    UINT packedNormal;
    PackedVector::XMHALF2 texCoord;
};

struct Float3x4 {
    float m[12];
};
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

struct InstanceProperties {
    Float3x4 objectToWorld;
    Float3x4 objectToWorldInverse;
    Float3x4 objectToWorldNormal;
    UINT indexBase;
    UINT vertexBase;
    UINT materialBase;

    UINT triToLightBase;
    UINT opaqueTriCount;
    UINT _pad[2];

    UINT lightSlot;
    Float3x4 prevObjectToWorld;
};
static_assert(sizeof(InstanceProperties) == 224, "GPU record must stay in sync with Data_v8.hlsli");

struct InstancePropertiesCpu {
    XMMATRIX objectToWorld;
    XMMATRIX objectToWorldInverse;
    XMMATRIX prevObjectToWorld;
    XMMATRIX prevObjectToWorldInverse;
    XMMATRIX objectToWorldNormal;
    XMMATRIX prevObjectToWorldNormal;
    UINT indexBase;
    UINT vertexBase;
    UINT materialBase;
    UINT triToLightBase;
    UINT opaqueTriCount;
    UINT _pad[2];
    UINT lightSlot;
};

struct GeometryOffsets {
    UINT vertexBase;
    UINT indexBase;
    UINT materialBase;
};

struct IntegratorSettings {
    int integratorMode = 0;

    bool compactLightTree = false;
    bool lightTreeLearning = true;
    bool lightTreeReset = false;
    int lightTreeCellExponent = 0;
    float lightTreeLodScale = 0.05f;

    float lightTreeLearnRoughness = 0.3f;
    bool lightTreeDebug = false;
    bool sharcEnabled = true;
    bool sharcReset = false;
    int sharcDebugMode = 0;
    bool sharcDebugCoarse = false;
    int sharcCellSizeExponent = -3;
    float sharcLodScale = 0.01f;
    int sharcUpdateStride = 4;
    int sharcMinSamples = 8;
    int sharcHistoryFrames = 16;
    int sharcMaxAge = 512;

    float sharcQueryFootprint = 0.5f;

    int sharcTrainBounces = 8;
    int sharcTrainRrDepth = 5;

    bool sharcGuideEnabled = true;
    bool sharcGuideTrain = true;
    float sharcGuideMax = 0.75f;
    int sharcGuideLevelOffset = 2;
    int sharcGuideFreshness = 32; // frames without new evidence after which a receiver guides at half strength
    int sharcGuideDepth = 2;

    bool liteEnabled = true;
    bool liteSpatial = true;
    bool liteUnshadowedTargets = false;
    bool liteDebugView = false;
    int liteSpatMcap = 8;
    int liteSpatSlots = 3;
    float liteReuseSigma = 20.0f;

    int tempMcapGI = 8;
    int spatCountMaxGI = 2;
    int spatCountMinGI = 2;
    int spatRadMaxGI = 56;
    int spatRadMinGI = 8;
    int spatTriesGI = 8;

    int maxBounces = 32;
    int rrStartDepth = 2;
    float regularizeRoughness = 0.2f;

    int maxDiffuseBounces = 3;
    int initialSamples = 1;

    int texturePointFilter = 0;

    int dlssDebugLayer = 0;

    float dlssDebugDepthNear = 0.0f;
    float dlssDebugDepthFar = 50.0f;

    bool forceDiffuseMats = false;
    bool enableTempGI = false;
    bool enableSpatGI = false;
    bool disableCorrReduction = false;

    float corrReductionPow = 0.025f;

    bool noSpecReproj = false;
    bool disableReuseVis = false;
    bool disableFinalVis = false;
    float reuseRoughnessMin = 0.1f;
    float reuseRoughnessMax = 0.3f;

    float reconnectRoughnessMin = 0.15f;

    float rejNormalDot = 0.36f;
    float rejDistance = 0.10f;

    float tempNormalSimCos = 0.5f;
    float tempPlaneDist = 0.10f;
    float tempJacClamp = 1.0f;

    float ucwClampMax = 10000.0f;

    int spmisReuseN = 2;
    int spmisRisN = 8;
    int spmisMcap = 20;
    int spmisTileSize = 32;

    bool spmisCellJitter = true;
    float spmisNormalFuzz = 0.2f;
    int spmisNormalBits = 2;
    float spmisSearchR0 = 20.0f;
    float spmisSearchGrow = 1.25f;
    int spmisSearchIters = 12;
    float spmisJacThreshold = 15.0f;
    float spmisNormalSimCos = -1.0f;
    float spmisPlaneDist = 0.111f;

    bool spmisConfidenceAdjust = true;

    bool hybridShift = true;

    float reconnectDistMin = 0.01f;

    int rcMaxK = 8;

    bool rcFootprint = true;
    float rcFpKappa = 0.02f;

    bool dualMotionVectors = true;

    bool rgbShadeWeights = true;

    bool lobeIndexedPss = true;

    // Identify settings changes that require reconstruction history to reset.
    auto ReconstructionKey() const {
        const bool pt = integratorMode == 0;
        const bool lite = pt && liteEnabled;
        return std::make_tuple(integratorMode, maxBounces, maxDiffuseBounces, texturePointFilter, forceDiffuseMats,
                               compactLightTree, pt && lightTreeLearning, lightTreeCellExponent, lightTreeLodScale, pt && sharcEnabled,
                               lite, lite && liteDebugView, lite && liteUnshadowedTargets, ucwClampMax,
                               pt ? 0u : Flags());
    }

    // Bit positions must match RS_FLAG_* in the shared shader settings.
    UINT Flags() const {
        return (enableTempGI ? 2u : 0u) | (enableSpatGI ? 8u : 0u) | (enableSpatGI ? 0x10u : 0u) |
               ((enableSpatGI && spmisConfidenceAdjust) ? 0x2000u : 0u) | (disableCorrReduction ? 0x40u : 0u) |
               (noSpecReproj ? 0x200u : 0u) | (disableReuseVis ? 0x800u : 0u) | (disableFinalVis ? 0x1000u : 0u) |
               (hybridShift ? 0x4000u : 0u) | (forceDiffuseMats ? 0x20000u : 0u) | (spmisCellJitter ? 0x40000u : 0u) |
               (rcFootprint ? 0x100000u : 0u) | (dualMotionVectors ? 0x200000u : 0u) |
               (rgbShadeWeights ? 0x400000u : 0u) | ((lobeIndexedPss && hybridShift) ? 0x800000u : 0u);
    }
};

struct DLSSGSettings {
    bool available = false;
    bool enabled = false;
    int framesToGenerate = 1;
    int maxFrames = 1;
};

struct CumulusSettings {
    float enabled = 0.0f;
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
    float lightingSamples = 2.0f;
    float fineDetail = 1.0f;

    auto LightingKey() const {
        return std::make_tuple(enabled, coverage, baseKm, thicknessKm, scale, extinction, detail, windX, windZ,
                               multipleScattering, ambient, viewSteps, reflectionSteps, seed, lightingSamples,
                               fineDetail);
    }
};
static_assert(sizeof(CumulusSettings) == 72);

struct SunSettings {
    float latitude = 48.52f;
    float longitude = 11.405f;
    float dayOfYear = 172.0f;
    float simSpeed = 10.0f;
    float startUTCHours = 6.0f;
    float nightSpeedup = 2.0f;
    float turbidity = 2.0f;
    float sunIntensity = 5.0f;

    float skyIntensity = 1.0f;
    float globalEmissionStrength = 1.0f;

    float dofApertureRadius = 0.0f;
    float dofFocusDistance = 10.0f;

    float skyStarIntensity = 0.047f;
    float skyStarGamma = 1.57f;
    float skyStarLodBias = -1.0f;
    float skyStarThreshold = 0.0f;

    float skyNightBaseIntensity = 0.57f;

    float atmosViewSteps = 12.0f;

    float atmosLightSteps = 8.0f;
    float atmosAerialViewSteps = 4.0f;
    float atmosAerialLightSteps = 4.0f;

    float atmosMultiScatterFactor = 1.0f;

    float atmosEarthShadowSoftness = 0.005f;
};

struct GpuPassTiming {
    std::string name;
    float gpuMs = 0;
    UINT calls = 0;
};

struct LightBvhStats {
    UINT nodes = 0, slots = 0, voxelLeaves = 0;
    UINT64 nodeBytes = 0, slotBytes = 0, trailBytes = 0;
    float buildCpuMs = 0;
    bool buildMeasured = false, incremental = false, pending = false;
};

struct FrameStats {
    static constexpr UINT GpuTimingCount = 10;
    float cpuFrameMs = 0;
    float cpuUpdateMs = 0;
    float cpuInstanceMs = 0;
    float cpuPopulateMs = 0;
    float cpuStreamingMs = 0;
    float gpuWaitMs = 0; // CPU fence wait, not a GPU timestamp
    float gpuFrameMs = 0;
    bool gpuTimingsValid = false, gpuTimingsTruncated = false;
    std::vector<GpuPassTiming> gpuPasses;
    LightBvhStats lightBvh;
    float cachePassMs[GpuTimingCount] = {};
    UINT cacheTimingMask = 0;
    UINT instanceCount = 0;
    UINT meshCount = 0;
};

inline float Halton(uint32_t index, uint32_t base) {
    float f = 1.0f, r = 0.0f;
    while (index > 0) {
        f /= base;
        r += f * (index % base);
        index /= base;
    }
    return r;
}

inline UINT EncodeNormalOct(const XMVECTOR& n) {
    XMVECTOR p = n / (abs(XMVectorGetX(n)) + abs(XMVectorGetY(n)) + abs(XMVectorGetZ(n)));
    if (XMVectorGetZ(p) < 0.0f) {
        float oldX = XMVectorGetX(p), oldY = XMVectorGetY(p);
        p = XMVectorSetX(p, (1.0f - abs(oldY)) * (oldX >= 0.0f ? 1.0f : -1.0f));
        p = XMVectorSetY(p, (1.0f - abs(oldX)) * (oldY >= 0.0f ? 1.0f : -1.0f));
    }
    return (static_cast<uint16_t>(static_cast<int>(XMVectorGetY(p) * 32767.0f)) << 16) |
           static_cast<uint16_t>(static_cast<int>(XMVectorGetX(p) * 32767.0f));
}

inline float Luminance(const XMFLOAT3& c) {
    return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}
