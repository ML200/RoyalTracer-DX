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
#include "../shaders/OceanLayout.h"

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
static constexpr UINT SCRATCH_LAYER_COUNT = 14;
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


static constexpr UINT BLUE_NOISE_HEAP_SLOT = 86;
static constexpr UINT BLUE_NOISE_MASK_SIZE = 128;
// Ocean descriptors occupy the range declared in OceanLayout.h; textures start above it.
static constexpr UINT BINDLESS_HEAP_START = 256;

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
    bool compactLightTree = false;
    bool lightTreeLearning = true;
    bool lightTreeReset = false;
    int lightTreeCellExponent = 0;
    float lightTreeLodScale = 0.05f;

    float lightTreeLearnRoughness = 0.8f;
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

    float sharcQueryFootprint = 2.0f;

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
    float liteNormalSimCos = 0.5f;
    float litePlaneDist = 0.10f;

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

    // Identify settings changes that require reconstruction history to reset.
    auto ReconstructionKey() const {
        const bool lite = liteEnabled;
        return std::make_tuple(maxBounces, maxDiffuseBounces, texturePointFilter, forceDiffuseMats, compactLightTree,
                               lightTreeLearning, lightTreeCellExponent, lightTreeLodScale, sharcEnabled, lite,
                               lite && liteDebugView, lite && liteUnshadowedTargets);
    }
};

struct DLSSGSettings {
    bool available = false;
    bool enabled = false;
    int framesToGenerate = 1;
    int maxFrames = 1;
};

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

    // Distance over which the solar aureole fades in, km. The halo around the sun is the forward
    // lobe of the Mie phase function, and it belongs to the depth of atmosphere a ray actually
    // crosses rather than to the direction it points: without this a wall a few metres away picks
    // up the same halo as the sky behind it. 0 restores the undamped lobe.
    float atmosHaloDistanceKm = 1.0f;
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
    static constexpr UINT GpuTimingCount = 8;
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
