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
//NRC reserves 5 UAVs at heap 58..62, autoexpose at heap 63, bindless starts at 64
static constexpr UINT  AUTOEXPOSE_HEAP_SLOT = 63;
static constexpr UINT  BINDLESS_HEAP_START  = 64;

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
    float reuseRoughnessMin = 0.2f;
    float reuseRoughnessMax = 0.5f;

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
    float skyIntensity  = 8.5f;
    float globalEmissionStrength = 1.0f;
    //thin-lens DoF, populated from Camera::apertureRadius / focusDistance
    //during UploadGPUBuffer, lives in this struct so the cbuffer tail stays 16-byte aligned
    float dofApertureRadius = 0.0f;
    float dofFocusDistance  = 10.0f;
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
