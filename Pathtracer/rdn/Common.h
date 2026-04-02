#pragma once
// ═══════════════════════════════════════════════════════════════════
// Common.h — Shared types, macros, and forward declarations
// ═══════════════════════════════════════════════════════════════════

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

// ── Logging ──────────────────────────────────────────────────────
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

// ── Scoped CPU Timer ─────────────────────────────────────────────
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

// ── Constants ────────────────────────────────────────────────────
static constexpr UINT  FRAME_COUNT          = 2;
static constexpr UINT  MAX_STACKS           = 4;
static constexpr UINT  MAX_INDIRECT_COMMANDS = MAX_STACKS;
static constexpr UINT  SORT_BUCKETS         = 65536;
static constexpr int   NUM_LUTS             = 2;
static constexpr int   LUT_RESOLUTION       = 16;
static constexpr int   NUM_SAMPLES_LUT      = 32000;
static constexpr UINT  BINDLESS_HEAP_START  = 60;

static constexpr D3D12_RESOURCE_STATES kSRV =
    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE |
    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;

// ── GPU Vertex Layout (for global buffers) ───────────────────────
struct BTriVertex {
    XMFLOAT3                       vertex;
    UINT                           packedNormal;
    PackedVector::XMHALF2         texCoord;
};

// ── Per-Instance data uploaded to GPU ────────────────────────────
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

// ── Geometry offsets for SRV setup ───────────────────────────────
struct GeometryOffsets {
    UINT vertexBase;
    UINT indexBase;
    UINT materialBase;
};

// ── ReSTIR runtime settings ─────────────────────────────────────
struct ReSTIRSettings {
    int   tempMcapDI       = 8;
    int   tempMcapGI       = 8;
    int   spatCountMaxDI   = 1;
    int   spatCountMinDI   = 1;
    int   spatRadMaxDI     = 48;
    int   spatRadMinDI     = 32;
    int   spatCountMaxGI   = 1;
    int   spatCountMinGI   = 1;
    int   spatRadMaxGI     = 32;
    int   spatRadMinGI     = 32;
    bool  enableTempDI     = true;
    bool  enableTempGI     = true;
    bool  enableSpatDI     = true;
    bool  enableSpatGI     = true;
    float reuseRoughnessMin = 0.15f;
    float reuseRoughnessMax = 0.6f;

    UINT Flags() const {
        return (enableTempDI ? 1u : 0u) | (enableTempGI ? 2u : 0u)
             | (enableSpatDI ? 4u : 0u) | (enableSpatGI ? 8u : 0u);
    }
};

// ── Halton sequence for jitter ───────────────────────────────────
inline float Halton(uint32_t index, uint32_t base) {
    float f = 1.0f, r = 0.0f;
    while (index > 0) { f /= base; r += f * (index % base); index /= base; }
    return r;
}

// ── Octahedral normal encoding ───────────────────────────────────
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
