#pragma once

#ifdef __cplusplus
#include <DirectXMath.h>
#include <cstdint>
namespace lt {
#define LT_PACKED_FLOAT3 DirectX::XMFLOAT3
#define LT_PACKED_UINT uint32_t
#else
#define LT_PACKED_FLOAT3 float3
#define LT_PACKED_UINT uint
#endif

// Leaves use index as the slot ID.
struct LightTLASNodePacked {
    LT_PACKED_FLOAT3 bmin; float power;
    LT_PACKED_FLOAT3 bmax; float cosTheta_o;
    LT_PACKED_UINT axis; float sinTheta_o;
    LT_PACKED_UINT index; LT_PACKED_UINT childCount;
};

// Per mesh: 32-byte header (FP32 min.xyz, max.xyz), then nodes; IDs exclude it.
struct LightBLASNodePacked {
    LT_PACKED_UINT boundsX, boundsY, boundsZ; float power; // bounds: lower | upper << 16
    LT_PACKED_UINT axis; float cosTheta_o;
    LT_PACKED_UINT index; LT_PACKED_UINT childCount; // leaf: triangle offset, one triangle
};

#ifdef __cplusplus
static_assert(sizeof(LightTLASNodePacked) == 48);
static_assert(sizeof(LightBLASNodePacked) == 32);
constexpr uint32_t LightBLASNodeStride(bool compact) { return compact ? 32u : 64u; }
constexpr uint32_t LightTLASNodeStride(bool compact) { return compact ? 48u : 64u; }
} // namespace lt
#endif
#undef LT_PACKED_FLOAT3
#undef LT_PACKED_UINT
