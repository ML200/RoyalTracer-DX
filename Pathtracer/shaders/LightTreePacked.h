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

// Spherical Gaussian light clusters (Tokuyoshi et al. 2024). A node holds the flux-weighted
// mean and total variance of its light positions, the mean resultant vector of its emission
// directions (a von Mises-Fisher lobe; leaves use half the triangle normal), the radiant flux,
// the radius of the sphere around the mean that encloses every light, and the cosine of a
// cone around the mean direction that bounds every emitter normal (a conservative back-face
// cull; -1 disables it).

// World-space top-level node. The mean resultant length and the cone cosine share a word as
// two halves. Leaves use index as the slot ID.
struct LightTLASNodePacked {
    LT_PACKED_FLOAT3 mean; float power;
    float variance; float radius; LT_PACKED_UINT axis; LT_PACKED_UINT lengthCos;
    LT_PACKED_UINT index; LT_PACKED_UINT childCount; LT_PACKED_UINT _pad0; LT_PACKED_UINT _pad1;
};

// Each mesh starts with one 32-byte header, followed by its nodes. Header word 0 holds the
// unit length of the mesh (twice the radius of its root), the other words are reserved. Node
// IDs exclude the header. The standard deviation and the radius are halves relative to the
// unit; the mean keeps FP32. index holds the leaf-triangle offset of a leaf or the first child
// of an inner node, with the child count in its top three bits.
struct LightBLASNodePacked {
    LT_PACKED_FLOAT3 mean; float power;
    LT_PACKED_UINT axis; LT_PACKED_UINT sigmaRadius; LT_PACKED_UINT lengthCos; LT_PACKED_UINT indexCount;
};

#define LT_PACKED_INDEX_BITS 29u
#define LT_PACKED_INDEX_MASK ((1u << LT_PACKED_INDEX_BITS) - 1u)

#ifdef __cplusplus
static_assert(sizeof(LightTLASNodePacked) == 48);
static_assert(sizeof(LightBLASNodePacked) == 32);
constexpr uint32_t LightBLASNodeStride(bool compact) { return compact ? 32u : 64u; }
constexpr uint32_t LightTLASNodeStride(bool compact) { return compact ? 48u : 64u; }
} // namespace lt
#endif
#undef LT_PACKED_FLOAT3
#undef LT_PACKED_UINT
