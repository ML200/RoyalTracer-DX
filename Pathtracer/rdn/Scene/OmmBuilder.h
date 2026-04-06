#pragma once
// ═══════════════════════════════════════════════════════════════════
// Scene/OmmBuilder.h — Opacity Micro-Map baking (CPU) and GPU build.
//
// For each mesh with alpha-tested geometry, the builder:
//   1. Classifies every micro-triangle as opaque/transparent/unknown
//      by sampling the albedo alpha channel on the CPU.
//   2. Packs the results into an OMM array built on the GPU via
//      D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_OPACITY_MICROMAP_ARRAY.
//   3. Produces an OMM index buffer that links each alpha triangle
//      to its OMM entry (or to a special whole-triangle state).
// ═══════════════════════════════════════════════════════════════════

#include "../Common.h"
#include <DirectXTex.h>

// ── Configuration ────────────────────────────────���──────────────
static constexpr uint32_t OMM_SUBDIV_LEVEL       = 5;   // 4^5 = 1024 micro-tris per triangle
static constexpr uint32_t OMM_MICROTRIS_PER_TRI  = 1024; // 1 << (2 * OMM_SUBDIV_LEVEL)
static constexpr uint32_t OMM_BYTES_PER_OMM      = OMM_MICROTRIS_PER_TRI * 2 / 8; // 256 bytes (4-state = 2 bits)

// ── CPU bake result (one per mesh that has alpha triangles) ─────
struct OmmBakeResult {
    // Concatenated per-microtri opacity data for all non-special OMMs
    std::vector<uint8_t> rawData;

    // One descriptor per baked OMM (non-special triangles only)
    std::vector<D3D12_RAYTRACING_OPACITY_MICROMAP_DESC> ommDescs;

    // One entry per alpha triangle: positive = index into ommDescs,
    // negative = D3D12_RAYTRACING_OPACITY_MICROMAP_SPECIAL_INDEX_*
    std::vector<int32_t> triOmmIndices;

    // Histogram required for the OMM array build
    std::vector<D3D12_RAYTRACING_OPACITY_MICROMAP_HISTOGRAM_ENTRY> histogram;

    uint32_t alphaTriCount = 0;
    bool     empty() const { return alphaTriCount == 0; }
};

// ── GPU resources produced by BuildGPU ──────────────────────────
struct OmmGpuData {
    ComPtr<ID3D12Resource> ommArray;        // built OMM array (or null if special-only)
    ComPtr<ID3D12Resource> ommIndexBuffer;  // per-alpha-tri signed int32 indices
    bool valid = false;

    // Intermediate upload/scratch resources — must stay alive until the
    // command list that builds the OMM array has been executed (flushed).
    ComPtr<ID3D12Resource> _scratchBuffer;
    ComPtr<ID3D12Resource> _inputBuffer;
    ComPtr<ID3D12Resource> _descBuffer;
};

// ── Forward declarations ────────────────────────────────────────
struct MeshGPU;
struct Material;

// ── Builder ─────────────────────────────────────────────────────
class OmmBuilder {
public:
    // CPU bake: sample alpha textures and classify micro-triangles.
    // albedoTextures are the loaded ScratchImage list (mip level 0 used).
    // Materials' albedoTexID at this point is the *local* index into
    // albedoTextures (before bindless offset is applied).
    static OmmBakeResult BakeMesh(
        const MeshGPU& mesh,
        const std::vector<Material>& materials,
        const std::vector<DirectX::ScratchImage*>& albedoImages,
        const std::vector<DirectX::XMFLOAT2>& albedoUVScales);

    // GPU build: create the OMM array resource and index buffer.
    static OmmGpuData BuildGPU(
        const OmmBakeResult& bake,
        ID3D12Device5* device,
        ID3D12GraphicsCommandList4* cmdList);

private:
    // Compute the centroid barycentric (u,v) of micro-triangle `index`
    // at the given subdivision level using recursive midpoint splitting.
    static void MicroTriCentroid(uint32_t index, uint32_t level,
                                 float& outU, float& outV);

    // Compute all 3 vertex barycentrics of micro-triangle `index`.
    static void MicroTriVertices(uint32_t index, uint32_t level,
                                 float& v0u, float& v0v,
                                 float& v1u, float& v1v,
                                 float& v2u, float& v2v);

    // Bilinear sample of the alpha channel from an RGBA8 image.
    // Returns alpha in [0,1].  UVs are wrapped to [0,1).
    static float SampleAlpha(const DirectX::Image& img, float u, float v);
};
