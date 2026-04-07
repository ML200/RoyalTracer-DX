#pragma once
// ═══════════════════════════════════════════════════════════════════
// Scene/OmmBuilder.h — Opacity Micro-Map baking via NVIDIA OMM SDK
//                      and D3D12 GPU build.
// ═══════════════════════════════════════════════════════════════════

#include "../Common.h"
#include <DirectXTex.h>

// ── CPU bake result (one per mesh that has alpha triangles) ─────
struct OmmBakeResult {
    std::vector<uint8_t> rawData;
    std::vector<D3D12_RAYTRACING_OPACITY_MICROMAP_DESC> ommDescs;
    std::vector<int32_t> triOmmIndices;
    std::vector<D3D12_RAYTRACING_OPACITY_MICROMAP_HISTOGRAM_ENTRY> histogram;

    uint32_t alphaTriCount = 0;
    bool     empty() const { return alphaTriCount == 0; }
};

// ── GPU resources produced by BuildGPU ──────────────────────────
struct OmmGpuData {
    ComPtr<ID3D12Resource> ommArray;
    ComPtr<ID3D12Resource> ommIndexBuffer;
    bool valid = false;

    ComPtr<ID3D12Resource> _scratchBuffer;
    ComPtr<ID3D12Resource> _inputBuffer;
    ComPtr<ID3D12Resource> _descBuffer;
};

struct MeshGPU;
struct Material;

class OmmBuilder {
public:
    // Batch-bake all meshes at once. Bakes once per unique texture,
    // then distributes results back to per-mesh OmmBakeResult.
    static void BakeAll(
        std::vector<MeshGPU>& meshes,
        const std::vector<Material>& materials,
        const std::vector<DirectX::ScratchImage*>& albedoImages);

    // GPU build: create the OMM array resource and index buffer.
    static OmmGpuData BuildGPU(
        const OmmBakeResult& bake,
        ID3D12Device5* device,
        ID3D12GraphicsCommandList4* cmdList);
};
