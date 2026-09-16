#pragma once

#include "../Common.h"
#include <DirectXTex.h>

struct OmmBakeResult {
    std::vector<uint8_t> rawData;
    std::vector<D3D12_RAYTRACING_OPACITY_MICROMAP_DESC> ommDescs;
    std::vector<int32_t> triOmmIndices;
    std::vector<D3D12_RAYTRACING_OPACITY_MICROMAP_HISTOGRAM_ENTRY> histogram;

    uint32_t alphaTriCount = 0;
    bool empty() const { return alphaTriCount == 0; }
};

struct OmmGpuData {
    ComPtr<ID3D12Resource> ommArray;
    ComPtr<ID3D12Resource> ommIndexBuffer;
    bool valid = false;

    ComPtr<ID3D12Resource> _scratchBuffer;
    ComPtr<ID3D12Resource> _inputBuffer;
    ComPtr<ID3D12Resource> _descBuffer;
};

struct MeshGPU;
struct MaterialSoA;

class OmmBuilder {
  public:
    static void BakeAll(std::vector<MeshGPU>& meshes, const MaterialSoA& materials,
                        const std::vector<DirectX::ScratchImage*>& albedoImages);

    static OmmGpuData BuildGPU(const OmmBakeResult& bake, ID3D12Device5* device, ID3D12GraphicsCommandList4* cmdList);
};
