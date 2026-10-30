#pragma once

#include "../Common.h"
#include "../Clusters/ClusterPages.h"
#include <DirectXTex.h>
#include <functional>
#include <unordered_map>

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

// Cutout micromaps for cluster pages, one page at a time; keeps each texture's alpha between pages.
class OmmPageBaker {
  public:
    // albedo: the image of an albedo texture ID, or null.
    OmmPageBaker(const MaterialSoA& materials, std::function<const DirectX::Image*(int texID)> albedo);
    ~OmmPageBaker();
    OmmPageBaker(const OmmPageBaker&) = delete;
    OmmPageBaker& operator=(const OmmPageBaker&) = delete;
    // The page's micromaps for its material, over indices into uvs (two floats per vertex). Transmissive and thin-glass
    // materials get unknown-opaque special indices, untextured or uniformly opaque ones fully-opaque special indices.
    void Bake(uint32_t material, int maxLevel, const float* uvs, uint32_t vertexCount, const uint32_t* indices,
              uint32_t triCount, clusters::PageOmm& out);

  private:
    struct Texture {
        void* handle = nullptr; // ommCpuTexture
        std::vector<uint8_t> alpha;
        uint8_t lo = 255, hi = 0;
    };
    const MaterialSoA& m_materials;
    std::function<const DirectX::Image*(int)> m_albedo;
    void* m_baker = nullptr; // ommBaker
    std::unordered_map<int, Texture> m_textures;
};

class OmmBuilder {
  public:
    static void BakeAll(std::vector<MeshGPU>& meshes, const MaterialSoA& materials,
                        const std::vector<DirectX::ScratchImage*>& albedoImages);

    static OmmGpuData BuildGPU(const OmmBakeResult& bake, ID3D12Device5* device, ID3D12GraphicsCommandList4* cmdList);
};
