#pragma once
#include "../rdn/Scene/Scene.h"
#include "RoyalTracerApi.h"
#include <functional>

// Scene data staged by the host. Build turns it into the renderer's Scene the way AssetLoader::LoadModels does
// for model files, but without merging meshes or baking transforms: the host moves every instance.
struct StagedScene {
    struct Texture {
        uint32_t width = 0, height = 0, kind = RT_TEXTURE_ALBEDO;
        std::vector<uint8_t> rgba;
    };
    struct Mesh {
        std::vector<Vertex> vertices;
        std::vector<UINT> indices;
        std::vector<UINT> triangleMaterials;
    };
    struct Instance {
        UINT mesh = 0;
        XMMATRIX world = XMMatrixIdentity();
    };
    std::vector<Texture> textures;
    std::vector<RT_MaterialDesc> materials;
    std::vector<Mesh> meshes;
    std::vector<Instance> instances;
};

namespace scene_bridge {
// Fills an empty Scene. Textures are uploaded through cmdList, which flush() submits and reopens. Releases the
// staged texture pixels and mesh data once they are in the Scene.
void Build(StagedScene& staged, Scene& scene, ID3D12Device* device, ID3D12GraphicsCommandList* cmdList,
           const std::function<void()>& flush);

// Row-major 3x4 object-to-world (column vectors) to the renderer's row-vector XMMATRIX.
XMMATRIX ToMatrix(const float* rowMajor3x4);
} // namespace scene_bridge
