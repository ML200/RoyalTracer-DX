#pragma once
//====================================
//ASSET LOADER
//====================================
//loads models into Scene, creates SceneModel entries for editor

#include "Scene.h"
#include "../src/Util/ObjLoader.h"

struct ModelEntry {
    std::string path;
    XMMATRIX    transform = XMMatrixIdentity();
    std::string name      = "";
};

struct MeshSplitResult {
    std::vector<UINT> reorderedIndices;
    std::vector<UINT> reorderedMaterialIDs;
    UINT opaqueTriCount;
    UINT alphaTriCount;
};

using FlushFn = std::function<void()>;

class AssetLoader {
public:
    static void LoadModels(
        const std::vector<ModelEntry>& modelEntries,
        Scene& scene,
        ID3D12Device* device,
        ID3D12GraphicsCommandList* cmdList,
        const FlushFn& flushAndReset);

    static void CreateBindlessTextures(
        std::vector<TextureData>& textures,
        UINT heapBaseSlot,
        const std::wstring& debugPrefix,
        ID3D12Device* device,
        ID3D12GraphicsCommandList* cmdList,
        std::vector<ComPtr<ID3D12Resource>>& outGpuTextures,
        const FlushFn& flushAndReset);

private:
    static MeshSplitResult SplitOpaqueAlpha(
        const std::vector<UINT>& indices,
        const std::vector<UINT>& perTriMatIDs,
        const MaterialSoA& allMaterials);
};
