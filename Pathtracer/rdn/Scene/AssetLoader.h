#pragma once
// ═══════════════════════════════════════════════════════════════════
// Scene/AssetLoader.h — Loads models into Scene, creating
//                       SceneModel entries for editor control.
// ═══════════════════════════════════════════════════════════════════

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

class AssetLoader {
public:
    static void LoadModels(
        const std::vector<ModelEntry>& models,
        Scene& scene,
        ID3D12Device* device,
        ID3D12GraphicsCommandList* cmdList);

    static void CreateBindlessTextures(
        std::vector<TextureData>& textures,
        UINT heapBaseSlot,
        const std::wstring& debugPrefix,
        ID3D12Device* device,
        ID3D12GraphicsCommandList* cmdList,
        std::vector<ComPtr<ID3D12Resource>>& outGpuTextures,
        std::vector<ComPtr<ID3D12Resource>>& outUploadHeaps);

private:
    static MeshSplitResult SplitOpaqueAlpha(
        const std::vector<UINT>& indices,
        const std::vector<UINT>& perTriMatIDs,
        const std::vector<Material>& allMaterials);
};
