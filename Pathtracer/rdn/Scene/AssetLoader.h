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

    // Spatial split threshold: meshes larger than this get recursively split
    // along the longest centroid-AABB axis at the centroid median so each BLAS
    // submission stays inside the driver builder's VRAM budget. At ~25M tris,
    // BLAS scratch+result stays under ~3 GB combined.
    static constexpr UINT MAX_TRIS_PER_MESH = 25'000'000;

private:
    static MeshSplitResult SplitOpaqueAlpha(
        const std::vector<UINT>& indices,
        const std::vector<UINT>& perTriMatIDs,
        const MaterialSoA& allMaterials);

    // Recursive binary split. Each output piece has at most maxTris triangles
    // and carries only the vertices its triangles reference.
    static std::vector<LoadedMesh> SplitMeshSpatial(LoadedMesh mesh, UINT maxTris);

    // Walks scene.meshes; oversized meshes get replaced by their split pieces,
    // and the corresponding instances fan out 1→N.
    static void SplitOversizedMeshes(LoadedScene& scene, UINT maxTris);
};
