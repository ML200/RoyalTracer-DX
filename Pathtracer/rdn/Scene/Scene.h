#pragma once

#include "../Common.h"
#include "../LightTree.h"
#include "OmmBuilder.h"

struct MeshGPU {
    ComPtr<ID3D12Resource> blas;
    ComPtr<ID3D12Resource> vertexBuffer;
    ComPtr<ID3D12Resource> indexBuffer;

    UINT vertexCount = 0;
    UINT indexCount = 0;
    UINT opaqueTriCount = 0;
    UINT alphaTriCount = 0;

    UINT globalVertexBase = 0;
    UINT globalIndexBase = 0;
    UINT materialIDBase = 0;

    std::vector<Vertex> cpuVertices;
    std::vector<UINT> cpuIndices;
    std::vector<UINT> cpuMaterialIDs;

    OmmBakeResult ommBake;
    ComPtr<ID3D12Resource> ommArray;
    ComPtr<ID3D12Resource> ommIndexBuffer;
    bool hasOmm = false;

    void CreateBlasBuildInputs(ID3D12Device* device);
};

struct SceneInstance {
    UINT meshIndex = 0;
    UINT modelIndex = 0;
    XMMATRIX localTransform = XMMatrixIdentity();
    XMMATRIX worldTransform = XMMatrixIdentity();
    XMMATRIX prevWorldTransform = XMMatrixIdentity();
    std::string name = "Instance";
};

struct SceneModel {
    std::string name = "Model";
    std::string filePath = "";

    XMFLOAT3 position = {0, 0, 0};
    XMFLOAT3 rotation = {0, 0, 0};
    XMFLOAT3 scale = {1, 1, 1};
    XMMATRIX worldTransform = XMMatrixIdentity();

    UINT instanceStart = 0;
    UINT instanceCount = 0;

    UINT meshStart = 0;
    UINT meshCount = 0;

    void RebuildTransform() {
        worldTransform = XMMatrixScaling(scale.x, scale.y, scale.z) *
                         XMMatrixRotationRollPitchYaw(XMConvertToRadians(rotation.x), XMConvertToRadians(rotation.y),
                                                      XMConvertToRadians(rotation.z)) *
                         XMMatrixTranslation(position.x, position.y, position.z);
    }
};

struct Scene {
    std::vector<MeshGPU> meshes;
    std::vector<SceneInstance> instances;
    std::vector<SceneModel> models;
    MaterialSoA materials;
    std::vector<std::string> materialNames;
    std::vector<UINT> materialIDs;

    struct TLASInstance {
        ComPtr<ID3D12Resource> blas;
        XMMATRIX transform;
        UINT hitGroupContribution;
        D3D12_RAYTRACING_INSTANCE_FLAGS flags;
    };
    std::vector<TLASInstance> tlasInstances;

    ComPtr<ID3D12Resource> vertexGlobal;
    ComPtr<ID3D12Resource> indexGlobal;
    uint8_t* vertexGlobalMapped = nullptr;
    uint8_t* indexGlobalMapped = nullptr;

    ComPtr<ID3D12Resource> vertexGlobalUpload;
    ComPtr<ID3D12Resource> indexGlobalUpload;
    UINT totalVertexCount = 0;
    UINT totalIndexCount = 0;

    UINT terrainVertexElems = 0;
    UINT terrainIndexElems = 0;
    UINT terrainMatIDElems = 0;
    UINT terrainTriLightElems = 0;
    UINT terrainInstanceSlots = 0;
    UINT terrainVertexBase = 0;
    UINT terrainIndexBase = 0;
    UINT terrainMatIDBase = 0;
    UINT terrainTriLightBase = 0xFFFFFFFFu;
    UINT terrainMatIndex = 0;

    UINT terrainPropsBase = 0;

    UINT rockPropsBase = 0;
    UINT rockInstanceSlots = 0;

    UINT voxelVertexElems = 0;
    UINT voxelIndexElems = 0;
    UINT voxelMatIDElems = 0;
    UINT voxelInstanceSlots = 0;
    UINT voxelVertexBase = 0;
    UINT voxelIndexBase = 0;
    UINT voxelMatIDBase = 0;
    UINT voxelPropsBase = 0;
    bool voxelMatIDReserved = false;
    UINT combinedVertexCount() const { return totalVertexCount + terrainVertexElems + voxelVertexElems; }
    UINT combinedIndexCount() const { return totalIndexCount + terrainIndexElems + voxelIndexElems; }

    UINT instancePropsCount() const {
        // Property ranges are reserved independently for terrain, rocks, and voxels.
        const UINT base = terrainInstanceSlots ? (terrainPropsBase + terrainInstanceSlots) : (UINT)instances.size();
        const UINT withRocks = base + rockInstanceSlots;
        return voxelInstanceSlots ? std::max(withRocks, voxelPropsBase + voxelInstanceSlots) : withRocks;
    }

    UINT sceneInstanceCap() const {
        UINT cap = terrainInstanceSlots ? terrainPropsBase : 0u;
        if (voxelInstanceSlots)
            cap = cap ? std::min(cap, voxelPropsBase) : voxelPropsBase;
        return cap;
    }

    ComPtr<ID3D12Resource> materialBuffer;
    ComPtr<ID3D12Resource> materialIndexBuffer;

    std::vector<uint32_t> materialPacked;

    ComPtr<ID3D12Resource> instanceProperties;

    std::vector<LightTriangle> emissiveTriangles;
    ComPtr<ID3D12Resource> emissiveTrianglesBuffer;

    std::vector<ComPtr<ID3D12Resource>> pendingLightUploads;
    std::vector<uint32_t> triToLightId;
    std::vector<uint32_t> meshLightBase;
    std::vector<uint32_t> meshRecordBase;
    std::vector<lt::LightInstanceRef> lightInstances;
    std::vector<uint32_t> instanceLightSlot;
    bool HasMeshLights() const { return !lightInstances.empty(); }

    enum LightClass : uint8_t { LightClassScene = 0, LightClassCubes = 1, LightClassCount = 2 };
    std::vector<uint8_t> meshLightClass;
    bool lightClassEnabled[LightClassCount] = {true, true};
    void SetMeshLightClass(UINT meshIndex, uint8_t cls) {
        if (meshLightClass.size() <= meshIndex)
            meshLightClass.resize((size_t)meshIndex + 1, (uint8_t)LightClassScene);
        meshLightClass[meshIndex] = cls;
    }
    uint8_t MeshLightClass(UINT meshIndex) const {
        return meshIndex < meshLightClass.size() ? meshLightClass[meshIndex] : (uint8_t)LightClassScene;
    }
    bool InstanceLightsEnabled(UINT instanceIndex) const {
        return instanceIndex < instances.size() &&
               lightClassEnabled[MeshLightClass(instances[instanceIndex].meshIndex)];
    }
    ComPtr<ID3D12Resource> triToLightIdBuffer;

    std::vector<GeometryOffsets> geoOffsets;

    UINT bindlessAlbedoBase = 0;
    UINT bindlessNormalBase = 0;
    UINT bindlessRmaBase = 0;
    UINT totalBindlessTextures = 0;
    std::vector<ComPtr<ID3D12Resource>> bindlessGpuTextures;

    std::vector<InstancePropertiesCpu> cpuInstanceProps;

    std::vector<uint8_t> instanceDirty;

    std::vector<uint8_t> instanceInitialized;

    std::vector<uint32_t> dirtyInstanceList;

    bool tlasDirty = true;
    bool tlasFullRebuild = true;
    bool lightTreeDirty = true;
    bool materialsDirty = false;
    bool emissivesDirty = false;

    XMFLOAT3 sceneOriginWorld = {0.0f, 0.0f, 0.0f};

    XMFLOAT3 prevSceneOriginWorld = {0.0f, 0.0f, 0.0f};

    void PropagateModelTransforms();

    void MarkModelMoved(UINT modelIndex);

    void MarkMaterialsDirty(bool emissionChanged = false);
    void MarkInstanceDirty(UINT instanceIndex);
    void MarkAllInstancesDirty();

    void ReserveTerrain(UINT vertexElems, UINT indexElems, UINT matIDElems, UINT triLightElems, UINT instanceSlots,
                        UINT propsBase);

    void ReserveRocks(UINT instanceSlots);

    void ReserveVoxels(UINT vertexElems, UINT indexElems, UINT matIDElems, UINT instanceSlots, UINT minPropsBase);

    void BuildGlobalMeshBuffers(ID3D12Device* device, ID3D12GraphicsCommandList* cmdList);
    void CreateInstancePropertiesBuffer(ID3D12Device* device);

    void PrepareInstanceProperties();

    void UploadInstanceProperties();
    void RebuildTLASInstanceList();
    void CollectEmissiveTriangles();
    void CreateEmissiveTrianglesBuffer(ID3D12Device* device, ID3D12GraphicsCommandList* cmdList,
                                       ID3D12CommandQueue* queue, ID3D12CommandAllocator* alloc);
    void CreateTriToLightIdBuffer(ID3D12Device* device, ID3D12GraphicsCommandList* cmdList);
    void ReleaseLightUploadStaging() { pendingLightUploads.clear(); }
    void UploadMaterials(ID3D12Device* device);
    void UpdateMaterialBuffer();

    float ComputeTriangleWeight(const XMFLOAT3& v0, const XMFLOAT3& v1, const XMFLOAT3& v2, const XMFLOAT3& emissive,
                                const XMMATRIX& M);
};
