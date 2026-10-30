#pragma once

#include "../Common.h"
#include "../LightTree.h"
#include "OmmBuilder.h"
#include "BvhQuality.h"
#include <memory>
#include "../Clusters/ClusterPages.h"

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

    // Clustered geometry: the first triangle of each octree cell OrderForClusters made, the CLAS the cluster BLAS (blas)
    // references (a buffer per build piece), and the mesh's cluster IDs.
    std::vector<UINT> clusterCuts;
    std::vector<ComPtr<ID3D12Resource>> clas;
    UINT clusterBase = 0, clusterCount = 0;
    // With a cluster LOD hierarchy: every cluster by ID offset (in place of clusterCuts), and the hierarchy. blas is then
    // its full-detail cut.
    std::vector<clusters::Span> clusterSpans;
    clusters::Lod lod;
    // Paged: the hierarchy streams from its page file (source, until the pager takes it) and the mesh keeps its fixed
    // triangles, in its own range of the scene's page pool.
    bool paged = false;
    bool lodAlpha = false; // some of its hierarchy is alpha-tested or transmissive (alpha pages)
    UINT poolVertexBase = 0, poolTriBase = 0;
    std::shared_ptr<clusters::PagedSource> pagedSource;
    XMFLOAT3 boundsMin = {0, 0, 0}, boundsMax = {0, 0, 0}; // object space

    void CreateBlasBuildInputs(ID3D12Device* device);
    // Reorders the opaque and the alpha triangles, each among themselves, into compact octree cells: a cluster is a run
    // of the triangle order. Before anything derives from that order (material IDs, micromaps, lights).
    void OrderForClusters();
    // In place of OrderForClusters: a cluster LOD hierarchy over the opaque triangles with a region (per triangle; their
    // material, or clusters::NO_REGION), every level's triangles appended to the opaque ones, all of them non-emissive.
    // Regions below clusters::MIN_LOD_TRIS stay outside. False (and ordered for clusters) when nothing qualified.
    bool BuildClusterLod(std::vector<uint32_t> regions);
    // BuildClusterLod's paged form: the hierarchy is baked once into a page file in clusters::cache_dir() (named after
    // the mesh's content), read back as tables, and the mesh keeps only the triangles outside it. materialBase: the
    // model's first material, so the file holds model-relative material IDs.
    // Alpha regions join too: foliage(region) simplifies with dilated borders, opacity bakes their pages' micromaps.
    bool BuildPagedLod(std::vector<uint32_t> regions, uint32_t materialBase, std::function<bool(uint32_t)> foliage,
                       decltype(clusters::BakeMesh::opacity) opacity);
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
    bvh::Survey bvhSurvey; // import-time; motion can change overlap
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
    // The cluster page pool at the front of the vertex, index (3 per triangle) and material-ID arrays, so a triangle
    // position means the same in the latter two; paged meshes' own ranges and their streamed pages live there.
    UINT pagePoolVertices = 0, pagePoolTris = 0;
    clusters::RangeAllocator pagePoolVertexAlloc, pagePoolTriAlloc;
    D3D12_RESOURCE_STATES globalReadState = D3D12_RESOURCE_STATE_GENERIC_READ;
    // Before any mesh: false when meshes were loaded already.
    bool ReservePagePool();
    // With no paged mesh: before anything but the meshes' material IDs used the positions.
    void ReleasePagePool();

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
    bool hostWritesGeometry = false; // a host stream copies into the global buffers on the compute queue

    UINT oceanVertexElems = 0;
    UINT oceanIndexElems = 0;
    UINT oceanMatIDElems = 0;
    UINT oceanInstanceSlots = 0;
    UINT oceanVertexBase = 0;
    UINT oceanIndexBase = 0;
    UINT oceanMatIDBase = 0;
    UINT oceanPropsBase = 0;
    UINT oceanMatIndex = 0;
    bool oceanMaterialEdited = false; // editor edits override generated values
    bool oceanMatIDReserved = false;

    // The test scatter's object (RDN_SCATTER with a name part): its parts, meshes of their own placed as in the scene
    // (relative to the scene origin then); the whole of them, one mesh (the scene instance's that shows it); and its
    // far-field stand-in, three textured planes in the whole's space.
    struct ScatterObject {
        std::vector<std::pair<UINT, XMMATRIX>> parts;
        UINT whole = ~0u, wholeInstance = ~0u, standIn = ~0u;
    };
    ScatterObject scatter;

    // Instance sets (clusters::InstanceSets): a record per set mesh, the mesh in world space; the sets' instances
    // carry a record's index as their instance ID and their own transforms in the TLAS.
    UINT setPropsBase = 0, setSlots = 0;
    void ReserveInstanceSets(UINT slots);
    // The set record of slot for mesh: identity transforms, no light records.
    void WriteSetRecord(UINT slot, UINT meshIndex);

    UINT combinedVertexCount() const {
        return totalVertexCount + terrainVertexElems + voxelVertexElems + oceanVertexElems;
    }
    UINT combinedIndexCount() const {
        return totalIndexCount + terrainIndexElems + voxelIndexElems + oceanIndexElems;
    }

    UINT instancePropsCount() const {
        // Each stream reserves its own range.
        const UINT base = terrainInstanceSlots ? (terrainPropsBase + terrainInstanceSlots) : (UINT)instances.size();
        const UINT withRocks = base + rockInstanceSlots;
        const UINT withVoxels =
            voxelInstanceSlots ? std::max(withRocks, voxelPropsBase + voxelInstanceSlots) : withRocks;
        const UINT withSets = setSlots ? std::max(withVoxels, setPropsBase + setSlots) : withVoxels;
        return oceanInstanceSlots ? std::max(withSets, oceanPropsBase + oceanInstanceSlots) : withSets;
    }

    UINT sceneInstanceCap() const {
        UINT cap = terrainInstanceSlots ? terrainPropsBase : 0u;
        if (voxelInstanceSlots)
            cap = cap ? std::min(cap, voxelPropsBase) : voxelPropsBase;
        if (setSlots)
            cap = cap ? std::min(cap, setPropsBase) : setPropsBase;
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

    void MarkModelMoved(UINT modelIndex);

    void MarkMaterialsDirty(bool emissionChanged = false);
    void MarkInstanceDirty(UINT instanceIndex);
    void MarkAllInstancesDirty();

    void ReserveTerrain(UINT vertexElems, UINT indexElems, UINT matIDElems, UINT triLightElems, UINT instanceSlots,
                        UINT propsBase);

    void ReserveRocks(UINT instanceSlots);

    void ReserveVoxels(UINT vertexElems, UINT indexElems, UINT matIDElems, UINT instanceSlots, UINT minPropsBase);

    // Ocean instances come last; shaders detect them by index.
    void ReserveOcean(UINT vertexElems, UINT indexElems, UINT matIDElems, UINT instanceSlots, const Material& mat,
                      float foamAlbedo);

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
