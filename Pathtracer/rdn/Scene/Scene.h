#pragma once
//====================================
//SCENE GRAPH
//====================================
//model-level transforms, dirty tracking for TLAS and light tree

#include "../Common.h"
#include "../LightTree.h"
#include "OmmBuilder.h"

//====================================
//MESH GPU
//====================================
//one unique geometry, one BLAS
struct MeshGPU {
    ComPtr<ID3D12Resource> blas;
    ComPtr<ID3D12Resource> vertexBuffer;
    ComPtr<ID3D12Resource> indexBuffer;

    UINT vertexCount    = 0;
    UINT indexCount     = 0;
    UINT opaqueTriCount = 0;
    UINT alphaTriCount  = 0;

    UINT globalVertexBase = 0;
    UINT globalIndexBase  = 0;
    UINT materialIDBase   = 0;

    std::vector<Vertex>  cpuVertices;
    std::vector<UINT>    cpuIndices;
    std::vector<UINT>    cpuMaterialIDs;

    //object-space AABB over cpuVertices, used by Renderer for NRC position normalization
    XMFLOAT3 localAabbMin = {  FLT_MAX,  FLT_MAX,  FLT_MAX };
    XMFLOAT3 localAabbMax = { -FLT_MAX, -FLT_MAX, -FLT_MAX };

    //Opacity Micro-Maps
    OmmBakeResult          ommBake;
    ComPtr<ID3D12Resource> ommArray;
    ComPtr<ID3D12Resource> ommIndexBuffer;
    bool                   hasOmm = false;
};

//====================================
//SCENE INSTANCE
//====================================
//placed sub-object from GLB scene graph
struct SceneInstance {
    UINT     meshIndex     = 0;
    UINT     modelIndex    = 0;
    XMMATRIX localTransform  = XMMatrixIdentity();
    XMMATRIX worldTransform  = XMMatrixIdentity();
    XMMATRIX prevWorldTransform = XMMatrixIdentity();
    std::string name = "Instance";
};

//====================================
//SCENE MODEL
//====================================
//loaded model (GLB/OBJ), editor entry
struct SceneModel {
    std::string name     = "Model";
    std::string filePath = "";

    //world transform, editor-editable
    XMFLOAT3 position = { 0, 0, 0 };
    XMFLOAT3 rotation = { 0, 0, 0 };
    XMFLOAT3 scale    = { 1, 1, 1 };
    XMMATRIX worldTransform = XMMatrixIdentity();

    //instance range
    UINT instanceStart = 0;
    UINT instanceCount = 0;

    //mesh range
    UINT meshStart = 0;
    UINT meshCount = 0;

    void RebuildTransform() {
        worldTransform = XMMatrixScaling(scale.x, scale.y, scale.z)
                       * XMMatrixRotationRollPitchYaw(
                             XMConvertToRadians(rotation.x),
                             XMConvertToRadians(rotation.y),
                             XMConvertToRadians(rotation.z))
                       * XMMatrixTranslation(position.x, position.y, position.z);
    }
};

//====================================
//SCENE
//====================================
struct Scene {
    //data
    std::vector<MeshGPU>        meshes;
    std::vector<SceneInstance>  instances;
    std::vector<SceneModel>     models;
    MaterialSoA                 materials;
    std::vector<std::string>    materialNames;
    std::vector<UINT>           materialIDs;

    //TLAS instance data
    struct TLASInstance {
        ComPtr<ID3D12Resource> blas;
        XMMATRIX transform;
        UINT hitGroupContribution;
        D3D12_RAYTRACING_INSTANCE_FLAGS flags;
    };
    std::vector<TLASInstance> tlasInstances;

    //global merged GPU buffers. PLANET: these now hold a SCENE region followed
    //by a reserved TERRAIN region (the streamed planet writes its chunk meshes
    //there so terrain shades through the unified EvalSurfaceState). They are
    //UPLOAD-heap + persistently mapped so the planet tessellator can write a
    //cell's slot directly (write-once before the cell goes live -> no GPU copy,
    //no barrier). vertexGlobalMapped/indexGlobalMapped point at the buffer base.
    ComPtr<ID3D12Resource> vertexGlobal;
    ComPtr<ID3D12Resource> indexGlobal;
    uint8_t* vertexGlobalMapped = nullptr;   // persistent map (UPLOAD)
    uint8_t* indexGlobalMapped  = nullptr;
    UINT totalVertexCount = 0;                // scene vertices (terrain region starts here)
    UINT totalIndexCount  = 0;                // scene indices

    //====================================
    //PLANET TERRAIN REGION (reserved in the combined buffers)
    //====================================
    //Element counts reserved AFTER the scene data, set by Renderer::ReserveTerrain
    //before the buffers are built. The per-triangle material/light regions are
    //SHARED by every terrain cell (all terrain triangles use one flat material
    //and no light), so they are sized to one cell's worst case.
    UINT terrainVertexElems     = 0;   // = terrainLeafSlots * MAX_CHUNK_VERTS
    UINT terrainIndexElems      = 0;   // = terrainLeafSlots * MAX_CHUNK_TRIS*3
    UINT terrainMatIDElems      = 0;   // = max_leaves_per_cell * MAX_CHUNK_TRIS (shared)
    UINT terrainTriLightElems   = 0;   // = max_leaves_per_cell * MAX_CHUNK_TRIS (shared)
    UINT terrainInstanceSlots   = 0;   // = MAX_TERRAIN_CELLS
    UINT terrainVertexBase      = 0;   // element offset where terrain verts begin (= totalVertexCount)
    UINT terrainIndexBase       = 0;   // element offset where terrain indices begin (= totalIndexCount)
    UINT terrainMatIDBase       = 0;   // element offset of the shared terrain materialID region
    UINT terrainTriLightBase    = 0;   // element offset of the shared terrain triToLightId region
    UINT terrainMatIndex        = 0;   // g_mat index of the flat terrain material
    //FIXED base where terrain InstanceProperties begin (= scene-instance
    //capacity). Terrain InstanceID = terrainPropsBase + stable_id, so it never
    //shifts when scene instances are added/removed at runtime. Scene instances
    //occupy [0, terrainPropsBase) (capped there by the TLAS), terrain occupies
    //[terrainPropsBase, terrainPropsBase + terrainInstanceSlots) - disjoint.
    UINT terrainPropsBase       = 0;
    //PLANET ROCKS: a second reserved instanceProps range after the terrain
    //region, for camera-streamed scatter rocks. Set by ReserveRocks AFTER
    //ReserveTerrain: rockPropsBase = terrainPropsBase + terrainInstanceSlots.
    UINT rockPropsBase          = 0;
    UINT rockInstanceSlots      = 0;
    UINT combinedVertexCount() const { return totalVertexCount + terrainVertexElems; }
    UINT combinedIndexCount()  const { return totalIndexCount  + terrainIndexElems;  }
    //element count of the instanceProperties buffer / SRV. With terrain enabled
    //this is FIXED (terrainPropsBase + terrainInstanceSlots) so the buffer never
    //needs reallocation when scene instances change, and the terrain region is
    //always covered. Without terrain it tracks the scene instance count.
    UINT instancePropsCount() const {
        const UINT base = terrainInstanceSlots ? (terrainPropsBase + terrainInstanceSlots)
                                               : (UINT)instances.size();
        return base + rockInstanceSlots;
    }

    //material GPU, 40B compressed AoS, see Material_Decoder_v8.hlsli
    ComPtr<ID3D12Resource> materialBuffer;
    ComPtr<ID3D12Resource> materialIndexBuffer;

    //CPU packed backing store, 10 uint32 per material
    std::vector<uint32_t>  materialPacked;

    ComPtr<ID3D12Resource> instanceProperties;

    //emissive / light data
    std::vector<LightTriangle>  emissiveTriangles;
    ComPtr<ID3D12Resource>      emissiveTrianglesBuffer;
    std::vector<uint32_t>       instTriOffset;
    std::vector<uint32_t>       triToLightId;
    ComPtr<ID3D12Resource>      triToLightIdBuffer;

    std::vector<GeometryOffsets> geoOffsets;

    //bindless textures
    UINT bindlessAlbedoBase  = 0;
    UINT bindlessNormalBase  = 0;
    UINT bindlessRmaBase     = 0;
    UINT totalBindlessTextures = 0;
    std::vector<ComPtr<ID3D12Resource>> bindlessGpuTextures;

    //CPU shadow of GPU instance properties, avoids WC upload-heap reads
    std::vector<InstanceProperties> cpuInstanceProps;

    //per-instance dirty flags
    std::vector<uint8_t> instanceDirty;

    //per-instance init flags, false until first UpdateInstanceProperties
    std::vector<uint8_t> instanceInitialized;

    //dirty instance list, rebuilt each frame from instanceDirty
    std::vector<uint32_t> dirtyInstanceList;

    //====================================
    //DIRTY FLAGS
    //====================================
    bool tlasDirty       = true;
    bool tlasFullRebuild = true;
    bool lightTreeDirty  = true;
    bool materialsDirty  = false;
    bool emissivesDirty  = false;

    //====================================
    //FLOATING ORIGIN
    //====================================
    //Per-frame shift applied to every instance transform before it lands
    //in cpuInstanceProps / tlasInstances. Source of truth (si.worldTransform)
    //stays in absolute world coords; this is the GPU side compensation.
    //Updated by the renderer from Camera::getSceneOriginWorld() before
    //PrepareInstanceProperties / RebuildTLASInstanceList run.
    XMFLOAT3 sceneOriginWorld = { 0.0f, 0.0f, 0.0f };
    //Origin used by the PREVIOUS PrepareInstanceProperties call. The delta
    //between the two is applied to every instance's prevObjectToWorld so it
    //stays consistent with Camera::m_prevView after a floating-origin snap;
    //without this, prevObjectToWorld lives in the old shifted frame while
    //prevView expects the new one, and motion vectors gain a shiftDelta-
    //sized offset on every snap (visible as a tearing/destabilization
    //past ~500 m of camera drift).
    XMFLOAT3 prevSceneOriginWorld = { 0.0f, 0.0f, 0.0f };

    //====================================
    //METHODS
    //====================================
    //recompute instance worldTransforms from model worldTransform, sets tlasDirty
    void PropagateModelTransforms();

    void MarkModelMoved(UINT modelIndex);
    //emission changes also set lightTreeDirty
    void MarkMaterialsDirty(bool emissionChanged = false);
    void MarkInstanceDirty(UINT instanceIndex);
    void MarkAllInstancesDirty();

    //PLANET: reserve the terrain region in the combined buffers + append the
    //flat terrain material. MUST be called after assets/materials are loaded
    //and BEFORE BuildGlobalMeshBuffers / UploadMaterials / CreateTriToLightIdBuffer
    /// CreateInstancePropertiesBuffer. Element counts are computed by the
    //Renderer from the planet config + chunk constants.
    void ReserveTerrain(UINT vertexElems, UINT indexElems, UINT matIDElems,
                        UINT triLightElems, UINT instanceSlots, UINT propsBase);
    //PLANET ROCKS: reserve `instanceSlots` instanceProps slots after the terrain
    //region. Call AFTER ReserveTerrain (needs terrainPropsBase/terrainInstanceSlots).
    void ReserveRocks(UINT instanceSlots);

    void BuildGlobalMeshBuffers(ID3D12Device* device, ID3D12GraphicsCommandList* cmdList);
    void CreateInstancePropertiesBuffer(ID3D12Device* device);
    //CPU math on shadow buffer, safe to overlap with GPU
    void PrepareInstanceProperties();
    //memcpy to upload heap, must wait for GPU first
    void UploadInstanceProperties();
    void RebuildTLASInstanceList();
    void CollectEmissiveTriangles();
    void CreateEmissiveTrianglesBuffer(ID3D12Device* device, ID3D12GraphicsCommandList* cmdList, ID3D12CommandQueue* queue, ID3D12CommandAllocator* alloc);
    void CreateTriToLightIdBuffer(ID3D12Device* device, ID3D12GraphicsCommandList* cmdList);
    void UploadMaterials(ID3D12Device* device);
    void UpdateMaterialBuffer();

    float ComputeTriangleWeight(const XMFLOAT3& v0, const XMFLOAT3& v1,
                                const XMFLOAT3& v2, const XMFLOAT3& emissive,
                                const XMMATRIX& M);
};
