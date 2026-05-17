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

    //global merged GPU buffers
    ComPtr<ID3D12Resource> vertexGlobal;
    ComPtr<ID3D12Resource> indexGlobal;
    UINT totalVertexCount = 0;
    UINT totalIndexCount  = 0;

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
