#pragma once
// ═══════════════════════════════════════════════════════════════════
// Scene/Scene.h — Scene graph with model-level transforms, dirty
//                 tracking for TLAS and light tree.
// ═══════════════════════════════════════════════════════════════════

#include "../Common.h"
#include "../LightTree.h"

// ── One unique piece of geometry (one BLAS) ──────────────────────
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
};

// ── One placed sub-object (from the GLB scene graph) ─────────────
struct SceneInstance {
    UINT     meshIndex     = 0;
    UINT     modelIndex    = 0;       // which SceneModel owns this instance
    XMMATRIX localTransform  = XMMatrixIdentity();  // from GLB scene graph
    XMMATRIX worldTransform  = XMMatrixIdentity();  // localTransform * model.worldTransform
    XMMATRIX prevWorldTransform = XMMatrixIdentity();
    std::string name = "Instance";
};

// ── One loaded model (GLB/OBJ) — single editor entry ─────────────
struct SceneModel {
    std::string name     = "Model";
    std::string filePath = "";

    // World transform (editor-editable)
    XMFLOAT3 position = { 0, 0, 0 };
    XMFLOAT3 rotation = { 0, 0, 0 };   // Euler degrees
    XMFLOAT3 scale    = { 1, 1, 1 };
    XMMATRIX worldTransform = XMMatrixIdentity();

    // Range of instances belonging to this model
    UINT instanceStart = 0;
    UINT instanceCount = 0;

    // Range of meshes belonging to this model
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

// ── Scene holds everything ───────────────────────────────────────
struct Scene {
    // ── Data ─────────────────────────────────────────────────────
    std::vector<MeshGPU>        meshes;
    std::vector<SceneInstance>  instances;
    std::vector<SceneModel>     models;
    std::vector<Material>       materials;
    std::vector<UINT>           materialIDs;

    // TLAS instance pairs (BLAS resource + transform)
    std::vector<std::pair<ComPtr<ID3D12Resource>, XMMATRIX>> tlasInstances;

    // Global merged GPU buffers
    ComPtr<ID3D12Resource> vertexGlobal;
    ComPtr<ID3D12Resource> indexGlobal;
    UINT totalVertexCount = 0;
    UINT totalIndexCount  = 0;

    // Material GPU buffers
    ComPtr<ID3D12Resource> materialBuffer;
    ComPtr<ID3D12Resource> materialIndexBuffer;

    // Instance property buffer
    ComPtr<ID3D12Resource> instanceProperties;

    // Emissive / light data
    std::vector<LightTriangle>  emissiveTriangles;
    ComPtr<ID3D12Resource>      emissiveTrianglesBuffer;
    std::vector<uint32_t>       instTriOffset;
    std::vector<uint32_t>       triToLightId;
    ComPtr<ID3D12Resource>      triToLightIdBuffer;

    // Geometry offsets
    std::vector<GeometryOffsets> geoOffsets;

    // Bindless textures
    UINT bindlessAlbedoBase  = 0;
    UINT bindlessNormalBase  = 0;
    UINT bindlessRmaBase     = 0;
    UINT totalBindlessTextures = 0;
    std::vector<ComPtr<ID3D12Resource>> bindlessGpuTextures;

    // ── Dirty flags ──────────────────────────────────────────────
    bool tlasDirty       = true;   // need DXR TLAS rebuild or refit
    bool tlasFullRebuild = true;   // true = full rebuild, false = refit only
    bool lightTreeDirty  = true;   // need light tree TLAS refit
    bool materialsDirty  = false;  // need material buffer re-upload
    bool emissivesDirty  = false;  // emission values changed → recompute BLAS roots

    // ── Methods ──────────────────────────────────────────────────

    // Recompute all instance worldTransforms from their model's worldTransform.
    // Call after any model transform changes. Sets tlasDirty.
    void PropagateModelTransforms();

    // Mark a specific model as moved (sets dirty flags)
    void MarkModelMoved(UINT modelIndex);

    // Mark materials changed (emission changes also set lightTreeDirty)
    void MarkMaterialsDirty(bool emissionChanged = false);

    void BuildGlobalMeshBuffers(ID3D12Device* device, ID3D12GraphicsCommandList* cmdList);
    void CreateInstancePropertiesBuffer(ID3D12Device* device);
    void UpdateInstanceProperties();
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
