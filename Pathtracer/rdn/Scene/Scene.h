#pragma once
// ═══════════════════════════════════════════════════════════════════
// Scene/Scene.h — Scene graph with model-level transforms, dirty
//                 tracking for TLAS and light tree.
// ═══════════════════════════════════════════════════════════════════

#include "../Common.h"
#include "../LightTree.h"
#include "OmmBuilder.h"

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

    // Object-space AABB over cpuVertices. Computed once at load time
    // by AssetLoader; used by Renderer to derive the scene-wide world
    // AABB that feeds NRC's position normalization (the Frequency
    // encoding expects inputs roughly in [0, 1]).
    XMFLOAT3 localAabbMin = {  FLT_MAX,  FLT_MAX,  FLT_MAX };
    XMFLOAT3 localAabbMax = { -FLT_MAX, -FLT_MAX, -FLT_MAX };

    // Opacity Micro-Maps
    OmmBakeResult          ommBake;
    ComPtr<ID3D12Resource> ommArray;
    ComPtr<ID3D12Resource> ommIndexBuffer;
    bool                   hasOmm = false;
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
    MaterialSoA                 materials;
    std::vector<std::string>    materialNames;  // parallel to materials (CPU-only, for editor)
    std::vector<UINT>           materialIDs;

    // TLAS instance data (BLAS resource, transform, hit group contribution, flags)
    struct TLASInstance {
        ComPtr<ID3D12Resource> blas;
        XMMATRIX transform;
        UINT hitGroupContribution;
        D3D12_RAYTRACING_INSTANCE_FLAGS flags;
    };
    std::vector<TLASInstance> tlasInstances;

    // Global merged GPU buffers
    ComPtr<ID3D12Resource> vertexGlobal;
    ComPtr<ID3D12Resource> indexGlobal;
    UINT totalVertexCount = 0;
    UINT totalIndexCount  = 0;

    // Material GPU buffer — single compressed AoS struct, 40 B / material.
    // See Material_Decoder_v8.hlsli for the field layout.
    ComPtr<ID3D12Resource> materialBuffer;        // t5 — StructuredBuffer<MatPacked>
    ComPtr<ID3D12Resource> materialIndexBuffer;

    // CPU-side packed backing store (10 × uint32 per material).
    std::vector<uint32_t>  materialPacked;

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

    // CPU-side shadow of GPU instance properties (avoids WC upload-heap reads)
    std::vector<InstanceProperties> cpuInstanceProps;

    // Per-instance dirty flags (set when transform changes)
    std::vector<uint8_t> instanceDirty;

    // Per-instance initialized flags (false until first UpdateInstanceProperties)
    std::vector<uint8_t> instanceInitialized;

    // Indices of dirty instances (rebuilt each frame from instanceDirty)
    std::vector<uint32_t> dirtyInstanceList;

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

    // Mark a specific instance as having a dirty transform
    void MarkInstanceDirty(UINT instanceIndex);

    // Mark all instances dirty (e.g. after structural change)
    void MarkAllInstancesDirty();

    void BuildGlobalMeshBuffers(ID3D12Device* device, ID3D12GraphicsCommandList* cmdList);
    void CreateInstancePropertiesBuffer(ID3D12Device* device);
    void PrepareInstanceProperties();   // CPU math on shadow buffer (safe to overlap with GPU)
    void UploadInstanceProperties();    // memcpy to GPU upload heap (must wait for GPU first)
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
