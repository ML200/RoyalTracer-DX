# Renderer Refactoring Guide — Exact Instructions

---

## TABLE OF CONTENTS

1. [Remove ML/ONNX/DML Code](#1-remove-mlonnxdml-code)
2. [Remove Alias Table Code](#2-remove-alias-table-code)
3. [Remove Non-Bindless Texture Code](#3-remove-non-bindless-texture-code)
4. [Instance Architecture Overhaul](#4-instance-architecture-overhaul)

---

## 1. REMOVE ML/ONNX/DML CODE

**Keep:** All `#include` lines at the top of both files.
**Remove:** Every ML-related member variable, function body, and call site.

### 1A. Renderer.cpp — Remove function bodies (delete entirely)

| Function | Approximate location | Action |
|----------|---------------------|--------|
| `Renderer::InitML_ONNX_DML(...)` | Near bottom, ~2750 lines in | **Delete entire function body** (from `void Renderer::InitML_ONNX_DML` through its closing `}`) |
| `Renderer::CreateMLBuffers()` | Right after `InitML_ONNX_DML` | **Delete entire function body** |
| `Renderer::RunMLPass()` | Right after `CreateMLBuffers` | **Delete entire function body** |

### 1B. Renderer.cpp — Remove call sites

**In `LoadPipeline()`**, find and **delete** this line:
```cpp
InitML_ONNX_DML(L"./models/denoiser_model_fp16.onnx");
```
It sits between the swap chain descriptor setup and the command queue creation — specifically right after:
```cpp
ThrowIfFailed(
    m_device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&m_commandQueue)));
```

**In `OnInit()`**, find and **delete** this line:
```cpp
CreateMLBuffers();
```
It sits in the big initialization block, between `CreateRaytracingOutputBuffer()` and `CreateReadbackBuffer()`.

**In `PopulateCommandList()`**, inside the pass execution `switch`, the `case Stage::ML:` block:
```cpp
case Stage::ML:
{
    if (m_enableML) {
        RunMLPass();
    }
    break;
}
```
**Delete the entire case.** (You can leave `case Stage::ML: break;` as a stub if your `Stage` enum still has it, or remove `ML` from the enum too.)

**In `CreateRaytracingPipeline()`**, find and **delete** this guard:
```cpp
if (p.stage == Stage::ML) continue;
```

**In `CreateShaderBindingTable()`**, find and **delete** this guard:
```cpp
if (entry == L"ml") continue;
```

**In the pass sequence string** (`m_passSequence` in the constructor), **remove** the `L"ml"` entry if it exists. (It's not in the current sequence shown, but check.)

### 1C. Renderer.h — Remove member declarations

Delete all of these member variables (search for each name):

```cpp
// ONNX Runtime
std::unique_ptr<Ort::Env>           m_ortEnv;
Ort::SessionOptions                 m_ortSessionOptions;
std::unique_ptr<Ort::Session>       m_ortSession;
std::string                         m_mlInputNameStr;
std::string                         m_mlOutputNameStr;
const char*                         m_mlInputName = nullptr;
const char*                         m_mlOutputName = nullptr;

// DirectML
ComPtr<IDMLDevice>                  m_dmlDevice;

// ML Buffers
ComPtr<ID3D12Resource>              m_mlReadbackBuffer;
ComPtr<ID3D12Resource>              m_mlUploadBuffer;
D3D12_PLACED_SUBRESOURCE_FOOTPRINT  m_mlFootprint{};
UINT64                              m_mlSliceBytes = 0;
UINT64                              m_mlSliceBytesAligned = 0;
UINT                                m_mlRowPitch = 0;
std::vector<UINT>                   m_mlInputSlices;
UINT                                m_mlOutputSlice = 0;
std::vector<int64_t>                m_mlInputShape;
std::vector<int64_t>                m_mlOutputShape;
size_t                              m_mlInputElems = 0;
size_t                              m_mlOutputElems = 0;
std::vector<uint16_t>               m_mlInputCPU;
std::vector<uint16_t>               m_mlOutputCPU;
bool                                m_enableML = true; // or false
```

Delete these member function declarations:
```cpp
void InitML_ONNX_DML(const wchar_t* onnxPath);
void CreateMLBuffers();
void RunMLPass();
```

### 1D. Remove from pass sequence and pass parser

In the constructor's `m_passSequence`, if there's an `L"ml"` string entry → delete it.

In your `ParsePass()` function (not shown but referenced), remove the branch that handles `"ml"` strings.

In the `Stage` enum, remove `ML` if you want to be thorough, or just leave it unused.

---

## 2. REMOVE ALIAS TABLE CODE

**Keep:** `m_emissiveTriangles`, `CollectEmissiveTriangles()`, `CreateEmissiveTrianglesBuffer()`, `CreateTriToLightIdBuffer()`, and all LightTree code.
**Remove:** Alias table construction, buffers, and heap slots.

### 2A. Renderer.cpp — Delete function bodies

| Function | Action |
|----------|--------|
| `Renderer::BuildAliasTableSoA(...)` | **Delete entire function** |
| `Renderer::CreateAliasBuffers()` | **Delete entire function** |

### 2B. Renderer.cpp — Remove call sites

**In `CreateAccelerationStructures()`**, find and **delete** these two lines:
```cpp
{SCOPE_TIMER("BuildAliasTableSoA");BuildAliasTableSoA(m_emissiveTriangles);}
{SCOPE_TIMER("CreateAliasBuffers");CreateAliasBuffers();}
```

### 2C. Renderer.cpp — Remove from descriptor heap (`CreateShaderResourceHeap`)

Find these two blocks and **delete** them:

**RANGE 16: SRV t7 (Alias Prob)** — the entire `if (m_aliasProbBuffer) { ... } else { createNullSRV(); } nextSlot();` block at slot 16.

**RANGE 17: SRV t8 (Alias Idx)** — the entire `if (m_aliasIdxBuffer) { ... } else { createNullSRV(); } nextSlot();` block at slot 17.

**IMPORTANT:** Since every slot after 16/17 is positionally dependent, you have two choices:
- **(Recommended)** Replace both blocks with `createNullSRV(); nextSlot();` to keep slot numbering intact and avoid re-numbering everything downstream. This is a no-op placeholder.
- **(Thorough)** Remove them and re-number all subsequent slots and corresponding shader register bindings — this is error-prone.

### 2D. Renderer.cpp — Remove from root signatures

In both `CreateRayGenSignature()` and `CreateComputeSignature()`, the ranges for t7 and t8 are:
```cpp
// Slot 16: SRV t7 (Alias Prob)
ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 7, 0, STATIC, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);

// Slot 17: SRV t8 (Alias Idx)
ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 8, 0, STATIC, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
```
**Same recommendation:** Either keep as-is (shader just won't read them) or remove and re-number. Keeping is safer.

### 2E. Renderer.h — Remove member declarations

```cpp
std::vector<float>       m_aliasProb;
std::vector<uint32_t>    m_aliasIdx;
ComPtr<ID3D12Resource>   m_aliasProbBuffer;
ComPtr<ID3D12Resource>   m_aliasIdxBuffer;
```

Delete function declarations:
```cpp
void BuildAliasTableSoA(const std::vector<LightTriangle>& tris);
void CreateAliasBuffers();
```

### 2F. HLSL side

In your shaders, remove any reads from `t7` (alias prob) and `t8` (alias idx), and any alias-table sampling logic. Replace light sampling with pure light-tree traversal.

---

## 3. REMOVE NON-BINDLESS TEXTURE CODE

The old path created `Texture2DArray` resources (`m_albedoTextureArray`, `m_normalTextureArray`, `m_rmaTextureArray`) via `CreateTextureArrays()`. The new bindless path uses `CreateBindlessTextures()` which creates individual `Texture2D` SRVs in the heap. The old arrays are already unused for albedo/normal/RMA (slots 28-30 are filled with `createNullTex2D()`) but the code and members still exist.

### 3A. Renderer.cpp — Delete function body

| Function | Action |
|----------|--------|
| `Renderer::CreateTextureArrays(...)` | **Delete the entire function** (~100 lines). It starts with `void Renderer::CreateTextureArrays(` and creates `m_albedoTextureArray`, `m_normalTextureArray`, `m_rmaTextureArray` |

### 3B. Renderer.cpp — Remove from `CreateShaderResourceHeap`

In `CreateShaderResourceHeap()`, the old texture array SRV slots (28-30) are already stubbed out with:
```cpp
{
    auto createNullTex2D = [&]() {
        D3D12_SHADER_RESOURCE_VIEW_DESC desc = {};
        desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        desc.Texture2D.MipLevels = 1;
        m_device->CreateShaderResourceView(nullptr, &desc, handle);
        nextSlot();
    };
    createNullTex2D(); // slot 28 (was albedo array)
    createNullTex2D(); // slot 29 (was normal array)
    createNullTex2D(); // slot 30 (was RMA array)
}
```
**Keep these null stubs** (they maintain heap layout). Or if your shaders no longer reference t30-t32 for arrays, you can eventually reclaim these slots.

### 3C. Renderer.cpp — Remove from root signatures

In both `CreateRayGenSignature()` and `CreateComputeSignature()`, the ranges at slots 28-31:
```cpp
// Slot 28-31: Texture Arrays (t30-t33)
ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 30, 0, STATIC, ...);
ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 31, 0, STATIC, ...);
ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 32, 0, STATIC, ...);
ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 33, 0, STATIC, ...);
```
Slots 28-30 were for the old arrays. Slot 31 is the LUT array (keep it). You can either:
- **Leave all four** (safe, LUT at t33 still works), or
- Remove the three dead ranges for t30-t32 and re-number (risky).

Since your shaders now use `ResourceDescriptorHeap[texID]` for bindless, the old array ranges are dead code. Safest: leave them and let the null SRVs fill them.

### 3D. Renderer.h — Remove member declarations

```cpp
ComPtr<ID3D12Resource>  m_albedoTextureArray;
ComPtr<ID3D12Resource>  m_normalTextureArray;
ComPtr<ID3D12Resource>  m_rmaTextureArray;
std::vector<ComPtr<ID3D12Resource>> m_textureUploadHeaps; // already cleared after init, but remove the member
```

Delete the function declaration:
```cpp
void CreateTextureArrays(
    const std::vector<TextureData>& albedoTextures,
    const std::vector<TextureData>& normalTextures,
    const std::vector<TextureData>& rmaTextures);
```

### 3E. Verify: `m_textureUploadHeaps` cleanup

In `OnInit()`, this line already exists:
```cpp
m_textureUploadHeaps.clear();
```
Once you remove the member entirely, delete this line too.

---

## 4. INSTANCE ARCHITECTURE OVERHAUL

### Current Problem

Right now, your loader (`loadGlbFile`) flattens an entire glTF scene into a single mesh:
- All mesh primitives from all nodes become one giant vertex/index buffer
- The `m_instances` vector has one entry per *file loaded*, not per *glTF node*
- Node transforms from the glTF scene graph are baked into vertex positions during load
- This means you can't independently move/animate individual objects from a glTF file

### Target Architecture

```
Scene
 └─ Instance[] — one per visible mesh-node in the scene
       ├── BLAS index (into m_blasList[])
       ├── Transform (XMMATRIX, updated per-frame)
       ├── Geometry offsets (into global vertex/index buffers)
       └── Material base offset

BLAS[] — one per unique mesh-primitive combo (shared via instancing)
       ├── pResult (compacted acceleration structure)
       ├── vertex range in global VB
       └── index range in global IB
```

### 4A. New Data Structures (add to Renderer.h)

```cpp
// Represents a unique piece of geometry (one BLAS)
struct MeshGPU {
    ComPtr<ID3D12Resource>  blas;           // compacted BLAS
    UINT                    vertexBase;     // offset into global VB
    UINT                    indexBase;       // offset into global IB
    UINT                    vertexCount;
    UINT                    indexCount;
    UINT                    materialBase;   // offset into m_materialIDs
    UINT                    opaqueTriCount;
    UINT                    alphaTriCount;
};

// Represents one visible object in the scene
struct SceneInstance {
    UINT        meshIndex;      // index into m_meshes[]
    XMMATRIX    transform;      // current object-to-world
    XMMATRIX    prevTransform;  // previous frame (for motion vectors)
    // Add any per-instance data: visibility flags, animation state, etc.
};
```

Replace or augment existing members:
```cpp
// OLD (remove these):
std::vector<ComPtr<ID3D12Resource>>  m_VB;
std::vector<ComPtr<ID3D12Resource>>  m_IB;
std::vector<D3D12_VERTEX_BUFFER_VIEW> m_VBView;
std::vector<D3D12_INDEX_BUFFER_VIEW>  m_IBView;
std::vector<UINT>                    m_VertexCount;
std::vector<UINT>                    m_IndexCount;
std::vector<UINT>                    m_opaqueTriCount;
std::vector<UINT>                    m_alphaTriCount;
std::vector<UINT>                    m_materialIDOffsets;
std::vector<UINT>                    m_instanceModelIndices;
std::vector<std::pair<ComPtr<ID3D12Resource>, XMMATRIX>> m_instances;
std::vector<std::vector<Vertex>>     m_cpuVertexData;
std::vector<std::vector<UINT>>       m_cpuIndexData;

// NEW (add these):
std::vector<MeshGPU>                m_meshes;         // unique geometries
std::vector<SceneInstance>          m_sceneInstances;  // all visible objects
```

### 4B. Changes to ObjLoader.h — Return structured data

Instead of pushing directly into renderer vectors, the loaders should return a clean intermediate structure. **Add this struct** above the `ObjLoader` class:

```cpp
struct LoadedMesh {
    std::vector<Vertex>   vertices;
    std::vector<UINT>     indices;
    std::vector<UINT>     perTriMaterialIDs;
    UINT                  opaqueTriCount;
    UINT                  alphaTriCount;
};

struct LoadedScene {
    std::vector<LoadedMesh>  meshes;      // unique geometries
    std::vector<Material>    materials;
    // Per-instance: which mesh + what transform
    std::vector<std::pair<UINT, XMMATRIX>> instances; // (meshIndex, transform)
};
```

### 4C. Rewrite `loadGlbFile` to populate `LoadedScene`

The key change is in the geometry loop. Currently you have:

```cpp
// CURRENT: flattens everything into one mesh
for (const auto& [meshIdx, worldTransform] : meshInstances) {
    // ...bakes worldTransform into vertex positions...
    // ...pushes all triangles into one shared vertices/indices...
}
```

**Change to:**

```cpp
// NEW: one LoadedMesh per unique glTF mesh, instances reference them
static LoadedScene loadGlbFile(...) {
    LoadedScene scene;
    
    // ... parse glTF as before ...
    
    // 1. Build one LoadedMesh per glTF mesh (not per instance)
    //    Do NOT apply node transforms to vertices — keep them in local space
    std::unordered_map<int, UINT> meshToLoadedIdx;
    
    for (uint32_t mi = 0; mi < model.meshes_count; ++mi) {
        LoadedMesh lm;
        // ... extract vertices in LOCAL space (no worldTransform applied) ...
        // ... extract indices ...
        // ... run SplitOpaqueAlpha ...
        meshToLoadedIdx[mi] = (UINT)scene.meshes.size();
        scene.meshes.push_back(std::move(lm));
    }
    
    // 2. Collect instances from scene graph
    //    Each node that references a mesh becomes an instance
    for (const auto& [meshIdx, worldTransform] : meshInstances) {
        UINT loadedIdx = meshToLoadedIdx[meshIdx];
        scene.instances.push_back({ loadedIdx, worldTransform });
    }
    
    return scene;
}
```

**Critical:** Do NOT call `XMVector3TransformCoord(pos, worldTransform)` on vertices anymore. Keep them in mesh-local space. The transform goes into the TLAS instance descriptor.

### 4D. Rewrite `LoadAssets()` in Renderer.cpp

```cpp
void Renderer::LoadAssets() {
    // ... command list creation as before ...
    
    // 1. Load scenes
    LoadedScene scene = ObjLoader::loadGlbFile("./pavillion.glb", ...);
    
    // 2. Copy materials
    UINT materialBaseOffset = (UINT)m_materials.size();
    m_materials.insert(m_materials.end(), 
                       scene.materials.begin(), scene.materials.end());
    
    // 3. Create one MeshGPU per unique LoadedMesh
    for (auto& lm : scene.meshes) {
        MeshGPU gpu;
        // Create VB, IB (upload heap for now, or accumulate for global buffer)
        gpu.vertexCount    = (UINT)lm.vertices.size();
        gpu.indexCount     = (UINT)lm.indices.size();
        gpu.opaqueTriCount = lm.opaqueTriCount;
        gpu.alphaTriCount  = lm.alphaTriCount;
        // ... create D3D12 buffers, upload ...
        // Store cpu data for BuildGlobalMeshBuffers
        m_meshes.push_back(std::move(gpu));
    }
    
    // 4. Create instances — multiple instances can share the same mesh
    for (auto& [meshIdx, transform] : scene.instances) {
        SceneInstance inst;
        inst.meshIndex     = meshIdx;  // + offset if loading multiple files
        inst.transform     = transform;
        inst.prevTransform = transform;
        m_sceneInstances.push_back(inst);
    }
    
    // ... rest of LoadAssets (textures, fence, etc.) ...
}
```

### 4E. Rewrite `CreateAccelerationStructures()`

```cpp
void Renderer::CreateAccelerationStructures() {
    // 1. Build one BLAS per unique MeshGPU
    for (size_t i = 0; i < m_meshes.size(); ++i) {
        auto& mesh = m_meshes[i];
        auto buffers = CreateBottomLevelAS(
            {{ mesh.vb.Get(), mesh.vertexCount }},
            {{ mesh.ib.Get(), mesh.indexCount }},
            mesh.opaqueTriCount, mesh.alphaTriCount);
        mesh.blas = buffers.pResult;
    }
    
    // 2. Build TLAS with one instance per SceneInstance
    //    Multiple instances can point to the same BLAS
    m_tlasInstances.clear();
    for (auto& inst : m_sceneInstances) {
        m_tlasInstances.emplace_back(
            m_meshes[inst.meshIndex].blas,
            inst.transform);
    }
    CreateTopLevelAS(m_tlasInstances);
    
    // ... light collection, etc. ...
}
```

### 4F. Update `BuildGlobalMeshBuffers()`

Currently iterates `m_VB.size()`. Change to iterate `m_meshes.size()`:

```cpp
void Renderer::BuildGlobalMeshBuffers() {
    m_geoOffsets.clear();
    m_geoOffsets.resize(m_meshes.size());
    size_t totalVerts = 0, totalIdx = 0;
    
    for (size_t m = 0; m < m_meshes.size(); ++m) {
        m_geoOffsets[m].vertexBase  = (UINT)totalVerts;
        m_geoOffsets[m].indexBase   = (UINT)totalIdx;
        m_geoOffsets[m].materialBase = m_meshes[m].materialBase;
        totalVerts += m_meshes[m].vertexCount;
        totalIdx   += m_meshes[m].indexCount;
    }
    // ... rest is same pattern, iterate m_meshes instead of m_VB ...
}
```

### 4G. Update `UpdateInstancePropertiesBuffer()`

Change from `m_instances.size()` to `m_sceneInstances.size()`:

```cpp
void Renderer::UpdateInstancePropertiesBuffer() {
    InstanceProperties* dst = nullptr;
    CD3DX12_RANGE r(0, 0);
    ThrowIfFailed(m_instanceProperties->Map(0, &r, (void**)&dst));
    
    for (size_t i = 0; i < m_sceneInstances.size(); ++i, ++dst) {
        auto& inst = m_sceneInstances[i];
        const XMMATRIX& M = inst.transform;
        XMVECTOR det;
        
        dst->prevObjectToWorld = ...;  // from inst.prevTransform
        dst->objectToWorld     = M;
        dst->objectToWorldInverse = XMMatrixInverse(&det, M);
        
        // Geometry offsets from the mesh this instance references
        auto& mesh = m_meshes[inst.meshIndex];
        auto& go   = m_geoOffsets[inst.meshIndex];
        dst->indexBase       = go.indexBase;
        dst->vertexBase      = go.vertexBase;
        dst->materialBase    = go.materialBase;
        dst->opaqueTriCount  = mesh.opaqueTriCount;
        dst->triToLightBase  = m_instTriOffset[i];
        
        // Save for next frame
        inst.prevTransform = inst.transform;
    }
    m_instanceProperties->Unmap(0, nullptr);
}
```

### 4H. Update `CollectEmissiveTriangles()`

Change from `m_instances.size()` / `m_instanceModelIndices` to `m_sceneInstances` / `inst.meshIndex`:

```cpp
void Renderer::CollectEmissiveTriangles() {
    m_emissiveTriangles.clear();
    m_instTriOffset.resize(m_sceneInstances.size());
    
    size_t totalTris = 0;
    for (size_t i = 0; i < m_sceneInstances.size(); ++i) {
        auto& mesh = m_meshes[m_sceneInstances[i].meshIndex];
        totalTris += mesh.indexCount / 3;
    }
    m_triToLightId.assign(totalTris, 0xFFFFFFFFu);
    
    uint32_t runningBase = 0;
    for (size_t i = 0; i < m_sceneInstances.size(); ++i) {
        m_instTriOffset[i] = runningBase;
        auto& mesh = m_meshes[m_sceneInstances[i].meshIndex];
        runningBase += mesh.indexCount / 3;
    }
    
    // ... rest uses m_sceneInstances[i].meshIndex instead of
    //     m_instanceModelIndices[i], and m_sceneInstances[i].transform
    //     instead of m_instances[i].second ...
}
```

### 4I. Update `CreateTopLevelAS()`

The signature changes. Instead of:
```cpp
std::vector<std::pair<ComPtr<ID3D12Resource>, XMMATRIX>> m_instances;
```
Use the new `m_tlasInstances` vector (same type, but now built from `m_sceneInstances`).

### 4J. Key principle: BLAS sharing

If a glTF file has 50 instances of the same lamp mesh, you should have:
- **1 BLAS** (the lamp geometry)
- **50 TLAS instance descriptors** (each with a different transform)
- **50 entries** in `m_sceneInstances`

The TLAS instance's `InstanceContributionToHitGroupIndex` should be `2 * instanceIdx` (as you already do), and the hit shader uses `InstanceID()` to look up the correct `InstanceProperties` which in turn gives the correct geometry offsets.

### 4K. Summary of member variable changes in Renderer.h

**Remove:**
```
m_VB, m_IB, m_VBView, m_IBView, m_VertexCount, m_IndexCount
m_opaqueTriCount, m_alphaTriCount  (per-file vectors)
m_materialIDOffsets  (folded into MeshGPU)
m_instanceModelIndices
m_instances  (the old pair<Resource,Matrix> vector)
m_cpuVertexData, m_cpuIndexData  (move into loading, don't store permanently)
```

**Add:**
```
std::vector<MeshGPU>        m_meshes;
std::vector<SceneInstance>  m_sceneInstances;
// Keep m_tlasInstances as a rebuild-time vector (same type as old m_instances)
std::vector<std::pair<ComPtr<ID3D12Resource>, XMMATRIX>> m_tlasInstances;
```

---

## EXECUTION ORDER

1. **ML removal** — cleanest, no dependencies on other changes
2. **Alias table removal** — also independent
3. **Non-bindless texture removal** — independent
4. **Instance architecture** — do this last, as it touches the most systems

For step 4, recommend a phased approach:
- **Phase A:** Add new structs, rewrite loaders to return `LoadedScene`, update `LoadAssets()` to populate `m_meshes` / `m_sceneInstances` while also filling the *old* vectors for compatibility. Verify nothing breaks.
- **Phase B:** Update `BuildGlobalMeshBuffers`, `CreateAccelerationStructures`, `UpdateInstancePropertiesBuffer` to use new structs. Delete old vectors.
- **Phase C:** Update `CollectEmissiveTriangles` and light tree to use new structs.
- **Phase D:** Remove all references to old vectors from the header.

---

## FILES AFFECTED

| File | Changes |
|------|---------|
| `Renderer.cpp` | Remove ML functions, alias functions, texture array function; rewrite LoadAssets, CreateAccelerationStructures, BuildGlobalMeshBuffers, UpdateInstancePropertiesBuffer, CollectEmissiveTriangles, CreateShaderResourceHeap |
| `Renderer.h` | Remove ~30 member variables, add MeshGPU/SceneInstance structs, add new vectors |
| `ObjLoader.h` | Add LoadedMesh/LoadedScene structs, rewrite loadGlbFile to not bake transforms, rewrite loadObjFile similarly |
| `HLSL shaders` | Remove alias table sampling (t7/t8 reads), update any array texture sampling to use bindless |
