//====================================
//SCENE
//====================================

#include "../stdafx.h"
#include <fstream>
#include "Scene.h"
#include "../DXRHelper.h"

// ─────────────────────────────────────────────────────────────────
void Scene::PropagateModelTransforms() {
    for (auto& model : models) {
        for (UINT i = model.instanceStart; i < model.instanceStart + model.instanceCount; ++i) {
            auto& inst = instances[i];
            inst.worldTransform = inst.localTransform * model.worldTransform;
        }
    }
}

void Scene::MarkModelMoved(UINT modelIndex) {
    if (modelIndex >= models.size()) return;
    auto& model = models[modelIndex];
    model.RebuildTransform();

    // Update only this model's instances
    for (UINT i = model.instanceStart; i < model.instanceStart + model.instanceCount; ++i) {
        instances[i].worldTransform = instances[i].localTransform * model.worldTransform;
        MarkInstanceDirty(i);
    }

    tlasDirty      = true;
    lightTreeDirty = true;
}

void Scene::MarkMaterialsDirty(bool emissionChanged) {
    materialsDirty = true;
    if (emissionChanged) {
        lightTreeDirty = true;
        emissivesDirty = true;  // need to recompute BLAS local roots
    }
}

void Scene::MarkInstanceDirty(UINT instanceIndex) {
    if (instanceIndex < instanceDirty.size())
        instanceDirty[instanceIndex] = 1;
}

void Scene::MarkAllInstancesDirty() {
    std::fill(instanceDirty.begin(), instanceDirty.end(), 1);
}

// ─────────────────────────────────────────────────────────────────
void Scene::ReserveTerrain(UINT vertexElems, UINT indexElems, UINT matIDElems,
                           UINT triLightElems, UINT instanceSlots, UINT propsBase) {
    terrainVertexElems   = vertexElems;
    terrainIndexElems    = indexElems;
    terrainMatIDElems    = matIDElems;
    terrainTriLightElems = triLightElems;
    terrainInstanceSlots = instanceSlots;
    terrainPropsBase     = propsBase;

    //Append one flat terrain material (mirrors TERRAIN_ALBEDO / TERRAIN_ROUGHNESS
    //in Constants_v8.hlsli): opaque, no textures, no emission. Every terrain
    //triangle's materialID points here, so terrain decodes through the normal
    //material path with no shader branch.
    Material tm;
    tm.Kd             = XMFLOAT4(0.42f, 0.36f, 0.30f, 1.0f);   // w=1 -> opaque
    tm.Ke             = XMFLOAT3(0.0f, 0.0f, 0.0f);
    tm.Ni             = 1.0f;
    tm.Pr_Pm_Ps_Pc    = XMFLOAT4(1.0f, 0.0f, 0.0f, 0.0f);      // roughness=1 (fully diffuse), metalness=0
    tm.albedoTexID    = -1;
    tm.normalTexID    = -1;
    tm.rmaTexID       = -1;
    tm.alphaThreshold = 1.0f;
    materials.push_back(tm);
    terrainMatIndex = (UINT)materials.size() - 1;
}

void Scene::ReserveRocks(UINT instanceSlots) {
    //Rocks live in a disjoint instanceProps range right after the terrain
    //region; terrain occupies [terrainPropsBase, +terrainInstanceSlots).
    rockInstanceSlots = instanceSlots;
    rockPropsBase     = terrainPropsBase + terrainInstanceSlots;
}

// ─────────────────────────────────────────────────────────────────
void Scene::BuildGlobalMeshBuffers(ID3D12Device* device, ID3D12GraphicsCommandList* cmdList) {
    SCOPE_TIMER("BuildGlobalMeshBuffers");

    geoOffsets.resize(meshes.size());
    size_t totalV = 0, totalI = 0;

    for (size_t m = 0; m < meshes.size(); ++m) {
        meshes[m].globalVertexBase = (UINT)totalV;
        meshes[m].globalIndexBase  = (UINT)totalI;
        geoOffsets[m] = { (UINT)totalV, (UINT)totalI, meshes[m].materialIDBase };
        totalV += meshes[m].vertexCount;
        totalI += meshes[m].indexCount;
    }
    totalVertexCount = (UINT)totalV;
    totalIndexCount  = (UINT)totalI;

    //PLANET: the terrain region follows the scene data in the same buffers.
    terrainVertexBase = totalVertexCount;
    terrainIndexBase  = totalIndexCount;

    const uint64_t vbBytes = (uint64_t)combinedVertexCount() * sizeof(BTriVertex);
    const uint64_t ibBytes = (uint64_t)combinedIndexCount()  * sizeof(uint32_t);

    //A reserved terrain region means the planet tessellator writes chunk meshes
    //IN PLACE at runtime, which needs a persistently-mapped UPLOAD buffer. With
    //NO terrain (planet off) the geometry is static — so put it in a DEFAULT
    //(VRAM) heap: EvalSurfaceState fetches ~9 vertex attributes PER HIT, and from
    //an UPLOAD (system-RAM) buffer those stream over PCIe and dominate throughput
    //(camera AND bounce). One-time staging copy, mirroring CreateTriToLightIdBuffer.
    const bool hasTerrain = combinedVertexCount() > totalVertexCount;

    uint8_t* dstVertsRaw;
    uint8_t* dstIdxRaw;
    if (hasTerrain)
    {
        //UPLOAD + persistent map: tessellator writes the terrain region in place.
        vertexGlobal = nv_helpers_dx12::CreateBuffer(device, vbBytes,
            D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ,
            nv_helpers_dx12::kUploadHeapProps);
        vertexGlobal->SetName(L"GlobalVertexBuffer");
        indexGlobal = nv_helpers_dx12::CreateBuffer(device, ibBytes,
            D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ,
            nv_helpers_dx12::kUploadHeapProps);
        indexGlobal->SetName(L"GlobalIndexBuffer");

        //persistent map (never unmapped; the planet writes into the terrain region).
        CD3DX12_RANGE noRead(0, 0);
        vertexGlobal->Map(0, &noRead, (void**)&vertexGlobalMapped);
        indexGlobal->Map (0, &noRead, (void**)&indexGlobalMapped);
        dstVertsRaw = vertexGlobalMapped;
        dstIdxRaw   = indexGlobalMapped;
    }
    else
    {
        //DEFAULT (VRAM): static scene geometry, shader-read every hit — keep it
        //off the PCIe bus. Build into UPLOAD staging (held as members until the
        //caller's FlushAndReset runs the copy), copy to DEFAULT, barrier to read.
        vertexGlobal = nv_helpers_dx12::CreateBuffer(device, vbBytes,
            D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST,
            nv_helpers_dx12::kDefaultHeapProps);
        vertexGlobal->SetName(L"GlobalVertexBuffer");
        indexGlobal = nv_helpers_dx12::CreateBuffer(device, ibBytes,
            D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST,
            nv_helpers_dx12::kDefaultHeapProps);
        indexGlobal->SetName(L"GlobalIndexBuffer");

        vertexGlobalUpload = nv_helpers_dx12::CreateBuffer(device, vbBytes,
            D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ,
            nv_helpers_dx12::kUploadHeapProps);
        indexGlobalUpload = nv_helpers_dx12::CreateBuffer(device, ibBytes,
            D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ,
            nv_helpers_dx12::kUploadHeapProps);

        CD3DX12_RANGE noRead(0, 0);
        vertexGlobalUpload->Map(0, &noRead, (void**)&dstVertsRaw);
        indexGlobalUpload->Map (0, &noRead, (void**)&dstIdxRaw);
        vertexGlobalMapped = nullptr;   // DEFAULT buffer has no CPU mapping
        indexGlobalMapped  = nullptr;
    }

    auto* dstVerts = reinterpret_cast<BTriVertex*>(dstVertsRaw);
    auto* dstIdx   = reinterpret_cast<uint32_t*>(dstIdxRaw);

    for (size_t m = 0; m < meshes.size(); ++m) {
        const auto& mesh = meshes[m];
        const UINT vBase = mesh.globalVertexBase;

        BTriVertex* outV = dstVerts + vBase;
        for (UINT i = 0; i < mesh.vertexCount; ++i) {
            const Vertex& sv = mesh.cpuVertices[i];
            outV[i].vertex = sv.position;
            XMVECTOR normal = XMVector3Normalize(
                XMVectorSet(sv.normal_material.x, sv.normal_material.y,
                            sv.normal_material.z, 0.0f));
            outV[i].packedNormal = EncodeNormalOct(normal);
            PackedVector::XMStoreHalf2(&outV[i].texCoord, XMLoadFloat2(&sv.texCoord));
        }

        uint32_t* outI = dstIdx + mesh.globalIndexBase;
        for (UINT i = 0; i < mesh.indexCount; ++i)
            outI[i] = mesh.cpuIndices[i] + vBase;
    }

    if (!hasTerrain)
    {
        //finish the DEFAULT-heap upload: staging -> VRAM, then barrier to read.
        //The caller (InitSceneGPU) runs FlushAndReset right after and then frees
        //the staging via vertexGlobalUpload / indexGlobalUpload .Reset().
        vertexGlobalUpload->Unmap(0, nullptr);
        indexGlobalUpload->Unmap(0, nullptr);
        cmdList->CopyBufferRegion(vertexGlobal.Get(), 0, vertexGlobalUpload.Get(), 0, vbBytes);
        cmdList->CopyBufferRegion(indexGlobal.Get(),  0, indexGlobalUpload.Get(),  0, ibBytes);
        const D3D12_RESOURCE_BARRIER toRead[2] = {
            CD3DX12_RESOURCE_BARRIER::Transition(vertexGlobal.Get(),
                D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_GENERIC_READ),
            CD3DX12_RESOURCE_BARRIER::Transition(indexGlobal.Get(),
                D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_GENERIC_READ),
        };
        cmdList->ResourceBarrier(2, toRead);
    }

    //terrain region left uninitialized: only LIVE cells (with a built BLAS and
    //valid instanceProps) are referenced by the TLAS, and those slots are
    //always written by the tessellator before the cell goes live.
}

// ─────────────────────────────────────────────────────────────────
void Scene::CreateInstancePropertiesBuffer(ID3D12Device* device) {
    //PLANET: with terrain enabled the count is FIXED (terrainPropsBase +
    //terrainInstanceSlots) - scene instances occupy [0, terrainPropsBase),
    //terrain occupies [terrainPropsBase, +terrainInstanceSlots). Because the
    //size is fixed, this buffer is created ONCE and reused across scene
    //structural changes; that is what keeps the orchestrator's bound pointer
    //(StreamOrchestrator::m_instanceProps) valid - reallocating it would dangle
    //that pointer and leave terrain writing freed memory.
    const uint32_t count = instancePropsCount();
    const uint32_t size  = ROUND_UP(
        count * sizeof(InstanceProperties),
        D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);

    //idempotent: keep the existing buffer if it already covers `size`. (Terrain
    //mode: always true after the first create. No-terrain mode: grows as the
    //scene instance count grows, matching the original behaviour.)
    if (instanceProperties) {
        const D3D12_RESOURCE_DESC d = instanceProperties->GetDesc();
        if (d.Width >= size) return;
    }
    instanceProperties = nv_helpers_dx12::CreateBuffer(
        device, size, D3D12_RESOURCE_FLAG_NONE,
        D3D12_RESOURCE_STATE_GENERIC_READ, nv_helpers_dx12::kUploadHeapProps);
}

void Scene::PrepareInstanceProperties() {
    const size_t count = instances.size();

    // Ensure shadow buffer and dirty flags are sized correctly
    cpuInstanceProps.resize(count);
    instanceDirty.resize(count, 1);       // default dirty on first frame
    instanceInitialized.resize(count, 0); // new entries are uninitialized

    // Build list of dirty indices and update only those
    dirtyInstanceList.clear();
    for (size_t i = 0; i < count; ++i) {
        if (!instanceDirty[i]) continue;
        dirtyInstanceList.push_back(static_cast<uint32_t>(i));
    }

    //Floating origin shift. The source of truth si.worldTransform stays in
    //absolute world coords; everything that lands in cpuInstanceProps /
    //tlasInstances is shifted by -sceneOriginWorld so the GPU sees a
    //camera local frame. Shift only touches the translation row; rotation
    /// scale (and hence normals) are unaffected.
    const XMVECTOR shift = XMVectorSet(sceneOriginWorld.x,
                                        sceneOriginWorld.y,
                                        sceneOriginWorld.z, 0.0f);

    //Origin delta since the last call. When non-zero (floating-origin snap
    //or camera reset), every existing instance's prevObjectToWorld needs the
    //same delta applied so it lives in the SAME shifted frame as prevView
    //(which Camera::PollSceneOrigin / Camera::ResetView already adjusted).
    //Without this, prevWorldPos = prevObjectToWorld * localPos lands in the
    //OLD shifted frame, prevView projects it as if it were in the NEW
    //frame, and the resulting MV is off by exactly shiftDelta. DLSS reads
    //that as a giant per pixel jump and rejects all temporal history; the
    //ReSTIR temporal pass reprojects into garbage pixels.
    const XMVECTOR prevShift = XMVectorSet(prevSceneOriginWorld.x,
                                            prevSceneOriginWorld.y,
                                            prevSceneOriginWorld.z, 0.0f);
    const XMVECTOR originDelta  = XMVectorSubtract(shift, prevShift);
    const bool     originShifted = XMVector3LengthSq(originDelta).m128_f32[0] > 0.0f;

    // Only process dirty instances — reads/writes go to CPU shadow buffer (fast),
    // NOT the GPU upload heap (write-combined memory where reads are 10-100x slower)
    for (uint32_t idx : dirtyInstanceList) {
        auto& dst = cpuInstanceProps[idx];
        auto& si  = instances[idx];
        XMMATRIX M = si.worldTransform;
        //subtract origin from translation row (row index 3 in DirectXMath
        //row-vector convention). XMVectorSubtract zeroes only xyz because
        //shift.w == 0, preserving the .w = 1 of the translation row.
        M.r[3] = XMVectorSubtract(M.r[3], shift);
        XMVECTOR det;

        bool isNew = !instanceInitialized[idx];

        // Compute new current values (only 1 inverse + 1 normal inverse)
        if (!isNew) {
            // Shift current → prev (reuse cached inverses — no extra inversions)
            dst.prevObjectToWorld        = dst.objectToWorld;
            dst.prevObjectToWorldInverse = dst.objectToWorldInverse;
            dst.prevObjectToWorldNormal  = dst.objectToWorldNormal;

            //Re-express prev in the NEW shifted frame after a floating-origin
            //snap. Upper 3x3 (rotation/scale, and hence the normal matrix)
            //is unaffected — only the translation row picks up -delta. The
            //inverse has to be recomputed since changing the translation row
            //of a 4x4 doesn't translate cleanly through XMMatrixInverse on
            //the cached value; cost is one mat-inverse per existing instance
            //per snap, which only happens when the origin actually moves.
            if (originShifted) {
                dst.prevObjectToWorld.r[3]   = XMVectorSubtract(dst.prevObjectToWorld.r[3], originDelta);
                dst.prevObjectToWorldInverse = XMMatrixInverse(&det, dst.prevObjectToWorld);
            }
        }

        dst.objectToWorld        = M;
        dst.objectToWorldInverse = XMMatrixInverse(&det, M);

        XMMATRIX upper3x3 = M;
        upper3x3.r[0].m128_f32[3] = upper3x3.r[1].m128_f32[3] = upper3x3.r[2].m128_f32[3] = 0.f;
        upper3x3.r[3] = { 0, 0, 0, 1.f };
        dst.objectToWorldNormal = XMMatrixTranspose(XMMatrixInverse(&det, upper3x3));

        if (isNew) {
            // First appearance: prev = current (no motion on first frame)
            dst.prevObjectToWorld        = dst.objectToWorld;
            dst.prevObjectToWorldInverse = dst.objectToWorldInverse;
            dst.prevObjectToWorldNormal  = dst.objectToWorldNormal;
            instanceInitialized[idx] = 1;
        }

        const MeshGPU& mesh = meshes[si.meshIndex];
        dst.opaqueTriCount = mesh.opaqueTriCount;
        dst.indexBase      = mesh.globalIndexBase;
        dst.vertexBase     = mesh.globalVertexBase;
        dst.materialBase   = mesh.materialIDBase;
        dst.triToLightBase = instTriOffset.empty() ? 0 : instTriOffset[idx];

        // Keep TLAS in sync (shifted transform; the TopLevelASGenerator holds
        // a reference to this XMMATRIX so the next refit picks up the shift).
        if (idx < tlasInstances.size())
            tlasInstances[idx].transform = M;
    }

    //Latch the origin used this frame so the next call can detect a snap.
    prevSceneOriginWorld = sceneOriginWorld;
}

void Scene::UploadInstanceProperties() {
    // Write only dirty instances to GPU upload heap, packing the CPU working
    // struct (full 4x4s) into the 224B GPU record (affine 3x4s, hot fields
    // first, only prevObjectToWorld kept of the prev matrices).
    if (!dirtyInstanceList.empty()) {
        uint8_t* gpuDst = nullptr;
        CD3DX12_RANGE readRange(0, 0);
        ThrowIfFailed(instanceProperties->Map(0, &readRange, reinterpret_cast<void**>(&gpuDst)));
        for (uint32_t idx : dirtyInstanceList) {
            const InstancePropertiesCpu& src = cpuInstanceProps[idx];
            InstanceProperties gp;
            gp.objectToWorld        = MakeFloat3x4(src.objectToWorld);
            gp.objectToWorldInverse = MakeFloat3x4(src.objectToWorldInverse);
            gp.objectToWorldNormal  = MakeFloat3x4(src.objectToWorldNormal);
            gp.indexBase      = src.indexBase;
            gp.vertexBase     = src.vertexBase;
            gp.materialBase   = src.materialBase;
            gp.triToLightBase = src.triToLightBase;
            gp.opaqueTriCount = src.opaqueTriCount;
            gp._pad[0] = src._pad[0]; gp._pad[1] = src._pad[1]; gp._pad[2] = src._pad[2];
            gp.prevObjectToWorld = MakeFloat3x4(src.prevObjectToWorld);
            memcpy(gpuDst + idx * sizeof(InstanceProperties), &gp, sizeof(gp));
        }
        instanceProperties->Unmap(0, nullptr);
    }

    // Finalize prev = current and keep dirty for one settle frame so the
    // GPU receives the zeroed-terrain motion (prev == current) on the next upload.
    for (uint32_t idx : dirtyInstanceList) {
        auto& p = cpuInstanceProps[idx];
        bool alreadySettled = (memcmp(&p.prevObjectToWorld, &p.objectToWorld, sizeof(XMMATRIX)) == 0);
        p.prevObjectToWorld        = p.objectToWorld;
        p.prevObjectToWorldInverse = p.objectToWorldInverse;
        p.prevObjectToWorldNormal  = p.objectToWorldNormal;
        // Clear dirty only when prev already equaled current (settle frame uploaded)
        instanceDirty[idx] = alreadySettled ? 0 : 1;
    }
}

// ─────────────────────────────────────────────────────────────────
// PLANET_INTEGRATION: TLAS instance list = triangle mesh instances only, no
// procedural primitives. SBT layout is 2 hit-group entries per instance (i*2).
// The floating-origin shift (sceneOriginWorld) is already applied here. Phase 5
// adds streamed terrain chunks as instances sharing one hit group.
void Scene::RebuildTLASInstanceList() {
    tlasInstances.clear();
    tlasInstances.reserve(instances.size());
    //Floating origin shift: subtract sceneOriginWorld from each transform
    //so the TLAS is built in camera local space. See PrepareInstanceProperties
    //for the per-frame refit equivalent.
    const XMVECTOR shift = XMVectorSet(sceneOriginWorld.x,
                                        sceneOriginWorld.y,
                                        sceneOriginWorld.z, 0.0f);
    for (size_t i = 0; i < instances.size(); ++i) {
        const auto& si   = instances[i];
        const auto& mesh = meshes[si.meshIndex];

        // Two SBT hit-group entries per instance: [opaque, alpha]
        UINT hitGroupContrib = static_cast<UINT>(i) * 2;

        // Fully-opaque instances skip any-hit entirely at the hardware level
        auto flags = (mesh.alphaTriCount == 0)
            ? D3D12_RAYTRACING_INSTANCE_FLAG_FORCE_OPAQUE
            : D3D12_RAYTRACING_INSTANCE_FLAG_NONE;

        XMMATRIX shifted = si.worldTransform;
        shifted.r[3] = XMVectorSubtract(shifted.r[3], shift);

        tlasInstances.push_back({ mesh.blas, shifted, hitGroupContrib, flags });
    }
}

// ─────────────────────────────────────────────────────────────────
void Scene::CollectEmissiveTriangles() {
    emissiveTriangles.clear();
    instTriOffset.resize(instances.size());

    size_t totalTris = 0;
    for (size_t i = 0; i < instances.size(); ++i)
        totalTris += meshes[instances[i].meshIndex].indexCount / 3;

    triToLightId.assign(totalTris, 0xFFFFFFFFu);
    uint32_t base = 0;
    for (size_t i = 0; i < instances.size(); ++i) {
        instTriOffset[i] = base;
        base += meshes[instances[i].meshIndex].indexCount / 3;
    }

    for (size_t inst = 0; inst < instances.size(); ++inst) {
        const auto& si   = instances[inst];
        const auto& mesh = meshes[si.meshIndex];
        const UINT  tris = mesh.indexCount / 3;
        const uint32_t triBase = instTriOffset[inst];

        for (UINT t = 0; t < tris; ++t) {
            UINT mid = mesh.cpuMaterialIDs[t];
            const XMFLOAT3& Ke = materials.Ke[mid];
            if (Ke.x + Ke.y + Ke.z <= 0.0f) continue;

            LightTriangle lt{};
            lt.x          = mesh.cpuVertices[mesh.cpuIndices[3*t+0]].position;
            lt.y          = mesh.cpuVertices[mesh.cpuIndices[3*t+1]].position;
            lt.z          = mesh.cpuVertices[mesh.cpuIndices[3*t+2]].position;
            lt.instanceID = (UINT)inst;
            lt.weight     = ComputeTriangleWeight(lt.x, lt.y, lt.z, Ke, si.worldTransform);
            lt.emission   = Ke;

            triToLightId[triBase + t] = (uint32_t)emissiveTriangles.size();
            emissiveTriangles.push_back(lt);
        }
    }

    if (emissiveTriangles.empty())
        emissiveTriangles.push_back(LightTriangle{});

    LOG(L"Emissive Triangles: " << emissiveTriangles.size());
}

// ─────────────────────────────────────────────────────────────────
float Scene::ComputeTriangleWeight(
    const XMFLOAT3& v0, const XMFLOAT3& v1, const XMFLOAT3& v2,
    const XMFLOAT3& emissive, const XMMATRIX& M)
{
    XMVECTOR p0 = XMVector3TransformCoord(XMLoadFloat3(&v0), M);
    XMVECTOR p1 = XMVector3TransformCoord(XMLoadFloat3(&v1), M);
    XMVECTOR p2 = XMVector3TransformCoord(XMLoadFloat3(&v2), M);
    float area = 0.5f * XMVectorGetX(XMVector3Length(XMVector3Cross(p1 - p0, p2 - p0)));
    return std::max(area, 1e-10f) * Luminance(emissive);
}

// ─────────────────────────────────────────────────────────────────
// Upload the compressed material buffer (48 B / material) and the
// per-primitive materialID buffer. See MaterialSoA::BuildGpuPacked
// and shaders/Material_Decoder_v8.hlsli for the layout.
void Scene::UploadMaterials(ID3D12Device* device) {
    //PLANET: append the shared terrain materialID region (every terrain
    //triangle uses the one flat terrain material appended in ReserveTerrain).
    //terrainMatIDBase is the element start; every terrain cell points its
    //materialBase here. Done before the upload below picks up materialIDs.size().
    if (terrainMatIDElems && terrainMatIDBase == 0) {
        terrainMatIDBase = (UINT)materialIDs.size();
        materialIDs.resize((size_t)terrainMatIDBase + terrainMatIDElems, terrainMatIndex);
    }

    materials.BuildGpuPacked(materialPacked);

    {
        const UINT sz = (UINT)materialPacked.size() * sizeof(uint32_t);
        auto d = CD3DX12_RESOURCE_DESC::Buffer(sz);
        ThrowIfFailed(device->CreateCommittedResource(
            &nv_helpers_dx12::kUploadHeapProps, D3D12_HEAP_FLAG_NONE, &d,
            D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&materialBuffer)));
        UINT8* p; materialBuffer->Map(0, nullptr, (void**)&p);
        memcpy(p, materialPacked.data(), sz);
        materialBuffer->Unmap(0, nullptr);
    }

    {
        //A scene with no triangle meshes has an empty materialIDs array. D3D12
        //rejects a zero-width buffer with E_INVALIDARG, so round up to a
        //minimal valid allocation that nothing then reads.
        const UINT idCount = (UINT)materialIDs.size();
        const UINT sz = std::max<UINT>(idCount * (UINT)sizeof(UINT), 256u);
        auto d = CD3DX12_RESOURCE_DESC::Buffer(sz);
        ThrowIfFailed(device->CreateCommittedResource(
            &nv_helpers_dx12::kUploadHeapProps, D3D12_HEAP_FLAG_NONE, &d,
            D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&materialIndexBuffer)));
        UINT8* p; materialIndexBuffer->Map(0, nullptr, (void**)&p);
        if (idCount) memcpy(p, materialIDs.data(), idCount * sizeof(UINT));
        materialIndexBuffer->Unmap(0, nullptr);
    }
}

void Scene::UpdateMaterialBuffer() {
    if (!materialBuffer || materials.empty()) return;

    // Repack everything; edits are infrequent and material count is small.
    materials.BuildGpuPacked(materialPacked);

    const UINT sz = (UINT)materialPacked.size() * sizeof(uint32_t);
    UINT8* p = nullptr;
    CD3DX12_RANGE readRange(0, 0);
    if (SUCCEEDED(materialBuffer->Map(0, &readRange, (void**)&p))) {
        memcpy(p, materialPacked.data(), sz);
        materialBuffer->Unmap(0, nullptr);
    }
}

// ─────────────────────────────────────────────────────────────────
void Scene::CreateTriToLightIdBuffer(ID3D12Device* device, ID3D12GraphicsCommandList* cmdList) {
    //PLANET: append the shared terrain triToLightId region (all 0xFFFFFFFF =
    //"not a light"), so terrain - shaded through the unified path - reports no
    //emission and lightID = none. Done before the empty check so a scene with
    //no emissive triangles still gets a valid buffer when terrain is enabled.
    if (terrainTriLightElems && terrainTriLightBase == 0) {
        terrainTriLightBase = (UINT)triToLightId.size();
        triToLightId.resize((size_t)terrainTriLightBase + terrainTriLightElems, 0xFFFFFFFFu);
    }
    if (triToLightId.empty()) return;
    const UINT bytes = (UINT)(triToLightId.size() * sizeof(uint32_t));
    ComPtr<ID3D12Resource> upload = nv_helpers_dx12::CreateBuffer(
        device, bytes, D3D12_RESOURCE_FLAG_NONE,
        D3D12_RESOURCE_STATE_GENERIC_READ, nv_helpers_dx12::kUploadHeapProps);
    { void* p = nullptr; CD3DX12_RANGE r(0,0);
      ThrowIfFailed(upload->Map(0, &r, &p));
      memcpy(p, triToLightId.data(), bytes);
      upload->Unmap(0, nullptr); }
    triToLightIdBuffer = nv_helpers_dx12::CreateBuffer(
        device, bytes, D3D12_RESOURCE_FLAG_NONE,
        D3D12_RESOURCE_STATE_COPY_DEST, nv_helpers_dx12::kDefaultHeapProps);
    cmdList->CopyBufferRegion(triToLightIdBuffer.Get(), 0, upload.Get(), 0, bytes);
    auto br = CD3DX12_RESOURCE_BARRIER::Transition(
        triToLightIdBuffer.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_STATE_GENERIC_READ);
    cmdList->ResourceBarrier(1, &br);
}

void Scene::CreateEmissiveTrianglesBuffer(
    ID3D12Device* device, ID3D12GraphicsCommandList* cmdList,
    ID3D12CommandQueue* queue, ID3D12CommandAllocator* alloc)
{
    size_t bufferSize = emissiveTriangles.size() * sizeof(LightTriangle);
    for (auto& t : emissiveTriangles)
        t.triCount = (UINT)emissiveTriangles.size();
    ComPtr<ID3D12Resource> upload = nv_helpers_dx12::CreateBuffer(
        device, (UINT)bufferSize, D3D12_RESOURCE_FLAG_NONE,
        D3D12_RESOURCE_STATE_GENERIC_READ, nv_helpers_dx12::kUploadHeapProps);
    { void* p = nullptr; CD3DX12_RANGE r(0,0);
      ThrowIfFailed(upload->Map(0, &r, &p));
      memcpy(p, emissiveTriangles.data(), bufferSize);
      upload->Unmap(0, nullptr); }
    emissiveTrianglesBuffer = nv_helpers_dx12::CreateBuffer(
        device, (UINT)bufferSize, D3D12_RESOURCE_FLAG_NONE,
        D3D12_RESOURCE_STATE_COPY_DEST, nv_helpers_dx12::kDefaultHeapProps);
    cmdList->CopyBufferRegion(emissiveTrianglesBuffer.Get(), 0, upload.Get(), 0, bufferSize);
    auto barrier = CD3DX12_RESOURCE_BARRIER::Transition(
        emissiveTrianglesBuffer.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_STATE_GENERIC_READ);
    cmdList->ResourceBarrier(1, &barrier);
}
