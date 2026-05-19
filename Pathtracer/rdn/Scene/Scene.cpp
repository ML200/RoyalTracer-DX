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

    const UINT vbBytes = totalVertexCount * sizeof(BTriVertex);
    const UINT ibBytes = totalIndexCount  * sizeof(uint32_t);

    auto makeDefault = [&](UINT bytes, ComPtr<ID3D12Resource>& dst, const wchar_t* name) {
        dst = nv_helpers_dx12::CreateBuffer(device, bytes,
            D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST,
            nv_helpers_dx12::kDefaultHeapProps);
        dst->SetName(name);
    };
    makeDefault(vbBytes, vertexGlobal, L"GlobalVertexBuffer");
    makeDefault(ibBytes, indexGlobal,  L"GlobalIndexBuffer");

    ComPtr<ID3D12Resource> upload = nv_helpers_dx12::CreateBuffer(
        device, vbBytes + ibBytes, D3D12_RESOURCE_FLAG_NONE,
        D3D12_RESOURCE_STATE_GENERIC_READ, nv_helpers_dx12::kUploadHeapProps);

    uint8_t* pUpload = nullptr;
    CD3DX12_RANGE r(0, 0);
    upload->Map(0, &r, (void**)&pUpload);

    auto* dstVerts = reinterpret_cast<BTriVertex*>(pUpload);
    auto* dstIdx   = reinterpret_cast<uint32_t*>(pUpload + vbBytes);

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
    upload->Unmap(0, nullptr);

    cmdList->CopyBufferRegion(vertexGlobal.Get(), 0, upload.Get(), 0, vbBytes);
    cmdList->CopyBufferRegion(indexGlobal.Get(),  0, upload.Get(), vbBytes, ibBytes);

    CD3DX12_RESOURCE_BARRIER br[] = {
        CD3DX12_RESOURCE_BARRIER::Transition(vertexGlobal.Get(),
            D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_GENERIC_READ),
        CD3DX12_RESOURCE_BARRIER::Transition(indexGlobal.Get(),
            D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_GENERIC_READ),
    };
    cmdList->ResourceBarrier(_countof(br), br);
}

// ─────────────────────────────────────────────────────────────────
void Scene::CreateInstancePropertiesBuffer(ID3D12Device* device) {
    uint32_t size = ROUND_UP(
        static_cast<uint32_t>(instances.size()) * sizeof(InstanceProperties),
        D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);
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
    // Write only dirty instances to GPU upload heap
    if (!dirtyInstanceList.empty()) {
        uint8_t* gpuDst = nullptr;
        CD3DX12_RANGE readRange(0, 0);
        ThrowIfFailed(instanceProperties->Map(0, &readRange, reinterpret_cast<void**>(&gpuDst)));
        for (uint32_t idx : dirtyInstanceList) {
            memcpy(gpuDst + idx * sizeof(InstanceProperties),
                   &cpuInstanceProps[idx], sizeof(InstanceProperties));
        }
        instanceProperties->Unmap(0, nullptr);
    }

    // Finalize prev = current and keep dirty for one settle frame so the
    // GPU receives the zeroed-out motion (prev == current) on the next upload.
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
// Upload the compressed material buffer (40 B / material) and the
// per-primitive materialID buffer. See MaterialSoA::BuildGpuPacked
// and shaders/Material_Decoder_v8.hlsli for the layout.
void Scene::UploadMaterials(ID3D12Device* device) {
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
        const UINT sz = (UINT)materialIDs.size() * sizeof(UINT);
        auto d = CD3DX12_RESOURCE_DESC::Buffer(sz);
        ThrowIfFailed(device->CreateCommittedResource(
            &nv_helpers_dx12::kUploadHeapProps, D3D12_HEAP_FLAG_NONE, &d,
            D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&materialIndexBuffer)));
        UINT8* p; materialIndexBuffer->Map(0, nullptr, (void**)&p);
        memcpy(p, materialIDs.data(), sz);
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
