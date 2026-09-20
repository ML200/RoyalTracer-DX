#include "../stdafx.h"
#include <fstream>
#include "Scene.h"
#include "../DXRHelper.h"

void MeshGPU::CreateBlasBuildInputs(ID3D12Device* device) {
    auto upload = [&](const void* data, size_t bytes, ComPtr<ID3D12Resource>& resource) {
        if (resource)
            return;
        resource = nv_helpers_dx12::CreateBuffer(device, bytes, D3D12_RESOURCE_FLAG_NONE,
                                                 D3D12_RESOURCE_STATE_GENERIC_READ, nv_helpers_dx12::kUploadHeapProps);
        void* mapped = nullptr;
        CD3DX12_RANGE noRead(0, 0);
        ThrowIfFailed(resource->Map(0, &noRead, &mapped));
        if (bytes)
            memcpy(mapped, data, bytes);
        resource->Unmap(0, nullptr);
    };
    upload(cpuVertices.data(), cpuVertices.size() * sizeof(Vertex), vertexBuffer);
    upload(cpuIndices.data(), cpuIndices.size() * sizeof(UINT), indexBuffer);
}

void Scene::PropagateModelTransforms() {
    for (auto& model : models) {
        for (UINT i = model.instanceStart; i < model.instanceStart + model.instanceCount; ++i) {
            auto& inst = instances[i];
            inst.worldTransform = inst.localTransform * model.worldTransform;
        }
    }
}

void Scene::MarkModelMoved(UINT modelIndex) {
    if (modelIndex >= models.size())
        return;
    auto& model = models[modelIndex];
    model.RebuildTransform();

    for (UINT i = model.instanceStart; i < model.instanceStart + model.instanceCount; ++i) {
        instances[i].worldTransform = instances[i].localTransform * model.worldTransform;
        MarkInstanceDirty(i);
    }

    tlasDirty = true;
    lightTreeDirty = true;
}

void Scene::MarkMaterialsDirty(bool emissionChanged) {
    materialsDirty = true;
    if (emissionChanged) {
        lightTreeDirty = true;
        emissivesDirty = true;
    }
}

void Scene::MarkInstanceDirty(UINT instanceIndex) {
    if (instanceIndex < instanceDirty.size())
        instanceDirty[instanceIndex] = 1;
}

void Scene::MarkAllInstancesDirty() {
    std::fill(instanceDirty.begin(), instanceDirty.end(), 1);
}

void Scene::ReserveTerrain(UINT vertexElems, UINT indexElems, UINT matIDElems, UINT triLightElems, UINT instanceSlots,
                           UINT propsBase) {
    terrainVertexElems = vertexElems;
    terrainIndexElems = indexElems;
    terrainMatIDElems = matIDElems;
    terrainTriLightElems = triLightElems;
    terrainInstanceSlots = instanceSlots;
    terrainPropsBase = propsBase;

    Material tm;
    tm.Kd = XMFLOAT4(0.42f, 0.36f, 0.30f, 1.0f);
    tm.Ke = XMFLOAT3(0.0f, 0.0f, 0.0f);
    tm.Ni = 1.0f;
    tm.Pr_Pm_Ps_Pc = XMFLOAT4(1.0f, 0.0f, 0.0f, 0.0f);
    tm.albedoTexID = -1;
    tm.normalTexID = -1;
    tm.rmaTexID = -1;
    tm.alphaThreshold = 1.0f;
    materials.push_back(tm);
    terrainMatIndex = (UINT)materials.size() - 1;
}

void Scene::ReserveRocks(UINT instanceSlots) {
    rockInstanceSlots = instanceSlots;
    rockPropsBase = terrainPropsBase + terrainInstanceSlots;
}

void Scene::ReserveVoxels(UINT vertexElems, UINT indexElems, UINT matIDElems, UINT instanceSlots, UINT minPropsBase) {
    voxelVertexElems = vertexElems;
    voxelIndexElems = indexElems;
    voxelMatIDElems = matIDElems;
    voxelInstanceSlots = instanceSlots;

    const UINT afterRocks = terrainInstanceSlots ? (rockPropsBase + rockInstanceSlots) : 0u;
    voxelPropsBase = std::max(minPropsBase, afterRocks);
}

void Scene::BuildGlobalMeshBuffers(ID3D12Device* device, ID3D12GraphicsCommandList* cmdList) {
    SCOPE_TIMER("BuildGlobalMeshBuffers");

    // Global bases let terrain, voxel, and imported geometry share buffers.
    geoOffsets.resize(meshes.size());
    size_t totalV = 0, totalI = 0;

    for (size_t m = 0; m < meshes.size(); ++m) {
        meshes[m].globalVertexBase = (UINT)totalV;
        meshes[m].globalIndexBase = (UINT)totalI;
        geoOffsets[m] = {(UINT)totalV, (UINT)totalI, meshes[m].materialIDBase};
        totalV += meshes[m].vertexCount;
        totalI += meshes[m].indexCount;
    }
    totalVertexCount = (UINT)totalV;
    totalIndexCount = (UINT)totalI;

    terrainVertexBase = totalVertexCount;
    terrainIndexBase = totalIndexCount;

    voxelVertexBase = totalVertexCount + terrainVertexElems;
    voxelIndexBase = totalIndexCount + terrainIndexElems;

    const uint64_t vbBytes = (uint64_t)combinedVertexCount() * sizeof(BTriVertex);
    const uint64_t ibBytes = (uint64_t)combinedIndexCount() * sizeof(uint32_t);

    const uint64_t sceneVbBytes = (uint64_t)totalVertexCount * sizeof(BTriVertex);
    const uint64_t sceneIbBytes = (uint64_t)totalIndexCount * sizeof(uint32_t);

    // Only the planet's terrain generator writes vertices from the CPU and needs the buffers
    // mapped in host memory. Voxel chunks arrive through GPU copies, which need a default-heap
    // destination; an upload heap can neither be copied into nor be fetched from at speed.
    const bool hasTerrain = terrainVertexElems > 0 || terrainIndexElems > 0;

    uint8_t* dstVertsRaw;
    uint8_t* dstIdxRaw;
    if (hasTerrain) {
        vertexGlobal =
            nv_helpers_dx12::CreateBuffer(device, vbBytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ,
                                          nv_helpers_dx12::kUploadHeapProps);
        vertexGlobal->SetName(L"GlobalVertexBuffer");
        indexGlobal =
            nv_helpers_dx12::CreateBuffer(device, ibBytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ,
                                          nv_helpers_dx12::kUploadHeapProps);
        indexGlobal->SetName(L"GlobalIndexBuffer");

        CD3DX12_RANGE noRead(0, 0);
        vertexGlobal->Map(0, &noRead, (void**)&vertexGlobalMapped);
        indexGlobal->Map(0, &noRead, (void**)&indexGlobalMapped);
        dstVertsRaw = vertexGlobalMapped;
        dstIdxRaw = indexGlobalMapped;
    } else {
        vertexGlobal =
            nv_helpers_dx12::CreateBuffer(device, vbBytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST,
                                          nv_helpers_dx12::kDefaultHeapProps);
        vertexGlobal->SetName(L"GlobalVertexBuffer");
        indexGlobal = nv_helpers_dx12::CreateBuffer(device, ibBytes, D3D12_RESOURCE_FLAG_NONE,
                                                    D3D12_RESOURCE_STATE_COPY_DEST, nv_helpers_dx12::kDefaultHeapProps);
        indexGlobal->SetName(L"GlobalIndexBuffer");

        vertexGlobalUpload =
            nv_helpers_dx12::CreateBuffer(device, sceneVbBytes, D3D12_RESOURCE_FLAG_NONE,
                                          D3D12_RESOURCE_STATE_GENERIC_READ, nv_helpers_dx12::kUploadHeapProps);
        indexGlobalUpload =
            nv_helpers_dx12::CreateBuffer(device, sceneIbBytes, D3D12_RESOURCE_FLAG_NONE,
                                          D3D12_RESOURCE_STATE_GENERIC_READ, nv_helpers_dx12::kUploadHeapProps);

        CD3DX12_RANGE noRead(0, 0);
        vertexGlobalUpload->Map(0, &noRead, (void**)&dstVertsRaw);
        indexGlobalUpload->Map(0, &noRead, (void**)&dstIdxRaw);
        vertexGlobalMapped = nullptr;
        indexGlobalMapped = nullptr;
    }

    auto* dstVerts = reinterpret_cast<BTriVertex*>(dstVertsRaw);
    auto* dstIdx = reinterpret_cast<uint32_t*>(dstIdxRaw);

    for (size_t m = 0; m < meshes.size(); ++m) {
        const auto& mesh = meshes[m];
        const UINT vBase = mesh.globalVertexBase;

        BTriVertex* outV = dstVerts + vBase;
        for (UINT i = 0; i < mesh.vertexCount; ++i) {
            const Vertex& sv = mesh.cpuVertices[i];
            outV[i].vertex = sv.position;
            XMVECTOR normal =
                XMVector3Normalize(XMVectorSet(sv.normal_material.x, sv.normal_material.y, sv.normal_material.z, 0.0f));
            outV[i].packedNormal = EncodeNormalOct(normal);
            PackedVector::XMStoreHalf2(&outV[i].texCoord, XMLoadFloat2(&sv.texCoord));
        }

        uint32_t* outI = dstIdx + mesh.globalIndexBase;
        for (UINT i = 0; i < mesh.indexCount; ++i)
            outI[i] = mesh.cpuIndices[i] + vBase;
    }

    if (!hasTerrain) {
        vertexGlobalUpload->Unmap(0, nullptr);
        indexGlobalUpload->Unmap(0, nullptr);
        if (sceneVbBytes)
            cmdList->CopyBufferRegion(vertexGlobal.Get(), 0, vertexGlobalUpload.Get(), 0, sceneVbBytes);
        if (sceneIbBytes)
            cmdList->CopyBufferRegion(indexGlobal.Get(), 0, indexGlobalUpload.Get(), 0, sceneIbBytes);
        const D3D12_RESOURCE_BARRIER toRead[2] = {
            CD3DX12_RESOURCE_BARRIER::Transition(vertexGlobal.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                                                 D3D12_RESOURCE_STATE_GENERIC_READ),
            CD3DX12_RESOURCE_BARRIER::Transition(indexGlobal.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                                                 D3D12_RESOURCE_STATE_GENERIC_READ),
        };
        cmdList->ResourceBarrier(2, toRead);
    }
}

void Scene::CreateInstancePropertiesBuffer(ID3D12Device* device) {
    const uint32_t count = instancePropsCount();
    const uint32_t size = ROUND_UP(count * sizeof(InstanceProperties), D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);

    if (instanceProperties) {
        const D3D12_RESOURCE_DESC d = instanceProperties->GetDesc();
        if (d.Width >= size)
            return;
    }
    instanceProperties = nv_helpers_dx12::CreateBuffer(
        device, size, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ, nv_helpers_dx12::kUploadHeapProps);
}

void Scene::PrepareInstanceProperties() {
    const size_t count = instances.size();

    cpuInstanceProps.resize(count);
    instanceDirty.resize(count, 1);
    instanceInitialized.resize(count, 0);

    dirtyInstanceList.clear();
    for (size_t i = 0; i < count; ++i) {
        if (!instanceDirty[i])
            continue;
        dirtyInstanceList.push_back(static_cast<uint32_t>(i));
    }

    // Store transforms relative to the current floating origin.
    const XMVECTOR shift = XMVectorSet(sceneOriginWorld.x, sceneOriginWorld.y, sceneOriginWorld.z, 0.0f);

    const XMVECTOR prevShift =
        XMVectorSet(prevSceneOriginWorld.x, prevSceneOriginWorld.y, prevSceneOriginWorld.z, 0.0f);
    const XMVECTOR originDelta = XMVectorSubtract(shift, prevShift);
    const bool originShifted = XMVector3LengthSq(originDelta).m128_f32[0] > 0.0f;

    for (uint32_t idx : dirtyInstanceList) {
        auto& dst = cpuInstanceProps[idx];
        auto& si = instances[idx];
        XMMATRIX M = si.worldTransform;

        M.r[3] = XMVectorSubtract(M.r[3], shift);
        XMVECTOR det;

        bool isNew = !instanceInitialized[idx];

        if (!isNew) {
            dst.prevObjectToWorld = dst.objectToWorld;
            dst.prevObjectToWorldInverse = dst.objectToWorldInverse;
            dst.prevObjectToWorldNormal = dst.objectToWorldNormal;

            if (originShifted) {
                dst.prevObjectToWorld.r[3] = XMVectorSubtract(dst.prevObjectToWorld.r[3], originDelta);
                dst.prevObjectToWorldInverse = XMMatrixInverse(&det, dst.prevObjectToWorld);
            }
        }

        dst.objectToWorld = M;
        dst.objectToWorldInverse = XMMatrixInverse(&det, M);

        XMMATRIX upper3x3 = M;
        upper3x3.r[0].m128_f32[3] = upper3x3.r[1].m128_f32[3] = upper3x3.r[2].m128_f32[3] = 0.f;
        upper3x3.r[3] = {0, 0, 0, 1.f};
        dst.objectToWorldNormal = XMMatrixTranspose(XMMatrixInverse(&det, upper3x3));

        if (isNew) {
            dst.prevObjectToWorld = dst.objectToWorld;
            dst.prevObjectToWorldInverse = dst.objectToWorldInverse;
            dst.prevObjectToWorldNormal = dst.objectToWorldNormal;
            instanceInitialized[idx] = 1;
        }

        const MeshGPU& mesh = meshes[si.meshIndex];
        dst.opaqueTriCount = mesh.opaqueTriCount;
        dst.indexBase = mesh.globalIndexBase;
        dst.vertexBase = mesh.globalVertexBase;
        dst.materialBase = mesh.materialIDBase;

        const bool lit = lightClassEnabled[MeshLightClass(si.meshIndex)];
        dst.triToLightBase = (lit && si.meshIndex < meshLightBase.size()) ? meshLightBase[si.meshIndex] : 0xFFFFFFFFu;
        dst.lightSlot = (lit && idx < instanceLightSlot.size()) ? instanceLightSlot[idx] : 0xFFFFFFFFu;

        if (idx < tlasInstances.size())
            tlasInstances[idx].transform = M;
    }

    prevSceneOriginWorld = sceneOriginWorld;
}

void Scene::UploadInstanceProperties() {
    if (!dirtyInstanceList.empty()) {
        uint8_t* gpuDst = nullptr;
        CD3DX12_RANGE readRange(0, 0);
        ThrowIfFailed(instanceProperties->Map(0, &readRange, reinterpret_cast<void**>(&gpuDst)));
        for (uint32_t idx : dirtyInstanceList) {
            const InstancePropertiesCpu& src = cpuInstanceProps[idx];
            InstanceProperties gp;
            gp.objectToWorld = MakeFloat3x4(src.objectToWorld);
            gp.objectToWorldInverse = MakeFloat3x4(src.objectToWorldInverse);
            gp.objectToWorldNormal = MakeFloat3x4(src.objectToWorldNormal);
            gp.indexBase = src.indexBase;
            gp.vertexBase = src.vertexBase;
            gp.materialBase = src.materialBase;
            gp.triToLightBase = src.triToLightBase;
            gp.opaqueTriCount = src.opaqueTriCount;
            gp._pad[0] = src._pad[0];
            gp._pad[1] = src._pad[1];
            gp.lightSlot = src.lightSlot;
            gp.prevObjectToWorld = MakeFloat3x4(src.prevObjectToWorld);
            memcpy(gpuDst + idx * sizeof(InstanceProperties), &gp, sizeof(gp));
        }
        instanceProperties->Unmap(0, nullptr);
    }

    for (uint32_t idx : dirtyInstanceList) {
        auto& p = cpuInstanceProps[idx];
        bool alreadySettled = (memcmp(&p.prevObjectToWorld, &p.objectToWorld, sizeof(XMMATRIX)) == 0);
        p.prevObjectToWorld = p.objectToWorld;
        p.prevObjectToWorldInverse = p.objectToWorldInverse;
        p.prevObjectToWorldNormal = p.objectToWorldNormal;

        instanceDirty[idx] = alreadySettled ? 0 : 1;
    }
}

void Scene::RebuildTLASInstanceList() {
    tlasInstances.clear();
    tlasInstances.reserve(instances.size());

    const XMVECTOR shift = XMVectorSet(sceneOriginWorld.x, sceneOriginWorld.y, sceneOriginWorld.z, 0.0f);
    for (size_t i = 0; i < instances.size(); ++i) {
        const auto& si = instances[i];
        const auto& mesh = meshes[si.meshIndex];

        UINT hitGroupContrib = static_cast<UINT>(i) * 2;

        auto flags = (mesh.alphaTriCount == 0) ? D3D12_RAYTRACING_INSTANCE_FLAG_FORCE_OPAQUE
                                               : D3D12_RAYTRACING_INSTANCE_FLAG_NONE;

        XMMATRIX shifted = si.worldTransform;
        shifted.r[3] = XMVectorSubtract(shifted.r[3], shift);

        tlasInstances.push_back({mesh.blas, shifted, hitGroupContrib, flags});
    }
}

void Scene::CollectEmissiveTriangles() {
    // Build triangle and instance light records from the current material set.
    emissiveTriangles.clear();
    triToLightId.clear();
    meshLightBase.assign(meshes.size(), 0xFFFFFFFFu);
    meshRecordBase.assign(meshes.size(), 0xFFFFFFFFu);

    terrainTriLightBase = 0xFFFFFFFFu;

    for (size_t m = 0; m < meshes.size(); ++m) {
        const auto& mesh = meshes[m];
        const UINT tris = mesh.indexCount / 3;
        bool anyEmissive = false;
        for (UINT t = 0; t < tris && !anyEmissive; ++t) {
            const XMFLOAT3& Ke = materials.Ke[mesh.cpuMaterialIDs[t]];
            anyEmissive = Ke.x + Ke.y + Ke.z > 0.0f;
        }
        if (!anyEmissive)
            continue;

        meshLightBase[m] = (uint32_t)triToLightId.size();
        meshRecordBase[m] = (uint32_t)emissiveTriangles.size();
        triToLightId.resize(triToLightId.size() + tris, 0xFFFFFFFFu);
        for (UINT t = 0; t < tris; ++t) {
            const XMFLOAT3& Ke = materials.Ke[mesh.cpuMaterialIDs[t]];
            if (Ke.x + Ke.y + Ke.z <= 0.0f)
                continue;

            LightTriangle lt{};
            lt.x = mesh.cpuVertices[mesh.cpuIndices[3 * t + 0]].position;
            lt.y = mesh.cpuVertices[mesh.cpuIndices[3 * t + 1]].position;
            lt.z = mesh.cpuVertices[mesh.cpuIndices[3 * t + 2]].position;
            lt.meshID = (UINT)m;

            lt.weight = ComputeTriangleWeight(lt.x, lt.y, lt.z, Ke, XMMatrixIdentity());
            lt.emission = Ke;

            triToLightId[(size_t)meshLightBase[m] + t] = (uint32_t)emissiveTriangles.size();
            emissiveTriangles.push_back(lt);
        }
    }

    lightInstances.clear();
    instanceLightSlot.assign(instances.size(), 0xFFFFFFFFu);
    for (size_t i = 0; i < instances.size(); ++i) {
        const UINT m = instances[i].meshIndex;
        if (m >= meshes.size() || meshLightBase[m] == 0xFFFFFFFFu)
            continue;
        instanceLightSlot[i] = (uint32_t)lightInstances.size();
        lightInstances.push_back({(UINT)i, m});
    }

    if (emissiveTriangles.empty())
        emissiveTriangles.push_back(LightTriangle{});

    LOG(L"Emissive triangles: " << emissiveTriangles.size() << L" unique records, " << lightInstances.size()
                                << L" light instances of " << instances.size());
}

float Scene::ComputeTriangleWeight(const XMFLOAT3& v0, const XMFLOAT3& v1, const XMFLOAT3& v2, const XMFLOAT3& emissive,
                                   const XMMATRIX& M) {
    XMVECTOR p0 = XMVector3TransformCoord(XMLoadFloat3(&v0), M);
    XMVECTOR p1 = XMVector3TransformCoord(XMLoadFloat3(&v1), M);
    XMVECTOR p2 = XMVector3TransformCoord(XMLoadFloat3(&v2), M);
    float area = 0.5f * XMVectorGetX(XMVector3Length(XMVector3Cross(p1 - p0, p2 - p0)));
    return std::max(area, 1e-10f) * Luminance(emissive);
}

void Scene::UploadMaterials(ID3D12Device* device) {
    if (terrainMatIDElems && terrainMatIDBase == 0) {
        terrainMatIDBase = (UINT)materialIDs.size();
        materialIDs.resize((size_t)terrainMatIDBase + terrainMatIDElems, terrainMatIndex);
    }

    if (voxelMatIDElems && !voxelMatIDReserved) {
        voxelMatIDBase = (UINT)materialIDs.size();
        materialIDs.resize((size_t)voxelMatIDBase + voxelMatIDElems, 0u);
        voxelMatIDReserved = true;
    }

    materials.BuildGpuPacked(materialPacked);

    {
        const UINT sz = (UINT)materialPacked.size() * sizeof(uint32_t);
        auto d = CD3DX12_RESOURCE_DESC::Buffer(sz);
        ThrowIfFailed(device->CreateCommittedResource(&nv_helpers_dx12::kUploadHeapProps, D3D12_HEAP_FLAG_NONE, &d,
                                                      D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                      IID_PPV_ARGS(&materialBuffer)));
        UINT8* p;
        materialBuffer->Map(0, nullptr, (void**)&p);
        memcpy(p, materialPacked.data(), sz);
        materialBuffer->Unmap(0, nullptr);
    }

    {
        const size_t idCount = materialIDs.size();
        const uint64_t sz = std::max<uint64_t>((uint64_t)idCount * sizeof(UINT), 256u);
        auto d = CD3DX12_RESOURCE_DESC::Buffer(sz);
        ThrowIfFailed(device->CreateCommittedResource(&nv_helpers_dx12::kUploadHeapProps, D3D12_HEAP_FLAG_NONE, &d,
                                                      D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                      IID_PPV_ARGS(&materialIndexBuffer)));
        UINT8* p;
        materialIndexBuffer->Map(0, nullptr, (void**)&p);
        if (idCount)
            memcpy(p, materialIDs.data(), idCount * sizeof(UINT));
        materialIndexBuffer->Unmap(0, nullptr);
    }
}

void Scene::UpdateMaterialBuffer() {
    if (!materialBuffer || materials.empty())
        return;

    materials.BuildGpuPacked(materialPacked);

    const UINT sz = (UINT)materialPacked.size() * sizeof(uint32_t);
    UINT8* p = nullptr;
    CD3DX12_RANGE readRange(0, 0);
    if (SUCCEEDED(materialBuffer->Map(0, &readRange, (void**)&p))) {
        memcpy(p, materialPacked.data(), sz);
        materialBuffer->Unmap(0, nullptr);
    }
}

void Scene::CreateTriToLightIdBuffer(ID3D12Device* device, ID3D12GraphicsCommandList* cmdList) {
    if (triToLightId.empty())
        return;
    const uint64_t bytes = (uint64_t)triToLightId.size() * sizeof(uint32_t);
    ComPtr<ID3D12Resource> upload = nv_helpers_dx12::CreateBuffer(
        device, bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ, nv_helpers_dx12::kUploadHeapProps);
    {
        void* p = nullptr;
        CD3DX12_RANGE r(0, 0);
        ThrowIfFailed(upload->Map(0, &r, &p));
        memcpy(p, triToLightId.data(), bytes);
        upload->Unmap(0, nullptr);
    }
    triToLightIdBuffer = nv_helpers_dx12::CreateBuffer(
        device, bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST, nv_helpers_dx12::kDefaultHeapProps);
    cmdList->CopyBufferRegion(triToLightIdBuffer.Get(), 0, upload.Get(), 0, bytes);
    auto br = CD3DX12_RESOURCE_BARRIER::Transition(triToLightIdBuffer.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                                                   D3D12_RESOURCE_STATE_GENERIC_READ);
    cmdList->ResourceBarrier(1, &br);
    // Keep upload memory alive until the GPU copy completes.
    pendingLightUploads.push_back(std::move(upload));
}

void Scene::CreateEmissiveTrianglesBuffer(ID3D12Device* device, ID3D12GraphicsCommandList* cmdList,
                                          ID3D12CommandQueue* queue, ID3D12CommandAllocator* alloc) {
    size_t bufferSize = emissiveTriangles.size() * sizeof(LightTriangle);
    for (auto& t : emissiveTriangles)
        t.triCount = (UINT)emissiveTriangles.size();
    ComPtr<ID3D12Resource> upload =
        nv_helpers_dx12::CreateBuffer(device, bufferSize, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ,
                                      nv_helpers_dx12::kUploadHeapProps);
    {
        void* p = nullptr;
        CD3DX12_RANGE r(0, 0);
        ThrowIfFailed(upload->Map(0, &r, &p));
        memcpy(p, emissiveTriangles.data(), bufferSize);
        upload->Unmap(0, nullptr);
    }
    emissiveTrianglesBuffer =
        nv_helpers_dx12::CreateBuffer(device, bufferSize, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST,
                                      nv_helpers_dx12::kDefaultHeapProps);
    cmdList->CopyBufferRegion(emissiveTrianglesBuffer.Get(), 0, upload.Get(), 0, bufferSize);
    auto barrier = CD3DX12_RESOURCE_BARRIER::Transition(emissiveTrianglesBuffer.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                                                        D3D12_RESOURCE_STATE_GENERIC_READ);
    cmdList->ResourceBarrier(1, &barrier);
    // Retain the staging resource through the asynchronous submission.
    pendingLightUploads.push_back(std::move(upload));
}
