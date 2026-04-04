// ═══════════════════════════════════════════════════════════════════
// Scene/AssetLoader.cpp — Creates SceneModel + SceneInstance entries
//                         Batched texture uploads to avoid TDR on
//                         large scenes (e.g. 4K Sponza).
// ═══════════════════════════════════════════════════════════════════

#include "../stdafx.h"
#include <fstream>
#include "AssetLoader.h"
#include "../DXRHelper.h"

// How many textures to upload per command-list submission.
// Each 4K RGBA mip chain is ~85 MB of upload + default-heap traffic.
// 4 textures ≈ 340 MB per batch — well under the 2-second TDR window
// on any modern GPU, and keeps peak staging memory around 340 MB.
static constexpr UINT TEXTURE_BATCH_SIZE = 4;

MeshSplitResult AssetLoader::SplitOpaqueAlpha(
    const std::vector<UINT>& indices,
    const std::vector<UINT>& perTriMatIDs,
    const std::vector<Material>& allMaterials)
{
    MeshSplitResult r;
    const UINT triCount = (UINT)indices.size() / 3;
    std::vector<UINT> opaqueIdx, alphaIdx, opaqueMatIDs, alphaMatIDs;

    for (UINT t = 0; t < triCount; ++t) {
        bool isAlpha = (allMaterials[perTriMatIDs[t]].alphaThreshold < 1.0f);
        auto& dstIdx = isAlpha ? alphaIdx  : opaqueIdx;
        auto& dstMat = isAlpha ? alphaMatIDs : opaqueMatIDs;
        dstIdx.push_back(indices[3*t+0]);
        dstIdx.push_back(indices[3*t+1]);
        dstIdx.push_back(indices[3*t+2]);
        dstMat.push_back(perTriMatIDs[t]);
    }

    r.opaqueTriCount = (UINT)opaqueIdx.size() / 3;
    r.alphaTriCount  = (UINT)alphaIdx.size()  / 3;
    r.reorderedIndices = std::move(opaqueIdx);
    r.reorderedIndices.insert(r.reorderedIndices.end(), alphaIdx.begin(), alphaIdx.end());
    r.reorderedMaterialIDs = std::move(opaqueMatIDs);
    r.reorderedMaterialIDs.insert(r.reorderedMaterialIDs.end(), alphaMatIDs.begin(), alphaMatIDs.end());
    return r;
}

// ─────────────────────────────────────────────────────────────────
void AssetLoader::LoadModels(
    const std::vector<ModelEntry>& modelEntries,
    Scene& scene,
    ID3D12Device* device,
    ID3D12GraphicsCommandList* cmdList,
    const FlushFn& flushAndReset)
{
    std::map<std::string, uint32_t> textureMap;
    std::vector<TextureData> albedoTextures, normalTextures, rmaTextures;

    for (const auto& entry : modelEntries) {
        const auto& modelPath      = entry.path;
        const XMMATRIX& modelXform = entry.transform;

        // ── Create SceneModel ────────────────────────────────────
        SceneModel model;
        model.filePath      = modelPath;
        model.name          = entry.name.empty() ? modelPath : entry.name;
        model.worldTransform = modelXform;
        model.meshStart      = (UINT)scene.meshes.size();
        model.instanceStart  = (UINT)scene.instances.size();

        // Decompose initial transform for editor
        XMVECTOR scaleV, rotQ, transV;
        XMMatrixDecompose(&scaleV, &rotQ, &transV, modelXform);
        XMStoreFloat3(&model.position, transV);
        XMStoreFloat3(&model.scale, scaleV);
        XMFLOAT4 q; XMStoreFloat4(&q, rotQ);
        float sinr = 2.0f * (q.w * q.x + q.y * q.z);
        float cosr = 1.0f - 2.0f * (q.x * q.x + q.y * q.y);
        model.rotation.x = XMConvertToDegrees(atan2f(sinr, cosr));
        float sinp = 2.0f * (q.w * q.y - q.z * q.x);
        model.rotation.y = XMConvertToDegrees(fabsf(sinp) >= 1.0f ? copysignf(XM_PIDIV2, sinp) : asinf(sinp));
        float siny = 2.0f * (q.w * q.z + q.x * q.y);
        float cosy = 1.0f - 2.0f * (q.y * q.y + q.z * q.z);
        model.rotation.z = XMConvertToDegrees(atan2f(siny, cosy));

        // ── Load file ────────────────────────────────────────────
        std::string matSearchPath = "./";
        auto lastSlash = modelPath.find_last_of("/\\");
        if (lastSlash != std::string::npos)
            matSearchPath = modelPath.substr(0, lastSlash + 1);

        std::string ext = modelPath.substr(modelPath.find_last_of('.') + 1);
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        LoadedScene loaded;
        if (ext == "glb" || ext == "gltf")
            loaded = ObjLoader::loadGlbFile(modelPath, textureMap,
                albedoTextures, normalTextures, rmaTextures, matSearchPath);
        else
            loaded = ObjLoader::loadObjFile(modelPath, textureMap,
                albedoTextures, normalTextures, rmaTextures, matSearchPath);

        // ── Merge materials ──────────────────────────────────────
        const UINT globalMatBase = (UINT)scene.materials.size();
        scene.materials.insert(scene.materials.end(),
            loaded.materials.begin(), loaded.materials.end());

        const UINT meshBaseIdx = (UINT)scene.meshes.size();

        // ── Process sub-meshes ───────────────────────────────────
        for (size_t mi = 0; mi < loaded.meshes.size(); ++mi) {
            auto& srcMesh = loaded.meshes[mi];
            MeshGPU gpu{};

            std::vector<UINT> globalMatIDs = srcMesh.perTriMaterialIDs;
            for (auto& mid : globalMatIDs) mid += globalMatBase;

            auto split = SplitOpaqueAlpha(srcMesh.indices, globalMatIDs, scene.materials);
            gpu.cpuVertices    = srcMesh.vertices;
            gpu.cpuIndices     = std::move(split.reorderedIndices);
            gpu.cpuMaterialIDs = std::move(split.reorderedMaterialIDs);
            gpu.vertexCount    = (UINT)gpu.cpuVertices.size();
            gpu.indexCount     = (UINT)gpu.cpuIndices.size();
            gpu.opaqueTriCount = split.opaqueTriCount;
            gpu.alphaTriCount  = split.alphaTriCount;
            gpu.materialIDBase = (UINT)scene.materialIDs.size();

            scene.materialIDs.insert(scene.materialIDs.end(),
                gpu.cpuMaterialIDs.begin(), gpu.cpuMaterialIDs.end());

            const UINT vbSize = gpu.vertexCount * sizeof(Vertex);
            auto vbDesc = CD3DX12_RESOURCE_DESC::Buffer(vbSize);
            ThrowIfFailed(device->CreateCommittedResource(
                &nv_helpers_dx12::kUploadHeapProps, D3D12_HEAP_FLAG_NONE,
                &vbDesc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                IID_PPV_ARGS(&gpu.vertexBuffer)));
            { UINT8* p; gpu.vertexBuffer->Map(0, nullptr, (void**)&p);
              memcpy(p, gpu.cpuVertices.data(), vbSize);
              gpu.vertexBuffer->Unmap(0, nullptr); }

            const UINT ibSize = gpu.indexCount * sizeof(UINT);
            auto ibDesc = CD3DX12_RESOURCE_DESC::Buffer(ibSize);
            ThrowIfFailed(device->CreateCommittedResource(
                &nv_helpers_dx12::kUploadHeapProps, D3D12_HEAP_FLAG_NONE,
                &ibDesc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                IID_PPV_ARGS(&gpu.indexBuffer)));
            { UINT8* p; gpu.indexBuffer->Map(0, nullptr, (void**)&p);
              memcpy(p, gpu.cpuIndices.data(), ibSize);
              gpu.indexBuffer->Unmap(0, nullptr); }

            scene.meshes.push_back(std::move(gpu));
        }

        // ── Create instances (localTransform from GLB scene graph) ──
        int subIdx = 0;
        for (const auto& [meshIdx, xform] : loaded.instances) {
            SceneInstance si{};
            si.meshIndex       = meshBaseIdx + meshIdx;
            si.modelIndex      = (UINT)scene.models.size();
            si.localTransform  = xform;
            si.worldTransform  = xform * modelXform;
            si.prevWorldTransform = si.worldTransform;
            si.name            = model.name + "_sub" + std::to_string(subIdx++);
            scene.instances.push_back(si);
        }

        model.meshCount     = (UINT)scene.meshes.size() - model.meshStart;
        model.instanceCount = (UINT)scene.instances.size() - model.instanceStart;

        UINT modelIdx = (UINT)scene.models.size();
        for (UINT i = model.instanceStart; i < model.instanceStart + model.instanceCount; ++i)
            scene.instances[i].modelIndex = modelIdx;

        scene.models.push_back(std::move(model));
    }

    // ── Bindless texture setup ───────────────────────────────────
    scene.bindlessAlbedoBase = BINDLESS_HEAP_START;
    scene.bindlessNormalBase = scene.bindlessAlbedoBase + (UINT)albedoTextures.size();
    scene.bindlessRmaBase    = scene.bindlessNormalBase + (UINT)normalTextures.size();
    scene.totalBindlessTextures = (UINT)(albedoTextures.size() + normalTextures.size() + rmaTextures.size());

    for (auto& mat : scene.materials) {
        if (mat.albedoTexID >= 0) mat.albedoTexID += (int)scene.bindlessAlbedoBase;
        if (mat.normalTexID >= 0) mat.normalTexID += (int)scene.bindlessNormalBase;
        if (mat.rmaTexID    >= 0) mat.rmaTexID    += (int)scene.bindlessRmaBase;
    }

    // Materials uploaded later by InitSceneGPU (after procedural meshes may add more)

    // Upload textures in batches — flushes every TEXTURE_BATCH_SIZE textures
    // to keep each submission under the TDR window and limit peak staging memory.
    CreateBindlessTextures(albedoTextures, scene.bindlessAlbedoBase, L"Albedo",
        device, cmdList, scene.bindlessGpuTextures, flushAndReset);
    CreateBindlessTextures(normalTextures, scene.bindlessNormalBase, L"Normal",
        device, cmdList, scene.bindlessGpuTextures, flushAndReset);
    CreateBindlessTextures(rmaTextures,    scene.bindlessRmaBase,    L"RMA",
        device, cmdList, scene.bindlessGpuTextures, flushAndReset);
}

// ─────────────────────────────────────────────────────────────────
void AssetLoader::CreateBindlessTextures(
    std::vector<TextureData>& textures,
    UINT heapBaseSlot,
    const std::wstring& debugPrefix,
    ID3D12Device* device,
    ID3D12GraphicsCommandList* cmdList,
    std::vector<ComPtr<ID3D12Resource>>& outGpuTextures,
    const FlushFn& flushAndReset)
{
    // Staging heaps for the current batch — released after each flush.
    std::vector<ComPtr<ID3D12Resource>> batchUploadHeaps;
    UINT batchCount = 0;

    for (size_t i = 0; i < textures.size(); ++i) {
        auto& tex = textures[i];
        const auto& meta = tex.image.GetMetadata();

        D3D12_RESOURCE_DESC d = {};
        d.Dimension        = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        d.Width            = meta.width;
        d.Height           = (UINT)meta.height;
        d.DepthOrArraySize = 1;
        d.MipLevels        = (UINT16)meta.mipLevels;
        d.Format           = meta.format;
        d.SampleDesc.Count = 1;

        ComPtr<ID3D12Resource> gpuTex;
        ThrowIfFailed(device->CreateCommittedResource(
            &nv_helpers_dx12::kDefaultHeapProps, D3D12_HEAP_FLAG_NONE,
            &d, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&gpuTex)));

        const UINT subCount = (UINT)meta.mipLevels;
        std::vector<D3D12_SUBRESOURCE_DATA> sr(subCount);
        for (UINT m = 0; m < subCount; ++m) {
            const auto* img = tex.image.GetImage(m, 0, 0);
            sr[m].pData      = img->pixels;
            sr[m].RowPitch   = (LONG_PTR)img->rowPitch;
            sr[m].SlicePitch = (LONG_PTR)img->slicePitch;
        }

        UINT64 uploadSize = GetRequiredIntermediateSize(gpuTex.Get(), 0, subCount);
        ComPtr<ID3D12Resource> upload;
        auto ud = CD3DX12_RESOURCE_DESC::Buffer(uploadSize);
        ThrowIfFailed(device->CreateCommittedResource(
            &nv_helpers_dx12::kUploadHeapProps, D3D12_HEAP_FLAG_NONE,
            &ud, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&upload)));

        UpdateSubresources(cmdList, gpuTex.Get(), upload.Get(), 0, 0, subCount, sr.data());

        auto b = CD3DX12_RESOURCE_BARRIER::Transition(gpuTex.Get(),
            D3D12_RESOURCE_STATE_COPY_DEST, kSRV);
        cmdList->ResourceBarrier(1, &b);

        outGpuTextures.push_back(gpuTex);
        batchUploadHeaps.push_back(std::move(upload));
        ++batchCount;

        // ── Flush batch ──────────────────────────────────────────
        // Submit the current command list, GPU-wait, reset, and
        // release staging heaps so the next batch starts clean.
        if (batchCount >= TEXTURE_BATCH_SIZE) {
            std::wcout << L"[AssetLoader] Flushing " << debugPrefix
                       << L" texture batch (" << batchCount << L" textures, "
                       << (i + 1) << L"/" << textures.size() << L")" << std::endl;

            flushAndReset();           // close → execute → fence wait → reset
            batchUploadHeaps.clear();  // release staging memory
            batchCount = 0;

            // Also release CPU-side mip data for already-uploaded textures
            // in this batch to reduce peak memory.
            for (size_t j = (i + 1 >= TEXTURE_BATCH_SIZE ? i + 1 - TEXTURE_BATCH_SIZE : 0);
                 j <= i; ++j) {
                textures[j].image.Release();
            }
        }
    }

    // Remaining textures (< TEXTURE_BATCH_SIZE) stay on the open command list.
    // They'll be flushed by the caller's next FlushAndReset().
    // But we still need to keep their upload heaps alive until that flush,
    // so stash them in the gpu textures vector (they'll be released after
    // the caller flushes).  Actually, we need a separate mechanism —
    // let's just flush the remainder here too for consistency.
    if (batchCount > 0) {
        std::wcout << L"[AssetLoader] Flushing final " << debugPrefix
                   << L" batch (" << batchCount << L" textures)" << std::endl;

        flushAndReset();
        batchUploadHeaps.clear();

        // Release remaining CPU mip data
        size_t start = textures.size() >= batchCount ? textures.size() - batchCount : 0;
        for (size_t j = start; j < textures.size(); ++j) {
            textures[j].image.Release();
        }
    }
}
