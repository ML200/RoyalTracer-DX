// ═══════════════════════════════════════════════════════════════════
// Scene/OmmBuilder.cpp — OMM baking via NVIDIA OMM SDK + D3D12 build
//
// BakeAll: one ommCpuBake call per unique texture across ALL meshes,
//          then distributes results back per-mesh.
// ═══════════════════════════════════════════════════════════════════

#include "../stdafx.h"
#include "OmmBuilder.h"
#include "Scene.h"
#include "../DXRHelper.h"
#include <omm.h>
#include <unordered_map>
#include <chrono>

using namespace DirectX;

// ═════════════════════════════════════════════════════════════════
// ExtractAlphaChannel — pull single-channel UNORM8 from RGBA8
// ═════════════════════════════════════════════════════════════════
static std::vector<uint8_t> ExtractAlphaChannel(const Image& img)
{
    std::vector<uint8_t> alpha(img.width * img.height);
    for (size_t y = 0; y < img.height; ++y) {
        const uint8_t* row = img.pixels + y * img.rowPitch;
        for (size_t x = 0; x < img.width; ++x)
            alpha[y * img.width + x] = row[x * 4 + 3];
    }
    return alpha;
}

// ═════════════════════════════════════════════════════════════════
// BakeAll — batch bake: one SDK call per unique texture
// ═════════════════════════════════════════════════════════════════

// A triangle reference: which mesh, which local alpha-tri index
struct TriRef {
    uint32_t meshIdx;
    uint32_t alphaLocalIdx; // [0..mesh.alphaTriCount)
};

void OmmBuilder::BakeAll(
    std::vector<MeshGPU>& meshes,
    const MaterialSoA& materials,
    const std::vector<ScratchImage*>& albedoImages)
{
    auto t0 = std::chrono::high_resolution_clock::now();

    // ── 1. Init per-mesh results and collect global triangle groups ──
    struct TexGroup {
        int                   texID;
        float                 threshold;
        XMFLOAT2              uvScale;
        std::vector<TriRef>   tris;
    };
    std::unordered_map<int, TexGroup> globalGroups;

    for (size_t mi = 0; mi < meshes.size(); ++mi) {
        auto& mesh = meshes[mi];
        mesh.ommBake = {};
        mesh.ommBake.alphaTriCount = mesh.alphaTriCount;
        if (mesh.alphaTriCount == 0) continue;

        mesh.ommBake.triOmmIndices.resize(mesh.alphaTriCount);
        // Default all to FULLY_OPAQUE; overwritten below for baked tris
        std::fill(mesh.ommBake.triOmmIndices.begin(),
                  mesh.ommBake.triOmmIndices.end(),
                  D3D12_RAYTRACING_OPACITY_MICROMAP_SPECIAL_INDEX_FULLY_OPAQUE);

        const uint32_t alphaStart = mesh.opaqueTriCount;
        for (uint32_t t = 0; t < mesh.alphaTriCount; ++t) {
            uint32_t triIdx = alphaStart + t;
            uint32_t matID  = mesh.cpuMaterialIDs[triIdx];

            int texID = materials.albedoTexID[matID];
            if (texID < 0 || (uint32_t)texID >= albedoImages.size() || !albedoImages[texID])
                continue; // stays FULLY_OPAQUE

            auto& g = globalGroups[texID];
            if (g.tris.empty()) {
                g.texID     = texID;
                g.threshold = materials.alphaThreshold[matID];
                g.uvScale   = materials.albedoUVScale[matID];
            }
            g.tris.push_back({ (uint32_t)mi, t });
        }
    }

    // Pre-scan: count how many textures have actual alpha variation
    uint32_t texWithAlpha = 0;
    uint32_t texSkipped = 0;
    uint32_t totalAlphaTris = 0;
    for (auto& [texID, group] : globalGroups) {
        const Image* img = albedoImages[texID]->GetImage(0, 0, 0);
        if (!img) { texSkipped++; continue; }
        // Quick alpha range check
        const uint8_t* pixels = img->pixels;
        uint8_t lo = 255, hi = 0;
        for (size_t y = 0; y < img->height && !(lo == 0 && hi == 255); ++y) {
            const uint8_t* row = pixels + y * img->rowPitch;
            for (size_t x = 0; x < img->width; ++x) {
                uint8_t a = row[x * 4 + 3];
                if (a < lo) lo = a;
                if (a > hi) hi = a;
            }
        }
        uint8_t tb = (uint8_t)(group.threshold * 255.0f);
        if (lo >= tb || hi < tb) { texSkipped++; continue; }
        texWithAlpha++;
        totalAlphaTris += (uint32_t)group.tris.size();
    }

    LOG(L"[OMM] " << globalGroups.size() << L" unique textures: "
        << texWithAlpha << L" need baking (" << totalAlphaTris << L" tris), "
        << texSkipped << L" skipped (uniform alpha)");

    // ── 2. Create ONE baker for the entire scene ────────────────
    ommBakerCreationDesc bakerDesc{};
    bakerDesc.type = ommBakerType_CPU;
    ommBaker baker = nullptr;
    if (ommCreateBaker(&bakerDesc, &baker) != ommResult_SUCCESS) {
        LOG(L"[OMM] Failed to create baker");
        return;
    }

    uint32_t totalBakedOMMs = 0;
    size_t   totalBytes = 0;
    uint32_t bakeIdx = 0;

    // ── 3. Bake each texture ONCE ───────────────────────────────
    for (auto& [texID, group] : globalGroups) {
        const Image* img = albedoImages[texID]->GetImage(0, 0, 0);
        if (!img) continue;

        // 3a. Extract alpha (once per texture) and check if it has any variation
        std::vector<uint8_t> alphaData = ExtractAlphaChannel(*img);

        uint8_t minA = 255, maxA = 0;
        for (uint8_t a : alphaData) { if (a < minA) minA = a; if (a > maxA) maxA = a; }
        uint8_t threshByte = (uint8_t)(group.threshold * 255.0f);
        // If entire texture is above or below threshold, no OMM needed
        if (minA >= threshByte || maxA < threshByte)
            continue; // all tris stay FULLY_OPAQUE (default)

        ommCpuTextureMipDesc mip{};
        mip.width       = (uint32_t)img->width;
        mip.height      = (uint32_t)img->height;
        mip.rowPitch    = (uint32_t)img->width;
        mip.textureData = alphaData.data();

        ommCpuTextureDesc texDesc{};
        texDesc.format      = ommCpuTextureFormat_UNORM8;
        texDesc.flags       = ommCpuTextureFlags_None;
        texDesc.mips        = &mip;
        texDesc.mipCount    = 1;
        texDesc.alphaCutoff = group.threshold;

        ommCpuTexture texture = nullptr;
        if (ommCpuCreateTexture(baker, &texDesc, &texture) != ommResult_SUCCESS)
            continue;

        // 3b. Build combined UV + index arrays for ALL triangles using this texture
        std::unordered_map<uint64_t, uint32_t> vertRemap; // (meshIdx<<32|vertIdx) → localIdx
        std::vector<float>    scaledUVs;
        std::vector<uint32_t> localIndices;

        for (const auto& ref : group.tris) {
            auto& mesh = meshes[ref.meshIdx];
            uint32_t triIdx = mesh.opaqueTriCount + ref.alphaLocalIdx;
            for (uint32_t c = 0; c < 3; ++c) {
                uint32_t vi = mesh.cpuIndices[triIdx * 3 + c];
                uint64_t key = ((uint64_t)ref.meshIdx << 32) | vi;
                auto it = vertRemap.find(key);
                uint32_t localVI;
                if (it == vertRemap.end()) {
                    localVI = (uint32_t)(scaledUVs.size() / 2);
                    vertRemap[key] = localVI;
                    const auto& uv = mesh.cpuVertices[vi].texCoord;
                    scaledUVs.push_back(uv.x * group.uvScale.x);
                    scaledUVs.push_back(uv.y * group.uvScale.y);
                } else {
                    localVI = it->second;
                }
                localIndices.push_back(localVI);
            }
        }

        // 3c. Bake
        ++bakeIdx;
        LOG(L"[OMM] [" << bakeIdx << L"/" << texWithAlpha << L"] Baking texID "
            << texID << L": " << group.tris.size() << L" tris, "
            << img->width << L"x" << img->height << L" tex, "
            << (localIndices.size() / 3) << L" unique tris");
        ommCpuBakeInputDesc input = ommCpuBakeInputDescDefault();
        input.bakeFlags              = ommCpuBakeFlags_EnableInternalThreads;
        input.texture                = texture;
        input.runtimeSamplerDesc.addressingMode = ommTextureAddressMode_Wrap;
        input.runtimeSamplerDesc.filter         = ommTextureFilterMode_Linear;
        input.runtimeSamplerDesc.borderAlpha    = 0.0f;
        input.alphaMode              = ommAlphaMode_Test;
        input.texCoordFormat         = ommTexCoordFormat_UV32_FLOAT;
        input.texCoords              = scaledUVs.data();
        input.texCoordStrideInBytes  = 0;
        input.indexFormat            = ommIndexFormat_UINT_32;
        input.indexBuffer            = localIndices.data();
        input.indexCount             = (uint32_t)localIndices.size();
        input.dynamicSubdivisionScale = 0.5f;  // ~0.5 texels per micro-tri (high precision)
        input.rejectionThreshold     = 0.0f;
        input.alphaCutoff            = group.threshold;
        input.format                 = ommFormat_OC1_4_State;
        input.unknownStatePromotion  = ommUnknownStatePromotion_ForceOpaque;
        input.maxSubdivisionLevel    = 8;

        ommCpuBakeResult bakeResult = nullptr;
        ommResult res = ommCpuBake(baker, &input, &bakeResult);
        if (res != ommResult_SUCCESS) {
            LOG(L"[OMM] Bake failed (err=" << (int)res << L") texID=" << texID
                << L" tris=" << group.tris.size());
            ommCpuDestroyTexture(baker, texture);
            continue;
        }

        const ommCpuBakeResultDesc* desc = nullptr;
        ommCpuGetBakeResultDesc(bakeResult, &desc);

        // 3d. Distribute results to each mesh
        // The SDK output has one index per triangle in the combined batch.
        // We need to split this back to per-mesh OmmBakeResult.

        // For each mesh that has tris in this group, we need to:
        //   - Append the OMM array data to the mesh's rawData
        //   - Append the OMM descriptors (with adjusted offsets)
        //   - Map the index buffer entries to the mesh's triOmmIndices

        // Since the SDK deduplicates OMMs, the arrayData + descArray is shared.
        // Each mesh that uses this texture gets a COPY of the shared data,
        // with its own desc base offset. Not ideal for memory but correct.

        // Group tris by mesh for distribution
        std::unordered_map<uint32_t, std::vector<uint32_t>> meshTriPositions;
        for (uint32_t i = 0; i < (uint32_t)group.tris.size(); ++i)
            meshTriPositions[group.tris[i].meshIdx].push_back(i);

        for (auto& [meshIdx, positions] : meshTriPositions) {
            auto& result = meshes[meshIdx].ommBake;

            // Append shared OMM data to this mesh's result
            size_t rawOffset = result.rawData.size();
            result.rawData.insert(result.rawData.end(),
                (const uint8_t*)desc->arrayData,
                (const uint8_t*)desc->arrayData + desc->arrayDataSize);

            uint32_t descBase = (uint32_t)result.ommDescs.size();
            for (uint32_t d = 0; d < desc->descArrayCount; ++d) {
                const auto& sd = desc->descArray[d];
                D3D12_RAYTRACING_OPACITY_MICROMAP_DESC dx{};
                dx.ByteOffset       = (UINT)(sd.offset + rawOffset);
                dx.SubdivisionLevel = sd.subdivisionLevel;
                dx.Format           = (D3D12_RAYTRACING_OPACITY_MICROMAP_FORMAT)sd.format;
                result.ommDescs.push_back(dx);
            }

            // Map indices
            for (uint32_t pos : positions) {
                uint32_t alphaLocal = group.tris[pos].alphaLocalIdx;
                int32_t sdkIdx;

                if (desc->indexFormat == ommIndexFormat_UINT_32)
                    sdkIdx = ((const int32_t*)desc->indexBuffer)[pos];
                else if (desc->indexFormat == ommIndexFormat_UINT_16) {
                    uint16_t v = ((const uint16_t*)desc->indexBuffer)[pos];
                    sdkIdx = (v >= 0xFFFC) ? (int32_t)(int16_t)v : (int32_t)v;
                } else {
                    uint8_t v = ((const uint8_t*)desc->indexBuffer)[pos];
                    sdkIdx = (v >= 0xFC) ? (int32_t)(int8_t)v : (int32_t)v;
                }

                if (sdkIdx < 0)
                    result.triOmmIndices[alphaLocal] = sdkIdx;
                else
                    result.triOmmIndices[alphaLocal] = (int32_t)(sdkIdx + descBase);
            }

            // Merge histogram
            for (uint32_t h = 0; h < desc->descArrayHistogramCount; ++h) {
                const auto& sh = desc->descArrayHistogram[h];
                bool merged = false;
                for (auto& existing : result.histogram) {
                    if (existing.SubdivisionLevel == sh.subdivisionLevel &&
                        existing.Format == (D3D12_RAYTRACING_OPACITY_MICROMAP_FORMAT)sh.format) {
                        existing.Count += sh.count;
                        merged = true;
                        break;
                    }
                }
                if (!merged) {
                    D3D12_RAYTRACING_OPACITY_MICROMAP_HISTOGRAM_ENTRY entry{};
                    entry.Count            = sh.count;
                    entry.SubdivisionLevel = sh.subdivisionLevel;
                    entry.Format           = (D3D12_RAYTRACING_OPACITY_MICROMAP_FORMAT)sh.format;
                    result.histogram.push_back(entry);
                }
            }
        }

        totalBakedOMMs += desc->descArrayCount;
        totalBytes += desc->arrayDataSize;

        if (desc->descArrayCount > 0)
            LOG(L"[OMM] texID " << texID << L": "
                << group.tris.size() << L" tris (across "
                << meshTriPositions.size() << L" meshes), "
                << desc->descArrayCount << L" unique OMMs, "
                << (desc->arrayDataSize / 1024) << L" KB");

        ommCpuDestroyBakeResult(bakeResult);
        ommCpuDestroyTexture(baker, texture);
    }

    ommDestroyBaker(baker);

    auto t1 = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    LOG(L"[OMM] BakeAll done: " << totalBakedOMMs << L" unique OMMs, "
        << (totalBytes / 1024) << L" KB, " << ms << L" ms");
}

// ═════════════════════════════════════════════════════════════════
// BuildGPU — build OMM array and upload index buffer
// ═════════════════════════════════════════════════════════════════
OmmGpuData OmmBuilder::BuildGPU(
    const OmmBakeResult& bake,
    ID3D12Device5* device,
    ID3D12GraphicsCommandList4* cmdList)
{
    OmmGpuData gpu;
    if (bake.empty()) return gpu;

    // 1. Upload OMM index buffer
    {
        UINT64 ibSize = bake.triOmmIndices.size() * sizeof(int32_t);
        auto desc = CD3DX12_RESOURCE_DESC::Buffer(ibSize);
        ThrowIfFailed(device->CreateCommittedResource(
            &nv_helpers_dx12::kUploadHeapProps, D3D12_HEAP_FLAG_NONE,
            &desc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
            IID_PPV_ARGS(&gpu.ommIndexBuffer)));
        gpu.ommIndexBuffer->SetName(L"OMM Index Buffer");
        void* p;
        gpu.ommIndexBuffer->Map(0, nullptr, &p);
        memcpy(p, bake.triOmmIndices.data(), ibSize);
        gpu.ommIndexBuffer->Unmap(0, nullptr);
    }

    if (bake.ommDescs.empty()) {
        gpu.valid = true;
        return gpu;
    }

    // 2. Upload OMM raw data
    {
        UINT64 sz = bake.rawData.size();
        auto desc = CD3DX12_RESOURCE_DESC::Buffer(sz);
        ThrowIfFailed(device->CreateCommittedResource(
            &nv_helpers_dx12::kUploadHeapProps, D3D12_HEAP_FLAG_NONE,
            &desc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
            IID_PPV_ARGS(&gpu._inputBuffer)));
        void* p;
        gpu._inputBuffer->Map(0, nullptr, &p);
        memcpy(p, bake.rawData.data(), sz);
        gpu._inputBuffer->Unmap(0, nullptr);
    }

    // 3. Upload per-OMM descriptors
    {
        UINT64 sz = bake.ommDescs.size() * sizeof(D3D12_RAYTRACING_OPACITY_MICROMAP_DESC);
        auto desc = CD3DX12_RESOURCE_DESC::Buffer(sz);
        ThrowIfFailed(device->CreateCommittedResource(
            &nv_helpers_dx12::kUploadHeapProps, D3D12_HEAP_FLAG_NONE,
            &desc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
            IID_PPV_ARGS(&gpu._descBuffer)));
        void* p;
        gpu._descBuffer->Map(0, nullptr, &p);
        memcpy(p, bake.ommDescs.data(), sz);
        gpu._descBuffer->Unmap(0, nullptr);
    }

    // 4. Query prebuild info
    D3D12_RAYTRACING_OPACITY_MICROMAP_ARRAY_DESC ommArrayDesc{};
    ommArrayDesc.NumOmmHistogramEntries = (UINT)bake.histogram.size();
    ommArrayDesc.pOmmHistogram = bake.histogram.data();
    ommArrayDesc.InputBuffer   = gpu._inputBuffer->GetGPUVirtualAddress();
    ommArrayDesc.PerOmmDescs.StartAddress  = gpu._descBuffer->GetGPUVirtualAddress();
    ommArrayDesc.PerOmmDescs.StrideInBytes = sizeof(D3D12_RAYTRACING_OPACITY_MICROMAP_DESC);

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS prebuildInputs{};
    prebuildInputs.Type    = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_OPACITY_MICROMAP_ARRAY;
    prebuildInputs.NumDescs = 1;
    prebuildInputs.pOpacityMicromapArrayDesc = &ommArrayDesc;
    prebuildInputs.Flags   = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;

    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO prebuildInfo{};
    device->GetRaytracingAccelerationStructurePrebuildInfo(&prebuildInputs, &prebuildInfo);

    // 5. Allocate scratch + result
    {
        UINT64 sz = ROUND_UP(prebuildInfo.ScratchDataSizeInBytes,
                             D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);
        gpu._scratchBuffer = nv_helpers_dx12::CreateBuffer(
            device, sz, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COMMON, nv_helpers_dx12::kDefaultHeapProps);
    }
    {
        UINT64 sz = ROUND_UP(prebuildInfo.ResultDataMaxSizeInBytes,
                             D3D12_RAYTRACING_OPACITY_MICROMAP_ARRAY_BYTE_ALIGNMENT);
        gpu.ommArray = nv_helpers_dx12::CreateBuffer(
            device, sz, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
            nv_helpers_dx12::kDefaultHeapProps);
        gpu.ommArray->SetName(L"OMM Array");
    }

    // 6. Build the OMM array
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC buildDesc{};
    buildDesc.Inputs.Type    = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_OPACITY_MICROMAP_ARRAY;
    buildDesc.Inputs.NumDescs = 1;
    buildDesc.Inputs.pOpacityMicromapArrayDesc = &ommArrayDesc;
    buildDesc.Inputs.Flags   = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
    buildDesc.DestAccelerationStructureData    = gpu.ommArray->GetGPUVirtualAddress();
    buildDesc.ScratchAccelerationStructureData = gpu._scratchBuffer->GetGPUVirtualAddress();

    cmdList->BuildRaytracingAccelerationStructure(&buildDesc, 0, nullptr);

    D3D12_RESOURCE_BARRIER uavBarrier{};
    uavBarrier.Type          = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    uavBarrier.UAV.pResource = gpu.ommArray.Get();
    uavBarrier.Flags         = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    cmdList->ResourceBarrier(1, &uavBarrier);

    gpu.valid = true;
    return gpu;
}
