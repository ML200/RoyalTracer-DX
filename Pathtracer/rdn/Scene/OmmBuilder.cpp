// ═��═════════════════════════════════════════════════════════════════
// Scene/OmmBuilder.cpp — Opacity Micro-Map CPU bake + GPU build
// ═══════════════════════════════════════════════════════════════════

#include "../stdafx.h"
#include "OmmBuilder.h"
#include "Scene.h"
#include "../DXRHelper.h"

using namespace DirectX;

// ─────────────────────────────────────────────────────────────────
// MicroTriCentroid — recursive midpoint subdivision to get the
// centroid (u,v) of micro-triangle `index` in the bird curve at
// the given subdivision level.  Works for both upright and inverted
// sub-triangles because the vertex update is orientation-agnostic.
// ─────────────────────────────────────────────────────────────────
void OmmBuilder::MicroTriCentroid(uint32_t index, uint32_t level,
                                  float& outU, float& outV)
{
    // Start with the full unit triangle: A=(0,0), B=(1,0), C=(0,1)
    // Bird curve child ordering (D3D12/Vulkan):
    //   0: W-corner (A) — upright
    //   1: M-center     — inverted winding (A=AC, B=BC, C=AB)
    //   2: U-corner (B) — upright
    //   3: V-corner (C) — inverted winding (A=BC, B=AC, C=C)
    float ax = 0, ay = 0;
    float bx = 1, by = 0;
    float cx = 0, cy = 1;

    for (int32_t l = (int32_t)level - 1; l >= 0; --l) {
        uint32_t child = (index >> (2 * l)) & 3;

        float abx = (ax + bx) * 0.5f, aby = (ay + by) * 0.5f;
        float bcx = (bx + cx) * 0.5f, bcy = (by + cy) * 0.5f;
        float acx = (ax + cx) * 0.5f, acy = (ay + cy) * 0.5f;

        switch (child) {
            case 0: // W-corner: A stays, B=AB, C=AC
                bx = abx; by = aby;
                cx = acx; cy = acy;
                break;
            case 1: // M-center (inverted): A=AC, B=BC, C=AB
            {
                float ta = acx, tb = acy;
                ax = ta;  ay = tb;
                bx = bcx; by = bcy;
                cx = abx; cy = aby;
                break;
            }
            case 2: // U-corner: A=AB, B stays, C=BC
                ax = abx; ay = aby;
                cx = bcx; cy = bcy;
                break;
            case 3: // V-corner (inverted): A=BC, B=AC, C stays
            {
                ax = bcx; ay = bcy;
                bx = acx; by = acy;
                break;
            }
        }
    }

    outU = (ax + bx + cx) * (1.0f / 3.0f);
    outV = (ay + by + cy) * (1.0f / 3.0f);
}

// ─────────────────────────────────────────────────────────────────
// MicroTriVertices — same subdivision as MicroTriCentroid but returns
// all 3 vertex barycentrics instead of just the centroid.
// ─────────────────────────────────────────────────────────────────
void OmmBuilder::MicroTriVertices(uint32_t index, uint32_t level,
                                  float& v0u, float& v0v,
                                  float& v1u, float& v1v,
                                  float& v2u, float& v2v)
{
    float ax = 0, ay = 0;
    float bx = 1, by = 0;
    float cx = 0, cy = 1;

    for (int32_t l = (int32_t)level - 1; l >= 0; --l) {
        uint32_t child = (index >> (2 * l)) & 3;
        float abx = (ax + bx) * 0.5f, aby = (ay + by) * 0.5f;
        float bcx = (bx + cx) * 0.5f, bcy = (by + cy) * 0.5f;
        float acx = (ax + cx) * 0.5f, acy = (ay + cy) * 0.5f;

        switch (child) {
            case 0: bx = abx; by = aby; cx = acx; cy = acy; break;
            case 1: { float ta = acx, tb = acy; ax = ta; ay = tb;
                      bx = bcx; by = bcy; cx = abx; cy = aby; break; }
            case 2: ax = abx; ay = aby; cx = bcx; cy = bcy; break;
            case 3: { ax = bcx; ay = bcy; bx = acx; by = acy; break; }
        }
    }
    v0u = ax; v0v = ay;
    v1u = bx; v1v = by;
    v2u = cx; v2v = cy;
}

// ─────────────────────────────────────────────────────────────────
// SampleAlpha — bilinear sample of alpha channel from RGBA8 image.
// UVs are wrapped to [0,1).
// ─────────────────────────────────────────────────────────────────
float OmmBuilder::SampleAlpha(const Image& img, float u, float v)
{
    // Wrap UVs
    u = u - floorf(u);
    v = v - floorf(v);

    float fx = u * (img.width  - 1);
    float fy = v * (img.height - 1);

    int x0 = (int)fx;
    int y0 = (int)fy;
    int x1 = (x0 + 1) % (int)img.width;   // wrap for tiling textures
    int y1 = (y0 + 1) % (int)img.height;

    float sx = fx - x0;
    float sy = fy - y0;

    auto fetchA = [&](int x, int y) -> float {
        return img.pixels[y * img.rowPitch + x * 4 + 3] / 255.0f;
    };

    float a00 = fetchA(x0, y0);
    float a10 = fetchA(x1, y0);
    float a01 = fetchA(x0, y1);
    float a11 = fetchA(x1, y1);

    return (a00 * (1 - sx) + a10 * sx) * (1 - sy)
         + (a01 * (1 - sx) + a11 * sx) * sy;
}

// ═════════════════════════════════════════════════════════════════
// BakeMesh — CPU-side OMM classification
// ═════════════════════════════════════════════════════════════════
OmmBakeResult OmmBuilder::BakeMesh(
    const MeshGPU& mesh,
    const std::vector<Material>& materials,
    const std::vector<ScratchImage*>& albedoImages,
    const std::vector<XMFLOAT2>& albedoUVScales)
{
    OmmBakeResult result;
    result.alphaTriCount = mesh.alphaTriCount;
    if (result.alphaTriCount == 0) return result;

    const uint32_t alphaStart = mesh.opaqueTriCount; // first alpha tri in index buffer
    result.triOmmIndices.resize(mesh.alphaTriCount);

    // Track histogram: how many OMMs at each (subdivLevel, format)
    uint32_t bakedOmmCount = 0;
    uint32_t specialOpaqueCount = 0;
    uint32_t specialTransparentCount = 0;

    for (uint32_t t = 0; t < mesh.alphaTriCount; ++t) {
        uint32_t triIdx = alphaStart + t;
        uint32_t matID  = mesh.cpuMaterialIDs[triIdx];
        const Material& mat = materials[matID];

        // Get the alpha texture image (mip 0)
        const Image* alphaImg = nullptr;
        XMFLOAT2 uvScale = mat.albedoUVScale; // use per-material scale, not per-texture
        if (mat.albedoTexID >= 0 && (uint32_t)mat.albedoTexID < albedoImages.size()
            && albedoImages[mat.albedoTexID] != nullptr)
        {
            alphaImg = albedoImages[mat.albedoTexID]->GetImage(0, 0, 0);
        }

        // No texture → treat entire triangle as opaque
        if (!alphaImg) {
            result.triOmmIndices[t] = D3D12_RAYTRACING_OPACITY_MICROMAP_SPECIAL_INDEX_FULLY_OPAQUE;
            specialOpaqueCount++;
            continue;
        }

        // Get triangle vertex UVs
        uint32_t i0 = mesh.cpuIndices[triIdx * 3 + 0];
        uint32_t i1 = mesh.cpuIndices[triIdx * 3 + 1];
        uint32_t i2 = mesh.cpuIndices[triIdx * 3 + 2];

        XMFLOAT2 uv0 = mesh.cpuVertices[i0].texCoord;
        XMFLOAT2 uv1 = mesh.cpuVertices[i1].texCoord;
        XMFLOAT2 uv2 = mesh.cpuVertices[i2].texCoord;

        // Apply UV scale
        uv0.x *= uvScale.x; uv0.y *= uvScale.y;
        uv1.x *= uvScale.x; uv1.y *= uvScale.y;
        uv2.x *= uvScale.x; uv2.y *= uvScale.y;

        float threshold = mat.alphaThreshold;

        // ── Phase 1: quick whole-triangle classification ────────
        // Sample vertices of coarse subdivision level 2 (16 micro-tris,
        // ~15 unique vertices × 3 corners each = up to 48 samples).
        // Much better coverage than the previous 7-point heuristic.
        bool allOpaque = true, allTransparent = true;
        {
            static constexpr uint32_t kCoarseLevel = 2;
            static constexpr uint32_t kCoarseCount = 16; // 4^2
            bool sawOpaque = false, sawTransparent = false;
            for (uint32_t ci = 0; ci < kCoarseCount && !(sawOpaque && sawTransparent); ++ci) {
                float va0, vb0, va1, vb1, va2, vb2;
                MicroTriVertices(ci, kCoarseLevel, va0, vb0, va1, vb1, va2, vb2);
                float pts[][2] = { {va0,vb0}, {va1,vb1}, {va2,vb2} };
                for (auto& [bu, bv] : pts) {
                    float bw = 1.0f - bu - bv;
                    float su = uv0.x * bw + uv1.x * bu + uv2.x * bv;
                    float sv = uv0.y * bw + uv1.y * bu + uv2.y * bv;
                    float a = SampleAlpha(*alphaImg, su, sv);
                    if (a >= threshold) sawOpaque = true;
                    else                sawTransparent = true;
                }
            }
            allOpaque      = sawOpaque && !sawTransparent;
            allTransparent = !sawOpaque && sawTransparent;
        }

        if (allOpaque) {
            result.triOmmIndices[t] = D3D12_RAYTRACING_OPACITY_MICROMAP_SPECIAL_INDEX_FULLY_OPAQUE;
            specialOpaqueCount++;
            continue;
        }
        if (allTransparent) {
            result.triOmmIndices[t] = D3D12_RAYTRACING_OPACITY_MICROMAP_SPECIAL_INDEX_FULLY_TRANSPARENT;
            specialTransparentCount++;
            continue;
        }

        // ── Phase 2: per-microtriangle conservative classification ──
        // Sample at all 3 corners + centroid of each micro-tri.
        // If samples disagree across the threshold → UNKNOWN.
        uint32_t ommIdx = bakedOmmCount++;
        result.triOmmIndices[t] = (int32_t)ommIdx;

        D3D12_RAYTRACING_OPACITY_MICROMAP_DESC desc{};
        desc.ByteOffset     = (UINT)(result.rawData.size());
        desc.SubdivisionLevel = OMM_SUBDIV_LEVEL;
        desc.Format         = D3D12_RAYTRACING_OPACITY_MICROMAP_FORMAT_OC1_4_STATE;
        result.ommDescs.push_back(desc);

        // Allocate space for this OMM's micro-tri data
        size_t dataStart = result.rawData.size();
        result.rawData.resize(dataStart + OMM_BYTES_PER_OMM, 0);

        for (uint32_t mi = 0; mi < OMM_MICROTRIS_PER_TRI; ++mi) {
            float va0, vb0, va1, vb1, va2, vb2;
            MicroTriVertices(mi, OMM_SUBDIV_LEVEL, va0, vb0, va1, vb1, va2, vb2);
            float cu = (va0 + va1 + va2) * (1.0f / 3.0f);
            float cv = (vb0 + vb1 + vb2) * (1.0f / 3.0f);

            // Sample at 3 corners + 3 edge midpoints + centroid (7 points)
            float e01u = (va0+va1)*0.5f, e01v = (vb0+vb1)*0.5f;
            float e12u = (va1+va2)*0.5f, e12v = (vb1+vb2)*0.5f;
            float e20u = (va2+va0)*0.5f, e20v = (vb2+vb0)*0.5f;
            float pts[][2] = {
                {va0,vb0}, {va1,vb1}, {va2,vb2},
                {e01u,e01v}, {e12u,e12v}, {e20u,e20v},
                {cu,cv}
            };
            float minA = 1.0f, maxA = 0.0f;
            for (auto& [bu, bv] : pts) {
                float bw = 1.0f - bu - bv;
                float su = uv0.x * bw + uv1.x * bu + uv2.x * bv;
                float sv = uv0.y * bw + uv1.y * bu + uv2.y * bv;
                float a = SampleAlpha(*alphaImg, su, sv);
                minA = (std::min)(minA, a);
                maxA = (std::max)(maxA, a);
            }

            // Conservative classify
            uint8_t state;
            bool anyAbove = (maxA >= threshold);
            bool anyBelow = (minA <  threshold);
            float margin = 0.05f;

            if (anyAbove && anyBelow) {
                // Mixed: samples straddle threshold → UNKNOWN
                float mid = (minA + maxA) * 0.5f;
                state = (mid >= threshold)
                    ? D3D12_RAYTRACING_OPACITY_MICROMAP_STATE_UNKNOWN_OPAQUE
                    : D3D12_RAYTRACING_OPACITY_MICROMAP_STATE_UNKNOWN_TRANSPARENT;
            } else if (!anyBelow) {
                // All >= threshold; use margin for extra safety
                state = (minA >= threshold + margin)
                    ? D3D12_RAYTRACING_OPACITY_MICROMAP_STATE_OPAQUE
                    : D3D12_RAYTRACING_OPACITY_MICROMAP_STATE_UNKNOWN_OPAQUE;
            } else {
                // All < threshold
                state = (maxA < threshold - margin)
                    ? D3D12_RAYTRACING_OPACITY_MICROMAP_STATE_TRANSPARENT
                    : D3D12_RAYTRACING_OPACITY_MICROMAP_STATE_UNKNOWN_TRANSPARENT;
            }

            // Pack 2-bit state into the byte array (4 microtris per byte)
            uint32_t byteIdx = mi / 4;
            uint32_t bitShift = (mi % 4) * 2;
            result.rawData[dataStart + byteIdx] |= (state << bitShift);
        }
    }

    // ── Build histogram ─────────────────────────────────────────
    if (bakedOmmCount > 0) {
        D3D12_RAYTRACING_OPACITY_MICROMAP_HISTOGRAM_ENTRY entry{};
        entry.Count            = bakedOmmCount;
        entry.SubdivisionLevel = OMM_SUBDIV_LEVEL;
        entry.Format           = D3D12_RAYTRACING_OPACITY_MICROMAP_FORMAT_OC1_4_STATE;
        result.histogram.push_back(entry);
    }

    LOG(L"[OMM] Mesh baked: " << mesh.alphaTriCount << L" alpha tris → "
        << specialOpaqueCount << L" opaque, "
        << specialTransparentCount << L" transparent, "
        << bakedOmmCount << L" baked OMMs ("
        << (result.rawData.size() / 1024) << L" KB)");

    return result;
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

    // ── 1. Upload OMM index buffer ──────────────────────────────
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

    // If there are no baked OMMs (all triangles got special indices),
    // we still need the index buffer but no OMM array.
    if (bake.ommDescs.empty()) {
        gpu.valid = true;
        return gpu;
    }

    // ── 2. Upload OMM raw data to GPU ───────────────────────────
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

    // ── 3. Upload per-OMM descriptors ───────────────────────────
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

    // ── 4. Query prebuild info for OMM array ────────────────────
    // pOmmHistogram is always a CPU pointer (both for prebuild and build).
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

    // ── 5. Allocate scratch + result ────────────────────────────
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

    // ── 6. Build the OMM array ──────────────────────────────────
    // pOmmHistogram stays a CPU pointer — the runtime reads it during the call.
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC buildDesc{};
    buildDesc.Inputs.Type    = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_OPACITY_MICROMAP_ARRAY;
    buildDesc.Inputs.NumDescs = 1;
    buildDesc.Inputs.pOpacityMicromapArrayDesc = &ommArrayDesc;
    buildDesc.Inputs.Flags   = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
    buildDesc.DestAccelerationStructureData    = gpu.ommArray->GetGPUVirtualAddress();
    buildDesc.ScratchAccelerationStructureData = gpu._scratchBuffer->GetGPUVirtualAddress();

    cmdList->BuildRaytracingAccelerationStructure(&buildDesc, 0, nullptr);

    // UAV barrier on the OMM array
    D3D12_RESOURCE_BARRIER uavBarrier{};
    uavBarrier.Type          = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    uavBarrier.UAV.pResource = gpu.ommArray.Get();
    uavBarrier.Flags         = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    cmdList->ResourceBarrier(1, &uavBarrier);

    gpu.valid = true;
    return gpu;
}
