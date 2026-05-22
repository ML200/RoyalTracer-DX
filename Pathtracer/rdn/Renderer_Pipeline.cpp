//====================================
//RENDERER PIPELINE
//====================================
//accel structures, RT pipeline, descriptor heap, SBT, LUT, compaction, readback
//split from Renderer.cpp, same TU

#include "stdafx.h"
#include "Renderer.h"
#include "ReuseTextureGen.h"
#include "Scene/OmmBuilder.h"
#include "nv_helpers_dx12/BottomLevelASGenerator.h"
#include "nv_helpers_dx12/RaytracingPipelineGenerator.h"
#include "nv_helpers_dx12/RootSignatureGenerator.h"
#include <algorithm>
#include <cmath>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <random>
#include <d3dcompiler.h>
#include <DirectXPackedVector.h>
#include "../DirectXTex/DirectXTex.h"
//Must match the TINYEXR_USE_* defines in ThirdPartyImpl.cpp so the extern
//declarations and the implementation agree on zlib backend.
#define TINYEXR_USE_MINIZ    0
#define TINYEXR_USE_STB_ZLIB 1
#include "../lib/tinyexr/tinyexr.h"

//====================================
//ACCELERATION STRUCTURES
//====================================

static constexpr bool kUseBlasCompaction = true;

Renderer::AccelerationStructureBuffers
Renderer::CreateBottomLevelAS(
    std::vector<std::pair<ComPtr<ID3D12Resource>, uint32_t>> vVertexBuffers,
    std::vector<std::pair<ComPtr<ID3D12Resource>, uint32_t>> vIndexBuffers,
    UINT opaqueTriCount, UINT alphaTriCount,
    MeshGPU* meshOmm)
{
    std::wcout << L"[BLAS]   CreateBottomLevelAS ENTER: vbufs=" << vVertexBuffers.size()
               << L" ibufs=" << vIndexBuffers.size()
               << L" opaqueTris=" << opaqueTriCount
               << L" alphaTris=" << alphaTriCount
               << L" omm=" << (meshOmm && meshOmm->hasOmm ? L"yes" : L"no")
               << std::endl;
    for (size_t i = 0; i < vVertexBuffers.size(); ++i) {
        std::wcout << L"[BLAS]     geom[" << i << L"] verts=" << vVertexBuffers[i].second
                   << L" vbufPtr=" << vVertexBuffers[i].first.Get()
                   << L" ibufPtr=" << vIndexBuffers[i].first.Get()
                   << std::endl;
    }

    nv_helpers_dx12::BottomLevelASGenerator blasGen;

    for (size_t i = 0; i < vVertexBuffers.size(); i++) {
        UINT opaqueIdxCount = opaqueTriCount * 3;
        UINT alphaIdxCount  = alphaTriCount  * 3;

        if (opaqueIdxCount > 0)
            blasGen.AddVertexBuffer(
                vVertexBuffers[i].first.Get(), 0, vVertexBuffers[i].second,
                sizeof(Vertex), vIndexBuffers[i].first.Get(), 0,
                opaqueIdxCount, nullptr, 0, true);

        if (alphaIdxCount > 0) {
            if (meshOmm && meshOmm->hasOmm) {
                // Use OMM-linked geometry for alpha triangles
                blasGen.AddVertexBufferWithOMM(
                    vVertexBuffers[i].first.Get(), 0, vVertexBuffers[i].second,
                    sizeof(Vertex), vIndexBuffers[i].first.Get(),
                    opaqueIdxCount * sizeof(UINT), alphaIdxCount,
                    nullptr, 0,
                    meshOmm->ommArray ? meshOmm->ommArray->GetGPUVirtualAddress() : 0,
                    meshOmm->ommIndexBuffer->GetGPUVirtualAddress(),
                    meshOmm->ommBake.alphaTriCount);
            } else {
                // Fallback: regular non-opaque geometry (any-hit shader for all)
                blasGen.AddVertexBuffer(
                    vVertexBuffers[i].first.Get(), 0, vVertexBuffers[i].second,
                    sizeof(Vertex), vIndexBuffers[i].first.Get(),
                    opaqueIdxCount * sizeof(UINT), alphaIdxCount, nullptr, 0, false);
            }
        }
    }

    UINT64 scratchSize = 0, resultSize = 0;
    auto buildFlags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
    if constexpr (kUseBlasCompaction)
        buildFlags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_COMPACTION;

    blasGen.ComputeASBufferSizes(m_ctx.Device(), buildFlags, &scratchSize, &resultSize);
    std::wcout << L"[BLAS]     sizes scratch=" << (scratchSize / (1024 * 1024)) << L" MB"
               << L" result=" << (resultSize / (1024 * 1024)) << L" MB" << std::endl;

    AccelerationStructureBuffers buffers;
    buffers.pScratch = nv_helpers_dx12::CreateBuffer(m_ctx.Device(), scratchSize,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON,
        nv_helpers_dx12::kDefaultHeapProps);
    std::wcout << L"[BLAS]     scratch buf created (" << buffers.pScratch.Get() << L")" << std::endl;

    if constexpr (!kUseBlasCompaction) {
        buffers.pResult = nv_helpers_dx12::CreateBuffer(m_ctx.Device(), resultSize,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
            nv_helpers_dx12::kDefaultHeapProps);
        blasGen.Generate(m_ctx.CmdList(), buffers.pScratch.Get(), buffers.pResult.Get(), false, nullptr);
    } else {
        // Build uncompacted, read back size, compact
        buffers.pResultUncompacted = nv_helpers_dx12::CreateBuffer(m_ctx.Device(), resultSize,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
            nv_helpers_dx12::kDefaultHeapProps);
        std::wcout << L"[BLAS]     uncompacted result buf created (" << buffers.pResultUncompacted.Get() << L")" << std::endl;

        ComPtr<ID3D12Resource> compactedSizeBuf = nv_helpers_dx12::CreateBuffer(
            m_ctx.Device(), sizeof(UINT64), D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nv_helpers_dx12::kDefaultHeapProps);

        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC postInfo = {};
        postInfo.DestBuffer = compactedSizeBuf->GetGPUVirtualAddress();
        postInfo.InfoType   = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_COMPACTED_SIZE;

        std::wcout << L"[BLAS]     calling blasGen.Generate (uncompacted build)..." << std::endl;
        blasGen.Generate(m_ctx.CmdList(), buffers.pScratch.Get(),
            buffers.pResultUncompacted.Get(), false, nullptr);
        std::wcout << L"[BLAS]     blasGen.Generate returned" << std::endl;

        D3D12_GPU_VIRTUAL_ADDRESS src = buffers.pResultUncompacted->GetGPUVirtualAddress();
        m_ctx.CmdList()->EmitRaytracingAccelerationStructurePostbuildInfo(&postInfo, 1, &src);

        ComPtr<ID3D12Resource> readback = nv_helpers_dx12::CreateBuffer(
            m_ctx.Device(), sizeof(UINT64), D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_COPY_DEST, nv_helpers_dx12::kReadbackHeapProps);

        auto barrier = CD3DX12_RESOURCE_BARRIER::Transition(
            compactedSizeBuf.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_SOURCE);
        m_ctx.CmdList()->ResourceBarrier(1, &barrier);
        m_ctx.CmdList()->CopyResource(readback.Get(), compactedSizeBuf.Get());

        std::wcout << L"[BLAS]     FlushAndReset before reading compacted size..." << std::endl;
        m_ctx.FlushAndReset();
        std::wcout << L"[BLAS]     FlushAndReset returned" << std::endl;

        UINT64 compactedSize;
        void* pMap;
        ThrowIfFailed(readback->Map(0, nullptr, &pMap));
        memcpy(&compactedSize, pMap, sizeof(UINT64));
        readback->Unmap(0, nullptr);
        std::wcout << L"[BLAS]     compactedSize=" << (compactedSize / (1024 * 1024)) << L" MB" << std::endl;

        buffers.pResult = nv_helpers_dx12::CreateBuffer(m_ctx.Device(), compactedSize,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
            nv_helpers_dx12::kDefaultHeapProps);
        buffers.pResult->SetName(L"Compacted BLAS");

        m_ctx.CmdList()->CopyRaytracingAccelerationStructure(
            buffers.pResult->GetGPUVirtualAddress(),
            buffers.pResultUncompacted->GetGPUVirtualAddress(),
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_COMPACT);

        auto uavB = CD3DX12_RESOURCE_BARRIER::UAV(buffers.pResult.Get());
        m_ctx.CmdList()->ResourceBarrier(1, &uavB);
    }
    std::wcout << L"[BLAS]   CreateBottomLevelAS EXIT: result=" << buffers.pResult.Get() << std::endl;
    return buffers;
}

// PLANET_INTEGRATION: TLAS build + buffer allocation. pScratch/pResult/pInstanceDesc
// are reallocated on every full rebuild. Phase 5's per-frame rebuild needs them
// pre-sized to a max instance count so no per-frame CreateBuffer happens on the hot path.
void Renderer::CreateTopLevelAS(
    const std::vector<Scene::TLASInstance>& instances,
    bool updateOnly)
{
    std::wcout << L"[TLAS] CreateTopLevelAS ENTER: instances=" << instances.size()
               << L" updateOnly=" << (updateOnly ? L"yes" : L"no") << std::endl;

    if (!updateOnly) {
        //Per-instance logging suits a handful of GLB instances; a large scene
        //has too many to log without stalling on console I/O, so gate it.
        const bool logEachInstance = instances.size() <= 32;
        for (size_t i = 0; i < instances.size(); i++) {
            if (logEachInstance)
                std::wcout << L"[TLAS]   instance[" << i << L"] blas=" << instances[i].blas.Get()
                           << L" hitGroup=" << instances[i].hitGroupContribution
                           << L" flags=" << (UINT)instances[i].flags << std::endl;
            m_topLevelASGenerator.AddInstance(
                instances[i].blas.Get(), instances[i].transform,
                static_cast<UINT>(i), instances[i].hitGroupContribution,
                instances[i].flags);
        }

        UINT64 scratchSize, resultSize, instanceDescsSize;
        m_topLevelASGenerator.ComputeASBufferSizes(
            m_ctx.Device(), true, &scratchSize, &resultSize, &instanceDescsSize);
        std::wcout << L"[TLAS]   sizes scratch=" << (scratchSize / 1024) << L" KB"
                   << L" result=" << (resultSize / 1024) << L" KB"
                   << L" instanceDescs=" << instanceDescsSize << L" B" << std::endl;

        m_topLevelASBuffers.pScratch = nv_helpers_dx12::CreateBuffer(
            m_ctx.Device(), scratchSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nv_helpers_dx12::kDefaultHeapProps);
        m_topLevelASBuffers.pResult = nv_helpers_dx12::CreateBuffer(
            m_ctx.Device(), resultSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
            nv_helpers_dx12::kDefaultHeapProps);
        m_topLevelASBuffers.pInstanceDesc = nv_helpers_dx12::CreateBuffer(
            m_ctx.Device(), instanceDescsSize, D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_GENERIC_READ, nv_helpers_dx12::kUploadHeapProps);
    }

    std::wcout << L"[TLAS]   calling m_topLevelASGenerator.Generate..." << std::endl;
    m_topLevelASGenerator.Generate(m_ctx.CmdList(),
        m_topLevelASBuffers.pScratch.Get(), m_topLevelASBuffers.pResult.Get(),
        m_topLevelASBuffers.pInstanceDesc.Get(), updateOnly,
        m_topLevelASBuffers.pResult.Get());
    std::wcout << L"[TLAS]   Generate returned. TLAS=" << m_topLevelASBuffers.pResult.Get() << std::endl;
}

void Renderer::CreateAccelerationStructures() {
    std::wcout << L"[AS] CreateAccelerationStructures ENTER: meshes=" << m_scene.meshes.size()
               << L" instances=" << m_scene.instances.size() << std::endl;

    // One BLAS per unique mesh (skip if already built, e.g. procedural meshes)
    for (size_t m = 0; m < m_scene.meshes.size(); ++m) {
        auto& mesh = m_scene.meshes[m];
        std::wcout << L"[AS] mesh[" << m << L"/" << m_scene.meshes.size() << L"] "
                   << L"verts=" << mesh.vertexCount
                   << L" indices=" << mesh.indexCount
                   << L" opaqueTris=" << mesh.opaqueTriCount
                   << L" alphaTris=" << mesh.alphaTriCount
                   << L" totalTris=" << (mesh.opaqueTriCount + mesh.alphaTriCount)
                   << std::endl;
        if (mesh.blas) {
            std::wcout << L"[AS]   already has BLAS, skipping" << std::endl;
            continue;
        }

        // Build OMM array on GPU if bake data exists.
        // ommGpu must outlive CreateBottomLevelAS (which flushes the cmd list).
        OmmGpuData ommGpu;
        if (!mesh.ommBake.triOmmIndices.empty()) {
            ommGpu = OmmBuilder::BuildGPU(
                mesh.ommBake, m_ctx.Device(), m_ctx.CmdList());
            if (ommGpu.valid) {
                mesh.ommArray       = ommGpu.ommArray;
                mesh.ommIndexBuffer = ommGpu.ommIndexBuffer;
                mesh.hasOmm         = true;
            }
        }

        MeshGPU* ommPtr = mesh.hasOmm ? &mesh : nullptr;
        auto buffers = CreateBottomLevelAS(
            {{ mesh.vertexBuffer.Get(), mesh.vertexCount }},
            {{ mesh.indexBuffer.Get(),  mesh.indexCount  }},
            mesh.opaqueTriCount, mesh.alphaTriCount, ommPtr);
        mesh.blas = buffers.pResult;
        std::wcout << L"[AS]   mesh[" << m << L"] BLAS=" << mesh.blas.Get() << std::endl;
    }
    std::wcout << L"[AS] All BLAS built. Rebuilding TLAS instance list..." << std::endl;

    // TLAS from scene instances
    m_scene.RebuildTLASInstanceList();
    std::wcout << L"[AS] tlasInstances.size()=" << m_scene.tlasInstances.size() << std::endl;
    CreateTopLevelAS(m_scene.tlasInstances);
    std::wcout << L"[AS] TLAS built" << std::endl;

    // Emissive triangles & light tree
    m_scene.CollectEmissiveTriangles();

    std::vector<InstanceXformCPU> ltXforms;
    ltXforms.reserve(m_scene.instances.size());
    //Apply floating origin shift (same as BuildXformsFromScene in Renderer.cpp)
    //so light tree positions match shifted ray origins. At init time the
    //scene origin is (0,0,0) so this is a no-op, but kept symmetric for
    //correctness if init ever runs after a camera shift.
    const XMVECTOR ltShift = XMVectorSet(m_scene.sceneOriginWorld.x,
                                          m_scene.sceneOriginWorld.y,
                                          m_scene.sceneOriginWorld.z, 0.0f);
    for (const auto& si : m_scene.instances) {
        InstanceXformCPU x{};
        XMMATRIX shifted = si.worldTransform;
        shifted.r[3] = XMVectorSubtract(shifted.r[3], ltShift);
        XMStoreFloat4x4(&x.objectToWorld, shifted);
        ltXforms.push_back(x);
    }
    m_lightTree.Build(m_scene.emissiveTriangles, ltXforms);
    m_lightTree.PrintMetrics();

    { SCOPE_TIMER("LightTree.UploadAll");
      m_lightTree.UploadAll(m_ctx.Device(), m_ctx.CmdList()); }

    { SCOPE_TIMER("CreateEmissiveTrianglesBuffer");
      m_scene.CreateEmissiveTrianglesBuffer(
          m_ctx.Device(), m_ctx.CmdList(), m_ctx.CmdQueue(),
          nullptr /* alloc unused in new API */); }

    { SCOPE_TIMER("CreateTriToLightIdBuffer");
      m_scene.CreateTriToLightIdBuffer(m_ctx.Device(), m_ctx.CmdList()); }

    m_ctx.FlushAndReset();
    m_lightTree.ReleaseStaging();

    // The mesh.vertexBuffer/indexBuffer pair is only consumed by BLAS construction.
    // Once the BLAS exists, shaders sample geometry via vertexGlobal/indexGlobal
    // and these copies are dead VRAM. cpuVertices/cpuIndices stay around so a
    // future BLAS rebuild can recreate the upload buffers from CPU data.
    {
        size_t freed = 0;
        for (auto& mesh : m_scene.meshes) {
            if (mesh.vertexBuffer) { mesh.vertexBuffer.Reset(); ++freed; }
            if (mesh.indexBuffer)  { mesh.indexBuffer.Reset(); }
        }
        LOG(L"[AS] Released per-mesh source buffers on " << freed << L" meshes");
    }
}

//====================================
//ROOT SIGNATURES
//====================================

ComPtr<ID3D12RootSignature> Renderer::CreateRayGenSignature() {
    CD3DX12_ROOT_PARAMETER1 rootParameters[2];
    std::vector<CD3DX12_DESCRIPTOR_RANGE1> ranges;
    ranges.reserve(40);
    const auto VOLATILE = D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE;
    const auto STATIC   = D3D12_DESCRIPTOR_RANGE_FLAG_DATA_STATIC_WHILE_SET_AT_EXECUTE;

    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0, 0, VOLATILE, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 1, 0, VOLATILE, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, STATIC,   D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 1, 0, STATIC,   D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 2, 0, STATIC,   D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_CBV, 1, 0, 0, STATIC,   D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 3, 0, STATIC,   D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 4, 0, STATIC,   D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 5, 0, STATIC,   D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 6, 0, STATIC,   D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    for (UINT u = 2; u <= 7; ++u)
        ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, u, 0, VOLATILE, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 7, 0, STATIC,   D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 8, 0, STATIC,   D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 8, 0, VOLATILE, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 9, 0, VOLATILE, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    for (UINT t = 9; t <= 12; ++t)
        ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, t, 0, STATIC, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 15, 0, STATIC,  D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    for (UINT t = 16; t <= 18; ++t)
        ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, t, 0, STATIC, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    for (UINT t = 30; t <= 33; ++t)
        ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, t, 0, STATIC, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 10, 0, VOLATILE, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 34, 0, VOLATILE, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 35, 0, VOLATILE, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 4, 36, 0, VOLATILE, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 13, 11, 0, VOLATILE, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 3, 60, 0, VOLATILE, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    // Paired reuse textures (t19, t20, t21) — 3 Texture2D<int2> SRVs at heap slots 55..57
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 3, 19, 0, STATIC, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    // NRC shared D3D12/CUDA UAVs (u40..u44) at heap slots 58..62 — see
    // shaders/Nrc_v8.hlsli for binding order.
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 5, 40, 0, VOLATILE, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    // Auto-expose persistent state (u24) at heap slot 63
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 24, 0, VOLATILE, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    // Star / Milky Way skybox texture (t40, Includes_v8.hlsli) at heap slot 64
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 40, 0, STATIC,   D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    // Volumetric cloud noise 3D texture (t42, g_cloudNoise in Includes_v8.hlsli)
    // at heap slot CLOUD_NOISE_HEAP_SLOT (65). Baked once by BakeCloudNoiseTexture
    // — see Pass_cloudnoise_bake_v8.hlsl.
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 42, 0, STATIC,   D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    // Planet-scale cloud coverage 2D texture (t43, g_cloudCoverage in
    // Includes_v8.hlsli) at heap slot CLOUD_COVERAGE_HEAP_SLOT (66). NASA
    // Blue Marble equirectangular 8192×4096 luminance map, loaded once by
    // InitCloudCoverageTexture.
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 43, 0, STATIC,   D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    // Spatiotemporal blue noise Texture2DArray (t41, g_cloudSTBN in
    // Clouds_v8.hlsli) at heap slot CLOUD_STBN_HEAP_SLOT (67). 128×128×64
    // RGBA8 baked once by BakeCloudSTBNTexture (Pass_stbn_bake_v8.hlsl).
    // Placed last in the descriptor table because t41 was added after
    // t42 / t43 were already wired up; the register binding is what the
    // shader sees, the heap order doesn't matter.
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 41, 0, STATIC,   D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    // PLANET_INTEGRATION: terrain instance table StructuredBuffer (t44,
    // g_terrainTable in Includes_v8.hlsli) at heap slot TERRAIN_TABLE_HEAP_SLOT
    // (68). Refilled each frame by the planet StreamOrchestrator; the terrain
    // shader reads it to reconstruct chunk vertices and drop stale GI samples.
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 44, 0, STATIC,   D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);

    rootParameters[0].InitAsDescriptorTable((UINT)ranges.size(), ranges.data(), D3D12_SHADER_VISIBILITY_ALL);
    // 24 ReSTIR constants [0..23] + 8 NRC control constants [24..31] = 32.
    // (Last 5 of the NRC block are scene-bounds normalization for the
    // position input, see Includes_v8.hlsli.)
    rootParameters[1].InitAsConstants(32, 1, 0, D3D12_SHADER_VISIBILITY_ALL);

    CD3DX12_STATIC_SAMPLER_DESC staticSamplers[3];
    staticSamplers[0].Init(0, D3D12_FILTER_ANISOTROPIC,
        D3D12_TEXTURE_ADDRESS_MODE_WRAP, D3D12_TEXTURE_ADDRESS_MODE_WRAP,
        D3D12_TEXTURE_ADDRESS_MODE_WRAP);
    staticSamplers[0].MaxAnisotropy = 16;
    staticSamplers[1].Init(1, D3D12_FILTER_MIN_MAG_MIP_LINEAR,
        D3D12_TEXTURE_ADDRESS_MODE_CLAMP, D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        D3D12_TEXTURE_ADDRESS_MODE_CLAMP);
    // s2: plain bilinear + WRAP, no anisotropy. Used for the cloud
    // coverage equirectangular map (Includes_v8.hlsli → g_samplerLinearWrap).
    // The atan2-derived U produces large UV derivatives at the
    // longitude=±180° meridian — an anisotropic filter sees that as a
    // huge footprint and selects a coarse mip / mis-computes the
    // sample footprint, producing a visible seam streak. A pure
    // bilinear sampler computes only the in-mip 4-tap bilinear
    // average, so the seam disappears (WRAP addressing handles the
    // texel-level wrap correctly without aniso confusion).
    staticSamplers[2].Init(2, D3D12_FILTER_MIN_MAG_LINEAR_MIP_POINT,
        D3D12_TEXTURE_ADDRESS_MODE_WRAP, D3D12_TEXTURE_ADDRESS_MODE_WRAP,
        D3D12_TEXTURE_ADDRESS_MODE_WRAP);

    CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC desc;
    desc.Init_1_1(_countof(rootParameters), rootParameters, _countof(staticSamplers),
        staticSamplers, D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED);

    ComPtr<ID3DBlob> signature, error;
    HRESULT hr = D3D12SerializeVersionedRootSignature(&desc, &signature, &error);
    if (FAILED(hr)) {
        if (error) OutputDebugStringA(static_cast<char*>(error->GetBufferPointer()));
        ThrowIfFailed(hr);
    }

    ComPtr<ID3D12RootSignature> rs;
    ThrowIfFailed(m_ctx.Device()->CreateRootSignature(0,
        signature->GetBufferPointer(), signature->GetBufferSize(), IID_PPV_ARGS(&rs)));
    return rs;
}

ComPtr<ID3D12RootSignature> Renderer::CreateComputeSignature() {
    return CreateRayGenSignature();  // identical layout
}

ComPtr<ID3D12RootSignature> Renderer::CreateHitSignature() {
    nv_helpers_dx12::RootSignatureGenerator rsc;
    return rsc.Generate(m_ctx.Device(), true);
}

ComPtr<ID3D12RootSignature> Renderer::CreateMissSignature() {
    nv_helpers_dx12::RootSignatureGenerator rsc;
    return rsc.Generate(m_ctx.Device(), true);
}

//====================================
//RAYTRACING PIPELINE
//====================================

void Renderer::CreateRaytracingPipeline() {
    nv_helpers_dx12::RayTracingPipelineGenerator pipeline(m_ctx.Device());

    m_rayGenSignature  = CreateRayGenSignature();
    m_computeSignature = CreateComputeSignature();
    m_missSignature    = CreateMissSignature();
    m_hitSignature     = CreateHitSignature();
    pipeline.SetGlobalRootSignature(m_rayGenSignature.Get());

    m_csPSOs.clear();
    m_callableShaderNames.clear();
    uint32_t nextCs = 0, rgSlot = 0;
    //export name of the first raygen pass — drives the stack-size query below
    //so the pass list can be swapped (e.g. to a debug pass) without a hardcode.
    std::wstring firstRayGenName;

    for (auto& p : m_passes.Passes()) {
        if (p.stage == Stage::Barrier || p.stage == Stage::LoopStart ||
            p.stage == Stage::LoopEnd || p.stage == Stage::PingSwap ||
            p.stage == Stage::ClearSort || p.stage == Stage::DLSS ||
            p.stage == Stage::CudaOp)
            continue;

        // Work graph
        if (p.isWorkGraph) {
            ComPtr<IDxcBlob> lib = nv_helpers_dx12::CompileWG(p.file.c_str());
            D3D12_DXIL_LIBRARY_DESC dxilDesc{};
            dxilDesc.DXILLibrary = { lib->GetBufferPointer(), lib->GetBufferSize() };

            D3D12_STATE_SUBOBJECT subobjects[3]{};
            subobjects[0].Type = D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY;
            subobjects[0].pDesc = &dxilDesc;
            subobjects[1].Type = D3D12_STATE_SUBOBJECT_TYPE_GLOBAL_ROOT_SIGNATURE;
            subobjects[1].pDesc = m_computeSignature.GetAddressOf();

            static const LPCWSTR kWGName = L"main";
            D3D12_WORK_GRAPH_DESC wgDesc = {};
            wgDesc.ProgramName = kWGName;
            wgDesc.Flags = D3D12_WORK_GRAPH_FLAG_INCLUDE_ALL_AVAILABLE_NODES;
            subobjects[2].Type = D3D12_STATE_SUBOBJECT_TYPE_WORK_GRAPH;
            subobjects[2].pDesc = &wgDesc;

            D3D12_STATE_OBJECT_DESC soDesc{};
            soDesc.Type = D3D12_STATE_OBJECT_TYPE_EXECUTABLE;
            soDesc.NumSubobjects = 3;
            soDesc.pSubobjects = subobjects;

            ComPtr<ID3D12StateObject> so;
            ThrowIfFailed(m_ctx.Device()->CreateStateObject(&soDesc, IID_PPV_ARGS(&so)));
            ComPtr<ID3D12StateObjectProperties1> soProps1;
            ThrowIfFailed(so->QueryInterface(IID_PPV_ARGS(&soProps1)));
            ComPtr<ID3D12WorkGraphProperties> wgProps;
            ThrowIfFailed(so->QueryInterface(IID_PPV_ARGS(&wgProps)));

            WgRuntimeData rt{};
            rt.id = soProps1->GetProgramIdentifier(L"main");
            D3D12_WORK_GRAPH_MEMORY_REQUIREMENTS mem{};
            wgProps->GetWorkGraphMemoryRequirements(0, &mem);
            if (mem.MaxSizeInBytes) {
                auto hp = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
                auto buf = CD3DX12_RESOURCE_DESC::Buffer(mem.MaxSizeInBytes,
                    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
                ThrowIfFailed(m_ctx.Device()->CreateCommittedResource(
                    &hp, D3D12_HEAP_FLAG_NONE, &buf,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                    IID_PPV_ARGS(&rt.backingRes)));
                rt.backing.StartAddress = rt.backingRes->GetGPUVirtualAddress();
                rt.backing.SizeInBytes  = mem.MaxSizeInBytes;
            }

            p.wgIdx = (uint32_t)m_wgStateObjects.size();
            m_wgStateObjects.push_back(so);
            m_wgProps.push_back(wgProps);
            m_wgRuntime.push_back(std::move(rt));
            continue;
        }

        // Compute / wavefront / fixed-compute
        if ((p.stage == Stage::Compute || p.stage == Stage::Wavefront ||
             p.stage == Stage::FixedCompute) && !p.isWorkGraph) {
            ComPtr<IDxcBlob> cs = nv_helpers_dx12::CompileCS(p.file.c_str(), L"main");
            D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
            desc.pRootSignature = m_computeSignature.Get();
            desc.CS = { cs->GetBufferPointer(), cs->GetBufferSize() };
            ComPtr<ID3D12PipelineState> pso;
            ThrowIfFailed(m_ctx.Device()->CreateComputePipelineState(&desc, IID_PPV_ARGS(&pso)));
            m_csPSOs.push_back(pso);
            p.psoIdx = nextCs++;
            continue;
        }

        // Callable
        if (p.stage == Stage::Callable) {
            std::wstring base = p.file.substr(p.file.find_last_of(L"/\\") + 1);
            base = base.substr(0, base.rfind(L'.'));
            ComPtr<IDxcBlob> lib = nv_helpers_dx12::CompileShaderLibrary(p.file.c_str());
            pipeline.AddLibrary(lib.Get(), { base.c_str() });
            m_callableShaderNames.push_back(base);
            continue;
        }

        // RayGen
        std::wstring base = p.file.substr(p.file.find_last_of(L"/\\") + 1);
        base = base.substr(0, base.rfind(L'.'));
        if (firstRayGenName.empty()) firstRayGenName = base;
        ComPtr<IDxcBlob> lib = nv_helpers_dx12::CompileShaderLibrary(p.file.c_str());
        pipeline.AddLibrary(lib.Get(), { base.c_str() });
        m_passes.RegisterPassIndex(p.file, rgSlot);
        rgSlot++;
    }

    // Fixed shaders
    ComPtr<IDxcBlob> missLib    = nv_helpers_dx12::CompileShaderLibrary(L"Miss_v8.hlsl");
    ComPtr<IDxcBlob> hitLib     = nv_helpers_dx12::CompileShaderLibrary(L"Hit_v8.hlsl");
    ComPtr<IDxcBlob> anyHitLib  = nv_helpers_dx12::CompileShaderLibrary(L"AnyHit.hlsl");

    pipeline.AddLibrary(missLib.Get(),    { L"Miss" });
    pipeline.AddLibrary(hitLib.Get(),     { L"ClosestHit" });
    pipeline.AddLibrary(anyHitLib.Get(),  { L"AlphaTestAnyHit" });

    // PLANET_INTEGRATION: streamed terrain chunks + the 6-face fallback share
    // ONE hit group (FORCE_OPAQUE, no any-hit). Its closest-hit is the same
    // stub - terrain is shaded procedurally in raygen, keyed on the terrain
    // InstanceID range. The matching SBT entry is appended in CreateShaderBindingTable.
    // Two hit groups: opaque (no any-hit, fast path) and alpha (any-hit for transparency)
    pipeline.AddHitGroup(L"OpaqueHitGroup", L"ClosestHit");
    pipeline.AddHitGroup(L"AlphaHitGroup",  L"ClosestHit", L"AlphaTestAnyHit");
    pipeline.AddHitGroup(L"TerrainHitGroup", L"ClosestHit");

    pipeline.AddRootSignatureAssociation(m_missSignature.Get(), { L"Miss" });
    pipeline.AddRootSignatureAssociation(m_hitSignature.Get(),
        { L"OpaqueHitGroup", L"AlphaHitGroup", L"TerrainHitGroup" });

    //TracePayload is a single uint sentinel (4 B), HitObject SER carries the real hit data
    pipeline.SetMaxPayloadSize(4);
    //triangle barycentrics (float2, 8 B)
    pipeline.SetMaxAttributeSize(2 * sizeof(float));
    pipeline.SetMaxRecursionDepth(1);
    pipeline.SetPipelineFlags(D3D12_RAYTRACING_PIPELINE_FLAG_ALLOW_OPACITY_MICROMAPS);

    m_rtStateObject = pipeline.Generate();
    ThrowIfFailed(m_rtStateObject->QueryInterface(IID_PPV_ARGS(&m_rtStateObjectProps)));

    // Compute pipeline stack size. The raygen export name is taken from the
    // first raygen pass (not hardcoded) so swapping the pass list — e.g. to the
    // debug pass — does not break this query.
    UINT64 rgStack = firstRayGenName.empty()
        ? 0
        : m_rtStateObjectProps->GetShaderStackSize(firstRayGenName.c_str());
    UINT64 maxCallable = 0;
    for (const auto& name : m_callableShaderNames) {
        UINT64 sz = m_rtStateObjectProps->GetShaderStackSize(name.c_str());
        if (sz > maxCallable) maxCallable = sz;
    }
    UINT64 missStack = m_rtStateObjectProps->GetShaderStackSize(L"Miss");
    UINT64 total = (rgStack + std::max(maxCallable, missStack) + 255) & ~255;
    m_rtStateObjectProps->SetPipelineStackSize(total);
}

//====================================
//RENDER TARGETS AND PATH STATE
//====================================

void Renderer::CreateRaytracingOutputBuffer() {
    auto* dev = m_ctx.Device();
    UINT w = GetWidth(), h = GetHeight(), px = w * h;

    // Main output array (6 layers: 0=noisy, 1=denoised, 2=accumulated,
    // 3=NRC debug, 4=x1 sharp-reflection contribution, 5=DLSS albedo debug)
    D3D12_RESOURCE_DESC rd = {};
    rd.DepthOrArraySize = 6;
    rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    rd.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    rd.Width = w; rd.Height = h;
    rd.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    rd.MipLevels = 1; rd.SampleDesc.Count = 1;
    ThrowIfFailed(dev->CreateCommittedResource(
        &nv_helpers_dx12::kDefaultHeapProps, D3D12_HEAP_FLAG_NONE,
        &rd, D3D12_RESOURCE_STATE_COPY_SOURCE, nullptr,
        IID_PPV_ARGS(&m_outputResource)));

    // Permanent data (R32G32B32A32_FLOAT 2D)
    ResourceFactory rf(dev);
    m_permanentDataTexture = rf.CreateTexture2D(w, h, DXGI_FORMAT_R32G32B32A32_FLOAT, 1,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, L"PermanentData");

    // Scratch ping (16-layer float4)
    D3D12_RESOURCE_DESC sd = {};
    sd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    sd.Width = w; sd.Height = h;
    sd.DepthOrArraySize = 16; sd.MipLevels = 1;
    sd.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    sd.SampleDesc.Count = 1;
    sd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    ThrowIfFailed(dev->CreateCommittedResource(
        &nv_helpers_dx12::kDefaultHeapProps, D3D12_HEAP_FLAG_NONE,
        &sd, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
        IID_PPV_ARGS(&m_scratchPing)));

    // Reservoir and sample buffers
    auto MakeRaw = [&](ComPtr<ID3D12Resource>& res, UINT bytes, const std::wstring& name) {
        res = rf.CreateUAVBuffer(bytes, name);
    };
    MakeRaw(m_reservoirBuffer_3,    px * sizeof(Reservoir_GI),  L"Reservoir_GI_1");
    MakeRaw(m_reservoirBuffer_4,    px * sizeof(Reservoir_GI),  L"Reservoir_GI_2");
    MakeRaw(m_sampleBuffer_current, px * sizeof(SampleData),    L"Sample_Current");
    MakeRaw(m_sampleBuffer_last,    px * sizeof(SampleData),    L"Sample_Last");

    CreatePathStateBuffer();
}

void Renderer::CreatePathStateBuffer() {
    ResourceFactory rf(m_ctx.Device());
    m_pathStateBuffer = rf.CreateUAVBuffer(GetWidth() * GetHeight() * 88, L"PathStateBuffer");
    // 32B persistent: [0]=sumLog2LumFixed (u32), [4]=smoothedLog2Lum (f32),
    // [8]=isInitialized (u32 flag), [12]=tileCount (u32), [16]=prevTime (f32),
    // [20..31]=pad. 32 B keeps the UAV 16 B aligned for D3D12. Cleared once at
    // create; finalize shader manages it after that.
    m_autoExposeBuffer = rf.CreateUAVBuffer(32, L"AutoExposeState");
}

//====================================
//SHADER RESOURCE HEAP DESCRIPTOR LAYOUT
//====================================

void Renderer::CreateShaderResourceHeap() {
    auto* dev = m_ctx.Device();
    m_srvUavHeap = nv_helpers_dx12::CreateDescriptorHeap(dev, 1000000,
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, true);

    D3D12_DESCRIPTOR_HEAP_DESC stagingDesc = {};
    stagingDesc.NumDescriptors = 4;
    stagingDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    ThrowIfFailed(dev->CreateDescriptorHeap(&stagingDesc, IID_PPV_ARGS(&m_stagingUavHeap)));

    CD3DX12_CPU_DESCRIPTOR_HANDLE handle(m_srvUavHeap->GetCPUDescriptorHandleForHeapStart());
    CD3DX12_GPU_DESCRIPTOR_HANDLE gpuHandle(m_srvUavHeap->GetGPUDescriptorHandleForHeapStart());
    CD3DX12_CPU_DESCRIPTOR_HANDLE stagingHandle(m_stagingUavHeap->GetCPUDescriptorHandleForHeapStart());
    const UINT inc = dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    auto next     = [&]() { handle.Offset(1, inc); gpuHandle.Offset(1, inc); };
    auto nextStg  = [&]() { stagingHandle.Offset(1, inc); };

    auto nullSRV = [&](D3D12_SRV_DIMENSION dim = D3D12_SRV_DIMENSION_BUFFER) {
        D3D12_SHADER_RESOURCE_VIEW_DESC sd = {};
        sd.Format = DXGI_FORMAT_R32_UINT; sd.ViewDimension = dim;
        sd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        if (dim == D3D12_SRV_DIMENSION_BUFFER) sd.Buffer.NumElements = 1;
        dev->CreateShaderResourceView(nullptr, &sd, handle); next();
    };

    // Slot 0: UAV u0 — output array
    { D3D12_UNORDERED_ACCESS_VIEW_DESC ud = {};
      ud.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
      ud.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
      ud.Texture2DArray.ArraySize = m_outputResource->GetDesc().DepthOrArraySize;
      dev->CreateUnorderedAccessView(m_outputResource.Get(), nullptr, &ud, handle); next(); }

    // Slot 1: UAV u1 — permanent data
    { D3D12_UNORDERED_ACCESS_VIEW_DESC ud = {};
      ud.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
      ud.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
      dev->CreateUnorderedAccessView(m_permanentDataTexture.Get(), nullptr, &ud, handle); next(); }

    // Slot 2: SRV t0 — TLAS
    // PLANET_INTEGRATION: points at the planet StreamOrchestrator's unified TLAS
    // (scene meshes + terrain chunks + fallback), rebuilt every frame on the
    // compute queue. The result buffer is preallocated once at orchestrator init,
    // so this SRV is created once here and never re-pointed.
    { D3D12_SHADER_RESOURCE_VIEW_DESC sd = {};
      sd.Format = DXGI_FORMAT_UNKNOWN;
      sd.ViewDimension = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE;
      sd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
      sd.RaytracingAccelerationStructure.Location = m_planet.tlas_result()->GetGPUVirtualAddress();
      dev->CreateShaderResourceView(nullptr, &sd, handle); next(); }

    // Slot 3: SRV t1 — Global IB
    { D3D12_SHADER_RESOURCE_VIEW_DESC sd = {};
      sd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
      sd.Format = DXGI_FORMAT_R32_UINT;
      sd.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
      //A scene with no triangle geometry has global index / vertex /
      //material-ID element counts of 0. D3D12 rejects a buffer SRV with
      //NumElements==0, so clamp to 1; the backing buffers are valid 256 B
      //placeholders (CreateBuffer's size-0 guard) that nothing samples.
      sd.Buffer.NumElements = std::max(m_scene.totalIndexCount, 1u);
      dev->CreateShaderResourceView(m_scene.indexGlobal.Get(), &sd, handle); next(); }

    // Slot 4: SRV t2 — Global VB
    { D3D12_SHADER_RESOURCE_VIEW_DESC sd = {};
      sd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
      sd.Format = DXGI_FORMAT_UNKNOWN;
      sd.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
      sd.Buffer.NumElements = std::max(m_scene.totalVertexCount, 1u);  //see Slot 3
      sd.Buffer.StructureByteStride = sizeof(BTriVertex);
      dev->CreateShaderResourceView(m_scene.vertexGlobal.Get(), &sd, handle); next(); }

    // Slot 5: CBV b0 — Camera
    { D3D12_CONSTANT_BUFFER_VIEW_DESC cd = {};
      cd.BufferLocation = m_camera.GPUBuffer()->GetGPUVirtualAddress();
      cd.SizeInBytes = m_camera.BufferSize();
      dev->CreateConstantBufferView(&cd, handle); next(); }

    // Slot 6: SRV t3 — Instance properties
    { D3D12_SHADER_RESOURCE_VIEW_DESC sd = {};
      sd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
      sd.Format = DXGI_FORMAT_UNKNOWN;
      sd.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
      sd.Buffer.NumElements = (UINT)m_scene.instances.size();
      sd.Buffer.StructureByteStride = sizeof(InstanceProperties);
      dev->CreateShaderResourceView(m_scene.instanceProperties.Get(), &sd, handle); next(); }

    // Slot 7: SRV t4 — Material IDs
    { D3D12_SHADER_RESOURCE_VIEW_DESC sd = {};
      sd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
      sd.Format = DXGI_FORMAT_R32_UINT;
      sd.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
      sd.Buffer.NumElements = std::max((UINT)m_scene.materialIDs.size(), 1u);  //see Slot 3
      dev->CreateShaderResourceView(m_scene.materialIndexBuffer.Get(), &sd, handle); next(); }

    // Slot 8: SRV t5 — Materials (compressed AoS, 40 B / material).
    // Fields: Kd.rgb RGB9E5, Kd.w+Ni half2, Pr/Pm/Ps/Pc u8, Tf RGB9E5,
    // Pcr/aniso/anisoRot/alphaThreshold u8, 3× texID i16, 3× UV scale half2.
    { D3D12_SHADER_RESOURCE_VIEW_DESC sd = {};
      sd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
      sd.Format = DXGI_FORMAT_UNKNOWN;
      sd.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
      sd.Buffer.NumElements = (UINT)m_scene.materials.size();
      sd.Buffer.StructureByteStride = 40;
      dev->CreateShaderResourceView(m_scene.materialBuffer.Get(), &sd, handle); next(); }

    // Slot 9: SRV t6 — Emissive triangles
    if (m_scene.emissiveTrianglesBuffer) {
        D3D12_SHADER_RESOURCE_VIEW_DESC sd = {};
        sd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        sd.Format = DXGI_FORMAT_UNKNOWN;
        sd.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        sd.Buffer.NumElements = (UINT)m_scene.emissiveTriangles.size();
        sd.Buffer.StructureByteStride = sizeof(LightTriangle);
        dev->CreateShaderResourceView(m_scene.emissiveTrianglesBuffer.Get(), &sd, handle);
        next();
    } else nullSRV();   //nullSRV() advances the heap handle itself — no trailing next()

    // Slots 10-15: Reservoir / sample UAVs
    auto rawUAV = [&](ComPtr<ID3D12Resource>& res, UINT bytes) {
        D3D12_UNORDERED_ACCESS_VIEW_DESC ud = {};
        ud.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        ud.Format = DXGI_FORMAT_R32_TYPELESS;
        ud.Buffer.NumElements = bytes / 4;
        ud.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
        dev->CreateUnorderedAccessView(res.Get(), nullptr, &ud, handle);
        next();
    };
    UINT px = GetWidth() * GetHeight();
    // Slots 10/11 map to root-sig u2/u3 which no shader binds. Two null
    // raw UAVs keep the contiguous u2..u7 descriptor range valid without
    // backing memory. Inlined to avoid colliding with the nullRawUAV
    // lambda defined later in this same function for the NRC slots.
    for (int i = 0; i < 2; ++i) {
        D3D12_UNORDERED_ACCESS_VIEW_DESC ud = {};
        ud.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        ud.Format = DXGI_FORMAT_R32_TYPELESS;
        ud.Buffer.NumElements = 1;
        ud.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
        dev->CreateUnorderedAccessView(nullptr, nullptr, &ud, handle);
        next();
    }
    rawUAV(m_reservoirBuffer_3,    px * sizeof(Reservoir_GI));
    rawUAV(m_reservoirBuffer_4,    px * sizeof(Reservoir_GI));
    rawUAV(m_sampleBuffer_current, px * sizeof(SampleData));
    rawUAV(m_sampleBuffer_last,    px * sizeof(SampleData));

    // Slots 16-17: placeholders (t7, t8 registered in the root signature
    // but unused now that the material buffer is a single SRV on t5).
    nullSRV(); nullSRV();

    // Slot 18: scratch ping UAV
    { D3D12_UNORDERED_ACCESS_VIEW_DESC ud = {};
      ud.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
      ud.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
      ud.Texture2DArray.ArraySize = 16;
      dev->CreateUnorderedAccessView(m_scratchPing.Get(), nullptr, &ud, handle); next(); }

    // Slot 19 maps to root-sig u9, which no shader binds. Null UAV keeps
    // the descriptor table layout intact.
    {
        D3D12_UNORDERED_ACCESS_VIEW_DESC ud = {};
        ud.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        ud.Format = DXGI_FORMAT_R32_TYPELESS;
        ud.Buffer.NumElements = 1;
        ud.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
        dev->CreateUnorderedAccessView(nullptr, nullptr, &ud, handle);
        next();
    }

    // Slots 20-23: light tree SRVs
    m_lightTree.WriteSrvs(dev, handle);
    handle.Offset(4, inc); gpuHandle.Offset(4, inc);

    // Slot 24: triToLightId SRV
    if (m_scene.triToLightIdBuffer) {
        D3D12_SHADER_RESOURCE_VIEW_DESC sd = {};
        sd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        sd.Format = DXGI_FORMAT_R32_UINT;
        sd.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        sd.Buffer.NumElements = (UINT)m_scene.triToLightId.size();
        dev->CreateShaderResourceView(m_scene.triToLightIdBuffer.Get(), &sd, handle);
        next();
    } else nullSRV();   //nullSRV() advances the heap handle itself — no trailing next()

    // Slots 25-27: light tree lookup SRVs
    m_lightTree.WriteLookupSrvs(dev, handle);
    handle.Offset(3, inc); gpuHandle.Offset(3, inc);

    // Slots 28-30: null tex2D placeholders
    auto nullTex2D = [&]() {
        D3D12_SHADER_RESOURCE_VIEW_DESC d = {};
        d.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        d.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        d.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        d.Texture2D.MipLevels = 1;
        dev->CreateShaderResourceView(nullptr, &d, handle); next();
    };
    nullTex2D(); nullTex2D(); nullTex2D();

    // Slot 31: LUT texture array
    if (m_lutTextureArray) {
        D3D12_SHADER_RESOURCE_VIEW_DESC d = {};
        d.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        d.Format = m_lutTextureArray->GetDesc().Format;
        d.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
        d.Texture2DArray.MipLevels = m_lutTextureArray->GetDesc().MipLevels;
        d.Texture2DArray.ArraySize = m_lutTextureArray->GetDesc().DepthOrArraySize;
        dev->CreateShaderResourceView(m_lutTextureArray.Get(), &d, handle); next();
    } else { nullSRV(D3D12_SRV_DIMENSION_TEXTURE2DARRAY); }

    // Slot 32: path state UAV
    rawUAV(m_pathStateBuffer, GetWidth() * GetHeight() * 88);

    // Slot 33: global counters UAV
    { D3D12_UNORDERED_ACCESS_VIEW_DESC ud = {};
      ud.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
      ud.Format = DXGI_FORMAT_R32_TYPELESS;
      ud.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
      ud.Buffer.NumElements = MAX_STACKS;
      dev->CreateUnorderedAccessView(m_globalCounterBuffer.Get(), nullptr, &ud, handle); next(); }

    // Slot 34: indirect args UAV
    { D3D12_UNORDERED_ACCESS_VIEW_DESC ud = {};
      ud.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
      ud.Format = DXGI_FORMAT_UNKNOWN;
      ud.Buffer.NumElements = MAX_INDIRECT_COMMANDS;
      ud.Buffer.StructureByteStride = sizeof(D3D12_DISPATCH_ARGUMENTS);
      dev->CreateUnorderedAccessView(m_indirectArgsBuffer.Get(), nullptr, &ud, handle); next(); }

    // Slots 35-38: stack buffers UAV
    for (int s = 0; s < 4; ++s) {
        D3D12_UNORDERED_ACCESS_VIEW_DESC ud = {};
        ud.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        ud.Format = DXGI_FORMAT_UNKNOWN;
        ud.Buffer.NumElements = GetWidth() * GetHeight();
        ud.Buffer.StructureByteStride = 8;
        dev->CreateUnorderedAccessView(m_stackBuffers[s].Get(), nullptr, &ud, handle); next();
    }

    // Slots 39-51: DLSS UAVs
    auto dlssUAV = [&](ID3D12Resource* res, DXGI_FORMAT fmt) {
        D3D12_UNORDERED_ACCESS_VIEW_DESC ud = {};
        ud.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        ud.Format = fmt;
        dev->CreateUnorderedAccessView(res, nullptr, &ud, handle); next();
    };
    dlssUAV(m_dlss.Depth(),            DXGI_FORMAT_R32_FLOAT);
    dlssUAV(m_dlss.MVec(),             DXGI_FORMAT_R16G16_FLOAT);
    dlssUAV(m_dlss.Normals(),          DXGI_FORMAT_R16G16B16A16_FLOAT);
    dlssUAV(m_dlss.DiffuseAlbedo(),    DXGI_FORMAT_R16G16B16A16_FLOAT);
    dlssUAV(m_dlss.Output(),           DXGI_FORMAT_R16G16B16A16_FLOAT);
    dlssUAV(m_dlss.SpecularAlbedo(),   DXGI_FORMAT_R16G16B16A16_FLOAT);
    dlssUAV(m_dlss.Roughness(),        DXGI_FORMAT_R16_FLOAT);
    dlssUAV(m_dlss.SpecMVec(),         DXGI_FORMAT_R16G16_FLOAT);
    dlssUAV(m_dlss.SpecHitDist(),      DXGI_FORMAT_R16_FLOAT);
    dlssUAV(m_dlss.Transparency(),     DXGI_FORMAT_R16G16B16A16_FLOAT);
    dlssUAV(m_dlss.ColorBeforeTrans(), DXGI_FORMAT_R16G16B16A16_FLOAT);
    dlssUAV(m_dlss.Input(),            DXGI_FORMAT_R16G16B16A16_FLOAT);
    dlssUAV(m_dlss.BiasHint(),         DXGI_FORMAT_R8_UNORM);

    // Sort buffer UAVs (staging + main heap)
    auto sortUAV = [&](ComPtr<ID3D12Resource>& buf, UINT numElements) {
        D3D12_UNORDERED_ACCESS_VIEW_DESC ud = {};
        ud.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        ud.Format = DXGI_FORMAT_R32_TYPELESS;
        ud.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
        ud.Buffer.NumElements = numElements;
        dev->CreateUnorderedAccessView(buf.Get(), nullptr, &ud, stagingHandle);
        dev->CopyDescriptorsSimple(1, handle, stagingHandle, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        auto cpuH = stagingHandle;
        auto gpuH = gpuHandle;
        next(); nextStg();
        return std::make_pair(cpuH, gpuH);
    };
    { auto [c,g] = sortUAV(m_sortCountBuffer,  SORT_BUCKETS); m_sortCountCpuHandle  = c; m_sortCountGpuHandle  = g; }
    { auto [c,g] = sortUAV(m_sortOffsetBuffer, SORT_BUCKETS); m_sortOffsetCpuHandle = c; m_sortOffsetGpuHandle = g; }
    { auto [c,g] = sortUAV(m_sortBoundsBuffer, 8);            m_sortBoundsCpuHandle = c; m_sortBoundsGpuHandle = g; }

    // Slots 55-57: paired reuse textures (t19, t20, t21) — R16G16_SINT
    for (int i = 0; i < 3; ++i) {
        if (m_reuseTexture[i]) {
            D3D12_SHADER_RESOURCE_VIEW_DESC d = {};
            d.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            d.Format                  = DXGI_FORMAT_R16G16_SINT;
            d.ViewDimension           = D3D12_SRV_DIMENSION_TEXTURE2D;
            d.Texture2D.MipLevels     = 1;
            dev->CreateShaderResourceView(m_reuseTexture[i].Get(), &d, handle); next();
        } else {
            nullSRV(D3D12_SRV_DIMENSION_TEXTURE2D);
        }
    }

    // Slots 58-62: NRC shared D3D12/CUDA UAVs (u40..u44). CudaInterop has
    // already allocated the backing buffers; here we just create views.
    // NullUAVs when interop init failed keep the descriptor table valid
    // so the rest of the pipeline still runs.
    auto nrcUAV = [&](const CudaInterop::Buffer& b) {
        D3D12_UNORDERED_ACCESS_VIEW_DESC ud = {};
        ud.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        ud.Format        = DXGI_FORMAT_R32_TYPELESS;
        ud.Buffer.Flags  = D3D12_BUFFER_UAV_FLAG_RAW;
        ud.Buffer.NumElements = (UINT)((b.sizeBytes + 3u) / 4u);
        dev->CreateUnorderedAccessView(b.resource.Get(), nullptr, &ud, handle);
        next();
    };
    auto nullRawUAV = [&]() {
        D3D12_UNORDERED_ACCESS_VIEW_DESC ud = {};
        ud.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        ud.Format        = DXGI_FORMAT_R32_TYPELESS;
        ud.Buffer.Flags  = D3D12_BUFFER_UAV_FLAG_RAW;
        ud.Buffer.NumElements = 1;
        dev->CreateUnorderedAccessView(nullptr, nullptr, &ud, handle);
        next();
    };
    if (m_nrcInferenceIn.resource)  nrcUAV(m_nrcInferenceIn);  else nullRawUAV();  // u40
    if (m_nrcInferenceOut.resource) nrcUAV(m_nrcInferenceOut); else nullRawUAV();  // u41
    if (m_nrcPendingGI.resource)    nrcUAV(m_nrcPendingGI);    else nullRawUAV();  // u42
    if (m_nrcTrainRecords.resource) nrcUAV(m_nrcTrainRecords); else nullRawUAV();  // u43
    if (m_nrcCounters.resource)     nrcUAV(m_nrcCounters);     else nullRawUAV();  // u44

    // Slot 63: autoexpose persistent state (u24, 16 B raw)
    { D3D12_UNORDERED_ACCESS_VIEW_DESC ud = {};
      ud.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
      ud.Format        = DXGI_FORMAT_R32_TYPELESS;
      ud.Buffer.Flags  = D3D12_BUFFER_UAV_FLAG_RAW;
      ud.Buffer.NumElements = 4;  // 16 B / 4
      dev->CreateUnorderedAccessView(m_autoExposeBuffer.Get(), nullptr, &ud, handle); next(); }

    // Slot 64: star / Milky Way skybox SRV (t40, gSkyStars in Includes_v8.hlsli).
    // Null SRV until InitSkyStarsTexture has run successfully so the descriptor
    // table stays valid even if the EXR file is missing.
    if (m_skyStarsTexture) {
        D3D12_SHADER_RESOURCE_VIEW_DESC d = {};
        d.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        d.Format                  = m_skyStarsTexture->GetDesc().Format;
        d.ViewDimension           = D3D12_SRV_DIMENSION_TEXTURE2D;
        d.Texture2D.MipLevels     = m_skyStarsTexture->GetDesc().MipLevels;
        dev->CreateShaderResourceView(m_skyStarsTexture.Get(), &d, handle); next();
    } else {
        nullSRV(D3D12_SRV_DIMENSION_TEXTURE2D);
    }

    // Slot 65: volumetric cloud noise 3D texture SRV (t42, g_cloudNoise in
    // Includes_v8.hlsli). Filled at startup by BakeCloudNoiseTexture (compute
    // pass dispatching Pass_cloudnoise_bake_v8.hlsl). Null fallback keeps the
    // descriptor table valid if the bake somehow failed to produce a texture.
    if (m_cloudNoiseTexture) {
        D3D12_SHADER_RESOURCE_VIEW_DESC d = {};
        d.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        d.Format                  = m_cloudNoiseTexture->GetDesc().Format;
        d.ViewDimension           = D3D12_SRV_DIMENSION_TEXTURE3D;
        d.Texture3D.MipLevels     = m_cloudNoiseTexture->GetDesc().MipLevels;
        dev->CreateShaderResourceView(m_cloudNoiseTexture.Get(), &d, handle); next();
    } else {
        nullSRV(D3D12_SRV_DIMENSION_TEXTURE3D);
    }

    // Slot 66: planet-scale cloud coverage 2D texture SRV (t43, g_cloudCoverage
    // in Includes_v8.hlsli). NASA Blue Marble cloud_combined_8192.tif loaded
    // by InitCloudCoverageTexture. Null fallback so missing file (download
    // failed, user removed) doesn't break the descriptor table — shader reads
    // 0 and falls back to pure procedural coverage.
    if (m_cloudCoverageTexture) {
        D3D12_SHADER_RESOURCE_VIEW_DESC d = {};
        d.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        d.Format                  = m_cloudCoverageTexture->GetDesc().Format;
        d.ViewDimension           = D3D12_SRV_DIMENSION_TEXTURE2D;
        d.Texture2D.MipLevels     = m_cloudCoverageTexture->GetDesc().MipLevels;
        dev->CreateShaderResourceView(m_cloudCoverageTexture.Get(), &d, handle); next();
    } else {
        nullSRV(D3D12_SRV_DIMENSION_TEXTURE2D);
    }

    // Slot 67: spatiotemporal blue noise Texture2DArray SRV (t41, g_cloudSTBN
    // in Clouds_v8.hlsli). 128x128x64 RGBA8 filled at startup by
    // BakeCloudSTBNTexture (Pass_stbn_bake_v8.hlsl). Cloud shader's CloudRand4
    // samples this via .Load(uint4(px, slice, 0)) to drive cone shadow taps,
    // step jitter, and atmospheric cloud shadow cone with a blue noise spatial
    // spectrum that DLSS RR cleanly removes.
    if (m_cloudSTBNTexture) {
        D3D12_SHADER_RESOURCE_VIEW_DESC d = {};
        d.Shader4ComponentMapping       = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        d.Format                        = m_cloudSTBNTexture->GetDesc().Format;
        d.ViewDimension                 = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
        d.Texture2DArray.MipLevels      = m_cloudSTBNTexture->GetDesc().MipLevels;
        d.Texture2DArray.ArraySize      = m_cloudSTBNTexture->GetDesc().DepthOrArraySize;
        dev->CreateShaderResourceView(m_cloudSTBNTexture.Get(), &d, handle); next();
    } else {
        nullSRV(D3D12_SRV_DIMENSION_TEXTURE2DARRAY);
    }

    // Slot 68 (TERRAIN_TABLE_HEAP_SLOT): planet terrain instance table SRV
    // (t44, g_terrainTable in Includes_v8.hlsli). StructuredBuffer<uint3>, one
    // entry per BLAS slot, refilled each frame by the planet StreamOrchestrator
    // (created in StreamOrchestrator::init, so always present).
    {
        D3D12_SHADER_RESOURCE_VIEW_DESC d = {};
        d.Shader4ComponentMapping     = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        d.Format                      = DXGI_FORMAT_UNKNOWN;
        d.ViewDimension               = D3D12_SRV_DIMENSION_BUFFER;
        d.Buffer.NumElements          = planet::StreamOrchestrator::terrain_table_count();
        d.Buffer.StructureByteStride  = sizeof(planet::TerrainSlotGPU);
        dev->CreateShaderResourceView(m_planet.terrain_table(), &d, handle); next();
    }

    // Bindless textures
    UINT globalTexIdx = 0;
    auto writeBatch = [&](UINT heapBase, UINT count) {
        for (UINT i = 0; i < count; ++i) {
            CD3DX12_CPU_DESCRIPTOR_HANDLE dst(m_srvUavHeap->GetCPUDescriptorHandleForHeapStart(), heapBase + i, inc);
            auto* res = m_scene.bindlessGpuTextures[globalTexIdx].Get();
            D3D12_SHADER_RESOURCE_VIEW_DESC srv = {};
            srv.Format = res->GetDesc().Format;
            srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            srv.Texture2D.MipLevels = res->GetDesc().MipLevels;
            dev->CreateShaderResourceView(res, &srv, dst);
            globalTexIdx++;
        }
    };
    UINT albedoCount = m_scene.bindlessNormalBase - m_scene.bindlessAlbedoBase;
    UINT normalCount = m_scene.bindlessRmaBase    - m_scene.bindlessNormalBase;
    UINT rmaCount    = m_scene.totalBindlessTextures - albedoCount - normalCount;
    writeBatch(m_scene.bindlessAlbedoBase, albedoCount);
    writeBatch(m_scene.bindlessNormalBase, normalCount);
    writeBatch(m_scene.bindlessRmaBase,    rmaCount);
}

//====================================
//SHADER BINDING TABLE
//====================================

void Renderer::CreateShaderBindingTable() {
    m_sbtHelper.Reset();
    D3D12_GPU_DESCRIPTOR_HANDLE heapHandle = m_srvUavHeap->GetGPUDescriptorHandleForHeapStart();
    auto heapPointer = reinterpret_cast<UINT64*>(heapHandle.ptr);

    for (const auto& entry : m_passes.Tokens()) {
        if (entry == L"barrier" || entry.rfind(L"loop:", 0) == 0 ||
            entry == L"endloop" || entry == L"pingswap" ||
            entry == L"clearsort" || entry == L"dlss" ||
            entry.rfind(L"cuda:", 0) == 0)
            continue;
        if (entry.find(L"|cs:") != std::wstring::npos ||
            entry.find(L"|wf:") != std::wstring::npos ||
            entry.find(L"|wg:") != std::wstring::npos ||
            entry.find(L"|fx:") != std::wstring::npos ||
            entry.find(L"|call") != std::wstring::npos)
            continue;

        std::wstring base = entry.substr(entry.find_last_of(L"/\\") + 1);
        base = base.substr(0, base.rfind(L'.'));
        m_sbtHelper.AddRayGenerationProgram(base.c_str(), { heapPointer });
    }

    m_sbtHelper.AddMissProgram(L"Miss", {});

    // Two SBT hit-group entries per instance (stride 2):
    //   entry i*2+0  → opaque geometry (GeometryIndex 0)
    //   entry i*2+1  → alpha geometry  (GeometryIndex 1)
    // For meshes with only one geometry type, the unused entry is harmless.
    for (size_t i = 0; i < m_scene.instances.size(); ++i) {
        const auto& mesh = m_scene.meshes[m_scene.instances[i].meshIndex];
        bool hasAlpha  = mesh.alphaTriCount  > 0;
        bool hasOpaque = mesh.opaqueTriCount > 0;

        // GeometryIndex 0: opaque if present, else alpha
        if (hasOpaque)
            m_sbtHelper.AddHitGroup(L"OpaqueHitGroup", {});
        else
            m_sbtHelper.AddHitGroup(L"AlphaHitGroup", {});

        // GeometryIndex 1: alpha (only reached if BLAS has 2 geometries)
        m_sbtHelper.AddHitGroup(L"AlphaHitGroup", {});
    }

    // PLANET_INTEGRATION: one shared TerrainHitGroup entry for every terrain
    // instance (streamed chunks + the 6-face fallback). It lands at SBT index
    // 2 * sceneInstanceCount; the orchestrator sets exactly that value as the
    // hitGroupContribution on every terrain instance in the unified TLAS.
    m_sbtHelper.AddHitGroup(L"TerrainHitGroup", {});

    for (const auto& name : m_callableShaderNames)
        m_sbtHelper.AddCallableProgram(name, { heapPointer });

    uint32_t sbtSize = m_sbtHelper.ComputeSBTSize();
    m_sbtStorage = nv_helpers_dx12::CreateBuffer(m_ctx.Device(), sbtSize,
        D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ,
        nv_helpers_dx12::kUploadHeapProps);
    m_sbtHelper.Generate(m_sbtStorage.Get(), m_rtStateObjectProps.Get());
}

//====================================
//STREAMING COMPACTION AND INDIRECT DISPATCH
//====================================

void Renderer::CreateStreamingCompactionBuffers() {
    auto* dev = m_ctx.Device();
    ResourceFactory rf(dev);
    UINT totalPx = GetWidth() * GetHeight();

    for (int i = 0; i < MAX_STACKS; ++i)
        m_stackBuffers[i] = rf.CreateUAVBuffer(totalPx * 8,
            L"PixelStack_" + std::to_wstring(i));

    m_globalCounterBuffer = rf.CreateUAVBuffer(MAX_STACKS * sizeof(uint32_t), L"GlobalCounters");
    m_indirectArgsBuffer  = rf.CreateUAVBuffer(
        MAX_INDIRECT_COMMANDS * sizeof(D3D12_DISPATCH_ARGUMENTS), L"IndirectArgs");

    // Zero buffer (upload)
    { auto hp = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
      auto bd = CD3DX12_RESOURCE_DESC::Buffer(MAX_STACKS * sizeof(uint32_t));
      ThrowIfFailed(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &bd,
          D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&m_zeroBuffer)));
      void* p; m_zeroBuffer->Map(0, nullptr, &p);
      memset(p, 0, MAX_STACKS * sizeof(uint32_t));
      m_zeroBuffer->Unmap(0, nullptr); }

    // Sort buffers
    m_sortCountBuffer  = rf.CreateUAVBuffer(SORT_BUCKETS * sizeof(uint32_t), L"SortCount");
    m_sortOffsetBuffer = rf.CreateUAVBuffer(SORT_BUCKETS * sizeof(uint32_t), L"SortOffset");
    m_sortBoundsBuffer = rf.CreateUAVBuffer(8 * sizeof(uint32_t), L"SortBounds");

    // Sort bounds reset
    { const UINT MAX_U = 0xFFFFFFFF, MIN_U = 0;
      const UINT initData[8] = { MAX_U, MAX_U, MAX_U, MIN_U, MIN_U, MIN_U, MAX_U, MIN_U };
      auto hp = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
      auto bd = CD3DX12_RESOURCE_DESC::Buffer(sizeof(initData));
      ThrowIfFailed(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &bd,
          D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&m_sortBoundsResetBuffer)));
      void* p; CD3DX12_RANGE rr(0,0);
      ThrowIfFailed(m_sortBoundsResetBuffer->Map(0, &rr, &p));
      memcpy(p, initData, sizeof(initData));
      m_sortBoundsResetBuffer->Unmap(0, nullptr); }
}

void Renderer::CreateIndirectCommandSignature() {
    D3D12_INDIRECT_ARGUMENT_DESC ad = {};
    ad.Type = D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH;
    D3D12_COMMAND_SIGNATURE_DESC cd = {};
    cd.ByteStride       = sizeof(D3D12_DISPATCH_ARGUMENTS);
    cd.NumArgumentDescs  = 1;
    cd.pArgumentDescs    = &ad;
    ThrowIfFailed(m_ctx.Device()->CreateCommandSignature(
        &cd, nullptr, IID_PPV_ARGS(&m_commandSignature)));
}

void Renderer::CompileSetupIndirectShader() {
    const char* shaderCode = R"(
        RWByteAddressBuffer Counters : register(u0);
        RWStructuredBuffer<uint3> IndirectArgs : register(u1);
        cbuffer Params : register(b0) { uint counterReadIdx; uint argWriteIdx; uint counterClearIdx; uint groupSize; };
        [numthreads(1,1,1)] void main() {
            uint ac = Counters.Load(counterReadIdx*4);
            IndirectArgs[argWriteIdx] = uint3((ac+groupSize-1)/groupSize, 1, 1);
        #if CLEAR_COUNTER
            Counters.Store(counterClearIdx*4, 0);
        #endif
        })";

    // Root signature
    { CD3DX12_ROOT_PARAMETER1 rp[2]; CD3DX12_DESCRIPTOR_RANGE1 range[1];
      range[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 2, 0, 0, D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE);
      rp[0].InitAsDescriptorTable(1, range);
      rp[1].InitAsConstants(4, 0);
      CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC rd; rd.Init_1_1(2, rp, 0, nullptr);
      ComPtr<ID3DBlob> sb, eb;
      D3D12SerializeVersionedRootSignature(&rd, &sb, &eb);
      ThrowIfFailed(m_ctx.Device()->CreateRootSignature(0,
          sb->GetBufferPointer(), sb->GetBufferSize(), IID_PPV_ARGS(&m_rsSetupIndirect))); }

    auto Compile = [&](const char* name, bool doClear, ComPtr<ID3D12PipelineState>& pso) {
        D3D_SHADER_MACRO macros[] = { { "CLEAR_COUNTER", doClear ? "1" : "0" }, { NULL, NULL } };
        ComPtr<ID3DBlob> sb, eb;
        ThrowIfFailed(D3DCompile(shaderCode, strlen(shaderCode), name, macros, nullptr,
            "main", "cs_5_0", 0, 0, &sb, &eb));
        D3D12_COMPUTE_PIPELINE_STATE_DESC pd = {};
        pd.pRootSignature = m_rsSetupIndirect.Get();
        pd.CS = { sb->GetBufferPointer(), sb->GetBufferSize() };
        ThrowIfFailed(m_ctx.Device()->CreateComputePipelineState(&pd, IID_PPV_ARGS(&pso)));
    };
    Compile("SetupIndirect_Clear",   true,  m_psoSetupIndirect);
    Compile("SetupIndirect_NoClear", false, m_psoSetupIndirectNoClear);
}

void Renderer::ClearSortBuffers(ID3D12GraphicsCommandList* cmdList) {
    UINT cv[4] = {0,0,0,0};
    cmdList->ClearUnorderedAccessViewUint(
        m_sortCountGpuHandle, m_sortCountCpuHandle,
        m_sortCountBuffer.Get(), cv, 0, nullptr);

    auto b1 = CD3DX12_RESOURCE_BARRIER::Transition(m_sortBoundsBuffer.Get(),
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST);
    cmdList->ResourceBarrier(1, &b1);
    cmdList->CopyResource(m_sortBoundsBuffer.Get(), m_sortBoundsResetBuffer.Get());
    auto b2 = CD3DX12_RESOURCE_BARRIER::Transition(m_sortBoundsBuffer.Get(),
        D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    cmdList->ResourceBarrier(1, &b2);
}

//====================================
//LUT TEXTURES
//====================================

void Renderer::GenerateLutTextures() {
    SCOPE_TIMER("GenerateLutTextures");
    std::vector<std::vector<float>> allData(NUM_LUTS, std::vector<float>(LUT_RESOLUTION * LUT_RESOLUTION));
    const XMFLOAT3 N = {0,0,1};
    Material tempMat;
    std::random_device rd; std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0, 1);

    for (int y = 0; y < LUT_RESOLUTION; ++y) {
        float cosTheta = std::max(0.01f, (float)y / (LUT_RESOLUTION-1));
        float sinTheta = sqrt(1 - cosTheta*cosTheta);
        const XMFLOAT3 V = { sinTheta, 0, cosTheta };

        for (int x = 0; x < LUT_RESOLUTION; ++x) {
            float roughness = std::max(0.01f, (float)x / (LUT_RESOLUTION-1));
            tempMat.Pr_Pm_Ps_Pc.x = roughness;
            size_t pi = (size_t)y * LUT_RESOLUTION + x;

            allData[0][pi] = ComputeSheenDirectionalAlbedo(N, V, roughness, NUM_SAMPLES_LUT);

            float Ess = 0;
            for (int i = 0; i < NUM_SAMPLES_LUT; ++i) {
                XMFLOAT3 L;
                SampleGGX(tempMat, V, N, L, dist(gen), dist(gen));
                if (dot(N,L) <= 0) continue;
                XMFLOAT3 b = EvaluateBRDF_GGX(V, L, N, {}, roughness);
                float pdf = BRDF_PDF_GGX(roughness, N, L*-1.0f, V);
                if (pdf > 1e-6f) Ess += (b.x * dot(N,L)) / pdf;
            }
            allData[1][pi] = NUM_SAMPLES_LUT > 0 ? Ess / NUM_SAMPLES_LUT : 0;
        }
    }
    CreateAndUploadLutArray(allData, m_lutTextureArray, L"LutTextureArray");
}

void Renderer::CreateAndUploadLutArray(
    const std::vector<std::vector<float>>& allData,
    ComPtr<ID3D12Resource>& tar, const std::wstring& rn)
{
    if (allData.empty()) return;
    UINT arraySize = (UINT)allData.size();

    D3D12_RESOURCE_DESC td = {};
    td.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    td.Width = LUT_RESOLUTION; td.Height = LUT_RESOLUTION;
    td.DepthOrArraySize = arraySize; td.MipLevels = 1;
    td.Format = DXGI_FORMAT_R32_FLOAT; td.SampleDesc.Count = 1;
    ThrowIfFailed(m_ctx.Device()->CreateCommittedResource(
        &nv_helpers_dx12::kDefaultHeapProps, D3D12_HEAP_FLAG_NONE,
        &td, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&tar)));
    tar->SetName(rn.c_str());

    UINT64 uploadSize = GetRequiredIntermediateSize(tar.Get(), 0, arraySize);
    m_lutUploadHeaps.emplace_back();
    auto& uh = m_lutUploadHeaps.back();
    auto ud = CD3DX12_RESOURCE_DESC::Buffer(uploadSize);
    ThrowIfFailed(m_ctx.Device()->CreateCommittedResource(
        &nv_helpers_dx12::kUploadHeapProps, D3D12_HEAP_FLAG_NONE,
        &ud, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&uh)));

    std::vector<D3D12_SUBRESOURCE_DATA> sr(arraySize);
    for (UINT i = 0; i < arraySize; ++i) {
        sr[i].pData      = allData[i].data();
        sr[i].RowPitch   = LUT_RESOLUTION * sizeof(float);
        sr[i].SlicePitch = sr[i].RowPitch * LUT_RESOLUTION;
    }
    UpdateSubresources(m_ctx.CmdList(), tar.Get(), uh.Get(), 0, 0, arraySize, sr.data());

    auto b = CD3DX12_RESOURCE_BARRIER::Transition(tar.Get(),
        D3D12_RESOURCE_STATE_COPY_DEST, kSRV);
    m_ctx.CmdList()->ResourceBarrier(1, &b);
}

//====================================
//PAIRED SPATIAL REUSE TEXTURES
//====================================
//Lin et al. 2026

void Renderer::InitReuseTextures() {
    const int   kSizes[3] = { 254, 230, 210 };
    const float kSigma    = 20.0f;
    auto*       dev       = m_ctx.Device();

    for (int i = 0; i < 3; ++i) {
        std::vector<int16_t> rg;
        GenerateReuseTexture(kSizes[i], kSigma,
                             static_cast<uint32_t>(i + 1), rg);

        int bad = -1;
        if (!ValidateReuseTexture(kSizes[i], rg, &bad)) {
            LOG(L"[ReuseTex] " << kSizes[i] << L"x" << kSizes[i]
                << L" self-inversion FAILED at texel " << bad);
            continue;
        }
        LOG(L"[ReuseTex] " << kSizes[i] << L"x" << kSizes[i]
            << L" self-inversion OK ("
            << (rg.size() * sizeof(int16_t)) << L" bytes)");

        // Default-heap Texture2D, R16G16_SINT
        D3D12_RESOURCE_DESC td = {};
        td.Dimension        = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        td.Width            = static_cast<UINT64>(kSizes[i]);
        td.Height           = static_cast<UINT>(kSizes[i]);
        td.DepthOrArraySize = 1;
        td.MipLevels        = 1;
        td.Format           = DXGI_FORMAT_R16G16_SINT;
        td.SampleDesc.Count = 1;
        ThrowIfFailed(dev->CreateCommittedResource(
            &nv_helpers_dx12::kDefaultHeapProps, D3D12_HEAP_FLAG_NONE,
            &td, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
            IID_PPV_ARGS(&m_reuseTexture[i])));
        std::wstring name = L"ReuseTexture_" + std::to_wstring(i);
        m_reuseTexture[i]->SetName(name.c_str());

        // Upload heap + staged copy
        UINT64 uploadSize = GetRequiredIntermediateSize(m_reuseTexture[i].Get(), 0, 1);
        m_reuseTextureUploadHeaps.emplace_back();
        auto& uh = m_reuseTextureUploadHeaps.back();
        auto  ub = CD3DX12_RESOURCE_DESC::Buffer(uploadSize);
        ThrowIfFailed(dev->CreateCommittedResource(
            &nv_helpers_dx12::kUploadHeapProps, D3D12_HEAP_FLAG_NONE,
            &ub, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
            IID_PPV_ARGS(&uh)));

        D3D12_SUBRESOURCE_DATA sr = {};
        sr.pData      = rg.data();
        sr.RowPitch   = static_cast<LONG_PTR>(kSizes[i]) * sizeof(int16_t) * 2;
        sr.SlicePitch = sr.RowPitch * kSizes[i];
        UpdateSubresources(m_ctx.CmdList(), m_reuseTexture[i].Get(),
                           uh.Get(), 0, 0, 1, &sr);

        auto bar = CD3DX12_RESOURCE_BARRIER::Transition(
            m_reuseTexture[i].Get(), D3D12_RESOURCE_STATE_COPY_DEST, kSRV);
        m_ctx.CmdList()->ResourceBarrier(1, &bar);
    }
}

//====================================
//STAR / MILKY WAY SKYBOX (NASA SVS EXR)
//====================================
//Loads an equirectangular EXR (e.g. NASA SVS 4851 Deep Star Maps, 16K) and
//uploads it as DXGI_FORMAT_R16G16B16A16_FLOAT. Sampled in EvaluateStars
//(SunSampler_v8.hlsli) via gSkyStars at register t40, descriptor heap slot
//SKY_STARS_HEAP_SLOT. Path is fixed so users can drop replacement files in
//place without rebuilding. On failure the descriptor is left as a null SRV;
//the shader returns black and the night sky reads as just the night base.

//Runtime working directory is the build dir, with `include/*` flattened in
//(same convention the GLB loader uses with paths like "./harbor2.glb").
//Resolution agnostic name so swapping 4k/8k/16k variants doesn't need a
//code change. Just drop the new EXR in include/ as sky_stars.exr and
//rebuild (the include/ copy step picks it up). The loader logs the actual
//dimensions on startup.
static constexpr const char* SKY_STARS_EXR_PATH = "./sky_stars.exr";

void Renderer::InitSkyStarsTexture() {
    SCOPE_TIMER("InitSkyStarsTexture");

    if (!std::filesystem::exists(SKY_STARS_EXR_PATH)) {
        LOG(L"[SkyStars] EXR not found at " << SKY_STARS_EXR_PATH
            << L" -- sky will use night base only");
        return;
    }

    //TinyEXR loads RGBA float regardless of source channel count (alpha is
    //forced to 1 if absent). For a 4K x 2K texture this is ~128 MB of host
    //memory temporarily; we free it immediately after converting to half.
    float* rgbaF = nullptr;
    int    width = 0, height = 0;
    const char* errStr = nullptr;
    int ret = LoadEXR(&rgbaF, &width, &height, SKY_STARS_EXR_PATH, &errStr);
    if (ret != TINYEXR_SUCCESS) {
        LOG(L"[SkyStars] LoadEXR failed: "
            << (errStr ? std::string(errStr).c_str() : "(no error string)"));
        if (errStr) FreeEXRErrorMessage(errStr);
        return;
    }

    //float RGBA -> half RGBA for the base mip
    const size_t numFloats = (size_t)width * height * 4;
    std::vector<uint16_t> halfBase(numFloats);
    DirectX::PackedVector::XMConvertFloatToHalfStream(
        halfBase.data(),     sizeof(uint16_t),
        rgbaF,               sizeof(float),
        numFloats);
    free(rgbaF);

    //Mipmap generation: critical for clean rendering under camera jitter +
    //DLSS RR. Without mips, sub-pixel jitter samples random texels of bright
    //stars between frames, looking like noise to the temporal denoiser and
    //smearing. With a proper mip chain + footprint based LOD in the shader,
    //each pixel covers ~1 texel of the chosen mip so jitter only blends
    //inside the bilinear filter footprint, which is stable.
    DirectX::Image baseImg = {};
    baseImg.width      = (size_t)width;
    baseImg.height     = (size_t)height;
    baseImg.format     = DXGI_FORMAT_R16G16B16A16_FLOAT;
    baseImg.rowPitch   = (size_t)width * 8;
    baseImg.slicePitch = baseImg.rowPitch * (size_t)height;
    baseImg.pixels     = reinterpret_cast<uint8_t*>(halfBase.data());

    DirectX::ScratchImage mipChain;
    HRESULT hr = DirectX::GenerateMipMaps(baseImg, DirectX::TEX_FILTER_LINEAR,
                                          0 /*full chain*/, mipChain);
    const bool haveMips = SUCCEEDED(hr);
    const size_t mipCount = haveMips ? mipChain.GetImageCount() : 1u;
    if (!haveMips) {
        LOG(L"[SkyStars] GenerateMipMaps failed (hr=" << std::hex << hr
            << std::dec << L"), falling back to single mip");
    }
    LOG(L"[SkyStars] Loaded " << width << L"x" << height
        << L" EXR with " << mipCount << L" mip"
        << (mipCount == 1 ? L"" : L"s"));

    //R16G16B16A16_FLOAT default heap texture with the full mip chain
    auto* dev = m_ctx.Device();
    D3D12_RESOURCE_DESC td = {};
    td.Dimension        = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    td.Width            = (UINT64)width;
    td.Height           = (UINT)height;
    td.DepthOrArraySize = 1;
    td.MipLevels        = (UINT16)mipCount;
    td.Format           = DXGI_FORMAT_R16G16B16A16_FLOAT;
    td.SampleDesc.Count = 1;
    ThrowIfFailed(dev->CreateCommittedResource(
        &nv_helpers_dx12::kDefaultHeapProps, D3D12_HEAP_FLAG_NONE,
        &td, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
        IID_PPV_ARGS(&m_skyStarsTexture)));
    m_skyStarsTexture->SetName(L"SkyStarsTexture");

    //Upload heap sized for the entire mip chain
    UINT64 uploadSize = GetRequiredIntermediateSize(
        m_skyStarsTexture.Get(), 0, (UINT)mipCount);
    auto ub = CD3DX12_RESOURCE_DESC::Buffer(uploadSize);
    ThrowIfFailed(dev->CreateCommittedResource(
        &nv_helpers_dx12::kUploadHeapProps, D3D12_HEAP_FLAG_NONE,
        &ub, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
        IID_PPV_ARGS(&m_skyStarsUploadHeap)));
    m_skyStarsUploadHeap->SetName(L"SkyStarsUploadHeap");

    //Subresource descriptors for each mip
    std::vector<D3D12_SUBRESOURCE_DATA> sr(mipCount);
    if (haveMips) {
        for (size_t i = 0; i < mipCount; ++i) {
            const DirectX::Image* img = mipChain.GetImage(i, 0, 0);
            sr[i].pData      = img->pixels;
            sr[i].RowPitch   = (LONG_PTR)img->rowPitch;
            sr[i].SlicePitch = (LONG_PTR)img->slicePitch;
        }
    } else {
        sr[0].pData      = halfBase.data();
        sr[0].RowPitch   = (LONG_PTR)((UINT64)width * 8);
        sr[0].SlicePitch = sr[0].RowPitch * height;
    }

    UpdateSubresources(m_ctx.CmdList(), m_skyStarsTexture.Get(),
                       m_skyStarsUploadHeap.Get(), 0, 0, (UINT)mipCount, sr.data());

    auto bar = CD3DX12_RESOURCE_BARRIER::Transition(
        m_skyStarsTexture.Get(), D3D12_RESOURCE_STATE_COPY_DEST, kSRV);
    m_ctx.CmdList()->ResourceBarrier(1, &bar);
}

//====================================
//VOLUMETRIC CLOUD NOISE BAKE
//====================================
//Bakes the runtime cloud-noise 3D texture once at startup. The compute
//shader (Pass_cloudnoise_bake_v8.hlsl) evaluates the analytical
//Perlin/Worley/value noises with tiling hashes and writes the result
//into a 256³ RGBA8 texture. From that point on the cloud integrator
//replaces every Perlin/Worley evaluation with one Texture3D sample
//(~50-200× fewer ALU ops per density tap, which is the dominant cost
//in the volumetric march on the current shader).
//
//Why a private root sig / heap / PSO rather than reusing the global
//compute infrastructure: this runs BEFORE CreateShaderResourceHeap, so
//m_srvUavHeap doesn't exist yet. Allocating a UAV inside the persistent
//heap (which only ever needs an SRV view of the texture at runtime)
//would have meant either reordering init or wasting a heap slot. A
//throwaway 1-UAV heap with its own minimal root sig is simpler and
//keeps the bake completely decoupled from the rest of the pipeline.
void Renderer::BakeCloudNoiseTexture() {
    auto* dev = m_ctx.Device();
    constexpr UINT kRes = 256;
    const auto t_start = std::chrono::high_resolution_clock::now();
    const size_t bytes = (size_t)kRes * kRes * kRes * 4;
    LOG(L"[CloudNoise] Baking " << kRes << L"³ RGBA8 noise texture ("
        << (bytes / (1024 * 1024)) << L" MB, periods R/G/B/A = 32/16/48/32)...");

    //--- 1. Create the destination 3D texture (UAV-capable, default heap).
    D3D12_RESOURCE_DESC td = {};
    td.Dimension          = D3D12_RESOURCE_DIMENSION_TEXTURE3D;
    td.Width              = kRes;
    td.Height             = kRes;
    td.DepthOrArraySize   = kRes;
    td.MipLevels          = 1;
    td.Format             = DXGI_FORMAT_R8G8B8A8_UNORM;
    td.SampleDesc.Count   = 1;
    td.Layout             = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    td.Flags              = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    ThrowIfFailed(dev->CreateCommittedResource(
        &nv_helpers_dx12::kDefaultHeapProps, D3D12_HEAP_FLAG_NONE,
        &td, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
        IID_PPV_ARGS(&m_cloudNoiseTexture)));
    m_cloudNoiseTexture->SetName(L"CloudNoiseTexture");

    //--- 2. Private shader-visible heap for the bake's UAV (1 descriptor).
    {
        D3D12_DESCRIPTOR_HEAP_DESC hd = {};
        hd.NumDescriptors = 1;
        hd.Type           = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        hd.Flags          = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        ThrowIfFailed(dev->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&m_cloudNoiseBakeHeap)));

        D3D12_UNORDERED_ACCESS_VIEW_DESC ud = {};
        ud.ViewDimension     = D3D12_UAV_DIMENSION_TEXTURE3D;
        ud.Format            = DXGI_FORMAT_R8G8B8A8_UNORM;
        ud.Texture3D.WSize   = kRes;
        dev->CreateUnorderedAccessView(m_cloudNoiseTexture.Get(), nullptr, &ud,
            m_cloudNoiseBakeHeap->GetCPUDescriptorHandleForHeapStart());
    }

    //--- 3. Minimal root signature: one descriptor table holding one UAV
    //       at register(u0) — matches Pass_cloudnoise_bake_v8.hlsl.
    {
        CD3DX12_DESCRIPTOR_RANGE1 range;
        range.Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0, 0,
                   D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE);
        CD3DX12_ROOT_PARAMETER1 param;
        param.InitAsDescriptorTable(1, &range);

        CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC desc;
        desc.Init_1_1(1, &param, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE);

        ComPtr<ID3DBlob> sig, err;
        HRESULT hr = D3D12SerializeVersionedRootSignature(&desc, &sig, &err);
        if (FAILED(hr)) {
            if (err) OutputDebugStringA((char*)err->GetBufferPointer());
            ThrowIfFailed(hr);
        }
        ThrowIfFailed(dev->CreateRootSignature(0,
            sig->GetBufferPointer(), sig->GetBufferSize(),
            IID_PPV_ARGS(&m_cloudNoiseBakeSig)));
    }

    //--- 4. Compile the CS and build the compute PSO.
    {
        // Filename only — the codebase's shader-loader finds Pass_*.hlsl
        // files relative to the runtime working directory (matches the
        // PassSystem convention in Renderer.cpp).
        ComPtr<IDxcBlob> cs = nv_helpers_dx12::CompileCS(
            L"Pass_cloudnoise_bake_v8.hlsl", L"main");
        D3D12_COMPUTE_PIPELINE_STATE_DESC pd = {};
        pd.pRootSignature = m_cloudNoiseBakeSig.Get();
        pd.CS             = { cs->GetBufferPointer(), cs->GetBufferSize() };
        ThrowIfFailed(dev->CreateComputePipelineState(&pd,
            IID_PPV_ARGS(&m_cloudNoiseBakePSO)));
    }

    //--- 5. Dispatch on the init cmd list.
    auto* cmd = m_ctx.CmdList();
    ID3D12DescriptorHeap* heaps[] = { m_cloudNoiseBakeHeap.Get() };
    cmd->SetDescriptorHeaps(1, heaps);
    cmd->SetComputeRootSignature(m_cloudNoiseBakeSig.Get());
    cmd->SetPipelineState(m_cloudNoiseBakePSO.Get());
    cmd->SetComputeRootDescriptorTable(0,
        m_cloudNoiseBakeHeap->GetGPUDescriptorHandleForHeapStart());
    cmd->Dispatch(kRes / 8, kRes / 8, kRes / 8);

    //--- 6. UAV barrier + transition to SRV state. After this the texture
    //       is read-only for the rest of the renderer's lifetime, and the
    //       runtime SRV view created in CreateShaderResourceHeap sees the
    //       baked contents.
    D3D12_RESOURCE_BARRIER bars[2] = {};
    bars[0].Type          = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    bars[0].UAV.pResource = m_cloudNoiseTexture.Get();
    bars[1] = CD3DX12_RESOURCE_BARRIER::Transition(m_cloudNoiseTexture.Get(),
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, kSRV);
    cmd->ResourceBarrier(2, bars);

    const auto t_end = std::chrono::high_resolution_clock::now();
    const auto cpu_ms = std::chrono::duration_cast<std::chrono::microseconds>(
                            t_end - t_start).count() / 1000.0;
    LOG(L"[CloudNoise] Recorded bake dispatch in " << cpu_ms
        << L" ms CPU (GPU work completes at init cmd list flush)");
}

//====================================
//SPATIOTEMPORAL BLUE NOISE ARRAY (CloudRand4 source)
//====================================
//Bakes a 128x128x64 RGBA8 Texture2DArray of spatiotemporal blue noise. The
//runtime cloud shader's CloudRand4 fetches from this array at register t41
//instead of evaluating a white noise hash, so the cone shadow taps and per
//pixel step jitter carry a blue noise spatial spectrum. DLSS RR's spatial
//filter cleanly removes blue noise (energy lives in the high frequencies
//the filter passes) but leaves visible per pixel grain on white noise.
//
//Single dispatch — see Pass_stbn_bake_v8.hlsl for the bake algorithm and
//its limitations vs true void and cluster STBN.
//
//Mirrors the BakeCloudNoiseTexture structure: private 1 UAV heap, root
//signature, and PSO, because the bake runs before CreateShaderResourceHeap.
void Renderer::BakeCloudSTBNTexture() {
    SCOPE_TIMER("BakeCloudSTBNTexture");
    auto* dev = m_ctx.Device();

    constexpr UINT kW      = 128;
    constexpr UINT kH      = 128;
    constexpr UINT kSlices = 64;

    const auto t_start = std::chrono::high_resolution_clock::now();
    LOG(L"[CloudSTBN] Baking " << kW << L"x" << kH << L"x" << kSlices
        << L" RGBA8 spatiotemporal blue noise array ("
        << ((size_t)kW * kH * kSlices * 4u / 1024u) << L" KB)...");

    //--- 1. Create the destination Texture2DArray (UAV-capable).
    D3D12_RESOURCE_DESC td = {};
    td.Dimension          = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    td.Width              = kW;
    td.Height             = kH;
    td.DepthOrArraySize   = (UINT16)kSlices;
    td.MipLevels          = 1;
    td.Format             = DXGI_FORMAT_R8G8B8A8_UNORM;
    td.SampleDesc.Count   = 1;
    td.Layout             = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    td.Flags              = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    ThrowIfFailed(dev->CreateCommittedResource(
        &nv_helpers_dx12::kDefaultHeapProps, D3D12_HEAP_FLAG_NONE,
        &td, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
        IID_PPV_ARGS(&m_cloudSTBNTexture)));
    m_cloudSTBNTexture->SetName(L"CloudSTBNTexture");

    //--- 2. Private shader-visible heap for the bake's UAV (1 descriptor).
    {
        D3D12_DESCRIPTOR_HEAP_DESC hd = {};
        hd.NumDescriptors = 1;
        hd.Type           = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        hd.Flags          = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        ThrowIfFailed(dev->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&m_cloudSTBNBakeHeap)));

        D3D12_UNORDERED_ACCESS_VIEW_DESC ud = {};
        ud.ViewDimension                  = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
        ud.Format                         = DXGI_FORMAT_R8G8B8A8_UNORM;
        ud.Texture2DArray.ArraySize       = kSlices;
        dev->CreateUnorderedAccessView(m_cloudSTBNTexture.Get(), nullptr, &ud,
            m_cloudSTBNBakeHeap->GetCPUDescriptorHandleForHeapStart());
    }

    //--- 3. Minimal root signature: one descriptor table holding one UAV at
    //       register(u0) — matches Pass_stbn_bake_v8.hlsl.
    {
        CD3DX12_DESCRIPTOR_RANGE1 range;
        range.Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0, 0,
                   D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE);
        CD3DX12_ROOT_PARAMETER1 param;
        param.InitAsDescriptorTable(1, &range);

        CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC desc;
        desc.Init_1_1(1, &param, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE);

        ComPtr<ID3DBlob> sig, err;
        HRESULT hr = D3D12SerializeVersionedRootSignature(&desc, &sig, &err);
        if (FAILED(hr)) {
            if (err) OutputDebugStringA((char*)err->GetBufferPointer());
            ThrowIfFailed(hr);
        }
        ThrowIfFailed(dev->CreateRootSignature(0,
            sig->GetBufferPointer(), sig->GetBufferSize(),
            IID_PPV_ARGS(&m_cloudSTBNBakeSig)));
    }

    //--- 4. Compile the CS and build the compute PSO.
    {
        ComPtr<IDxcBlob> cs = nv_helpers_dx12::CompileCS(
            L"Pass_stbn_bake_v8.hlsl", L"main");
        D3D12_COMPUTE_PIPELINE_STATE_DESC pd = {};
        pd.pRootSignature = m_cloudSTBNBakeSig.Get();
        pd.CS             = { cs->GetBufferPointer(), cs->GetBufferSize() };
        ThrowIfFailed(dev->CreateComputePipelineState(&pd,
            IID_PPV_ARGS(&m_cloudSTBNBakePSO)));
    }

    //--- 5. Dispatch on the init cmd list.
    auto* cmd = m_ctx.CmdList();
    ID3D12DescriptorHeap* heaps[] = { m_cloudSTBNBakeHeap.Get() };
    cmd->SetDescriptorHeaps(1, heaps);
    cmd->SetComputeRootSignature(m_cloudSTBNBakeSig.Get());
    cmd->SetPipelineState(m_cloudSTBNBakePSO.Get());
    cmd->SetComputeRootDescriptorTable(0,
        m_cloudSTBNBakeHeap->GetGPUDescriptorHandleForHeapStart());
    cmd->Dispatch(kW / 8, kH / 8, kSlices);

    //--- 6. UAV barrier + transition to SRV state.
    D3D12_RESOURCE_BARRIER bars[2] = {};
    bars[0].Type          = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    bars[0].UAV.pResource = m_cloudSTBNTexture.Get();
    bars[1] = CD3DX12_RESOURCE_BARRIER::Transition(m_cloudSTBNTexture.Get(),
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, kSRV);
    cmd->ResourceBarrier(2, bars);

    const auto t_end = std::chrono::high_resolution_clock::now();
    const auto cpu_ms = std::chrono::duration_cast<std::chrono::microseconds>(
                            t_end - t_start).count() / 1000.0;
    LOG(L"[CloudSTBN] Recorded bake dispatch in " << cpu_ms
        << L" ms CPU (GPU work completes at init cmd list flush)");
}

//====================================
//PLANET-SCALE CLOUD COVERAGE MAP (NASA Blue Marble)
//====================================
//Loads the NASA Blue Marble "cloud_combined_8192.tif" equirectangular cloud
//cover image. The TIFF is RGB color (clouds rendered as white/grey opacity
//over dark Earth), so we collapse to single-channel R8 luminance during the
//upload — gives us a 33 MB texture instead of 134 MB and matches the
//"coverage scalar" semantic the shader needs. The shader samples this with
//hardware bilinear filtering, so coverage transitions between source texels
//appear as smooth gradients rather than blocky steps.
//
//Path is fixed at include/cloud_coverage.tif (CMake downloads on configure).
//On failure the texture stays null and CreateShaderResourceHeap binds a null
//SRV — the shader treats that as "no coverage map" and falls back to pure
//procedural coverage so the missing-file path is non-fatal.

static constexpr const wchar_t* CLOUD_COVERAGE_TIF_PATH = L"./cloud_coverage.tif";

//Bind-a-fallback helper: when the TIFF is missing or fails to decode we
//still need *some* texture in the SRV slot so the shader sample evaluates
//to a useful constant (rather than 0, which would mean "no clouds anywhere").
//A 1×1 mid-grey (0.5) makes `saturate(base * map * 2.0)` collapse to the
//user's `base` coverage — i.e., the map has no effect and the renderer
//behaves as if the map system wasn't there. Identical visual outcome to
//pre-coverage-map behaviour.
void Renderer::CreateCloudCoverageFallback(uint8_t value) {
    auto* dev = m_ctx.Device();
    D3D12_RESOURCE_DESC td = {};
    td.Dimension          = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    td.Width              = 1;
    td.Height             = 1;
    td.DepthOrArraySize   = 1;
    td.MipLevels          = 1;
    td.Format             = DXGI_FORMAT_R8_UNORM;
    td.SampleDesc.Count   = 1;
    ThrowIfFailed(dev->CreateCommittedResource(
        &nv_helpers_dx12::kDefaultHeapProps, D3D12_HEAP_FLAG_NONE,
        &td, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
        IID_PPV_ARGS(&m_cloudCoverageTexture)));
    m_cloudCoverageTexture->SetName(L"CloudCoverageTexture(Fallback)");

    UINT64 uploadSize = GetRequiredIntermediateSize(
        m_cloudCoverageTexture.Get(), 0, 1);
    auto ub = CD3DX12_RESOURCE_DESC::Buffer(uploadSize);
    ThrowIfFailed(dev->CreateCommittedResource(
        &nv_helpers_dx12::kUploadHeapProps, D3D12_HEAP_FLAG_NONE,
        &ub, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
        IID_PPV_ARGS(&m_cloudCoverageUploadHeap)));

    D3D12_SUBRESOURCE_DATA sr = {};
    sr.pData      = &value;
    sr.RowPitch   = 1;
    sr.SlicePitch = 1;
    UpdateSubresources(m_ctx.CmdList(), m_cloudCoverageTexture.Get(),
                       m_cloudCoverageUploadHeap.Get(), 0, 0, 1, &sr);
    auto bar = CD3DX12_RESOURCE_BARRIER::Transition(
        m_cloudCoverageTexture.Get(), D3D12_RESOURCE_STATE_COPY_DEST, kSRV);
    m_ctx.CmdList()->ResourceBarrier(1, &bar);
}

void Renderer::InitCloudCoverageTexture() {
    SCOPE_TIMER("InitCloudCoverageTexture");
    const auto t_start = std::chrono::high_resolution_clock::now();

    if (!std::filesystem::exists(CLOUD_COVERAGE_TIF_PATH)) {
        LOG(L"[CloudCoverage] TIFF not found at " << CLOUD_COVERAGE_TIF_PATH
            << L" — binding 1×1 grey fallback (clouds use procedural coverage only)");
        CreateCloudCoverageFallback(128);
        return;
    }

    //Decode via DirectXTex's WIC backend (handles TIFF, JPG, PNG, etc.).
    //WIC_FLAGS_NONE lets WIC pick the best in-memory format from the source;
    //we'll convert to RGBA8 below if it doesn't land there directly.
    DirectX::TexMetadata meta = {};
    DirectX::ScratchImage scratch;
    HRESULT hr = DirectX::LoadFromWICFile(CLOUD_COVERAGE_TIF_PATH,
        DirectX::WIC_FLAGS_NONE, &meta, scratch);
    if (FAILED(hr)) {
        LOG(L"[CloudCoverage] LoadFromWICFile failed (HRESULT 0x"
            << std::hex << hr << std::dec
            << L") — binding 1×1 grey fallback");
        CreateCloudCoverageFallback(128);
        return;
    }

    //Convert to RGBA8 if WIC handed us anything else (TIFF can be 24bpp RGB
    //which DXGI doesn't have a direct format for, so WIC may give us 32bpp
    //BGRA or even something exotic). RGBA8 is the lowest common denominator.
    if (meta.format != DXGI_FORMAT_R8G8B8A8_UNORM) {
        DirectX::ScratchImage rgba;
        hr = DirectX::Convert(*scratch.GetImage(0, 0, 0),
            DXGI_FORMAT_R8G8B8A8_UNORM,
            DirectX::TEX_FILTER_DEFAULT, DirectX::TEX_THRESHOLD_DEFAULT, rgba);
        if (FAILED(hr)) {
            LOG(L"[CloudCoverage] DirectX::Convert to RGBA8 failed (HRESULT 0x"
                << std::hex << hr << std::dec
                << L") — binding 1×1 grey fallback");
            CreateCloudCoverageFallback(128);
            return;
        }
        scratch = std::move(rgba);
        meta    = scratch.GetMetadata();
    }

    const UINT width  = (UINT)meta.width;
    const UINT height = (UINT)meta.height;
    LOG(L"[CloudCoverage] Loaded " << width << L"x" << height
        << L" TIFF (" << (scratch.GetPixelsSize() / (1024 * 1024))
        << L" MB RGBA8 on CPU)");

    //Collapse RGBA → R8 luminance on the CPU. Rec. 709 luminance weights
    //(0.2126/0.7152/0.0722) so cloud highlights map to ~1 and dark Earth
    //maps to ~0. The output is a flat 8 MB → 33 MB R8 buffer ready for
    //upload as DXGI_FORMAT_R8_UNORM.
    const uint8_t* src = scratch.GetImage(0, 0, 0)->pixels;
    const size_t srcRowPitch = scratch.GetImage(0, 0, 0)->rowPitch;
    std::vector<uint8_t> r8(width * height);
    for (UINT y = 0; y < height; ++y) {
        const uint8_t* srow = src + y * srcRowPitch;
        uint8_t*       drow = r8.data() + y * width;
        for (UINT x = 0; x < width; ++x) {
            const float r = srow[x * 4 + 0] / 255.0f;
            const float g = srow[x * 4 + 1] / 255.0f;
            const float b = srow[x * 4 + 2] / 255.0f;
            const float lum = 0.2126f * r + 0.7152f * g + 0.0722f * b;
            drow[x] = (uint8_t)(std::clamp(lum, 0.0f, 1.0f) * 255.0f + 0.5f);
        }
    }

    //Create the GPU texture (R8_UNORM, no mips — the shader samples this at
    //fairly low frequency relative to the source, and the path tracer's TAA
    //hides any residual minification aliasing).
    auto* dev = m_ctx.Device();
    D3D12_RESOURCE_DESC td = {};
    td.Dimension          = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    td.Width              = width;
    td.Height             = height;
    td.DepthOrArraySize   = 1;
    td.MipLevels          = 1;
    td.Format             = DXGI_FORMAT_R8_UNORM;
    td.SampleDesc.Count   = 1;
    ThrowIfFailed(dev->CreateCommittedResource(
        &nv_helpers_dx12::kDefaultHeapProps, D3D12_HEAP_FLAG_NONE,
        &td, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
        IID_PPV_ARGS(&m_cloudCoverageTexture)));
    m_cloudCoverageTexture->SetName(L"CloudCoverageTexture");

    UINT64 uploadSize = GetRequiredIntermediateSize(
        m_cloudCoverageTexture.Get(), 0, 1);
    auto ub = CD3DX12_RESOURCE_DESC::Buffer(uploadSize);
    ThrowIfFailed(dev->CreateCommittedResource(
        &nv_helpers_dx12::kUploadHeapProps, D3D12_HEAP_FLAG_NONE,
        &ub, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
        IID_PPV_ARGS(&m_cloudCoverageUploadHeap)));
    m_cloudCoverageUploadHeap->SetName(L"CloudCoverageUploadHeap");

    D3D12_SUBRESOURCE_DATA sr = {};
    sr.pData      = r8.data();
    sr.RowPitch   = (LONG_PTR)width;        // R8: one byte per texel
    sr.SlicePitch = sr.RowPitch * height;
    UpdateSubresources(m_ctx.CmdList(), m_cloudCoverageTexture.Get(),
                       m_cloudCoverageUploadHeap.Get(), 0, 0, 1, &sr);

    auto bar = CD3DX12_RESOURCE_BARRIER::Transition(
        m_cloudCoverageTexture.Get(), D3D12_RESOURCE_STATE_COPY_DEST, kSRV);
    m_ctx.CmdList()->ResourceBarrier(1, &bar);

    const auto t_end = std::chrono::high_resolution_clock::now();
    const auto ms = std::chrono::duration_cast<std::chrono::microseconds>(
                        t_end - t_start).count() / 1000.0;
    LOG(L"[CloudCoverage] Uploaded " << width << L"x" << height
        << L" R8 luminance ("
        << ((size_t)width * height / (1024 * 1024))
        << L" MB) in " << ms << L" ms");
}

//====================================
//READBACK AND SIMULATION DATA
//====================================

void Renderer::CreateReadbackBuffer() {
    D3D12_RESOURCE_DESC td = m_scratchPing->GetDesc();
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp;
    UINT64 totalBytes = 0;
    m_ctx.Device()->GetCopyableFootprints(&td, 0, 1, 0, &fp, nullptr, nullptr, &totalBytes);

    auto hp = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK);
    auto bd = CD3DX12_RESOURCE_DESC::Buffer(totalBytes);
    ThrowIfFailed(m_ctx.Device()->CreateCommittedResource(
        &hp, D3D12_HEAP_FLAG_NONE, &bd,
        D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&m_readbackBuffer)));
}

void Renderer::SaveSimulationData(uint32_t stepIndex) {
    namespace fs = std::filesystem;
    if (!fs::exists("output")) fs::create_directory("output");

    D3D12_RESOURCE_DESC td = m_scratchPing->GetDesc();
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp;
    m_ctx.Device()->GetCopyableFootprints(&td, 0, 1, 0, &fp, nullptr, nullptr, nullptr);

    UINT width = (UINT)td.Width, height = td.Height, rowPitch = fp.Footprint.RowPitch;
    auto* cmdList = m_ctx.CmdList();

    auto ProcessSlice = [&](UINT slice, auto func) {
        auto b1 = CD3DX12_RESOURCE_BARRIER::Transition(m_scratchPing.Get(),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        cmdList->ResourceBarrier(1, &b1);

        UINT sub = D3D12CalcSubresource(0, slice, 0, 1, td.DepthOrArraySize);
        CD3DX12_TEXTURE_COPY_LOCATION dst(m_readbackBuffer.Get(), fp);
        CD3DX12_TEXTURE_COPY_LOCATION src(m_scratchPing.Get(), sub);
        cmdList->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);

        auto b2 = CD3DX12_RESOURCE_BARRIER::Transition(m_scratchPing.Get(),
            D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        cmdList->ResourceBarrier(1, &b2);

        m_ctx.FlushAndReset();

        uint8_t* p = nullptr;
        CD3DX12_RANGE rr(0, fp.Footprint.RowPitch * height);
        ThrowIfFailed(m_readbackBuffer->Map(0, &rr, (void**)&p));
        func(p);
        CD3DX12_RANGE wr(0,0);
        m_readbackBuffer->Unmap(0, &wr);
    };

    auto WriteBin = [&](const std::string& suffix, const std::vector<float>& data) {
        std::ofstream f("output/" + std::to_string(stepIndex) + "_" + suffix + ".bin", std::ios::binary);
        if (f) f.write((const char*)data.data(), data.size() * sizeof(float));
    };

    ProcessSlice(7, [&](uint8_t* rd) {
        std::vector<float> a(width*height), b(width*height), c(width*height), d(width*height);
        for (UINT y = 0; y < height; ++y) {
            float* row = (float*)(rd + y * rowPitch);
            for (UINT x = 0; x < width; ++x) {
                a[y*width+x] = row[x*4]; b[y*width+x] = row[x*4+1];
                c[y*width+x] = row[x*4+2]; d[y*width+x] = row[x*4+3];
            }
        }
        WriteBin("restir", a); WriteBin("gt", b); WriteBin("init", c); WriteBin("albedo", d);
    });

    ProcessSlice(8, [&](uint8_t* rd) {
        std::vector<float> e(width*height), f(width*height);
        for (UINT y = 0; y < height; ++y) {
            float* row = (float*)(rd + y * rowPitch);
            for (UINT x = 0; x < width; ++x) {
                e[y*width+x] = row[x*4]; f[y*width+x] = row[x*4+1];
            }
        }
        WriteBin("roughness", e); WriteBin("depth", f);
    });

    ProcessSlice(9, [&](uint8_t* rd) {
        std::vector<float> g(width*height*3);
        for (UINT y = 0; y < height; ++y) {
            float* row = (float*)(rd + y * rowPitch);
            for (UINT x = 0; x < width; ++x) {
                size_t idx = (y*width+x)*3;
                g[idx] = row[x*4]; g[idx+1] = row[x*4+1]; g[idx+2] = row[x*4+2];
            }
        }
        WriteBin("normal", g);
    });
}
