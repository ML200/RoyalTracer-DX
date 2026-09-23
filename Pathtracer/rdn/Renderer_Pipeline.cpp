#include "stdafx.h"
#include "Renderer.h"
#include "Scene/OmmBuilder.h"
#include "nv_helpers_dx12/BottomLevelASGenerator.h"
#include "nv_helpers_dx12/RaytracingPipelineGenerator.h"
#include "nv_helpers_dx12/RootSignatureGenerator.h"
#include <algorithm>
#include <cmath>
#include <chrono>
#include <limits>
#include <fstream>
#include <filesystem>
#include <random>
#include <unordered_set>
#include <d3dcompiler.h>
#include <DirectXPackedVector.h>
#include "../DirectXTex/DirectXTex.h"

#define TINYEXR_USE_MINIZ 0
#define TINYEXR_USE_STB_ZLIB 1
#include "../lib/tinyexr/tinyexr.h"

static constexpr bool kUseBlasCompaction = true;

// Keep opaque and alpha geometry in the shader binding table's order.
Renderer::AccelerationStructureBuffers
Renderer::CreateBottomLevelAS(const std::vector<std::pair<ComPtr<ID3D12Resource>, uint32_t>>& vVertexBuffers,
                              const std::vector<std::pair<ComPtr<ID3D12Resource>, uint32_t>>& vIndexBuffers,
                              UINT opaqueTriCount, UINT alphaTriCount, MeshGPU* meshOmm) {
    if (vVertexBuffers.size() != vIndexBuffers.size())
        throw std::invalid_argument("BLAS vertex/index buffer counts differ");

    nv_helpers_dx12::BottomLevelASGenerator blasGen;

    for (size_t i = 0; i < vVertexBuffers.size(); i++) {
        UINT opaqueIdxCount = opaqueTriCount * 3;
        UINT alphaIdxCount = alphaTriCount * 3;

        if (opaqueIdxCount > 0)
            blasGen.AddVertexBuffer(vVertexBuffers[i].first.Get(), 0, vVertexBuffers[i].second, sizeof(Vertex),
                                    vIndexBuffers[i].first.Get(), 0, opaqueIdxCount, nullptr, 0, true);

        if (alphaIdxCount > 0) {
            if (meshOmm && meshOmm->hasOmm) {
                blasGen.AddVertexBufferWithOMM(
                    vVertexBuffers[i].first.Get(), 0, vVertexBuffers[i].second, sizeof(Vertex),
                    vIndexBuffers[i].first.Get(), opaqueIdxCount * sizeof(UINT), alphaIdxCount, nullptr, 0,
                    meshOmm->ommArray ? meshOmm->ommArray->GetGPUVirtualAddress() : 0,
                    meshOmm->ommIndexBuffer->GetGPUVirtualAddress(), meshOmm->ommBake.alphaTriCount);
            } else {
                blasGen.AddVertexBuffer(vVertexBuffers[i].first.Get(), 0, vVertexBuffers[i].second, sizeof(Vertex),
                                        vIndexBuffers[i].first.Get(), opaqueIdxCount * sizeof(UINT), alphaIdxCount,
                                        nullptr, 0, false);
            }
        }
    }

    UINT64 scratchSize = 0, resultSize = 0;
    auto buildFlags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
    if constexpr (kUseBlasCompaction)
        buildFlags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_COMPACTION;

    blasGen.ComputeASBufferSizes(m_ctx.Device(), buildFlags, &scratchSize, &resultSize);
    std::wcout << L"[BLAS]     sizes scratch=" << (scratchSize / (1024 * 1024)) << L" MB" << L" result="
               << (resultSize / (1024 * 1024)) << L" MB" << std::endl;

    AccelerationStructureBuffers buffers;
    buffers.pScratch =
        nv_helpers_dx12::CreateBuffer(m_ctx.Device(), scratchSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                      D3D12_RESOURCE_STATE_COMMON, nv_helpers_dx12::kDefaultHeapProps);

    if constexpr (!kUseBlasCompaction) {
        buffers.pResult = nv_helpers_dx12::CreateBuffer(
            m_ctx.Device(), resultSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, nv_helpers_dx12::kDefaultHeapProps);
        blasGen.Generate(m_ctx.CmdList(), buffers.pScratch.Get(), buffers.pResult.Get(), false, nullptr);
    } else {
        buffers.pResultUncompacted = nv_helpers_dx12::CreateBuffer(
            m_ctx.Device(), resultSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, nv_helpers_dx12::kDefaultHeapProps);
        std::wcout << L"[BLAS]     uncompacted result buf created (" << buffers.pResultUncompacted.Get() << L")"
                   << std::endl;

        ComPtr<ID3D12Resource> compactedSizeBuf =
            nv_helpers_dx12::CreateBuffer(m_ctx.Device(), sizeof(UINT64), D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                          D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nv_helpers_dx12::kDefaultHeapProps);

        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC postInfo = {};
        postInfo.DestBuffer = compactedSizeBuf->GetGPUVirtualAddress();
        postInfo.InfoType = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_COMPACTED_SIZE;

        blasGen.Generate(m_ctx.CmdList(), buffers.pScratch.Get(), buffers.pResultUncompacted.Get(), false, nullptr);

        D3D12_GPU_VIRTUAL_ADDRESS src = buffers.pResultUncompacted->GetGPUVirtualAddress();
        m_ctx.CmdList()->EmitRaytracingAccelerationStructurePostbuildInfo(&postInfo, 1, &src);

        ComPtr<ID3D12Resource> readback =
            nv_helpers_dx12::CreateBuffer(m_ctx.Device(), sizeof(UINT64), D3D12_RESOURCE_FLAG_NONE,
                                          D3D12_RESOURCE_STATE_COPY_DEST, nv_helpers_dx12::kReadbackHeapProps);

        auto barrier = CD3DX12_RESOURCE_BARRIER::Transition(
            compactedSizeBuf.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        m_ctx.CmdList()->ResourceBarrier(1, &barrier);
        m_ctx.CmdList()->CopyResource(readback.Get(), compactedSizeBuf.Get());

        m_ctx.FlushAndReset();

        // The GPU-reported size is available only after the build fence.
        UINT64 compactedSize;
        void* pMap;
        ThrowIfFailed(readback->Map(0, nullptr, &pMap));
        memcpy(&compactedSize, pMap, sizeof(UINT64));
        readback->Unmap(0, nullptr);
        std::wcout << L"[BLAS]     compactedSize=" << (compactedSize / (1024 * 1024)) << L" MB" << std::endl;

        buffers.pResult = nv_helpers_dx12::CreateBuffer(
            m_ctx.Device(), compactedSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, nv_helpers_dx12::kDefaultHeapProps);
        buffers.pResult->SetName(L"Compacted BLAS");

        m_ctx.CmdList()->CopyRaytracingAccelerationStructure(buffers.pResult->GetGPUVirtualAddress(),
                                                             buffers.pResultUncompacted->GetGPUVirtualAddress(),
                                                             D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_COMPACT);

        auto uavB = CD3DX12_RESOURCE_BARRIER::UAV(buffers.pResult.Get());
        m_ctx.CmdList()->ResourceBarrier(1, &uavB);
    }

    return buffers;
}

void Renderer::CreateTopLevelAS(const std::vector<Scene::TLASInstance>& instances, bool updateOnly) {
    std::wcout << L"[TLAS] CreateTopLevelAS ENTER: instances=" << instances.size() << L" updateOnly="
               << (updateOnly ? L"yes" : L"no") << std::endl;

    if (!updateOnly) {
        const bool logEachInstance = instances.size() <= 32;
        for (size_t i = 0; i < instances.size(); i++) {
            if (logEachInstance)
                std::wcout << L"[TLAS]   instance[" << i << L"] blas=" << instances[i].blas.Get() << L" hitGroup="
                           << instances[i].hitGroupContribution << L" flags=" << (UINT)instances[i].flags << std::endl;
            m_topLevelASGenerator.AddInstance(instances[i].blas.Get(), instances[i].transform, static_cast<UINT>(i),
                                              instances[i].hitGroupContribution, instances[i].flags);
        }

        UINT64 scratchSize, resultSize, instanceDescsSize;
        m_topLevelASGenerator.ComputeASBufferSizes(m_ctx.Device(), true, &scratchSize, &resultSize, &instanceDescsSize);
        std::wcout << L"[TLAS]   sizes scratch=" << (scratchSize / 1024) << L" KB" << L" result=" << (resultSize / 1024)
                   << L" KB" << L" instanceDescs=" << instanceDescsSize << L" B" << std::endl;

        m_topLevelASBuffers.pScratch =
            nv_helpers_dx12::CreateBuffer(m_ctx.Device(), scratchSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                          D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nv_helpers_dx12::kDefaultHeapProps);
        m_topLevelASBuffers.pResult = nv_helpers_dx12::CreateBuffer(
            m_ctx.Device(), resultSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, nv_helpers_dx12::kDefaultHeapProps);
        m_topLevelASBuffers.pInstanceDesc =
            nv_helpers_dx12::CreateBuffer(m_ctx.Device(), instanceDescsSize, D3D12_RESOURCE_FLAG_NONE,
                                          D3D12_RESOURCE_STATE_GENERIC_READ, nv_helpers_dx12::kUploadHeapProps);
    }

    std::wcout << L"[TLAS]   calling m_topLevelASGenerator.Generate..." << std::endl;
    m_topLevelASGenerator.Generate(m_ctx.CmdList(), m_topLevelASBuffers.pScratch.Get(),
                                   m_topLevelASBuffers.pResult.Get(), m_topLevelASBuffers.pInstanceDesc.Get(),
                                   updateOnly, m_topLevelASBuffers.pResult.Get());
    std::wcout << L"[TLAS]   Generate returned. TLAS=" << m_topLevelASBuffers.pResult.Get() << std::endl;
}

void Renderer::CreateAccelerationStructures() {
    std::wcout << L"[AS] CreateAccelerationStructures ENTER: meshes=" << m_scene.meshes.size() << L" instances="
               << m_scene.instances.size() << std::endl;

    // Retain the source allocation until its compaction copy completes.
    ComPtr<ID3D12Resource> pendingCompactionSource;

    for (size_t m = 0; m < m_scene.meshes.size(); ++m) {
        auto& mesh = m_scene.meshes[m];
        std::wcout << L"[AS] mesh[" << m << L"/" << m_scene.meshes.size() << L"] " << L"verts=" << mesh.vertexCount
                   << L" indices=" << mesh.indexCount << L" opaqueTris=" << mesh.opaqueTriCount << L" alphaTris="
                   << mesh.alphaTriCount << L" totalTris=" << (mesh.opaqueTriCount + mesh.alphaTriCount) << std::endl;
        if (mesh.blas) {
            std::wcout << L"[AS]   already has BLAS, skipping" << std::endl;
            continue;
        }
        mesh.CreateBlasBuildInputs(m_ctx.Device());

        OmmGpuData ommGpu;
        if (!mesh.ommBake.triOmmIndices.empty()) {
            ommGpu = OmmBuilder::BuildGPU(mesh.ommBake, m_ctx.Device(), m_ctx.CmdList());
            if (ommGpu.valid) {
                mesh.ommArray = ommGpu.ommArray;
                mesh.ommIndexBuffer = ommGpu.ommIndexBuffer;
                mesh.hasOmm = true;
            }
        }

        MeshGPU* ommPtr = mesh.hasOmm ? &mesh : nullptr;
        auto buffers = CreateBottomLevelAS({{mesh.vertexBuffer.Get(), mesh.vertexCount}},
                                           {{mesh.indexBuffer.Get(), mesh.indexCount}}, mesh.opaqueTriCount,
                                           mesh.alphaTriCount, ommPtr);
        if constexpr (kUseBlasCompaction) {
            pendingCompactionSource = std::move(buffers.pResultUncompacted);
        } else {
            m_ctx.FlushAndReset(); // scratch/source lifetime for the non-compacted path
        }
        mesh.blas = buffers.pResult;

        mesh.vertexBuffer.Reset();
        mesh.indexBuffer.Reset();
        std::wcout << L"[AS]   mesh[" << m << L"] BLAS=" << mesh.blas.Get() << std::endl;
    }
    std::wcout << L"[AS] All BLAS built. Rebuilding TLAS instance list..." << std::endl;

    m_scene.RebuildTLASInstanceList();

    m_scene.CollectEmissiveTriangles();

    std::vector<InstanceXformCPU> ltXforms;
    ltXforms.reserve(m_scene.instances.size());

    const XMVECTOR ltShift =
        XMVectorSet(m_scene.sceneOriginWorld.x, m_scene.sceneOriginWorld.y, m_scene.sceneOriginWorld.z, 0.0f);
    for (const auto& si : m_scene.instances) {
        InstanceXformCPU x{};
        XMMATRIX shifted = si.worldTransform;
        shifted.r[3] = XMVectorSubtract(shifted.r[3], ltShift);
        XMStoreFloat4x4(&x.objectToWorld, shifted);
        ltXforms.push_back(x);
    }
    m_lightTreeCompact = m_integratorSettings.compactLightTree;
    lt::LightTreeBuilder::Settings lightTreeSettings;
    lightTreeSettings.compactGpuNodes = m_lightTreeCompact;
    m_lightTree.Build(m_scene.emissiveTriangles, m_scene.lightInstances, ltXforms, lightTreeSettings);
    m_publishedLightTLAS = m_lightTree.GetCpuTLASNodes();
    m_frameStats.lightBvh.slots = m_lightTree.SlotCount();
    m_lightTree.PrintMetrics();

    {
        SCOPE_TIMER("LightTree.UploadAll");
        m_lightTree.UploadAll(m_ctx.Device(), m_ctx.CmdList());
    }

    {
        SCOPE_TIMER("CreateEmissiveTrianglesBuffer");
        m_scene.CreateEmissiveTrianglesBuffer(m_ctx.Device(), m_ctx.CmdList(), m_ctx.CmdQueue(), nullptr);
    }

    {
        SCOPE_TIMER("CreateTriToLightIdBuffer");
        m_scene.CreateTriToLightIdBuffer(m_ctx.Device(), m_ctx.CmdList());
    }

    m_ctx.FlushAndReset();
    pendingCompactionSource.Reset();
    m_scene.ReleaseLightUploadStaging();
    m_lightTree.ReleaseStaging();

    {
        size_t freed = 0;
        for (auto& mesh : m_scene.meshes) {
            if (mesh.vertexBuffer) {
                mesh.vertexBuffer.Reset();
                ++freed;
            }
            if (mesh.indexBuffer) {
                mesh.indexBuffer.Reset();
            }
        }
        LOG(L"[AS] Released per-mesh source buffers on " << freed << L" meshes");
    }
}

// Shared descriptor layout for ray generation and compute passes.
ComPtr<ID3D12RootSignature> Renderer::CreateRayGenSignature() {
    CD3DX12_ROOT_PARAMETER1 rootParameters[4];
    std::vector<CD3DX12_DESCRIPTOR_RANGE1> ranges;
    ranges.reserve(40);
    const auto VOLATILE = D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE;
    const auto STATIC = D3D12_DESCRIPTOR_RANGE_FLAG_DATA_STATIC_WHILE_SET_AT_EXECUTE;
    // Buffer SRVs take volatile descriptors. With static ones the driver (NVIDIA 610.88, RTX 5090)
    // drops the bounds check of StructuredBuffer reads in raygen shaders, even with
    // DESCRIPTORS_STATIC_KEEPING_BUFFER_BOUNDS_CHECKS: a read past the end of a view returns the
    // memory behind it, and past the allocation it page-faults (the TDRs in LT_LoadTLAS /
    // LT_LoadBLAS). Compute shaders, UAVs and typed buffers stay bounds-checked either way.
    // Measured cost: none (1920x1080 raygen, 64 dependent reads each: 1.198 ms static, 1.197 ms
    // volatile).
    const auto BUFFER_SRV =
        D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE | D3D12_DESCRIPTOR_RANGE_FLAG_DATA_STATIC_WHILE_SET_AT_EXECUTE;

    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0, 0, VOLATILE,
                               D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 1, 0, VOLATILE,
                               D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, STATIC, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 1, 0, BUFFER_SRV, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 2, 0, BUFFER_SRV, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_CBV, 1, 0, 0, STATIC, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 3, 0, BUFFER_SRV, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 4, 0, BUFFER_SRV, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 5, 0, BUFFER_SRV, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 6, 0, BUFFER_SRV, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    for (UINT u = 2; u <= 7; ++u)
        ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, u, 0, VOLATILE,
                                   D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 7, 0, BUFFER_SRV, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 8, 0, BUFFER_SRV, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 8, 0, VOLATILE,
                               D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 9, 0, VOLATILE,
                               D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    for (UINT t = 9; t <= 12; ++t)
        ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, t, 0, BUFFER_SRV,
                                   D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 15, 0, BUFFER_SRV, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    for (UINT t = 16; t <= 18; ++t)
        ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, t, 0, BUFFER_SRV,
                                   D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    for (UINT t = 30; t <= 33; ++t)
        ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, t, 0, STATIC,
                                   D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 10, 0, VOLATILE,
                               D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 13, 11, 0, VOLATILE,
                               D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    // The reconstruction responsivity mask follows the guides in the heap but not in register
    // space: u24 already belongs to the auto-exposure buffer.
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 26, 0, VOLATILE,
                               D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);

    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 3, 19, 0, STATIC, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);

    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 24, 0, VOLATILE, AUTOEXPOSE_HEAP_SLOT);

    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 40, 0, STATIC, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);

    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 44, 0, STATIC, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);

    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 45, 0, STATIC, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 46, 0, STATIC, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 47, 0, STATIC, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 49, 0, VOLATILE,
                               D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 51, 0, VOLATILE,
                               D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);

    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 59, 0, STATIC, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);

    rootParameters[0].InitAsDescriptorTable((UINT)ranges.size(), ranges.data(), D3D12_SHADER_VISIBILITY_ALL);

    rootParameters[1].InitAsConstants(SHARC_ROOT_CONSTANTS, 1, 0, D3D12_SHADER_VISIBILITY_ALL);

    rootParameters[2].InitAsUnorderedAccessView(25, 0);

    rootParameters[3].InitAsUnorderedAccessView(27, 0);

    CD3DX12_STATIC_SAMPLER_DESC staticSamplers[3];
    staticSamplers[0].Init(0, D3D12_FILTER_ANISOTROPIC, D3D12_TEXTURE_ADDRESS_MODE_WRAP,
                           D3D12_TEXTURE_ADDRESS_MODE_WRAP, D3D12_TEXTURE_ADDRESS_MODE_WRAP);
    staticSamplers[0].MaxAnisotropy = 16;
    staticSamplers[1].Init(1, D3D12_FILTER_MIN_MAG_MIP_LINEAR, D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                           D3D12_TEXTURE_ADDRESS_MODE_CLAMP, D3D12_TEXTURE_ADDRESS_MODE_CLAMP);

    staticSamplers[2].Init(3, D3D12_FILTER_MIN_MAG_POINT_MIP_LINEAR, D3D12_TEXTURE_ADDRESS_MODE_WRAP,
                           D3D12_TEXTURE_ADDRESS_MODE_WRAP, D3D12_TEXTURE_ADDRESS_MODE_WRAP);

    CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC desc;

    desc.Init_1_1(_countof(rootParameters), rootParameters, _countof(staticSamplers), staticSamplers,
                  D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED |
                      D3D12_ROOT_SIGNATURE_FLAG_SAMPLER_HEAP_DIRECTLY_INDEXED);

    ComPtr<ID3DBlob> signature, error;
    HRESULT hr = D3D12SerializeVersionedRootSignature(&desc, &signature, &error);
    if (FAILED(hr)) {
        if (error)
            OutputDebugStringA(static_cast<char*>(error->GetBufferPointer()));
        ThrowIfFailed(hr);
    }

    ComPtr<ID3D12RootSignature> rs;
    ThrowIfFailed(m_ctx.Device()->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(),
                                                      IID_PPV_ARGS(&rs)));
    return rs;
}

ComPtr<ID3D12RootSignature> Renderer::CreateComputeSignature() {
    return CreateRayGenSignature();
}

ComPtr<ID3D12RootSignature> Renderer::CreateHitSignature() {
    nv_helpers_dx12::RootSignatureGenerator rsc;
    return rsc.Generate(m_ctx.Device(), true);
}

ComPtr<ID3D12RootSignature> Renderer::CreateMissSignature() {
    nv_helpers_dx12::RootSignatureGenerator rsc;
    return rsc.Generate(m_ctx.Device(), true);
}

void Renderer::CreateRaytracingPipeline() {
    m_skyLutsReady = false;
    m_lightLearningResetPending = true;
    m_sharcResetPending = true;
    nv_helpers_dx12::RayTracingPipelineGenerator pipeline(m_ctx.Device());

    m_rayGenSignature = CreateRayGenSignature();
    m_computeSignature = CreateComputeSignature();
    m_missSignature = CreateMissSignature();
    m_hitSignature = CreateHitSignature();
    pipeline.SetGlobalRootSignature(m_rayGenSignature.Get());

    m_csPSOs.clear();
    uint32_t nextCs = 0, rgSlot = 0;

    std::vector<std::wstring> rayGenNames;

    for (auto& p : m_passes.Passes()) {
        if (p.stage == Stage::Barrier || p.stage == Stage::LoopStart || p.stage == Stage::LoopEnd ||
            p.stage == Stage::DLSS)
            continue;

        if (p.stage == Stage::Compute || p.stage == Stage::FixedCompute) {
            ComPtr<IDxcBlob> cs = nv_helpers_dx12::CompileCS(p.file.c_str(), L"main");
            D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
            desc.pRootSignature = m_computeSignature.Get();
            desc.CS = {cs->GetBufferPointer(), cs->GetBufferSize()};
            ComPtr<ID3D12PipelineState> pso;
            LOG(L"[Pipeline] Creating compute PSO: " << p.file);
            ThrowIfFailed(m_ctx.Device()->CreateComputePipelineState(&desc, IID_PPV_ARGS(&pso)));
            LOG(L"[Pipeline] Compute PSO ready: " << p.file);
            m_csPSOs.push_back(pso);
            p.psoIdx = nextCs++;
            continue;
        }

        if (m_passes.PassIndexByFile(p.file) != UINT32_MAX)
            continue;

        std::wstring base = p.file.substr(p.file.find_last_of(L"/\\") + 1);
        base = base.substr(0, base.rfind(L'.'));
        rayGenNames.push_back(base);
        ComPtr<IDxcBlob> lib = nv_helpers_dx12::CompileShaderLibrary(p.file.c_str());
        pipeline.AddLibrary(lib.Get(), {base.c_str()});
        m_passes.RegisterPassIndex(p.file, rgSlot);
        rgSlot++;

    }

    // The raygens shade their hits themselves from the hit object and never invoke it, so the
    // closest-hit and miss shaders are empty. They still exist: the driver fails
    // CreateStateObject (DXGI_ERROR_DRIVER_INTERNAL_ERROR) for a raytracing pipeline in which no
    // hit group has a closest-hit shader, whether the groups are empty or any-hit only, and the
    // SDK layers crash while reporting that failure. The alpha-test any-hit is the only shader a
    // traversal ever runs.
    ComPtr<IDxcBlob> missLib = nv_helpers_dx12::CompileShaderLibrary(L"Miss_v8.hlsl");
    ComPtr<IDxcBlob> hitLib = nv_helpers_dx12::CompileShaderLibrary(L"Hit_v8.hlsl");
    ComPtr<IDxcBlob> anyHitLib = nv_helpers_dx12::CompileShaderLibrary(L"AnyHit.hlsl");

    pipeline.AddLibrary(missLib.Get(), {L"Miss"});
    pipeline.AddLibrary(hitLib.Get(), {L"ClosestHit"});
    pipeline.AddLibrary(anyHitLib.Get(), {L"AlphaTestAnyHit"});

    pipeline.AddHitGroup(L"OpaqueHitGroup", L"ClosestHit");
    pipeline.AddHitGroup(L"AlphaHitGroup", L"ClosestHit", L"AlphaTestAnyHit");
    pipeline.AddHitGroup(L"TerrainHitGroup", L"ClosestHit");

    pipeline.AddRootSignatureAssociation(m_missSignature.Get(), {L"Miss"});
    pipeline.AddRootSignatureAssociation(m_hitSignature.Get(),
                                         {L"OpaqueHitGroup", L"AlphaHitGroup", L"TerrainHitGroup"});

    pipeline.SetMaxPayloadSize(4); // TracePayload in Inline_RT_v8.hlsli carries nothing

    pipeline.SetMaxAttributeSize(2 * sizeof(float));
    pipeline.SetMaxRecursionDepth(1);
    pipeline.SetPipelineFlags(D3D12_RAYTRACING_PIPELINE_FLAG_ALLOW_OPACITY_MICROMAPS);

    LOG(L"[Pipeline] Creating ray-tracing state object (" << rayGenNames.size() << L" raygen exports)");
    m_rtStateObject = pipeline.Generate();
    LOG(L"[Pipeline] Ray-tracing state object ready");
    ThrowIfFailed(m_rtStateObject->QueryInterface(IID_PPV_ARGS(&m_rtStateObjectProps)));

    // With a trace recursion depth of one the driver's default pipeline stack is exactly the
    // raygen plus the deepest shader a traversal invokes, so it is left in place: an explicit
    // size that comes out too small corrupts memory long before the device fails. The queried
    // sizes are logged for reference only.
    auto stackOf = [&](const wchar_t* exportName) -> UINT64 {
        const UINT64 sz = m_rtStateObjectProps->GetShaderStackSize(exportName);
        return sz >= 0xFFFFFFFFull ? 0ull : sz;
    };
    // A raygen's stack holds the state it keeps live across TraceRay, so it tracks the live state
    // that the traces (and the reorders next to them) have to save and restore.
    UINT64 rgStack = 0;
    for (const auto& name : rayGenNames) {
        const UINT64 sz = stackOf(name.c_str());
        LOG(L"[RT] raygen " << name << L" stack " << sz);
        rgStack = std::max(rgStack, sz);
    }
    LOG(L"[RT] shader stack sizes: raygen " << rgStack << L", any-hit " << stackOf(L"AlphaHitGroup::anyhit")
                                            << L", miss " << stackOf(L"Miss") << L" (driver default pipeline stack)");
}

void Renderer::CreateRaytracingOutputBuffer() {
    auto* dev = m_ctx.Device();

    UINT w = GetWidth(), h = GetHeight(), px = TileAlignedPx(w, h);

    D3D12_RESOURCE_DESC rd = {};
    rd.DepthOrArraySize = 6;
    rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    rd.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    rd.Width = w;
    rd.Height = h;
    rd.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    rd.MipLevels = 1;
    rd.SampleDesc.Count = 1;
    ThrowIfFailed(dev->CreateCommittedResource(&nv_helpers_dx12::kDefaultHeapProps, D3D12_HEAP_FLAG_NONE, &rd,
                                               D3D12_RESOURCE_STATE_COPY_SOURCE, nullptr,
                                               IID_PPV_ARGS(&m_outputResource)));

    ResourceFactory rf(dev);
    m_permanentDataTexture = rf.CreateTexture2D(w, h, DXGI_FORMAT_R32G32B32A32_FLOAT, 1,
                                                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, L"PermanentData");

    D3D12_RESOURCE_DESC sd = {};
    sd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    sd.Width = w;
    sd.Height = h;
    sd.DepthOrArraySize = SCRATCH_LAYER_COUNT;
    sd.MipLevels = 1;
    sd.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    sd.SampleDesc.Count = 1;
    sd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    ThrowIfFailed(dev->CreateCommittedResource(&nv_helpers_dx12::kDefaultHeapProps, D3D12_HEAP_FLAG_NONE, &sd,
                                               D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                                               IID_PPV_ARGS(&m_scratchPing)));

    auto MakeRaw = [&](ComPtr<ID3D12Resource>& res, UINT bytes, const std::wstring& name) {
        res = rf.CreateUAVBuffer(bytes, name);
    };
    MakeRaw(m_liteReservoirs, px * LITE_RESERVOIR_BYTES, L"LiteReservoirs");
    MakeRaw(m_sampleBuffer_current, px * sizeof(SampleData), L"Sample_Current");
    MakeRaw(m_sampleBuffer_last, px * sizeof(SampleData), L"Sample_Last");

    CreatePathStateBuffer();
}

void Renderer::CreatePathStateBuffer() {
    m_gpuProfiler.Init(m_ctx.Device(), m_ctx.CmdQueue());
    ResourceFactory rf(m_ctx.Device());
    // World-space cache history survives changes to the render resolution.
    if (!m_sharcBuffer) {
        m_sharcBuffer = rf.CreateUAVBuffer(SHARC_BUFFER_BYTES, L"SHaRC surface radiance cache");
        m_sharcResetPending = true;
    }

    m_pathStateBuffer =
        rf.CreateUAVBuffer(TileAlignedPx(GetWidth(), GetHeight()) * kPathStateBytesPerPx, L"PathStateBuffer");

    m_skyBakeBuffer = rf.CreateUAVBuffer(SKYBAKE_BYTES, L"SkyBake");

    if (!m_autoExposeBuffer)
        m_autoExposeBuffer = rf.CreateUAVBuffer(128, L"AutoExposeState");
}

// Fixed renderer slots precede the dynamically populated material texture range.
void Renderer::CreateShaderResourceHeap() {
    auto* dev = m_ctx.Device();
    m_srvUavHeap = nv_helpers_dx12::CreateDescriptorHeap(dev, 1000000, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, true);

    D3D12_DESCRIPTOR_HEAP_DESC stagingDesc = {};
    stagingDesc.NumDescriptors = 4;
    stagingDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    ThrowIfFailed(dev->CreateDescriptorHeap(&stagingDesc, IID_PPV_ARGS(&m_stagingUavHeap)));

    {
        D3D12_DESCRIPTOR_HEAP_DESC shd = {};
        shd.NumDescriptors = 2;
        shd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER;
        shd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        ThrowIfFailed(dev->CreateDescriptorHeap(&shd, IID_PPV_ARGS(&m_samplerHeap)));

        D3D12_SAMPLER_DESC sd = {};
        sd.AddressU = sd.AddressV = sd.AddressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
        sd.MipLODBias = 0.0f;
        sd.MaxAnisotropy = 16;
        sd.ComparisonFunc = D3D12_COMPARISON_FUNC_LESS_EQUAL;
        sd.BorderColor[0] = sd.BorderColor[1] = sd.BorderColor[2] = sd.BorderColor[3] = 1.0f;
        sd.MinLOD = 0.0f;
        sd.MaxLOD = D3D12_FLOAT32_MAX;

        const UINT sinc = dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
        CD3DX12_CPU_DESCRIPTOR_HANDLE sh(m_samplerHeap->GetCPUDescriptorHandleForHeapStart());
        sd.Filter = D3D12_FILTER_ANISOTROPIC;
        dev->CreateSampler(&sd, sh);
        sh.Offset(1, sinc);
        sd.Filter = D3D12_FILTER_MIN_MAG_POINT_MIP_LINEAR;
        dev->CreateSampler(&sd, sh);
    }

    CD3DX12_CPU_DESCRIPTOR_HANDLE handle(m_srvUavHeap->GetCPUDescriptorHandleForHeapStart());
    CD3DX12_GPU_DESCRIPTOR_HANDLE gpuHandle(m_srvUavHeap->GetGPUDescriptorHandleForHeapStart());
    CD3DX12_CPU_DESCRIPTOR_HANDLE stagingHandle(m_stagingUavHeap->GetCPUDescriptorHandleForHeapStart());
    const UINT inc = dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    auto next = [&]() {
        handle.Offset(1, inc);
        gpuHandle.Offset(1, inc);
    };
    auto nextStg = [&]() { stagingHandle.Offset(1, inc); };

    auto nullSRV = [&](D3D12_SRV_DIMENSION dim = D3D12_SRV_DIMENSION_BUFFER) {
        D3D12_SHADER_RESOURCE_VIEW_DESC sd = {};
        sd.Format = DXGI_FORMAT_R32_UINT;
        sd.ViewDimension = dim;
        sd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        if (dim == D3D12_SRV_DIMENSION_BUFFER)
            sd.Buffer.NumElements = 1;
        dev->CreateShaderResourceView(nullptr, &sd, handle);
        next();
    };

    {
        D3D12_UNORDERED_ACCESS_VIEW_DESC ud = {};
        ud.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
        ud.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        ud.Texture2DArray.ArraySize = m_outputResource->GetDesc().DepthOrArraySize;
        dev->CreateUnorderedAccessView(m_outputResource.Get(), nullptr, &ud, handle);
        next();
    }

    {
        D3D12_UNORDERED_ACCESS_VIEW_DESC ud = {};
        ud.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        ud.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
        dev->CreateUnorderedAccessView(m_permanentDataTexture.Get(), nullptr, &ud, handle);
        next();
    }

    {
        D3D12_SHADER_RESOURCE_VIEW_DESC sd = {};
        sd.Format = DXGI_FORMAT_UNKNOWN;
        sd.ViewDimension = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE;
        sd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        sd.RaytracingAccelerationStructure.Location = m_planet.tlas_result()->GetGPUVirtualAddress();
        dev->CreateShaderResourceView(nullptr, &sd, handle);
        next();
    }

    {
        D3D12_SHADER_RESOURCE_VIEW_DESC sd = {};
        sd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        sd.Format = DXGI_FORMAT_R32_UINT;
        sd.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;

        sd.Buffer.NumElements = std::max(m_scene.combinedIndexCount(), 1u);
        dev->CreateShaderResourceView(m_scene.indexGlobal.Get(), &sd, handle);
        next();
    }

    {
        D3D12_SHADER_RESOURCE_VIEW_DESC sd = {};
        sd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        sd.Format = DXGI_FORMAT_UNKNOWN;
        sd.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        sd.Buffer.NumElements = std::max(m_scene.combinedVertexCount(), 1u);
        sd.Buffer.StructureByteStride = sizeof(BTriVertex);
        dev->CreateShaderResourceView(m_scene.vertexGlobal.Get(), &sd, handle);
        next();
    }

    {
        D3D12_CONSTANT_BUFFER_VIEW_DESC cd = {};
        cd.BufferLocation = m_camera.GPUBuffer()->GetGPUVirtualAddress();
        cd.SizeInBytes = m_camera.BufferSize();
        dev->CreateConstantBufferView(&cd, handle);
        next();
    }

    {
        D3D12_SHADER_RESOURCE_VIEW_DESC sd = {};
        sd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        sd.Format = DXGI_FORMAT_UNKNOWN;
        sd.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        sd.Buffer.NumElements = m_scene.instancePropsCount();
        sd.Buffer.StructureByteStride = sizeof(InstanceProperties);
        dev->CreateShaderResourceView(m_scene.instanceProperties.Get(), &sd, handle);
        next();
    }

    {
        D3D12_SHADER_RESOURCE_VIEW_DESC sd = {};
        sd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        sd.Format = DXGI_FORMAT_R32_UINT;
        sd.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        sd.Buffer.NumElements = std::max((UINT)m_scene.materialIDs.size(), 1u);
        dev->CreateShaderResourceView(m_scene.materialIndexBuffer.Get(), &sd, handle);
        next();
    }

    {
        D3D12_SHADER_RESOURCE_VIEW_DESC sd = {};
        sd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        sd.Format = DXGI_FORMAT_UNKNOWN;
        sd.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        sd.Buffer.NumElements = (UINT)m_scene.materials.size();
        sd.Buffer.StructureByteStride = (UINT)(MaterialPack::kMatPackedU32 * sizeof(uint32_t));
        dev->CreateShaderResourceView(m_scene.materialBuffer.Get(), &sd, handle);
        next();
    }

    if (m_scene.emissiveTrianglesBuffer) {
        D3D12_SHADER_RESOURCE_VIEW_DESC sd = {};
        sd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        sd.Format = DXGI_FORMAT_UNKNOWN;
        sd.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        sd.Buffer.NumElements = (UINT)m_scene.emissiveTriangles.size();
        sd.Buffer.StructureByteStride = sizeof(LightTriangle);
        dev->CreateShaderResourceView(m_scene.emissiveTrianglesBuffer.Get(), &sd, handle);
        next();
    } else
        nullSRV();

    auto rawUAV = [&](ComPtr<ID3D12Resource>& res, UINT bytes) {
        D3D12_UNORDERED_ACCESS_VIEW_DESC ud = {};
        ud.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        ud.Format = DXGI_FORMAT_R32_TYPELESS;
        ud.Buffer.NumElements = bytes / 4;
        ud.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
        dev->CreateUnorderedAccessView(res.Get(), nullptr, &ud, handle);
        next();
    };
    UINT px = TileAlignedPx(GetWidth(), GetHeight());

    for (int i = 0; i < 2; ++i) {
        D3D12_UNORDERED_ACCESS_VIEW_DESC ud = {};
        ud.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        ud.Format = DXGI_FORMAT_R32_TYPELESS;
        ud.Buffer.NumElements = 1;
        ud.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
        dev->CreateUnorderedAccessView(nullptr, nullptr, &ud, handle);
        next();
    }
    rawUAV(m_liteReservoirs, px * LITE_RESERVOIR_BYTES);
    {
        D3D12_UNORDERED_ACCESS_VIEW_DESC ud = {};
        ud.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        ud.Format = DXGI_FORMAT_R32_TYPELESS;
        ud.Buffer.NumElements = 1;
        ud.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
        dev->CreateUnorderedAccessView(nullptr, nullptr, &ud, handle);
        next();
    }
    rawUAV(m_sampleBuffer_current, px * sizeof(SampleData));
    rawUAV(m_sampleBuffer_last, px * sizeof(SampleData));

    m_lightTree.WriteSlotSrv(dev, handle);
    next();
    nullSRV();

    {
        D3D12_UNORDERED_ACCESS_VIEW_DESC ud = {};
        ud.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
        ud.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
        ud.Texture2DArray.ArraySize = SCRATCH_LAYER_COUNT;
        dev->CreateUnorderedAccessView(m_scratchPing.Get(), nullptr, &ud, handle);
        next();
    }

    {
        D3D12_UNORDERED_ACCESS_VIEW_DESC ud = {};
        ud.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        ud.Format = DXGI_FORMAT_R32_TYPELESS;
        ud.Buffer.NumElements = 1;
        ud.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
        dev->CreateUnorderedAccessView(nullptr, nullptr, &ud, handle);
        next();
    }

    m_lightTree.WriteSrvs(dev, handle);
    handle.Offset(4, inc);
    gpuHandle.Offset(4, inc);

    if (m_scene.triToLightIdBuffer) {
        D3D12_SHADER_RESOURCE_VIEW_DESC sd = {};
        sd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        sd.Format = DXGI_FORMAT_R32_UINT;
        sd.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        sd.Buffer.NumElements = (UINT)m_scene.triToLightId.size();
        dev->CreateShaderResourceView(m_scene.triToLightIdBuffer.Get(), &sd, handle);
        next();
    } else
        nullSRV();

    m_lightTree.WriteLookupSrvs(dev, handle);
    handle.Offset(3, inc);
    gpuHandle.Offset(3, inc);

    auto nullTex2D = [&]() {
        D3D12_SHADER_RESOURCE_VIEW_DESC d = {};
        d.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        d.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        d.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        d.Texture2D.MipLevels = 1;
        dev->CreateShaderResourceView(nullptr, &d, handle);
        next();
    };
    nullTex2D();
    nullTex2D();
    nullTex2D();

    if (m_lutTextureArray) {
        D3D12_SHADER_RESOURCE_VIEW_DESC d = {};
        d.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        d.Format = m_lutTextureArray->GetDesc().Format;
        d.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
        d.Texture2DArray.MipLevels = m_lutTextureArray->GetDesc().MipLevels;
        d.Texture2DArray.ArraySize = m_lutTextureArray->GetDesc().DepthOrArraySize;
        dev->CreateShaderResourceView(m_lutTextureArray.Get(), &d, handle);
        next();
    } else {
        nullSRV(D3D12_SRV_DIMENSION_TEXTURE2DARRAY);
    }

    rawUAV(m_pathStateBuffer, px * kPathStateBytesPerPx);

    auto dlssUAV = [&](ID3D12Resource* res, DXGI_FORMAT fmt) {
        D3D12_UNORDERED_ACCESS_VIEW_DESC ud = {};
        ud.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        ud.Format = fmt;
        dev->CreateUnorderedAccessView(res, nullptr, &ud, handle);
        next();
    };
    dlssUAV(m_dlss.Depth(), DXGI_FORMAT_R32_FLOAT);
    dlssUAV(m_dlss.MVec(), DXGI_FORMAT_R16G16_FLOAT);
    dlssUAV(m_dlss.Normals(), DXGI_FORMAT_R16G16B16A16_FLOAT);
    dlssUAV(m_dlss.DiffuseAlbedo(), DXGI_FORMAT_R16G16B16A16_FLOAT);
    dlssUAV(m_dlss.Output(), DXGI_FORMAT_R16G16B16A16_FLOAT);
    dlssUAV(m_dlss.SpecularAlbedo(), DXGI_FORMAT_R16G16B16A16_FLOAT);
    dlssUAV(m_dlss.Roughness(), DXGI_FORMAT_R16_FLOAT);
    dlssUAV(m_dlss.SpecMVec(), DXGI_FORMAT_R16G16_FLOAT);
    dlssUAV(m_dlss.SpecHitDist(), DXGI_FORMAT_R16_FLOAT);
    dlssUAV(m_dlss.Transparency(), DXGI_FORMAT_R16G16B16A16_FLOAT);
    dlssUAV(m_dlss.ColorBeforeTrans(), DXGI_FORMAT_R16G16B16A16_FLOAT);
    dlssUAV(m_dlss.Input(), DXGI_FORMAT_R16G16B16A16_FLOAT);
    dlssUAV(m_dlss.BiasHint(), DXGI_FORMAT_R8_UNORM);
    dlssUAV(m_dlss.ResponsivityMask(), DXGI_FORMAT_R16_FLOAT);

    nullSRV(D3D12_SRV_DIMENSION_TEXTURE2D);
    nullSRV(D3D12_SRV_DIMENSION_TEXTURE2D);
    nullSRV(D3D12_SRV_DIMENSION_TEXTURE2D);

    // Fixed slots keep later resources independent of gaps in the descriptor table.
    handle =
        CD3DX12_CPU_DESCRIPTOR_HANDLE(m_srvUavHeap->GetCPUDescriptorHandleForHeapStart(), AUTOEXPOSE_HEAP_SLOT, inc);
    gpuHandle =
        CD3DX12_GPU_DESCRIPTOR_HANDLE(m_srvUavHeap->GetGPUDescriptorHandleForHeapStart(), AUTOEXPOSE_HEAP_SLOT, inc);

    {
        D3D12_UNORDERED_ACCESS_VIEW_DESC ud = {};
        ud.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        ud.Format = DXGI_FORMAT_R32_TYPELESS;
        ud.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
        ud.Buffer.NumElements = 32;
        dev->CreateUnorderedAccessView(m_autoExposeBuffer.Get(), nullptr, &ud, handle);
        next();
    }

    if (m_skyStarsTexture) {
        D3D12_SHADER_RESOURCE_VIEW_DESC d = {};
        d.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        d.Format = m_skyStarsTexture->GetDesc().Format;
        d.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        d.Texture2D.MipLevels = m_skyStarsTexture->GetDesc().MipLevels;
        dev->CreateShaderResourceView(m_skyStarsTexture.Get(), &d, handle);
        next();
    } else {
        nullSRV(D3D12_SRV_DIMENSION_TEXTURE2D);
    }

    {
        D3D12_SHADER_RESOURCE_VIEW_DESC d = {};
        d.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        d.Format = DXGI_FORMAT_UNKNOWN;
        d.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        d.Buffer.NumElements = planet::StreamOrchestrator::terrain_table_count();
        d.Buffer.StructureByteStride = sizeof(planet::TerrainSlotGPU);
        dev->CreateShaderResourceView(m_planet.terrain_table(), &d, handle);
        next();
    }

    if (m_terrainHeightmapTexture) {
        D3D12_SHADER_RESOURCE_VIEW_DESC d = {};
        d.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        d.Format = DXGI_FORMAT_R32_FLOAT;
        d.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
        d.Texture2DArray.MostDetailedMip = 0;
        d.Texture2DArray.MipLevels = m_terrainHeightmapTexture->GetDesc().MipLevels;
        d.Texture2DArray.FirstArraySlice = 0;
        d.Texture2DArray.ArraySize = 6;
        d.Texture2DArray.PlaneSlice = 0;
        d.Texture2DArray.ResourceMinLODClamp = 0.0f;
        dev->CreateShaderResourceView(m_terrainHeightmapTexture.Get(), &d, handle);
        next();
    } else {
        D3D12_SHADER_RESOURCE_VIEW_DESC d = {};
        d.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        d.Format = DXGI_FORMAT_R32_FLOAT;
        d.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
        d.Texture2DArray.ArraySize = 6;
        dev->CreateShaderResourceView(nullptr, &d, handle);
        next();
    }

    {
        D3D12_SHADER_RESOURCE_VIEW_DESC d = {};
        d.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        d.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        d.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
        d.Texture2DArray.ArraySize = 6;

        d.Texture2DArray.MipLevels = 1;
        dev->CreateShaderResourceView(m_terrainSurfaceColorTexture.Get(), &d, handle);
        next();
    }

    {
        D3D12_SHADER_RESOURCE_VIEW_DESC d = {};
        d.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        d.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        d.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
        d.Texture2DArray.ArraySize = 6;
        d.Texture2DArray.MipLevels = 1;
        dev->CreateShaderResourceView(m_terrainNormalTexture.Get(), &d, handle);
        next();
    }

    {
        D3D12_SHADER_RESOURCE_VIEW_DESC d = {};
        d.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        d.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
        d.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        d.Texture2D.MipLevels = 1;
        dev->CreateShaderResourceView(m_skyTransmittanceLUT.Get(), &d, handle);
        next();
        dev->CreateShaderResourceView(m_skyMultiScatterLUT.Get(), &d, handle);
        next();
    }

    {
        D3D12_SHADER_RESOURCE_VIEW_DESC d{};
        d.Format = DXGI_FORMAT_R32_FLOAT;
        d.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        d.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        d.Texture2D.MipLevels = 1;
        dev->CreateShaderResourceView(m_blueNoiseTexture.Get(), &d, handle);
        next();
    }

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
    UINT normalCount = m_scene.bindlessRmaBase - m_scene.bindlessNormalBase;
    UINT rmaCount = m_scene.totalBindlessTextures - albedoCount - normalCount;
    writeBatch(m_scene.bindlessAlbedoBase, albedoCount);
    writeBatch(m_scene.bindlessNormalBase, normalCount);
    writeBatch(m_scene.bindlessRmaBase, rmaCount);

    // The ocean owns a fixed block of slots below the bindless range and reaches all of them
    // through direct heap indexing, so it needs no descriptor table of its own.
    static_assert(OCEAN_HEAP_BASE + OCEAN_HEAP_COUNT <= BINDLESS_HEAP_START,
                  "Ocean descriptors overlap the bindless texture range");
    m_ocean.CreateDescriptors(dev, m_srvUavHeap.Get());
}

// Match pass indices and instance hit-group offsets used during dispatch.
void Renderer::CreateShaderBindingTable() {
    m_sbtHelper.Reset();
    D3D12_GPU_DESCRIPTOR_HANDLE heapHandle = m_srvUavHeap->GetGPUDescriptorHandleForHeapStart();
    auto heapPointer = reinterpret_cast<UINT64*>(heapHandle.ptr);

    // Repeated pass tokens share one ray-generation record.
    uint32_t rgEntryCount = 0;
    std::unordered_set<std::wstring> seenRayGenFiles;

    for (const auto& entry : m_passes.Tokens()) {
        if (entry == L"barrier" || entry.rfind(L"loop:", 0) == 0 || entry == L"endloop" || entry == L"dlss")
            continue;
        if (entry.find(L"|cs:") != std::wstring::npos || entry.find(L"|fx:") != std::wstring::npos)
            continue;

        if (!seenRayGenFiles.insert(entry.substr(0, entry.find(L'|'))).second)
            continue;

        std::wstring base = entry.substr(entry.find_last_of(L"/\\") + 1);
        base = base.substr(0, base.rfind(L'.'));
        m_sbtHelper.AddRayGenerationProgram(base.c_str(), {heapPointer});
        rgEntryCount++;
    }

    m_sbtHelper.AddMissProgram(L"Miss", {});

    for (size_t i = 0; i < m_scene.instances.size(); ++i) {
        const auto& mesh = m_scene.meshes[m_scene.instances[i].meshIndex];
        bool hasAlpha = mesh.alphaTriCount > 0;
        bool hasOpaque = mesh.opaqueTriCount > 0;

        if (hasOpaque)
            m_sbtHelper.AddHitGroup(L"OpaqueHitGroup", {});
        else
            m_sbtHelper.AddHitGroup(L"AlphaHitGroup", {});

        m_sbtHelper.AddHitGroup(L"AlphaHitGroup", {});
    }

    // Streamed terrain and voxel records follow the regular scene instances.
    m_sbtHelper.AddHitGroup(L"TerrainHitGroup", {});

    m_sbtHelper.AddHitGroup(L"OpaqueHitGroup", {});
    m_sbtHelper.AddHitGroup(L"AlphaHitGroup", {});

    uint32_t sbtSize = m_sbtHelper.ComputeSBTSize();
    m_sbtStorage = nv_helpers_dx12::CreateBuffer(m_ctx.Device(), sbtSize, D3D12_RESOURCE_FLAG_NONE,
                                                 D3D12_RESOURCE_STATE_GENERIC_READ, nv_helpers_dx12::kUploadHeapProps);
    m_sbtHelper.Generate(m_sbtStorage.Get(), m_rtStateObjectProps.Get());
}

void Renderer::GenerateLutTextures() {
    SCOPE_TIMER("GenerateLutTextures");
    std::vector<std::vector<float>> allData(NUM_LUTS, std::vector<float>(LUT_RESOLUTION * LUT_RESOLUTION * 4));
    const XMFLOAT3 N = {0, 0, 1};
    Material tempMat;
    std::mt19937 gen(0x4c5554u);
    std::uniform_real_distribution<float> dist(0, 1);

    for (int y = 0; y < LUT_RESOLUTION; ++y) {
        float cosTheta = std::max(0.01f, (float)y / (LUT_RESOLUTION - 1));
        float sinTheta = sqrt(1 - cosTheta * cosTheta);
        const XMFLOAT3 V = {sinTheta, 0, cosTheta};

        for (int x = 0; x < LUT_RESOLUTION; ++x) {
            float roughness = std::max(0.01f, (float)x / (LUT_RESOLUTION - 1));
            tempMat.Pr_Pm_Ps_Pc.x = roughness;
            size_t pi = ((size_t)y * LUT_RESOLUTION + x) * 4;

            allData[0][pi] = ComputeSheenDirectionalAlbedo(N, V, roughness, NUM_SAMPLES_LUT);

            float Ess = 0, schlick5 = 0, schlick10 = 0;
            for (int i = 0; i < NUM_SAMPLES_LUT; ++i) {
                XMFLOAT3 L;
                SampleGGX(tempMat, V, N, L, dist(gen), dist(gen));
                if (dot(N, L) <= 0)
                    continue;
                // With the legacy VNDF proposal, f_unit * cos / pdf = G1(L).
                float weight = G1_SmithGGX(dot(N, L), roughness * roughness);
                XMFLOAT3 H = normalize(V + L);
                float xh = std::clamp(1.0f - dot(V, H), 0.0f, 1.0f);
                float xh2 = xh * xh;
                float xh5 = xh2 * xh2 * xh;
                Ess += weight;
                schlick5 += weight * xh5;
                schlick10 += weight * xh5 * xh5;
            }
            allData[1][pi] = NUM_SAMPLES_LUT > 0 ? Ess / NUM_SAMPLES_LUT : 0;
            allData[1][pi + 1] = NUM_SAMPLES_LUT > 0 ? schlick5 / NUM_SAMPLES_LUT : 0;
            allData[1][pi + 2] = NUM_SAMPLES_LUT > 0 ? schlick10 / NUM_SAMPLES_LUT : 0;
        }
    }
    CreateAndUploadLutArray(allData, m_lutTextureArray, L"LutTextureArray");
}

void Renderer::CreateAndUploadLutArray(const std::vector<std::vector<float>>& allData, ComPtr<ID3D12Resource>& tar,
                                       const std::wstring& rn) {
    if (allData.empty())
        return;
    UINT arraySize = (UINT)allData.size();

    D3D12_RESOURCE_DESC td = {};
    td.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    td.Width = LUT_RESOLUTION;
    td.Height = LUT_RESOLUTION;
    td.DepthOrArraySize = arraySize;
    td.MipLevels = 1;
    td.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    td.SampleDesc.Count = 1;
    ThrowIfFailed(m_ctx.Device()->CreateCommittedResource(&nv_helpers_dx12::kDefaultHeapProps, D3D12_HEAP_FLAG_NONE,
                                                          &td, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                          IID_PPV_ARGS(&tar)));
    tar->SetName(rn.c_str());

    UINT64 uploadSize = GetRequiredIntermediateSize(tar.Get(), 0, arraySize);
    m_lutUploadHeaps.emplace_back();
    auto& uh = m_lutUploadHeaps.back();
    auto ud = CD3DX12_RESOURCE_DESC::Buffer(uploadSize);
    ThrowIfFailed(m_ctx.Device()->CreateCommittedResource(&nv_helpers_dx12::kUploadHeapProps, D3D12_HEAP_FLAG_NONE, &ud,
                                                          D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                          IID_PPV_ARGS(&uh)));

    std::vector<D3D12_SUBRESOURCE_DATA> sr(arraySize);
    for (UINT i = 0; i < arraySize; ++i) {
        sr[i].pData = allData[i].data();
        sr[i].RowPitch = LUT_RESOLUTION * 4 * sizeof(float);
        sr[i].SlicePitch = sr[i].RowPitch * LUT_RESOLUTION;
    }
    UpdateSubresources(m_ctx.CmdList(), tar.Get(), uh.Get(), 0, 0, arraySize, sr.data());

    auto b = CD3DX12_RESOURCE_BARRIER::Transition(tar.Get(), D3D12_RESOURCE_STATE_COPY_DEST, kSRV);
    m_ctx.CmdList()->ResourceBarrier(1, &b);
}

static constexpr const char* SKY_STARS_EXR_PATH = "./sky_stars.exr";

void Renderer::InitSkyStarsTexture() {
    SCOPE_TIMER("InitSkyStarsTexture");

    if (!std::filesystem::exists(SKY_STARS_EXR_PATH)) {
        LOG(L"[SkyStars] EXR not found at " << SKY_STARS_EXR_PATH << L" -- sky will use night base only");
        return;
    }

    float* rgbaF = nullptr;
    int width = 0, height = 0;
    const char* errStr = nullptr;
    int ret = LoadEXR(&rgbaF, &width, &height, SKY_STARS_EXR_PATH, &errStr);
    if (ret != TINYEXR_SUCCESS) {
        LOG(L"[SkyStars] LoadEXR failed: " << (errStr ? std::string(errStr).c_str() : "(no error string)"));
        if (errStr)
            FreeEXRErrorMessage(errStr);
        return;
    }

    const size_t numFloats = (size_t)width * height * 4;
    std::vector<uint16_t> halfBase(numFloats);
    DirectX::PackedVector::XMConvertFloatToHalfStream(halfBase.data(), sizeof(uint16_t), rgbaF, sizeof(float),
                                                      numFloats);
    free(rgbaF);

    DirectX::Image baseImg = {};
    baseImg.width = (size_t)width;
    baseImg.height = (size_t)height;
    baseImg.format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    baseImg.rowPitch = (size_t)width * 8;
    baseImg.slicePitch = baseImg.rowPitch * (size_t)height;
    baseImg.pixels = reinterpret_cast<uint8_t*>(halfBase.data());

    DirectX::ScratchImage mipChain;
    HRESULT hr = DirectX::GenerateMipMaps(baseImg, DirectX::TEX_FILTER_LINEAR, 0, mipChain);
    const bool haveMips = SUCCEEDED(hr);
    const size_t mipCount = haveMips ? mipChain.GetImageCount() : 1u;
    if (!haveMips) {
        LOG(L"[SkyStars] GenerateMipMaps failed (hr=" << std::hex << hr << std::dec
                                                      << L"), falling back to single mip");
    }
    LOG(L"[SkyStars] Loaded " << width << L"x" << height << L" EXR with " << mipCount << L" mip"
                              << (mipCount == 1 ? L"" : L"s"));

    auto* dev = m_ctx.Device();
    D3D12_RESOURCE_DESC td = {};
    td.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    td.Width = (UINT64)width;
    td.Height = (UINT)height;
    td.DepthOrArraySize = 1;
    td.MipLevels = (UINT16)mipCount;
    td.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    td.SampleDesc.Count = 1;
    ThrowIfFailed(dev->CreateCommittedResource(&nv_helpers_dx12::kDefaultHeapProps, D3D12_HEAP_FLAG_NONE, &td,
                                               D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                               IID_PPV_ARGS(&m_skyStarsTexture)));
    m_skyStarsTexture->SetName(L"SkyStarsTexture");

    UINT64 uploadSize = GetRequiredIntermediateSize(m_skyStarsTexture.Get(), 0, (UINT)mipCount);
    auto ub = CD3DX12_RESOURCE_DESC::Buffer(uploadSize);
    ThrowIfFailed(dev->CreateCommittedResource(&nv_helpers_dx12::kUploadHeapProps, D3D12_HEAP_FLAG_NONE, &ub,
                                               D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                               IID_PPV_ARGS(&m_skyStarsUploadHeap)));
    m_skyStarsUploadHeap->SetName(L"SkyStarsUploadHeap");

    std::vector<D3D12_SUBRESOURCE_DATA> sr(mipCount);
    if (haveMips) {
        for (size_t i = 0; i < mipCount; ++i) {
            const DirectX::Image* img = mipChain.GetImage(i, 0, 0);
            sr[i].pData = img->pixels;
            sr[i].RowPitch = (LONG_PTR)img->rowPitch;
            sr[i].SlicePitch = (LONG_PTR)img->slicePitch;
        }
    } else {
        sr[0].pData = halfBase.data();
        sr[0].RowPitch = (LONG_PTR)((UINT64)width * 8);
        sr[0].SlicePitch = sr[0].RowPitch * height;
    }

    UpdateSubresources(m_ctx.CmdList(), m_skyStarsTexture.Get(), m_skyStarsUploadHeap.Get(), 0, 0, (UINT)mipCount,
                       sr.data());

    auto bar = CD3DX12_RESOURCE_BARRIER::Transition(m_skyStarsTexture.Get(), D3D12_RESOURCE_STATE_COPY_DEST, kSRV);
    m_ctx.CmdList()->ResourceBarrier(1, &bar);
}

static std::vector<float> GenerateBlueNoiseMask(uint32_t size, float sigma, uint32_t seed) {
    const uint32_t n = size * size;
    const int radius = (int)std::ceil(sigma * 4.0f);
    const int width = 2 * radius + 1;
    std::vector<float> kernel((size_t)width * width);
    float kernelSum = 0.0f;
    for (int dy = -radius; dy <= radius; ++dy)
        for (int dx = -radius; dx <= radius; ++dx) {
            const float k = std::exp(-(float)(dx * dx + dy * dy) / (2.0f * sigma * sigma));
            kernel[(size_t)(dy + radius) * width + (dx + radius)] = k;
            kernelSum += k;
        }
    auto splat = [&](std::vector<float>& e, uint32_t idx, float sign) {
        const int cx = (int)(idx % size), cy = (int)(idx / size);
        for (int dy = -radius; dy <= radius; ++dy) {
            const uint32_t y = (uint32_t)(cy + dy + (int)size) % size;
            for (int dx = -radius; dx <= radius; ++dx) {
                const uint32_t x = (uint32_t)(cx + dx + (int)size) % size;
                e[(size_t)y * size + x] += sign * kernel[(size_t)(dy + radius) * width + (dx + radius)];
            }
        }
    };

    auto extreme = [&](const std::vector<float>& e, const std::vector<uint8_t>& p, uint8_t state, bool largest) {
        uint32_t best = 0;
        float bestE = largest ? -std::numeric_limits<float>::max() : std::numeric_limits<float>::max();
        for (uint32_t i = 0; i < n; ++i) {
            if (p[i] != state)
                continue;
            if (largest ? e[i] > bestE : e[i] < bestE) {
                bestE = e[i];
                best = i;
            }
        }
        return best;
    };

    std::vector<uint8_t> pattern(n, 0);
    std::vector<float> energy(n, 0.0f);
    uint32_t rng = seed;
    const uint32_t ones = std::max(1u, n / 10);
    for (uint32_t placed = 0; placed < ones;) {
        rng = rng * 1664525u + 1013904223u;
        const uint32_t i = (rng >> 8) % n;
        if (pattern[i])
            continue;
        pattern[i] = 1;
        splat(energy, i, 1.0f);
        ++placed;
    }
    for (uint32_t iter = 0; iter < n; ++iter) {
        const uint32_t c = extreme(energy, pattern, 1, true);
        pattern[c] = 0;
        splat(energy, c, -1.0f);
        const uint32_t v = extreme(energy, pattern, 0, false);
        pattern[v] = 1;
        splat(energy, v, 1.0f);
        if (v == c)
            break;
    }

    std::vector<uint32_t> rank(n, 0);
    {
        std::vector<uint8_t> p = pattern;
        std::vector<float> e = energy;
        for (uint32_t r = ones; r-- > 0;) {
            const uint32_t c = extreme(e, p, 1, true);
            rank[c] = r;
            p[c] = 0;
            splat(e, c, -1.0f);
        }
    }
    std::vector<uint8_t> p = pattern;
    std::vector<float> e = energy;
    for (uint32_t r = ones; r < n / 2; ++r) {
        const uint32_t v = extreme(e, p, 0, false);
        rank[v] = r;
        p[v] = 1;
        splat(e, v, 1.0f);
    }
    for (uint32_t i = 0; i < n; ++i)
        e[i] = kernelSum - e[i];
    for (uint32_t r = n / 2; r < n; ++r) {
        const uint32_t c = extreme(e, p, 0, true);
        rank[c] = r;
        p[c] = 1;
        splat(e, c, -1.0f);
    }
    std::vector<float> mask(n);
    for (uint32_t i = 0; i < n; ++i)
        mask[i] = ((float)rank[i] + 0.5f) / (float)n;
    return mask;
}

void Renderer::InitBlueNoiseTexture() {
    SCOPE_TIMER("InitBlueNoiseTexture");
    constexpr UINT kSize = BLUE_NOISE_MASK_SIZE;
    const std::vector<float> mask = GenerateBlueNoiseMask(kSize, 1.5f, 0x9E3779B9u);

    auto* dev = m_ctx.Device();
    D3D12_RESOURCE_DESC td = {};
    td.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    td.Width = kSize;
    td.Height = kSize;
    td.DepthOrArraySize = 1;
    td.MipLevels = 1;
    td.Format = DXGI_FORMAT_R32_FLOAT;
    td.SampleDesc.Count = 1;
    ThrowIfFailed(dev->CreateCommittedResource(&nv_helpers_dx12::kDefaultHeapProps, D3D12_HEAP_FLAG_NONE, &td,
                                               D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                               IID_PPV_ARGS(&m_blueNoiseTexture)));
    m_blueNoiseTexture->SetName(L"BlueNoiseMask");

    const UINT64 uploadSize = GetRequiredIntermediateSize(m_blueNoiseTexture.Get(), 0, 1);
    auto ub = CD3DX12_RESOURCE_DESC::Buffer(uploadSize);
    ThrowIfFailed(dev->CreateCommittedResource(&nv_helpers_dx12::kUploadHeapProps, D3D12_HEAP_FLAG_NONE, &ub,
                                               D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                               IID_PPV_ARGS(&m_blueNoiseUploadHeap)));
    m_blueNoiseUploadHeap->SetName(L"BlueNoiseUploadHeap");

    D3D12_SUBRESOURCE_DATA sr = {};
    sr.pData = mask.data();
    sr.RowPitch = (LONG_PTR)kSize * sizeof(float);
    sr.SlicePitch = sr.RowPitch * (LONG_PTR)kSize;
    UpdateSubresources(m_ctx.CmdList(), m_blueNoiseTexture.Get(), m_blueNoiseUploadHeap.Get(), 0, 0, 1, &sr);

    auto bar = CD3DX12_RESOURCE_BARRIER::Transition(m_blueNoiseTexture.Get(), D3D12_RESOURCE_STATE_COPY_DEST, kSRV);
    m_ctx.CmdList()->ResourceBarrier(1, &bar);
}

void Renderer::InitSkyLUTBake() {
    m_skyLutsReady = false;
    SCOPE_TIMER("InitSkyLUTBake");
    auto* dev = m_ctx.Device();

    constexpr UINT kTransW = 256, kTransH = 64;
    constexpr UINT kMsDim = 32;

    auto makeLut = [&](UINT w, UINT h, ComPtr<ID3D12Resource>& out, LPCWSTR name) {
        D3D12_RESOURCE_DESC td = {};
        td.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        td.Width = w;
        td.Height = h;
        td.DepthOrArraySize = 1;
        td.MipLevels = 1;
        td.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
        td.SampleDesc.Count = 1;
        td.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
        td.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        ThrowIfFailed(dev->CreateCommittedResource(&nv_helpers_dx12::kDefaultHeapProps, D3D12_HEAP_FLAG_NONE, &td,
                                                   D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&out)));
        out->SetName(name);
    };
    makeLut(kTransW, kTransH, m_skyTransmittanceLUT, L"SkyTransmittanceLUT");
    makeLut(kMsDim, kMsDim, m_skyMultiScatterLUT, L"SkyMultiScatterLUT");

    {
        D3D12_DESCRIPTOR_HEAP_DESC hd = {};
        hd.NumDescriptors = 2;
        hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        ThrowIfFailed(dev->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&m_skyLutBakeHeap)));

        const UINT inc = dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        CD3DX12_CPU_DESCRIPTOR_HANDLE h(m_skyLutBakeHeap->GetCPUDescriptorHandleForHeapStart());

        D3D12_UNORDERED_ACCESS_VIEW_DESC ud = {};
        ud.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        ud.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
        dev->CreateUnorderedAccessView(m_skyTransmittanceLUT.Get(), nullptr, &ud, h);
        h.Offset(1, inc);
        dev->CreateUnorderedAccessView(m_skyMultiScatterLUT.Get(), nullptr, &ud, h);
    }

    {
        CD3DX12_DESCRIPTOR_RANGE1 ranges[2];
        ranges[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 25, 0, D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE,
                       D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
        ranges[1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 27, 0, D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE,
                       D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
        CD3DX12_ROOT_PARAMETER1 params[2];
        params[0].InitAsDescriptorTable(2, ranges);
        params[1].InitAsConstantBufferView(0);

        CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC desc;
        desc.Init_1_1(2, params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE);

        ComPtr<ID3DBlob> sig, err;
        HRESULT hr = D3D12SerializeVersionedRootSignature(&desc, &sig, &err);
        if (FAILED(hr)) {
            if (err)
                OutputDebugStringA((char*)err->GetBufferPointer());
            ThrowIfFailed(hr);
        }
        ThrowIfFailed(
            dev->CreateRootSignature(0, sig->GetBufferPointer(), sig->GetBufferSize(), IID_PPV_ARGS(&m_skyLutBakeSig)));
    }

    {
        ComPtr<IDxcBlob> csT = nv_helpers_dx12::CompileCS(L"Pass_skylut_bake_v8.hlsl", L"mainTransmittance");
        D3D12_COMPUTE_PIPELINE_STATE_DESC pd = {};
        pd.pRootSignature = m_skyLutBakeSig.Get();
        pd.CS = {csT->GetBufferPointer(), csT->GetBufferSize()};
        ThrowIfFailed(dev->CreateComputePipelineState(&pd, IID_PPV_ARGS(&m_skyLutTransmittancePSO)));

        ComPtr<IDxcBlob> csM = nv_helpers_dx12::CompileCS(L"Pass_skylut_bake_v8.hlsl", L"mainMultiScatter");
        pd.CS = {csM->GetBufferPointer(), csM->GetBufferSize()};
        ThrowIfFailed(dev->CreateComputePipelineState(&pd, IID_PPV_ARGS(&m_skyLutMultiScatterPSO)));
    }

    D3D12_RESOURCE_BARRIER bars[2] = {
        CD3DX12_RESOURCE_BARRIER::Transition(m_skyTransmittanceLUT.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, kSRV),
        CD3DX12_RESOURCE_BARRIER::Transition(m_skyMultiScatterLUT.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, kSRV),
    };
    m_ctx.CmdList()->ResourceBarrier(2, bars);

    LOG(L"[SkyLUT] Created atmospheric transmittance and multiple-scattering LUTs");
}

// Rebuild atmospheric lookup tables only when turbidity changes.
void Renderer::RecordSkyLUTBake(ID3D12GraphicsCommandList4* cmd) {
    if (!m_skyLutTransmittancePSO)
        return;

    if (m_skyLutsReady && m_skyLutTurbidity == m_camera.sunSettings.turbidity)
        return;

    D3D12_RESOURCE_BARRIER toUav[2] = {
        CD3DX12_RESOURCE_BARRIER::Transition(m_skyTransmittanceLUT.Get(), kSRV, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
        CD3DX12_RESOURCE_BARRIER::Transition(m_skyMultiScatterLUT.Get(), kSRV, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
    };
    cmd->ResourceBarrier(2, toUav);

    ID3D12DescriptorHeap* heaps[] = {m_skyLutBakeHeap.Get()};
    cmd->SetDescriptorHeaps(1, heaps);
    cmd->SetComputeRootSignature(m_skyLutBakeSig.Get());
    cmd->SetComputeRootDescriptorTable(0, m_skyLutBakeHeap->GetGPUDescriptorHandleForHeapStart());
    cmd->SetComputeRootConstantBufferView(1, m_camera.GPUBuffer()->GetGPUVirtualAddress());

    const UINT transmittanceTimer = m_gpuProfiler.BeginPass(cmd, "Sky transmittance LUT");
    cmd->SetPipelineState(m_skyLutTransmittancePSO.Get());
    cmd->Dispatch(256 / 8, 64 / 8, 1);
    m_gpuProfiler.EndPass(cmd, transmittanceTimer);
    {
        // Multiple scattering reads the transmittance written by the first pass.
        D3D12_RESOURCE_BARRIER transDone = CD3DX12_RESOURCE_BARRIER::UAV(m_skyTransmittanceLUT.Get());
        cmd->ResourceBarrier(1, &transDone);
    }
    const UINT scatterTimer = m_gpuProfiler.BeginPass(cmd, "Sky multiscatter LUT");
    cmd->SetPipelineState(m_skyLutMultiScatterPSO.Get());

    cmd->Dispatch(32, 32, 1);
    m_gpuProfiler.EndPass(cmd, scatterTimer);
    D3D12_RESOURCE_BARRIER toSrv[2] = {
        CD3DX12_RESOURCE_BARRIER::Transition(m_skyTransmittanceLUT.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, kSRV),
        CD3DX12_RESOURCE_BARRIER::Transition(m_skyMultiScatterLUT.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, kSRV),
    };
    cmd->ResourceBarrier(2, toSrv);
    m_skyLutTurbidity = m_camera.sunSettings.turbidity;
    m_skyLutsReady = true;
}

static constexpr UINT TERRAIN_HEIGHTMAP_GPU_RESOLUTION = 8192;

void Renderer::InitTerrainHeightmapTexture() {
    SCOPE_TIMER("InitTerrainHeightmapTexture");
    const auto t_start = std::chrono::high_resolution_clock::now();

    const planet::HeightmapCubemap& src = m_planet.heightmap();
    if (!src.loaded()) {
        LOG(L"[TerrainHeightmap] planet heightmap not loaded - binding null SRV "
            L"(shader treats as flat sphere)");
        return;
    }

    const UINT dstN = TERRAIN_HEIGHTMAP_GPU_RESOLUTION;
    if (src.resolution() % dstN != 0) {
        LOG(L"[TerrainHeightmap] CPU resolution " << src.resolution()
                                                  << L" is not an integer multiple of GPU resolution " << dstN
                                                  << L" - skipping upload; sin-bump fallback active");
        return;
    }

    auto* dev = m_ctx.Device();
    D3D12_RESOURCE_DESC td = {};
    td.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    td.Width = dstN;
    td.Height = dstN;
    td.DepthOrArraySize = 6;
    td.MipLevels = 1;
    td.Format = DXGI_FORMAT_R32_FLOAT;
    td.SampleDesc.Count = 1;
    ThrowIfFailed(dev->CreateCommittedResource(&nv_helpers_dx12::kDefaultHeapProps, D3D12_HEAP_FLAG_NONE, &td,
                                               D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                               IID_PPV_ARGS(&m_terrainHeightmapTexture)));
    m_terrainHeightmapTexture->SetName(L"TerrainHeightmapCubemap");

    const size_t face_floats = (size_t)dstN * (size_t)dstN;
    std::vector<float> face_buf(face_floats);

    const UINT64 uploadSizeOne = GetRequiredIntermediateSize(m_terrainHeightmapTexture.Get(), 0, 1);
    auto ub = CD3DX12_RESOURCE_DESC::Buffer(uploadSizeOne);
    ThrowIfFailed(dev->CreateCommittedResource(&nv_helpers_dx12::kUploadHeapProps, D3D12_HEAP_FLAG_NONE, &ub,
                                               D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                               IID_PPV_ARGS(&m_terrainHeightmapUploadHeap)));
    m_terrainHeightmapUploadHeap->SetName(L"TerrainHeightmapUploadHeap");

    for (UINT f = 0; f < 6; ++f) {
        if (!src.downsample_face_km(static_cast<uint8_t>(f), dstN, face_buf.data())) {
            LOG(L"[TerrainHeightmap] downsample failed for face " << f << L" - aborting upload");
            return;
        }

        D3D12_SUBRESOURCE_DATA sr = {};
        sr.pData = face_buf.data();
        sr.RowPitch = (LONG_PTR)dstN * sizeof(float);
        sr.SlicePitch = sr.RowPitch * (LONG_PTR)dstN;

        UpdateSubresources(m_ctx.CmdList(), m_terrainHeightmapTexture.Get(), m_terrainHeightmapUploadHeap.Get(), 0, f,
                           1, &sr);

        m_ctx.FlushAndReset();
    }

    auto bar =
        CD3DX12_RESOURCE_BARRIER::Transition(m_terrainHeightmapTexture.Get(), D3D12_RESOURCE_STATE_COPY_DEST, kSRV);
    m_ctx.CmdList()->ResourceBarrier(1, &bar);

    m_terrainHeightmapUploadHeap.Reset();

    const auto t_end = std::chrono::high_resolution_clock::now();
    const auto ms = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count() / 1000.0;
    LOG(L"[TerrainHeightmap] Uploaded 6 x "
        << dstN << L"x" << dstN << L" R32F faces (" << (6 * face_floats * sizeof(float) / (1024 * 1024)) << L" MB) in "
        << ms << L" ms (downsampled from " << src.resolution() << L"); upload heap released");
}

namespace {
ComPtr<ID3D12Resource> create_terrain_array_texture(ID3D12Device* dev, UINT n, DXGI_FORMAT fmt, const wchar_t* name,
                                                    UINT mipLevels = 1) {
    D3D12_RESOURCE_DESC td = {};
    td.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    td.Width = n;
    td.Height = n;
    td.DepthOrArraySize = 6;
    td.MipLevels = static_cast<UINT16>(mipLevels);
    td.Format = fmt;
    td.SampleDesc.Count = 1;
    ComPtr<ID3D12Resource> tex;
    ThrowIfFailed(dev->CreateCommittedResource(&nv_helpers_dx12::kDefaultHeapProps, D3D12_HEAP_FLAG_NONE, &td,
                                               D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&tex)));
    tex->SetName(name);
    return tex;
}
} // namespace

void Renderer::InitTerrainSurfaceColorTexture() {
    SCOPE_TIMER("InitTerrainSurfaceColorTexture");
    const auto t_start = std::chrono::high_resolution_clock::now();

    const planet::HeightmapCubemap& src = m_planet.heightmap();
    if (!src.surface_color_loaded()) {
        LOG(L"[TerrainSurfaceColor] not present in bake - leaving null SRV "
            L"(shader falls back to TERRAIN_ALBEDO constant)");
        return;
    }

    const UINT dstN = TERRAIN_HEIGHTMAP_GPU_RESOLUTION;
    if (src.surface_color_resolution() % dstN != 0) {
        LOG(L"[TerrainSurfaceColor] CPU resolution " << src.surface_color_resolution()
                                                     << L" is not an integer multiple of GPU resolution " << dstN
                                                     << L" - skipping upload");
        return;
    }

    auto* dev = m_ctx.Device();
    m_terrainSurfaceColorTexture =
        create_terrain_array_texture(dev, dstN, DXGI_FORMAT_R8G8B8A8_UNORM, L"TerrainSurfaceColorCubemap");

    const size_t face_bytes = static_cast<size_t>(dstN) * dstN * 4;
    std::vector<std::uint8_t> face_buf(face_bytes);

    const UINT64 uploadSizeOne = GetRequiredIntermediateSize(m_terrainSurfaceColorTexture.Get(), 0, 1);
    auto ub = CD3DX12_RESOURCE_DESC::Buffer(uploadSizeOne);
    ThrowIfFailed(dev->CreateCommittedResource(&nv_helpers_dx12::kUploadHeapProps, D3D12_HEAP_FLAG_NONE, &ub,
                                               D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                               IID_PPV_ARGS(&m_terrainCompanionUploadHeap)));
    m_terrainCompanionUploadHeap->SetName(L"TerrainCompanionUploadHeap");

    for (UINT f = 0; f < 6; ++f) {
        if (!src.surface_color_face(static_cast<uint8_t>(f), dstN, face_buf.data())) {
            LOG(L"[TerrainSurfaceColor] downsample failed for face " << f);
            m_terrainSurfaceColorTexture.Reset();
            m_terrainCompanionUploadHeap.Reset();
            return;
        }
        D3D12_SUBRESOURCE_DATA sr = {};
        sr.pData = face_buf.data();
        sr.RowPitch = (LONG_PTR)dstN * 4;
        sr.SlicePitch = sr.RowPitch * (LONG_PTR)dstN;
        UpdateSubresources(m_ctx.CmdList(), m_terrainSurfaceColorTexture.Get(), m_terrainCompanionUploadHeap.Get(), 0,
                           f, 1, &sr);
        m_ctx.FlushAndReset();
    }

    auto bar =
        CD3DX12_RESOURCE_BARRIER::Transition(m_terrainSurfaceColorTexture.Get(), D3D12_RESOURCE_STATE_COPY_DEST, kSRV);
    m_ctx.CmdList()->ResourceBarrier(1, &bar);

    m_terrainCompanionUploadHeap.Reset();

    const auto ms =
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - t_start)
            .count() /
        1000.0;
    LOG(L"[TerrainSurfaceColor] Uploaded 6 x " << dstN << L"x" << dstN << L" RGBA8 faces ("
                                               << (6 * face_bytes / (1024 * 1024)) << L" MB) in " << ms
                                               << L" ms (downsampled from " << src.surface_color_resolution() << L")");
}

void Renderer::InitTerrainNormalTexture() {
    SCOPE_TIMER("InitTerrainNormalTexture");
    const auto t_start = std::chrono::high_resolution_clock::now();

    const planet::HeightmapCubemap& src = m_planet.heightmap();
    if (!src.normal_loaded()) {
        LOG(L"[TerrainNormal] not present in bake - leaving null SRV "
            L"(shader falls back to analytic finite-diff normal)");
        return;
    }

    const UINT dstN = TERRAIN_HEIGHTMAP_GPU_RESOLUTION;
    if (src.normal_resolution() % dstN != 0) {
        LOG(L"[TerrainNormal] CPU resolution " << src.normal_resolution()
                                               << L" is not an integer multiple of GPU resolution " << dstN
                                               << L" - skipping upload");
        return;
    }

    UINT numMips = 1;
    {
        UINT n = dstN;
        while (n > 1u) {
            n >>= 1;
            ++numMips;
        }
    }

    auto* dev = m_ctx.Device();
    m_terrainNormalTexture =
        create_terrain_array_texture(dev, dstN, DXGI_FORMAT_R8G8B8A8_UNORM, L"TerrainNormalCubemap", numMips);

    std::vector<size_t> mipOffsets(numMips + 1, 0);
    std::vector<UINT> mipResolutions(numMips);
    for (UINT m = 0; m < numMips; ++m) {
        const UINT mn = std::max<UINT>(1u, dstN >> m);
        mipResolutions[m] = mn;
        mipOffsets[m + 1] = mipOffsets[m] + static_cast<size_t>(mn) * mn * 4;
    }
    const size_t face_bytes_all_mips = mipOffsets[numMips];
    std::vector<std::uint8_t> face_buf(face_bytes_all_mips);

    const UINT64 uploadSizeFace = GetRequiredIntermediateSize(m_terrainNormalTexture.Get(), 0, numMips);
    auto ub = CD3DX12_RESOURCE_DESC::Buffer(uploadSizeFace);
    ThrowIfFailed(dev->CreateCommittedResource(&nv_helpers_dx12::kUploadHeapProps, D3D12_HEAP_FLAG_NONE, &ub,
                                               D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                               IID_PPV_ARGS(&m_terrainCompanionUploadHeap)));
    m_terrainCompanionUploadHeap->SetName(L"TerrainCompanionUploadHeap");

    std::vector<D3D12_SUBRESOURCE_DATA> srs(numMips);

    for (UINT f = 0; f < 6; ++f) {
        if (!src.normal_face(static_cast<uint8_t>(f), dstN, face_buf.data())) {
            LOG(L"[TerrainNormal] downsample failed for face " << f);
            m_terrainNormalTexture.Reset();
            m_terrainCompanionUploadHeap.Reset();
            return;
        }

        for (UINT m = 1; m < numMips; ++m) {
            const UINT srcN = mipResolutions[m - 1];
            const UINT dstM = mipResolutions[m];
            const std::uint8_t* srcMip = face_buf.data() + mipOffsets[m - 1];
            std::uint8_t* dstMip = face_buf.data() + mipOffsets[m];
            for (UINT j = 0; j < dstM; ++j) {
                for (UINT i = 0; i < dstM; ++i) {
                    const UINT si = i * 2u;
                    const UINT sj = j * 2u;
                    for (UINT c = 0; c < 4u; ++c) {
                        const uint32_t s00 = srcMip[((sj)*srcN + si) * 4 + c];
                        const uint32_t s10 = srcMip[((sj)*srcN + si + 1) * 4 + c];
                        const uint32_t s01 = srcMip[((sj + 1) * srcN + si) * 4 + c];
                        const uint32_t s11 = srcMip[((sj + 1) * srcN + si + 1) * 4 + c];
                        const uint32_t sum = s00 + s10 + s01 + s11;
                        dstMip[(j * dstM + i) * 4 + c] = static_cast<std::uint8_t>((sum + 2u) / 4u);
                    }
                }
            }
        }

        for (UINT m = 0; m < numMips; ++m) {
            const UINT mn = mipResolutions[m];
            srs[m].pData = face_buf.data() + mipOffsets[m];
            srs[m].RowPitch = static_cast<LONG_PTR>(mn) * 4;
            srs[m].SlicePitch = srs[m].RowPitch * static_cast<LONG_PTR>(mn);
        }

        UpdateSubresources(m_ctx.CmdList(), m_terrainNormalTexture.Get(), m_terrainCompanionUploadHeap.Get(), 0,
                           f * numMips, numMips, srs.data());
        m_ctx.FlushAndReset();
    }

    auto bar = CD3DX12_RESOURCE_BARRIER::Transition(m_terrainNormalTexture.Get(), D3D12_RESOURCE_STATE_COPY_DEST, kSRV);
    m_ctx.CmdList()->ResourceBarrier(1, &bar);

    m_terrainCompanionUploadHeap.Reset();

    const size_t mip0_bytes = static_cast<size_t>(dstN) * dstN * 4;
    const size_t total_bytes = 6 * face_bytes_all_mips;
    const auto ms =
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - t_start)
            .count() /
        1000.0;
    LOG(L"[TerrainNormal] Uploaded 6 x " << dstN << L"x" << dstN << L" RGBA8 faces, " << numMips << L" mips ("
                                         << (total_bytes / (1024 * 1024)) << L" MB, mip 0 = "
                                         << (6 * mip0_bytes / (1024 * 1024)) << L" MB) in " << ms << L" ms (from "
                                         << src.normal_resolution() << L" bake)");
}

void Renderer::CreateReadbackBuffer() {
    D3D12_RESOURCE_DESC td = m_scratchPing->GetDesc();
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp;
    UINT64 totalBytes = 0;
    m_ctx.Device()->GetCopyableFootprints(&td, 0, 1, 0, &fp, nullptr, nullptr, &totalBytes);

    auto hp = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK);
    auto bd = CD3DX12_RESOURCE_DESC::Buffer(totalBytes);
    ThrowIfFailed(m_ctx.Device()->CreateCommittedResource(
        &hp, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&m_readbackBuffer)));
}

void Renderer::SaveSimulationData(uint32_t stepIndex) {
    namespace fs = std::filesystem;
    if (!fs::exists("output"))
        fs::create_directory("output");

    D3D12_RESOURCE_DESC td = m_scratchPing->GetDesc();
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp;
    m_ctx.Device()->GetCopyableFootprints(&td, 0, 1, 0, &fp, nullptr, nullptr, nullptr);

    UINT width = (UINT)td.Width, height = td.Height, rowPitch = fp.Footprint.RowPitch;
    auto* cmdList = m_ctx.CmdList();

    auto ProcessSlice = [&](UINT slice, auto func) {
        auto b1 = CD3DX12_RESOURCE_BARRIER::Transition(m_scratchPing.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                                       D3D12_RESOURCE_STATE_COPY_SOURCE);
        cmdList->ResourceBarrier(1, &b1);

        UINT sub = D3D12CalcSubresource(0, slice, 0, 1, td.DepthOrArraySize);
        CD3DX12_TEXTURE_COPY_LOCATION dst(m_readbackBuffer.Get(), fp);
        CD3DX12_TEXTURE_COPY_LOCATION src(m_scratchPing.Get(), sub);
        cmdList->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);

        auto b2 = CD3DX12_RESOURCE_BARRIER::Transition(m_scratchPing.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
                                                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        cmdList->ResourceBarrier(1, &b2);

        m_ctx.FlushAndReset();

        uint8_t* p = nullptr;
        CD3DX12_RANGE rr(0, fp.Footprint.RowPitch * height);
        ThrowIfFailed(m_readbackBuffer->Map(0, &rr, (void**)&p));
        func(p);
        CD3DX12_RANGE wr(0, 0);
        m_readbackBuffer->Unmap(0, &wr);
    };

    auto WriteBin = [&](const std::string& suffix, const std::vector<float>& data) {
        std::ofstream f("output/" + std::to_string(stepIndex) + "_" + suffix + ".bin", std::ios::binary);
        if (f)
            f.write((const char*)data.data(), data.size() * sizeof(float));
    };

    ProcessSlice(7, [&](uint8_t* rd) {
        std::vector<float> a(width * height), b(width * height), c(width * height), d(width * height);
        for (UINT y = 0; y < height; ++y) {
            float* row = (float*)(rd + y * rowPitch);
            for (UINT x = 0; x < width; ++x) {
                a[y * width + x] = row[x * 4];
                b[y * width + x] = row[x * 4 + 1];
                c[y * width + x] = row[x * 4 + 2];
                d[y * width + x] = row[x * 4 + 3];
            }
        }
        WriteBin("restir", a);
        WriteBin("gt", b);
        WriteBin("init", c);
        WriteBin("albedo", d);
    });

    ProcessSlice(8, [&](uint8_t* rd) {
        std::vector<float> e(width * height), f(width * height);
        for (UINT y = 0; y < height; ++y) {
            float* row = (float*)(rd + y * rowPitch);
            for (UINT x = 0; x < width; ++x) {
                e[y * width + x] = row[x * 4];
                f[y * width + x] = row[x * 4 + 1];
            }
        }
        WriteBin("roughness", e);
        WriteBin("depth", f);
    });

    ProcessSlice(9, [&](uint8_t* rd) {
        std::vector<float> g(width * height * 3);
        for (UINT y = 0; y < height; ++y) {
            float* row = (float*)(rd + y * rowPitch);
            for (UINT x = 0; x < width; ++x) {
                size_t idx = (y * width + x) * 3;
                g[idx] = row[x * 4];
                g[idx + 1] = row[x * 4 + 1];
                g[idx + 2] = row[x * 4 + 2];
            }
        }
        WriteBin("normal", g);
    });
}
