// ═══════════════════════════════════════════════════════════════════
// Renderer.cpp — Slim orchestrator: init, update, render, destroy.
//                All heavy logic lives in the modules.
// ═══════════════════════════════════════════════════════════════════

#include "stdafx.h"
#include "Renderer.h"
#include "Windowsx.h"

// ─────────────────────────────────────────────────────────────────
Renderer::Renderer(UINT width, UINT height)
    : m_width(width), m_height(height),
      m_aspectRatio(static_cast<float>(width) / static_cast<float>(height))
{
    // Define the rendering pass pipeline (data-driven)
    m_passes.Build({
        L"Pass_raygen_v8.hlsl|rg",          L"barrier",
        L"Pass_temp_di_v8.hlsl|cs:16x8",    L"barrier",
        L"Pass_temp_gi_v8.hlsl|rg",     L"barrier",
        L"Pass_spat_di_v8.hlsl|cs:16x16",   L"barrier",
        L"Pass_spat_gi_v8_1.hlsl|rg", L"barrier",
        L"Pass_shading_v8.hlsl|cs:16x16",   L"barrier",
        L"dlss",                             L"barrier",
        L"Pass_postprocess_v8.hlsl|cs:8x4",  L"barrier",
    });
}

// ═════════════════════════════════════════════════════════════════
// Init
// ═════════════════════════════════════════════════════════════════
void Renderer::InitDevice() {
    try {
        m_ctx.Init(Win32Application::GetHwnd(), GetWidth(), GetHeight());
        m_simulator.PromptUserConfiguration();
        m_recorder.Initialize();
        m_camera.Init(m_ctx.Device(), GetWidth(), GetHeight());

        if (!m_ctx.viewportHandle) {
            sl::Result r = slAllocateResources(m_ctx.CmdList(), sl::kFeatureDLSS_RR, m_ctx.viewportHandle);
            if (r != sl::Result::eOk)
                std::wcout << L"[SL] slAllocateResources failed: " << (int)r << std::endl;
        }
        GenerateLutTextures();

        D3D12_FEATURE_DATA_D3D12_OPTIONS5 opts5 = {};
        ThrowIfFailed(m_ctx.Device()->CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS5, &opts5, sizeof(opts5)));
        if (opts5.RaytracingTier < D3D12_RAYTRACING_TIER_1_0)
            throw std::runtime_error("Raytracing not supported on device");
    } catch (const std::exception& e) {
        wchar_t wMsg[4096];
        MultiByteToWideChar(CP_UTF8, 0, e.what(), -1, wMsg, 4096);
        MessageBoxW(NULL, wMsg, L"Fatal Init Error", MB_OK | MB_ICONERROR);
        exit(1);
    }
}

void Renderer::LoadScene(const std::vector<ModelEntry>& models) {
    try {
        auto flushFn = [this]() { m_ctx.FlushAndReset(); };
        AssetLoader::LoadModels(models, m_scene, m_ctx.Device(), m_ctx.CmdList(), flushFn);
        m_ctx.FlushAndReset();
    } catch (const std::exception& e) {
        wchar_t wMsg[4096];
        MultiByteToWideChar(CP_UTF8, 0, e.what(), -1, wMsg, 4096);
        MessageBoxW(NULL, wMsg, L"Fatal Init Error", MB_OK | MB_ICONERROR);
        exit(1);
    }
}

void Renderer::InitSceneGPU() {
    try {
        CreateAccelerationStructures();
        m_scene.BuildGlobalMeshBuffers(m_ctx.Device(), m_ctx.CmdList());
        m_ctx.FlushAndReset();
        m_lutUploadHeaps.clear();

        CreateRaytracingPipeline();
        CreateStreamingCompactionBuffers();
        CreateIndirectCommandSignature();
        CompileSetupIndirectShader();
        CreateRaytracingOutputBuffer();
        CreateReadbackBuffer();
        m_scene.CreateInstancePropertiesBuffer(m_ctx.Device());

        m_scene.UploadMaterials(m_ctx.Device());
        m_dlss.CreateResources(m_ctx.Device(), GetWidth(), GetHeight());
        CreateShaderResourceHeap();
        CreateShaderBindingTable();

        m_scene.tlasDirty       = false;
        m_scene.tlasFullRebuild = false;
        m_scene.lightTreeDirty  = false;
        m_scene.materialsDirty  = false;

        m_blasLocalRoots = lt::ComputeBLASLocalRoots(m_scene.emissiveTriangles);

        {
            const UINT inc = m_ctx.Device()->GetDescriptorHandleIncrementSize(
                D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            CD3DX12_CPU_DESCRIPTOR_HANDLE fontCpu(
                m_srvUavHeap->GetCPUDescriptorHandleForHeapStart(), IMGUI_FONT_HEAP_SLOT, inc);
            CD3DX12_GPU_DESCRIPTOR_HANDLE fontGpu(
                m_srvUavHeap->GetGPUDescriptorHandleForHeapStart(), IMGUI_FONT_HEAP_SLOT, inc);
            m_editor.Init(Win32Application::GetHwnd(), m_ctx.Device(), FRAME_COUNT,
                m_srvUavHeap.Get(), fontCpu, fontGpu);
        }
        // Command list left open — EngineApp closes it.
    } catch (const std::exception& e) {
        wchar_t wMsg[4096];
        MultiByteToWideChar(CP_UTF8, 0, e.what(), -1, wMsg, 4096);
        MessageBoxW(NULL, wMsg, L"Fatal Init Error", MB_OK | MB_ICONERROR);
        exit(1);
    }
}

// ═════════════════════════════════════════════════════════════════
// Update
// ═════════════════════════════════════════════════════════════════
void Renderer::UpdateRenderer(float dt) {
    using hrc = std::chrono::high_resolution_clock;

    // Simulation path
    if (m_simulator.IsActive()) {
        bool shouldCapture = false;
        bool finished = m_simulator.Update(dt, m_camera.Manipulator(), shouldCapture);
        if (shouldCapture) SaveSimulationData(m_simulator.GetLastCaptureIndex());
        if (finished) { LOG(L"\n[Sim] Data Generation Complete."); PostQuitMessage(0); return; }
    }

    auto t_updateStart = hrc::now();
    m_time++;

    // Check for DLSS mode change — flag for reallocation in PopulateCommandList
    if (m_dlss.UpdateMode(m_ctx.Device())) {
        m_dlssModeChanged = true;
        RebuildDLSSDescriptors();
        LOG(L"[DLSS] Mode changed → render res: "
            << m_dlss.RenderWidth() << L"x" << m_dlss.RenderHeight());
    }

    // Kick async light tree TLAS refit when dirty and no refit in flight
    if (m_scene.lightTreeDirty && !m_lightTreeRefit.IsPending()) {
        // If emission values changed, recompute everything from CPU data
        if (m_scene.emissivesDirty) {
            m_scene.CollectEmissiveTriangles();
            m_blasLocalRoots = lt::ComputeBLASLocalRoots(m_scene.emissiveTriangles);
            m_emissiveGpuDirty = true;  // GPU buffers need re-upload
            m_scene.emissivesDirty = false;
            LOG(L"[LightTree] Recomputed emissives + BLAS roots (emission change)");
        }
        KickLightTreeRefit();
        m_scene.lightTreeDirty = false;
    }

    // Poll for completed async refit — stage data for GPU upload
    lt::TLASRefitResult refitResult;
    if (m_lightTreeRefit.PollResult(refitResult)) {
        m_pendingTLASUpload  = std::move(refitResult.nodes);
        m_pendingBLASToItem  = std::move(refitResult.blasToItem);
        LOG(L"[LightTree] Async TLAS refit ready: " << m_pendingTLASUpload.size() << L" nodes");
    }

    // Build editor UI — must run before UpdateInstanceProperties so
    // MarkModelMoved() changes are picked up in the same frame
    m_editor.Draw(m_scene, m_camera, m_passes, m_dlss, m_restirSettings, m_fps, m_frameStats);

    // Prepare instance data on CPU shadow buffer (overlaps with GPU)
    auto t_instStart = hrc::now();
    m_scene.PrepareInstanceProperties();
    auto t_instEnd = hrc::now();

    m_frameStats.cpuInstanceMs = std::chrono::duration<float, std::milli>(t_instEnd - t_instStart).count();
    m_frameStats.cpuUpdateMs   = std::chrono::duration<float, std::milli>(t_instEnd - t_updateStart).count();
    m_frameStats.instanceCount = (UINT)m_scene.instances.size();
    m_frameStats.meshCount     = (UINT)m_scene.meshes.size();
}

// ─────────────────────────────────────────────────────────────────
std::vector<InstanceXformCPU> Renderer::BuildXformsFromScene() const {
    std::vector<InstanceXformCPU> xf;
    xf.reserve(m_scene.instances.size());
    for (const auto& si : m_scene.instances) {
        InstanceXformCPU x{};
        XMStoreFloat4x4(&x.objectToWorld, si.worldTransform);
        xf.push_back(x);
    }
    return xf;
}

void Renderer::KickLightTreeRefit() {
    // Copy data for the background thread (must be self-contained)
    auto xforms = BuildXformsFromScene();
    auto roots  = m_blasLocalRoots;  // copy — thread safety
    m_lightTreeRefit.RequestRefit(std::move(roots), std::move(xforms));
}

void Renderer::UploadLightTreeTLAS(ID3D12GraphicsCommandList* cmdList) {
    if (m_pendingTLASUpload.empty()) return;

    auto* dev = m_ctx.Device();
    const UINT inc = dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    const UINT nodeCount = (UINT)m_pendingTLASUpload.size();
    const UINT nodeBytes = nodeCount * sizeof(lt::LightTLASNodeGpu);
    const UINT itemCount = (UINT)m_pendingBLASToItem.size();
    const UINT itemBytes = itemCount * sizeof(uint32_t);
    CD3DX12_RANGE readRange(0, 0);

    // ── Helper: grow a default-heap buffer + update its SRV ──────
    auto growBuffer = [&](ComPtr<ID3D12Resource>& gpu, ComPtr<ID3D12Resource>& upload,
                         UINT& capacity, UINT needed, UINT stride,
                         DXGI_FORMAT fmt, UINT srvSlot, const wchar_t* name)
    {
        const UINT elemSize = stride ? stride : 4;
        const UINT byteCount = needed * elemSize;

        // Grow GPU buffer if needed
        if (needed > capacity) {
            capacity = needed * 2;  // 2x headroom
            const UINT allocBytes = capacity * elemSize;

            auto hp = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
            auto desc = CD3DX12_RESOURCE_DESC::Buffer(allocBytes);
            ThrowIfFailed(dev->CreateCommittedResource(
                &hp, D3D12_HEAP_FLAG_NONE, &desc,
                D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&gpu)));
            gpu->SetName(name);

            // Update SRV descriptor in the heap
            CD3DX12_CPU_DESCRIPTOR_HANDLE srvHandle(
                m_srvUavHeap->GetCPUDescriptorHandleForHeapStart(), srvSlot, inc);

            D3D12_SHADER_RESOURCE_VIEW_DESC sd{};
            sd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            sd.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
            if (stride > 0) {
                sd.Format = DXGI_FORMAT_UNKNOWN;
                sd.Buffer.NumElements = capacity;
                sd.Buffer.StructureByteStride = stride;
            } else {
                sd.Format = fmt;
                sd.Buffer.NumElements = capacity;
            }
            dev->CreateShaderResourceView(gpu.Get(), &sd, srvHandle);

            LOG(L"[LightTree] Grew " << name << L": " << capacity << L" elements");
        } else {
            // Transition existing buffer to COPY_DEST
            auto b = CD3DX12_RESOURCE_BARRIER::Transition(gpu.Get(),
                D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_STATE_COPY_DEST);
            cmdList->ResourceBarrier(1, &b);
        }

        // Grow upload staging if needed
        if (!upload || byteCount > upload->GetDesc().Width) {
            auto hp = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
            auto desc = CD3DX12_RESOURCE_DESC::Buffer(capacity * elemSize);
            ThrowIfFailed(dev->CreateCommittedResource(
                &hp, D3D12_HEAP_FLAG_NONE, &desc,
                D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&upload)));
        }
    };

    // ── TLAS nodes ───────────────────────────────────────────────
    growBuffer(m_ltTlasGpu, m_tlasUploadStaging, m_ltTlasGpuCapacity,
              nodeCount, sizeof(lt::LightTLASNodeGpu), DXGI_FORMAT_UNKNOWN,
              LT_TLAS_SRV_SLOT, L"LT_TLAS_Refit");

    { void* p = nullptr;
      ThrowIfFailed(m_tlasUploadStaging->Map(0, &readRange, &p));
      memcpy(p, m_pendingTLASUpload.data(), nodeBytes);
      m_tlasUploadStaging->Unmap(0, nullptr); }

    cmdList->CopyBufferRegion(m_ltTlasGpu.Get(), 0, m_tlasUploadStaging.Get(), 0, nodeBytes);

    auto b1 = CD3DX12_RESOURCE_BARRIER::Transition(m_ltTlasGpu.Get(),
        D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_GENERIC_READ);
    cmdList->ResourceBarrier(1, &b1);

    // ── BLASToItem ───────────────────────────────────────────────
    if (itemCount > 0) {
        growBuffer(m_ltBtIGpu, m_blasToItemUploadStaging, m_ltBtIGpuCapacity,
                  itemCount, 0, DXGI_FORMAT_R32_UINT,
                  LT_BLASTOITEM_SRV_SLOT, L"LT_BLASToItem_Refit");

        { void* p = nullptr;
          ThrowIfFailed(m_blasToItemUploadStaging->Map(0, &readRange, &p));
          memcpy(p, m_pendingBLASToItem.data(), itemBytes);
          m_blasToItemUploadStaging->Unmap(0, nullptr); }

        cmdList->CopyBufferRegion(m_ltBtIGpu.Get(), 0, m_blasToItemUploadStaging.Get(), 0, itemBytes);

        auto b2 = CD3DX12_RESOURCE_BARRIER::Transition(m_ltBtIGpu.Get(),
            D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_GENERIC_READ);
        cmdList->ResourceBarrier(1, &b2);
    }

    m_pendingTLASUpload.clear();
    m_pendingBLASToItem.clear();
}

// ─────────────────────────────────────────────────────────────────
void Renderer::UploadEmissiveBuffers(ID3D12GraphicsCommandList* cmdList) {
    if (!m_emissiveGpuDirty) return;
    m_emissiveGpuDirty = false;

    auto* dev = m_ctx.Device();
    const UINT inc = dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    CD3DX12_RANGE readRange(0, 0);

    const auto& tris = m_scene.emissiveTriangles;
    const auto& triMap = m_scene.triToLightId;

    // ── Emissive triangles buffer ────────────────────────────────
    if (!tris.empty()) {
        const UINT count = (UINT)tris.size();
        const UINT bytes = count * sizeof(LightTriangle);

        // Grow GPU buffer if needed
        if (count > m_emissiveGpuCapacity) {
            m_emissiveGpuCapacity = count * 2;
            const UINT allocBytes = m_emissiveGpuCapacity * sizeof(LightTriangle);

            m_ownedEmissiveGpu = nv_helpers_dx12::CreateBuffer(
                dev, allocBytes, D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_COPY_DEST, nv_helpers_dx12::kDefaultHeapProps);
            m_ownedEmissiveGpu->SetName(L"EmissiveTris_Refit");

            // Update SRV at slot 9
            CD3DX12_CPU_DESCRIPTOR_HANDLE srvH(
                m_srvUavHeap->GetCPUDescriptorHandleForHeapStart(), EMISSIVE_TRI_SRV_SLOT, inc);
            D3D12_SHADER_RESOURCE_VIEW_DESC sd{};
            sd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            sd.Format = DXGI_FORMAT_UNKNOWN;
            sd.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
            sd.Buffer.NumElements = m_emissiveGpuCapacity;
            sd.Buffer.StructureByteStride = sizeof(LightTriangle);
            dev->CreateShaderResourceView(m_ownedEmissiveGpu.Get(), &sd, srvH);

            LOG(L"[Emissive] Grew GPU buffer: " << m_emissiveGpuCapacity << L" tris");
        } else if (m_ownedEmissiveGpu) {
            auto b = CD3DX12_RESOURCE_BARRIER::Transition(m_ownedEmissiveGpu.Get(),
                D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_STATE_COPY_DEST);
            cmdList->ResourceBarrier(1, &b);
        } else {
            // First time: transition the original scene buffer
            auto b = CD3DX12_RESOURCE_BARRIER::Transition(m_scene.emissiveTrianglesBuffer.Get(),
                D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_STATE_COPY_DEST);
            cmdList->ResourceBarrier(1, &b);
        }

        // Grow upload staging
        if (!m_emissiveUploadStaging || bytes > m_emissiveUploadStaging->GetDesc().Width) {
            auto hp = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
            auto desc = CD3DX12_RESOURCE_DESC::Buffer(m_emissiveGpuCapacity * sizeof(LightTriangle));
            ThrowIfFailed(dev->CreateCommittedResource(
                &hp, D3D12_HEAP_FLAG_NONE, &desc,
                D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                IID_PPV_ARGS(&m_emissiveUploadStaging)));
        }

        { void* p = nullptr;
          ThrowIfFailed(m_emissiveUploadStaging->Map(0, &readRange, &p));
          memcpy(p, tris.data(), bytes);
          m_emissiveUploadStaging->Unmap(0, nullptr); }

        auto* dst = m_ownedEmissiveGpu ? m_ownedEmissiveGpu.Get() : m_scene.emissiveTrianglesBuffer.Get();
        cmdList->CopyBufferRegion(dst, 0, m_emissiveUploadStaging.Get(), 0, bytes);

        auto b = CD3DX12_RESOURCE_BARRIER::Transition(dst,
            D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_GENERIC_READ);
        cmdList->ResourceBarrier(1, &b);
    }

    // ── TriToLightId buffer ──────────────────────────────────────
    if (!triMap.empty()) {
        const UINT count = (UINT)triMap.size();
        const UINT bytes = count * sizeof(uint32_t);

        if (count > m_triToLightIdGpuCapacity) {
            m_triToLightIdGpuCapacity = count * 2;
            const UINT allocBytes = m_triToLightIdGpuCapacity * sizeof(uint32_t);

            m_ownedTriToLightIdGpu = nv_helpers_dx12::CreateBuffer(
                dev, allocBytes, D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_COPY_DEST, nv_helpers_dx12::kDefaultHeapProps);
            m_ownedTriToLightIdGpu->SetName(L"TriToLightId_Refit");

            // Update SRV at slot 24
            CD3DX12_CPU_DESCRIPTOR_HANDLE srvH(
                m_srvUavHeap->GetCPUDescriptorHandleForHeapStart(), TRI_TO_LIGHTID_SRV_SLOT, inc);
            D3D12_SHADER_RESOURCE_VIEW_DESC sd{};
            sd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            sd.Format = DXGI_FORMAT_R32_UINT;
            sd.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
            sd.Buffer.NumElements = m_triToLightIdGpuCapacity;
            dev->CreateShaderResourceView(m_ownedTriToLightIdGpu.Get(), &sd, srvH);

            LOG(L"[Emissive] Grew TriToLightId: " << m_triToLightIdGpuCapacity);
        } else if (m_ownedTriToLightIdGpu) {
            auto b = CD3DX12_RESOURCE_BARRIER::Transition(m_ownedTriToLightIdGpu.Get(),
                D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_STATE_COPY_DEST);
            cmdList->ResourceBarrier(1, &b);
        } else {
            auto b = CD3DX12_RESOURCE_BARRIER::Transition(m_scene.triToLightIdBuffer.Get(),
                D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_STATE_COPY_DEST);
            cmdList->ResourceBarrier(1, &b);
        }

        if (!m_triToLightIdUploadStaging || bytes > m_triToLightIdUploadStaging->GetDesc().Width) {
            auto hp = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
            auto desc = CD3DX12_RESOURCE_DESC::Buffer(m_triToLightIdGpuCapacity * sizeof(uint32_t));
            ThrowIfFailed(dev->CreateCommittedResource(
                &hp, D3D12_HEAP_FLAG_NONE, &desc,
                D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                IID_PPV_ARGS(&m_triToLightIdUploadStaging)));
        }

        { void* p = nullptr;
          ThrowIfFailed(m_triToLightIdUploadStaging->Map(0, &readRange, &p));
          memcpy(p, triMap.data(), bytes);
          m_triToLightIdUploadStaging->Unmap(0, nullptr); }

        auto* dst = m_ownedTriToLightIdGpu ? m_ownedTriToLightIdGpu.Get() : m_scene.triToLightIdBuffer.Get();
        cmdList->CopyBufferRegion(dst, 0, m_triToLightIdUploadStaging.Get(), 0, bytes);

        auto b = CD3DX12_RESOURCE_BARRIER::Transition(dst,
            D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_GENERIC_READ);
        cmdList->ResourceBarrier(1, &b);
    }

    LOG(L"[Emissive] GPU buffers updated: " << tris.size() << L" tris, " << triMap.size() << L" triToLightId");
}

// ─────────────────────────────────────────────────────────────────
void Renderer::RebuildDLSSDescriptors() {
    auto* dev = m_ctx.Device();
    const UINT inc = dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    CD3DX12_CPU_DESCRIPTOR_HANDLE handle(
        m_srvUavHeap->GetCPUDescriptorHandleForHeapStart(), DLSS_UAV_HEAP_START, inc);

    auto dlssUAV = [&](ID3D12Resource* res, DXGI_FORMAT fmt) {
        D3D12_UNORDERED_ACCESS_VIEW_DESC ud = {};
        ud.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        ud.Format = fmt;
        dev->CreateUnorderedAccessView(res, nullptr, &ud, handle);
        handle.ptr += inc;
    };

    // Must match order in CreateShaderResourceHeap (slots 39-50)
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
}

// ═════════════════════════════════════════════════════════════════
// Render
// ═════════════════════════════════════════════════════════════════
void Renderer::RenderFrame() {
    using hrc = std::chrono::high_resolution_clock;
    static auto s_lastTime = hrc::now();
    static int s_frameCount = 0;
    auto t_frameStart = hrc::now();

    // Wait for previous frame's GPU to finish, then upload shared buffers.
    // PrepareInstanceProperties already ran in UpdateRenderer (overlapped with GPU).
    auto t_waitStart = hrc::now();
    m_ctx.WaitForPreviousFrame();
    m_frameStats.gpuMs = std::chrono::duration<float, std::milli>(hrc::now() - t_waitStart).count();

    // Upload shared GPU buffers (must happen after GPU wait)
    m_camera.UploadGPUBuffer(m_aspectRatio);
    m_scene.UploadInstanceProperties();
    if (m_scene.materialsDirty) {
        m_scene.UpdateMaterialBuffer();
        m_scene.materialsDirty = false;
    }

    m_ctx.BeginFrame();

    auto t_popStart = hrc::now();
    PopulateCommandList();
    auto t_popEnd = hrc::now();
    m_frameStats.cpuPopulateMs = std::chrono::duration<float, std::milli>(t_popEnd - t_popStart).count();

    m_ctx.ExecuteAndPresent();

    // CPU = total CPU work across UpdateRenderer + PopulateCommandList
    m_frameStats.cpuFrameMs = m_frameStats.cpuUpdateMs + m_frameStats.cpuPopulateMs;

    // FPS display
    s_frameCount++;
    auto now = hrc::now();
    float elapsed = std::chrono::duration<float>(now - s_lastTime).count();
    if (elapsed >= 1.0f) {
        m_fps = s_frameCount / elapsed;
        std::wstringstream ss;
        ss << std::fixed << std::setprecision(2)
           << L"Frame Time: " << 1000.0f / m_fps << L" ms (" << m_fps << L" fps)";
        SetWindowTextW(Win32Application::GetHwnd(), ss.str().c_str());
        s_frameCount = 0;
        s_lastTime = now;
    }
}

// ═════════════════════════════════════════════════════════════════
// Destroy
// ═════════════════════════════════════════════════════════════════
void Renderer::DestroyRenderer() {
    m_editor.Shutdown();
    m_ctx.Shutdown();
}

// ═════════════════════════════════════════════════════════════════
UINT Renderer::CreateProceduralMesh(
    const std::vector<Vertex>& vertices, const std::vector<UINT>& indices,
    const Material& material)
{
    UINT matIdx = (UINT)m_scene.materials.size();
    UINT triCount = (UINT)indices.size() / 3;
    m_scene.materials.push_back(material);
    for (UINT t = 0; t < triCount; ++t) m_scene.materialIDs.push_back(matIdx);

    MeshGPU mesh;
    mesh.cpuVertices = vertices; mesh.cpuIndices = indices;
    mesh.cpuMaterialIDs = std::vector<UINT>(triCount, matIdx);
    mesh.vertexCount = (UINT)vertices.size(); mesh.indexCount = (UINT)indices.size();
    mesh.opaqueTriCount = triCount; mesh.alphaTriCount = 0; mesh.materialIDBase = matIdx;

    { UINT bytes = mesh.vertexCount * sizeof(Vertex);
      mesh.vertexBuffer = nv_helpers_dx12::CreateBuffer(m_ctx.Device(), bytes,
          D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ, nv_helpers_dx12::kUploadHeapProps);
      void* p = nullptr; mesh.vertexBuffer->Map(0, nullptr, &p);
      memcpy(p, vertices.data(), bytes); mesh.vertexBuffer->Unmap(0, nullptr); }
    { UINT bytes = mesh.indexCount * sizeof(UINT);
      mesh.indexBuffer = nv_helpers_dx12::CreateBuffer(m_ctx.Device(), bytes,
          D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ, nv_helpers_dx12::kUploadHeapProps);
      void* p = nullptr; mesh.indexBuffer->Map(0, nullptr, &p);
      memcpy(p, indices.data(), bytes); mesh.indexBuffer->Unmap(0, nullptr); }

    auto blasBuf = CreateBottomLevelAS(
        {{ mesh.vertexBuffer, mesh.vertexCount }}, {{ mesh.indexBuffer, mesh.indexCount }},
        mesh.opaqueTriCount, mesh.alphaTriCount);
    mesh.blas = blasBuf.pResult;
    m_ctx.FlushAndReset();

    UINT meshIndex = (UINT)m_scene.meshes.size();
    m_scene.meshes.push_back(std::move(mesh));
    LOG(L"[Engine] Created procedural mesh " << meshIndex << L" (mat " << matIdx << L")");
    return meshIndex;
}

// ═════════════════════════════════════════════════════════════════
UINT Renderer::CreateMeshInstance(UINT sourceMeshIndex, const Material& material)
{
    const auto& src = m_scene.meshes[sourceMeshIndex];
    UINT matIdx   = (UINT)m_scene.materials.size();
    UINT triCount = src.indexCount / 3;

    m_scene.materials.push_back(material);
    for (UINT t = 0; t < triCount; ++t) m_scene.materialIDs.push_back(matIdx);

    MeshGPU mesh;
    mesh.cpuVertices    = src.cpuVertices;
    mesh.cpuIndices     = src.cpuIndices;
    mesh.cpuMaterialIDs = std::vector<UINT>(triCount, matIdx);
    mesh.vertexCount    = src.vertexCount;
    mesh.indexCount     = src.indexCount;
    mesh.opaqueTriCount = src.opaqueTriCount;
    mesh.alphaTriCount  = src.alphaTriCount;
    mesh.materialIDBase = matIdx;

    // Share geometry buffers and BLAS — no GPU work needed
    mesh.vertexBuffer = src.vertexBuffer;
    mesh.indexBuffer  = src.indexBuffer;
    mesh.blas         = src.blas;

    UINT meshIndex = (UINT)m_scene.meshes.size();
    m_scene.meshes.push_back(std::move(mesh));
    return meshIndex;
}

// ═════════════════════════════════════════════════════════════════
void Renderer::HandleSceneStructuralChange() {
    m_scene.RebuildTLASInstanceList();
    m_scene.CreateInstancePropertiesBuffer(m_ctx.Device());

    // Resize dirty tracking and mark all instances dirty (new layout)
    m_scene.instanceDirty.assign(m_scene.instances.size(), 1);
    m_scene.instanceInitialized.assign(m_scene.instances.size(), 0);
    m_scene.cpuInstanceProps.clear();  // force re-init in UpdateInstanceProperties

    // Update SRV slot 6 → new instanceProperties buffer
    { const UINT inc = m_ctx.Device()->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
      CD3DX12_CPU_DESCRIPTOR_HANDLE h(m_srvUavHeap->GetCPUDescriptorHandleForHeapStart(), INSTANCE_PROPS_SRV_SLOT, inc);
      D3D12_SHADER_RESOURCE_VIEW_DESC sd = {};
      sd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
      sd.Format = DXGI_FORMAT_UNKNOWN; sd.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
      sd.Buffer.NumElements = (UINT)m_scene.instances.size();
      sd.Buffer.StructureByteStride = sizeof(InstanceProperties);
      m_ctx.Device()->CreateShaderResourceView(m_scene.instanceProperties.Get(), &sd, h); }

    CreateShaderBindingTable();
    m_scene.CollectEmissiveTriangles();
    m_blasLocalRoots = lt::ComputeBLASLocalRoots(m_scene.emissiveTriangles);
    m_emissiveGpuDirty = true;
    m_scene.tlasDirty        = true;
    m_scene.tlasFullRebuild  = true;
    m_scene.lightTreeDirty   = true;
}

// ═════════════════════════════════════════════════════════════════
bool Renderer::WantsKeyboard() const { return m_editor.IsVisible() && ImGui::GetIO().WantCaptureKeyboard; }
bool Renderer::WantsMouse() const    { return m_editor.IsVisible() && ImGui::GetIO().WantCaptureMouse; }
void Renderer::HandleKeyUp(UINT8 key) {
    if (key == 'C') m_currentDisplayLevel = (m_currentDisplayLevel + 1) % m_displayLevels.size();
    if (key == 'K') m_recorder.CaptureKeyframe(m_camera.Manipulator());
    if (key == VK_F1) m_editor.ToggleVisibility();
}

// ═════════════════════════════════════════════════════════════════
// PopulateCommandList — the frame's GPU work
// ═════════════════════════════════════════════════════════════════
void Renderer::PopulateCommandList() {
    auto* cmdList = m_ctx.CmdList();

    // Reallocate Streamline DLSS feature if mode changed (needs open command list)
    if (m_dlssModeChanged) {
        m_dlssModeChanged = false;
        sl::Result fr = slFreeResources(sl::kFeatureDLSS_RR, m_ctx.viewportHandle);
        if (fr != sl::Result::eOk)
            std::wcout << L"[SL] slFreeResources failed: " << (int)fr << std::endl;
        sl::Result ar = slAllocateResources(cmdList, sl::kFeatureDLSS_RR, m_ctx.viewportHandle);
        if (ar != sl::Result::eOk)
            std::wcout << L"[SL] slAllocateResources failed: " << (int)ar << std::endl;
    }

    // Transition back buffer → render target
    {   auto b = CD3DX12_RESOURCE_BARRIER::Transition(
            m_ctx.BackBuffer(), D3D12_RESOURCE_STATE_PRESENT,
            D3D12_RESOURCE_STATE_RENDER_TARGET);
        cmdList->ResourceBarrier(1, &b); }

    auto rtv = m_ctx.CurrentRTV();
    auto dsv = m_ctx.DSV();
    cmdList->OMSetRenderTargets(1, &rtv, FALSE, &dsv);

    // TLAS update: full rebuild on structural change, periodic in-place rebuild
    // to prevent BVH degradation, or partial refit for transform-only changes.
    static constexpr uint32_t TLAS_REBUILD_INTERVAL = 120;
    bool periodicRebuild = m_scene.tlasDirty && !m_scene.tlasFullRebuild
                        && (m_time % TLAS_REBUILD_INTERVAL) == 0;

    auto t_tlasStart = std::chrono::high_resolution_clock::now();
    m_frameStats.tlasWasRefit = false;
    m_frameStats.tlasWasRebuilt = false;
    if (m_scene.tlasDirty) {
        if (m_scene.tlasFullRebuild) {
            // Structural change: new buffers, new generator
            m_topLevelASGenerator = nv_helpers_dx12::TopLevelASGenerator();
            CreateTopLevelAS(m_scene.tlasInstances, false);
            m_scene.tlasFullRebuild = false;
            m_frameStats.tlasWasRebuilt = true;

            // Full rebuild allocates a new pResult buffer → update SRV
            const UINT inc = m_ctx.Device()->GetDescriptorHandleIncrementSize(
                D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            CD3DX12_CPU_DESCRIPTOR_HANDLE tlasSrv(
                m_srvUavHeap->GetCPUDescriptorHandleForHeapStart(), 2, inc);
            D3D12_SHADER_RESOURCE_VIEW_DESC sd = {};
            sd.Format = DXGI_FORMAT_UNKNOWN;
            sd.ViewDimension = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE;
            sd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            sd.RaytracingAccelerationStructure.Location =
                m_topLevelASBuffers.pResult->GetGPUVirtualAddress();
            m_ctx.Device()->CreateShaderResourceView(nullptr, &sd, tlasSrv);
        } else if (!m_scene.dirtyInstanceList.empty()) {
            if (periodicRebuild) {
                // Periodic rebuild: reuse buffers, full BVH build (PREFER_FAST_BUILD)
                m_topLevelASGenerator.RebuildInPlace(
                    m_ctx.CmdList(),
                    m_topLevelASBuffers.pScratch.Get(),
                    m_topLevelASBuffers.pResult.Get(),
                    m_topLevelASBuffers.pInstanceDesc.Get(),
                    m_scene.dirtyInstanceList);
                m_frameStats.tlasWasRebuilt = true;
            } else {
                // Transform-only: partial refit — only update dirty instance descriptors
                m_topLevelASGenerator.UpdateAndRefit(
                    m_ctx.CmdList(),
                    m_topLevelASBuffers.pScratch.Get(),
                    m_topLevelASBuffers.pResult.Get(),
                    m_topLevelASBuffers.pInstanceDesc.Get(),
                    m_scene.dirtyInstanceList);
                m_frameStats.tlasWasRefit = true;
            }
        }
        m_scene.tlasDirty = false;
    }
    m_frameStats.tlasMs = std::chrono::duration<float, std::milli>(
        std::chrono::high_resolution_clock::now() - t_tlasStart).count();
    { auto b = CD3DX12_RESOURCE_BARRIER::UAV(m_topLevelASBuffers.pResult.Get());
      cmdList->ResourceBarrier(1, &b); }

    // Bind main descriptor heap
    ID3D12DescriptorHeap* heaps[] = { m_srvUavHeap.Get() };
    cmdList->SetDescriptorHeaps(1, heaps);

    // ── Build ray dispatch descriptor ────────────────────────────
    // Pre-DLSS passes render at internal resolution; post-DLSS at display res
    const UINT renderW = m_dlss.RenderWidth();
    const UINT renderH = m_dlss.RenderHeight();

    D3D12_DISPATCH_RAYS_DESC raysDesc{};
    raysDesc.Width = renderW; raysDesc.Height = renderH; raysDesc.Depth = 1;
    const uint64_t sbtStart = m_sbtStorage->GetGPUVirtualAddress();
    const uint32_t rgSize   = m_sbtHelper.GetRayGenEntrySize();

    raysDesc.MissShaderTable.StartAddress  = sbtStart + m_sbtHelper.GetRayGenSectionSize();
    raysDesc.MissShaderTable.SizeInBytes   = m_sbtHelper.GetMissSectionSize();
    raysDesc.MissShaderTable.StrideInBytes = m_sbtHelper.GetMissEntrySize();

    raysDesc.HitGroupTable.StartAddress    = raysDesc.MissShaderTable.StartAddress + raysDesc.MissShaderTable.SizeInBytes;
    raysDesc.HitGroupTable.SizeInBytes     = m_sbtHelper.GetHitGroupSectionSize();
    raysDesc.HitGroupTable.StrideInBytes   = m_sbtHelper.GetHitGroupEntrySize();

    if (m_sbtHelper.GetCallableSectionSize() > 0) {
        raysDesc.CallableShaderTable.StartAddress  = raysDesc.HitGroupTable.StartAddress + raysDesc.HitGroupTable.SizeInBytes;
        raysDesc.CallableShaderTable.SizeInBytes   = m_sbtHelper.GetCallableSectionSize();
        raysDesc.CallableShaderTable.StrideInBytes = m_sbtHelper.GetCallableEntrySize();
    }

    // Upload async light tree TLAS refit data if available
    UploadLightTreeTLAS(cmdList);

    // Re-upload emissive triangle data if emission values changed
    UploadEmissiveBuffers(cmdList);

    // Output → UAV
    { auto b = CD3DX12_RESOURCE_BARRIER::Transition(m_outputResource.Get(),
          D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
      cmdList->ResourceBarrier(1, &b); }

    // Clear global counters
    { auto b = CD3DX12_RESOURCE_BARRIER::Transition(m_globalCounterBuffer.Get(),
          D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST);
      cmdList->ResourceBarrier(1, &b);
      cmdList->CopyBufferRegion(m_globalCounterBuffer.Get(), 0, m_zeroBuffer.Get(), 0, MAX_STACKS * sizeof(uint32_t));
      auto b2 = CD3DX12_RESOURCE_BARRIER::Transition(m_globalCounterBuffer.Get(),
          D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
      cmdList->ResourceBarrier(1, &b2); }

    // ── Execute pass pipeline ────────────────────────────────────
    uint32_t currentStack = 0, nextStack = 1;
    std::vector<std::pair<int, uint32_t>> loopStack;
    // Pre-DLSS: dispatch at render resolution. Post-DLSS: display resolution.
    UINT dispW = renderW, dispH = renderH;

    // Precompute ReSTIR root constants (slots 4-19, reused across all dispatches)
    // Clamp to safe ranges: min <= max, all >= 1 to prevent uint wrap / div-by-zero
    auto& rs = m_restirSettings;
    rs.tempMcapDI     = std::max(rs.tempMcapDI, 1);
    rs.tempMcapGI     = std::max(rs.tempMcapGI, 1);
    rs.spatCountMaxDI = std::max(rs.spatCountMaxDI, 1);
    rs.spatCountMinDI = std::clamp(rs.spatCountMinDI, 1, rs.spatCountMaxDI);
    rs.spatRadMaxDI   = std::max(rs.spatRadMaxDI, 4);
    rs.spatRadMinDI   = std::clamp(rs.spatRadMinDI, 4, rs.spatRadMaxDI);
    rs.spatCountMaxGI = std::max(rs.spatCountMaxGI, 1);
    rs.spatCountMinGI = std::clamp(rs.spatCountMinGI, 1, rs.spatCountMaxGI);
    rs.spatRadMaxGI   = std::max(rs.spatRadMaxGI, 4);
    rs.spatRadMinGI   = std::clamp(rs.spatRadMinGI, 4, rs.spatRadMaxGI);

    UINT rsConsts[20] = {};
    rsConsts[4]  = (UINT)rs.tempMcapDI;
    rsConsts[5]  = (UINT)rs.tempMcapGI;
    rsConsts[6]  = (UINT)rs.spatCountMaxDI;
    rsConsts[7]  = (UINT)rs.spatCountMinDI;
    rsConsts[8]  = (UINT)rs.spatRadMaxDI;
    rsConsts[9]  = (UINT)rs.spatRadMinDI;
    rsConsts[10] = (UINT)rs.spatCountMaxGI;
    rsConsts[11] = (UINT)rs.spatCountMinGI;
    rsConsts[12] = (UINT)rs.spatRadMaxGI;
    rsConsts[13] = (UINT)rs.spatRadMinGI;
    rsConsts[14] = rs.Flags();
    memcpy(&rsConsts[15], &rs.reuseRoughnessMin, 4);
    memcpy(&rsConsts[16], &rs.reuseRoughnessMax, 4);

    auto setConsts = [&](UINT w, UINT h, UINT stackIn, UINT stackOut) {
        rsConsts[0] = w; rsConsts[1] = h; rsConsts[2] = stackIn; rsConsts[3] = stackOut;
        cmdList->SetComputeRoot32BitConstants(1, 20, rsConsts, 0);
    };

    for (size_t i = 0; i < m_passes.Passes().size(); ++i) {
        auto& p = m_passes.Passes()[i];

        switch (p.stage) {
        case Stage::LoopStart:
            loopStack.push_back({ -1, p.loopCount });
            break;

        case Stage::PingSwap:
            std::swap(currentStack, nextStack);
            break;

        case Stage::LoopEnd:
            if (!loopStack.empty()) {
                loopStack.back().second--;
                if (loopStack.back().second > 0) i = p.targetIdx;
                else loopStack.pop_back();
            }
            break;

        case Stage::Barrier:
            { auto u = CD3DX12_RESOURCE_BARRIER::UAV(nullptr);
              cmdList->ResourceBarrier(1, &u); }
            break;

        case Stage::ClearSort:
            ClearSortBuffers(cmdList);
            break;

        case Stage::RayGen:
        {
            cmdList->SetPipelineState1(m_rtStateObject.Get());
            cmdList->SetComputeRootSignature(m_rayGenSignature.Get());
            cmdList->SetComputeRootDescriptorTable(0, m_srvUavHeap->GetGPUDescriptorHandleForHeapStart());
            setConsts(dispW, dispH, 0, 0);

            uint32_t rgSlot = m_passes.PassIndexByFile(p.file);
            raysDesc.RayGenerationShaderRecord.StartAddress = sbtStart + rgSlot * rgSize;
            raysDesc.RayGenerationShaderRecord.SizeInBytes  = rgSize;
            cmdList->DispatchRays(&raysDesc);
            break;
        }

        case Stage::Compute:
        {
            if (p.isWorkGraph) {
                const auto& rt = m_wgRuntime[p.wgIdx];
                cmdList->SetComputeRootSignature(m_computeSignature.Get());
                cmdList->SetComputeRootDescriptorTable(0, m_srvUavHeap->GetGPUDescriptorHandleForHeapStart());
                setConsts(dispW, dispH, currentStack, nextStack);
                D3D12_SET_PROGRAM_DESC sp{}; sp.Type = D3D12_PROGRAM_TYPE_WORK_GRAPH;
                sp.WorkGraph.ProgramIdentifier = rt.id; sp.WorkGraph.BackingMemory = rt.backing;
                static std::vector<bool> s_inited;
                if (s_inited.size() <= p.wgIdx) s_inited.resize(p.wgIdx + 1, false);
                sp.WorkGraph.Flags = s_inited[p.wgIdx] ? D3D12_SET_WORK_GRAPH_FLAG_NONE : D3D12_SET_WORK_GRAPH_FLAG_INITIALIZE;
                cmdList->SetProgram(&sp); s_inited[p.wgIdx] = true;
                D3D12_DISPATCH_GRAPH_DESC dg{}; dg.Mode = D3D12_DISPATCH_MODE_NODE_CPU_INPUT;
                dg.NodeCPUInput.EntrypointIndex = 0; dg.NodeCPUInput.NumRecords = 1;
                cmdList->DispatchGraph(&dg);
            } else {
                cmdList->SetPipelineState(m_csPSOs[p.psoIdx].Get());
                cmdList->SetComputeRootSignature(m_computeSignature.Get());
                cmdList->SetComputeRootDescriptorTable(0, m_srvUavHeap->GetGPUDescriptorHandleForHeapStart());
                setConsts(dispW, dispH, currentStack, nextStack);
                cmdList->Dispatch(
                    (dispW + p.groupX - 1) / p.groupX,
                    (dispH + p.groupY - 1) / p.groupY, 1);
            }
            break;
        }

        case Stage::FixedCompute:
        {
            cmdList->SetPipelineState(m_csPSOs[p.psoIdx].Get());
            cmdList->SetComputeRootSignature(m_computeSignature.Get());
            cmdList->SetComputeRootDescriptorTable(0, m_srvUavHeap->GetGPUDescriptorHandleForHeapStart());
            setConsts(dispW, dispH, currentStack, nextStack);
            cmdList->Dispatch(p.groupX, p.groupY, 1);
            break;
        }

        case Stage::Wavefront:
        {
            bool inPlace = (currentStack == nextStack);
            cmdList->SetPipelineState(inPlace ? m_psoSetupIndirectNoClear.Get() : m_psoSetupIndirect.Get());
            cmdList->SetComputeRootSignature(m_rsSetupIndirect.Get());
            UINT inc = m_ctx.Device()->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            auto gpuH = m_srvUavHeap->GetGPUDescriptorHandleForHeapStart();
            gpuH.ptr += 33 * inc;
            cmdList->SetComputeRootDescriptorTable(0, gpuH);
            UINT setupC[4] = { currentStack, 0, nextStack, p.groupX };
            cmdList->SetComputeRoot32BitConstants(1, 4, setupC, 0);
            cmdList->Dispatch(1, 1, 1);

            CD3DX12_RESOURCE_BARRIER pre[] = {
                CD3DX12_RESOURCE_BARRIER::UAV(m_indirectArgsBuffer.Get()),
                CD3DX12_RESOURCE_BARRIER::Transition(m_indirectArgsBuffer.Get(),
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT) };
            cmdList->ResourceBarrier(2, pre);

            cmdList->SetPipelineState(m_csPSOs[p.psoIdx].Get());
            cmdList->SetComputeRootSignature(m_computeSignature.Get());
            cmdList->SetComputeRootDescriptorTable(0, m_srvUavHeap->GetGPUDescriptorHandleForHeapStart());
            setConsts(dispW, dispH, currentStack, nextStack);
            cmdList->ExecuteIndirect(m_commandSignature.Get(), 1, m_indirectArgsBuffer.Get(), 0, nullptr, 0);

            CD3DX12_RESOURCE_BARRIER post[] = {
                CD3DX12_RESOURCE_BARRIER::Transition(m_indirectArgsBuffer.Get(),
                    D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
                CD3DX12_RESOURCE_BARRIER::UAV(m_stackBuffers[nextStack].Get()),
                CD3DX12_RESOURCE_BARRIER::UAV(m_globalCounterBuffer.Get()) };
            cmdList->ResourceBarrier(3, post);
            break;
        }

        case Stage::DLSS:
        {
            // UAV barriers on all DLSS inputs
            ID3D12Resource* uavs[] = {
                m_dlss.Depth(), m_dlss.MVec(), m_dlss.Normals(),
                m_dlss.DiffuseAlbedo(), m_dlss.SpecularAlbedo(),
                m_dlss.Roughness(), m_dlss.SpecMVec(), m_dlss.SpecHitDist(),
                m_dlss.Transparency(), m_dlss.ColorBeforeTrans(), m_dlss.Input()
            };
            for (auto* r : uavs) {
                if (r) { auto b = CD3DX12_RESOURCE_BARRIER::UAV(r);
                         cmdList->ResourceBarrier(1, &b); }
            }

            m_dlss.Evaluate(cmdList, m_ctx.Device(),
                *m_ctx.frameToken, m_ctx.viewportHandle,
                m_aspectRatio,
                m_camera.ViewMatrix(), m_camera.PrevView(), m_camera.PrevProj(),
                m_camera.JitterX(), m_camera.JitterY(), m_camera.JitterFrame());

            // Now safe to advance prev matrices for next frame
            m_camera.AdvanceFrame();

            // After DLSS: post-process passes run at display resolution
            dispW = GetWidth();
            dispH = GetHeight();

            // Rebind our heap (DLSS may have changed it)
            ID3D12DescriptorHeap* h[] = { m_srvUavHeap.Get() };
            cmdList->SetDescriptorHeaps(1, h);
            break;
        }

        default: break;
        } // switch
    } // for passes

    // ── Copy output → back buffer ────────────────────────────────
    { auto toSrc = CD3DX12_RESOURCE_BARRIER::Transition(m_outputResource.Get(),
          D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
      cmdList->ResourceBarrier(1, &toSrc); }

    { auto toDst = CD3DX12_RESOURCE_BARRIER::Transition(m_ctx.BackBuffer(),
          D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_DEST);
      cmdList->ResourceBarrier(1, &toDst); }

    UINT layer = m_displayLevels[m_currentDisplayLevel];
    UINT sub   = D3D12CalcSubresource(0, layer, 0, 1, 60);
    CD3DX12_TEXTURE_COPY_LOCATION src(m_outputResource.Get(), sub);
    CD3DX12_TEXTURE_COPY_LOCATION dst(m_ctx.BackBuffer(), 0);
    D3D12_BOX box = { 0, 0, 0, GetWidth(), GetHeight(), 1 };
    cmdList->CopyTextureRegion(&dst, 0, 0, 0, &src, &box);

    // Editor overlay — render ImGui on top of the composited frame
    if (m_editor.IsVisible()) {
        // Back buffer is in COPY_DEST after the texture copy — transition to RT
        auto toRT = CD3DX12_RESOURCE_BARRIER::Transition(m_ctx.BackBuffer(),
            D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_RENDER_TARGET);
        cmdList->ResourceBarrier(1, &toRT);

        // ImGui needs the render target bound and the SRV heap active
        auto rtv = m_ctx.CurrentRTV();
        cmdList->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
        ID3D12DescriptorHeap* heaps[] = { m_srvUavHeap.Get() };
        cmdList->SetDescriptorHeaps(1, heaps);

        m_editor.Render(cmdList);

        // Present barrier (from RENDER_TARGET)
        auto toPres = CD3DX12_RESOURCE_BARRIER::Transition(m_ctx.BackBuffer(),
            D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT);
        cmdList->ResourceBarrier(1, &toPres);
    } else {
        // No editor — go straight from COPY_DEST to PRESENT
        auto b = CD3DX12_RESOURCE_BARRIER::Transition(m_ctx.BackBuffer(),
            D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PRESENT);
        cmdList->ResourceBarrier(1, &b);
    }
}

// Old input handlers removed — EngineApp handles input now.
