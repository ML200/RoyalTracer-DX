#include "stdafx.h"
#include "Renderer.h"
#include "../shaders/DlssGuideLayout.h"
#include "ReuseTextureGen.h"
#include "Diagnostics.h"
#include "Core/PerformanceCapture.h"
#include "Windowsx.h"
#include <random>
#include <unordered_set>
#include <DirectXPackedVector.h>
#include <DirectXTex.h>

#undef SL_CHECK
#define SL_CHECK(x)                                                                                                    \
    do {                                                                                                               \
        sl::Result r = (x);                                                                                            \
        if (r != sl::Result::eOk) {                                                                                    \
            std::wcout << L"[SL] " << L#x << L" failed: " << (int)r << std::endl;                                      \
        }                                                                                                              \
    } while (0)

Renderer::Renderer(UINT width, UINT height)
    : m_width(width), m_height(height), m_aspectRatio(static_cast<float>(width) / static_cast<float>(height)) {
    m_passes.Build({
        L"Pass_camera_v8.hlsl|rg",
        L"barrier",
        L"Pass_pt_skybake_v8.hlsl|fx:512",
        L"barrier",
        L"Pass_light_learning_v8.hlsl|fx:256",
        L"barrier",
        L"Pass_sharc_prepare_v8.hlsl|fx:4096",
        L"barrier",
        L"Pass_sharc_update_v8.hlsl|rg",
        L"barrier",
        L"Pass_sharc_resolve_v8.hlsl|fx:4096",
        L"barrier",
        L"Pass_sharc_debug_v8.hlsl|cs:16x16",
        L"barrier",

        // Cache-driven path tracing: trace to the first wide vertex, pick a light there, shade it.
        L"loop:pt_samples",
        L"Pass_pt_trace_v8.hlsl|rg",
        L"barrier",
        L"Pass_pt_light_v8.hlsl|cs:16x16",
        L"barrier",
        L"Pass_pt_shade_v8.hlsl|cs:16x16",
        L"barrier",
        L"endloop",

        L"Pass_lite_shift_v8.hlsl|cs:16x16",
        L"barrier",
        L"Pass_lite_merge_v8.hlsl|cs:16x16",
        L"barrier",
        L"Pass_atmosphere_primary_v8.hlsl|cs:8x8",
        L"barrier",

        L"Pass_shading_v8.hlsl|cs:16x16",
        L"barrier",
        L"dlss",
        L"barrier",
        L"Pass_autoexpose_reduce_v8.hlsl|cs:8x8",
        L"barrier",
        L"Pass_autoexpose_finalize_v8.hlsl|fx:1",
        L"barrier",
        L"Pass_postprocess_v8.hlsl|cs:8x4",
        L"barrier",
    });
}

void Renderer::InitDevice() {
    try {
        m_ctx.Init(Win32Application::GetHwnd(), GetWidth(), GetHeight());

        {
            planet::StreamConfig cfg;

            cfg.enabled = false;
            cfg.planet.radius = 6371000.0;

            cfg.planet.center = {0.0, -cfg.planet.radius, 0.0};
            cfg.max_lod = 16;
            cfg.max_triangles = 3000000u;

            m_planet.init(m_ctx.Device(), &m_ctx, cfg);

            m_camera.planetCenter =
                glm::vec3((float)cfg.planet.center.x, (float)cfg.planet.center.y, (float)cfg.planet.center.z);
            m_camera.planetRadius = (float)cfg.planet.radius;

            m_camera.terrainHeightFrequency = 0.0f;
        }

        m_simulator.PromptUserConfiguration();
        m_recorder.Initialize();
        m_camera.Init(m_ctx.Device(), GetWidth(), GetHeight());

        GenerateLutTextures();
        InitSkyStarsTexture();
        InitBlueNoiseTexture();
        // Create atmospheric LUTs before their scene SRVs.
        InitSkyLUTBake();
        D3D12_FEATURE_DATA_D3D12_OPTIONS5 opts5 = {};
        ThrowIfFailed(m_ctx.Device()->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &opts5, sizeof(opts5)));
        if (opts5.RaytracingTier < D3D12_RAYTRACING_TIER_1_0)
            throw std::runtime_error("Raytracing not supported on device");

        {
            sl::ReflexState state{};
            sl::Result rr = slReflexGetState(state);
            if (rr == sl::Result::eOk) {
                m_reflexAvailable = state.lowLatencyAvailable;
                LOG(L"[Reflex] Low latency available: " << (m_reflexAvailable ? L"yes" : L"no"));
            } else {
                LOG(L"[Reflex] GetState failed: " << (int)rr);
            }
            sl::ReflexOptions options{};
            options.mode = sl::ReflexMode::eLowLatency;
            SL_CHECK(slReflexSetOptions(options));
        }

        {
            LUID luid = m_ctx.Device()->GetAdapterLuid();
            sl::AdapterInfo ai;
            ai.deviceLUID = (uint8_t*)&luid;
            ai.deviceLUIDSizeInBytes = sizeof(LUID);
            sl::Result sr = slIsFeatureSupported(sl::kFeatureDLSS_G, ai);
            if (sr == sl::Result::eOk) {
                m_dlssG.available = true;

                sl::DLSSGState gState{};
                sl::DLSSGOptions gOpts{};
                gOpts.mode = sl::DLSSGMode::eOff;
                if (slDLSSGGetState(m_ctx.viewportHandle, gState, &gOpts) == sl::Result::eOk) {
                    m_dlssG.maxFrames = std::max(1, (int)gState.numFramesToGenerateMax);
                }
                LOG(L"[DLSS-G] Feature supported, maxFrames=" << m_dlssG.maxFrames);
            } else {
                LOG(L"[DLSS-G] Not supported: " << (int)sr);
            }
        }
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
        m_rockMeshIndices.clear();
        if (m_planet.enabled()) {
            const auto rockVariants = planet::generate_rock_variants(6, 2, 0xB0DECA11u);
            Material rockMat;
            rockMat.Kd = DirectX::XMFLOAT4(0.30f, 0.27f, 0.24f, 1.0f);
            rockMat.Ke = DirectX::XMFLOAT3(0.0f, 0.0f, 0.0f);
            rockMat.Ni = 1.0f;
            rockMat.Pr_Pm_Ps_Pc = DirectX::XMFLOAT4(1.0f, 0.0f, 0.0f, 0.0f);
            rockMat.albedoTexID = -1;
            rockMat.normalTexID = -1;
            rockMat.rmaTexID = -1;
            rockMat.alphaThreshold = 1.0f;
            for (const auto& rm : rockVariants) {
                std::vector<Vertex> rvtx;
                rvtx.reserve(rm.vertices.size());
                for (const auto& rv : rm.vertices)
                    rvtx.emplace_back(DirectX::XMFLOAT3(rv.position.x, rv.position.y, rv.position.z),
                                      DirectX::XMFLOAT4(rv.normal.x, rv.normal.y, rv.normal.z, 0.0f),
                                      DirectX::XMFLOAT2(rv.u, rv.v));
                m_rockMeshIndices.push_back(CreateProceduralMesh(rvtx, rm.indices, rockMat));
            }
        }
        // MINECRAFT: attach the voxel streamer before the unified TLAS is sized.
        m_planet.set_external(m_voxels.enabled() ? &m_voxels : nullptr);
        m_planet.set_external2(m_ocean.Enabled() ? &m_ocean : nullptr);
        m_planet.reserve_scene_instances((uint32_t)m_scene.instances.size());
        CreateAccelerationStructures();

        {
            const auto tr = m_planet.terrain_reservation();
            m_scene.ReserveTerrain(tr.vertexElems, tr.indexElems, tr.matIDElems, tr.triLightElems, tr.instanceSlots,
                                   tr.propsBase);
        }

        if (m_planet.enabled())
            m_scene.ReserveRocks(planet::MAX_ROCK_INSTANCES);

        if (m_voxels.enabled()) {
            if (m_scene.instances.size() > 4096)
                throw std::runtime_error(
                    "Minecraft world: more than 4096 scene instances are not supported alongside it");
            const mc::StreamerConfig& sc = m_voxels.config();
            m_scene.ReserveVoxels(sc.vertexCapacity, sc.indexCapacity, sc.matIdCapacity, sc.maxInstances, 4096u);
        }

        if (m_ocean.Enabled()) {
            const auto orv = m_ocean.GetReservation();
            m_scene.ReserveOcean(orv.vertexElems, orv.indexElems, orv.matIDElems, orv.instanceSlots,
                                 ocean::OceanSystem::MakeMaterial(m_ocean.GetParams()));
        }

        m_scene.BuildGlobalMeshBuffers(m_ctx.Device(), m_ctx.CmdList());
        m_ctx.FlushAndReset();

        m_scene.vertexGlobalUpload.Reset();
        m_scene.indexGlobalUpload.Reset();

        if (m_planet.enabled() && !m_rockMeshIndices.empty()) {
            std::vector<planet::StreamOrchestrator::RockVariantGPU> rockDescs;
            rockDescs.reserve(m_rockMeshIndices.size());
            for (UINT mi : m_rockMeshIndices) {
                const auto& rmesh = m_scene.meshes[mi];
                planet::StreamOrchestrator::RockVariantGPU d;
                d.blas_va = rmesh.blas ? rmesh.blas->GetGPUVirtualAddress() : 0;
                d.vertexBase = rmesh.globalVertexBase;
                d.indexBase = rmesh.globalIndexBase;
                d.triCount = rmesh.opaqueTriCount;
                rockDescs.push_back(d);
            }
            m_planet.set_rock_variants(m_scene.rockPropsBase, rockDescs);

            planet::RockScatterConfig rc;
            rc.planet = m_planet.planet_geometry();
            rc.max_rocks = planet::MAX_ROCK_INSTANCES;
            m_rockScatter.configure(rc, (int)m_rockMeshIndices.size());
        }
        m_lutUploadHeaps.clear();
        m_skyStarsUploadHeap.Reset();

        CreateRaytracingPipeline();
        CreateRaytracingOutputBuffer();
        CreateReadbackBuffer();
        m_scene.CreateInstancePropertiesBuffer(m_ctx.Device());

        m_scene.UploadMaterials(m_ctx.Device());

        if (m_ocean.Enabled()) {
            // Bind before Init: the acceleration-structure size query needs the vertex span, which
            // depends on where the ocean's range landed in the global buffer.
            m_ocean.BindScene(m_scene.vertexGlobal.Get(), m_scene.indexGlobal.Get(), m_scene.oceanVertexBase,
                              m_scene.oceanIndexBase, m_scene.oceanMatIDBase, m_scene.oceanPropsBase,
                              m_scene.oceanMatIndex);
            m_ocean.Init(m_ctx.Device(), &m_ctx);
            // The orchestrator hands this pointer to every external stream so it can fill in its
            // own instance records; without it the ocean's tiles would have no geometry offsets.
            m_planet.bind_instance_properties(m_scene.instanceProperties.Get());
            m_camera.oceanInstanceBase = m_scene.oceanPropsBase;
            m_camera.oceanEnabled = true;
        }

        m_dlss.CreateResources(m_ctx.Device(), GetWidth(), GetHeight());

        m_dlssNR.Initialize(GetWidth(), GetHeight());
        CreateShaderResourceHeap();
        CreateShaderBindingTable();

        if (m_voxels.enabled()) {
            BindVoxelLights(m_ctx.CmdList());
            m_ctx.FlushAndReset();
        }

        if (m_planet.enabled()) {
            const auto tr = m_planet.terrain_reservation();
            m_planet.bind_geometry(m_scene.vertexGlobal.Get(), m_scene.vertexGlobalMapped, m_scene.indexGlobal.Get(),
                                   m_scene.indexGlobalMapped, m_scene.instanceProperties.Get(),
                                   m_scene.totalVertexCount, m_scene.totalIndexCount, m_scene.combinedVertexCount(),
                                   tr.propsBase, tr.leafSlots, m_scene.terrainMatIDBase, m_scene.terrainTriLightBase);
        }

        if (m_voxels.enabled()) {
            m_voxels.bind_geometry(m_scene.vertexGlobal.Get(), m_scene.indexGlobal.Get(), m_scene.combinedVertexCount(),
                                   m_scene.voxelVertexBase, m_scene.voxelIndexBase, m_scene.materialIndexBuffer.Get(),
                                   m_scene.voxelMatIDBase, m_scene.voxelPropsBase);
            if (m_vxOmm.valid && m_vxOmm.ommArray && m_vxOmmIndices)
                m_voxels.bind_omm(m_vxOmmIndices.Get(), m_vxOmm.ommArray->GetGPUVirtualAddress(), &m_vxOmmTable);
            m_planet.bind_instance_properties(m_scene.instanceProperties.Get());

            glm::vec3 eye, center, up;
            nv_helpers_dx12::CameraManip.getLookat(eye, center, up);
            const glm::vec3 origin = m_camera.getSceneOriginWorld();
            const double camPos[3] = {(double)eye.x + origin.x, (double)eye.y + origin.y, (double)eye.z + origin.z};
            m_voxels.warm_up(camPos);
        }

        m_scene.tlasDirty = false;
        m_scene.tlasFullRebuild = false;
        m_scene.lightTreeDirty = false;
        m_scene.materialsDirty = false;

        m_blasLocalRoots = lt::ComputeBLASLocalRoots(m_scene.emissiveTriangles);

        {
            const UINT inc = m_ctx.Device()->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            CD3DX12_CPU_DESCRIPTOR_HANDLE fontCpu(m_srvUavHeap->GetCPUDescriptorHandleForHeapStart(),
                                                  IMGUI_FONT_HEAP_SLOT, inc);
            CD3DX12_GPU_DESCRIPTOR_HANDLE fontGpu(m_srvUavHeap->GetGPUDescriptorHandleForHeapStart(),
                                                  IMGUI_FONT_HEAP_SLOT, inc);
            m_editor.Init(Win32Application::GetHwnd(), m_ctx.Device(), m_ctx.BufferCount(), m_srvUavHeap.Get(), fontCpu,
                          fontGpu);
        }

        dxdiag::DumpNewMessages();
    } catch (const std::exception& e) {
        dxdiag::CheckDeviceRemoved(m_ctx.Device(), 1000);
        wchar_t wMsg[4096];
        MultiByteToWideChar(CP_UTF8, 0, e.what(), -1, wMsg, 4096);
        MessageBoxW(NULL, wMsg, L"Fatal Init Error", MB_OK | MB_ICONERROR);
        exit(1);
    }
}

// Apply scene edits and synchronize resources before recording the next frame.
void Renderer::UpdateRenderer(float dt) {
    m_lastDt = dt;
    using hrc = std::chrono::high_resolution_clock;

    // Reflex sleep — must be called every frame regardless of mode
    slReflexSleep(*m_ctx.frameToken);

    slPCLSetMarker(sl::PCLMarker::eSimulationStart, *m_ctx.frameToken);

    if (m_simulator.IsActive()) {
        bool shouldCapture = false;
        bool finished = m_simulator.Update(dt, m_camera.Manipulator(), shouldCapture);
        if (shouldCapture)
            SaveSimulationData(m_simulator.GetLastCaptureIndex());
        if (finished) {
            LOG(L"\n[Sim] Data Generation Complete.");
            PostQuitMessage(0);
            return;
        }
    }

    auto t_updateStart = hrc::now();
    m_time++;

    if (m_integratorSettings.compactLightTree != m_lightTreeCompact) {
        // Drain users of the old descriptors before replacing the selected layout.
        m_ctx.WaitForGPU();
        m_lightTreeRefit.DiscardPending();
        m_lightTreeCompact = m_integratorSettings.compactLightTree;
        m_emissiveGpuDirty = true;
        m_lightLearningResetPending = true;
        m_dlss.ForceReset();
        m_dlssNR.ForceReset();
    }

    if (m_dlss.mode != m_dlss.ActiveMode()) {
        m_ctx.WaitForGPU(); // drain all in-flight GPU work BEFORE releasing old textures
        if (m_dlss.UpdateMode(m_ctx.Device())) {
            m_dlssModeChangedFrames = 2;
            m_camera.ResetJitter();

            m_dlssNR.ForceReset();
            RebuildDLSSDescriptors();
            LOG(L"[DLSS] Mode changed → render res: " << m_dlss.RenderWidth() << L"x" << m_dlss.RenderHeight());
        }
    }

    if (m_dlss.clampEmitterSpikes != m_dlssClampEmitterSpikesPrev) {
        m_dlssClampEmitterSpikesPrev = m_dlss.clampEmitterSpikes;
        m_dlss.ForceReset();
        LOG(L"[DLSS] Emitter-spike clamp " << (m_dlss.clampEmitterSpikes ? L"ON" : L"OFF"));
    }

    for (int c = 0; c < (int)Scene::LightClassCount; ++c) {
        if (m_lightClassApplied[c] == m_scene.lightClassEnabled[c])
            continue;
        m_lightClassApplied[c] = m_scene.lightClassEnabled[c];
        m_scene.MarkAllInstancesDirty();
        m_scene.lightTreeDirty = true;
    }

    bool voxelLightsDue = false;
    if (m_voxels.enabled() && m_voxels.lights_bound()) {
        const double nowS = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
        voxelLightsDue = m_voxels.light_version() != m_voxelLightKickedVersion && nowS - m_lastVoxelLightKick >= 0.1;
    }
    // Keep old light buffers alive while earlier frames may reference them.
    for (size_t i = 0; i < m_vxLightRetired.size();) {
        if (m_time > m_vxLightRetired[i].frame + 4) {
            m_vxLightRetired[i] = m_vxLightRetired.back();
            m_vxLightRetired.pop_back();
        } else
            ++i;
    }
    if ((m_scene.lightTreeDirty || voxelLightsDue) && !m_lightTreeRefit.IsPending()) {
        if (m_scene.emissivesDirty) {
            m_scene.CollectEmissiveTriangles();
            m_blasLocalRoots = lt::ComputeBLASLocalRoots(m_scene.emissiveTriangles);
            m_emissiveGpuDirty = true;

            m_scene.MarkAllInstancesDirty();
            m_scene.emissivesDirty = false;
            LOG(L"[LightTree] Recomputed emissives + BLAS roots (emission change)");
        }
        KickLightTreeRefit();
        m_scene.lightTreeDirty = false;
    }

    lt::TLASRefitResult refitResult;
    if (m_lightTreeRefit.PollResult(refitResult)) {
        m_pendingTLASUpload = std::move(refitResult.nodes);
        m_pendingPackedTLAS = std::move(refitResult.packedNodes);
        m_pendingBLASBitTrail = std::move(refitResult.blasBitTrails);

        m_pendingSlotRecords = std::move(refitResult.slots);
        m_pendingVoxelLightVersion = refitResult.extraVersion;
        m_pendingVoxelLeafCount = refitResult.extraLeafCount;
        m_pendingTlasIncremental = refitResult.incremental;
        m_frameStats.lightBvh.buildCpuMs = refitResult.worker_cpu_ms;
        m_frameStats.lightBvh.buildMeasured = true;
        m_frameStats.lightBvh.incremental = refitResult.incremental;

        LOG(L"[LightTree] Async TLAS refit ready: "
            << m_pendingTLASUpload.size() << L" nodes" << (m_pendingTlasIncremental ? L" (incremental)" : L" (rebuilt)")
            << (m_voxels.enabled() ? L", voxel light chunks " + std::to_wstring(m_pendingVoxelLeafCount) : L""));
    }

    auto& lightStats = m_frameStats.lightBvh;
    lightStats.nodes = static_cast<UINT>(m_publishedLightTLAS.size());
    lightStats.voxelLeaves = m_liveVoxelLightLeaves;
    lightStats.pending = m_lightTreeRefit.IsPending() || !m_pendingTLASUpload.empty();
    const auto bytes = [](ID3D12Resource* resource) -> UINT64 { return resource ? resource->GetDesc().Width : 0; };
    const auto& baseLightBuffers = m_lightTree.GetGpu();
    lightStats.nodeBytes = bytes(m_ltTlasGpu ? m_ltTlasGpu.Get() : baseLightBuffers.TLASNodes.Get());
    lightStats.slotBytes = bytes(m_ltSlotGpu ? m_ltSlotGpu.Get() : baseLightBuffers.Slots.Get());
    lightStats.trailBytes =
        bytes(m_ltBlasBitTrailGpu ? m_ltBlasBitTrailGpu.Get() : baseLightBuffers.BLASBitTrail.Get());

    static FlyCamController dummyFlyCam;
    m_editor.Draw(m_scene, m_camera, m_flyCam ? *m_flyCam : dummyFlyCam, m_passes, m_dlss, m_dlssNR, m_dlssG,
                  m_integratorSettings, m_fps, m_frameStats, m_planet.stats(),
                  m_voxels.enabled() ? &m_voxels : nullptr);
    // Transport changes invalidate accumulated reconstruction and learning history.
    if (m_ocean.Enabled() && m_scene.oceanInstanceSlots && !m_scene.oceanMaterialEdited) {
        const Material water = ocean::OceanSystem::MakeMaterial(m_ocean.GetParams());
        const UINT base = m_scene.oceanMatIndex;
        const auto oldTf = m_scene.materials.Tf[base];
        const auto oldSss = m_scene.materials.sssAlbedo[base];
        if (m_scene.materials.Kd[base].w != water.Kd.w ||
            oldTf.x != water.Tf.x || oldTf.y != water.Tf.y || oldTf.z != water.Tf.z ||
            oldSss.x != water.sssAlbedo.x || oldSss.y != water.sssAlbedo.y || oldSss.z != water.sssAlbedo.z ||
            m_scene.materials.sssRadius[base] != water.sssRadius ||
            m_scene.materials.sssPhaseG[base] != water.sssPhaseG ||
            m_scene.materials.sssWeight[base] != water.sssWeight ||
            m_scene.materials.sssEnable[base] != water.sssEnable) {
            for (UINT i=0;i<OCEAN_MATERIAL_COUNT;++i) {
                const float coverage = float(i % OCEAN_MATERIAL_LEVELS) / (OCEAN_MATERIAL_LEVELS-1);
                m_scene.materials.Kd[base+i].w = water.Kd.w + (1.0f-water.Kd.w)*coverage;
                m_scene.materials.Tf[base+i] = water.Tf;
                m_scene.materials.sssAlbedo[base+i] = water.sssAlbedo;
                m_scene.materials.sssRadius[base+i] = water.sssRadius;
                m_scene.materials.sssPhaseG[base+i] = water.sssPhaseG;
                m_scene.materials.sssWeight[base+i] = water.sssWeight * (1.0f-coverage);
                m_scene.materials.sssEnable[base+i] = coverage < 1.0f ? water.sssEnable : 0u;
            }
            m_scene.materialsDirty = true;
        }
    }
    const auto& settings = m_integratorSettings;
    if (!m_integratorHistoryValid || settings.forceDiffuseMats != m_previousIntegratorSettings.forceDiffuseMats)
        m_lightLearningResetPending = true;
    if (!m_integratorHistoryValid || settings.ReconstructionKey() != m_previousIntegratorSettings.ReconstructionKey() ||
        m_scene.materialsDirty) {
        m_dlss.ForceReset();
        m_dlssNR.ForceReset();
    }
    if (settings.forceDiffuseMats != m_previousIntegratorSettings.forceDiffuseMats)
        m_sharcResetPending = true;
    m_previousIntegratorSettings = settings;
    m_integratorHistoryValid = true;

    const bool sharcStructureChanged = m_sharcInstanceState.size() != m_scene.instances.size();
    if (sharcStructureChanged)
        m_sharcInstanceState.resize(m_scene.instances.size());
    if (m_scene.materialsDirty || sharcStructureChanged) {
        m_lightLearningResetPending = true;
        m_sharcResetPending = true;
    }
    for (size_t i = 0; i < m_scene.instances.size(); ++i) {
        if (!sharcStructureChanged && i < m_scene.instanceDirty.size() && !m_scene.instanceDirty[i])
            continue;
        const auto& instance = m_scene.instances[i];
        auto& previous = m_sharcInstanceState[i];

        if (sharcStructureChanged || previous.meshIndex != instance.meshIndex)
            m_lightLearningResetPending = true;

        if (sharcStructureChanged || previous.meshIndex != instance.meshIndex) {
            m_sharcResetPending = true;
        }
        previous.transform = instance.worldTransform;
        previous.meshIndex = instance.meshIndex;
    }
    m_camera.PollSceneOrigin();
    // Rebasing changes GPU transforms even when scene objects stay still.
    if (m_camera.consumeOriginShifted()) {
        m_scene.MarkAllInstancesDirty();

        m_scene.tlasDirty = true;

        m_scene.lightTreeDirty = true;
    }
    const auto camOrigin = m_camera.getSceneOriginWorld();
    m_scene.sceneOriginWorld = {camOrigin.x, camOrigin.y, camOrigin.z};

    auto t_instStart = hrc::now();
    m_scene.PrepareInstanceProperties();
    auto t_instEnd = hrc::now();

    m_frameStats.cpuInstanceMs = std::chrono::duration<float, std::milli>(t_instEnd - t_instStart).count();
    m_frameStats.cpuUpdateMs = std::chrono::duration<float, std::milli>(t_instEnd - t_updateStart).count();
    m_frameStats.instanceCount = (UINT)m_scene.instances.size();
    m_frameStats.meshCount = (UINT)m_scene.meshes.size();

    slPCLSetMarker(sl::PCLMarker::eSimulationEnd, *m_ctx.frameToken);

    // Reuse upload buffers and read timestamps only after the frame fence.
    auto t_waitStart = hrc::now();
    m_ctx.WaitForPreviousFrame();
    m_dlssNR.PrepareFrameGPUIdle();
    m_frameStats.gpuWaitMs = std::chrono::duration<float, std::milli>(hrc::now() - t_waitStart).count();

    m_gpuProfiler.Readback(m_frameStats);

    SwapSampleBuffers();

    m_camera.UploadGPUBuffer(m_aspectRatio);

    if (m_camera.ConsumeResetPending()) {
        m_dlss.ForceReset();
        m_dlssNR.ForceReset();
    }
    m_scene.UploadInstanceProperties();
    if (m_scene.materialsDirty) {
        m_scene.UpdateMaterialBuffer();
        m_scene.materialsDirty = false;
    }
}

// Express light instances relative to the current GPU scene origin.
std::vector<InstanceXformCPU> Renderer::BuildXformsFromScene() const {
    std::vector<InstanceXformCPU> xf;
    xf.reserve(m_scene.instances.size());

    const XMVECTOR shift =
        XMVectorSet(m_scene.sceneOriginWorld.x, m_scene.sceneOriginWorld.y, m_scene.sceneOriginWorld.z, 0.0f);
    for (const auto& si : m_scene.instances) {
        InstanceXformCPU x{};
        XMMATRIX shifted = si.worldTransform;
        shifted.r[3] = XMVectorSubtract(shifted.r[3], shift);
        XMStoreFloat4x4(&x.objectToWorld, shifted);
        xf.push_back(x);
    }
    return xf;
}

void Renderer::KickLightTreeRefit() {
    // The worker owns a snapshot independent of later scene edits.
    auto xforms = BuildXformsFromScene();
    auto roots = m_blasLocalRoots;
    auto slots = m_scene.lightInstances;

    for (lt::LightInstanceRef& s : slots)
        if (!m_scene.InstanceLightsEnabled(s.instanceID))
            s.meshID = 0xFFFFFFFFu;

    std::vector<lt::TLASExtraLeaf> extra;
    uint32_t extraVersion = 0, slotCount = 0;
    if (m_voxels.enabled() && m_voxels.lights_bound()) {
        const planet::DVec3 origin{m_scene.sceneOriginWorld.x, m_scene.sceneOriginWorld.y, m_scene.sceneOriginWorld.z};
        m_voxels.light_snapshot(origin, extra, extraVersion, slotCount);
        m_voxelLightKickedVersion = extraVersion;
        m_lastVoxelLightKick =
            std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    lt::RequestIncrementalRefit(m_lightTreeRefit, std::move(roots), std::move(slots), m_lightTree.SlotRecords(),
                                std::move(xforms), std::move(extra), slotCount, extraVersion, &m_liveLightTlas,
                                m_lightTlasForceRebuild, m_lightTreeCompact);
    m_lightTlasForceRebuild = false;
}

// Publish tree nodes, traversal trails, and instance slots from one refit.
void Renderer::UploadLightTreeTLAS(ID3D12GraphicsCommandList* cmdList) {
    if (m_pendingTLASUpload.empty())
        return;
    // Topology changes can invalidate learned branch identities.
    if (!lt::SameLightTreeTopology(m_publishedLightTLAS, m_pendingTLASUpload)) {
        if (m_pendingTlasIncremental)
            m_lightLearningRevalidatePending = true;
        else
            m_lightLearningResetPending = true;
    }
    m_publishedLightTLAS = m_pendingTLASUpload;

    auto* dev = m_ctx.Device();
    const UINT inc = dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    const UINT nodeCount = (UINT)m_pendingTLASUpload.size();
    const UINT nodeStride = lt::LightTLASNodeStride(m_lightTreeCompact);
    const UINT64 nodeBytes = (UINT64)nodeCount * nodeStride;
    const UINT itemCount = (UINT)m_pendingBLASBitTrail.size();
    const UINT64 itemBytes = (UINT64)itemCount * sizeof(lt::LightTreeTrail);
    CD3DX12_RANGE readRange(0, 0);

    auto growBuffer = [&](ComPtr<ID3D12Resource>& gpu, ComPtr<ID3D12Resource>& upload, UINT& capacity, UINT needed,
                          UINT stride, DXGI_FORMAT fmt, UINT srvSlot, const wchar_t* name) {
        const UINT elemSize = stride ? stride : (fmt == DXGI_FORMAT_R32G32_UINT ? 8u : 4u);
        const UINT64 byteCount = (UINT64)needed * elemSize;

        if (needed > capacity) {
            capacity = needed * 2;
            const UINT64 allocBytes = (UINT64)capacity * elemSize;

            auto hp = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
            auto desc = CD3DX12_RESOURCE_DESC::Buffer(allocBytes);
            ThrowIfFailed(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST,
                                                       nullptr, IID_PPV_ARGS(&gpu)));
            gpu->SetName(name);

            CD3DX12_CPU_DESCRIPTOR_HANDLE srvHandle(m_srvUavHeap->GetCPUDescriptorHandleForHeapStart(), srvSlot, inc);

            D3D12_SHADER_RESOURCE_VIEW_DESC sd{};
            sd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            sd.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
            if (stride > 0) {
                sd.Format = DXGI_FORMAT_UNKNOWN;
                const UINT srvStride = srvSlot == LT_TLAS_SRV_SLOT ? 16u : stride;
                sd.Buffer.NumElements = (UINT)(allocBytes / srvStride);
                sd.Buffer.StructureByteStride = srvStride;
            } else {
                sd.Format = fmt;
                sd.Buffer.NumElements = capacity;
            }
            dev->CreateShaderResourceView(gpu.Get(), &sd, srvHandle);

            LOG(L"[LightTree] Grew " << name << L": " << capacity << L" elements");
        } else {
            auto b = CD3DX12_RESOURCE_BARRIER::Transition(gpu.Get(), D3D12_RESOURCE_STATE_GENERIC_READ,
                                                          D3D12_RESOURCE_STATE_COPY_DEST);
            cmdList->ResourceBarrier(1, &b);
        }

        if (!upload || byteCount > upload->GetDesc().Width) {
            auto hp = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
            auto desc = CD3DX12_RESOURCE_DESC::Buffer((UINT64)capacity * elemSize);
            ThrowIfFailed(dev->CreateCommittedResource(
                &hp, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&upload)));
        }
    };

    growBuffer(m_ltTlasGpu, m_tlasUploadStaging, m_ltTlasGpuCapacity, nodeCount, nodeStride,
               DXGI_FORMAT_UNKNOWN, LT_TLAS_SRV_SLOT, L"LT_TLAS_Refit");

    {
        void* p = nullptr;
        ThrowIfFailed(m_tlasUploadStaging->Map(0, &readRange, &p));
        const void* nodes = m_lightTreeCompact ? static_cast<const void*>(m_pendingPackedTLAS.data())
                                               : static_cast<const void*>(m_pendingTLASUpload.data());
        memcpy(p, nodes, nodeBytes);
        m_tlasUploadStaging->Unmap(0, nullptr);
    }

    cmdList->CopyBufferRegion(m_ltTlasGpu.Get(), 0, m_tlasUploadStaging.Get(), 0, nodeBytes);

    auto b1 = CD3DX12_RESOURCE_BARRIER::Transition(m_ltTlasGpu.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                                                   D3D12_RESOURCE_STATE_GENERIC_READ);
    cmdList->ResourceBarrier(1, &b1);

    if (itemCount > 0) {
        growBuffer(m_ltBlasBitTrailGpu, m_blasBitTrailUploadStaging, m_ltBlasBitTrailGpuCapacity, itemCount, 0,
                   DXGI_FORMAT_R32G32_UINT, LT_BLASBITTRAIL_SRV_SLOT, L"LT_BLASBitTrail_Refit");

        {
            void* p = nullptr;
            ThrowIfFailed(m_blasBitTrailUploadStaging->Map(0, &readRange, &p));
            memcpy(p, m_pendingBLASBitTrail.data(), itemBytes);
            m_blasBitTrailUploadStaging->Unmap(0, nullptr);
        }

        cmdList->CopyBufferRegion(m_ltBlasBitTrailGpu.Get(), 0, m_blasBitTrailUploadStaging.Get(), 0, itemBytes);

        auto b2 = CD3DX12_RESOURCE_BARRIER::Transition(m_ltBlasBitTrailGpu.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                                                       D3D12_RESOURCE_STATE_GENERIC_READ);
        cmdList->ResourceBarrier(1, &b2);
    }

    const UINT slotCount = (UINT)m_pendingSlotRecords.size();
    m_frameStats.lightBvh.slots = slotCount;
    if (slotCount > 0) {
        const UINT64 slotBytes = (UINT64)slotCount * sizeof(lt::LightSlotGpu);

        growBuffer(m_ltSlotGpu, m_slotUploadStaging, m_ltSlotGpuCapacity, slotCount, sizeof(lt::LightSlotGpu),
                   DXGI_FORMAT_UNKNOWN, LT_SLOT_SRV_SLOT, L"LT_Slots_Refit");

        {
            void* p = nullptr;
            ThrowIfFailed(m_slotUploadStaging->Map(0, &readRange, &p));
            memcpy(p, m_pendingSlotRecords.data(), slotBytes);
            m_slotUploadStaging->Unmap(0, nullptr);
        }

        cmdList->CopyBufferRegion(m_ltSlotGpu.Get(), 0, m_slotUploadStaging.Get(), 0, slotBytes);

        auto b3 = CD3DX12_RESOURCE_BARRIER::Transition(m_ltSlotGpu.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                                                       D3D12_RESOURCE_STATE_GENERIC_READ);
        cmdList->ResourceBarrier(1, &b3);
    }

    m_pendingTLASUpload.clear();
    m_pendingPackedTLAS.clear();
    m_pendingBLASBitTrail.clear();
    m_pendingSlotRecords.clear();
}

void Renderer::UploadEmissiveBuffers(ID3D12GraphicsCommandList* cmdList) {
    if (!m_emissiveGpuDirty)
        return;
    m_emissiveGpuDirty = false;

    auto* dev = m_ctx.Device();
    const UINT inc = dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    CD3DX12_RANGE readRange(0, 0);

    const auto& tris = m_scene.emissiveTriangles;
    const auto& triMap = m_scene.triToLightId;

    m_lightTreeRefit.DiscardPending();
    m_pendingTLASUpload.clear();
    m_pendingPackedTLAS.clear();
    m_pendingBLASBitTrail.clear();
    m_pendingSlotRecords.clear();
    m_lightLearningResetPending = true;
    m_lightTlasForceRebuild = true;
    m_scene.lightTreeDirty = true;
    {
        m_lightTree.ReleaseStaging();
        lt::LightTreeBuilder::Settings settings;
        settings.compactGpuNodes = m_lightTreeCompact;
        m_lightTree.Build(tris, m_scene.lightInstances, BuildXformsFromScene(), settings);
        m_publishedLightTLAS = m_lightTree.GetCpuTLASNodes();
        m_frameStats.lightBvh.slots = m_lightTree.SlotCount();
        m_lightTree.UploadAll(dev, cmdList);
        CD3DX12_CPU_DESCRIPTOR_HANDLE treeHandle(m_srvUavHeap->GetCPUDescriptorHandleForHeapStart(), LT_TLAS_SRV_SLOT,
                                                 inc);
        m_lightTree.WriteSrvs(dev, treeHandle);
        CD3DX12_CPU_DESCRIPTOR_HANDLE lookupHandle(m_srvUavHeap->GetCPUDescriptorHandleForHeapStart(), 25, inc);
        m_lightTree.WriteLookupSrvs(dev, lookupHandle);
        CD3DX12_CPU_DESCRIPTOR_HANDLE slotHandle(m_srvUavHeap->GetCPUDescriptorHandleForHeapStart(), LT_SLOT_SRV_SLOT,
                                                 inc);
        m_lightTree.WriteSlotSrv(dev, slotHandle);

        m_ltTlasGpu.Reset();
        m_ltBlasBitTrailGpu.Reset();
        m_ltSlotGpu.Reset();
        m_ltTlasGpuCapacity = m_ltBlasBitTrailGpuCapacity = m_ltSlotGpuCapacity = 0;
    }

    if (!tris.empty()) {
        const UINT count = (UINT)tris.size();
        const UINT64 bytes = (UINT64)count * sizeof(LightTriangle);

        if (count > m_emissiveGpuCapacity) {
            m_emissiveGpuCapacity = count * 2;
            const UINT64 allocBytes = (UINT64)m_emissiveGpuCapacity * sizeof(LightTriangle);

            m_ownedEmissiveGpu =
                nv_helpers_dx12::CreateBuffer(dev, allocBytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST,
                                              nv_helpers_dx12::kDefaultHeapProps);
            m_ownedEmissiveGpu->SetName(L"EmissiveTris_Refit");

            CD3DX12_CPU_DESCRIPTOR_HANDLE srvH(m_srvUavHeap->GetCPUDescriptorHandleForHeapStart(),
                                               EMISSIVE_TRI_SRV_SLOT, inc);
            D3D12_SHADER_RESOURCE_VIEW_DESC sd{};
            sd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            sd.Format = DXGI_FORMAT_UNKNOWN;
            sd.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
            sd.Buffer.NumElements = m_emissiveGpuCapacity;
            sd.Buffer.StructureByteStride = sizeof(LightTriangle);
            dev->CreateShaderResourceView(m_ownedEmissiveGpu.Get(), &sd, srvH);

            LOG(L"[Emissive] Grew GPU buffer: " << m_emissiveGpuCapacity << L" tris");
        } else if (m_ownedEmissiveGpu) {
            auto b = CD3DX12_RESOURCE_BARRIER::Transition(m_ownedEmissiveGpu.Get(), D3D12_RESOURCE_STATE_GENERIC_READ,
                                                          D3D12_RESOURCE_STATE_COPY_DEST);
            cmdList->ResourceBarrier(1, &b);
        } else {
            auto b =
                CD3DX12_RESOURCE_BARRIER::Transition(m_scene.emissiveTrianglesBuffer.Get(),
                                                     D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_STATE_COPY_DEST);
            cmdList->ResourceBarrier(1, &b);
        }

        if (!m_emissiveUploadStaging || bytes > m_emissiveUploadStaging->GetDesc().Width) {
            auto hp = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
            auto desc = CD3DX12_RESOURCE_DESC::Buffer((UINT64)m_emissiveGpuCapacity * sizeof(LightTriangle));
            ThrowIfFailed(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &desc,
                                                       D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                       IID_PPV_ARGS(&m_emissiveUploadStaging)));
        }

        {
            void* p = nullptr;
            ThrowIfFailed(m_emissiveUploadStaging->Map(0, &readRange, &p));
            memcpy(p, tris.data(), bytes);
            m_emissiveUploadStaging->Unmap(0, nullptr);
        }

        auto* dst = m_ownedEmissiveGpu ? m_ownedEmissiveGpu.Get() : m_scene.emissiveTrianglesBuffer.Get();
        cmdList->CopyBufferRegion(dst, 0, m_emissiveUploadStaging.Get(), 0, bytes);

        auto b = CD3DX12_RESOURCE_BARRIER::Transition(dst, D3D12_RESOURCE_STATE_COPY_DEST,
                                                      D3D12_RESOURCE_STATE_GENERIC_READ);
        cmdList->ResourceBarrier(1, &b);
    }

    if (!triMap.empty()) {
        const UINT count = (UINT)triMap.size();
        const UINT64 bytes = (UINT64)count * sizeof(uint32_t);

        if (count > m_triToLightIdGpuCapacity) {
            m_triToLightIdGpuCapacity = count * 2;
            const UINT64 allocBytes = (UINT64)m_triToLightIdGpuCapacity * sizeof(uint32_t);

            m_ownedTriToLightIdGpu =
                nv_helpers_dx12::CreateBuffer(dev, allocBytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST,
                                              nv_helpers_dx12::kDefaultHeapProps);
            m_ownedTriToLightIdGpu->SetName(L"TriToLightId_Refit");

            CD3DX12_CPU_DESCRIPTOR_HANDLE srvH(m_srvUavHeap->GetCPUDescriptorHandleForHeapStart(),
                                               TRI_TO_LIGHTID_SRV_SLOT, inc);
            D3D12_SHADER_RESOURCE_VIEW_DESC sd{};
            sd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            sd.Format = DXGI_FORMAT_R32_UINT;
            sd.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
            sd.Buffer.NumElements = m_triToLightIdGpuCapacity;
            dev->CreateShaderResourceView(m_ownedTriToLightIdGpu.Get(), &sd, srvH);

            LOG(L"[Emissive] Grew TriToLightId: " << m_triToLightIdGpuCapacity);
        } else if (m_ownedTriToLightIdGpu) {
            auto b = CD3DX12_RESOURCE_BARRIER::Transition(
                m_ownedTriToLightIdGpu.Get(), D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_STATE_COPY_DEST);
            cmdList->ResourceBarrier(1, &b);
        } else {
            auto b = CD3DX12_RESOURCE_BARRIER::Transition(
                m_scene.triToLightIdBuffer.Get(), D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_STATE_COPY_DEST);
            cmdList->ResourceBarrier(1, &b);
        }

        if (!m_triToLightIdUploadStaging || bytes > m_triToLightIdUploadStaging->GetDesc().Width) {
            auto hp = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
            auto desc = CD3DX12_RESOURCE_DESC::Buffer((UINT64)m_triToLightIdGpuCapacity * sizeof(uint32_t));
            ThrowIfFailed(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &desc,
                                                       D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                       IID_PPV_ARGS(&m_triToLightIdUploadStaging)));
        }

        {
            void* p = nullptr;
            ThrowIfFailed(m_triToLightIdUploadStaging->Map(0, &readRange, &p));
            memcpy(p, triMap.data(), bytes);
            m_triToLightIdUploadStaging->Unmap(0, nullptr);
        }

        auto* dst = m_ownedTriToLightIdGpu ? m_ownedTriToLightIdGpu.Get() : m_scene.triToLightIdBuffer.Get();
        cmdList->CopyBufferRegion(dst, 0, m_triToLightIdUploadStaging.Get(), 0, bytes);

        auto b = CD3DX12_RESOURCE_BARRIER::Transition(dst, D3D12_RESOURCE_STATE_COPY_DEST,
                                                      D3D12_RESOURCE_STATE_GENERIC_READ);
        cmdList->ResourceBarrier(1, &b);
    }

    LOG(L"[Emissive] GPU buffers updated: " << tris.size() << L" tris, " << triMap.size() << L" triToLightId");

    if (m_voxels.enabled()) {
        m_voxelLightKickedVersion = 0xFFFFFFFFu;
        BindVoxelLights(cmdList);
    }
}

void Renderer::RebuildDLSSDescriptors() {
    auto* dev = m_ctx.Device();
    const UINT inc = dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    CD3DX12_CPU_DESCRIPTOR_HANDLE handle(m_srvUavHeap->GetCPUDescriptorHandleForHeapStart(), DLSS_UAV_HEAP_START, inc);

    auto dlssUAV = [&](ID3D12Resource* res, DXGI_FORMAT fmt) {
        D3D12_UNORDERED_ACCESS_VIEW_DESC ud = {};
        ud.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        ud.Format = fmt;
        dev->CreateUnorderedAccessView(res, nullptr, &ud, handle);
        handle.ptr += inc;
    };

    // Must match order in CreateShaderResourceHeap (slots 39-51)
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
}

void Renderer::OnResize(UINT newWidth, UINT newHeight) {
    if (newWidth == 0 || newHeight == 0)
        return;
    if (newWidth == m_width && newHeight == m_height)
        return;

    // Disable DLSS-G before resize to avoid deadlock with present hook
    if (m_dlssG.enabled) {
        sl::DLSSGOptions gOpts{};
        gOpts.mode = sl::DLSSGMode::eOff;
        slDLSSGSetOptions(m_ctx.viewportHandle, gOpts);
    }

    m_ctx.WaitForGPU();

    m_width = newWidth;
    m_height = newHeight;
    m_aspectRatio = static_cast<float>(newWidth) / static_cast<float>(newHeight);

    m_ctx.Resize(newWidth, newHeight);

    m_outputResource.Reset();
    m_permanentDataTexture.Reset();
    m_scratchPing.Reset();
    m_liteReservoirs.Reset();
    m_sampleBuffer_current.Reset();
    m_sampleBuffer_last.Reset();
    m_pathStateBuffer.Reset();
    m_skyBakeBuffer.Reset();

    CreateRaytracingOutputBuffer();

    m_dlss.CreateResources(m_ctx.Device(), newWidth, newHeight);

    m_dlssNR.OnDisplayResolution(newWidth, newHeight);

    RebuildResolutionDependentDescriptors();
    RebuildDLSSDescriptors();

    m_dlssModeChangedFrames = 2;
    m_camera.ResetJitter();

    // Re-enable DLSS-G after resize
    if (m_dlssG.enabled) {
        sl::DLSSGOptions gOpts{};
        gOpts.mode = sl::DLSSGMode::eOn;
        gOpts.numFramesToGenerate = m_dlssG.framesToGenerate;
        gOpts.numBackBuffers = m_ctx.BufferCount();
        gOpts.mvecDepthWidth = m_dlss.RenderWidth();
        gOpts.mvecDepthHeight = m_dlss.RenderHeight();
        gOpts.colorWidth = newWidth;
        gOpts.colorHeight = newHeight;
        slDLSSGSetOptions(m_ctx.viewportHandle, gOpts);
    }

    LOG(L"[Resize] " << newWidth << L"x" << newHeight);
}

void Renderer::RebuildResolutionDependentDescriptors() {
    auto* dev = m_ctx.Device();
    const UINT inc = dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    auto writeUAVAt = [&](UINT slot, auto& res, auto writeFn) {
        CD3DX12_CPU_DESCRIPTOR_HANDLE h(m_srvUavHeap->GetCPUDescriptorHandleForHeapStart(), slot, inc);
        writeFn(h);
    };

    UINT px = TileAlignedPx(GetWidth(), GetHeight());

    writeUAVAt(0, m_outputResource, [&](D3D12_CPU_DESCRIPTOR_HANDLE h) {
        D3D12_UNORDERED_ACCESS_VIEW_DESC ud = {};
        ud.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
        ud.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        ud.Texture2DArray.ArraySize = m_outputResource->GetDesc().DepthOrArraySize;
        dev->CreateUnorderedAccessView(m_outputResource.Get(), nullptr, &ud, h);
    });

    writeUAVAt(1, m_permanentDataTexture, [&](D3D12_CPU_DESCRIPTOR_HANDLE h) {
        D3D12_UNORDERED_ACCESS_VIEW_DESC ud = {};
        ud.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        ud.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
        dev->CreateUnorderedAccessView(m_permanentDataTexture.Get(), nullptr, &ud, h);
    });

    auto rawUAVAt = [&](UINT slot, ComPtr<ID3D12Resource>& res, UINT bytes) {
        CD3DX12_CPU_DESCRIPTOR_HANDLE h(m_srvUavHeap->GetCPUDescriptorHandleForHeapStart(), slot, inc);
        D3D12_UNORDERED_ACCESS_VIEW_DESC ud = {};
        ud.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        ud.Format = DXGI_FORMAT_R32_TYPELESS;
        ud.Buffer.NumElements = bytes / 4;
        ud.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
        dev->CreateUnorderedAccessView(res.Get(), nullptr, &ud, h);
    };

    {
        D3D12_UNORDERED_ACCESS_VIEW_DESC ud = {};
        ud.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        ud.Format = DXGI_FORMAT_R32_TYPELESS;
        ud.Buffer.NumElements = 1;
        ud.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
        for (UINT slot : {10u, 11u}) {
            CD3DX12_CPU_DESCRIPTOR_HANDLE h(m_srvUavHeap->GetCPUDescriptorHandleForHeapStart(), slot, inc);
            dev->CreateUnorderedAccessView(nullptr, nullptr, &ud, h);
        }
    }
    rawUAVAt(12, m_liteReservoirs, px * LITE_RESERVOIR_BYTES);
    rawUAVAt(14, m_sampleBuffer_current, px * sizeof(SampleData));
    rawUAVAt(15, m_sampleBuffer_last, px * sizeof(SampleData));

    writeUAVAt(18, m_scratchPing, [&](D3D12_CPU_DESCRIPTOR_HANDLE h) {
        D3D12_UNORDERED_ACCESS_VIEW_DESC ud = {};
        ud.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
        ud.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
        ud.Texture2DArray.ArraySize = m_scratchPing->GetDesc().DepthOrArraySize;
        dev->CreateUnorderedAccessView(m_scratchPing.Get(), nullptr, &ud, h);
    });

    {
        D3D12_UNORDERED_ACCESS_VIEW_DESC ud = {};
        ud.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        ud.Format = DXGI_FORMAT_R32_TYPELESS;
        ud.Buffer.NumElements = 1;
        ud.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
        CD3DX12_CPU_DESCRIPTOR_HANDLE h(m_srvUavHeap->GetCPUDescriptorHandleForHeapStart(), 19, inc);
        dev->CreateUnorderedAccessView(nullptr, nullptr, &ud, h);
    }

    rawUAVAt(32, m_pathStateBuffer, px * kPathStateBytesPerPx);

}

// Swap reservoir history and repoint its fixed descriptor slots.
void Renderer::SwapSampleBuffers() {
    m_sampleBuffer_current.Swap(m_sampleBuffer_last);

    auto* dev = m_ctx.Device();
    const UINT inc = dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    const UINT bytes = TileAlignedPx(GetWidth(), GetHeight()) * sizeof(SampleData);

    auto rawUAVAt = [&](UINT slot, ID3D12Resource* res) {
        CD3DX12_CPU_DESCRIPTOR_HANDLE h(m_srvUavHeap->GetCPUDescriptorHandleForHeapStart(), slot, inc);
        D3D12_UNORDERED_ACCESS_VIEW_DESC ud = {};
        ud.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        ud.Format = DXGI_FORMAT_R32_TYPELESS;
        ud.Buffer.NumElements = bytes / 4;
        ud.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
        dev->CreateUnorderedAccessView(res, nullptr, &ud, h);
    };
    rawUAVAt(14, m_sampleBuffer_current.Get());
    rawUAVAt(15, m_sampleBuffer_last.Get());
}

// Recover world coordinates from the camera-relative rendering transform.
planet::CameraView Renderer::MakePlanetCamera() const {
    const XMMATRIX view = m_camera.ViewMatrix();
    const XMMATRIX invView = XMMatrixInverse(nullptr, view);
    XMFLOAT3 pos, fwd, up;
    XMStoreFloat3(&pos, invView.r[3]);

    XMStoreFloat3(&fwd, XMVector3Normalize(XMVectorNegate(invView.r[2])));
    XMStoreFloat3(&up, XMVector3Normalize(invView.r[1]));
    const glm::vec3 origin = m_camera.getSceneOriginWorld();

    planet::CameraView cv;
    cv.position_world = {(double)pos.x + origin.x, (double)pos.y + origin.y, (double)pos.z + origin.z};

    cv.scene_origin = {origin.x, origin.y, origin.z};
    cv.forward = {fwd.x, fwd.y, fwd.z};
    cv.up = {up.x, up.y, up.z};
    cv.fov_y = m_camera.fovDegrees * 0.01745329252f;
    cv.aspect = m_aspectRatio;
    cv.near_plane = m_camera.nearPlane;
    cv.far_plane = m_camera.farPlane;
    return cv;
}

// Mirror changed scene instances into the streamer's unified TLAS input.
void Renderer::BuildPlanetSceneInstances() {
    const auto& src = m_scene.tlasInstances;
    const bool all = m_scene.tlasFullRebuild || m_planetSceneInstances.size() != src.size();
    m_planetSceneInstances.resize(src.size());
    auto update = [&](size_t i) {
        planet::SceneInstanceDesc& d = m_planetSceneInstances[i];
        d.blas = src[i].blas ? src[i].blas->GetGPUVirtualAddress() : 0;

        XMFLOAT4X4 t;
        XMStoreFloat4x4(&t, XMMatrixTranspose(src[i].transform));
        memcpy(d.transform, &t, sizeof(float) * 12);
        d.instance_id = (uint32_t)i;
        d.hit_group_index = src[i].hitGroupContribution;
        d.flags = (uint32_t)src[i].flags;
    };
    if (all) {
        for (size_t i = 0; i < src.size(); ++i)
            update(i);
    } else {
        for (uint32_t i : m_scene.dirtyInstanceList)
            if (i < src.size())
                update(i);
    }
}

// Submit streamed geometry before the passes that trace against it.
void Renderer::RenderFrame() {
    using hrc = std::chrono::high_resolution_clock;
    static auto s_lastTime = hrc::now();
    static int s_frameCount = 0;
    auto t_frameStart = hrc::now();

    m_ctx.BeginFrame();

    m_planet.begin_frame(m_planetFrame++, MakePlanetCamera());

    // The ocean picks its tiles from the same camera the terrain streamer uses, one frame ahead of
    // the work the orchestrator records for it.
    if (m_ocean.Enabled())
        m_ocean.BeginFrame(m_lastDt, MakePlanetCamera(), m_planetFrame);

    slPCLSetMarker(sl::PCLMarker::eRenderSubmitStart, *m_ctx.frameToken);

    try {
        if (m_scene.tlasFullRebuild || m_scene.tlasInstances.size() != m_scene.instances.size())
            m_scene.RebuildTLASInstanceList();
        BuildPlanetSceneInstances();

        if (m_planet.enabled() && !m_rockMeshIndices.empty()) {
            const planet::CameraView pcam = MakePlanetCamera();
            struct RockHeightAdapter : planet::IRockHeight {
                const planet::HeightmapCubemap* hm = nullptr;
                float sample_height_m(const planet::DVec3& d) const override { return hm ? hm->sample(d, 0) : 0.0f; }
            } adapter;
            adapter.hm = &m_planet.heightmap();
            m_rockScatter.update(pcam.position_world, adapter);
            m_planet.set_rock_instances(m_rockScatter.live().data(), (uint32_t)m_rockScatter.live().size());
        }

        if (m_voxels.enabled()) {
            if (m_emissiveGpuDirty) {
                m_voxels.on_light_tlas_published(0u, false);
                m_liveVoxelLightLeaves = 0;
            } else if (!m_pendingTLASUpload.empty()) {
                m_voxels.on_light_tlas_published(m_pendingVoxelLightVersion, m_pendingVoxelLeafCount > 0);
                m_liveVoxelLightLeaves = m_pendingVoxelLeafCount;
            }
            const planet::CameraView pcam = MakePlanetCamera();
            const double camPos[3] = {pcam.position_world.x, pcam.position_world.y, pcam.position_world.z};
            m_voxels.begin_frame(camPos);
        }
        const uint32_t terrainHitGroup = (uint32_t)m_scene.instances.size() * 2u;
        const uint32_t voxelHitGroup = terrainHitGroup + 1u; // [opaque, alpha] pair after the terrain entry
        const auto previousTlasAddress = m_planet.tlas_address();
        m_planet.submit_work(m_planetSceneInstances.data(), (uint32_t)m_planetSceneInstances.size(), terrainHitGroup,
                             voxelHitGroup);
        // TLAS growth can replace the allocation behind this descriptor.
        if (previousTlasAddress != m_planet.tlas_address()) {
            D3D12_SHADER_RESOURCE_VIEW_DESC sd{};
            sd.ViewDimension = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE;
            sd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            sd.RaytracingAccelerationStructure.Location = m_planet.tlas_address();
            const UINT inc = m_ctx.Device()->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            CD3DX12_CPU_DESCRIPTOR_HANDLE h(m_srvUavHeap->GetCPUDescriptorHandleForHeapStart(), 2, inc);
            m_ctx.Device()->CreateShaderResourceView(nullptr, &sd, h);
        }
        m_scene.tlasDirty = false;
        m_scene.tlasFullRebuild = false;

        auto t_popStart = hrc::now();
        m_frameStats.cpuStreamingMs = std::chrono::duration<float, std::milli>(t_popStart - t_frameStart).count();
        PopulateCommandList();
        auto t_popEnd = hrc::now();
        m_frameStats.cpuPopulateMs = std::chrono::duration<float, std::milli>(t_popEnd - t_popStart).count();

        slPCLSetMarker(sl::PCLMarker::eRenderSubmitEnd, *m_ctx.frameToken);

        slPCLSetMarker(sl::PCLMarker::ePresentStart, *m_ctx.frameToken);
        ID3D12Resource* presentedBuffer = m_ctx.BackBuffer();
        m_ctx.ExecuteAndPresent();

        // Opt-in image capture for reproducible, hidden-window rendering reviews.
        // Synchronous readback runs only at explicitly requested capture frames.
        static uint32_t captureFrame = 0;
        static const std::wstring capturePath = [] {
            wchar_t path[32768]{};
            const DWORD n = GetEnvironmentVariableW(L"RT_CAPTURE_PATH", path, _countof(path));
            return n && n < _countof(path) ? std::wstring(path) : std::wstring();
        }();
        if (!capturePath.empty()) {
            wchar_t frameText[32]{};
            GetEnvironmentVariableW(L"RT_CAPTURE_FRAME", frameText, _countof(frameText));
            const uint32_t requestedFrame = std::max(1ul, frameText[0] ? wcstoul(frameText, nullptr, 10) : 64ul);
            frameText[0] = 0;
            GetEnvironmentVariableW(L"RT_CAPTURE_EVERY", frameText, _countof(frameText));
            const uint32_t interval = wcstoul(frameText, nullptr, 10);
            ++captureFrame;
            if (captureFrame >= requestedFrame && (interval ? (captureFrame-requestedFrame)%interval == 0 : captureFrame == requestedFrame)) {
                const std::wstring outputPath = interval ? capturePath + L"-" + std::to_wstring(captureFrame) + L".png" : capturePath;
                DirectX::ScratchImage capture;
                ThrowIfFailed(DirectX::CaptureTexture(m_ctx.CmdQueue(), presentedBuffer, false, capture,
                    D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_PRESENT));
                ThrowIfFailed(DirectX::SaveToWICFile(*capture.GetImage(0, 0, 0), DirectX::WIC_FLAGS_NONE,
                    DirectX::GetWICCodec(DirectX::WIC_CODEC_PNG), outputPath.c_str()));
            }
        }

        slPCLSetMarker(sl::PCLMarker::ePresentEnd, *m_ctx.frameToken);
        m_editor.RenderPlatformWindows();
    } catch (const std::exception& e) {
        // Streamline hands out a proxy device; the removal state and the DRED data live on the
        // native one, which is polled first. The proxy is tried afterwards in case only it reports.
        dxdiag::CrashLogF(L"\n*** frame failed: %hs\n", e.what());
        dxdiag::CheckDeviceRemoved(m_ctx.NativeDevice(), 2000);
        dxdiag::CheckDeviceRemoved(m_ctx.Device(), 500);
        throw;
    } catch (...) {
        dxdiag::CrashLog(L"\n*** frame failed: non-standard exception\n");
        dxdiag::CheckDeviceRemoved(m_ctx.NativeDevice(), 2000);
        dxdiag::CheckDeviceRemoved(m_ctx.Device(), 500);
        throw;
    }

    m_planet.end_frame();

    m_frameStats.cpuFrameMs = m_frameStats.cpuUpdateMs + m_frameStats.cpuStreamingMs + m_frameStats.cpuPopulateMs;
    static PerformanceCapture capture;
    capture.record(m_frameStats, m_voxels.enabled() ? &m_voxels.stats() : nullptr);

    s_frameCount++;
    auto now = hrc::now();
    float elapsed = std::chrono::duration<float>(now - s_lastTime).count();
    if (elapsed >= 1.0f) {
        m_fps = s_frameCount / elapsed;
        std::wstringstream ss;
        ss << std::fixed << std::setprecision(2);
        if (m_dlssG.enabled && m_dlssG.framesToGenerate > 0) {
            float presentedFps = m_fps * (1 + m_dlssG.framesToGenerate);
            ss << L"Frame Time: " << 1000.0f / m_fps << L" ms (" << presentedFps << L" fps, " << m_fps
               << L" rendered + " << (1 + m_dlssG.framesToGenerate) << L"x FG)";
        } else {
            ss << L"Frame Time: " << 1000.0f / m_fps << L" ms (" << m_fps << L" fps)";
        }
        SetWindowTextW(Win32Application::GetHwnd(), ss.str().c_str());

        if (m_ocean.Enabled()) {
            const auto& os = m_ocean.GetStats();
            std::wcout << L"[ocean] tiles=" << os.tiles << L"/" << os.leaves << L" dropped=" << os.dropped
                       << L" tris=" << (os.triangles / 1000) << L"k blas[build=" << os.builds << L" refit="
                       << os.refits << L"] Hs=" << os.significantWaveHeight << L" m whitecap=" << (os.whitecapMeasured * 100.0) << L"%/" << (os.whitecapCoverage * 100.0) << L"% slopeVar=" << os.slopeVarSpectrum
                       << L"/" << os.slopeVarCoxMunk << L" chopGain=" << os.conditioningGain << std::endl;
        }

        const auto& ps = m_planet.stats();
        std::wcout << L"[planet] built=" << ps.built << L" leaves=" << ps.leaf_count << L" cells=" << ps.cell_count
                   << L" tris=" << ps.triangle_count << L" tlas=" << ps.tlas_instances << L" rebuilding="
                   << ps.rebuilding << L" dirty=" << ps.dirty_built << L"/" << ps.dirty_total << L" est="
                   << ps.rebuild_frames_est << L"f" << L" rec=" << ps.cells_recorded << L" pipe[p=" << ps.cells_pending
                   << L" r=" << ps.cells_ready << L" b=" << ps.cells_recorded_total << L" B=" << ps.dirty_built << L"]"
                   << L" cpu_ms[blas_rec=" << ps.blas_record_cpu_ms << L" plan=" << ps.plan_ms << L"]"
                   << L" gpu_ms[blas=" << ps.blas_gpu_ms << L" tlas=" << ps.tlas_gpu_ms << L"]" << L" DROPPED="
                   << ps.cells_dropped << L" geo_free=" << ps.geo_free_leaves << L" id_peak=" << ps.stable_id_peak
                   << std::endl;

        if (m_voxels.enabled()) {
            const auto& vs = m_voxels.stats();
            const auto& vc = m_voxels.config();
            std::printf(
                "[mc] shown=%u/%u ready=%u empty=%u pend=%u mesh=%u meshed=%u upl=%u tris=%.2fM (est %.1fM/%.1fM lod "
                "%.0f cut %.0f%% built)"
                " builds=%u up=%.1fMB job=%.2fms sel=%.2fms (adopt %.2f resolve %.2f adapt %.2f, %u nodes) rec=%.2fms "
                "vtx=%.1f/%uM idx=%.1f/%uM"
                " blas=%llu/%lluMB (+%lluMB building) fail=%u evict=%u tracked=%u lights=%.2fM/%uk in %u chunks (res "
                "%.2fM, dropped %u, v%u live%u%s)\n",
                vs.rendered, vs.desired, vs.ready, vs.empty, vs.pending, vs.meshing, vs.meshed, vs.uploading,
                (double)vs.trianglesRendered / 1.0e6, (double)vs.trianglesEstimated / 1.0e6,
                (double)vs.triangleBudget / 1.0e6, vs.lodFactorNow, vs.cutReadyFraction * 100.0f, vs.buildsThisFrame,
                (double)vs.uploadBytesThisFrame / 1048576.0, vs.meshMsAvg, vs.selectMs, vs.adoptMs, vs.resolveMs,
                vs.adaptMs, vs.cutNodes, vs.recordMs, (double)vs.vertexUsed / 1048576.0, vc.vertexCapacity >> 20,
                (double)vs.indexUsed / 1048576.0, vc.indexCapacity >> 20, (unsigned long long)(vs.blasUsed >> 20),
                (unsigned long long)(vc.blasPoolBytes >> 20), (unsigned long long)(vs.blasBuildUsed >> 20),
                vs.allocFailures, vs.evicted, vs.chunksTracked, (double)vs.lightTrisInTree / 1.0e6,
                vc.maxLightTris / 1000u, vs.lightChunksInTree, (double)vs.lightTrisResident / 1.0e6,
                vs.lightChunksDropped, vs.lightVersion, vs.lightLiveVersion, m_voxels.has_live_lights() ? "" : " none");
        }

        s_frameCount = 0;
        s_lastTime = now;
    }
}

void Renderer::DestroyRenderer() {
    m_ctx.WaitForGPU();
    if (m_dlssG.enabled) {
        sl::DLSSGOptions gOpts{};
        gOpts.mode = sl::DLSSGMode::eOff;
        slDLSSGSetOptions(m_ctx.viewportHandle, gOpts);
        slFreeResources(sl::kFeatureDLSS_G, m_ctx.viewportHandle);
        m_dlssG.enabled = false;
    }
    m_editor.Shutdown();
    // Release the independent NR runtime before Streamline/device shutdown.
    m_dlssNR.Shutdown(m_ctx.Device());
    m_dlssHudlessColor.Reset();
    m_ctx.Shutdown();
}

UINT Renderer::CreateProceduralMesh(const std::vector<Vertex>& vertices, const std::vector<UINT>& indices,
                                    const Material& material) {
    UINT matIdx = (UINT)m_scene.materials.size();
    UINT triCount = (UINT)indices.size() / 3;
    const UINT materialIDBase = (UINT)m_scene.materialIDs.size();
    m_scene.materials.push_back(material);
    for (UINT t = 0; t < triCount; ++t)
        m_scene.materialIDs.push_back(matIdx);

    MeshGPU mesh;
    mesh.cpuVertices = vertices;
    mesh.cpuIndices = indices;
    mesh.cpuMaterialIDs = std::vector<UINT>(triCount, matIdx);
    mesh.vertexCount = (UINT)vertices.size();
    mesh.indexCount = (UINT)indices.size();
    mesh.opaqueTriCount = triCount;
    mesh.alphaTriCount = 0;
    mesh.materialIDBase = materialIDBase;
    {
        UINT bytes = mesh.vertexCount * sizeof(Vertex);
        mesh.vertexBuffer =
            nv_helpers_dx12::CreateBuffer(m_ctx.Device(), bytes, D3D12_RESOURCE_FLAG_NONE,
                                          D3D12_RESOURCE_STATE_GENERIC_READ, nv_helpers_dx12::kUploadHeapProps);
        void* p = nullptr;
        mesh.vertexBuffer->Map(0, nullptr, &p);
        memcpy(p, vertices.data(), bytes);
        mesh.vertexBuffer->Unmap(0, nullptr);
    }
    {
        UINT bytes = mesh.indexCount * sizeof(UINT);
        mesh.indexBuffer =
            nv_helpers_dx12::CreateBuffer(m_ctx.Device(), bytes, D3D12_RESOURCE_FLAG_NONE,
                                          D3D12_RESOURCE_STATE_GENERIC_READ, nv_helpers_dx12::kUploadHeapProps);
        void* p = nullptr;
        mesh.indexBuffer->Map(0, nullptr, &p);
        memcpy(p, indices.data(), bytes);
        mesh.indexBuffer->Unmap(0, nullptr);
    }

    auto blasBuf = CreateBottomLevelAS({{mesh.vertexBuffer, mesh.vertexCount}}, {{mesh.indexBuffer, mesh.indexCount}},
                                       mesh.opaqueTriCount, mesh.alphaTriCount);
    mesh.blas = blasBuf.pResult;
    m_ctx.FlushAndReset();

    mesh.vertexBuffer.Reset();
    mesh.indexBuffer.Reset();

    UINT meshIndex = (UINT)m_scene.meshes.size();
    m_scene.meshes.push_back(std::move(mesh));
    LOG(L"[Engine] Created procedural mesh " << meshIndex << L" (mat " << matIdx << L")");
    return meshIndex;
}

UINT Renderer::CreateMeshInstance(UINT sourceMeshIndex, const Material& material) {
    const auto& src = m_scene.meshes[sourceMeshIndex];
    UINT matIdx = (UINT)m_scene.materials.size();
    UINT triCount = src.indexCount / 3;
    const UINT materialIDBase = (UINT)m_scene.materialIDs.size();

    m_scene.materials.push_back(material);
    for (UINT t = 0; t < triCount; ++t)
        m_scene.materialIDs.push_back(matIdx);

    MeshGPU mesh;
    mesh.cpuVertices = src.cpuVertices;
    mesh.cpuIndices = src.cpuIndices;
    mesh.cpuMaterialIDs = std::vector<UINT>(triCount, matIdx);
    mesh.vertexCount = src.vertexCount;
    mesh.indexCount = src.indexCount;
    mesh.opaqueTriCount = src.opaqueTriCount;
    mesh.alphaTriCount = src.alphaTriCount;
    mesh.materialIDBase = materialIDBase;

    mesh.vertexBuffer = src.vertexBuffer;
    mesh.indexBuffer = src.indexBuffer;
    mesh.blas = src.blas;

    UINT meshIndex = (UINT)m_scene.meshes.size();
    m_scene.meshes.push_back(std::move(mesh));
    return meshIndex;
}

// Rebuild index-dependent bindings after instances are added or removed.
void Renderer::HandleSceneStructuralChange() {
    if (m_scene.sceneInstanceCap() && m_scene.instances.size() > m_scene.sceneInstanceCap())
        throw std::runtime_error("Scene instances exceed the reserved streamed instance range");

    m_ctx.WaitForGPU();
    m_scene.RebuildTLASInstanceList();
    m_scene.CreateInstancePropertiesBuffer(m_ctx.Device());

    m_scene.instanceDirty.assign(m_scene.instances.size(), 1);
    m_scene.instanceInitialized.assign(m_scene.instances.size(), 0);
    m_scene.cpuInstanceProps.clear();

    {
        const UINT inc = m_ctx.Device()->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        CD3DX12_CPU_DESCRIPTOR_HANDLE h(m_srvUavHeap->GetCPUDescriptorHandleForHeapStart(), INSTANCE_PROPS_SRV_SLOT,
                                        inc);
        D3D12_SHADER_RESOURCE_VIEW_DESC sd = {};
        sd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        sd.Format = DXGI_FORMAT_UNKNOWN;
        sd.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;

        sd.Buffer.NumElements = m_scene.instancePropsCount();
        sd.Buffer.StructureByteStride = sizeof(InstanceProperties);
        m_ctx.Device()->CreateShaderResourceView(m_scene.instanceProperties.Get(), &sd, h);
    }

    CreateShaderBindingTable();
    m_scene.CollectEmissiveTriangles();
    m_blasLocalRoots = lt::ComputeBLASLocalRoots(m_scene.emissiveTriangles);
    m_emissiveGpuDirty = true;
    m_scene.tlasDirty = true;
    m_scene.tlasFullRebuild = true;
    m_sharcResetPending = true;
    m_scene.lightTreeDirty = true;
}

bool Renderer::WantsKeyboard() const {
    return m_editor.IsVisible() && ImGui::GetIO().WantCaptureKeyboard;
}
bool Renderer::WantsMouse() const {
    return m_editor.IsVisible() && ImGui::GetIO().WantCaptureMouse;
}
void Renderer::HandleKeyUp(UINT8 key) {
    if (key == 'C')
        m_currentDisplayLevel = (m_currentDisplayLevel + 1) % m_displayLevels.size();
    if (key == 'K')
        m_recorder.CaptureKeyframe(m_camera.Manipulator());
    if (key == VK_F1)
        m_editor.ToggleVisibility();
}

// Execute the configured pass graph with shared queues and resource barriers.
void Renderer::PopulateCommandList() {
    auto* cmdList = m_ctx.CmdList();
    m_gpuProfiler.BeginFrame(cmdList);

    bool dlssResChanged = (m_dlssModeChangedFrames > 0);
    if (m_dlssModeChangedFrames > 0) {
        if (m_dlssModeChangedFrames == 2) { // Evaluate recreates resources after setting the updated options.
            sl::Result fr = slFreeResources(sl::kFeatureDLSS_RR, m_ctx.viewportHandle);
            if (fr != sl::Result::eOk)
                std::wcout << L"[SL] slFreeResources failed: " << (int)fr << std::endl;
        }
        m_dlssModeChangedFrames--;
    }

    {
        auto b = CD3DX12_RESOURCE_BARRIER::Transition(m_ctx.BackBuffer(), D3D12_RESOURCE_STATE_PRESENT,
                                                      D3D12_RESOURCE_STATE_RENDER_TARGET);
        cmdList->ResourceBarrier(1, &b);
    }

    auto rtv = m_ctx.CurrentRTV();
    auto dsv = m_ctx.DSV();
    cmdList->OMSetRenderTargets(1, &rtv, FALSE, &dsv);

    RecordSkyLUTBake(cmdList);

    ID3D12DescriptorHeap* heaps[] = {m_srvUavHeap.Get(), m_samplerHeap.Get()};
    cmdList->SetDescriptorHeaps(2, heaps);

    const UINT renderW = m_dlss.RenderWidth();
    const UINT renderH = m_dlss.RenderHeight();

    D3D12_DISPATCH_RAYS_DESC raysDesc{};
    raysDesc.Width = renderW;
    raysDesc.Height = renderH;
    raysDesc.Depth = 1;
    const uint64_t sbtStart = m_sbtStorage->GetGPUVirtualAddress();
    const uint32_t rgSize = m_sbtHelper.GetRayGenEntrySize();

    raysDesc.MissShaderTable.StartAddress = sbtStart + m_sbtHelper.GetRayGenSectionSize();
    raysDesc.MissShaderTable.SizeInBytes = m_sbtHelper.GetMissSectionSize();
    raysDesc.MissShaderTable.StrideInBytes = m_sbtHelper.GetMissEntrySize();

    raysDesc.HitGroupTable.StartAddress = raysDesc.MissShaderTable.StartAddress + raysDesc.MissShaderTable.SizeInBytes;
    raysDesc.HitGroupTable.SizeInBytes = m_sbtHelper.GetHitGroupSectionSize();
    raysDesc.HitGroupTable.StrideInBytes = m_sbtHelper.GetHitGroupEntrySize();


    if (m_emissiveGpuDirty) {
        const UINT timer = m_gpuProfiler.BeginPass(cmdList, "Emissive buffers");
        UploadEmissiveBuffers(cmdList);
        m_gpuProfiler.EndPass(cmdList, timer);
    }
    if (!m_pendingTLASUpload.empty()) {
        const UINT timer = m_gpuProfiler.BeginPass(cmdList, "Light BVH upload");
        UploadLightTreeTLAS(cmdList);
        m_gpuProfiler.EndPass(cmdList, timer);
    }

    {
        auto b = CD3DX12_RESOURCE_BARRIER::Transition(m_outputResource.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
                                                      D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        cmdList->ResourceBarrier(1, &b);
    }

    if (m_integratorSettings.liteReuseSigma != m_liteReuseSigma) {
        m_liteReuseSigma = m_integratorSettings.liteReuseSigma;
        BuildLiteReuseTables(m_liteReuseSigma);
    }
    if (m_liteReusePending && m_liteReuseUpload && m_sharcBuffer) {
        auto b = CD3DX12_RESOURCE_BARRIER::Transition(m_sharcBuffer.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                                      D3D12_RESOURCE_STATE_COPY_DEST);
        cmdList->ResourceBarrier(1, &b);
        cmdList->CopyBufferRegion(m_sharcBuffer.Get(), LITE_REUSE_OFFSET, m_liteReuseUpload.Get(), 0,
                                  (UINT64)LITE_REUSE_TEXELS * 4u);
        auto b2 = CD3DX12_RESOURCE_BARRIER::Transition(m_sharcBuffer.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                                                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        cmdList->ResourceBarrier(1, &b2);
        m_liteReusePending = false;
    }


    struct LoopFrame {
        uint32_t remaining;
        const std::wstring* tag;
    };
    std::vector<LoopFrame> loopStack;
    uint32_t ptSampleIndex = 0; // packed into pt_initialSamples for the sample loop

    UINT dispW = renderW, dispH = renderH;

    auto& rs = m_integratorSettings;

    // Slots and bit fields must match the shared shader root constants (Globals_v8.hlsli).
    UINT rsConsts[SHARC_ROOT_CONSTANTS] = {};
    auto setFloat = [&](UINT slot, float value) { memcpy(&rsConsts[slot], &value, 4); };

    uint32_t baseFlags = 0;
    if (m_lightTreeCompact)
        baseFlags |= RS_FLAG_COMPACT_LIGHT_TREE;
    if (!MeshLightsActive())
        baseFlags |= RS_FLAG_NO_MESH_LIGHTS;
    if (m_dlss.clampEmitterSpikes)
        baseFlags |= RS_FLAG_CLAMP_EMITTERS;
    if (rs.forceDiffuseMats)
        baseFlags |= RS_FLAG_FORCE_DIFFUSE;
    baseFlags |= m_dlss.GuideOffFlags();

    const bool useSharc = rs.sharcEnabled;
    const bool useLightLearning = rs.lightTreeLearning && MeshLightsActive();
    rs.lightTreeCellExponent = std::clamp(rs.lightTreeCellExponent, -4, 8);
    rs.lightTreeLodScale = std::clamp(rs.lightTreeLodScale, 0.001f, 0.25f);
    if (rs.lightTreeReset || (useLightLearning && !m_lightLearningWasEnabled) ||
        rs.lightTreeCellExponent != m_lightLearningCellExponent || rs.lightTreeLodScale != m_lightLearningLodScale ||
        rs.texturePointFilter != m_sharcTextureFilter ||
        rs.forceDiffuseMats != m_previousIntegratorSettings.forceDiffuseMats)
        m_lightLearningResetPending = true;
    m_lightLearningWasEnabled = useLightLearning;
    m_lightLearningCellExponent = rs.lightTreeCellExponent;
    rs.lightTreeReset = false;
    m_lightLearningLodScale = rs.lightTreeLodScale;
    if (useLightLearning)
        baseFlags |= LT_FLAG_LEARNING;
    if (useLightLearning && rs.lightTreeDebug)
        baseFlags |= LT_FLAG_DEBUG;
    const UINT sharcDebugMode = useSharc ? (UINT)std::clamp(rs.sharcDebugMode, 0, 3) : 0u;
    if (!m_sharcLightingValid || std::memcmp(&m_sharcSunSettings, &m_camera.sunSettings, sizeof(SunSettings)) != 0) {
        m_sharcResetPending = true;
        m_sharcSunSettings = m_camera.sunSettings;
        m_sharcLightingValid = true;
    }
    rs.sharcCellSizeExponent = std::clamp(rs.sharcCellSizeExponent, -6, 4);
    rs.sharcTrainBounces = std::clamp(rs.sharcTrainBounces, 4, 64);
    rs.sharcGuideLevelOffset = std::clamp(rs.sharcGuideLevelOffset, 0, 7);
    if (rs.sharcCellSizeExponent != m_sharcCellExponent || rs.sharcTrainBounces != m_sharcBounceLimit ||
        rs.sharcGuideLevelOffset != m_sharcGuideLevel || rs.texturePointFilter != m_sharcTextureFilter ||
        rs.sharcReset || (useSharc && !m_sharcWasEnabled))
        m_sharcResetPending = true;
    m_sharcCellExponent = rs.sharcCellSizeExponent;
    m_sharcGuideLevel = rs.sharcGuideLevelOffset;
    m_sharcBounceLimit = rs.sharcTrainBounces;
    m_sharcTextureFilter = rs.texturePointFilter;
    m_sharcWasEnabled = useSharc;
    rs.sharcReset = false;

    const bool liteActive = rs.liteEnabled;
    if (liteActive) {
        baseFlags |= LITE_FLAG_ENABLED | (rs.liteSpatial ? LITE_FLAG_SPATIAL : 0u) |
                     (rs.liteUnshadowedTargets ? LITE_FLAG_UNSHADOWED : 0u) | (rs.liteDebugView ? LITE_FLAG_DEBUG : 0u);
    }

    rsConsts[2] = baseFlags;
    rsConsts[3] = (UINT)std::clamp(rs.maxBounces, 2, 32);
    rsConsts[4] = (UINT)std::clamp(rs.rrStartDepth, 1, 32);
    rsConsts[5] = (UINT)std::clamp(rs.maxDiffuseBounces, 1, std::clamp(rs.maxBounces, 2, 32));
    rsConsts[7] = rs.texturePointFilter ? 1u : 0u;
    if (liteActive) {
        rsConsts[8] = (UINT)std::clamp(rs.liteSpatMcap, 1, 255);
        rsConsts[9] = (UINT)std::clamp(rs.liteSpatSlots, 0, (int)LITE_SLOTS_MAX);
        const UINT sizes[3] = {LITE_REUSE_SIZE0, LITE_REUSE_SIZE1, LITE_REUSE_SIZE2};
        for (int t = 0; t < 3; ++t) {
            const UINT ox = m_liteRng() % sizes[t];
            const UINT oy = m_liteRng() % sizes[t];
            const UINT flags = m_liteRng() & 7u;
            rsConsts[10 + t] = ox | (oy << 8) | (flags << LITE_REUSE_FLAGS_SHIFT);
        }
    }
    setFloat(13, std::clamp(rs.liteNormalSimCos, -1.0f, 1.0f));
    setFloat(14, std::max(rs.litePlaneDist, 0.0f));
    setFloat(15, std::exp2(float(rs.lightTreeCellExponent)));
    setFloat(16, rs.lightTreeLodScale);
    rsConsts[17] = static_cast<UINT>(
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch())
            .count());
    setFloat(18, std::clamp(rs.lightTreeLearnRoughness, 0.0f, 0.8f));
    rsConsts[19] = (UINT)std::clamp(rs.dlssDebugLayer, 0, 13) | (m_dlss.guideOffPsr ? DLSS_GUIDE_OPT_NO_PSR : 0u) |
                   (m_dlss.guideOffMvBlend ? DLSS_GUIDE_OPT_NO_MV_BLEND : 0u);
    {
        using DirectX::PackedVector::XMConvertFloatToHalf;
        const float dn = std::clamp(rs.dlssDebugDepthNear, 0.0f, 60000.0f);
        const float df = std::clamp(rs.dlssDebugDepthFar, dn + 0.01f, 65000.0f);
        rsConsts[20] = (UINT)XMConvertFloatToHalf(dn) | ((UINT)XMConvertFloatToHalf(df) << 16);
    }
    setFloat(21, std::clamp(m_dlss.sharpness, 0.0f, 1.0f));

    rsConsts[22] = useSharc ? 1u : 0u;
    rsConsts[22] |= sharcDebugMode << SHARC_DEBUG_MODE_SHIFT;
    if (sharcDebugMode != 0u && rs.sharcDebugCoarse)
        rsConsts[22] |= SHARC_DEBUG_OTHER_LEVEL_BIT;
    rsConsts[23] = (m_sharcResetPending ? 1u : 0u) | (m_lightLearningResetPending ? LT_RESET_BIT : 0u) |
                   (m_lightLearningRevalidatePending ? LT_REVALIDATE_BIT : 0u);
    if (useLightLearning) {
        m_lightLearningResetPending = false;
        m_lightLearningRevalidatePending = false;
    }
    rsConsts[24] = ++m_sharcFrame; // monotonic; unsigned age works across wrap
    setFloat(25, std::exp2((float)rs.sharcCellSizeExponent));
    setFloat(26, std::clamp(rs.sharcLodScale, 0.001f, 0.1f));
    rsConsts[27] = (UINT)std::clamp(rs.sharcMinSamples, 8, 256);
    rsConsts[28] = (UINT)std::clamp(rs.sharcHistoryFrames, 8, 256);
    rsConsts[29] = (UINT)std::clamp(rs.sharcMaxAge, 32, 4096);
    setFloat(30, std::clamp(rs.sharcQueryFootprint, 0.5f, 8.0f));
    rsConsts[31] = (UINT)rs.sharcTrainBounces;
    rsConsts[32] = (UINT)std::clamp(rs.sharcTrainRrDepth, 2, rs.sharcTrainBounces);

    const UINT guideQ = (UINT)std::lround(std::clamp(rs.sharcGuideMax, 0.0f, 0.9f) * 255.0f);
    const UINT guideFreshness = (UINT)std::clamp(rs.sharcGuideFreshness / 8 - 1, 0, 255);
    rs.sharcGuideDepth = std::clamp(rs.sharcGuideDepth, 1, 7);
    rsConsts[33] =
        (useSharc && rs.sharcGuideEnabled ? GUIDE_PARAM_ENABLED : 0u) | (guideQ << GUIDE_PARAM_QMAX_SHIFT) |
        ((UINT)rs.sharcGuideLevelOffset << GUIDE_PARAM_LEVEL_SHIFT) | (guideFreshness << GUIDE_PARAM_FRESHNESS_SHIFT) |
        (rs.sharcGuideTrain ? GUIDE_PARAM_TRAIN : 0u) |
        ((UINT)std::lround(std::clamp(rs.regularizeRoughness, 0.0f, 0.63f) * 100.0f) << GUIDE_PARAM_REGULARIZE_SHIFT) |
        ((UINT)rs.sharcGuideDepth << GUIDE_PARAM_DEPTH_SHIFT);
    rsConsts[34] = LT_BUFFER_OFFSET;
    rsConsts[35] = (UINT)std::clamp(rs.sharcUpdateStride, 2, 8);
    if (useSharc)
        m_sharcResetPending = false;

    // Tagged loops take their trip count from the integrator settings.
    auto resolveLoopCount = [&](const PassDesc& pass) -> uint32_t {
        if (pass.loopTag.empty())
            return std::max(pass.loopCount, 1u);
        if (pass.loopTag == L"pt_samples")
            return (uint32_t)std::clamp(rs.initialSamples, 1, 8);
        return 1u;
    };

    auto setConsts = [&](UINT w, UINT h) {
        rsConsts[0] = w;
        rsConsts[1] = h;
        rsConsts[6] = (UINT)std::clamp(rs.initialSamples, 1, 8) | (ptSampleIndex << 16);
        cmdList->SetComputeRoot32BitConstants(1, SHARC_ROOT_CONSTANTS, rsConsts, 0);
        cmdList->SetComputeRootUnorderedAccessView(2, m_skyBakeBuffer->GetGPUVirtualAddress());
        cmdList->SetComputeRootUnorderedAccessView(3, m_sharcBuffer->GetGPUVirtualAddress());
    };

    uint32_t activeFeatures = 0;
    if (useSharc)
        activeFeatures |= pass_feature::Sharc;
    if (sharcDebugMode != 0u)
        activeFeatures |= pass_feature::SharcDebug;
    if (useLightLearning)
        activeFeatures |= pass_feature::LightLearning;
    if (liteActive)
        activeFeatures |= pass_feature::DiffuseReuse;
    if (rs.liteSpatial)
        activeFeatures |= pass_feature::SpatialReuse;
    if (MeshLightsActive())
        activeFeatures |= pass_feature::MeshLights;
    for (auto& pass : m_passes.Passes())
        pass.executedLastFrame = false;
    bool dlssEvaluatedThisFrame = false;
    for (size_t i = 0; i < m_passes.Passes().size(); ++i) {
        auto& p = m_passes.Passes()[i];

        if (!p.IsEnabled(activeFeatures)) {
            if (i + 1 < m_passes.Passes().size() && m_passes.Passes()[i + 1].stage == Stage::Barrier)
                ++i;
            continue;
        }
        p.executedLastFrame = true;
        int cacheGroup = -1;
        if (p.file == L"Pass_sharc_prepare_v8.hlsl")
            cacheGroup = 0;
        else if (p.file == L"Pass_sharc_update_v8.hlsl")
            cacheGroup = 1;
        else if (p.file == L"Pass_sharc_resolve_v8.hlsl")
            cacheGroup = 2;
        else if (p.file.rfind(L"Pass_pt_", 0) == 0 && p.file != L"Pass_pt_skybake_v8.hlsl")
            cacheGroup = 3;
        else if (p.file == L"Pass_lite_shift_v8.hlsl")
            cacheGroup = 4;
        else if (p.file == L"Pass_lite_merge_v8.hlsl")
            cacheGroup = 5;
        else if (p.file == L"Pass_atmosphere_primary_v8.hlsl")
            cacheGroup = 6;
        else if (p.file == L"Pass_light_learning_v8.hlsl")
            cacheGroup = 7;

        UINT passTimer = GpuProfiler::InvalidPass;
        if (p.stage != Stage::LoopStart && p.stage != Stage::LoopEnd) {
            const int length = static_cast<int>(p.file.size());
            const int bytes = WideCharToMultiByte(CP_UTF8, 0, p.file.data(), length, nullptr, 0, nullptr, nullptr);
            std::string name(bytes, '\0');
            WideCharToMultiByte(CP_UTF8, 0, p.file.data(), length, name.data(), bytes, nullptr, nullptr);
            if (p.stage == Stage::Barrier)
                name = "UAV barriers";
            else if (p.stage == Stage::DLSS)
                name = "DLSS Ray Reconstruction";
            passTimer = m_gpuProfiler.BeginPass(cmdList, std::move(name), cacheGroup);
        }

        switch (p.stage) {
        case Stage::LoopStart:
            loopStack.push_back({resolveLoopCount(p), &p.loopTag});
            if (p.loopTag == L"pt_samples")
                ptSampleIndex = 0;
            break;

        case Stage::LoopEnd:
            if (!loopStack.empty()) {
                LoopFrame& frame = loopStack.back();
                frame.remaining--;
                if (frame.remaining > 0) {
                    if (*frame.tag == L"pt_samples")
                        ++ptSampleIndex;
                    i = p.targetIdx;
                } else {
                    if (*frame.tag == L"pt_samples")
                        ptSampleIndex = 0;
                    loopStack.pop_back();
                }
            }
            break;

        case Stage::Barrier: {
            auto u = CD3DX12_RESOURCE_BARRIER::UAV(nullptr);
            cmdList->ResourceBarrier(1, &u);
        } break;

        case Stage::RayGen: {
            cmdList->SetPipelineState1(m_rtStateObject.Get());
            cmdList->SetComputeRootSignature(m_rayGenSignature.Get());
            cmdList->SetComputeRootDescriptorTable(0, m_srvUavHeap->GetGPUDescriptorHandleForHeapStart());
            setConsts(dispW, dispH);

            const uint32_t rgSlot = m_passes.PassIndexByFile(p.file);
            raysDesc.RayGenerationShaderRecord.StartAddress = sbtStart + rgSlot * rgSize;
            raysDesc.RayGenerationShaderRecord.SizeInBytes = rgSize;
            raysDesc.Depth = 1;
            if (p.file == L"Pass_sharc_update_v8.hlsl") {
                // One training lane per tile of the strided schedule.
                const UINT stride = rsConsts[35];
                raysDesc.Width = (dispW + stride - 1u) / stride;
                raysDesc.Height = (dispH + stride - 1u) / stride;
            } else {
                raysDesc.Width = dispW;
                raysDesc.Height = dispH;
            }
            cmdList->DispatchRays(&raysDesc);
            break;
        }

        case Stage::Compute: {
            cmdList->SetPipelineState(m_csPSOs[p.psoIdx].Get());
            cmdList->SetComputeRootSignature(m_computeSignature.Get());
            cmdList->SetComputeRootDescriptorTable(0, m_srvUavHeap->GetGPUDescriptorHandleForHeapStart());
            setConsts(dispW, dispH);
            cmdList->Dispatch((dispW + p.groupX - 1) / p.groupX, (dispH + p.groupY - 1) / p.groupY, 1);
            break;
        }

        case Stage::FixedCompute: {
            cmdList->SetPipelineState(m_csPSOs[p.psoIdx].Get());
            cmdList->SetComputeRootSignature(m_computeSignature.Get());
            cmdList->SetComputeRootDescriptorTable(0, m_srvUavHeap->GetGPUDescriptorHandleForHeapStart());
            setConsts(dispW, dispH);
            const UINT groups = p.file == L"Pass_sharc_resolve_v8.hlsl"    ? SHARC_RESOLVE_GROUPS
                                : p.file == L"Pass_light_learning_v8.hlsl" ? LT_LEARNING_GROUPS
                                : p.file == L"Pass_sharc_prepare_v8.hlsl"  ? SHARC_CAPACITY / SHARC_GROUP_SIZE
                                                                           : p.groupX;
            cmdList->Dispatch(groups, p.groupY, 1);
            if (p.file == L"Pass_light_learning_v8.hlsl" && (rsConsts[23] & LT_RESET_BIT) == 0u) {
                auto learningBarrier = CD3DX12_RESOURCE_BARRIER::UAV(m_sharcBuffer.Get());
                cmdList->ResourceBarrier(1, &learningBarrier);
                cmdList->SetComputeRoot32BitConstant(1, rsConsts[23] | LT_INITIALIZE_BIT, 23);
                cmdList->Dispatch(groups, p.groupY, 1);
                cmdList->SetComputeRoot32BitConstant(1, rsConsts[23], 23);
            }
            break;
        }

        case Stage::DLSS: {
            if (!m_sentinelReadback[0]) {
                for (int i = 0; i < 3; ++i) {
                    auto hp = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK);
                    auto bd = CD3DX12_RESOURCE_DESC::Buffer(32);
                    ThrowIfFailed(m_ctx.Device()->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &bd,
                                                                          D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                                          IID_PPV_ARGS(&m_sentinelReadback[i])));
                    m_sentinelReadback[i]->SetName(L"DlssSentinelReadback");
                    void* p = nullptr;
                    ThrowIfFailed(m_sentinelReadback[i]->Map(0, nullptr, &p));
                    m_sentinelMapped[i] = static_cast<const uint32_t*>(p);
                }
            }
            // Read two frames behind to avoid stalling for diagnostics.
            if (m_sentinelFrame >= 2) {
                const uint32_t* s = m_sentinelMapped[(m_sentinelFrame + 1) % 3];
                auto bitsToF = [](uint32_t u) {
                    float x;
                    std::memcpy(&x, &u, 4);
                    return x;
                };
                auto& gs = m_dlss.sentinel;
                gs.mask = s[0];
                gs.maxLuma = bitsToF(s[1]);
                gs.maxMV = bitsToF(s[2]);
                gs.maxSpecMV = bitsToF(s[3]);
                gs.capCount = s[4];
                gs.badCount = s[5];
                gs.firstBad = s[6];
                gs.frame = m_sentinelFrame - 2;
                if (gs.mask != 0) {
                    gs.lastMask = gs.mask;
                    gs.lastBad = gs.firstBad;
                    gs.lastFrame = gs.frame;
                    const uint32_t bx = gs.firstBad & 0xFFFFu, by = gs.firstBad >> 16;
                    std::wcout << L"[DLSS-SENTINEL] frame " << gs.frame << L" mask 0x" << std::hex << gs.mask
                               << std::dec << L" badPixels " << gs.badCount << L" first (" << (bx ? bx - 1 : 0) << L","
                               << (by ? by - 1 : 0) << L")" << L" maxMV " << gs.maxMV << L" maxLuma " << gs.maxLuma
                               << std::endl;
                }
            }
            {
                auto toCopy = CD3DX12_RESOURCE_BARRIER::Transition(
                    m_autoExposeBuffer.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
                cmdList->ResourceBarrier(1, &toCopy);
                cmdList->CopyBufferRegion(m_sentinelReadback[m_sentinelFrame % 3].Get(), 0, m_autoExposeBuffer.Get(),
                                          32, 32);
                auto backToUav = CD3DX12_RESOURCE_BARRIER::Transition(
                    m_autoExposeBuffer.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                cmdList->ResourceBarrier(1, &backToUav);
            }
            ++m_sentinelFrame;

            ID3D12Resource* uavs[] = {m_dlss.Depth(),          m_dlss.MVec(),
                                      m_dlss.Normals(),        m_dlss.DiffuseAlbedo(),
                                      m_dlss.SpecularAlbedo(), m_dlss.Roughness(),
                                      m_dlss.SpecMVec(),       m_dlss.SpecHitDist(),
                                      m_dlss.Transparency(),   m_dlss.ColorBeforeTrans(),
                                      m_dlss.Input()};
            D3D12_RESOURCE_BARRIER guideBarriers[std::size(uavs)];
            UINT guideBarrierCount = 0;
            for (auto* resource : uavs)
                if (resource)
                    guideBarriers[guideBarrierCount++] = CD3DX12_RESOURCE_BARRIER::UAV(resource);
            if (guideBarrierCount)
                cmdList->ResourceBarrier(guideBarrierCount, guideBarriers);

            m_dlss.Evaluate(cmdList, m_ctx.Device(), *m_ctx.frameToken, m_ctx.viewportHandle, m_aspectRatio,
                            m_camera.ViewMatrix(), m_camera.PrevView(), m_camera.PrevProj(), m_camera.JitterX(),
                            m_camera.JitterY(), m_camera.JitterFrame(), m_camera.fovDegrees, m_camera.nearPlane,
                            m_camera.farPlane);
            dlssEvaluatedThisFrame = m_dlss.LastEvaluationSucceeded();
            p.executedLastFrame = dlssEvaluatedThisFrame;
            if (!dlssEvaluatedThisFrame || m_dlss.LastEvaluationReset())
                m_dlssNR.ForceReset();

            if (m_dlssG.available && m_dlssG.enabled) {
                sl::DLSSGOptions gOpts{};
                gOpts.mode = sl::DLSSGMode::eOn;
                gOpts.numFramesToGenerate = m_dlssG.framesToGenerate;
                gOpts.numBackBuffers = m_ctx.BufferCount();
                gOpts.mvecDepthWidth = m_dlss.RenderWidth();
                gOpts.mvecDepthHeight = m_dlss.RenderHeight();
                gOpts.colorWidth = GetWidth();
                gOpts.colorHeight = GetHeight();
                SL_CHECK(slDLSSGSetOptions(m_ctx.viewportHandle, gOpts));
            } else if (m_dlssG.available && !m_dlssG.enabled) {
                sl::DLSSGOptions gOpts{};
                gOpts.mode = sl::DLSSGMode::eOff;
                slDLSSGSetOptions(m_ctx.viewportHandle, gOpts);
            }

            m_camera.AdvanceFrame();

            // After DLSS: post-process passes run at display resolution
            dispW = GetWidth();
            dispH = GetHeight();

            ID3D12DescriptorHeap* h[] = {m_srvUavHeap.Get(), m_samplerHeap.Get()};
            cmdList->SetDescriptorHeaps(2, h);
            break;
        }

        default:
            break;
        }
        m_gpuProfiler.EndPass(cmdList, passTimer);
    }

    {
        auto toSrc = CD3DX12_RESOURCE_BARRIER::Transition(m_outputResource.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                                          D3D12_RESOURCE_STATE_COPY_SOURCE);
        cmdList->ResourceBarrier(1, &toSrc);
    }

    const bool inspectBuffers =
        sharcDebugMode != 0u || rs.dlssDebugLayer != 0 || (useLightLearning && rs.lightTreeDebug);
    UINT layer = inspectBuffers ? 3u : m_displayLevels[m_currentDisplayLevel];
    UINT sub = D3D12CalcSubresource(0, layer, 0, 1, 4);

    ID3D12Resource* presentSrc = m_outputResource.Get();
    UINT presentSub = sub;
    const bool nrSceneView = dlssEvaluatedThisFrame && sharcDebugMode == 0u && layer == 1u;
    if (!nrSceneView)
        m_dlssNR.ForceReset();
    if (nrSceneView) {
        const UINT timer = m_dlssNR.settings.enabled ? m_gpuProfiler.BeginPass(cmdList, "DLSS Neural Rendering")
                                                     : GpuProfiler::InvalidPass;
        if (m_dlssNR.Evaluate(cmdList, m_ctx.Device(), m_outputResource.Get(), sub, m_dlss.Depth(), m_dlss.MVec(),
                              m_dlss.RenderWidth(), m_dlss.RenderHeight())) {
            presentSrc = m_dlssNR.Output();
            presentSub = 0;
        }
        m_gpuProfiler.EndPass(cmdList, timer);
    }
    const UINT outputTimer = m_gpuProfiler.BeginPass(cmdList, "Output copy");

    {
        ID3D12DescriptorHeap* h[] = {m_srvUavHeap.Get(), m_samplerHeap.Get()};
        cmdList->SetDescriptorHeaps(2, h);
    }

    if (m_dlssG.enabled) {
        bool fresh = !m_dlssHudlessColor || m_dlssHudlessColor->GetDesc().Width != GetWidth() ||
                     m_dlssHudlessColor->GetDesc().Height != GetHeight();
        if (fresh) {
            // Previous-frame fence completed before this command list began.
            m_dlssHudlessColor.Reset();
            auto desc = CD3DX12_RESOURCE_DESC::Tex2D(DXGI_FORMAT_R8G8B8A8_UNORM, GetWidth(), GetHeight(), 1, 1);
            auto heap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
            ThrowIfFailed(m_ctx.Device()->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
                                                                  D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                                  IID_PPV_ARGS(&m_dlssHudlessColor)));
            m_dlssHudlessColor->SetName(L"DLSSG_HUDLessColor");
        } else {
            auto barrier = CD3DX12_RESOURCE_BARRIER::Transition(m_dlssHudlessColor.Get(),
                                                                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                                                D3D12_RESOURCE_STATE_COPY_DEST);
            cmdList->ResourceBarrier(1, &barrier);
        }
        CD3DX12_TEXTURE_COPY_LOCATION hudSrc(presentSrc, presentSub), hudDst(m_dlssHudlessColor.Get(), 0);
        cmdList->CopyTextureRegion(&hudDst, 0, 0, 0, &hudSrc, nullptr);
        auto barrier = CD3DX12_RESOURCE_BARRIER::Transition(m_dlssHudlessColor.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                                                            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        cmdList->ResourceBarrier(1, &barrier);
    }

    {
        auto toDst = CD3DX12_RESOURCE_BARRIER::Transition(m_ctx.BackBuffer(), D3D12_RESOURCE_STATE_RENDER_TARGET,
                                                          D3D12_RESOURCE_STATE_COPY_DEST);
        cmdList->ResourceBarrier(1, &toDst);
    }

    CD3DX12_TEXTURE_COPY_LOCATION src(presentSrc, presentSub);
    CD3DX12_TEXTURE_COPY_LOCATION dst(m_ctx.BackBuffer(), 0);
    D3D12_BOX box = {0, 0, 0, GetWidth(), GetHeight(), 1};
    cmdList->CopyTextureRegion(&dst, 0, 0, 0, &src, &box);

    m_gpuProfiler.EndPass(cmdList, outputTimer);
    if (m_editor.IsVisible()) {
        const UINT editorTimer = m_gpuProfiler.BeginPass(cmdList, "Editor");
        // Back buffer is in COPY_DEST after the texture copy — transition to RT
        auto toRT = CD3DX12_RESOURCE_BARRIER::Transition(m_ctx.BackBuffer(), D3D12_RESOURCE_STATE_COPY_DEST,
                                                         D3D12_RESOURCE_STATE_RENDER_TARGET);
        cmdList->ResourceBarrier(1, &toRT);

        auto rtv = m_ctx.CurrentRTV();
        cmdList->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
        ID3D12DescriptorHeap* heaps[] = {m_srvUavHeap.Get()};
        cmdList->SetDescriptorHeaps(1, heaps);

        m_editor.Render(cmdList);
        m_gpuProfiler.EndPass(cmdList, editorTimer);

        auto toPres = CD3DX12_RESOURCE_BARRIER::Transition(m_ctx.BackBuffer(), D3D12_RESOURCE_STATE_RENDER_TARGET,
                                                           D3D12_RESOURCE_STATE_PRESENT);
        cmdList->ResourceBarrier(1, &toPres);
    } else {
        auto b = CD3DX12_RESOURCE_BARRIER::Transition(m_ctx.BackBuffer(), D3D12_RESOURCE_STATE_COPY_DEST,
                                                      D3D12_RESOURCE_STATE_PRESENT);
        cmdList->ResourceBarrier(1, &b);
    }

    // ── DLSS-G: tag resources after back buffer has final content ──
    if (m_dlssG.enabled) {
        constexpr D3D12_RESOURCE_STATES stateUAV = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        constexpr D3D12_RESOURCE_STATES statePresent = D3D12_RESOURCE_STATE_PRESENT;
        sl::Resource slFgDepth(sl::ResourceType::eTex2d, m_dlss.Depth(), (uint32_t)stateUAV);
        sl::Resource slFgMVec(sl::ResourceType::eTex2d, m_dlss.MVec(), (uint32_t)stateUAV);
        // Contains the displayed scene (including NR), captured before ImGui.
        sl::Resource slFgHud(sl::ResourceType::eTex2d, m_dlssHudlessColor.Get(),
                             (uint32_t)D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        sl::Resource slFgBB(sl::ResourceType::eTex2d, m_ctx.BackBuffer(), (uint32_t)statePresent);

        sl::Extent renderExt{0, 0, m_dlss.RenderWidth(), m_dlss.RenderHeight()};
        sl::Extent displayExt{0, 0, m_dlss.DisplayWidth(), m_dlss.DisplayHeight()};
        auto fgLife = sl::ResourceLifecycle::eValidUntilPresent;

        sl::ResourceTag fgTags[] = {
            {&slFgDepth, sl::kBufferTypeDepth, fgLife, &renderExt},
            {&slFgMVec, sl::kBufferTypeMotionVectors, fgLife, &renderExt},
            {&slFgHud, sl::kBufferTypeHUDLessColor, fgLife, &displayExt},
            {&slFgBB, sl::kBufferTypeBackbuffer, fgLife, &displayExt},
            {nullptr, sl::kBufferTypeUIColorAndAlpha, fgLife, &displayExt},
        };

        SL_CHECK(slSetTagForFrame(*m_ctx.frameToken, m_ctx.viewportHandle, fgTags, _countof(fgTags), nullptr));
    }
    m_gpuProfiler.EndFrame(cmdList);
}

void Renderer::BuildLiteReuseTables(float sigma) {
    const int sizes[3] = {(int)LITE_REUSE_SIZE0, (int)LITE_REUSE_SIZE1, (int)LITE_REUSE_SIZE2};
    std::vector<uint32_t> words;
    words.reserve(LITE_REUSE_TEXELS);
    for (int t = 0; t < 3; ++t) {
        std::vector<int16_t> rg;
        GenerateReuseTexture(sizes[t], std::clamp(sigma, 1.0f, 64.0f), (uint32_t)(t + 1), rg);
        int bad = -1;
        if (!ValidateReuseTexture(sizes[t], rg, &bad)) {
            LOG(L"[ReSTIR lite] reuse table " << sizes[t] << L" failed self-inversion at texel " << bad);
            return;
        }
        for (size_t i = 0; i < rg.size(); i += 2)
            words.push_back((uint32_t)(uint16_t)rg[i] | ((uint32_t)(uint16_t)rg[i + 1] << 16));
    }

    if (m_liteReuseUpload)
        m_liteReuseRetired.push_back(m_liteReuseUpload);
    if (m_liteReuseRetired.size() > 8)
        m_liteReuseRetired.erase(m_liteReuseRetired.begin());
    ResourceFactory rf(m_ctx.Device());
    m_liteReuseUpload = rf.CreateUploadBufferWithData(words.data(), (UINT)(words.size() * sizeof(uint32_t)));
    m_liteReusePending = true;
}
