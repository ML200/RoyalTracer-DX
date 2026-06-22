//====================================
//RENDERER ORCHESTRATOR
//====================================
//init, update, render, destroy, heavy logic in modules

#include "stdafx.h"
#include "Renderer.h"
#include "Diagnostics.h"
#include "Windowsx.h"
#include "ReuseTextureGen.h"
#include "NRC/NrcNetwork.h"
#include <random>

#undef SL_CHECK
#define SL_CHECK(x) do { sl::Result r = (x); if (r != sl::Result::eOk) { \
    std::wcout << L"[SL] " << L#x << L" failed: " << (int)r << std::endl; } \
} while(0)

// ─────────────────────────────────────────────────────────────────
Renderer::Renderer(UINT width, UINT height)
    : m_width(width), m_height(height),
      m_aspectRatio(static_cast<float>(width) / static_cast<float>(height))
{
    /*m_passes.Build({
        L"Pass_debug_v8.hlsl|rg",                       L"barrier",
    });*/
    m_passes.Build({
        L"cuda:nrc_frame_begin",                        L"barrier",
        L"Pass_spmis_reset_v8.hlsl|cs:8x4",             L"barrier",
        L"Pass_raygen_v8.hlsl|rg",                      L"barrier",
        L"Pass_clouds_primary_v8.hlsl|cs:16x16",        L"barrier",
        L"cuda:nrc_inference",                          L"barrier",
        L"Pass_nrc_resolve_v8.hlsl|cs:8x8",             L"barrier",
        L"Pass_temp_gi_v8.hlsl|rg",                     L"barrier",
        L"Pass_spat_gi_select_v8.hlsl|cs:16x16",        L"barrier",
        L"Pass_spat_gi_shift_v8.hlsl|rg",               L"barrier",
        L"Pass_spat_gi_v8_1.hlsl|cs:16x16",             L"barrier",
        L"Pass_spmis_count_v8.hlsl|cs:8x4",             L"barrier",
        L"Pass_spmis_offsets_v8.hlsl|cs:8x4",           L"barrier",
        L"Pass_spmis_sort_v8.hlsl|cs:8x4",              L"barrier",
        L"Pass_spmis_reuse_v8.hlsl|cs:8x8",             L"barrier",
        L"Pass_dup_gi_v8.hlsl|cs:16x16",                L"barrier",
        L"Pass_nrc_train_gather_v8.hlsl|cs:8x8",        L"barrier",
        L"cuda:nrc_train",                              L"barrier",
        L"Pass_nrc_debug_query_v8.hlsl|cs:8x8",         L"barrier",
        L"cuda:nrc_debug_inference",                    L"barrier",
        L"Pass_nrc_debug_present_v8.hlsl|cs:8x8",       L"barrier",
        L"Pass_shading_v8.hlsl|cs:16x16",               L"barrier",
        L"dlss",                                        L"barrier",
        L"Pass_autoexpose_reduce_v8.hlsl|cs:8x8",       L"barrier",
        L"Pass_autoexpose_finalize_v8.hlsl|fx:1",       L"barrier",
        L"Pass_postprocess_v8.hlsl|cs:8x4",             L"barrier",
    });

}

//====================================
//INIT
//====================================
void Renderer::InitDevice() {
    try {
        m_ctx.Init(Win32Application::GetHwnd(), GetWidth(), GetHeight());

        // PLANET_INTEGRATION: bring up the planet generation pipeline.
        {
            planet::StreamConfig cfg;
            //==== PLANET ON/OFF — flip to false to disable terrain entirely ====
            //false: scene-only unified TLAS, no generation built. Use this to
            //bisect the planet terrain of the renderer.
            // PLANET DISABLED 2026-06-10: terrain mesh system turned off (the
            // perf cost and image-quality gain weren't worth it). With this
            // false, begin_frame() is a no-op, terrain_reservation() returns
            // zero, and the rock scatter / bind_geometry blocks below are
            // skipped — but submit_work() still rebuilds the scene-only unified
            // TLAS every frame, so the renderer is unaffected. All planet code
            // stays compiled and in the tree; flip back to true to re-enable.
            cfg.enabled = false;
            cfg.planet.radius = 6371000.0;
            //The planet sits BELOW the world origin: its surface passes exactly
            //through (0,0,0), so the camera spawns standing ON the surface and
            //the planet body extends downward (-Y). center.y = -radius.
            cfg.planet.center = { 0.0, -cfg.planet.radius, 0.0 };
            cfg.max_lod       = 16;            // hard depth cap; the budget sets actual depth
            cfg.max_triangles = 3000000u;      // planet-wide triangle budget - the quadtree
                                               // auto-tunes its leaf cut to fit under this
            //heightmap defaults (StreamConfig) are tiny test bumps - tune here.
            m_planet.init(m_ctx.Device(), &m_ctx, cfg);

            //mirror the planet params into the camera cbuffer so the HLSL
            //terrain shader samples the exact surface the tessellator built.
            m_camera.planetCenter = glm::vec3((float)cfg.planet.center.x,
                                              (float)cfg.planet.center.y,
                                              (float)cfg.planet.center.z);
            m_camera.planetRadius           = (float)cfg.planet.radius;
            //terrainHeightAmplitude / terrainHeightFrequency are vestigial from
            //the sin/cos era - the shader now samples a baked heightmap. Left
            //at zero until the cbuffer slots are repurposed.
            m_camera.terrainHeightAmplitude = 0.0f;
            m_camera.terrainHeightFrequency = 0.0f;
        }

        // CUDA/D3D12 interop. Optional — if this fails (no CUDA device, LUID
        // mismatch, etc.) the renderer runs fine; cuda:* passes become no-ops.
        if (m_cudaInterop.Init(m_ctx.Device())) {
            m_cudaFence = m_cudaInterop.CreateFence(L"Cuda_Interop_Fence");
            LOG(L"[CUDA] Interop ready");

            // Smoke test callback: exercises the fence split + tcnn once on first
            // frame, then no-ops but still round-trips the fence every frame so
            // the D3D12<->CUDA sync path stays live and exercised.
            RegisterCudaOp(L"nrc_smoke", [this]{
                static bool s_ranOnce = false;
                if (s_ranOnce) return;
                s_ranOnce = true;
                const bool ok = nrc::SmokeTest(m_cudaInterop.Stream());
                LOG(L"[NRC] SmokeTest " << (ok ? L"OK" : L"FAILED"));
            });

            // ── NRC setup ─────────────────────────────────────────
            // Cap inference capacity at 1 × pixels: each pixel fits at most one
            // cache-termination record, and the debug view's inference (slot =
            // MapPixelID, max TileAlignedPx) reuses the same buffer after the
            // main chain has consumed it. Round up to tcnn's batch granularity.
            // (Was 2× to also reserve a depth-0 sharp-reflection record per
            // pixel; that feature was dead and has been removed, so the second
            // half was never written — halving this reclaims the VRAM.)
            //
            // Use the TILE-ALIGNED pixel count, not W*H. PendingGI is indexed
            // by MapPixelID (the 8×4 tile swizzle), whose max index is
            // TileAlignedPx(W,H) == ps_numPx() — strictly greater than W*H at
            // any render resolution that isn't a multiple of (8,4), which is
            // common at DLSS render scales. Sizing at W*H let raygen's
            // NrcWriteTerminationRecord / NrcClearPendingGI scribble past the
            // buffer end and let resolve read uninitialised slots → garbage
            // NRC output → corrupted reservoirs. Every other MapPixelID-indexed
            // buffer (path-state, reservoirs, sample data) already uses
            // TileAlignedPx; NRC was the one that didn't.
            const uint32_t pixelCount = TileAlignedPx(GetWidth(), GetHeight());
            m_nrcInferenceCapacity   = nrc::AlignBatch(pixelCount);
            m_nrcDynamicInferenceCap = m_nrcInferenceCapacity;

            m_nrcInferenceIn  = m_cudaInterop.CreateBuffer(nrc::InferenceInputBytes (m_nrcInferenceCapacity), L"NRC_InferenceIn");
            m_nrcInferenceOut = m_cudaInterop.CreateBuffer(nrc::InferenceOutputBytes(m_nrcInferenceCapacity), L"NRC_InferenceOut");
            m_nrcPendingGI    = m_cudaInterop.CreateBuffer(nrc::PendingGIBytes      (pixelCount),             L"NRC_PendingGI");
            m_nrcTrainRecords = m_cudaInterop.CreateBuffer(nrc::TrainingBytes       (),                       L"NRC_TrainRecords");
            m_nrcCounters     = m_cudaInterop.CreateBuffer(nrc::CountersBytes       (),                       L"NRC_Counters");

            const bool allocOK = m_nrcInferenceIn.resource && m_nrcInferenceOut.resource
                              && m_nrcPendingGI.resource   && m_nrcTrainRecords.resource
                              && m_nrcCounters.resource;
            if (!allocOK) {
                LOG(L"[NRC] Shared buffer allocation failed — NRC disabled");
            } else if (!m_nrcNetwork.Init()) {
                LOG(L"[NRC] tcnn network init failed — NRC disabled");
            } else {
                m_nrcReady = true;
                LOG(L"[NRC] Ready: infCapacity=" << m_nrcInferenceCapacity
                    << L" trainingPaths="       << nrc::kMaxTrainingPaths);
            }

            // Frame start: zero the counters, invalidate every PendingGI
            // record, and clear the training path-meta section (leaves
            // last frame's stale numVertices=0 so the backward-fill
            // kernel skips unused paths).
            RegisterCudaOp(L"nrc_frame_begin", [this]{
                if (!m_nrcReady || !m_nrcSettings.enabled) return;
                void* s = m_cudaInterop.Stream();
                // Gate the train records memset on the previous frame's
                // auxStream training (specifically its fill kernel, which
                // reads m_nrcTrainRecords meta + per vertex). Without this
                // wait, the cudaMemsetAsync below races the still in flight
                // fill kernel: a torn 16 byte meta read can land
                // tailKind==kTailCache alongside a garbage inferenceSlot,
                // and the fill kernel's deref into m_nrcInferenceOut MMU
                // faults inside the kernel which TDRs. The fill kernel
                // belt is paired with a slot bound check in TrainFrame's
                // launch, this is the suspenders.
                m_nrcNetwork.WaitTrainDoneOnStream(s);
                nrc::Memzero(s, m_nrcCounters.cudaPtr,    m_nrcCounters.sizeBytes);
                nrc::Memfill(s, m_nrcPendingGI.cudaPtr,   0xFF, m_nrcPendingGI.sizeBytes);
                // Only clear the meta header — per-vertex payload is
                // overwritten by raygen for freshly-allocated path ids.
                nrc::Memzero(s, m_nrcTrainRecords.cudaPtr, nrc::kPathMetaTotalBytes);
            },
            //"Enable cache" off in the editor short circuits every cache
            //subsystem: skip the whole D3D12<->CUDA interop round trip here.
            //Paired with nrc_inference on the same predicate so the counter,
            //PendingGI and training meta clears stay matched to the work
            //that consumes them.
            [this]{ return m_nrcReady && m_nrcSettings.enabled; });

            // Main-chain inference. The dispatch size matches the dynamic cap
            // computed at frame start from prior frame's actual counter, so
            // raygen and the CUDA inference agree on the upper bound. The
            // counter readback is async (mirrors the training counter pattern),
            // so the host never stalls on the GPU. Records past raygen's
            // actual write count up to dynamicCap have stale features but
            // their outputs are never read: resolve only follows PendingGI
            // (set only for valid slots) and debug_query overwrites its own
            // slots before its own inference. Schedule the next frame's
            // readback before dispatching so the copy queues alongside
            // inference instead of after it.
            RegisterCudaOp(L"nrc_inference", [this]{
                if (!m_nrcReady || !m_nrcSettings.enabled) return;
                void* s = m_cudaInterop.Stream();
                m_nrcNetwork.ScheduleInferenceCounterReadback(s, m_nrcCounters.cudaPtr);
                const uint32_t padded = nrc::AlignBatch(m_nrcDynamicInferenceCap);
                m_nrcNetwork.Inference(
                    s,
                    static_cast<const float*>(m_nrcInferenceIn.cudaPtr),
                    static_cast<float*>      (m_nrcInferenceOut.cudaPtr),
                    padded);
                //async sum of |terrain| over the batch, harvested next frame by
                //the collapse detector in the NRC tick block
                m_nrcNetwork.ScheduleInferenceOutSumReadback(
                    s,
                    static_cast<const float*>(m_nrcInferenceOut.cudaPtr),
                    padded);

                //Training NO LONGER rides inside this op. It was moved to the
                //late cuda:nrc_train op (after the ReSTIR reuse chain) so the
                //x1 training rows can be gathered from the CONVERGED reservoir
                //(Pass_nrc_train_gather_v8). Inference must still run HERE --
                //the cache short-circuit candidates that Pass_nrc_resolve_v8
                //injects into the reservoir depend on this frame's inferenceOut,
                //and that has to be ready before the temporal/spatial passes.
            },
            //gated on the editor master switch: when the cache is disabled,
            //skip the whole inference round trip so the NRC stack goes dark,
            //matching nrc_frame_begin.
            [this]{ return m_nrcReady && m_nrcSettings.enabled; });

            // Additional inference for the debug view only. Runs AFTER
            // training, so main inferenceOut has already been consumed
            // by resolve + train — we're free to overwrite the buffers
            // with per-pixel predictions for Pass_nrc_debug_present.
            // Debug_query wrote one query per pixel at slot = pixelIdx,
            // so count is W*H.
            RegisterCudaOp(L"nrc_debug_inference",
                [this]{
                    if (!m_nrcReady || !m_nrcSettings.enabled || !m_nrcSettings.debugView) return;
                    void* s = m_cudaInterop.Stream();
                    // Debug query writes one record per pixel at slot = MapPixelID
                    // (the 8x4 tile swizzle), whose max index is TileAlignedPx, NOT
                    // W*H — at non-tile-divisible render resolutions the high tiles
                    // would be left un-inferred (stale -> black tiles in the debug
                    // view). Must match the buffer sizing.
                    const uint32_t count   = TileAlignedPx(GetWidth(), GetHeight());
                    const uint32_t clamped = (count < m_nrcInferenceCapacity) ? count : m_nrcInferenceCapacity;
                    const uint32_t padded  = nrc::AlignBatch(clamped);
                    m_nrcNetwork.Inference(
                        s,
                        static_cast<const float*>(m_nrcInferenceIn.cudaPtr),
                        static_cast<float*>      (m_nrcInferenceOut.cudaPtr),
                        padded);
                },
                //skip the entire fence round-trip when debug view is off;
                //typical play config never enables debug view so this kills
                //one full close/execute/fence/wait/reopen cycle per frame
                [this]{ return m_nrcReady && m_nrcSettings.enabled && m_nrcSettings.debugView; });

            // Backward-fill + 4 SGD steps. This op runs LATE in the frame,
            // right after Pass_nrc_train_gather_v8 has written the x1 training
            // rows from the converged ReSTIR reservoir (and after raygen wrote
            // the v2+ rows). Training was previously folded into
            // cuda:nrc_inference (early), but the x1-from-reservoir scheme needs
            // the reuse chain to finish first, so it gets its own late op again.
            //
            // SYNC: TrainFrame's fill kernel runs on auxStream and reads
            // m_nrcTrainRecords, which the gather pass (D3D12) just wrote. The
            // dispatcher's CudaWait(preVal) before this lambda makes the MAIN
            // stream `s` wait on the D3D12 fence (so the gather's UAV writes are
            // visible to anything ordered after `s`); TrainFrame records an
            // event on `s` and makes auxStream wait on it, ordering the fill
            // kernel after the gather. It also still waits on inferenceDoneEvent
            // (recorded back in cuda:nrc_inference) for the cache-tail
            // inferenceOut reads.
            RegisterCudaOp(L"nrc_train", [this]{
                if (!m_nrcReady || !m_nrcSettings.enabled || !m_nrcSettings.trainingEnabled) return;
                void* s = m_cudaInterop.Stream();
                //Live LR override: editor's learningRateScale scales the base
                //1e-2 so the Adam rate is tunable at runtime (default 1.0 = no
                //change).
                m_nrcNetwork.SetLearningRate(nrc::kBaseLearningRate * m_nrcSettings.learningRateScale);
                m_nrcNetwork.TrainFrame(
                    s,
                    m_nrcTrainRecords.cudaPtr,
                    m_nrcInferenceOut.cudaPtr,
                    m_nrcInferenceCapacity,
                    m_nrcTrainRecordsTarget);
            },
            //skip the interop round trip entirely when training is off
            [this]{ return m_nrcReady && m_nrcSettings.enabled && m_nrcSettings.trainingEnabled; });
        } else {
            LOG(L"[CUDA] Interop disabled (no matching CUDA device)");
        }

        m_simulator.PromptUserConfiguration();
        m_recorder.Initialize();
        m_camera.Init(m_ctx.Device(), GetWidth(), GetHeight());

        if (!m_ctx.viewportHandle) {
            sl::Result r = slAllocateResources(m_ctx.CmdList(), sl::kFeatureDLSS_RR, m_ctx.viewportHandle);
            if (r != sl::Result::eOk)
                std::wcout << L"[SL] slAllocateResources failed: " << (int)r << std::endl;
        }
        GenerateLutTextures();
        InitReuseTextures();
        InitSkyStarsTexture();
        //Bake the 256³ cloud noise texture before CreateShaderResourceHeap
        //runs, so the heap can pick up the resulting SRV at heap slot
        //CLOUD_NOISE_HEAP_SLOT. One-time GPU work on the init cmd list.
        BakeCloudNoiseTexture();
        //Load the NASA Blue Marble cloud coverage map (8192×4096 → R8 lum).
        //SRV will land at CLOUD_COVERAGE_HEAP_SLOT (register t43).
        //MUST precede InitSkyLUTBake: the cloud sun-OD bake heap binds SRVs to
        //the cloud noise + coverage textures, so both resources must exist.
        InitCloudCoverageTexture();
        //Create the per-frame sky LUT textures (sun transmittance t49 +
        //cloud ambient probes t50 + cloud sun-OD shell map t52) before
        //CreateShaderResourceHeap so their SRVs land at heap slots 73/74/76.
        //The actual bake dispatches run every frame from PopulateCommandList
        //(RecordSkyLUTBake).
        InitSkyLUTBake();
        //Upload the baked planet heightmap cubemap (downsample to 4k per
        //face). SRV at TERRAIN_HEIGHTMAP_HEAP_SLOT (register t45). Requires
        //m_planet.init to have run already so its HeightmapCubemap is loaded.
        // PLANET DISABLED 2026-06-10: skip the terrain texture uploads. The
        // orchestrator still loads the heightmap from disk at init(), so these
        // would otherwise upload regardless of cfg.enabled and keep perturbing
        // the sky. With them commented out, m_terrain*Texture stay null and the
        // SRV heap binds NULL descriptors at slots 69-72 (t45-t48) — the
        // shaders already treat a null/zero sample as "no terrain":
        // TerrainHeight()/TerrainCloudBaseHeight() return 0 (flat sphere, so
        // clouds/atmosphere/sun get no terrain relief) and TerrainTintFromUV()
        // returns false. Also avoids the large 8192^2 x6 cubemap uploads.
        // Re-enable alongside cfg.enabled above.
        //InitTerrainHeightmapTexture();
        //Companion textures from the baker v8 output: surface_color (Mars
        //tint, t46), normal map (t47), and cloud_offset (smoothed elevation
        //for cloud base lookup, t48). Each Init is a no-op if the bake
        //didn't produce that layer; the shaders fall back to the legacy
        //constants in that case.
        //InitTerrainSurfaceColorTexture();
        //InitTerrainNormalTexture();
        //InitTerrainCloudOffsetTexture();
        //Bake the 128x128x64 spatiotemporal blue noise array. SRV lands at
        //CLOUD_STBN_HEAP_SLOT (register t41). Drives CloudRand4 so the
        //cloud march's per pixel jitter has blue noise spatial spectrum.
        BakeCloudSTBNTexture();

        D3D12_FEATURE_DATA_D3D12_OPTIONS5 opts5 = {};
        ThrowIfFailed(m_ctx.Device()->CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS5, &opts5, sizeof(opts5)));
        if (opts5.RaytracingTier < D3D12_RAYTRACING_TIER_1_0)
            throw std::runtime_error("Raytracing not supported on device");

        // Initialize Reflex
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

        // Initialize DLSS Frame Generation
        {
            LUID luid = m_ctx.Device()->GetAdapterLuid();
            sl::AdapterInfo ai;
            ai.deviceLUID = (uint8_t*)&luid;
            ai.deviceLUIDSizeInBytes = sizeof(LUID);
            sl::Result sr = slIsFeatureSupported(sl::kFeatureDLSS_G, ai);
            if (sr == sl::Result::eOk) {
                m_dlssG.available = true;
                // Query max generated frames from hardware
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
        //PLANET ROCKS: generate boulder variants and register them as ordinary
        //scene meshes (geometry -> combined buffers, one BLAS each) BEFORE the
        //buffers are built. No scene instances are created for them; the rock
        //streamer emits their TLAS instances per frame. CreateProceduralMesh
        //builds each BLAS, so CreateAccelerationStructures below skips them.
        m_rockMeshIndices.clear();
        if (m_planet.enabled()) {
            const auto rockVariants =
                planet::generate_rock_variants(/*count*/6, /*subdiv*/2, /*seed*/0xB0DECA11u);
            Material rockMat;
            rockMat.Kd             = DirectX::XMFLOAT4(0.30f, 0.27f, 0.24f, 1.0f);
            rockMat.Ke             = DirectX::XMFLOAT3(0.0f, 0.0f, 0.0f);
            rockMat.Ni             = 1.0f;
            rockMat.Pr_Pm_Ps_Pc    = DirectX::XMFLOAT4(1.0f, 0.0f, 0.0f, 0.0f);
            rockMat.albedoTexID    = -1;
            rockMat.normalTexID    = -1;
            rockMat.rmaTexID       = -1;
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
        CreateAccelerationStructures();
        //PLANET: reserve the terrain region in the combined scene buffers (and
        //append the flat terrain material) BEFORE the buffers are built. No-op
        //when terrain is disabled (terrain_reservation() returns all zero).
        {
            const auto tr = m_planet.terrain_reservation();
            m_scene.ReserveTerrain(tr.vertexElems, tr.indexElems, tr.matIDElems,
                                   tr.triLightElems, tr.instanceSlots, tr.propsBase);
        }
        //PLANET ROCKS: reserve the rock instanceProps range after the terrain
        //region (must precede CreateInstancePropertiesBuffer below).
        if (m_planet.enabled()) m_scene.ReserveRocks(planet::MAX_ROCK_INSTANCES);
        m_scene.BuildGlobalMeshBuffers(m_ctx.Device(), m_ctx.CmdList());
        m_ctx.FlushAndReset();
        //PLANET ROCKS: variant geometry now lives in the combined buffers - read
        //back each variant's base offsets + BLAS, hand them to the orchestrator,
        //then configure the camera-following scatter.
        if (m_planet.enabled() && !m_rockMeshIndices.empty()) {
            std::vector<planet::StreamOrchestrator::RockVariantGPU> rockDescs;
            rockDescs.reserve(m_rockMeshIndices.size());
            for (UINT mi : m_rockMeshIndices) {
                const auto& rmesh = m_scene.meshes[mi];
                planet::StreamOrchestrator::RockVariantGPU d;
                d.blas_va    = rmesh.blas ? rmesh.blas->GetGPUVirtualAddress() : 0;
                d.vertexBase = rmesh.globalVertexBase;
                d.indexBase  = rmesh.globalIndexBase;
                d.triCount   = rmesh.opaqueTriCount;
                rockDescs.push_back(d);
            }
            m_planet.set_rock_variants(m_scene.rockPropsBase, rockDescs);

            planet::RockScatterConfig rc;
            rc.planet    = m_planet.planet_geometry();
            rc.max_rocks = planet::MAX_ROCK_INSTANCES;
            m_rockScatter.configure(rc, (int)m_rockMeshIndices.size());
        }
        m_lutUploadHeaps.clear();
        m_skyStarsUploadHeap.Reset();

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

        //PLANET: hand the unified scene+terrain buffers to the orchestrator.
        //Must run AFTER BuildGlobalMeshBuffers (combined buffers + mapped ptrs),
        //CreateInstancePropertiesBuffer (terrain-slot capacity), UploadMaterials
        //(terrainMatIDBase) and CreateShaderResourceHeap (terrainTriLightBase,
        //set when CreateTriToLightIdBuffer runs inside it).
        if (m_planet.enabled()) {
            const auto tr = m_planet.terrain_reservation();
            m_planet.bind_geometry(
                m_scene.vertexGlobal.Get(), m_scene.vertexGlobalMapped,
                m_scene.indexGlobal.Get(),  m_scene.indexGlobalMapped,
                m_scene.instanceProperties.Get(),
                m_scene.totalVertexCount, m_scene.totalIndexCount,
                m_scene.combinedVertexCount(), tr.propsBase,
                tr.leafSlots, m_scene.terrainMatIDBase, m_scene.terrainTriLightBase);
        }

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
            m_editor.Init(Win32Application::GetHwnd(), m_ctx.Device(), m_ctx.BufferCount(),
                m_srvUavHeap.Get(), fontCpu, fontGpu);
        }
        // DLSS-G resources will be allocated on the first frame (after slSetConstants)
        // because slAllocateResources requires constants to be set first.

        // Command list left open — EngineApp closes it.
    } catch (const std::exception& e) {
        //A lost device (DXGI_ERROR_DEVICE_REMOVED) surfaces here as a generic
        //failure from whatever resource call first noticed it — not where the
        //GPU actually faulted. Poll up to 1s for the removal to latch into
        //GetDeviceRemovedReason, then dump DRED: its breadcrumbs + page-fault
        //data name the real faulting op. Returns fast if the device is
        //healthy, so non-device-lost exceptions still reach the message box.
        dxdiag::CheckDeviceRemoved(m_ctx.Device(), 1000);
        wchar_t wMsg[4096];
        MultiByteToWideChar(CP_UTF8, 0, e.what(), -1, wMsg, 4096);
        MessageBoxW(NULL, wMsg, L"Fatal Init Error", MB_OK | MB_ICONERROR);
        exit(1);
    }
}

//====================================
//UPDATE
//====================================
void Renderer::UpdateRenderer(float dt) {
    using hrc = std::chrono::high_resolution_clock;

    // Reflex sleep — must be called every frame regardless of mode
    slReflexSleep(*m_ctx.frameToken);

    // PCL: simulation start
    slPCLSetMarker(sl::PCLMarker::eSimulationStart, *m_ctx.frameToken);

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
    if (m_dlss.mode != m_dlss.ActiveMode()) {
        m_ctx.WaitForGPU();  // drain all in-flight GPU work BEFORE releasing old textures
        if (m_dlss.UpdateMode(m_ctx.Device())) {
            m_dlssModeChangedFrames = 2;  // skip temporal reuse for 2 frames
            m_camera.ResetJitter();       // force DLSS temporal reset (fixes blur)
            RebuildDLSSDescriptors();
            LOG(L"[DLSS] Mode changed → render res: "
                << m_dlss.RenderWidth() << L"x" << m_dlss.RenderHeight());
        }
    }

    // Toggling emitter-spike clamping changes the DLSS input for emitter pixels;
    // drop temporal history so the two encodings don't blend/ghost during the swap.
    if (m_dlss.clampEmitterSpikes != m_dlssClampEmitterSpikesPrev) {
        m_dlssClampEmitterSpikesPrev = m_dlss.clampEmitterSpikes;
        m_dlss.ForceReset();
        LOG(L"[DLSS] Emitter-spike clamp " << (m_dlss.clampEmitterSpikes ? L"ON" : L"OFF"));
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
        m_pendingTLASUpload    = std::move(refitResult.nodes);
        m_pendingBLASBitTrail  = std::move(refitResult.blasBitTrails);

        // Update worldToLocal in BLASRanges from refit result
        auto& cpuRanges = m_lightTree.GetCpuBLASRanges();
        if (!refitResult.blasWorldToLocal.empty() && !cpuRanges.empty()) {
            m_pendingBLASRanges = cpuRanges;  // copy static fields
            for (size_t i = 0; i < m_pendingBLASRanges.size() && i < refitResult.blasWorldToLocal.size(); ++i)
                m_pendingBLASRanges[i].worldToLocal = refitResult.blasWorldToLocal[i];
        }

        LOG(L"[LightTree] Async TLAS refit ready: " << m_pendingTLASUpload.size() << L" nodes");
    }

    // Build editor UI — must run before UpdateInstanceProperties so
    // MarkModelMoved() changes are picked up in the same frame
    static FlyCamController dummyFlyCam;
    m_editor.Draw(m_scene, m_camera, m_flyCam ? *m_flyCam : dummyFlyCam,
                  m_passes, m_dlss, m_dlssG, m_restirSettings, m_nrcSettings,
                  m_fps, m_frameStats, m_planet.stats());
    // Debug-view checkbox only enables the calculation into slice 3.
    // Cycle to it with 'C' when you want to look at it.

    // ── Floating origin sync ─────────────────────────────────────
    // Snap the scene origin to a 1 km grid following the camera. If it
    // moved this frame, every instance transform needs re-shifting and
    // the TLAS BVH has to be rebuilt (refit can't handle the big jump).
    // Done BEFORE PrepareInstanceProperties so the new origin is in
    // effect when transforms get shifted.
    m_camera.PollSceneOrigin();
    if (m_camera.consumeOriginShifted()) {
        //Every instance needs its transform re-shifted to the new origin.
        //PrepareInstanceProperties picks these up via the dirty list and
        //rewrites cpuInstanceProps + tlasInstances.transform.
        m_scene.MarkAllInstancesDirty();
        //tlasDirty + non-empty dirtyInstanceList sends the TLAS through
        //UpdateAndRefit (microseconds for any reasonable instance count)
        //instead of tlasFullRebuild which re-allocates GPU buffers + the
        //instance descriptor heap and costs ~ms regardless of count. The
        //BVH may be slightly suboptimal after a big shift but the existing
        //periodic RebuildInPlace (every 120 frames) recovers structure.
        m_scene.tlasDirty       = true;
        //Light tree triangle positions are computed from xforms shifted by
        //sceneOriginWorld (see BuildXformsFromScene). Re-refit so NEE
        //queries from the new origin hit the right cluster nodes.
        m_scene.lightTreeDirty  = true;
    }
    const auto camOrigin = m_camera.getSceneOriginWorld();
    m_scene.sceneOriginWorld = { camOrigin.x, camOrigin.y, camOrigin.z };

    // Prepare instance data on CPU shadow buffer (overlaps with GPU)
    auto t_instStart = hrc::now();
    m_scene.PrepareInstanceProperties();
    auto t_instEnd = hrc::now();

    m_frameStats.cpuInstanceMs = std::chrono::duration<float, std::milli>(t_instEnd - t_instStart).count();
    m_frameStats.cpuUpdateMs   = std::chrono::duration<float, std::milli>(t_instEnd - t_updateStart).count();
    m_frameStats.instanceCount = (UINT)m_scene.instances.size();
    m_frameStats.meshCount     = (UINT)m_scene.meshes.size();

    // PCL: simulation end
    slPCLSetMarker(sl::PCLMarker::eSimulationEnd, *m_ctx.frameToken);

    // ── GPU sync + upload ────────────────────────────────────────
    // Wait for previous frame, then write all shared upload-heap buffers.
    auto t_waitStart = hrc::now();
    m_ctx.WaitForPreviousFrame();
    m_frameStats.gpuMs = std::chrono::duration<float, std::milli>(hrc::now() - t_waitStart).count();

    //current/last sample-buffer ping-pong - must come right after the full
    //GPU sync and before any command recording for this frame
    SwapSampleBuffers();

    m_camera.UploadGPUBuffer(m_aspectRatio);
    //ResetView (editor button) snapped prevView = view inside UploadGPUBuffer
    //so MVs come terrain at zero this frame; tell DLSS RR to discard its history
    //so it doesn't blend the pre-reset frame on top of the new pose. Must
    //run before slEvaluateFeature(DLSS_RR) downstream.
    if (m_camera.ConsumeResetPending()) {
        m_dlss.ForceReset();
    }
    m_scene.UploadInstanceProperties();
    if (m_scene.materialsDirty) {
        m_scene.UpdateMaterialBuffer();
        m_scene.materialsDirty = false;
    }
}

// ─────────────────────────────────────────────────────────────────
std::vector<InstanceXformCPU> Renderer::BuildXformsFromScene() const {
    std::vector<InstanceXformCPU> xf;
    xf.reserve(m_scene.instances.size());
    //Apply the floating origin shift so light tree positions match the
    //shifted ray origins. Without this, NEE distance lookups would be off
    //by sceneOriginWorld magnitude at orbital altitudes.
    const XMVECTOR shift = XMVectorSet(m_scene.sceneOriginWorld.x,
                                        m_scene.sceneOriginWorld.y,
                                        m_scene.sceneOriginWorld.z, 0.0f);
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
    const UINT itemCount = (UINT)m_pendingBLASBitTrail.size();
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

    // ── BLAS bit trails (per-BLAS TLAS descent path used by the PDF) ─
    if (itemCount > 0) {
        growBuffer(m_ltBlasBitTrailGpu, m_blasBitTrailUploadStaging, m_ltBlasBitTrailGpuCapacity,
                  itemCount, 0, DXGI_FORMAT_R32_UINT,
                  LT_BLASBITTRAIL_SRV_SLOT, L"LT_BLASBitTrail_Refit");

        { void* p = nullptr;
          ThrowIfFailed(m_blasBitTrailUploadStaging->Map(0, &readRange, &p));
          memcpy(p, m_pendingBLASBitTrail.data(), itemBytes);
          m_blasBitTrailUploadStaging->Unmap(0, nullptr); }

        cmdList->CopyBufferRegion(m_ltBlasBitTrailGpu.Get(), 0, m_blasBitTrailUploadStaging.Get(), 0, itemBytes);

        auto b2 = CD3DX12_RESOURCE_BARRIER::Transition(m_ltBlasBitTrailGpu.Get(),
            D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_GENERIC_READ);
        cmdList->ResourceBarrier(1, &b2);
    }

    // ── BLASRanges (updated worldToLocal) ────────────────────────
    const UINT rangeCount = (UINT)m_pendingBLASRanges.size();
    if (rangeCount > 0) {
        const UINT rangeBytes = rangeCount * sizeof(lt::BlasRangeGpu);

        growBuffer(m_ltRangesGpu, m_rangesUploadStaging, m_ltRangesGpuCapacity,
                  rangeCount, sizeof(lt::BlasRangeGpu), DXGI_FORMAT_UNKNOWN,
                  LT_BLASRANGES_SRV_SLOT, L"LT_BLASRanges_Refit");

        { void* p = nullptr;
          ThrowIfFailed(m_rangesUploadStaging->Map(0, &readRange, &p));
          memcpy(p, m_pendingBLASRanges.data(), rangeBytes);
          m_rangesUploadStaging->Unmap(0, nullptr); }

        cmdList->CopyBufferRegion(m_ltRangesGpu.Get(), 0, m_rangesUploadStaging.Get(), 0, rangeBytes);

        auto b3 = CD3DX12_RESOURCE_BARRIER::Transition(m_ltRangesGpu.Get(),
            D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_GENERIC_READ);
        cmdList->ResourceBarrier(1, &b3);
    }

    m_pendingTLASUpload.clear();
    m_pendingBLASBitTrail.clear();
    m_pendingBLASRanges.clear();
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

    // Must match order in CreateShaderResourceHeap (slots 39-51)
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
}

//====================================
//RESIZE
//====================================
void Renderer::OnResize(UINT newWidth, UINT newHeight) {
    if (newWidth == 0 || newHeight == 0) return;
    if (newWidth == m_width && newHeight == m_height) return;

    // Disable DLSS-G before resize to avoid deadlock with present hook
    if (m_dlssG.enabled) {
        sl::DLSSGOptions gOpts{};
        gOpts.mode = sl::DLSSGMode::eOff;
        slDLSSGSetOptions(m_ctx.viewportHandle, gOpts);
    }

    m_ctx.WaitForGPU();
    // The CUDA aux stream runs training in parallel with raygen and is
    // not covered by WaitForGPU. We must drain it before reallocating
    // any NRC buffers below — otherwise an in-flight training kernel
    // could be reading inferenceOut / trainRecords while we free them.
    if (m_nrcReady) m_nrcNetwork.WaitIdle();

    m_width       = newWidth;
    m_height      = newHeight;
    m_aspectRatio = static_cast<float>(newWidth) / static_cast<float>(newHeight);

    // Resize swap chain, RTVs, depth stencil
    m_ctx.Resize(newWidth, newHeight);

    // Recreate all display-resolution-dependent GPU resources
    m_outputResource.Reset();
    m_permanentDataTexture.Reset();
    m_scratchPing.Reset();
    m_reservoirBuffer_3.Reset();
    m_reservoirBuffer_4.Reset();
    m_sampleBuffer_current.Reset();
    m_sampleBuffer_last.Reset();
    m_pathStateBuffer.Reset();
    m_spmisBuffer.Reset();        // recreated by CreateRaytracingOutputBuffer -> CreatePathStateBuffer
    for (int i = 0; i < MAX_STACKS; ++i)
        m_stackBuffers[i].Reset();

    CreateRaytracingOutputBuffer();
    CreateStreamingCompactionBuffers();

    // Recreate the resolution-dependent NRC buffers. InferenceIn/Out are
    // sized to 1·px (each pixel issues at most one cache-termination record;
    // the debug-view inference reuses the same buffer at slot = MapPixelID),
    // and PendingGI is one slot per pixel — both grow if the user enlarges the
    // window and raygen would otherwise scribble past the end of the old
    // buffer. TrainRecords and Counters are fixed-size and don't need
    // recreation. px is the TILE-ALIGNED pixel count (MapPixelID's index
    // space), not W·H — see the init-time NRC setup for why this matters.
    // (Was 2·px for a dead sharp-reflection record; that feature is removed.)
    if (m_nrcReady) {
        const uint32_t pixelCount = TileAlignedPx(newWidth, newHeight);
        m_nrcInferenceCapacity   = nrc::AlignBatch(pixelCount);
        //full cap for first post resize frame, shrinks again once a new count lands
        m_nrcDynamicInferenceCap = m_nrcInferenceCapacity;

        // Reset old buffers FIRST so the cudaInterop allocator releases
        // the backing D3D12 / CUDA imports before we ask it for new ones
        // — avoids a momentary 2× VRAM peak on large windows.
        m_nrcInferenceIn  = {};
        m_nrcInferenceOut = {};
        m_nrcPendingGI    = {};

        m_nrcInferenceIn  = m_cudaInterop.CreateBuffer(nrc::InferenceInputBytes (m_nrcInferenceCapacity), L"NRC_InferenceIn");
        m_nrcInferenceOut = m_cudaInterop.CreateBuffer(nrc::InferenceOutputBytes(m_nrcInferenceCapacity), L"NRC_InferenceOut");
        m_nrcPendingGI    = m_cudaInterop.CreateBuffer(nrc::PendingGIBytes      (pixelCount),             L"NRC_PendingGI");

        if (!m_nrcInferenceIn.resource || !m_nrcInferenceOut.resource || !m_nrcPendingGI.resource) {
            LOG(L"[NRC] Resize realloc failed — disabling NRC");
            m_nrcReady = false;
        }
    }

    // Recreate DLSS resources at new display resolution
    m_dlss.CreateResources(m_ctx.Device(), newWidth, newHeight);

    // Update descriptors for all resolution-dependent resources. Includes
    // a re-bind of NRC slots 58-60 (the resolution-dependent NRC UAVs)
    // so the descriptor table doesn't dangle on the freed resources.
    RebuildResolutionDependentDescriptors();
    RebuildDLSSDescriptors();
    RebuildNrcDescriptors();

    // Disable temporal reuse for 2 frames (old reservoirs are stale)
    m_dlssModeChangedFrames = 2;
    m_camera.ResetJitter();

    // Re-enable DLSS-G after resize
    if (m_dlssG.enabled) {
        sl::DLSSGOptions gOpts{};
        gOpts.mode = sl::DLSSGMode::eOn;
        gOpts.numFramesToGenerate = m_dlssG.framesToGenerate;
        gOpts.numBackBuffers      = m_ctx.BufferCount();
        gOpts.mvecDepthWidth      = m_dlss.RenderWidth();
        gOpts.mvecDepthHeight     = m_dlss.RenderHeight();
        gOpts.colorWidth          = newWidth;
        gOpts.colorHeight         = newHeight;
        slDLSSGSetOptions(m_ctx.viewportHandle, gOpts);
    }

    LOG(L"[Resize] " << newWidth << L"x" << newHeight);
}

// Re-creates the UAV descriptors for the resolution-dependent NRC
// buffers (slots 58, 59, 60). The fixed-size train records / counters
// at slots 61-62 are unchanged so we leave their descriptors alone.
// Mirrors the layout in CreateShaderResourceHeap (Renderer_Pipeline.cpp
// slots 58..62).
void Renderer::RebuildNrcDescriptors() {
    auto* dev = m_ctx.Device();
    const UINT inc = dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    auto writeRawUAVAt = [&](UINT slot, const CudaInterop::Buffer& b) {
        CD3DX12_CPU_DESCRIPTOR_HANDLE h(m_srvUavHeap->GetCPUDescriptorHandleForHeapStart(), slot, inc);
        if (b.resource) {
            D3D12_UNORDERED_ACCESS_VIEW_DESC ud = {};
            ud.ViewDimension      = D3D12_UAV_DIMENSION_BUFFER;
            ud.Format             = DXGI_FORMAT_R32_TYPELESS;
            ud.Buffer.Flags       = D3D12_BUFFER_UAV_FLAG_RAW;
            ud.Buffer.NumElements = (UINT)((b.sizeBytes + 3u) / 4u);
            dev->CreateUnorderedAccessView(b.resource.Get(), nullptr, &ud, h);
        } else {
            // Null-UAV fallback so the table stays valid if interop init failed.
            D3D12_UNORDERED_ACCESS_VIEW_DESC ud = {};
            ud.ViewDimension      = D3D12_UAV_DIMENSION_BUFFER;
            ud.Format             = DXGI_FORMAT_R32_TYPELESS;
            ud.Buffer.Flags       = D3D12_BUFFER_UAV_FLAG_RAW;
            ud.Buffer.NumElements = 1;
            dev->CreateUnorderedAccessView(nullptr, nullptr, &ud, h);
        }
    };

    writeRawUAVAt(58, m_nrcInferenceIn);   // u40
    writeRawUAVAt(59, m_nrcInferenceOut);  // u41
    writeRawUAVAt(60, m_nrcPendingGI);     // u42
    // Slots 61 (TrainRecords) and 62 (Counters) are fixed-size buffers
    // whose descriptors set up at init are still valid after a resize.
}

void Renderer::RebuildResolutionDependentDescriptors() {
    auto* dev = m_ctx.Device();
    const UINT inc = dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    auto writeUAVAt = [&](UINT slot, auto& res, auto writeFn) {
        CD3DX12_CPU_DESCRIPTOR_HANDLE h(m_srvUavHeap->GetCPUDescriptorHandleForHeapStart(), slot, inc);
        writeFn(h);
    };

    //tile-aligned (MapPixelID 8x4 swizzle) - must match the allocations in
    //CreateRaytracingOutputBuffer / CreatePathStateBuffer.
    UINT px = TileAlignedPx(GetWidth(), GetHeight());

    // Slot 0: output array UAV
    writeUAVAt(0, m_outputResource, [&](D3D12_CPU_DESCRIPTOR_HANDLE h) {
        D3D12_UNORDERED_ACCESS_VIEW_DESC ud = {};
        ud.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
        ud.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        ud.Texture2DArray.ArraySize = m_outputResource->GetDesc().DepthOrArraySize;
        dev->CreateUnorderedAccessView(m_outputResource.Get(), nullptr, &ud, h);
    });

    // Slot 1: permanent data UAV
    writeUAVAt(1, m_permanentDataTexture, [&](D3D12_CPU_DESCRIPTOR_HANDLE h) {
        D3D12_UNORDERED_ACCESS_VIEW_DESC ud = {};
        ud.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        ud.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
        dev->CreateUnorderedAccessView(m_permanentDataTexture.Get(), nullptr, &ud, h);
    });

    // Slots 10-15: reservoir / sample raw UAVs
    auto rawUAVAt = [&](UINT slot, ComPtr<ID3D12Resource>& res, UINT bytes) {
        CD3DX12_CPU_DESCRIPTOR_HANDLE h(m_srvUavHeap->GetCPUDescriptorHandleForHeapStart(), slot, inc);
        D3D12_UNORDERED_ACCESS_VIEW_DESC ud = {};
        ud.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        ud.Format = DXGI_FORMAT_R32_TYPELESS;
        ud.Buffer.NumElements = bytes / 4;
        ud.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
        dev->CreateUnorderedAccessView(res.Get(), nullptr, &ud, h);
    };
    // Slots 10/11 map to root-sig u2/u3 which no shader binds. Two null
    // UAVs keep the descriptor table layout intact without backing memory.
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
    rawUAVAt(12, m_reservoirBuffer_3,    px * sizeof(Reservoir_GI));
    rawUAVAt(13, m_reservoirBuffer_4,    px * sizeof(Reservoir_GI));
    rawUAVAt(14, m_sampleBuffer_current, px * sizeof(SampleData));
    rawUAVAt(15, m_sampleBuffer_last,    px * sizeof(SampleData));

    // Slot 18: scratch ping UAV
    writeUAVAt(18, m_scratchPing, [&](D3D12_CPU_DESCRIPTOR_HANDLE h) {
        D3D12_UNORDERED_ACCESS_VIEW_DESC ud = {};
        ud.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
        ud.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
        ud.Texture2DArray.ArraySize = 16;
        dev->CreateUnorderedAccessView(m_scratchPing.Get(), nullptr, &ud, h);
    });

    // Slot 19 maps to root-sig u9, unbound by every shader. Null UAV
    // preserves the descriptor table shape without backing memory.
    {
        D3D12_UNORDERED_ACCESS_VIEW_DESC ud = {};
        ud.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        ud.Format = DXGI_FORMAT_R32_TYPELESS;
        ud.Buffer.NumElements = 1;
        ud.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
        CD3DX12_CPU_DESCRIPTOR_HANDLE h(m_srvUavHeap->GetCPUDescriptorHandleForHeapStart(), 19, inc);
        dev->CreateUnorderedAccessView(nullptr, nullptr, &ud, h);
    }

    // Slot 32: path state UAV
    rawUAVAt(32, m_pathStateBuffer, px * kPathStateBytesPerPx);

    // Slots 33 (global counters), 34 (indirect args), 52-54 (sort buffers) are
    // bound to non-resolution-dependent buffers and are NOT recreated on resize
    // (see the if-not-null guards in CreateStreamingCompactionBuffers), so their
    // init-time descriptors stay valid - nothing to refresh here.

    // Slots 35-38: stack buffers UAV
    for (int s = 0; s < 4; ++s) {
        CD3DX12_CPU_DESCRIPTOR_HANDLE h(m_srvUavHeap->GetCPUDescriptorHandleForHeapStart(), 35 + s, inc);
        D3D12_UNORDERED_ACCESS_VIEW_DESC ud = {};
        ud.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        ud.Format = DXGI_FORMAT_UNKNOWN;
        ud.Buffer.NumElements = GetWidth() * GetHeight();
        ud.Buffer.StructureByteStride = 8;
        dev->CreateUnorderedAccessView(m_stackBuffers[s].Get(), nullptr, &ud, h);
    }
}

//====================================
//SAMPLE BUFFER PING-PONG
//====================================
//Raygen fully rewrites g_sample_current (u6) every frame and g_sample_last
//(u7) is read-only within the frame, so swapping the two bindings each frame
//replaces the merge pass's copySampleData (72 B/px of pure copy traffic).
//Safe to rewrite the live descriptor heap here: UpdateRenderer fully syncs
//CPU<->GPU (WaitForPreviousFrame) before calling this, so no in-flight frame
//references the old contents. Slots match WriteDescriptors /
//RebuildResolutionDependentDescriptors (14 = u6 current, 15 = u7 last).
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

//====================================
//PLANET CAMERA ADAPTER
//====================================
planet::CameraView Renderer::MakePlanetCamera() const {
    //derive the camera basis from the (floating-origin) view matrix, then add
    //sceneOriginWorld back so the planet sees absolute FP64 world coordinates.
    const XMMATRIX view    = m_camera.ViewMatrix();
    const XMMATRIX invView = XMMatrixInverse(nullptr, view);
    XMFLOAT3 pos, fwd, up;
    XMStoreFloat3(&pos, invView.r[3]);
    //invView.r[2] is the world-space direction of the view-space +Z axis. The
    //view matrix is right-handed (glm::lookAt + XMMatrixPerspectiveFovRH), so
    //the camera looks down -Z and +Z is BACKWARD — forward is its negation.
    //Without this negate the planet LOD frustum (Frustum::from_camera) points
    //behind the camera: descend() refines terrain off-screen and frustum-culls
    //the terrain in view, so the visible surface never receives fine chunks.
    XMStoreFloat3(&fwd, XMVector3Normalize(XMVectorNegate(invView.r[2])));
    XMStoreFloat3(&up,  XMVector3Normalize(invView.r[1]));
    const glm::vec3 origin = m_camera.getSceneOriginWorld();

    planet::CameraView cv;
    cv.position_world = { (double)pos.x + origin.x,
                          (double)pos.y + origin.y,
                          (double)pos.z + origin.z };
    //unified-TLAS origin: the same floating origin the scene TLAS uses, so
    //terrain and scene instances land in one consistent camera-local frame.
    cv.scene_origin = { origin.x, origin.y, origin.z };
    cv.forward    = { fwd.x, fwd.y, fwd.z };
    cv.up         = { up.x,  up.y,  up.z  };
    cv.fov_y      = m_camera.fovDegrees * 0.01745329252f;   // degrees -> radians
    cv.aspect     = m_aspectRatio;
    cv.near_plane = m_camera.nearPlane;
    cv.far_plane  = m_camera.farPlane;
    return cv;
}

//====================================
//PLANET SCENE INSTANCES (unified TLAS)
//====================================
//Convert m_scene.tlasInstances (XMMATRIX, sceneOrigin-relative) into the planet
//module's D3D12-only SceneInstanceDesc layout. Rebuilt every frame; the scene
//is small enough that the loop is well under the TLAS-build budget.
void Renderer::BuildPlanetSceneInstances() {
    const auto& src = m_scene.tlasInstances;
    m_planetSceneInstances.resize(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        planet::SceneInstanceDesc& d = m_planetSceneInstances[i];
        d.blas = src[i].blas ? src[i].blas->GetGPUVirtualAddress() : 0;
        //XMMATRIX (row-vector) -> DXR row-major 3x4: transpose, take the first
        //12 floats - exactly what nv_helpers TopLevelASGenerator does.
        XMFLOAT4X4 t;
        XMStoreFloat4x4(&t, XMMatrixTranspose(src[i].transform));
        memcpy(d.transform, &t, sizeof(float) * 12);
        d.instance_id     = (uint32_t)i;            // scene InstanceID == TLAS index
        d.hit_group_index = src[i].hitGroupContribution;
        d.flags           = (uint32_t)src[i].flags;
    }
}

//====================================
//RENDER
//====================================
void Renderer::RenderFrame() {
    using hrc = std::chrono::high_resolution_clock;
    static auto s_lastTime = hrc::now();
    static int s_frameCount = 0;
    auto t_frameStart = hrc::now();

    m_ctx.BeginFrame();

    // PLANET_INTEGRATION: Phase 4 stream pipeline - visibility + async tessellation.
    // Pass a MONOTONIC counter - NOT m_ctx.FrameIndex() (the cycling swapchain
    // back-buffer index): the chunk manager ages chunks by frame number for
    // retire hysteresis, which is meaningless against an index that wraps every
    // few frames.
    m_planet.begin_frame(m_planetFrame++, MakePlanetCamera());

    // PCL: render submit start
    slPCLSetMarker(sl::PCLMarker::eRenderSubmitStart, *m_ctx.frameToken);

    try {
        // PLANET_INTEGRATION: rebuild the scene TLAS instance list, convert it
        // for the planet module, then submit the planet copy/BLAS work plus the
        // per-frame unified TLAS rebuild (scene meshes + terrain + fallback).
        //
        // ORDERING IS LOAD-BEARING: this MUST run BEFORE PopulateCommandList().
        // The unified SceneBVH raygen traverses is rebuilt IN PLACE on the planet
        // COMPUTE queue here, and the graphics queue gates on it via
        // planetComputeFence (DeviceContext::ExecuteAndPresent AND, now,
        // CloseExecuteAndSignal). When NRC is enabled, the cuda:* ops split the
        // graphics command list mid-frame, so raygen's segment is submitted from
        // inside PopulateCommandList — before this build used to run. With the old
        // ordering, planetComputeAtSlot[frameIndex] still held the PREVIOUS frame's
        // value during those splits, so the raygen segment executed with no wait
        // on this frame's in-place TLAS rebuild -> raygen traversed a half-written
        // TLAS -> warp/SM-aligned tiles missed ALL geometry (blocky dark tiles,
        // only with the cache on; invisible with the cache off because then the
        // whole frame is one list gated solely by ExecuteAndPresent). Submitting
        // the build first makes planetComputeAtSlot[frameIndex] current for the
        // splits. submit_work targets the compute queue + CPU-side instance data
        // only, so it has no dependency on PopulateCommandList's graphics recording.
        m_scene.RebuildTLASInstanceList();
        BuildPlanetSceneInstances();
        //PLANET ROCKS: refresh the camera-following live set (hysteresis-gated
        //inside update) and hand it to the orchestrator for this frame's TLAS.
        if (m_planet.enabled() && !m_rockMeshIndices.empty()) {
            const planet::CameraView pcam = MakePlanetCamera();
            struct RockHeightAdapter : planet::IRockHeight {
                const planet::HeightmapCubemap* hm = nullptr;
                float sample_height_m(const planet::DVec3& d) const override {
                    return hm ? hm->sample(d, 0) : 0.0f;
                }
            } adapter;
            adapter.hm = &m_planet.heightmap();
            m_rockScatter.update(pcam.position_world, adapter);
            m_planet.set_rock_instances(m_rockScatter.live().data(),
                                        (uint32_t)m_rockScatter.live().size());
        }
        const uint32_t terrainHitGroup = (uint32_t)m_scene.instances.size() * 2u;
        m_planet.submit_work(m_planetSceneInstances.data(),
                             (uint32_t)m_planetSceneInstances.size(),
                             terrainHitGroup);

        auto t_popStart = hrc::now();
        PopulateCommandList();
        auto t_popEnd = hrc::now();
        m_frameStats.cpuPopulateMs = std::chrono::duration<float, std::milli>(t_popEnd - t_popStart).count();

        // PCL: render submit end
        slPCLSetMarker(sl::PCLMarker::eRenderSubmitEnd, *m_ctx.frameToken);

        // PCL: present start
        slPCLSetMarker(sl::PCLMarker::ePresentStart, *m_ctx.frameToken);
        m_ctx.ExecuteAndPresent();
        // PCL: present end
        slPCLSetMarker(sl::PCLMarker::ePresentEnd, *m_ctx.frameToken);
    } catch (...) {
        //A per-frame D3D failure that throws is, in practice, a lost device
        //surfacing through whatever call first noticed it (a failed Map,
        //Present, ...) — not where the GPU actually faulted. Dump DRED before
        //the exception unwinds through the Win32 WindowProc as an opaque
        //STATUS_FATAL_USER_CALLBACK_EXCEPTION: its breadcrumbs + page-fault
        //data name the faulting GPU op. CheckDeviceRemoved polls 1s for the
        //removal to latch, then terminates if the device is gone; if it
        //returns, the device is healthy and the exception is something else.
        dxdiag::CheckDeviceRemoved(m_ctx.Device(), 1000);
        throw;
    }

    // PLANET_INTEGRATION: advance built chunks to Ready; reclaim GPU pool slots
    m_planet.end_frame();

    // CPU = total CPU work across UpdateRenderer + PopulateCommandList
    m_frameStats.cpuFrameMs = m_frameStats.cpuUpdateMs + m_frameStats.cpuPopulateMs;

    // FPS display
    s_frameCount++;
    auto now = hrc::now();
    float elapsed = std::chrono::duration<float>(now - s_lastTime).count();
    if (elapsed >= 1.0f) {
        m_fps = s_frameCount / elapsed;
        std::wstringstream ss;
        ss << std::fixed << std::setprecision(2);
        if (m_dlssG.enabled && m_dlssG.framesToGenerate > 0) {
            float presentedFps = m_fps * (1 + m_dlssG.framesToGenerate);
            ss << L"Frame Time: " << 1000.0f / m_fps << L" ms ("
               << presentedFps << L" fps, " << m_fps << L" rendered + "
               << (1 + m_dlssG.framesToGenerate) << L"x FG)";
        } else {
            ss << L"Frame Time: " << 1000.0f / m_fps << L" ms (" << m_fps << L" fps)";
        }
        SetWindowTextW(Win32Application::GetHwnd(), ss.str().c_str());

        // PLANET_INTEGRATION: per-second planet readout. With async generation
        // the render-thread cost during a rebuild is just blas_record_cpu_ms;
        // plan + tess + alloc all happen on the worker pool. plan_ms is the
        // one-off plan-job duration (off-thread). pipe[p/r/b/B] = cells in
        // Pending / Ready / BLAS-recorded-fence-pending / Built. gpu_ms[] is
        // a fence-gated timestamp readback that lags ~4 frames.
        const auto& ps = m_planet.stats();
        std::wcout << L"[planet] built=" << ps.built
                   << L" leaves=" << ps.leaf_count
                   << L" cells=" << ps.cell_count
                   << L" tris=" << ps.triangle_count
                   << L" tlas=" << ps.tlas_instances
                   << L" rebuilding=" << ps.rebuilding
                   << L" dirty=" << ps.dirty_built << L"/" << ps.dirty_total
                   << L" est=" << ps.rebuild_frames_est << L"f"
                   << L" rec=" << ps.cells_recorded
                   << L" pipe[p=" << ps.cells_pending
                   << L" r="     << ps.cells_ready
                   << L" b="     << ps.cells_recorded_total
                   << L" B="     << ps.dirty_built << L"]"
                   << L" cpu_ms[blas_rec=" << ps.blas_record_cpu_ms
                   << L" plan="            << ps.plan_ms << L"]"
                   << L" gpu_ms[blas=" << ps.blas_gpu_ms
                   << L" tlas="        << ps.tlas_gpu_ms << L"]"
                   << L" DROPPED=" << ps.cells_dropped
                   << L" geo_free=" << ps.geo_free_leaves
                   << L" id_peak=" << ps.stable_id_peak
                   << std::endl;

        s_frameCount = 0;
        s_lastTime = now;
    }
}

//====================================
//DESTROY
//====================================
void Renderer::DestroyRenderer() {
    if (m_dlssG.enabled) {
        sl::DLSSGOptions gOpts{};
        gOpts.mode = sl::DLSSGMode::eOff;
        slDLSSGSetOptions(m_ctx.viewportHandle, gOpts);
        slFreeResources(sl::kFeatureDLSS_G, m_ctx.viewportHandle);
        m_dlssG.enabled = false;
    }
    m_editor.Shutdown();
    m_ctx.Shutdown();
}

//====================================
//PROCEDURAL MESH CREATION
//====================================
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

    // Source buffers are only used for BLAS build, see CreateAccelerationStructures.
    mesh.vertexBuffer.Reset();
    mesh.indexBuffer.Reset();

    UINT meshIndex = (UINT)m_scene.meshes.size();
    m_scene.meshes.push_back(std::move(mesh));
    LOG(L"[Engine] Created procedural mesh " << meshIndex << L" (mat " << matIdx << L")");
    return meshIndex;
}

//====================================
//MESH INSTANCE CREATION
//====================================
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

//====================================
//SCENE STRUCTURAL CHANGE
//====================================
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
      //PLANET: cover the terrain region too (and the buffer is the same fixed-
      //size resource, since CreateInstancePropertiesBuffer is idempotent in
      //terrain mode - so this just re-points the SRV at the unchanged buffer).
      sd.Buffer.NumElements = m_scene.instancePropsCount();
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

//====================================
//INPUT CAPTURE AND KEY HANDLERS
//====================================
bool Renderer::WantsKeyboard() const { return m_editor.IsVisible() && ImGui::GetIO().WantCaptureKeyboard; }
bool Renderer::WantsMouse() const    { return m_editor.IsVisible() && ImGui::GetIO().WantCaptureMouse; }
void Renderer::HandleKeyUp(UINT8 key) {
    if (key == 'C') m_currentDisplayLevel = (m_currentDisplayLevel + 1) % m_displayLevels.size();
    if (key == 'K') m_recorder.CaptureKeyframe(m_camera.Manipulator());
    if (key == VK_F1) m_editor.ToggleVisibility();
}

//====================================
//POPULATE COMMAND LIST
//====================================
//the frame's GPU work
void Renderer::PopulateCommandList() {
    auto* cmdList = m_ctx.CmdList();

    // Reallocate Streamline DLSS feature if mode changed (needs open command list)
    bool dlssResChanged = (m_dlssModeChangedFrames > 0);
    if (m_dlssModeChangedFrames > 0) {
        if (m_dlssModeChangedFrames == 2) {  // first frame after change: reallocate SL resources
            sl::Result fr = slFreeResources(sl::kFeatureDLSS_RR, m_ctx.viewportHandle);
            if (fr != sl::Result::eOk)
                std::wcout << L"[SL] slFreeResources failed: " << (int)fr << std::endl;
            sl::Result ar = slAllocateResources(cmdList, sl::kFeatureDLSS_RR, m_ctx.viewportHandle);
            if (ar != sl::Result::eOk)
                std::wcout << L"[SL] slAllocateResources failed: " << (int)ar << std::endl;
        }
        m_dlssModeChangedFrames--;
    }

    // Transition back buffer → render target
    {   auto b = CD3DX12_RESOURCE_BARRIER::Transition(
            m_ctx.BackBuffer(), D3D12_RESOURCE_STATE_PRESENT,
            D3D12_RESOURCE_STATE_RENDER_TARGET);
        cmdList->ResourceBarrier(1, &b); }

    auto rtv = m_ctx.CurrentRTV();
    auto dsv = m_ctx.DSV();
    cmdList->OMSetRenderTargets(1, &rtv, FALSE, &dsv);

    // PLANET_INTEGRATION: the unified TLAS (scene meshes + terrain chunks +
    // fallback) is rebuilt every frame by the planet StreamOrchestrator on the
    // async compute queue (RenderFrame -> m_planet.submit_work). The SceneBVH SRV
    // (heap slot 2) points at that TLAS, and the graphics queue waits the planet
    // compute fence before ray dispatch (DeviceContext::ExecuteAndPresent). The
    // renderer no longer builds a TLAS here - the old 3-tier dirty scheme is
    // superseded; RebuildTLASInstanceList still runs each frame (in RenderFrame)
    // to feed the orchestrator the current scene instances.
    m_frameStats.tlasWasRefit   = false;
    m_frameStats.tlasWasRebuilt = false;
    m_frameStats.tlasMs         = 0.0f;

    // Per-frame sky LUT bake (sun transmittance + cloud ambient probes).
    // Recorded BEFORE the main heap bind — it uses its own private heap and
    // root signature, and every consumer pass below reads the results as
    // SRVs at t49/t50.
    RecordSkyLUTBake(cmdList);

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

    // Precompute ReSTIR root constants.
    auto& rs = m_restirSettings;
    rs.tempMcapGI     = std::max(rs.tempMcapGI, 1);
    rs.spatCountMaxGI = std::clamp(rs.spatCountMaxGI, 1, 2);
    rs.spatCountMinGI = rs.spatCountMaxGI;
    rs.spatRadMaxGI   = std::max(rs.spatRadMaxGI, 4);
    rs.spatRadMinGI   = std::clamp(rs.spatRadMinGI, 4, rs.spatRadMaxGI);
    rs.spatTriesGI    = std::clamp(rs.spatTriesGI, 2, 16);

    // Neighbor rejection thresholds
    rs.rejNormalDot   = std::clamp(rs.rejNormalDot, 0.0f, 1.0f);
    rs.rejDistance    = std::max(rs.rejDistance, 0.001f);

    UINT rsConsts[42] = {};
    rsConsts[4]  = (UINT)rs.tempMcapGI;
    rsConsts[5]  = (UINT)rs.spatCountMaxGI;
    rsConsts[6]  = (UINT)rs.spatCountMinGI;
    rsConsts[7]  = (UINT)rs.spatRadMaxGI;
    rsConsts[8]  = (UINT)rs.spatRadMinGI;
    // When DLSS render resolution changes, the "last" reservoir/sample buffers
    // use a different SoA layout (numPx changes). Reading them with the new
    // layout yields garbage positions → NaN rays → GPU hang. Disable temporal
    // reuse for 2 frames so those buffers are never read with stale layout.
    // Flags() returns bit1=tempGI (0x2) and bit3=spatGI (0x8); mask off both
    // lower bits on DLSS res change to drop the temporal pass.
    // RS_FLAG_CLAMP_EMITTERS (0x100, must match Includes_v8.hlsli) is a high bit
    // clear of the ReSTIR bits; Pass_shading reads it to clamp emitter spikes.
    constexpr uint32_t RS_FLAG_CLAMP_EMITTERS = 0x100u;
    uint32_t baseFlags = dlssResChanged ? (rs.Flags() & ~3u) : rs.Flags();
    if (m_dlss.clampEmitterSpikes) baseFlags |= RS_FLAG_CLAMP_EMITTERS;
    rsConsts[9]  = baseFlags;
    memcpy(&rsConsts[10], &rs.reuseRoughnessMin, 4);
    memcpy(&rsConsts[11], &rs.reuseRoughnessMax, 4);
    rsConsts[12] = (UINT)rs.spatTriesGI;

    // Per-frame reuse-texture transforms (offset.xy + flag bits, per slot).
    // Sizes must match shaders' hardcoded values and InitReuseTextures.
    {
        const UINT kSizes[3] = { 254u, 230u, 210u };
        std::mt19937 rng(static_cast<uint32_t>(m_time) * 0x9E3779B9u + 1u);
        std::uniform_int_distribution<uint32_t> dist;
        for (int i = 0; i < 3; ++i) {
            rsConsts[13 + i * 3 + 0] = dist(rng) % kSizes[i];  // offset.x
            rsConsts[13 + i * 3 + 1] = dist(rng) % kSizes[i];  // offset.y
            rsConsts[13 + i * 3 + 2] = dist(rng) & 7u;         // flags (3 bits)
        }
    }

    // Neighbor rejection thresholds (slots 22-23)
    memcpy(&rsConsts[22], &rs.rejNormalDot, 4);
    memcpy(&rsConsts[23], &rs.rejDistance,  4);

    // SPMIS spatial-reuse params (slots 32-37; read by the Pass_spmis_* kernels).
    // Selected by RS_FLAG_SPMIS_SPATIAL (0x10). Clamp to safe ranges.
    rsConsts[32] = (UINT)std::clamp(rs.spmisReuseN,   1, 32);      // Ntilde (reuse_neighbor_count)
    rsConsts[33] = (UINT)std::clamp(rs.spmisRisN,     1, 32);      // inner-RIS candidate count
    rsConsts[34] = (UINT)std::clamp(rs.spmisMcap,     0, 100000);  // output M-cap (0 disables)
    rsConsts[35] = (UINT)std::clamp(rs.spmisTileSize, 1, 256);     // screen-space cell tile size (px)
    {
        const float jac = std::clamp(rs.spmisJacThreshold, 1.0f, 1000.0f);
        const float ns  = std::clamp(rs.spmisNormalSimCos, -1.0f, 1.0f);
        memcpy(&rsConsts[36], &jac, 4);   // reconnection-shift jacobian reject band
        memcpy(&rsConsts[37], &ns,  4);   // neighbor-similarity normal cone (cos)
    }

    // Path-trace termination knobs (slots 38-39; read by Pass_raygen_v8). Editor
    // caps both at 32. maxBounces floored at 2 so the primary (depth==1) always runs.
    rsConsts[38] = (UINT)std::clamp(rs.maxBounces,    2, 32);
    rsConsts[39] = (UINT)std::clamp(rs.rrStartDepth,  1, 32);
    rsConsts[40] = (UINT)std::clamp(rs.initialSamples, 1, 8);

    // SPMIS cell-search plane-distance rejection (slot 41; fraction of camera distance).
    {
        const float pd = std::max(rs.spmisPlaneDist, 0.0f);
        memcpy(&rsConsts[41], &pd, 4);
    }

    // NRC control constants (slots 24-27). NRC is only driving the
    // pipeline when the interop + tcnn stack initialised successfully —
    // we mask terrain enabled/training when m_nrcReady is false so the
    // fallback path stays exactly like the pre-NRC pipeline.
    //
    // Scene AABB is auto-recomputed every frame from mesh localAabbs +
    // live instance transforms. Cost is O(N_instances · 8) corner
    // transforms, a handful of microseconds even for dense scenes, so
    // dynamic content (moving instances, live-edited transforms) keeps
    // NRC's position normalization in lockstep with the actual world.
    {
        using namespace DirectX;
        XMVECTOR bMin = XMVectorReplicate(+FLT_MAX);
        XMVECTOR bMax = XMVectorReplicate(-FLT_MAX);
        bool anyBounds = false;
        for (const auto& inst : m_scene.instances) {
            if (inst.meshIndex >= m_scene.meshes.size()) continue;
            const MeshGPU& mesh = m_scene.meshes[inst.meshIndex];
            // Skip unset / empty meshes.
            if (mesh.localAabbMin.x > mesh.localAabbMax.x) continue;
            const XMFLOAT3 lm = mesh.localAabbMin;
            const XMFLOAT3 lM = mesh.localAabbMax;
            const XMFLOAT3 corners[8] = {
                {lm.x, lm.y, lm.z}, {lM.x, lm.y, lm.z},
                {lm.x, lM.y, lm.z}, {lM.x, lM.y, lm.z},
                {lm.x, lm.y, lM.z}, {lM.x, lm.y, lM.z},
                {lm.x, lM.y, lM.z}, {lM.x, lM.y, lM.z},
            };
            const XMMATRIX& w = inst.worldTransform;
            for (int k = 0; k < 8; ++k) {
                XMVECTOR c = XMLoadFloat3(&corners[k]);
                c = XMVector3Transform(c, w);
                bMin = XMVectorMin(bMin, c);
                bMax = XMVectorMax(bMax, c);
            }
            anyBounds = true;
        }

        if (anyBounds) {
            const XMVECTOR center  = XMVectorScale(XMVectorAdd(bMin, bMax), 0.5f);
            const XMVECTOR halfExt = XMVectorScale(XMVectorSubtract(bMax, bMin), 0.5f);
            const float ext = std::max({
                XMVectorGetX(halfExt),
                XMVectorGetY(halfExt),
                XMVectorGetZ(halfExt)
            });
            XMFLOAT3 c3; XMStoreFloat3(&c3, center);
            // FLOATING-ORIGIN FIX: inst.worldTransform is ABSOLUTE world (see
            // BuildXformsFromScene, which subtracts sceneOriginWorld to produce
            // the SHIFTED frame the GPU/TLAS actually use). raygen feeds
            // NrcNormalizePosition the SHIFTED hitPos (absolute -
            // sceneOriginWorld), so nrc_scene_center MUST live in that same
            // shifted frame. Without this subtraction the center is off by
            // sceneOriginWorld — the camera's 1 km origin snap — so every
            // position feature lands far outside [0,1], saturate()s to a single
            // HashGrid corner, the position encoding collapses, and the cache
            // emits spatially-constant / blocky L_s. Extent is translation-
            // invariant, so only the center needs shifting. (Floating origin was
            // added while NRC was disabled, so this site never got migrated.)
            m_nrcSettings.sceneCenter = {
                c3.x - m_scene.sceneOriginWorld.x,
                c3.y - m_scene.sceneOriginWorld.y,
                c3.z - m_scene.sceneOriginWorld.z,
            };
            // Floor at 1.0 so empty / single-point scenes don't divide
            // by zero in the shader normalization.
            m_nrcSettings.sceneExtent = std::max(ext, 1.0f);
        }

        // Auto-reinit on weight collapse. When training on long stretches of
        // near-zero targets (ultra-dark scenes, all-shadow areas) a plain-ReLU
        // net could fall into a dead-ReLU state where every prediction is
        // literally zero and gradients vanish so it can't recover on its own.
        // The hidden activation is now LeakyReLU (see BuildNetworkConfig), whose
        // 0.01 negative-slope keeps dead units learning, so this should be rare
        // -- but the detector stays as a backstop for any residual collapse.
        // The previous frame's inference output magnitude (sampled
        // by ScheduleInferenceOutSumReadback) is the canary: if it stays below
        // kCollapseMean for kCollapseFrames consecutive frames AND training is
        // on, queue a reinit. Cooldown gates the detector after a reinit so
        // we don't retrigger before the freshly seeded network has had a
        // chance to learn (LR=3e-3 + JIT warmup needs ~120 frames to recover
        // meaningful output magnitudes from zero EMA).
        if (m_nrcReady && m_nrcSettings.trainingEnabled && m_nrcSettings.enabled) {
            constexpr float    kCollapseMean   = 1.0e-6f;
            constexpr uint32_t kCollapseFrames = 60u;
            constexpr uint32_t kReinitCooldown = 240u;

            if (m_nrcReinitCooldown > 0u) {
                --m_nrcReinitCooldown;
                m_nrcCollapseConsecutive = 0u;
            } else {
                //mean < 0 is the "no readback yet" sentinel; leave counter alone
                const float mean = m_nrcNetwork.LastInferenceOutMagnitudeMean();
                if (mean >= 0.0f) {
                    if (mean < kCollapseMean) ++m_nrcCollapseConsecutive;
                    else                       m_nrcCollapseConsecutive = 0u;
                }
                if (m_nrcCollapseConsecutive >= kCollapseFrames) {
                    LOG(L"[NRC] Auto-reinit triggered: mean output magnitude "
                        << mean << L" stayed below " << kCollapseMean
                        << L" for " << m_nrcCollapseConsecutive << L" frames");
                    m_nrcSettings.requestReinit = true;
                    m_nrcReinitCooldown      = kReinitCooldown;
                    m_nrcCollapseConsecutive = 0u;
                }
            }
        } else {
            m_nrcCollapseConsecutive = 0u;
        }

        // Consume editor-requested weight reinit before any NRC work fires
        // on this frame. Network::ReinitWeights drains auxStream internally
        // and the follow-up cudaMemset on EMA forces a global device sync,
        // so the main interop stream's prior-frame work is also flushed by
        // the time the new weights land. Keep this BEFORE the adaptive-tile
        // block so the post-reset lastValidVertices=0 is the value the
        // tile-size feedback reads.
        if (m_nrcReady && m_nrcSettings.requestReinit) {
            const bool ok = m_nrcNetwork.ReinitWeights();
            LOG(L"[NRC] ReinitWeights " << (ok ? L"OK" : L"FAILED"));
            m_nrcTrainTileSide = nrc::kInitialTrainingTileSide;
            m_nrcSettings.requestReinit = false;
        }

        // Runtime training records target is FIXED across resolutions
        // (kFixedTrainingRecords ~= 32400, the 1080p budget). NRC stability
        // tuning was done at 1080p and higher-resolution scaling was making
        // the trainer adapt too fast / overshoot in dark scenes. The
        // adaptive tile side below grows with resolution to hit this
        // constant target instead.
        {
            uint32_t target = nrc::kFixedTrainingRecords;
            // Floor at one full SGD batch so the trainer always has something
            // to chew on at tiny window sizes / editor previews.
            const uint32_t floorRecords = nrc::kTrainingBatchesPerFrame * nrc::kBatchGranularity;
            if (target < floorRecords)                 target = floorRecords;
            if (target > nrc::kTrainingRecordsPerFrame) target = nrc::kTrainingRecordsPerFrame;
            m_nrcTrainRecordsTarget = target;
        }

        // Adaptive tile side per paper §3.5. With T = target records,
        // V = previous frame's actual records, and S = previous tile
        // side, the new side is S · sqrt(V/T) — vertex count scales
        // ~quadratically with 1/tile_side (training pixel density).
        // Damp by averaging with the old side so transient variance
        // (sky-dominant frames, big visibility changes) doesn't yank
        // the size around. First frame uses the initial value because
        // LastValidVertexCount returns 0 before TrainFrame ever ran.
        if (m_nrcReady && m_nrcSettings.enabled && m_nrcSettings.trainingEnabled) {
            const uint32_t prev = m_nrcNetwork.LastValidVertexCount();
            if (prev > 0u) {
                const float target  = (float)m_nrcTrainRecordsTarget;
                const float ratio   = (float)prev / target;
                const float scaled  = (float)m_nrcTrainTileSide * sqrtf(ratio);
                const float blended = 0.5f * (float)m_nrcTrainTileSide + 0.5f * scaled;
                int32_t s = (int32_t)(blended + 0.5f);
                if (s < (int32_t)nrc::kMinTrainingTileSide) s = (int32_t)nrc::kMinTrainingTileSide;
                if (s > (int32_t)nrc::kMaxTrainingTileSide) s = (int32_t)nrc::kMaxTrainingTileSide;
                m_nrcTrainTileSide = (uint32_t)s;
            }
        }

        //"Enable cache" is the master switch. Training, debug view and sharp
        //reflections are subsystems of the cache, so they only light up when
        //the cache itself is on, otherwise raygen keeps writing training and
        //debug records that nothing downstream consumes.
        const bool nrcActive = m_nrcReady && m_nrcSettings.enabled;
        uint32_t nrcFlags = 0u;
        if (nrcActive)                                   nrcFlags |= nrc::flags::kEnabled;
        if (nrcActive && m_nrcSettings.trainingEnabled)  nrcFlags |= nrc::flags::kTrain;
        if (nrcActive && m_nrcSettings.debugView)        nrcFlags |= nrc::flags::kDebugView;
        nrcFlags |= (m_nrcTrainTileSide & nrc::flags::kTileMask) << nrc::flags::kTileShift;
        rsConsts[24] = nrcFlags;
        memcpy(&rsConsts[25], &m_nrcSettings.areaSpreadC,      4);
        memcpy(&rsConsts[26], &m_nrcSettings.learningRateScale, 4);
        // Slot 27: dynamic inference cap. Sized from the most recently
        // harvested counter (async readback, one frame lag), padded by
        // 25% plus a small floor for camera cuts and tcnn batch
        // granularity. Capped at the static buffer capacity. Stays at
        // the full buffer cap until the first non zero count lands so
        // the very first frame, post resize, and post weight reinit all
        // run with headroom. Slot 27 also serves as the 16B alignment
        // pad for nrc_scene_center at slot 28, so its size is fixed.
        if (m_nrcReady) {
            uint32_t target = m_nrcInferenceCapacity;
            const uint32_t lastCount = m_nrcNetwork.LastInferenceCount();
            if (lastCount > 0u) {
                const uint32_t grown = lastCount + (lastCount >> 2) + 4096u;
                target = std::min(m_nrcInferenceCapacity, grown);
            }
            m_nrcDynamicInferenceCap = nrc::AlignBatch(target);
        } else {
            m_nrcDynamicInferenceCap = m_nrcInferenceCapacity;
        }
        rsConsts[27] = m_nrcDynamicInferenceCap;
        // 0.5 / halfExtent maps the scene AABB to exactly [0, 1]³ via
        // x_norm = (x - center) * scale_inv + 0.5. HashGrid's lookup
        // table is indexed in [0, 1] — anything outside that range
        // wraps via the hash function and produces nonsensical features.
        // (Old factor was 1.0 / halfExtent for [-0.5, 1.5] which the
        // periodic TriangleWave tolerated; HashGrid does not.)
        const float sceneScaleInv = 0.5f / m_nrcSettings.sceneExtent;
        memcpy(&rsConsts[28], &m_nrcSettings.sceneCenter.x, 4);
        memcpy(&rsConsts[29], &m_nrcSettings.sceneCenter.y, 4);
        memcpy(&rsConsts[30], &m_nrcSettings.sceneCenter.z, 4);
        memcpy(&rsConsts[31], &sceneScaleInv, 4);
    }

    auto setConsts = [&](UINT w, UINT h, UINT stackIn, UINT stackOut) {
        rsConsts[0] = w; rsConsts[1] = h; rsConsts[2] = stackIn; rsConsts[3] = stackOut;
        cmdList->SetComputeRoot32BitConstants(1, 42, rsConsts, 0);
        // SPMIS hash-grid buffer as a root UAV (u25) — see Includes_v8.hlsli / the
        // Pass_spmis_* kernels + raygen hash insertion. Bound for every main-root-sig
        // pass; shaders that don't reference it strip the binding (DXC).
        cmdList->SetComputeRootUnorderedAccessView(2, m_spmisBuffer->GetGPUVirtualAddress());
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
                m_camera.JitterX(), m_camera.JitterY(), m_camera.JitterFrame(),
                m_camera.fovDegrees, m_camera.nearPlane, m_camera.farPlane);

            // DLSS Frame Generation: set options every frame
            if (m_dlssG.available && m_dlssG.enabled) {
                sl::DLSSGOptions gOpts{};
                gOpts.mode                = sl::DLSSGMode::eOn;
                gOpts.numFramesToGenerate = m_dlssG.framesToGenerate;
                gOpts.numBackBuffers      = m_ctx.BufferCount();
                gOpts.mvecDepthWidth      = m_dlss.RenderWidth();
                gOpts.mvecDepthHeight     = m_dlss.RenderHeight();
                gOpts.colorWidth          = GetWidth();
                gOpts.colorHeight         = GetHeight();
                SL_CHECK(slDLSSGSetOptions(m_ctx.viewportHandle, gOpts));
            } else if (m_dlssG.available && !m_dlssG.enabled) {
                sl::DLSSGOptions gOpts{};
                gOpts.mode = sl::DLSSGMode::eOff;
                slDLSSGSetOptions(m_ctx.viewportHandle, gOpts);
            }

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

        case Stage::CudaOp:
        {
            if (!m_cudaInterop.IsReady() || !m_cudaFence.fence) break;
            auto it = m_cudaOps.find(p.file);
            if (it == m_cudaOps.end()) break;

            // shouldRun lets the registrar skip the entire interop round
            // trip when no actual CUDA work is needed this frame. The cmd
            // list close + ExecuteCommandLists + fence signal + CUDA wait
            // + CUDA signal + queue wait + cmd list reopen cycle costs
            // several ms of WDDM cross context overhead even with an
            // empty CUDA stream, so skipping it for known no-op ops
            // (debug inference when debug view is off etc.) is a clear
            // win.
            if (!it->second.shouldRun()) break;

            // D3D12 -> CUDA: flush current list and signal fence at value N.
            const UINT64 preVal  = ++m_cudaFenceValue;
            m_ctx.CloseExecuteAndSignal(m_cudaFence.fence.Get(), preVal);
            m_cudaInterop.CudaWait(m_cudaFence, preVal);

            // CUDA work (tcnn kernels) enqueued on the interop stream.
            it->second.fn();

            // CUDA -> D3D12: signal N+1 from CUDA, queue-wait, reopen list.
            const UINT64 postVal = ++m_cudaFenceValue;
            m_cudaInterop.CudaSignal(m_cudaFence, postVal);
            m_ctx.WaitAndReopen(m_cudaFence.fence.Get(), postVal);

            // cmdList was Reset — rebind the descriptor heap for subsequent passes.
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
    UINT sub   = D3D12CalcSubresource(0, layer, 0, 1, 4);
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

    // ── DLSS-G: tag resources after back buffer has final content ──
    if (m_dlssG.enabled) {
        constexpr D3D12_RESOURCE_STATES stateUAV     = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        constexpr D3D12_RESOURCE_STATES statePresent  = D3D12_RESOURCE_STATE_PRESENT;
        sl::Resource slFgDepth(sl::ResourceType::eTex2d, m_dlss.Depth(),       (uint32_t)stateUAV);
        sl::Resource slFgMVec (sl::ResourceType::eTex2d, m_dlss.MVec(),        (uint32_t)stateUAV);
        // HUDLessColor = back buffer (final tonemapped frame, not pre-postprocess DLSS output)
        // so optical flow runs on the clean displayed image, not raw HDR/noisy data
        sl::Resource slFgHud  (sl::ResourceType::eTex2d, m_ctx.BackBuffer(),   (uint32_t)statePresent);
        sl::Resource slFgBB   (sl::ResourceType::eTex2d, m_ctx.BackBuffer(),   (uint32_t)statePresent);

        sl::Extent renderExt { 0, 0, m_dlss.RenderWidth(),  m_dlss.RenderHeight()  };
        sl::Extent displayExt{ 0, 0, m_dlss.DisplayWidth(), m_dlss.DisplayHeight() };
        auto fgLife = sl::ResourceLifecycle::eValidUntilPresent;

        sl::ResourceTag fgTags[] = {
            { &slFgDepth, sl::kBufferTypeDepth,            fgLife, &renderExt  },
            { &slFgMVec,  sl::kBufferTypeMotionVectors,    fgLife, &renderExt  },
            { &slFgHud,   sl::kBufferTypeHUDLessColor,     fgLife, &displayExt },
            { &slFgBB,    sl::kBufferTypeBackbuffer,        fgLife, &displayExt },
            { nullptr,    sl::kBufferTypeUIColorAndAlpha,   fgLife, &displayExt },
        };
        // nullptr for cmdBuffer: docs say OK when all tags are eValidUntilPresent
        SL_CHECK(slSetTagForFrame(*m_ctx.frameToken, m_ctx.viewportHandle,
            fgTags, _countof(fgTags), nullptr));
    }
}

// Old input handlers removed — EngineApp handles input now.
