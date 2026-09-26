#include "stdafx.h"
#include "Renderer.h"
#include "minecraft/mc_materials.h"
#include "minecraft/mc_omm_bake.h"
#include "minecraft/mc_zip.h"
#include "planet/blas_pool.h"
#include "planet/worker_pool.h"
#include <chrono>
#include <filesystem>

bool Renderer::LoadMinecraftWorld(const mc::MinecraftWorldConfig& cfg, const DirectX::XMMATRIX& placement) {
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    m_mcConfig = cfg;
    if (m_planet.enabled()) {
        LOG(L"[mc] the planet terrain keeps the geometry buffers in an upload heap; disable it to load a Minecraft "
            L"world");
        return false;
    }
    try {
        planet::WorkerPool pool;

        auto world = std::make_unique<mc::World>();
        mc::WorldLoadConfig wl;
        wl.worldDir = cfg.worldDir;
        std::string err;
        if (!world->load(wl, &pool, &err)) {
            LOG(L"[mc] world load failed: " << std::wstring(err.begin(), err.end()));
            return false;
        }

        // Archives outlive the resource stack's borrowed pointers.
        std::vector<std::unique_ptr<mc::ZipArchive>> archives;
        mc::ResourceStack stack;
        for (const std::string& packPath : cfg.resourcePacks) {
            auto z = std::make_unique<mc::ZipArchive>();
            if (z->open(packPath, &err)) {
                LOG(L"[mc] resource pack: " << std::wstring(packPath.begin(), packPath.end()) << L" ("
                                            << z->entry_count() << L" entries)");
                stack.push(z.get());
                archives.push_back(std::move(z));
            } else {
                LOG(L"[mc] resource pack skipped: " << std::wstring(err.begin(), err.end()));
            }
        }
        const std::string jar = cfg.jarPath.empty() ? mc::find_minecraft_jar() : cfg.jarPath;
        {
            auto z = std::make_unique<mc::ZipArchive>();
            if (!jar.empty() && z->open(jar, &err)) {
                LOG(L"[mc] block models + textures from " << std::wstring(jar.begin(), jar.end()));
                stack.push(z.get());
                archives.push_back(std::move(z));
            } else {
                LOG(L"[mc] no vanilla jar found (" << std::wstring(err.begin(), err.end())
                                                   << L"); blocks without pack models render as missing textures");
            }
        }

        std::vector<TextureData> textures;
        mc::MaterialBuilder mb;
        mc::MaterialBuildStats ms;
        const int texBase = (int)(BINDLESS_HEAP_START + m_scene.totalBindlessTextures);
        if (m_scene.materialNames.size() < m_scene.materials.size())
            m_scene.materialNames.resize(m_scene.materials.size());
        mb.lodOpaqueCoverage = cfg.lodOpaqueCoverage;
        mb.opaqueLeaves = cfg.opaqueLeaves;
        mb.build(world->registry(), stack, m_scene.materials, m_scene.materialNames, textures, texBase, ms, &err);
        {
            auto flushFn = [this]() { m_ctx.FlushAndReset(); };
            AssetLoader::CreateBindlessTextures(textures, (UINT)texBase, L"Minecraft", m_ctx.Device(), m_ctx.CmdList(),
                                                m_scene.bindlessGpuTextures, flushFn);

            if (m_scene.totalBindlessTextures == 0) {
                m_scene.bindlessAlbedoBase = BINDLESS_HEAP_START;
                m_scene.bindlessNormalBase = BINDLESS_HEAP_START;
                m_scene.bindlessRmaBase = BINDLESS_HEAP_START;
            }
            m_scene.totalBindlessTextures += (UINT)textures.size();
        }

        // Block-face OMMs, baked once before meshing.
        if (cfg.opacityMicromaps) {
            const auto tb = clock::now();
            std::vector<mc::OmmBakeTri> tris;
            mc::enumerate_omm_triangles(world->registry(), tris);
            OmmBakeResult bake;
            std::string oerr;
            m_vxOmmTable = mc::OmmTable{};
            if (mc::bake_omm_table(world->registry(), tris, m_vxOmmTable, bake, &oerr) && !bake.ommDescs.empty()) {
                m_vxOmm = OmmBuilder::BuildGPU(bake, m_ctx.Device(), m_ctx.CmdList());
                m_ctx.FlushAndReset();
                m_vxOmmIndices = planet::create_buffer(
                    m_ctx.Device(), (uint64_t)cfg.streamer.ommIndexCapacity * sizeof(int32_t), D3D12_RESOURCE_FLAG_NONE,
                    D3D12_RESOURCE_STATE_GENERIC_READ, planet::HEAP_UPLOAD);
                LOG(L"[mc] opacity micromaps: "
                    << tris.size() << L" face triangles -> " << bake.ommDescs.size() << L" micromaps, "
                    << (bake.rawData.size() >> 10) << L" KB, "
                    << std::chrono::duration<double>(clock::now() - tb).count() << L" s"
                    << (oerr.empty() ? L"" : L" (" + std::wstring(oerr.begin(), oerr.end()) + L")"));
            } else {
                LOG(L"[mc] opacity micromaps skipped: " << std::wstring(oerr.begin(), oerr.end()));
            }
        }

        world->build_lod(&pool, cfg.lodDecorMaxLevel);

        m_mcWorld = std::move(world);

        mc::Placement place;
        {
            DirectX::XMFLOAT4X4 m;
            DirectX::XMStoreFloat4x4(&m, placement);
            place.set(&m.m[0][0]);
        }

        // Ocean's full reservation comes from the world's budget (32-bit view).
        mc::StreamerConfig streamer = cfg.streamer;
        if (m_ocean.Enabled()) {
            constexpr uint32_t kOceanVerts = (uint32_t)(OCEAN_MAX_TILES * OCEAN_TILE_VERTS);
            constexpr uint32_t kOceanIndices = (uint32_t)(OCEAN_MAX_TILES * OCEAN_TILE_INDICES);
            streamer.vertexCapacity -= std::min(streamer.vertexCapacity / 2u, kOceanVerts);
            streamer.indexCapacity -= std::min(streamer.indexCapacity / 2u, kOceanIndices);
            LOG(L"[mc] the wave surface shares the scene's geometry buffers; world budget "
                << (streamer.vertexCapacity >> 20) << L"M vertices / " << (streamer.indexCapacity >> 20)
                << L"M indices");
        }
        m_voxels.init(m_ctx.Device(), &m_ctx, m_mcWorld.get(), streamer);

        // Sea only where the world holds water.
        if (m_ocean.Enabled()) {
            const auto t = clock::now();
            m_voxels.water_coverage_mut().build(*m_mcWorld, place);
            m_ocean.SetCoverage(&m_voxels.water_coverage());
            const auto& cov = m_voxels.water_coverage();
            LOG(L"[mc] water coverage: " << cov.columns_with_water() << L" of " << cov.columns_total()
                << L" section columns hold water ("
                << (cov.columns_total() ? 100.0 * cov.columns_with_water() / cov.columns_total() : 0.0)
                << L"%), " << std::chrono::duration<double>(clock::now() - t).count() << L" s");
        }

        m_voxels.set_placement(place);

        {
            const double lowB[3] = {0.0, (double)m_mcWorld->store().min_section_y(0) * 16.0 - 16.0, 0.0};
            double lowS[3];
            place.to_scene(lowB, lowS);
            m_camera.skyGroundY = (float)lowS[1];
        }

        const mc::LevelInfo& li = m_mcWorld->level();
        if (cfg.spawnCamera && li.hasSpawn) {
            const double eyeB[3] = {(double)li.spawnX + 0.5, (double)li.spawnY + cfg.cameraHeight,
                                    (double)li.spawnZ + 0.5};
            const double northB[3] = {0.0, -0.35, -1.0}, upB[3] = {0.0, 1.0, 0.0};
            double eyeS[3], northS[3], upS[3];
            place.to_scene(eyeB, eyeS);
            place.dir_to_scene(northB, northS);
            place.dir_to_scene(upB, upS);
            const glm::vec3 eye((float)eyeS[0], (float)eyeS[1], (float)eyeS[2]);
            const glm::vec3 center = eye + glm::vec3((float)northS[0], (float)northS[1], (float)northS[2]);
            nv_helpers_dx12::CameraManip.setLookat(
                eye, center, glm::normalize(glm::vec3((float)upS[0], (float)upS[1], (float)upS[2])));
        }
        if (cfg.pointFilter)
            m_integratorSettings.texturePointFilter = 1;
        if (cfg.flySpeed > 0.0f && m_flyCam)
            m_flyCam->moveSpeed = cfg.flySpeed;

        const double secs = std::chrono::duration<double>(clock::now() - t0).count();
        LOG(L"[mc] world ready: " << m_mcWorld->stats().chunks << L" chunks, " << m_mcWorld->registry().count()
                                  << L" block states, " << ms.materials << L" materials, " << textures.size()
                                  << L" textures, " << secs << L" s");
        return true;
    } catch (const std::exception& e) {
        LOG(L"[mc] world load failed: " << std::wstring(e.what(), e.what() + strlen(e.what())));
        m_mcWorld.reset();
        return false;
    }
}

// Streamed lights follow the scene's light records.
void Renderer::BindVoxelLights(ID3D12GraphicsCommandList* cmdList) {
    if (!m_voxels.enabled())
        return;
    auto* dev = m_ctx.Device();
    const mc::StreamerConfig& sc = m_voxels.config();
    const auto& gpu = m_lightTree.GetGpu();
    auto elems = [](ID3D12Resource* r, UINT64 stride) -> UINT { return r ? (UINT)(r->GetDesc().Width / stride) : 0u; };
    ID3D12Resource* srcRecords = m_ownedEmissiveGpu ? m_ownedEmissiveGpu.Get() : m_scene.emissiveTrianglesBuffer.Get();
    const UINT sceneRecords =
        std::min((UINT)m_scene.emissiveTriangles.size(), elems(srcRecords, sizeof(LightTriangle)));
    const UINT nodeStride = lt::LightBLASNodeStride(m_lightTreeCompact);
    const UINT sceneNodes = elems(gpu.BLASNodes.Get(), nodeStride);
    const UINT sceneLeaves = elems(gpu.LeafTriIndex.Get(), sizeof(uint32_t));
    const UINT sceneTrails = elems(gpu.TriBitTrail.Get(), sizeof(lt::LightTreeTrail));
    const UINT needRec = std::max({sceneRecords, sceneLeaves, sceneTrails, 1u});
    const bool recreate = !m_vxLightRecords || needRec > m_vxSceneRecordCap || sceneNodes > m_vxSceneNodeCap ||
                          m_vxLightNodeStride != nodeStride;
    if (recreate) {
        // Freed once in-flight frames finish.
        ComPtr<ID3D12Resource>* old[4] = {&m_vxLightRecords, &m_vxLightNodes, &m_vxLightLeaf, &m_vxLightTrails};
        for (ComPtr<ID3D12Resource>* r : old)
            if (*r)
                m_vxLightRetired.push_back({*r, m_time});
        m_vxSceneRecordCap = std::max(4096u, needRec * 2u);
        m_vxSceneNodeCap = std::max(8192u, sceneNodes * 2u);
        m_vxLightNodeStride = nodeStride;
        auto make = [&](ComPtr<ID3D12Resource>& res, UINT64 bytes, const wchar_t* name) {
            auto hp = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
            auto desc = CD3DX12_RESOURCE_DESC::Buffer(bytes);
            ThrowIfFailed(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON,
                                                       nullptr, IID_PPV_ARGS(&res)));
            res->SetName(name);
        };
        const UINT64 recElems = (UINT64)m_vxSceneRecordCap + sc.lightRecordCapacity;
        make(m_vxLightRecords, recElems * sizeof(LightTriangle), L"MinecraftLightRecords");
        make(m_vxLightNodes, ((UINT64)m_vxSceneNodeCap + sc.lightNodeCapacity) * nodeStride,
             L"MinecraftLightNodes");
        make(m_vxLightLeaf, recElems * sizeof(uint32_t), L"MinecraftLightLeafIndex");
        make(m_vxLightTrails, recElems * sizeof(lt::LightTreeTrail), L"MinecraftLightTrails");
        LOG(L"[mc] light buffers: scene part " << m_vxSceneRecordCap << L" records / " << m_vxSceneNodeCap
                                               << L" nodes, voxel regions " << sc.lightRecordCapacity << L" records / "
                                               << sc.lightNodeCapacity << L" nodes");
    }

    auto copy = [&](ID3D12Resource* dst, ID3D12Resource* src, UINT64 bytes) {
        if (!src || !bytes)
            return;
        cmdList->CopyBufferRegion(dst, 0, src, 0, bytes);
        auto b = CD3DX12_RESOURCE_BARRIER::Transition(dst, D3D12_RESOURCE_STATE_COPY_DEST,
                                                      D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        cmdList->ResourceBarrier(1, &b);
    };
    copy(m_vxLightRecords.Get(), srcRecords, (UINT64)sceneRecords * sizeof(LightTriangle));
    copy(m_vxLightNodes.Get(), gpu.BLASNodes.Get(), (UINT64)sceneNodes * nodeStride);
    copy(m_vxLightLeaf.Get(), gpu.LeafTriIndex.Get(), (UINT64)sceneLeaves * sizeof(uint32_t));
    copy(m_vxLightTrails.Get(), gpu.TriBitTrail.Get(), (UINT64)sceneTrails * sizeof(lt::LightTreeTrail));

    const UINT inc = dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    auto srv = [&](UINT slot, ID3D12Resource* res, UINT count, UINT stride, DXGI_FORMAT fmt) {
        CD3DX12_CPU_DESCRIPTOR_HANDLE h(m_srvUavHeap->GetCPUDescriptorHandleForHeapStart(), (INT)slot, inc);
        D3D12_SHADER_RESOURCE_VIEW_DESC sd{};
        sd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        sd.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        sd.Buffer.NumElements = count;
        if (stride) {
            sd.Format = DXGI_FORMAT_UNKNOWN;
            sd.Buffer.StructureByteStride = stride;
        } else
            sd.Format = fmt;
        dev->CreateShaderResourceView(res, &sd, h);
    };
    const UINT recCount = m_vxSceneRecordCap + sc.lightRecordCapacity;
    srv(EMISSIVE_TRI_SRV_SLOT, m_vxLightRecords.Get(), recCount, sizeof(LightTriangle), DXGI_FORMAT_UNKNOWN);
    srv(21u, m_vxLightNodes.Get(), (UINT)(((UINT64)m_vxSceneNodeCap + sc.lightNodeCapacity) * nodeStride / 16u), 16u,
        DXGI_FORMAT_UNKNOWN);
    srv(23u, m_vxLightLeaf.Get(), recCount, 0u, DXGI_FORMAT_R32_UINT);
    srv(26u, m_vxLightTrails.Get(), recCount, 0u, DXGI_FORMAT_R32G32_UINT);

    if (recreate) {
        mc::LightBinding lb;
        lb.records = m_vxLightRecords.Get();
        lb.recordBase = m_vxSceneRecordCap;
        lb.recordCapacity = sc.lightRecordCapacity;
        lb.nodes = m_vxLightNodes.Get();
        lb.nodeBase = m_vxSceneNodeCap;
        lb.nodeCapacity = sc.lightNodeCapacity;
        lb.leafIndex = m_vxLightLeaf.Get();
        lb.trails = m_vxLightTrails.Get();
        lb.slotBase = (uint32_t)m_scene.lightInstances.size();
        lb.compactNodes = m_lightTreeCompact;
        m_voxels.bind_lights(sc.lights ? lb : mc::LightBinding{});
    } else {
        m_voxels.set_light_slot_base((uint32_t)m_scene.lightInstances.size());
    }
    m_liveVoxelLightLeaves = 0;
}
