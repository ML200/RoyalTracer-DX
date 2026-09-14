#include "stream_orchestrator.h"
#include "../Core/DeviceContext.h"
#include "../Common.h"
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <unordered_set>

namespace {
constexpr uint32_t RETIRE_FRAMES = 8;

// Packs a translation-only transform in D3D12's row-major layout.
inline void make_translation(float m[12], const planet::DVec3& t) {
    m[0]=1.f; m[1]=0.f; m[2]=0.f;  m[3]=(float)t.x;
    m[4]=0.f; m[5]=1.f; m[6]=0.f;  m[7]=(float)t.y;
    m[8]=0.f; m[9]=0.f; m[10]=1.f; m[11]=(float)t.z;
}

// Converts cell-local geometry metadata into renderer instance properties.
inline void fill_terrain_props(InstanceProperties& p, const planet::CellInstance& c,
                               const planet::DVec3& origin,
                               uint32_t matIDBase, uint32_t triLightBase) {
    using namespace DirectX;
    const double dx  = c.anchor_world.x - origin.x;
    const double dy  = c.anchor_world.y - origin.y;
    const double dz  = c.anchor_world.z - origin.z;
    const XMMATRIX M    = XMMatrixTranslation( (float)dx,  (float)dy,  (float)dz);
    const XMMATRIX Minv = XMMatrixTranslation(-(float)dx, -(float)dy, -(float)dz);
    p.objectToWorld        = MakeFloat3x4(M);
    p.objectToWorldInverse = MakeFloat3x4(Minv);
    p.objectToWorldNormal  = MakeFloat3x4(XMMatrixIdentity());
    p.prevObjectToWorld    = MakeFloat3x4(M);
    p.indexBase      = c.idx_base_elems;
    p.vertexBase     = c.vtx_base_elems;
    p.materialBase   = matIDBase;
    p.triToLightBase = triLightBase;
    p.opaqueTriCount = c.geo_tri_count;
    const uint8_t face = planet::unpack_node_id(c.node_id).face;
    p._pad[0] = 1u + (uint32_t)(face & 0x7u);
    p._pad[1] = 1u;
    p.lightSlot = 0xFFFFFFFFu;
}

inline void fill_rock_props(InstanceProperties& p,
                            const planet::RockInstance& r,
                            const planet::StreamOrchestrator::RockVariantGPU& var,
                            const planet::DVec3& origin,
                            uint32_t matIDBase, uint32_t triLightBase) {
    using namespace DirectX;
    const float tx = (float)(r.anchor_world.x - origin.x);
    const float ty = (float)(r.anchor_world.y - origin.y);
    const float tz = (float)(r.anchor_world.z - origin.z);
    const XMMATRIX M = XMMatrixSet(
        r.rot_scale[0], r.rot_scale[3], r.rot_scale[6], 0.0f,
        r.rot_scale[1], r.rot_scale[4], r.rot_scale[7], 0.0f,
        r.rot_scale[2], r.rot_scale[5], r.rot_scale[8], 0.0f,
        tx,             ty,             tz,             1.0f);
    const XMMATRIX Minv = XMMatrixInverse(nullptr, M);
    p.objectToWorld        = MakeFloat3x4(M);
    p.objectToWorldInverse = MakeFloat3x4(Minv);
    p.objectToWorldNormal  = MakeFloat3x4(M);
    p.prevObjectToWorld    = MakeFloat3x4(M);
    p.indexBase      = var.indexBase;
    p.vertexBase     = var.vertexBase;
    p.materialBase   = matIDBase;
    p.triToLightBase = triLightBase;
    p.opaqueTriCount = var.triCount;
    p._pad[0] = 0u;
    p._pad[1] = 1u;
    p.lightSlot = 0xFFFFFFFFu;
}
}

namespace planet {

void StreamOrchestrator::init(ID3D12Device5* device, DeviceContext* ctx,
                              const StreamConfig& cfg) {
    m_device = device;
    m_ctx    = ctx;
    m_cfg    = cfg;

    if (cfg.enabled) {
        if (!m_heightmap.load(cfg.heightmap_dir)) {
            std::wcout << L"[planet] heightmap load failed; planet will render flat"
                       << std::endl;
        }
    }

    m_tlas.init(device, cfg.max_scene_instances +
        (cfg.enabled ? MAX_TERRAIN_CELLS + MAX_ROCK_INSTANCES : 0));

    m_terrainTable = create_buffer(device, (uint64_t)MAX_TERRAIN_CELLS * sizeof(TerrainSlotGPU),
                                   D3D12_RESOURCE_FLAG_NONE,
                                   D3D12_RESOURCE_STATE_GENERIC_READ, HEAP_UPLOAD);
    {
        void* mapped = nullptr;
        D3D12_RANGE no_read{ 0, 0 };
        if (FAILED(m_terrainTable->Map(0, &no_read, &mapped)))
            throw std::runtime_error("planet: terrain-table Map failed");
        m_terrainTableMapped = static_cast<TerrainSlotGPU*>(mapped);
    }
    for (uint32_t i = 0; i < MAX_TERRAIN_CELLS; ++i) {
        m_terrainTableMapped[i].node_lo = (uint32_t)(INVALID_NODE & 0xFFFFFFFFull);
        m_terrainTableMapped[i].node_hi = (uint32_t)(INVALID_NODE >> 32);
        m_terrainTableMapped[i].changed = 0;
        m_terrainNodePrev[i] = INVALID_NODE;
    }

    {
        D3D12_QUERY_HEAP_DESC qhd = {};
        qhd.Type  = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
        qhd.Count = TS_RING * TS_PER_SLOT;
        if (FAILED(device->CreateQueryHeap(&qhd, IID_PPV_ARGS(&m_queryHeap))))
            throw std::runtime_error("planet: timestamp query-heap create failed");

        const uint64_t tsBytes = (uint64_t)TS_RING * TS_PER_SLOT * sizeof(uint64_t);
        m_tsReadback = create_buffer(device, tsBytes,
                                     D3D12_RESOURCE_FLAG_NONE,
                                     D3D12_RESOURCE_STATE_COMMON, HEAP_READBACK);
        void* p = nullptr;
        D3D12_RANGE full_read{ 0, (SIZE_T)tsBytes };
        if (FAILED(m_tsReadback->Map(0, &full_read, &p)))
            throw std::runtime_error("planet: timestamp readback Map failed");
        m_tsReadbackMapped = static_cast<uint64_t*>(p);

        if (FAILED(ctx->PlanetComputeQueue()->GetTimestampFrequency(&m_tsFreq)))
            m_tsFreq = 0;
    }

    std::wcout << L"[planet] StreamOrchestrator ready (workers="
               << m_workers.thread_count() << L")" << std::endl;
}

void StreamOrchestrator::set_rock_variants(uint32_t propsBase,
                                           const std::vector<RockVariantGPU>& variants) {
    m_rockPropsBase = propsBase;
    m_rockVariants  = variants;
}

void StreamOrchestrator::set_rock_instances(const RockInstance* insts, uint32_t count) {
    if (insts && count) m_rockInstances.assign(insts, insts + count);
    else                m_rockInstances.clear();
}

void StreamOrchestrator::bind_geometry(ID3D12Resource* combinedVtx, uint8_t* vtxMapped,
                                       ID3D12Resource* combinedIdx, uint8_t* idxMapped,
                                       ID3D12Resource* instanceProps,
                                       uint32_t sceneVertexCount, uint32_t sceneIndexCount,
                                       uint32_t combinedVertexCount, uint32_t terrainPropsBase,
                                       uint32_t terrainLeafSlots, uint32_t terrainMatIDBase,
                                       uint32_t terrainTriLightBase) {
    m_geoPool.init(sceneVertexCount, sceneIndexCount, terrainLeafSlots);
    m_instanceProps       = instanceProps;
    m_terrainPropsBase    = terrainPropsBase;
    m_terrainMatIDBase    = terrainMatIDBase;
    m_terrainTriLightBase = terrainTriLightBase;
    m_builder.set_geometry(&m_geoPool, vtxMapped, idxMapped,
                           combinedVtx->GetGPUVirtualAddress(),
                           combinedIdx->GetGPUVirtualAddress(),
                           combinedVertexCount);
}

GenerationParams StreamOrchestrator::make_params() const {
    GenerationParams gp;
    gp.quadtree.planet           = m_cfg.planet;
    gp.quadtree.min_lod          = m_cfg.min_lod;
    gp.quadtree.max_lod          = m_cfg.max_lod;
    gp.quadtree.max_leaves       = m_cfg.max_triangles / MAX_CHUNK_TRIS;
    gp.cells.planet              = m_cfg.planet;
    gp.cells.max_leaves_per_cell = m_cfg.max_leaves_per_cell;
    gp.cells.max_cell_radius_m   = m_cfg.max_cell_radius_m;
    return gp;
}

void StreamOrchestrator::assign_stable_ids(Generation& g) {
    std::unordered_set<uint64_t> liveNodes;
    liveNodes.reserve(g.cells.size());
    for (CellInstance& c : g.cells) {
        if (c.node_id == INVALID_NODE) continue;
        c.stable_id = m_ids.get(c.node_id);
        liveNodes.insert(c.node_id);
    }
    m_ids.retain([&liveNodes](uint64_t node) { return liveNodes.count(node) != 0; });
}

// Advances generation state, chooses visible cells, and queues GPU work.
void StreamOrchestrator::begin_frame(uint32_t frame_index, const CameraView& cam) {
    m_frame       = frame_index;
    m_sceneOrigin = cam.scene_origin;

    if (!m_cfg.enabled) return;

    if (m_havePrev) {
        const DVec3  dv = cam.position_world - m_prevCamPos;
        const double a  = 0.2;
        m_camVel.x += (dv.x - m_camVel.x) * a;
        m_camVel.y += (dv.y - m_camVel.y) * a;
        m_camVel.z += (dv.z - m_camVel.z) * a;
    }
    m_prevCamPos = cam.position_world;
    m_havePrev   = true;

    if (m_builder.active())
        m_builder.reclaim(m_ctx->PlanetComputeCompleted());
    {
        size_t w = 0;
        for (size_t i = 0; i < m_retiring.size(); ++i)
            if (m_retiring[i].dropFrame > m_frame)
                m_retiring[w++] = std::move(m_retiring[i]);
        m_retiring.resize(w);
    }

    if (!m_haveLive) {
        m_builder.begin(m_device, make_params(), cam, nullptr, m_heightmap, m_workers);
        while (!m_builder.done()) {
            m_builder.poll();
            m_ctx->ResetPlanetLists();
            const uint64_t copyVal = m_ctx->SubmitPlanetCopy();
            const uint32_t recorded = m_builder.record_ready_blas(m_ctx->ComputeList(), 0u);
            const uint64_t cv       = m_ctx->SubmitPlanetCompute(copyVal);
            if (recorded > 0) m_builder.on_submitted(cv);
            m_ctx->PlanetComputeCpuWait(cv);
            m_builder.reclaim(m_ctx->PlanetComputeCompleted());
            if (recorded == 0)
                std::this_thread::sleep_for(std::chrono::microseconds(200));
        }
        m_live = m_builder.take();
        assign_stable_ids(m_live);
        m_haveLive   = true;
        m_liveCamPos = cam.position_world;
        std::wcout << L"[planet] first generation: " << m_live.cells.size()
                   << L" cells, " << m_live.leaf_count << L" leaves" << std::endl;
        return;
    }

    m_builder.poll();

    if (m_builder.active() && m_builder.done()) {
        m_throughput.on_rebuild_done(m_frame - m_triggerFrame);
        m_retiring.push_back({ std::move(m_live), m_frame + RETIRE_FRAMES });
        m_live       = m_builder.take();
        assign_stable_ids(m_live);
        m_liveCamPos = m_rebuildTargetPos;
    }

    if (!m_builder.active()) {
        const DVec3 d = cam.position_world - m_liveCamPos;
        if (length(d) > (double)m_cfg.rebuild_trigger_m) {
            DVec3 predicted = cam.position_world;
            if (m_cfg.predict) {
                DVec3        off    = m_camVel * (double)m_throughput.rebuild_frames;
                const double L      = length(off);
                const double maxoff = 256.0;
                if (L > maxoff && L > 0.0) off = off * (maxoff / L);
                predicted = cam.position_world + off;
            }
            CameraView predCam = cam;
            predCam.position_world = predicted;
            m_builder.begin(m_device, make_params(), predCam, &m_live,
                            m_heightmap, m_workers);
            m_rebuildTargetPos = predicted;
            m_triggerFrame     = m_frame;
        }
    }
}

// Submits uploads and BLAS work while preserving queue fence ordering.
void StreamOrchestrator::submit_work(const SceneInstanceDesc* scene, uint32_t scene_count,
                                     uint32_t terrain_hit_group, uint32_t external_hit_group) {
    m_ctx->ResetPlanetLists();
    ID3D12GraphicsCommandList10* cl = m_ctx->ComputeList();
    if (m_external) m_external->record_gpu_work(m_ctx->CopyList(), cl);
    const uint64_t copyVal = m_ctx->SubmitPlanetCopy();

    const uint32_t slot = m_tsWrite % TS_RING;
    if (m_tsRing[slot].pending &&
        m_ctx->PlanetComputeCompleted() >= m_tsRing[slot].fence) {
        const uint64_t* p = m_tsReadbackMapped + (size_t)slot * TS_PER_SLOT;
        const uint64_t  t_start = p[0];
        const uint64_t  t_blas  = p[1];
        const uint64_t  t_tlas  = p[2];
        const double inv_freq_ms = (m_tsFreq != 0) ? (1000.0 / double(m_tsFreq)) : 0.0;
        m_stats.blas_gpu_ms = float(double(t_blas - t_start) * inv_freq_ms);
        m_stats.tlas_gpu_ms = float(double(t_tlas - t_blas ) * inv_freq_ms);
        m_tsRing[slot].pending = false;
    }

    cl->EndQuery(m_queryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP,
                 slot * TS_PER_SLOT + 0);

    uint32_t recorded = 0;
    if (m_builder.active()) {
        const auto t0 = std::chrono::high_resolution_clock::now();
        recorded = m_builder.record_ready_blas(cl, m_cfg.build_budget);
        const auto t1 = std::chrono::high_resolution_clock::now();
        const float blas_rec_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
        m_throughput.on_step(blas_rec_ms);
        m_stats.blas_record_cpu_ms = blas_rec_ms;
    } else {
        m_stats.blas_record_cpu_ms = 0.0f;
    }
    m_stats.cells_recorded = recorded;

    cl->EndQuery(m_queryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP,
                 slot * TS_PER_SLOT + 1);

    record_tlas(scene, scene_count, terrain_hit_group, external_hit_group, cl);

    cl->EndQuery(m_queryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP,
                 slot * TS_PER_SLOT + 2);

    cl->ResolveQueryData(m_queryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP,
                         slot * TS_PER_SLOT, TS_PER_SLOT,
                         m_tsReadback.Get(),
                         (uint64_t)slot * TS_PER_SLOT * sizeof(uint64_t));

    const uint64_t cv = m_ctx->SubmitPlanetCompute(copyVal);
    if (recorded > 0) m_builder.on_submitted(cv);
    if (m_external) m_external->on_submitted(copyVal, cv);

    m_tsRing[slot] = TsSlot{ cv, true };
    m_tsWrite++;

    m_stats.built          = m_haveLive;
    m_stats.rebuilding     = m_builder.active();
    m_stats.leaf_count     = m_live.leaf_count;
    m_stats.cell_count     = (uint32_t)m_live.cells.size();
    m_stats.triangle_count = m_live.triangle_count;
    m_stats.tlas_instances = m_tlas.instance_count();
    m_stats.geo_free_leaves = m_geoPool.free_leaves();
    m_stats.stable_id_peak  = m_ids.peak();
    if (m_builder.active()) {
        m_stats.dirty_total          = m_builder.dirty_total();
        m_stats.dirty_built           = m_builder.dirty_built();
        m_stats.cells_pending         = m_builder.dirty_tessellating();
        m_stats.cells_ready           = m_builder.dirty_ready();
        m_stats.cells_recorded_total  = m_builder.dirty_recorded();
        m_stats.plan_ms               = m_builder.plan_ms();
    } else {
        m_stats.dirty_total           = 0;
        m_stats.dirty_built           = 0;
        m_stats.cells_pending         = 0;
        m_stats.cells_ready           = 0;
        m_stats.cells_recorded_total  = 0;
    }
    m_stats.rebuild_frames_est  = m_throughput.rebuild_frames;
    m_stats.step_ms             = m_throughput.step_ms;
    m_stats.last_rebuild_frames = m_throughput.last_rebuild_frames;
}

void StreamOrchestrator::reserve_scene_instances(uint32_t count) {
    if (m_cfg.enabled && count > m_cfg.max_scene_instances) {
        if (m_instanceProps)
            throw std::runtime_error("Scene instances exceed the fixed terrain instance range");
        m_cfg.max_scene_instances = count;
    }
    m_tlas.reserve(count + (m_cfg.enabled ? MAX_TERRAIN_CELLS + MAX_ROCK_INSTANCES : 0) + external_capacity());
}

void StreamOrchestrator::record_tlas(const SceneInstanceDesc* scene, uint32_t scene_count,
                                     uint32_t terrain_hit_group, uint32_t external_hit_group,
                                     ID3D12GraphicsCommandList4* compute_cl) {
    if (m_cfg.enabled && scene_count > m_cfg.max_scene_instances)
        throw std::runtime_error("Scene instances overlap the reserved terrain instance range");
    m_tlas.begin(scene_count + (uint32_t)m_live.cells.size() + (uint32_t)m_rockInstances.size() + external_capacity());

    for (uint32_t i = 0; i < scene_count; ++i) {
        const SceneInstanceDesc& s = scene[i];
        m_tlas.add_instance(s.blas, s.transform, s.instance_id, s.hit_group_index,
                            (D3D12_RAYTRACING_INSTANCE_FLAGS)s.flags);
    }

    InstanceProperties* props = nullptr;
    if (m_instanceProps) {
        D3D12_RANGE nr{ 0, 0 };
        if (FAILED(m_instanceProps->Map(0, &nr, (void**)&props))) props = nullptr;
    }

    uint32_t dropped = 0;
    for (const CellInstance& c : m_live.cells) {
        if (c.node_id == INVALID_NODE) continue;
        if (c.stable_id >= MAX_TERRAIN_CELLS) { ++dropped; continue; }
        if (c.blas_va == 0)                   { ++dropped; continue; }
        const uint32_t instID = m_terrainPropsBase + c.stable_id;
        float xform[12];
        make_translation(xform, c.anchor_world - m_sceneOrigin);
        m_tlas.add_instance(c.blas_va, xform, instID,
                            terrain_hit_group, D3D12_RAYTRACING_INSTANCE_FLAG_FORCE_OPAQUE);
        if (props)
            fill_terrain_props(props[instID], c, m_sceneOrigin,
                               m_terrainMatIDBase, m_terrainTriLightBase);
    }

    for (const RockInstance& r : m_rockInstances) {
        if (r.stable_id >= MAX_ROCK_INSTANCES)    continue;
        if (r.variant   >= m_rockVariants.size()) continue;
        const RockVariantGPU& var = m_rockVariants[r.variant];
        if (var.blas_va == 0)                     continue;
        const uint32_t instID = m_rockPropsBase + r.stable_id;
        const float tx = (float)(r.anchor_world.x - m_sceneOrigin.x);
        const float ty = (float)(r.anchor_world.y - m_sceneOrigin.y);
        const float tz = (float)(r.anchor_world.z - m_sceneOrigin.z);
        const float xform[12] = {
            r.rot_scale[0], r.rot_scale[1], r.rot_scale[2], tx,
            r.rot_scale[3], r.rot_scale[4], r.rot_scale[5], ty,
            r.rot_scale[6], r.rot_scale[7], r.rot_scale[8], tz,
        };
        m_tlas.add_instance(var.blas_va, xform, instID,
                            terrain_hit_group, D3D12_RAYTRACING_INSTANCE_FLAG_FORCE_OPAQUE);
        if (props)
            fill_rock_props(props[instID], r, var, m_sceneOrigin,
                            m_terrainMatIDBase, m_terrainTriLightBase);
    }

    bool externalForce = false;
    if (m_external) m_external->append_instances(m_tlas, props, m_sceneOrigin, external_hit_group, externalForce);

    if (props) m_instanceProps->Unmap(0, nullptr);
    m_stats.cells_dropped = dropped;

    m_tlas.build(compute_cl, m_cfg.enabled || externalForce);
}

void StreamOrchestrator::end_frame() {
}

}
