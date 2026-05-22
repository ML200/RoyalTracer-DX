//====================================
//PLANET - STREAM ORCHESTRATOR
//====================================

#include "stream_orchestrator.h"
#include "../Core/DeviceContext.h"
#include <chrono>
#include <iostream>
#include <stdexcept>

namespace {
//cells per batch when building the first generation synchronously at init.
constexpr uint32_t FIRST_BUILD_BATCH = 32;
//frames a retired generation is kept before release - long enough that no
//in-flight TLAS still references its BLASes.
constexpr uint32_t RETIRE_FRAMES = 8;

//row-major 3x4 translation-only transform in D3D12 instance-desc layout
inline void make_translation(float m[12], const planet::DVec3& t) {
    m[0]=1.f; m[1]=0.f; m[2]=0.f;  m[3]=(float)t.x;
    m[4]=0.f; m[5]=1.f; m[6]=0.f;  m[7]=(float)t.y;
    m[8]=0.f; m[9]=0.f; m[10]=1.f; m[11]=(float)t.z;
}
}

namespace planet {

//====================================
//INIT
//====================================
void StreamOrchestrator::init(ID3D12Device5* device, DeviceContext* ctx,
                              const StreamConfig& cfg) {
    m_device = device;
    m_ctx    = ctx;
    m_cfg    = cfg;

    m_heightmap.amplitude = cfg.heightmap_amplitude;
    m_heightmap.frequency = cfg.heightmap_frequency;

    //unified TLAS: every terrain cell + the scene-instance allowance.
    m_tlas.init(device, MAX_TERRAIN_CELLS + cfg.max_scene_instances);

    //terrain table: one TerrainSlotGPU per stable id, persistently-mapped upload heap.
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

    //GPU timestamp queries: one heap + one readback for the BLAS/TLAS GPU timings.
    {
        D3D12_QUERY_HEAP_DESC qhd = {};
        qhd.Type  = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
        qhd.Count = TS_RING * TS_PER_SLOT;
        if (FAILED(device->CreateQueryHeap(&qhd, IID_PPV_ARGS(&m_queryHeap))))
            throw std::runtime_error("planet: timestamp query-heap create failed");

        const uint64_t tsBytes = (uint64_t)TS_RING * TS_PER_SLOT * sizeof(uint64_t);
        //Enhanced-barriers devices ignore InitialResourceState for buffers and
        //warn about non-COMMON values; pass COMMON. Readback heaps still can't
        //transition - they stay effectively COMMON for the resource's lifetime,
        //which is what ResolveQueryData expects.
        m_tsReadback = create_buffer(device, tsBytes,
                                     D3D12_RESOURCE_FLAG_NONE,
                                     D3D12_RESOURCE_STATE_COMMON, HEAP_READBACK);
        void* p = nullptr;
        D3D12_RANGE full_read{ 0, (SIZE_T)tsBytes };
        if (FAILED(m_tsReadback->Map(0, &full_read, &p)))
            throw std::runtime_error("planet: timestamp readback Map failed");
        m_tsReadbackMapped = static_cast<uint64_t*>(p);

        //Timestamp frequency is per-queue. If the call fails, blas_gpu_ms /
        //tlas_gpu_ms stay 0 (the conversion guards on m_tsFreq).
        if (FAILED(ctx->PlanetComputeQueue()->GetTimestampFrequency(&m_tsFreq)))
            m_tsFreq = 0;
    }

    std::wcout << L"[planet] StreamOrchestrator ready (workers="
               << m_workers.thread_count() << L")" << std::endl;
}

//====================================
//PARAMS
//====================================
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

//assign each cell a stable terrain-table id by its node; a cell reused across a
//ping-pong swap keeps its id, so its ReSTIR/DLSS history survives.
void StreamOrchestrator::assign_stable_ids(Generation& g) {
    for (CellInstance& c : g.cells)
        c.stable_id = m_ids.get(c.node_id);
    m_ids.retain([&g](uint64_t node) { return g.cellSet.is_cell(node); });
}

//====================================
//BEGIN FRAME - reclaim, swap, trigger
//====================================
void StreamOrchestrator::begin_frame(uint32_t frame_index, const CameraView& cam) {
    m_frame       = frame_index;
    m_sceneOrigin = cam.scene_origin;

    //planet disabled (StreamConfig::enabled == false): skip all generation /
    //rebuild work. m_live stays empty, so record_tlas emits a scene-only
    //unified TLAS - the renderer runs as if there were no terrain at all.
    if (!m_cfg.enabled) return;

    //--- track camera velocity (EWMA of the per-frame world-space delta) ---
    if (m_havePrev) {
        const DVec3  dv = cam.position_world - m_prevCamPos;
        const double a  = 0.2;
        m_camVel.x += (dv.x - m_camVel.x) * a;
        m_camVel.y += (dv.y - m_camVel.y) * a;
        m_camVel.z += (dv.z - m_camVel.z) * a;
    }
    m_prevCamPos = cam.position_world;
    m_havePrev   = true;

    //--- reclaim: builder transient buffers, then retired generations ---
    if (m_builder.active())
        m_builder.reclaim(m_ctx->PlanetComputeCompleted());
    {
        size_t w = 0;
        for (size_t i = 0; i < m_retiring.size(); ++i)
            if (m_retiring[i].dropFrame > m_frame)
                m_retiring[w++] = std::move(m_retiring[i]);
        m_retiring.resize(w);
    }

    //--- first generation: build synchronously (an init-time stall is fine) ---
    if (!m_haveLive) {
        m_builder.begin(m_device, make_params(), cam, nullptr);
        while (!m_builder.done()) {
            m_ctx->ResetPlanetLists();
            const uint64_t copyVal = m_ctx->SubmitPlanetCopy();
            m_builder.step(m_device, m_heightmap, m_workers,
                           m_ctx->ComputeList(), FIRST_BUILD_BATCH);
            const uint64_t cv = m_ctx->SubmitPlanetCompute(copyVal);
            m_builder.on_submitted(cv);
            m_ctx->PlanetComputeCpuWait(cv);
            m_builder.reclaim(m_ctx->PlanetComputeCompleted());
        }
        m_live = m_builder.take();
        assign_stable_ids(m_live);
        m_haveLive   = true;
        m_liveCamPos = cam.position_world;
        std::wcout << L"[planet] first generation: " << m_live.cells.size()
                   << L" cells, " << m_live.leaf_count << L" leaves" << std::endl;
        return;
    }

    //--- a finished rebuild swaps in as the new LIVE generation ---
    if (m_builder.active() && m_builder.done()) {
        m_throughput.on_rebuild_done(m_frame - m_triggerFrame);
        m_retiring.push_back({ std::move(m_live), m_frame + RETIRE_FRAMES });
        m_live       = m_builder.take();
        assign_stable_ids(m_live);
        m_liveCamPos = m_rebuildTargetPos;
    }

    //--- trigger a rebuild when idle and the camera has drifted far enough ---
    if (!m_builder.active()) {
        const DVec3 d = cam.position_world - m_liveCamPos;
        if (length(d) > (double)m_cfg.rebuild_trigger_m) {
            //predict where the camera will be when this rebuild completes, and
            //aim the new LOD there - so it lands fresh, not already stale.
            DVec3 predicted = cam.position_world;
            if (m_cfg.predict) {
                DVec3        off    = m_camVel * (double)m_throughput.rebuild_frames;
                const double L      = length(off);
                const double maxoff = 4.0 * (double)m_cfg.rebuild_trigger_m;
                if (L > maxoff && L > 0.0) off = off * (maxoff / L);   // clamp a wild prediction
                predicted = cam.position_world + off;
            }
            CameraView predCam = cam;
            predCam.position_world = predicted;
            m_builder.begin(m_device, make_params(), predCam, &m_live);
            m_rebuildTargetPos = predicted;
            m_triggerFrame     = m_frame;
        }
    }
}

//====================================
//SUBMIT WORK - advance the rebuild, rebuild + submit the unified TLAS
//====================================
void StreamOrchestrator::submit_work(const SceneInstanceDesc* scene, uint32_t scene_count,
                                     uint32_t terrain_hit_group) {
    m_ctx->ResetPlanetLists();
    const uint64_t copyVal = m_ctx->SubmitPlanetCopy();          // empty copy list - just close it
    ID3D12GraphicsCommandList10* cl = m_ctx->ComputeList();

    //--- read back the timestamp slot we are about to overwrite ---
    //TS_RING > frames-in-flight, so the slot's fence has retired by now and the
    //readback is non-blocking. If pending=false (first uses) we just skip; if
    //the fence somehow hasn't retired (overflow), keep the prior datapoint.
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

    //T0: start of this frame's compute list (before BLAS recording).
    cl->EndQuery(m_queryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP,
                 slot * TS_PER_SLOT + 0);

    //--- advance an in-flight rebuild: tessellate + record this frame's budget
    //of dirty cells (their BLAS builds run async on the compute queue). ---
    uint32_t recorded = 0;
    if (m_builder.active() && !m_builder.all_recorded()) {
        const auto t0 = std::chrono::high_resolution_clock::now();
        recorded = m_builder.step(m_device, m_heightmap, m_workers, cl, m_cfg.build_budget);
        const auto t1 = std::chrono::high_resolution_clock::now();
        const float step_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
        m_throughput.on_step(step_ms);
        m_stats.step_cpu_ms        = step_ms;
        m_stats.tess_cpu_ms        = m_builder.tess_ms();
        m_stats.blas_record_cpu_ms = m_builder.blas_record_ms();
    } else {
        m_stats.step_cpu_ms        = 0.0f;
        m_stats.tess_cpu_ms        = 0.0f;
        m_stats.blas_record_cpu_ms = 0.0f;
    }
    m_stats.cells_recorded = recorded;

    //T1: after BLAS recording, before TLAS. (T1 - T0) = BLAS GPU time.
    cl->EndQuery(m_queryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP,
                 slot * TS_PER_SLOT + 1);

    //the unified TLAS always renders the LIVE generation. The compute queue is
    //submitted every frame: the renderer's graphics queue waits its fence.
    record_tlas(scene, scene_count, terrain_hit_group, cl);

    //T2: after TLAS. (T2 - T1) = TLAS GPU time.
    cl->EndQuery(m_queryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP,
                 slot * TS_PER_SLOT + 2);

    //Resolve the 3 timestamps into this slot of the readback buffer. The
    //readback is in COPY_DEST permanently (readback-heap rule) - no transitions.
    cl->ResolveQueryData(m_queryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP,
                         slot * TS_PER_SLOT, TS_PER_SLOT,
                         m_tsReadback.Get(),
                         (uint64_t)slot * TS_PER_SLOT * sizeof(uint64_t));

    const uint64_t cv = m_ctx->SubmitPlanetCompute(copyVal);
    if (recorded > 0) m_builder.on_submitted(cv);

    //park this slot for fence-gated readback next time we come round.
    m_tsRing[slot] = TsSlot{ cv, true };
    m_tsWrite++;

    m_stats.built          = m_haveLive;
    m_stats.rebuilding     = m_builder.active();
    m_stats.leaf_count     = m_live.leaf_count;
    m_stats.cell_count     = (uint32_t)m_live.cells.size();
    m_stats.triangle_count = m_live.triangle_count;
    m_stats.tlas_instances = m_tlas.instance_count();
    m_stats.dirty_total    = m_builder.active() ? m_builder.dirty_total() : 0;
    m_stats.dirty_built    = m_builder.active() ? m_builder.dirty_built() : 0;
    m_stats.rebuild_frames_est  = m_throughput.rebuild_frames;
    m_stats.step_ms             = m_throughput.step_ms;
    m_stats.last_rebuild_frames = m_throughput.last_rebuild_frames;
}

//====================================
//RECORD TLAS - scene meshes + every LIVE terrain cell
//====================================
void StreamOrchestrator::record_tlas(const SceneInstanceDesc* scene, uint32_t scene_count,
                                     uint32_t terrain_hit_group,
                                     ID3D12GraphicsCommandList4* compute_cl) {
    m_tlas.begin();

    //scene mesh instances - transforms already scene-origin-relative
    for (uint32_t i = 0; i < scene_count; ++i) {
        const SceneInstanceDesc& s = scene[i];
        m_tlas.add_instance(s.blas, s.transform, s.instance_id, s.hit_group_index,
                            (D3D12_RAYTRACING_INSTANCE_FLAGS)s.flags);
    }

    for (uint32_t i = 0; i < MAX_TERRAIN_CELLS; ++i) m_curNode[i] = INVALID_NODE;

    //one TLAS instance per LIVE cell. InstanceID = base + the cell's stable id
    //(stable across ping-pong swaps). The cell geometry is cell-local, so the
    //instance transform is a pure translation by the cell anchor.
    for (const CellInstance& c : m_live.cells) {
        if (c.stable_id >= MAX_TERRAIN_CELLS) continue;          // fail-soft
        float xform[12];
        make_translation(xform, c.anchor_world - m_sceneOrigin);
        m_tlas.add_instance(c.blas_va, xform, TERRAIN_INSTANCE_BASE + c.stable_id,
                            terrain_hit_group, D3D12_RAYTRACING_INSTANCE_FLAG_FORCE_OPAQUE);
        m_curNode[c.stable_id] = c.node_id;
    }

    //publish the terrain table: per slot the current cell node, and 'changed'
    //if a different cell took that slot since last frame (a swap reshuffled it).
    for (uint32_t i = 0; i < MAX_TERRAIN_CELLS; ++i) {
        m_terrainTableMapped[i].node_lo = (uint32_t)(m_curNode[i] & 0xFFFFFFFFull);
        m_terrainTableMapped[i].node_hi = (uint32_t)(m_curNode[i] >> 32);
        m_terrainTableMapped[i].changed = (m_curNode[i] != m_terrainNodePrev[i]) ? 1u : 0u;
        m_terrainNodePrev[i] = m_curNode[i];
    }

    m_tlas.build(compute_cl);
}

//====================================
//END FRAME
//====================================
void StreamOrchestrator::end_frame() {
    //reclaim + swap happen in begin_frame; nothing to do here.
}

} // namespace planet
