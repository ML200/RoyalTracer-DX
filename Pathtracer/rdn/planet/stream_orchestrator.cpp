//====================================
//PLANET - STREAM ORCHESTRATOR
//====================================

#include "stream_orchestrator.h"
#include "../Core/DeviceContext.h"
#include "../Common.h"          // InstanceProperties + DirectXMath (terrain instance props)
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <unordered_set>

namespace {
//frames a retired generation is kept before release - long enough that no
//in-flight TLAS still references its BLASes.
constexpr uint32_t RETIRE_FRAMES = 8;

//row-major 3x4 translation-only transform in D3D12 instance-desc layout
inline void make_translation(float m[12], const planet::DVec3& t) {
    m[0]=1.f; m[1]=0.f; m[2]=0.f;  m[3]=(float)t.x;
    m[4]=0.f; m[5]=1.f; m[6]=0.f;  m[7]=(float)t.y;
    m[8]=0.f; m[9]=0.f; m[10]=1.f; m[11]=(float)t.z;
}

//Fill a terrain cell's InstanceProperties. The cell geometry is cell-local
//(relative to anchor_world), so object->world is a pure translation by
//(anchor - sceneOrigin) - same as the TLAS instance transform - and the normal
//matrix is identity. index/vertex bases point at the cell's slot in the
//combined buffer; material/light bases at the shared terrain regions;
//opaqueTriCount = K*MAX_CHUNK_TRIS (single opaque geometry).
//
//prev == cur: terrain is WORLD-STATIC, and the camera's prevView is already
//re-expressed in the CURRENT floating-origin frame (the camera adjusts it on a
//snap - exactly how Scene handles static meshes), so the previous-frame
//object->world is the current one. Using (anchor - prevOrigin) here instead put
//prev in the OLD origin frame, skewing motion vectors by the snap delta on
//every floating-origin snap - rare when crawling, frequent when flying fast,
//which is why fast flight ghosted/mis-loaded freshly-streamed cells.
inline void fill_terrain_props(InstanceProperties& p, const planet::CellInstance& c,
                               const planet::DVec3& origin,
                               uint32_t matIDBase, uint32_t triLightBase) {
    using namespace DirectX;
    const double dx  = c.anchor_world.x - origin.x;
    const double dy  = c.anchor_world.y - origin.y;
    const double dz  = c.anchor_world.z - origin.z;
    const XMMATRIX M    = XMMatrixTranslation( (float)dx,  (float)dy,  (float)dz);
    const XMMATRIX Minv = XMMatrixTranslation(-(float)dx, -(float)dy, -(float)dz);
    p.objectToWorld            = M;
    p.objectToWorldInverse     = Minv;
    p.prevObjectToWorld        = M;
    p.prevObjectToWorldInverse = Minv;
    p.objectToWorldNormal      = XMMatrixIdentity();
    p.prevObjectToWorldNormal  = XMMatrixIdentity();
    p.indexBase      = c.idx_base_elems;
    p.vertexBase     = c.vtx_base_elems;
    p.materialBase   = matIDBase;
    p.triToLightBase = triLightBase;
    p.opaqueTriCount = c.geo_tri_count;
    //_pad[0] = terrain marker + cube face: 0 = scene mesh, 1..6 = terrain on
    //face 0..5. Scene InstanceProperties zero-init _pad and never write it, so
    //the raygen tint lookup keys on this alone (no instID-range assumption).
    //A cell is a single-face subtree, so node_id's face is the whole cell's.
    const uint8_t face = planet::unpack_node_id(c.node_id).face;
    p._pad[0] = 1u + (uint32_t)(face & 0x7u);
    p._pad[1] = 1u;   // flat per-triangle shading (planet terrain)
    p._pad[2] = 0u;
}

//PLANET ROCKS: fill a scene-style InstanceProperties for one scatter rock.
//Rocks shade as scene meshes (_pad[0]=0 -> no terrain tint / detail-normal) and
//reuse the terrain shared material + tri-light regions (both filled with the
//flat terrain material / no-light sentinel), so they render as rough terrain-
//coloured boulders. objectToWorld is built so HLSL's mul(objectToWorld,
//float4(local,1)) == rot_scale*local + (anchor-origin), i.e. it matches the
//TLAS instance 3x4 that record_tlas emits for the same rock.
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
    p.objectToWorld            = M;
    p.objectToWorldInverse     = Minv;
    p.prevObjectToWorld        = M;
    p.prevObjectToWorldInverse = Minv;
    p.objectToWorldNormal      = M;   // uniform scale -> the shading path renormalizes
    p.prevObjectToWorldNormal  = M;
    p.indexBase      = var.indexBase;
    p.vertexBase     = var.vertexBase;
    p.materialBase   = matIDBase;     // reuse terrain's shared flat-material region
    p.triToLightBase = triLightBase;  // reuse terrain's no-light region
    p.opaqueTriCount = var.triCount;
    p._pad[0] = 0u;                   // scene mesh: no terrain tint / detail-normal
    p._pad[1] = 1u;                   // but DO flat per-triangle shade (planet rock)
    p._pad[2] = 0u;
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

    //Load the baked cubemap heightmap from the runtime include/terrain/ copy.
    //Failure leaves the heightmap empty - the tessellator will see all zeros,
    //which renders a perfect sphere instead of crashing.
    // PLANET DISABLED 2026-06-10: skip the terrain-data load entirely when the
    // planet is off. The baked terrain set (include/terrain/: elevation_face*.r32,
    // surface_color/normal PNGs, cloud_offset_face*.r32, manifest.json) is too
    // large to commit, so it is no longer shipped. With the planet disabled
    // nothing consumes the heightmap anyway — generation is gated off and the
    // InitTerrain* GPU uploads are commented out in Renderer — so gating the
    // load on cfg.enabled keeps the runtime free of any ./terrain dependency
    // (no disk probe, no "load failed" log). Re-enabling the planet restores
    // this load, which still degrades gracefully to a flat sphere if the data
    // files happen to be absent.
    if (cfg.enabled) {
        if (!m_heightmap.load(cfg.heightmap_dir)) {
            std::wcout << L"[planet] heightmap load failed; planet will render flat"
                       << std::endl;
        }
    }

    //unified TLAS: every terrain cell + the scene-instance allowance.
    m_tlas.init(device, MAX_TERRAIN_CELLS + cfg.max_scene_instances + MAX_ROCK_INSTANCES);

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

void StreamOrchestrator::set_rock_variants(uint32_t propsBase,
                                           const std::vector<RockVariantGPU>& variants) {
    m_rockPropsBase = propsBase;
    m_rockVariants  = variants;
}

void StreamOrchestrator::set_rock_instances(const RockInstance* insts, uint32_t count) {
    if (insts && count) m_rockInstances.assign(insts, insts + count);
    else                m_rockInstances.clear();
}

//====================================
//BIND GEOMETRY - wire the unified scene+terrain buffers (after scene load)
//====================================
void StreamOrchestrator::bind_geometry(ID3D12Resource* combinedVtx, uint8_t* vtxMapped,
                                       ID3D12Resource* combinedIdx, uint8_t* idxMapped,
                                       ID3D12Resource* instanceProps,
                                       uint32_t sceneVertexCount, uint32_t sceneIndexCount,
                                       uint32_t combinedVertexCount, uint32_t terrainPropsBase,
                                       uint32_t terrainLeafSlots, uint32_t terrainMatIDBase,
                                       uint32_t terrainTriLightBase) {
    //the terrain region begins right after the scene data in the combined
    //buffers (vertex base = sceneVertexCount, index base = sceneIndexCount).
    m_geoPool.init(sceneVertexCount, sceneIndexCount, terrainLeafSlots);
    m_instanceProps       = instanceProps;
    m_terrainPropsBase    = terrainPropsBase;   // FIXED (= scene capacity), not the live scene count
    m_terrainMatIDBase    = terrainMatIDBase;
    m_terrainTriLightBase = terrainTriLightBase;
    m_builder.set_geometry(&m_geoPool, vtxMapped, idxMapped,
                           combinedVtx->GetGPUVirtualAddress(),
                           combinedIdx->GetGPUVirtualAddress(),
                           combinedVertexCount);
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
//ping-pong swap keeps its id, so its ReSTIR/DLSS history survives. Retain by the
//set of LIVE cell nodes (not just cellSet cells) so split-fallback sub-cells -
//whose node_id is a leaf, not a cellSet cell node - keep their id; without this
//their id would be recycled while they still render, colliding with another
//cell. INVALID_NODE cells are split originals that were replaced - skip them.
void StreamOrchestrator::assign_stable_ids(Generation& g) {
    std::unordered_set<uint64_t> liveNodes;
    liveNodes.reserve(g.cells.size());
    for (CellInstance& c : g.cells) {
        if (c.node_id == INVALID_NODE) continue;   // replaced-by-split original; not rendered
        c.stable_id = m_ids.get(c.node_id);
        liveNodes.insert(c.node_id);
    }
    m_ids.retain([&liveNodes](uint64_t node) { return liveNodes.count(node) != 0; });
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

    //--- first generation: build synchronously (an init-time stall is fine).
    //Uses the SAME async API but drains it on this thread: poll the plan job,
    //then record + submit + cpu-wait + reclaim in a loop until every cell is
    //Built. Spins briefly when nothing is Ready yet (workers still working).
    if (!m_haveLive) {
        m_builder.begin(m_device, make_params(), cam, nullptr, m_heightmap, m_workers);
        while (!m_builder.done()) {
            m_builder.poll();                            // plan -> tess fanout
            m_ctx->ResetPlanetLists();
            const uint64_t copyVal = m_ctx->SubmitPlanetCopy();
            //First-build path: no cap (drain everything ready) - it's an
            //init-time stall anyway, throughput beats latency.
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

    //--- advance the builder's state machine (plan-done -> fan terrain tess jobs) ---
    m_builder.poll();

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
            //
            //The clamp is a sanity cap against velocity spikes (a teleport
            //would otherwise push the predicted target hundreds of km off and
            //all chunks would be rebuilt at the wrong LOD). The earlier 4 x
            //trigger_m cap was way too tight: any camera moving faster than
            //~0.2 m/frame had its prediction clipped to 4 m, so the rebuilt
            //LOD landed several seconds behind the actual camera and chunks
            //read as stuck at the previous resolution. 256 m comfortably
            //covers normal flight (a 100 m/s camera + 30-frame rebuild moves
            //50 m) while still discarding genuine teleports.
            DVec3 predicted = cam.position_world;
            if (m_cfg.predict) {
                DVec3        off    = m_camVel * (double)m_throughput.rebuild_frames;
                const double L      = length(off);
                const double maxoff = 256.0;
                if (L > maxoff && L > 0.0) off = off * (maxoff / L);   // clamp a wild prediction
                predicted = cam.position_world + off;
            }
            CameraView predCam = cam;
            predCam.position_world = predicted;
            //async begin: enqueues the plan job, returns immediately. The
            //next poll() (above, next frame) sees plan_done and fans terrain the
            //per-cell tess jobs onto the worker pool.
            m_builder.begin(m_device, make_params(), predCam, &m_live,
                            m_heightmap, m_workers);
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

    //--- record BLAS builds for every cell whose tess is complete. The
    //tess + alloc happened on worker threads; this is the only render-thread
    //rebuild cost. ---
    uint32_t recorded = 0;
    if (m_builder.active()) {
        const auto t0 = std::chrono::high_resolution_clock::now();
        //build_budget caps BLAS-per-frame GPU work. With async tess, every cell
        //finishes around the same time; without a cap one frame would record
        //dozens of BLAS builds and the graphics queue (waiting the compute
        //fence) stalls for all of them.
        recorded = m_builder.record_ready_blas(cl, m_cfg.build_budget);
        const auto t1 = std::chrono::high_resolution_clock::now();
        const float blas_rec_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
        m_throughput.on_step(blas_rec_ms);
        m_stats.blas_record_cpu_ms = blas_rec_ms;
    } else {
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

    //Resolve the 3 timestamps into this slot of the readback buffer.
    cl->ResolveQueryData(m_queryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP,
                         slot * TS_PER_SLOT, TS_PER_SLOT,
                         m_tsReadback.Get(),
                         (uint64_t)slot * TS_PER_SLOT * sizeof(uint64_t));

    const uint64_t cv = m_ctx->SubmitPlanetCompute(copyVal);
    if (recorded > 0) m_builder.on_submitted(cv);

    //park this slot for fence-gated readback next time we come round.
    m_tsRing[slot] = TsSlot{ cv, true };
    m_tsWrite++;

    //--- publish stats ---
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

    //map the instance-properties terrain region for this frame's terrain entries
    //(disjoint from the scene region the Scene writes; both are UPLOAD heap).
    InstanceProperties* props = nullptr;
    if (m_instanceProps) {
        D3D12_RANGE nr{ 0, 0 };
        if (FAILED(m_instanceProps->Map(0, &nr, (void**)&props))) props = nullptr;
    }

    //one TLAS instance per LIVE cell. InstanceID = m_terrainPropsBase (FIXED =
    //scene capacity) + the cell's stable id - a REAL instanceProps index
    //(< TERRAIN_INSTANCE_BASE), so terrain shades through the unified
    //EvalSurfaceState path. The base is FIXED (independent of the live scene
    //instance count), and stable_id is hard-capped below, so instID is provably
    //in [m_terrainPropsBase, m_terrainPropsBase + MAX_TERRAIN_CELLS) - always
    //within the instanceProps buffer (sized propsBase + MAX_TERRAIN_CELLS) and
    //never overlapping the scene region [0, propsBase). The cell geometry is
    //cell-local, so the instance transform is a pure translation.
    uint32_t dropped = 0;
    for (const CellInstance& c : m_live.cells) {
        if (c.node_id == INVALID_NODE) continue;   // split original, replaced by sub-cells (not a hole)
        //HARD safety bound: stable_id MUST be < MAX_TERRAIN_CELLS or the write
        //below (and the shader's instanceProps[instID]) would run past the
        //reserved terrain region. The StableIdMap's high-water mark is bounded
        //by the peak live-cell count (< MAX_TERRAIN_CELLS for any sane budget),
        //but a pathological cut is dropped here rather than risking an OOB.
        if (c.stable_id >= MAX_TERRAIN_CELLS) { ++dropped; continue; }  // id overflow -> HOLE
        if (c.blas_va == 0)                   { ++dropped; continue; }  // pool exhausted -> HOLE
        const uint32_t instID = m_terrainPropsBase + c.stable_id;
        float xform[12];
        make_translation(xform, c.anchor_world - m_sceneOrigin);
        m_tlas.add_instance(c.blas_va, xform, instID,
                            terrain_hit_group, D3D12_RAYTRACING_INSTANCE_FLAG_FORCE_OPAQUE);
        if (props)
            fill_terrain_props(props[instID], c, m_sceneOrigin,
                               m_terrainMatIDBase, m_terrainTriLightBase);
    }

    //PLANET ROCKS: one TLAS instance + InstanceProperties per live scatter rock,
    //in the reserved range [m_rockPropsBase, m_rockPropsBase + MAX_ROCK_INSTANCES).
    //Rocks shade as scene meshes sharing the terrain material/tri-light regions.
    for (const RockInstance& r : m_rockInstances) {
        if (r.stable_id >= MAX_ROCK_INSTANCES)    continue;
        if (r.variant   >= m_rockVariants.size()) continue;
        const RockVariantGPU& var = m_rockVariants[r.variant];
        if (var.blas_va == 0)                     continue;
        const uint32_t instID = m_rockPropsBase + r.stable_id;
        const float tx = (float)(r.anchor_world.x - m_sceneOrigin.x);
        const float ty = (float)(r.anchor_world.y - m_sceneOrigin.y);
        const float tz = (float)(r.anchor_world.z - m_sceneOrigin.z);
        //row-major 3x4 (world = M * [local,1]); 3x3 = rot_scale, translation in col 3.
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

    if (props) m_instanceProps->Unmap(0, nullptr);
    m_stats.cells_dropped = dropped;

    m_tlas.build(compute_cl);
}

//====================================
//END FRAME
//====================================
void StreamOrchestrator::end_frame() {
    //reclaim + swap happen in begin_frame; nothing to do here.
}

} // namespace planet
