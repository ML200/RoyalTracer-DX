//====================================
//PLANET - GENERATION
//====================================

#include "generation.h"
#include "tessellator.h"
#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <intrin.h>          // _mm_sfence

namespace planet {

namespace {
//Cell state enum, packed into the atomic<uint8_t> on CellBuild.
enum CellState : uint8_t {
    CS_Pending   = 0,        // tess not yet started (or in flight)
    CS_Ready     = 1,        // tess complete, BLAS not yet recorded
    CS_Recorded  = 2,        // BLAS recorded on command list; fence pending
    CS_Built     = 3,        // BLAS build fence retired; BLAS is valid
};

//worst-case BLAS result + scratch size for a cell of K leaves. The cell is ONE
//merged geometry over K contiguous leaf slots, so the size depends only on K.
void cell_blas_sizes(ID3D12Device5* device, uint32_t K,
                     uint64_t& out_result, uint64_t& out_scratch) {
    D3D12_RAYTRACING_GEOMETRY_DESC g = {};
    g.Type  = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
    g.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
    g.Triangles.VertexBuffer.StrideInBytes = CHUNK_VERTEX_STRIDE;
    g.Triangles.VertexCount  = K * MAX_CHUNK_VERTS;
    g.Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
    g.Triangles.IndexCount   = K * MAX_CHUNK_TRIS * 3;
    g.Triangles.IndexFormat  = DXGI_FORMAT_R32_UINT;

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS in = {};
    in.Type           = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
    in.DescsLayout    = D3D12_ELEMENTS_LAYOUT_ARRAY;
    in.Flags          = PLANET_BLAS_BUILD_FLAGS;
    in.NumDescs       = 1;
    in.pGeometryDescs = &g;

    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO info = {};
    device->GetRaytracingAccelerationStructurePrebuildInfo(&in, &info);
    out_result  = info.ResultDataMaxSizeInBytes;
    out_scratch = info.ScratchDataSizeInBytes;
}
} // namespace

//====================================
//BEGIN - enqueue the plan job; returns immediately
//====================================
void GenerationBuilder::begin(ID3D12Device5* device, const GenerationParams& params,
                              const CameraView& cam, const Generation* live,
                              const IHeightmapSource& heightmap, WorkerPool& workers) {
    m_params = params;
    if (m_params.cells.max_leaves_per_cell == 0) m_params.cells.max_leaves_per_cell = 1;

    m_device    = device;
    m_heightmap = &heightmap;
    m_workers   = &workers;

    //reset every per-rebuild slot. The plan job writes m_gen / m_dirty /
    //m_resultSize / m_maxScratch / m_cellBuilds[] before the planDone flag.
    m_state = State::Planning;
    m_gen   = Generation{};
    m_dirty.clear();
    m_resultSize.clear();
    m_maxScratch = 0;
    m_cellBuilds.reset();
    m_cellBuildCount = 0;
    m_batches.clear();
    m_pending = Batch{};
    m_lastBlasRecordMs = 0.0f;
    m_planMs.store(0.0f, std::memory_order_relaxed);
    m_planDone.store(false, std::memory_order_release);

    //plan job: capture inputs by value (params, cam) and by pointer (live).
    //LIVE outlives this build (orchestrator only swaps on done()), so the
    //pointer is stable. The plan job touches no GPU command lists, only the
    //CPU device interface (which is thread-safe).
    workers.enqueue([this, params_copy = m_params, cam_copy = cam, live]() {
        plan_job_(params_copy, cam_copy, live);
    });
}

//====================================
//PLAN JOB - LOD select + cell cut + diff (worker thread)
//====================================
void GenerationBuilder::plan_job_(GenerationParams params, CameraView cam,
                                  const Generation* live) {
    using clock = std::chrono::high_resolution_clock;
    const auto t0 = clock::now();

    //Capture the scene origin this rebuild was triggered with. tess_job_
    //reads it for each chunk's FP32-quant-compensation; the orchestrator's
    //TLAS instance transform uses the current scene origin, which differs
    //from this one only by FP32-exact 1 km snap deltas - leaving the
    //per-vertex correction invariant. Written here BEFORE planDone is
    //released so the render thread observes it on the same acquire.
    m_sceneOrigin = cam.scene_origin;

    //--- target structure: leaf cut + cell cut ---
    //pass the heightmap so LOD distance tracks the DISPLACED surface (camera
    //standing on relief refines the ground under it, not the analytical sphere).
    m_gen.qt.select(params.quadtree, cam, m_heightmap);
    m_gen.cellSet.build(m_gen.qt, params.cells);
    m_gen.leaf_count     = m_gen.qt.leaf_count();
    m_gen.triangle_count = (uint64_t)m_gen.qt.leaf_count() * MAX_CHUNK_TRIS;

    const std::vector<BlasCellSet::Cell>& cells = m_gen.cellSet.cells();
    m_gen.cells.resize(cells.size());
    //Headroom for split-fallback sub-cells appended in poll() (worst case: every
    //dirty cell splits to 1-leaf sub-cells). Reserving here means those
    //push_backs never reallocate the vector mid-rebuild.
    m_gen.cells.reserve(cells.size() + m_gen.leaf_count);

    //--- BLAS prebuild sizes per geometry count K, and one shared scratch size ---
    //The device call is thread-safe per D3D12 contract; safe to do here.
    const uint32_t maxK = params.cells.max_leaves_per_cell;
    m_resultSize.assign(maxK + 1, 0);
    m_maxScratch = 0;
    for (uint32_t K = 1; K <= maxK; ++K) {
        uint64_t r = 0, s = 0;
        cell_blas_sizes(m_device, K, r, s);
        m_resultSize[K] = r;
        if (s > m_maxScratch) m_maxScratch = s;
    }

    //--- diff against LIVE: clean cells reuse the LIVE BLAS, dirty cells queue ---
    std::vector<uint8_t> dirty;
    if (live && live->valid())
        diff_generations(live->qt, live->cellSet, m_gen.qt, m_gen.cellSet, dirty);
    else
        dirty.assign(cells.size(), 1);                  // first generation: all dirty

    m_dirty.clear();
    m_dirty.reserve(cells.size());
    for (uint32_t i = 0; i < cells.size(); ++i) {
        CellInstance& ci = m_gen.cells[i];
        ci.node_id      = cells[i].node_id;
        ci.anchor_world = cells[i].anchor_world;
        ci.leaf_begin   = cells[i].leaf_begin;
        ci.leaf_count   = cells[i].leaf_count;

        if (!dirty[i] && live) {
            const int li = live->cellSet.cell_index(cells[i].node_id);
            //Only reuse a VALID live cell. A cell that was split last generation
            //(transient pool fragmentation) left an INVALID_NODE placeholder in
            //its cellSet slot; reusing that would copy the hole forward. Treat
            //it as dirty so it re-tessellates - and re-merges into one cell once
            //the pool has a contiguous run again.
            if (li >= 0 && live->cells[li].node_id != INVALID_NODE
                        && live->cells[li].blas_va != 0) {
                m_gen.cells[i] = live->cells[li];        // reuse: copies the shared BLAS ComPtr
                continue;
            }
        }
        m_dirty.push_back(i);                            // dirty -> async tess + BLAS build
    }

    //Pre-allocate the per-cell async-build slots (atomics default-zero =
    //CS_Pending). Sized for the WORST case where every dirty cell's contiguous
    //pool allocation fails and it splits into 1-leaf sub-cells (poll() appends
    //sub-cells and sets the real m_cellBuildCount). Total leaf count is the
    //upper bound on to-tessellate cells (each has >= 1 leaf).
    const uint32_t maxBuilds = m_gen.leaf_count > 0 ? m_gen.leaf_count : 1u;
    m_cellBuildCount = (uint32_t)m_dirty.size();
    m_cellBuilds.reset(new CellBuild[maxBuilds]);

    const auto t1 = clock::now();
    m_planMs.store(std::chrono::duration<float, std::milli>(t1 - t0).count(),
                   std::memory_order_relaxed);

    //RELEASE: every write above must be visible to the render thread once it
    //sees planDone=true via its acquire-load. The orchestrator's poll() then
    //fans terrain the tess jobs.
    m_planDone.store(true, std::memory_order_release);
}

//====================================
//POLL - render-thread tick: plan -> tess fanout
//====================================
void GenerationBuilder::poll() {
    if (m_state != State::Planning) return;
    if (!m_planDone.load(std::memory_order_acquire)) return;

    //plan finished. Allocate each dirty cell a geometry slot HERE (render
    //thread; the pool is not thread-safe), then fan one tess job per to-build
    //cell. The dirty list is REBUILT here because of the split fallback:
    //
    //  - allocate(K) succeeds  -> the cell keeps its slot (fast path, one BLAS).
    //  - allocate(K) FAILS     -> the pool has K free slots somewhere but not
    //    K CONTIGUOUS (fragmented under fast-flight churn). Instead of dropping
    //    the cell (a hole), split it into sub-cells that each take the largest
    //    free span (allocate_upto), so every leaf is placed and renders. The
    //    original cell is marked not-rendered (node_id = INVALID). Sub-cells are
    //    appended to m_gen.cells (capacity reserved in plan_job so no realloc).
    //
    //Big cells stay the default; only the few fragmented ones split, and they
    //re-merge on the next rebuild once the pool defragments.
    const std::vector<uint64_t>& leaves = m_gen.cellSet.cell_leaves();
    std::vector<uint32_t> newDirty;
    newDirty.reserve(m_dirty.size());

    for (uint32_t d = 0; d < (uint32_t)m_dirty.size(); ++d) {
        const uint32_t cellIdx = m_dirty[d];
        const uint32_t K       = m_gen.cells[cellIdx].leaf_count;
        if (K == 0 || !m_pool) continue;

        const uint32_t leaf_off = m_pool->allocate(K);
        if (leaf_off != TERRAIN_GEO_INVALID) {
            CellInstance& ci  = m_gen.cells[cellIdx];
            ci.vtx_base_elems = m_pool->vertex_base_elems(leaf_off);
            ci.idx_base_elems = m_pool->index_base_elems(leaf_off);
            ci.geo_tri_count  = K * MAX_CHUNK_TRIS;
            ci.geo            = std::make_shared<GeoSlot>(m_pool, leaf_off, K);
            newDirty.push_back(cellIdx);
            continue;
        }

        //--- split fallback: carve the cell into sub-cells over the free spans.
        const uint32_t origBegin = m_gen.cells[cellIdx].leaf_begin;
        const DVec3    anchor     = m_gen.cells[cellIdx].anchor_world;
        uint32_t cursor    = origBegin;
        uint32_t remaining = K;
        while (remaining > 0) {
            uint32_t got = 0;
            const uint32_t off = m_pool->allocate_upto(remaining, got);
            if (off == TERRAIN_GEO_INVALID || got == 0) break;  // pool truly full (capacity, not fragmentation)
            CellInstance sub;
            sub.node_id        = leaves[cursor];     // first leaf's node = unique sub-cell identity
            sub.anchor_world   = anchor;
            sub.leaf_begin     = cursor;
            sub.leaf_count     = got;
            sub.vtx_base_elems = m_pool->vertex_base_elems(off);
            sub.idx_base_elems = m_pool->index_base_elems(off);
            sub.geo_tri_count  = got * MAX_CHUNK_TRIS;
            sub.geo            = std::make_shared<GeoSlot>(m_pool, off, got);
            const uint32_t subIdx = (uint32_t)m_gen.cells.size();
            m_gen.cells.push_back(std::move(sub));   // capacity reserved in plan_job -> no realloc
            newDirty.push_back(subIdx);
            cursor    += got;
            remaining -= got;
        }
        //the original cell is fully replaced by its sub-cells (or, only if the
        //pool was truly full, partially) - mark it not-rendered so record_tlas
        //and assign_stable_ids skip it (it is NOT a dropped-cell hole).
        m_gen.cells[cellIdx].node_id    = INVALID_NODE;
        m_gen.cells[cellIdx].leaf_count = 0;
        m_gen.cells[cellIdx].geo.reset();
    }

    m_dirty          = std::move(newDirty);
    m_cellBuildCount = (uint32_t)m_dirty.size();
    for (uint32_t i = 0; i < m_cellBuildCount; ++i)
        m_workers->enqueue([this, i]() { tess_job_(i); });
    m_state = State::Streaming;
}

//====================================
//TESS JOB - one dirty cell (worker thread)
//====================================
void GenerationBuilder::tess_job_(uint32_t dirty_idx) {
    CellBuild&  cb       = m_cellBuilds[dirty_idx];
    const uint32_t cellIdx = m_dirty[dirty_idx];
    const CellInstance&      ci   = m_gen.cells[cellIdx];
    const std::vector<uint64_t>& leaves = m_gen.cellSet.cell_leaves();
    //Leaf range comes from the CellInstance (NOT cellSet.cells()[cellIdx]) so a
    //split fallback sub-cell - which is appended past the cellSet cell count -
    //tessellates its own sub-range.
    const uint32_t K = ci.leaf_count;

    //tessellate every leaf straight into its slot of the mapped combined
    //buffers (no per-cell upload buffer). The cell occupies K contiguous leaf
    //slots from ci.vtx_base_elems / ci.idx_base_elems; leaf s sits +s strides
    //in. Indices are ABSOLUTE into the combined vertex buffer so the cell's
    //merged single-geometry BLAS and the shader both read it directly.
    for (uint32_t s = 0; s < K; ++s) {
        const QuadNode leaf = unpack_node_id(leaves[ci.leaf_begin + s]);

        uint8_t mask = 0;
        for (int e = 0; e < 4; ++e)
            if (m_gen.qt.neighbor_lod(leaf, (QuadEdge)e) < leaf.lod)
                mask |= (uint8_t)(1u << e);
        //Corner-stitch bits: an interior chunk whose four edge-neighbours are
        //all same-LOD can still share a corner POINT with a coarser diagonal
        //chunk - edge balance never sees that diagonal. Without flagging it,
        //the corner vertex evaluates one extra noise octave compared to the 3
        //other chunks meeting there, leaving a sub-metre crack visible as a
        //dark spot at every LOD-transition corner (a ring around the camera).
        //bits 4..7 correspond to corner_idx 0..3 = (-s,-t)(+s,-t)(-s,+t)(+s,+t).
        for (int c = 0; c < 4; ++c)
            if (m_gen.qt.corner_lod(leaf, c) < leaf.lod)
                mask |= (uint8_t)(1u << (4 + c));

        const uint32_t leafVtxBase = ci.vtx_base_elems + s * MAX_CHUNK_VERTS;        // vertex elem
        const uint32_t leafIdxBase = ci.idx_base_elems + s * (MAX_CHUNK_TRIS * 3u);  // index elem

        TessJob tj;
        tj.node              = leaf;
        tj.planet            = m_params.quadtree.planet;
        tj.anchor_world      = ci.anchor_world;
        tj.scene_origin      = m_sceneOrigin;
        tj.grid              = CHUNK_GRID;
        tj.stitch_mask       = mask;
        tj.vertex_dest       = m_vtxMapped + (uint64_t)leafVtxBase * CHUNK_VERTEX_STRIDE;
        tj.index_dest        = m_idxMapped + (uint64_t)leafIdxBase * CHUNK_INDEX_STRIDE;
        tj.vertex_capacity   = CHUNK_VERTEX_BYTES;
        tj.index_capacity    = CHUNK_INDEX_BYTES;
        tj.index_vertex_base = leafVtxBase;
        tessellate_chunk(tj, *m_heightmap);
    }

    //UPLOAD heap memory is write-combining (WC); regular stores are buffered
    //in the CPU's WC buffers and become globally visible only after a
    //serialising instruction. SFENCE drains the WC buffers before the
    //release-store on the state flag, so the render thread (which sees
    //Ready via acquire-load) is guaranteed every byte of the tess output
    //has reached the heap before it records a BLAS build referencing it.
    //Missing this fence is what corrupted GPU output on the previous
    //async-tess attempt (project_planet_perf_revert).
    _mm_sfence();
    cb.state.store(CS_Ready, std::memory_order_release);
}

//====================================
//RECORD READY BLAS - render thread: drain Ready cells onto compute_cl
//====================================
uint32_t GenerationBuilder::record_ready_blas(ID3D12GraphicsCommandList4* compute_cl,
                                              uint32_t budget) {
    m_lastBlasRecordMs = 0.0f;
    if (m_state != State::Streaming) return 0;

    using clock = std::chrono::high_resolution_clock;
    const auto t0 = clock::now();

    //Walk every dirty cell, find the ones whose tess is complete and BLAS has
    //not yet been recorded. The 'budget' cap is GPU-throughput control: tess
    //jobs all finish roughly together (parallel workers) so without a cap the
    //compute submission can hold dozens of BLAS builds serialised on the
    //shared scratch, blocking the graphics queue (which waits the compute
    //fence). budget=0 means no cap.
    ComPtr<ID3D12Resource> scratch;                      // lazy: only if there's work

    uint32_t recorded = 0;
    for (uint32_t i = 0; i < m_cellBuildCount; ++i) {
        if (budget != 0 && recorded >= budget) break;
        CellBuild& cb = m_cellBuilds[i];
        if (cb.state.load(std::memory_order_acquire) != CS_Ready) continue;

        //first Ready of this frame: allocate the shared scratch lazily.
        if (!scratch) {
            scratch = create_buffer(m_device, m_maxScratch,
                                    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                    D3D12_RESOURCE_STATE_COMMON, HEAP_DEFAULT);
        }

        const uint32_t cellIdx = m_dirty[i];
        const CellInstance&      ci   = m_gen.cells[cellIdx];
        const uint32_t K = ci.leaf_count;   // from CellInstance (handles split sub-cells)

        ComPtr<ID3D12Resource> result = create_buffer(
            m_device, m_resultSize[K], D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, HEAP_DEFAULT);

        //ONE merged geometry over the cell's K contiguous leaf slots in the
        //combined buffer. Indices are ABSOLUTE into the combined vertex buffer,
        //so VertexBuffer.StartAddress is the buffer base (element 0) and the
        //index range is the cell's slice. No copy: the tessellator already
        //wrote the slot in place (UPLOAD heap).
        D3D12_RAYTRACING_GEOMETRY_DESC gd = {};
        gd.Type  = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
        gd.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
        gd.Triangles.VertexBuffer.StartAddress  = m_vtxVA;
        gd.Triangles.VertexBuffer.StrideInBytes = CHUNK_VERTEX_STRIDE;
        gd.Triangles.VertexCount  = m_combinedVertexCount;
        gd.Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
        gd.Triangles.IndexBuffer  = m_idxVA + (uint64_t)ci.idx_base_elems * CHUNK_INDEX_STRIDE;
        gd.Triangles.IndexCount   = K * MAX_CHUNK_TRIS * 3;
        gd.Triangles.IndexFormat  = DXGI_FORMAT_R32_UINT;

        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC desc = {};
        desc.Inputs.Type           = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
        desc.Inputs.DescsLayout    = D3D12_ELEMENTS_LAYOUT_ARRAY;
        desc.Inputs.Flags          = PLANET_BLAS_BUILD_FLAGS;
        desc.Inputs.NumDescs       = 1;
        desc.Inputs.pGeometryDescs = &gd;
        desc.DestAccelerationStructureData    = result->GetGPUVirtualAddress();
        desc.ScratchAccelerationStructureData = scratch->GetGPUVirtualAddress();
        compute_cl->BuildRaytracingAccelerationStructure(&desc, 0, nullptr);

        //next BLAS in this batch reuses the one scratch - serialise on it.
        D3D12_RESOURCE_BARRIER uav = {};
        uav.Type          = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        uav.UAV.pResource = scratch.Get();
        compute_cl->ResourceBarrier(1, &uav);

        m_gen.cells[cellIdx].blas    = result;
        m_gen.cells[cellIdx].blas_va = result->GetGPUVirtualAddress();

        //the cell's geometry lives in the persistent combined buffer (slot
        //freed via its GeoSlot when the cell is no longer referenced), so the
        //only transient is the shared scratch.
        m_pending.recordedDirty.push_back(i);

        cb.state.store(CS_Recorded, std::memory_order_release);
        ++recorded;
    }

    if (scratch) m_pending.transients.push_back(scratch);

    const auto t1 = clock::now();
    m_lastBlasRecordMs = std::chrono::duration<float, std::milli>(t1 - t0).count();
    return recorded;
}

//====================================
//SUBMIT / RECLAIM / TAKE
//====================================
void GenerationBuilder::on_submitted(uint64_t compute_fence) {
    if (m_pending.recordedDirty.empty()) return;     // nothing recorded since the last seal
    m_pending.fence = compute_fence;
    m_batches.push_back(std::move(m_pending));
    m_pending = Batch{};
}

void GenerationBuilder::reclaim(uint64_t completed_fence) {
    size_t w = 0;
    for (size_t r = 0; r < m_batches.size(); ++r) {
        if (m_batches[r].fence <= completed_fence) {
            //the GPU is done with these BLAS builds; the cells are now valid.
            for (uint32_t dirty_idx : m_batches[r].recordedDirty) {
                m_cellBuilds[dirty_idx].state.store(CS_Built,
                                                    std::memory_order_release);
            }
            //m_batches[r].transients ComPtrs drop here -> scratch freed. The
            //cell geometry stays in the persistent combined buffer.
        } else {
            m_batches[w++] = std::move(m_batches[r]);
        }
    }
    m_batches.resize(w);
}

bool GenerationBuilder::all_recorded() const {
    if (m_state != State::Streaming) return false;
    for (uint32_t i = 0; i < m_cellBuildCount; ++i) {
        const uint8_t s = m_cellBuilds[i].state.load(std::memory_order_acquire);
        if (s != CS_Recorded && s != CS_Built) return false;
    }
    return true;
}

bool GenerationBuilder::done() const {
    if (m_state != State::Streaming) return false;
    for (uint32_t i = 0; i < m_cellBuildCount; ++i) {
        if (m_cellBuilds[i].state.load(std::memory_order_acquire) != CS_Built)
            return false;
    }
    return true;
}

uint32_t GenerationBuilder::dirty_built() const {
    uint32_t n = 0;
    for (uint32_t i = 0; i < m_cellBuildCount; ++i)
        if (m_cellBuilds[i].state.load(std::memory_order_acquire) == CS_Built) ++n;
    return n;
}

uint32_t GenerationBuilder::dirty_ready() const {
    uint32_t n = 0;
    for (uint32_t i = 0; i < m_cellBuildCount; ++i)
        if (m_cellBuilds[i].state.load(std::memory_order_acquire) == CS_Ready) ++n;
    return n;
}

uint32_t GenerationBuilder::dirty_recorded() const {
    uint32_t n = 0;
    for (uint32_t i = 0; i < m_cellBuildCount; ++i)
        if (m_cellBuilds[i].state.load(std::memory_order_acquire) == CS_Recorded) ++n;
    return n;
}

uint32_t GenerationBuilder::dirty_tessellating() const {
    uint32_t n = 0;
    for (uint32_t i = 0; i < m_cellBuildCount; ++i)
        if (m_cellBuilds[i].state.load(std::memory_order_acquire) == CS_Pending) ++n;
    return n;
}

Generation GenerationBuilder::take() {
    m_state = State::Idle;
    Generation g = std::move(m_gen);
    m_gen = Generation{};
    m_dirty.clear();
    m_cellBuilds.reset();
    m_cellBuildCount = 0;
    m_batches.clear();
    m_pending = Batch{};
    m_planDone.store(false, std::memory_order_relaxed);
    m_device    = nullptr;
    m_heightmap = nullptr;
    m_workers   = nullptr;
    return g;
}

} // namespace planet
