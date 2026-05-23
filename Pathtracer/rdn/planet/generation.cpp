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

//worst-case BLAS result + scratch size for a cell of K full-grid leaf
//geometries. Every leaf tessellates to the same full CHUNK_GRID mesh, so the
//size depends only on the geometry count K.
void cell_blas_sizes(ID3D12Device5* device, uint32_t K,
                     uint64_t& out_result, uint64_t& out_scratch) {
    std::vector<D3D12_RAYTRACING_GEOMETRY_DESC> g(K);
    for (uint32_t k = 0; k < K; ++k) {
        g[k] = {};
        g[k].Type  = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
        g[k].Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
        g[k].Triangles.VertexBuffer.StrideInBytes = CHUNK_VERTEX_STRIDE;
        g[k].Triangles.VertexCount  = MAX_CHUNK_VERTS;
        g[k].Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
        g[k].Triangles.IndexCount   = MAX_CHUNK_TRIS * 3;
        g[k].Triangles.IndexFormat  = DXGI_FORMAT_R16_UINT;
    }
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS in = {};
    in.Type           = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
    in.DescsLayout    = D3D12_ELEMENTS_LAYOUT_ARRAY;
    in.Flags          = PLANET_BLAS_BUILD_FLAGS;
    in.NumDescs       = K;
    in.pGeometryDescs = g.data();

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

    //--- target structure: leaf cut + cell cut ---
    m_gen.qt.select(params.quadtree, cam);
    m_gen.cellSet.build(m_gen.qt, params.cells);
    m_gen.leaf_count     = m_gen.qt.leaf_count();
    m_gen.triangle_count = (uint64_t)m_gen.qt.leaf_count() * MAX_CHUNK_TRIS;

    const std::vector<BlasCellSet::Cell>& cells = m_gen.cellSet.cells();
    m_gen.cells.resize(cells.size());

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

        if (!dirty[i] && live) {
            const int li = live->cellSet.cell_index(cells[i].node_id);
            if (li >= 0) {
                m_gen.cells[i] = live->cells[li];        // reuse: copies the shared BLAS ComPtr
                continue;
            }
        }
        m_dirty.push_back(i);                            // dirty -> async tess + BLAS build
    }

    //Pre-allocate the per-cell async-build slots (atomics default-zero =
    //CS_Pending). Pre-allocate so the render thread doesn't have to grow
    //the buffer when it kicks off the tess jobs.
    m_cellBuildCount = (uint32_t)m_dirty.size();
    m_cellBuilds.reset(new CellBuild[m_cellBuildCount]);

    const auto t1 = clock::now();
    m_planMs.store(std::chrono::duration<float, std::milli>(t1 - t0).count(),
                   std::memory_order_relaxed);

    //RELEASE: every write above must be visible to the render thread once it
    //sees planDone=true via its acquire-load. The orchestrator's poll() then
    //fans out the tess jobs.
    m_planDone.store(true, std::memory_order_release);
}

//====================================
//POLL - render-thread tick: plan -> tess fanout
//====================================
void GenerationBuilder::poll() {
    if (m_state != State::Planning) return;
    if (!m_planDone.load(std::memory_order_acquire)) return;

    //plan finished: every dirty cell becomes one tess job. Each job allocates
    //its own upload buffers + tessellates + flushes WC writes + flips state
    //to Ready. The render thread picks them up via record_ready_blas().
    for (uint32_t i = 0; i < m_cellBuildCount; ++i) {
        m_workers->enqueue([this, i]() { tess_job_(i); });
    }
    m_state = State::Streaming;
}

//====================================
//TESS JOB - one dirty cell (worker thread)
//====================================
void GenerationBuilder::tess_job_(uint32_t dirty_idx) {
    CellBuild&  cb       = m_cellBuilds[dirty_idx];
    const uint32_t cellIdx = m_dirty[dirty_idx];
    const BlasCellSet::Cell& cell = m_gen.cellSet.cells()[cellIdx];
    const std::vector<uint64_t>& leaves = m_gen.cellSet.cell_leaves();
    const uint32_t K = cell.leaf_count;

    //per-cell upload buffers. CreateCommittedResource is thread-safe (D3D12
    //device methods are MT-safe), so the ~100-200us alloc cost lives on the
    //worker, not the render thread.
    cb.vtx = create_buffer(m_device, (uint64_t)K * CHUNK_VERTEX_BYTES,
                           D3D12_RESOURCE_FLAG_NONE,
                           D3D12_RESOURCE_STATE_GENERIC_READ, HEAP_UPLOAD);
    cb.idx = create_buffer(m_device, (uint64_t)K * CHUNK_INDEX_BYTES,
                           D3D12_RESOURCE_FLAG_NONE,
                           D3D12_RESOURCE_STATE_GENERIC_READ, HEAP_UPLOAD);
    D3D12_RANGE no_read{ 0, 0 };
    void* vp = nullptr; void* ip = nullptr;
    if (FAILED(cb.vtx->Map(0, &no_read, &vp)) ||
        FAILED(cb.idx->Map(0, &no_read, &ip)))
        throw std::runtime_error("planet: tess_job upload-buffer Map failed");
    cb.vp = static_cast<uint8_t*>(vp);
    cb.ip = static_cast<uint8_t*>(ip);

    //tessellate every leaf of this cell into its slot in the upload buffers.
    for (uint32_t s = 0; s < K; ++s) {
        const QuadNode leaf = unpack_node_id(leaves[cell.leaf_begin + s]);

        uint8_t mask = 0;
        for (int e = 0; e < 4; ++e)
            if (m_gen.qt.neighbor_lod(leaf, (QuadEdge)e) < leaf.lod)
                mask |= (uint8_t)(1u << e);

        TessJob tj;
        tj.node            = leaf;
        tj.planet          = m_params.quadtree.planet;
        tj.anchor_world    = cell.anchor_world;
        tj.grid            = CHUNK_GRID;
        tj.stitch_mask     = mask;
        tj.vertex_dest     = cb.vp + (uint64_t)s * CHUNK_VERTEX_BYTES;
        tj.index_dest      = cb.ip + (uint64_t)s * CHUNK_INDEX_BYTES;
        tj.vertex_capacity = CHUNK_VERTEX_BYTES;
        tj.index_capacity  = CHUNK_INDEX_BYTES;
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
    const std::vector<BlasCellSet::Cell>& cells = m_gen.cellSet.cells();
    std::vector<D3D12_RAYTRACING_GEOMETRY_DESC> geom(m_params.cells.max_leaves_per_cell);

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
        const BlasCellSet::Cell& cell = cells[cellIdx];
        const uint32_t K = cell.leaf_count;

        ComPtr<ID3D12Resource> result = create_buffer(
            m_device, m_resultSize[K], D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, HEAP_DEFAULT);

        for (uint32_t g = 0; g < K; ++g) {
            D3D12_RAYTRACING_GEOMETRY_DESC& gd = geom[g];
            gd = {};
            gd.Type  = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
            gd.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
            gd.Triangles.VertexBuffer.StartAddress  =
                cb.vtx->GetGPUVirtualAddress() + (uint64_t)g * CHUNK_VERTEX_BYTES;
            gd.Triangles.VertexBuffer.StrideInBytes = CHUNK_VERTEX_STRIDE;
            gd.Triangles.VertexCount  = MAX_CHUNK_VERTS;
            gd.Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
            gd.Triangles.IndexBuffer  =
                cb.idx->GetGPUVirtualAddress() + (uint64_t)g * CHUNK_INDEX_BYTES;
            gd.Triangles.IndexCount   = MAX_CHUNK_TRIS * 3;
            gd.Triangles.IndexFormat  = DXGI_FORMAT_R16_UINT;
        }

        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC desc = {};
        desc.Inputs.Type           = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
        desc.Inputs.DescsLayout    = D3D12_ELEMENTS_LAYOUT_ARRAY;
        desc.Inputs.Flags          = PLANET_BLAS_BUILD_FLAGS;
        desc.Inputs.NumDescs       = K;
        desc.Inputs.pGeometryDescs = geom.data();
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

        //transient buffers stay alive until the batch fence retires.
        m_pending.transients.push_back(cb.vtx);
        m_pending.transients.push_back(cb.idx);
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
                //tess upload mapping is no longer needed; drop the pointers
                //and the mapped ComPtrs (the batch's transients still hold
                //refs - they'll drop with the batch below).
                m_cellBuilds[dirty_idx].vp = nullptr;
                m_cellBuilds[dirty_idx].ip = nullptr;
                m_cellBuilds[dirty_idx].vtx.Reset();
                m_cellBuilds[dirty_idx].idx.Reset();
            }
            //m_batches[r].transients ComPtrs drop here -> upload + scratch freed.
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
