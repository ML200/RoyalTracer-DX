#include "generation.h"
#include "tessellator.h"
#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <intrin.h>

namespace planet {

namespace {
enum CellState : uint8_t {
    CS_Pending   = 0,
    CS_Ready     = 1,
    CS_Recorded  = 2,
    CS_Built     = 3,
};

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
}

// Starts a worker plan while retaining the live generation for diffing.
void GenerationBuilder::begin(ID3D12Device5* device, const GenerationParams& params,
                              const CameraView& cam, const Generation* live,
                              const IHeightmapSource& heightmap, WorkerPool& workers) {
    m_params = params;
    if (m_params.cells.max_leaves_per_cell == 0) m_params.cells.max_leaves_per_cell = 1;

    m_device    = device;
    m_heightmap = &heightmap;
    m_workers   = &workers;

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

    workers.enqueue([this, params_copy = m_params, cam_copy = cam, live]() {
        plan_job_(params_copy, cam_copy, live);
    });
}

// Builds the desired cut and records changed cells for tessellation.
void GenerationBuilder::plan_job_(GenerationParams params, CameraView cam,
                                  const Generation* live) {
    using clock = std::chrono::high_resolution_clock;
    const auto t0 = clock::now();

    m_sceneOrigin = cam.scene_origin;

    m_gen.qt.select(params.quadtree, cam, m_heightmap);
    m_gen.cellSet.build(m_gen.qt, params.cells);
    m_gen.leaf_count     = m_gen.qt.leaf_count();
    m_gen.triangle_count = (uint64_t)m_gen.qt.leaf_count() * MAX_CHUNK_TRIS;

    const std::vector<BlasCellSet::Cell>& cells = m_gen.cellSet.cells();
    m_gen.cells.resize(cells.size());
    m_gen.cells.reserve(cells.size() + m_gen.leaf_count);

    const uint32_t maxK = params.cells.max_leaves_per_cell;
    m_resultSize.assign(maxK + 1, 0);
    m_maxScratch = 0;
    for (uint32_t K = 1; K <= maxK; ++K) {
        uint64_t r = 0, s = 0;
        cell_blas_sizes(m_device, K, r, s);
        m_resultSize[K] = r;
        if (s > m_maxScratch) m_maxScratch = s;
    }

    std::vector<uint8_t> dirty;
    if (live && live->valid())
        diff_generations(live->qt, live->cellSet, m_gen.qt, m_gen.cellSet, dirty);
    else
        dirty.assign(cells.size(), 1);

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
            if (li >= 0 && live->cells[li].node_id != INVALID_NODE
                        && live->cells[li].blas_va != 0) {
                m_gen.cells[i] = live->cells[li];
                continue;
            }
        }
        m_dirty.push_back(i);
    }

    const uint32_t maxBuilds = m_gen.leaf_count > 0 ? m_gen.leaf_count : 1u;
    m_cellBuildCount = (uint32_t)m_dirty.size();
    m_cellBuilds.reset(new CellBuild[maxBuilds]);

    const auto t1 = clock::now();
    m_planMs.store(std::chrono::duration<float, std::milli>(t1 - t0).count(),
                   std::memory_order_relaxed);

    m_planDone.store(true, std::memory_order_release);
}

// Moves completed cells through tessellation and BLAS readiness states.
void GenerationBuilder::poll() {
    if (m_state != State::Planning) return;
    if (!m_planDone.load(std::memory_order_acquire)) return;

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

        const uint32_t origBegin = m_gen.cells[cellIdx].leaf_begin;
        const DVec3    anchor     = m_gen.cells[cellIdx].anchor_world;
        uint32_t cursor    = origBegin;
        uint32_t remaining = K;
        while (remaining > 0) {
            uint32_t got = 0;
            const uint32_t off = m_pool->allocate_upto(remaining, got);
            if (off == TERRAIN_GEO_INVALID || got == 0) break;
            CellInstance sub;
            sub.node_id        = leaves[cursor];
            sub.anchor_world   = anchor;
            sub.leaf_begin     = cursor;
            sub.leaf_count     = got;
            sub.vtx_base_elems = m_pool->vertex_base_elems(off);
            sub.idx_base_elems = m_pool->index_base_elems(off);
            sub.geo_tri_count  = got * MAX_CHUNK_TRIS;
            sub.geo            = std::make_shared<GeoSlot>(m_pool, off, got);
            const uint32_t subIdx = (uint32_t)m_gen.cells.size();
            m_gen.cells.push_back(std::move(sub));
            newDirty.push_back(subIdx);
            cursor    += got;
            remaining -= got;
        }
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

void GenerationBuilder::tess_job_(uint32_t dirty_idx) {
    CellBuild&  cb       = m_cellBuilds[dirty_idx];
    const uint32_t cellIdx = m_dirty[dirty_idx];
    const CellInstance&      ci   = m_gen.cells[cellIdx];
    const std::vector<uint64_t>& leaves = m_gen.cellSet.cell_leaves();
    const uint32_t K = ci.leaf_count;

    for (uint32_t s = 0; s < K; ++s) {
        const QuadNode leaf = unpack_node_id(leaves[ci.leaf_begin + s]);

        uint8_t mask = 0;
        for (int e = 0; e < 4; ++e)
            if (m_gen.qt.neighbor_lod(leaf, (QuadEdge)e) < leaf.lod)
                mask |= (uint8_t)(1u << e);
        for (int c = 0; c < 4; ++c)
            if (m_gen.qt.corner_lod(leaf, c) < leaf.lod)
                mask |= (uint8_t)(1u << (4 + c));

        const uint32_t leafVtxBase = ci.vtx_base_elems + s * MAX_CHUNK_VERTS;
        const uint32_t leafIdxBase = ci.idx_base_elems + s * (MAX_CHUNK_TRIS * 3u);

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

    _mm_sfence();
    cb.state.store(CS_Ready, std::memory_order_release);
}

// Records bounded BLAS builds and retains resources until their fence retires.
uint32_t GenerationBuilder::record_ready_blas(ID3D12GraphicsCommandList4* compute_cl,
                                              uint32_t budget) {
    m_lastBlasRecordMs = 0.0f;
    if (m_state != State::Streaming) return 0;

    using clock = std::chrono::high_resolution_clock;
    const auto t0 = clock::now();

    ComPtr<ID3D12Resource> scratch;

    uint32_t recorded = 0;
    for (uint32_t i = 0; i < m_cellBuildCount; ++i) {
        if (budget != 0 && recorded >= budget) break;
        CellBuild& cb = m_cellBuilds[i];
        if (cb.state.load(std::memory_order_acquire) != CS_Ready) continue;

        if (!scratch) {
            scratch = create_buffer(m_device, m_maxScratch,
                                    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                    D3D12_RESOURCE_STATE_COMMON, HEAP_DEFAULT);
        }

        const uint32_t cellIdx = m_dirty[i];
        const CellInstance&      ci   = m_gen.cells[cellIdx];
        const uint32_t K = ci.leaf_count;

        ComPtr<ID3D12Resource> result = create_buffer(
            m_device, m_resultSize[K], D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, HEAP_DEFAULT);

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

        D3D12_RESOURCE_BARRIER uav = {};
        uav.Type          = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        uav.UAV.pResource = scratch.Get();
        compute_cl->ResourceBarrier(1, &uav);

        m_gen.cells[cellIdx].blas    = result;
        m_gen.cells[cellIdx].blas_va = result->GetGPUVirtualAddress();

        m_pending.recordedDirty.push_back(i);

        cb.state.store(CS_Recorded, std::memory_order_release);
        ++recorded;
    }

    if (scratch) m_pending.transients.push_back(scratch);

    const auto t1 = clock::now();
    m_lastBlasRecordMs = std::chrono::duration<float, std::milli>(t1 - t0).count();
    return recorded;
}

void GenerationBuilder::on_submitted(uint64_t compute_fence) {
    if (m_pending.recordedDirty.empty()) return;
    m_pending.fence = compute_fence;
    m_batches.push_back(std::move(m_pending));
    m_pending = Batch{};
}

void GenerationBuilder::reclaim(uint64_t completed_fence) {
    size_t w = 0;
    for (size_t r = 0; r < m_batches.size(); ++r) {
        if (m_batches[r].fence <= completed_fence) {
            for (uint32_t dirty_idx : m_batches[r].recordedDirty) {
                m_cellBuilds[dirty_idx].state.store(CS_Built,
                                                    std::memory_order_release);
            }
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

}
