//====================================
//PLANET - GENERATION
//====================================

#include "generation.h"
#include "tessellator.h"
#include <algorithm>
#include <stdexcept>

namespace planet {

namespace {
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
//BEGIN - target cut + dirty/clean split
//====================================
void GenerationBuilder::begin(ID3D12Device5* device, const GenerationParams& params,
                              const CameraView& cam, const Generation* live) {
    m_params = params;
    if (m_params.cells.max_leaves_per_cell == 0) m_params.cells.max_leaves_per_cell = 1;

    m_active = true;
    m_cursor = 0;
    m_built  = 0;
    m_batches.clear();
    m_pending = Batch{};
    m_dirty.clear();

    //--- target structure: leaf cut + cell cut ---
    m_gen = Generation{};
    m_gen.qt.select(m_params.quadtree, cam);
    m_gen.cellSet.build(m_gen.qt, m_params.cells);
    m_gen.leaf_count     = m_gen.qt.leaf_count();
    m_gen.triangle_count = (uint64_t)m_gen.qt.leaf_count() * MAX_CHUNK_TRIS;

    const std::vector<BlasCellSet::Cell>& cells = m_gen.cellSet.cells();
    m_gen.cells.resize(cells.size());

    //--- BLAS prebuild sizes per geometry count K, and one shared scratch size ---
    const uint32_t maxK = m_params.cells.max_leaves_per_cell;
    m_resultSize.assign(maxK + 1, 0);
    m_maxScratch = 0;
    for (uint32_t K = 1; K <= maxK; ++K) {
        uint64_t r = 0, s = 0;
        cell_blas_sizes(device, K, r, s);
        m_resultSize[K] = r;
        if (s > m_maxScratch) m_maxScratch = s;
    }

    //--- diff against LIVE: clean cells reuse the LIVE BLAS, dirty cells queue ---
    std::vector<uint8_t> dirty;
    if (live && live->valid())
        diff_generations(live->qt, live->cellSet, m_gen.qt, m_gen.cellSet, dirty);
    else
        dirty.assign(cells.size(), 1);                  // first generation: all dirty

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
        m_dirty.push_back(i);                            // dirty -> must be tessellated + built
    }
}

//====================================
//STEP - tessellate + record a budget of dirty cells
//====================================
uint32_t GenerationBuilder::step(ID3D12Device5* device, const IHeightmapSource& heightmap,
                                 WorkerPool& workers, ID3D12GraphicsCommandList4* compute_cl,
                                 uint32_t budget) {
    if (!m_active) return 0;
    const uint32_t remaining = (uint32_t)m_dirty.size() - m_cursor;
    const uint32_t n = budget < remaining ? budget : remaining;
    if (n == 0) return 0;

    const std::vector<BlasCellSet::Cell>& cells  = m_gen.cellSet.cells();
    const std::vector<uint64_t>&          leaves = m_gen.cellSet.cell_leaves();

    //--- per-cell upload vertex/index buffers (transient) + one shared scratch ---
    struct CellBuild { ComPtr<ID3D12Resource> vtx, idx; uint8_t* vp = nullptr; uint8_t* ip = nullptr; };
    std::vector<CellBuild> cb(n);
    for (uint32_t k = 0; k < n; ++k) {
        const uint32_t K = cells[m_dirty[m_cursor + k]].leaf_count;
        cb[k].vtx = create_buffer(device, (uint64_t)K * CHUNK_VERTEX_BYTES, D3D12_RESOURCE_FLAG_NONE,
                                  D3D12_RESOURCE_STATE_GENERIC_READ, HEAP_UPLOAD);
        cb[k].idx = create_buffer(device, (uint64_t)K * CHUNK_INDEX_BYTES, D3D12_RESOURCE_FLAG_NONE,
                                  D3D12_RESOURCE_STATE_GENERIC_READ, HEAP_UPLOAD);
        D3D12_RANGE no_read{ 0, 0 };
        void* vp = nullptr; void* ip = nullptr;
        if (FAILED(cb[k].vtx->Map(0, &no_read, &vp)) || FAILED(cb[k].idx->Map(0, &no_read, &ip)))
            throw std::runtime_error("planet: GenerationBuilder upload-buffer Map failed");
        cb[k].vp = static_cast<uint8_t*>(vp);
        cb[k].ip = static_cast<uint8_t*>(ip);
    }
    ComPtr<ID3D12Resource> scratch = create_buffer(
        device, m_maxScratch, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, HEAP_DEFAULT);

    //--- tessellate every leaf of the batch in parallel (distinct destinations) ---
    struct LeafJob { uint32_t k; uint32_t slot; };
    std::vector<LeafJob> jobs;
    for (uint32_t k = 0; k < n; ++k) {
        const uint32_t K = cells[m_dirty[m_cursor + k]].leaf_count;
        for (uint32_t s = 0; s < K; ++s)
            jobs.push_back({ k, s });
    }
    workers.parallel_for((uint32_t)jobs.size(), [&](uint32_t j) {
        const LeafJob& job = jobs[j];
        const BlasCellSet::Cell& cell = cells[m_dirty[m_cursor + job.k]];
        const QuadNode leaf = unpack_node_id(leaves[cell.leaf_begin + job.slot]);

        uint8_t mask = 0;
        for (int e = 0; e < 4; ++e)
            if (m_gen.qt.neighbor_lod(leaf, (QuadEdge)e) < leaf.lod)
                mask |= (uint8_t)(1u << e);

        TessJob tj;
        tj.node            = leaf;
        tj.planet          = m_params.quadtree.planet;
        tj.anchor_world    = cell.anchor_world;          // all leaves of a cell share the anchor
        tj.grid            = CHUNK_GRID;
        tj.stitch_mask     = mask;
        tj.vertex_dest     = cb[job.k].vp + (uint64_t)job.slot * CHUNK_VERTEX_BYTES;
        tj.index_dest      = cb[job.k].ip + (uint64_t)job.slot * CHUNK_INDEX_BYTES;
        tj.vertex_capacity = CHUNK_VERTEX_BYTES;
        tj.index_capacity  = CHUNK_INDEX_BYTES;
        tessellate_chunk(tj, heightmap);
    });

    //--- record one multi-geometry BLAS build per cell ---
    std::vector<D3D12_RAYTRACING_GEOMETRY_DESC> geom(m_params.cells.max_leaves_per_cell);
    for (uint32_t k = 0; k < n; ++k) {
        const uint32_t cellIdx = m_dirty[m_cursor + k];
        const BlasCellSet::Cell& cell = cells[cellIdx];
        const uint32_t K = cell.leaf_count;

        ComPtr<ID3D12Resource> result = create_buffer(
            device, m_resultSize[K], D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, HEAP_DEFAULT);

        for (uint32_t g = 0; g < K; ++g) {
            D3D12_RAYTRACING_GEOMETRY_DESC& gd = geom[g];
            gd = {};
            gd.Type  = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
            gd.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
            gd.Triangles.VertexBuffer.StartAddress  =
                cb[k].vtx->GetGPUVirtualAddress() + (uint64_t)g * CHUNK_VERTEX_BYTES;
            gd.Triangles.VertexBuffer.StrideInBytes = CHUNK_VERTEX_STRIDE;
            gd.Triangles.VertexCount  = MAX_CHUNK_VERTS;
            gd.Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
            gd.Triangles.IndexBuffer  =
                cb[k].idx->GetGPUVirtualAddress() + (uint64_t)g * CHUNK_INDEX_BYTES;
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

        //the next cell reuses the one scratch buffer - serialise on it
        D3D12_RESOURCE_BARRIER uav = {};
        uav.Type          = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        uav.UAV.pResource = scratch.Get();
        compute_cl->ResourceBarrier(1, &uav);

        m_gen.cells[cellIdx].blas    = result;
        m_gen.cells[cellIdx].blas_va = result->GetGPUVirtualAddress();
    }

    //--- transient buffers join the pending batch; sealed by on_submitted() ---
    for (uint32_t k = 0; k < n; ++k) {
        m_pending.transient.push_back(cb[k].vtx);
        m_pending.transient.push_back(cb[k].idx);
    }
    m_pending.transient.push_back(scratch);
    m_pending.count += n;
    m_cursor += n;
    return n;
}

//====================================
//SUBMIT / RECLAIM / TAKE
//====================================
void GenerationBuilder::on_submitted(uint64_t compute_fence) {
    if (m_pending.count == 0) return;            // nothing recorded since the last seal
    m_pending.fence = compute_fence;
    m_batches.push_back(std::move(m_pending));
    m_pending = Batch{};
}

void GenerationBuilder::reclaim(uint64_t completed_fence) {
    size_t w = 0;
    for (size_t r = 0; r < m_batches.size(); ++r) {
        if (m_batches[r].fence <= completed_fence) {
            m_built += m_batches[r].count;       // these cells' BLAS builds have retired
            //m_batches[r].transient ComPtrs drop here -> the buffers are freed
        } else {
            m_batches[w++] = std::move(m_batches[r]);
        }
    }
    m_batches.resize(w);
}

Generation GenerationBuilder::take() {
    m_active = false;
    Generation g = std::move(m_gen);
    m_gen = Generation{};
    m_dirty.clear();
    m_cursor = 0;
    m_built  = 0;
    m_batches.clear();
    m_pending = Batch{};
    return g;
}

} // namespace planet
