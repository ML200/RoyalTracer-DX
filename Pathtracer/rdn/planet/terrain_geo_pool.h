#pragma once
//====================================
//PLANET - TERRAIN GEOMETRY POOL
//====================================
//Persistent leaf-slot allocator over the TERRAIN REGION of the unified
//(scene + terrain) global vertex / index buffers. The atomic unit is one
//"leaf slot" = MAX_CHUNK_VERTS vertices / MAX_CHUNK_TRIS*3 indices, the fixed
//mesh of a single quadtree leaf.
//
//A terrain CELL (1..max_leaves_per_cell leaves) becomes ONE BLAS geometry, so
//its vertices and indices must be CONTIGUOUS: a K-leaf cell takes K contiguous
//leaf slots. The cell's index buffer holds ABSOLUTE int indices into the
//unified vertex buffer, exactly like a scene mesh - so EvalSurfaceState /
//FlatPrimID shade it with no terrain branch (single opaque geometry,
//opaqueTriCount = K*MAX_CHUNK_TRIS).
//
//Lifetime: a slot is allocated when a dirty cell's geometry is built and freed
//when that geometry is no longer referenced by any live / in-flight / retiring
//generation (the orchestrator drives free()). Clean cells reused across a
//ping-pong swap keep their slot (the CellInstance carries the offset, copied
//with the shared BLAS).
//
//Pure CPU bookkeeping - the actual GPU bytes live in Scene's combined buffers;
//this only hands out element offsets into the terrain region.

#include <cstdint>
#include <vector>
#include <algorithm>
#include "chunk_mesh.h"

namespace planet {

//element counts of one leaf slot in the unified buffers.
constexpr uint32_t TERRAIN_LEAF_VERTS   = MAX_CHUNK_VERTS;        // 1089
constexpr uint32_t TERRAIN_LEAF_INDICES = MAX_CHUNK_TRIS * 3u;    // 6144

constexpr uint32_t TERRAIN_GEO_INVALID = 0xFFFFFFFFu;

class TerrainGeoPool {
public:
    //vbase_elems / ibase_elems = element offsets where the terrain region
    //begins in the combined vertex / index buffers (i.e. the scene vertex /
    //index counts). The scene vertex and index counts differ, so the two bases
    //are independent.
    //capacity   = number of leaf slots the terrain region holds.
    void init(uint32_t vbase_elems, uint32_t ibase_elems, uint32_t capacity_leaves) {
        m_vbase    = vbase_elems;
        m_ibase    = ibase_elems;
        m_capacity = capacity_leaves;
        m_free.assign(1, Span{ 0u, capacity_leaves });
        m_used = 0;
    }

    //Allocate k contiguous leaf slots. Returns the region-relative leaf offset,
    //or TERRAIN_GEO_INVALID if no span is large enough.
    //
    //BEST-FIT (smallest span that fits): small cells fill small gaps and large
    //contiguous spans stay available for large cells. First-fit instead carved
    //the largest spans for small cells, fragmenting the pool until a big cell
    //couldn't find a contiguous run even with plenty of total free slots - which
    //dropped cells (-> holes) under fast-flight churn. Keep max_leaves_per_cell
    //modest too, so k stays small relative to typical free spans.
    uint32_t allocate(uint32_t k) {
        if (k == 0) k = 1;
        size_t   best      = (size_t)-1;
        uint32_t best_count = 0xFFFFFFFFu;
        for (size_t i = 0; i < m_free.size(); ++i) {
            if (m_free[i].count >= k && m_free[i].count < best_count) {
                best = i;
                best_count = m_free[i].count;
            }
        }
        if (best == (size_t)-1) return TERRAIN_GEO_INVALID;
        const uint32_t off = m_free[best].first;
        m_free[best].first += k;
        m_free[best].count -= k;
        if (m_free[best].count == 0) m_free.erase(m_free.begin() + (long)best);
        m_used += k;
        return off;
    }

    //Allocate up to max_k contiguous leaf slots from the LARGEST free span (for
    //the split fallback: when allocate(K) can't fit a whole cell, carve it into
    //a few sub-cells that each take the biggest run currently available). Sets
    //out_count to how many were taken (1..max_k); returns TERRAIN_GEO_INVALID
    //with out_count=0 only if the pool is completely full. Because it always
    //takes from a real free span, repeated calls place every leaf as long as
    //free_leaves() >= the total needed - so a fragmented pool never holes.
    uint32_t allocate_upto(uint32_t max_k, uint32_t& out_count) {
        if (max_k == 0) max_k = 1;
        size_t   best       = (size_t)-1;
        uint32_t best_count = 0;
        for (size_t i = 0; i < m_free.size(); ++i)
            if (m_free[i].count > best_count) { best = i; best_count = m_free[i].count; }
        if (best == (size_t)-1) { out_count = 0; return TERRAIN_GEO_INVALID; }
        const uint32_t take = best_count < max_k ? best_count : max_k;
        const uint32_t off  = m_free[best].first;
        m_free[best].first += take;
        m_free[best].count -= take;
        if (m_free[best].count == 0) m_free.erase(m_free.begin() + (long)best);
        m_used   += take;
        out_count = take;
        return off;
    }

    //Return k leaf slots starting at region-relative offset `off`. Inserts the
    //span sorted by offset and coalesces with adjacent free spans.
    void free(uint32_t off, uint32_t k) {
        if (k == 0 || off == TERRAIN_GEO_INVALID) return;
        m_used -= k;
        //find insertion point (free list kept sorted by first)
        size_t i = 0;
        while (i < m_free.size() && m_free[i].first < off) ++i;
        m_free.insert(m_free.begin() + (long)i, Span{ off, k });
        //coalesce with previous
        if (i > 0 && m_free[i - 1].first + m_free[i - 1].count == m_free[i].first) {
            m_free[i - 1].count += m_free[i].count;
            m_free.erase(m_free.begin() + (long)i);
            --i;
        }
        //coalesce with next
        if (i + 1 < m_free.size() &&
            m_free[i].first + m_free[i].count == m_free[i + 1].first) {
            m_free[i].count += m_free[i + 1].count;
            m_free.erase(m_free.begin() + (long)(i + 1));
        }
    }

    //element offsets into the combined buffers for a region-relative leaf offset.
    uint32_t vertex_base_elems(uint32_t leaf_off) const {
        return m_vbase + leaf_off * TERRAIN_LEAF_VERTS;
    }
    uint32_t index_base_elems(uint32_t leaf_off) const {
        return m_ibase + leaf_off * TERRAIN_LEAF_INDICES;
    }

    uint32_t capacity_leaves() const { return m_capacity; }
    uint32_t used_leaves()     const { return m_used; }
    uint32_t free_leaves()     const { return m_capacity - m_used; }

private:
    struct Span { uint32_t first; uint32_t count; };   // leaf-slot range, region-relative
    std::vector<Span> m_free;                          // sorted by first, coalesced
    uint32_t m_vbase    = 0;                            // terrain region start, vertex elems
    uint32_t m_ibase    = 0;                            // terrain region start, index elems
    uint32_t m_capacity = 0;
    uint32_t m_used     = 0;
};

} // namespace planet
