#pragma once

#include <cstdint>
#include <vector>
#include <algorithm>
#include "chunk_mesh.h"

namespace planet {

constexpr uint32_t TERRAIN_LEAF_VERTS   = MAX_CHUNK_VERTS;
constexpr uint32_t TERRAIN_LEAF_INDICES = MAX_CHUNK_TRIS * 3u;

constexpr uint32_t TERRAIN_GEO_INVALID = 0xFFFFFFFFu;

class TerrainGeoPool {
public:
    // Leaf ranges within the combined scene buffers.
    void init(uint32_t vbase_elems, uint32_t ibase_elems, uint32_t capacity_leaves) {
        m_vbase    = vbase_elems;
        m_ibase    = ibase_elems;
        m_capacity = capacity_leaves;
        m_free.assign(1, Span{ 0u, capacity_leaves });
        m_used = 0;
    }

    // Best fit.
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

    void free(uint32_t off, uint32_t k) {
        if (k == 0 || off == TERRAIN_GEO_INVALID) return;
        m_used -= k;
        size_t i = 0;
        while (i < m_free.size() && m_free[i].first < off) ++i;
        m_free.insert(m_free.begin() + (long)i, Span{ off, k });
        if (i > 0 && m_free[i - 1].first + m_free[i - 1].count == m_free[i].first) {
            m_free[i - 1].count += m_free[i].count;
            m_free.erase(m_free.begin() + (long)i);
            --i;
        }
        if (i + 1 < m_free.size() &&
            m_free[i].first + m_free[i].count == m_free[i + 1].first) {
            m_free[i].count += m_free[i + 1].count;
            m_free.erase(m_free.begin() + (long)(i + 1));
        }
    }

    uint32_t vertex_base_elems(uint32_t leaf_off) const {
        return m_vbase + leaf_off * TERRAIN_LEAF_VERTS;
    }
    uint32_t index_base_elems(uint32_t leaf_off) const {
        return m_ibase + leaf_off * TERRAIN_LEAF_INDICES;
    }

    uint32_t free_leaves()     const { return m_capacity - m_used; }

private:
    struct Span { uint32_t first; uint32_t count; };
    std::vector<Span> m_free;
    uint32_t m_vbase    = 0;
    uint32_t m_ibase    = 0;
    uint32_t m_capacity = 0;
    uint32_t m_used     = 0;
};

}
