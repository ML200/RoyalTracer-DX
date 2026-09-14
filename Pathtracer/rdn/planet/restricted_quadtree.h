#pragma once

#include <cstdint>
#include <vector>
#include <unordered_set>
#include "coordinate_system.h"
#include "cube_sphere.h"

namespace planet {

class IHeightmapSource;

struct QuadtreeParams {
    PlanetGeometry planet{};
    uint8_t  min_lod    = 0;
    uint8_t  max_lod    = MAX_LOD;
    uint32_t max_leaves = 8192;
};

class RestrictedQuadtree {
public:
    // Selects balanced detail under the configured triangle budget.
    void select(const QuadtreeParams& params, const CameraView& cam,
                const IHeightmapSource* heightmap = nullptr);

    const std::vector<uint64_t>& leaves() const { return m_leaves; }
    uint32_t leaf_count() const { return (uint32_t)m_leaves.size(); }

    bool is_leaf(uint64_t node_id) const { return m_set.contains(node_id); }

    // Returns neighboring detail, including cube-face seam handling.
    uint8_t neighbor_lod(const QuadNode& leaf, QuadEdge edge) const;

    int adjacent_leaves(const QuadNode& leaf, QuadEdge edge, uint64_t out[2]) const;

    uint8_t corner_lod(const QuadNode& leaf, int corner_idx) const;

    int corner_leaf(const QuadNode& leaf, int corner_idx, uint64_t& out) const;

private:
    void balance();
    bool covering_leaf(const QuadNode& m, QuadNode& out) const;

    void do_split      (const QuadNode& n, std::vector<uint64_t>& record, uint32_t& leaf_count);
    void ensure_lod    (const QuadNode& region, uint8_t target,
                        std::vector<uint64_t>& record, uint32_t& leaf_count);
    void balanced_split(const QuadNode& N, std::vector<uint64_t>& record, uint32_t& leaf_count);

    std::unordered_set<uint64_t> m_set;
    std::vector<uint64_t>        m_leaves;
};

}
