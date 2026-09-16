#pragma once

#include <cstdint>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include "coordinate_system.h"
#include "cube_sphere.h"
#include "restricted_quadtree.h"

namespace planet {

struct CellCutParams {
    PlanetGeometry planet{};
    uint32_t max_leaves_per_cell = 8;
    double   max_cell_radius_m   = 1e30;
};

class BlasCellSet {
public:
    struct Cell {
        uint64_t node_id    = INVALID_NODE;
        DVec3    anchor_world{};
        uint32_t leaf_begin = 0;
        uint32_t leaf_count = 0;
    };

    void build(const RestrictedQuadtree& qt, const CellCutParams& params);

    const std::vector<Cell>&     cells()       const { return m_cells; }
    const std::vector<uint64_t>& cell_leaves() const { return m_cellLeaves; }
    uint32_t cell_count() const { return (uint32_t)m_cells.size(); }

    bool is_cell(uint64_t node_id) const { return m_cellSet.contains(node_id); }
    int  cell_index(uint64_t node_id) const;
    int  cell_of_leaf(uint64_t leaf_id) const;

private:
    void cut_descend(const QuadNode& node, const RestrictedQuadtree& qt,
                     const CellCutParams& p);
    void collect_leaves(const QuadNode& node, const RestrictedQuadtree& qt,
                        std::vector<uint64_t>& out) const;

    std::vector<Cell>     m_cells;
    std::vector<uint64_t> m_cellLeaves;
    std::unordered_map<uint64_t, uint32_t> m_cellSet;
    std::unordered_map<uint64_t, uint32_t> m_leafToCell;
};

void diff_generations(const RestrictedQuadtree& live_qt,   const BlasCellSet& live_cells,
                      const RestrictedQuadtree& target_qt, const BlasCellSet& target_cells,
                      std::vector<uint8_t>& out_dirty);

}
