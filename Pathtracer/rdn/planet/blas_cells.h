#pragma once
//====================================
//PLANET - BLAS CELLS
//====================================
//Cuts a restricted-quadtree leaf set into BLAS cells: a coarser cut where each
//cell is a quadtree node whose subtree holds at most max_leaves_per_cell leaves
//(every leaf tessellates to the same triangle count, so a leaf-count cap is a
//triangle-count cap) and whose extent stays within max_cell_radius_m (so the
//cell's vertices keep FP32 precision relative to the cell anchor). One cell ->
//one BLAS. A cell is a world-anchored quadtree node, so its identity is stable
//across generations - that is what makes ping-pong cell reuse possible.
//
//diff_generations() classifies each cell of a new (TARGET) generation as dirty
//(must be re-tessellated and rebuilt) or clean (reuse the LIVE generation's
//BLAS): a cell is dirty if it is not a LIVE cell, or if a leaf inside it or
//bordering it changed LOD (the seam stitch is baked into the cell mesh).
//
//Pure CPU; runs once per generation rebuild on a worker thread.

#include <cstdint>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include "coordinate_system.h"
#include "cube_sphere.h"
#include "restricted_quadtree.h"

namespace planet {

//====================================
//PARAMS
//====================================
struct CellCutParams {
    PlanetGeometry planet{};
    uint32_t max_leaves_per_cell = 8;        // BLAS triangle budget / MAX_CHUNK_TRIS
    double   max_cell_radius_m   = 1e30;     // FP32-precision cap on a cell's extent
};

//====================================
//BLAS CELL SET
//====================================
class BlasCellSet {
public:
    struct Cell {
        uint64_t node_id    = INVALID_NODE;  // the cell's quadtree node (stable identity)
        DVec3    anchor_world{};             // BLAS local-frame origin (node centre)
        uint32_t leaf_begin = 0;             // [leaf_begin, leaf_begin+leaf_count) into cell_leaves()
        uint32_t leaf_count = 0;
    };

    //cut the quadtree's leaf set into cells; replaces the previous cut.
    void build(const RestrictedQuadtree& qt, const CellCutParams& params);

    const std::vector<Cell>&     cells()       const { return m_cells; }
    const std::vector<uint64_t>& cell_leaves() const { return m_cellLeaves; }
    uint32_t cell_count() const { return (uint32_t)m_cells.size(); }

    //is 'node_id' (packed) the node of one of the cells?
    bool is_cell(uint64_t node_id) const { return m_cellSet.contains(node_id); }
    //index into cells() of the cell with this node, or -1.
    int  cell_index(uint64_t node_id) const;
    //index into cells() of the cell owning 'leaf_id', or -1.
    int  cell_of_leaf(uint64_t leaf_id) const;

private:
    void cut_descend(const QuadNode& node, const RestrictedQuadtree& qt,
                     const CellCutParams& p);
    void collect_leaves(const QuadNode& node, const RestrictedQuadtree& qt,
                        std::vector<uint64_t>& out) const;

    std::vector<Cell>     m_cells;
    std::vector<uint64_t> m_cellLeaves;                       // flattened per-cell leaf lists
    std::unordered_map<uint64_t, uint32_t> m_cellSet;         // cell node_id -> cell index
    std::unordered_map<uint64_t, uint32_t> m_leafToCell;      // leaf node_id -> cell index
};

//====================================
//GENERATION DIFF
//====================================
//Classify each TARGET cell: out_dirty[i] (parallel to target_cells.cells()) is
//1 if cell i must be rebuilt, 0 if the LIVE generation's BLAS can be reused.
void diff_generations(const RestrictedQuadtree& live_qt,   const BlasCellSet& live_cells,
                      const RestrictedQuadtree& target_qt, const BlasCellSet& target_cells,
                      std::vector<uint8_t>& out_dirty);

} // namespace planet
