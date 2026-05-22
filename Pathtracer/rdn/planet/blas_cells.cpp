//====================================
//PLANET - BLAS CELLS
//====================================

#include "blas_cells.h"

namespace planet {

//====================================
//CELL CUT
//====================================
void BlasCellSet::collect_leaves(const QuadNode& node, const RestrictedQuadtree& qt,
                                 std::vector<uint64_t>& out) const {
    const uint64_t id = pack_node_id(node);
    if (qt.is_leaf(id)) { out.push_back(id); return; }
    if (node.lod >= MAX_LOD) return;                  // a valid cut always terminates earlier
    for (int q = 0; q < 4; ++q)
        collect_leaves(child_node(node, q), qt, out);
}

void BlasCellSet::cut_descend(const QuadNode& node, const RestrictedQuadtree& qt,
                              const CellCutParams& p) {
    const uint64_t     id   = pack_node_id(node);
    const bool         leaf = qt.is_leaf(id);
    const NodeGeometry g    = compute_node_geometry(node, p.planet);

    //gather this node's subtree leaves at the tail of m_cellLeaves
    const uint32_t start = (uint32_t)m_cellLeaves.size();
    collect_leaves(node, qt, m_cellLeaves);
    const uint32_t count = (uint32_t)m_cellLeaves.size() - start;

    //a single leaf is always its own cell; otherwise a node becomes a cell once
    //its subtree is small enough AND spatially compact enough for FP32.
    const bool emit = leaf
        || (count <= p.max_leaves_per_cell && g.bounding_radius <= p.max_cell_radius_m);

    if (emit) {
        const uint32_t idx = (uint32_t)m_cells.size();
        m_cells.push_back(Cell{ id, g.center_world, start, count });
        m_cellSet[id] = idx;
        for (uint32_t i = 0; i < count; ++i)
            m_leafToCell[m_cellLeaves[start + i]] = idx;
    } else {
        m_cellLeaves.resize(start);                   // discard; the children re-collect
        for (int q = 0; q < 4; ++q)
            cut_descend(child_node(node, q), qt, p);
    }
}

void BlasCellSet::build(const RestrictedQuadtree& qt, const CellCutParams& params) {
    CellCutParams p = params;
    if (p.max_leaves_per_cell == 0) p.max_leaves_per_cell = 1;

    m_cells.clear();
    m_cellLeaves.clear();
    m_cellSet.clear();
    m_leafToCell.clear();

    for (uint8_t f = 0; f < CUBE_FACES; ++f)
        cut_descend(QuadNode{ f, 0, 0, 0 }, qt, p);
}

int BlasCellSet::cell_index(uint64_t node_id) const {
    const auto it = m_cellSet.find(node_id);
    return it == m_cellSet.end() ? -1 : (int)it->second;
}

int BlasCellSet::cell_of_leaf(uint64_t leaf_id) const {
    const auto it = m_leafToCell.find(leaf_id);
    return it == m_leafToCell.end() ? -1 : (int)it->second;
}

//====================================
//GENERATION DIFF
//====================================
void diff_generations(const RestrictedQuadtree& live_qt,   const BlasCellSet& live_cells,
                      const RestrictedQuadtree& target_qt, const BlasCellSet& target_cells,
                      std::vector<uint8_t>& out_dirty) {
    const std::vector<BlasCellSet::Cell>& cells = target_cells.cells();
    out_dirty.assign(cells.size(), 0);

    //(1) a TARGET cell that is not a LIVE cell has no BLAS to reuse - build it.
    for (size_t i = 0; i < cells.size(); ++i)
        if (!live_cells.is_cell(cells[i].node_id))
            out_dirty[i] = 1;

    //(2) a changed TARGET leaf dirties its own cell and the cells of every leaf
    //bordering it - those cells' seam stitching depended on the changed leaf.
    auto mark = [&](int cell_idx) {
        if (cell_idx >= 0) out_dirty[(size_t)cell_idx] = 1;
    };
    for (uint64_t leaf_id : target_qt.leaves()) {
        if (live_qt.is_leaf(leaf_id)) continue;              // leaf unchanged
        mark(target_cells.cell_of_leaf(leaf_id));
        const QuadNode L = unpack_node_id(leaf_id);
        for (int e = 0; e < 4; ++e) {
            uint64_t adj[2];
            const int n = target_qt.adjacent_leaves(L, (QuadEdge)e, adj);
            for (int k = 0; k < n; ++k)
                mark(target_cells.cell_of_leaf(adj[k]));
        }
    }
}

} // namespace planet
