#pragma once
//====================================
//PLANET - RESTRICTED QUADTREE
//====================================
//Triangle-budget LOD selection over the cube-sphere quadtree, producing a
//RESTRICTED (balanced) cut: a set of leaf nodes covering the whole closed
//sphere in which any two edge-adjacent leaves differ by at most one LOD level.
//
//That <=1 LOD-delta rule is what makes the mesh stitchable into a crack-free
//surface - the seam-aware tessellator then only ever has to bridge a 1:2 edge.
//
//select() is a BALANCE-PRESERVING budget greedy: from the 6 cube faces it
//repeatedly splits the highest screen-error leaf, and each split also pulls up
//the coarser neighbours the <=1 LOD-delta rule demands - the whole step
//counted against the max_leaves budget. A step that would exceed the budget is
//rolled back and refinement stops, so the cut is ALWAYS balanced (crack-free)
//AND within budget. Detail lands on the highest-error leaves first - always
//the ones nearest the camera - so it is spent fine-near / coarse-far
//automatically. No horizon cull: the far side stays in the mesh (low error,
//never refined) so the surface is a closed manifold.
//Pure CPU; runs once per generation rebuild on a worker thread.

#include <cstdint>
#include <vector>
#include <unordered_set>
#include "coordinate_system.h"
#include "cube_sphere.h"

namespace planet {

class IHeightmapSource;   // optional terrain displacement for the LOD distance metric

//====================================
//PARAMS
//====================================
struct QuadtreeParams {
    PlanetGeometry planet{};
    uint8_t  min_lod    = 0;          // coarsest a leaf may be (far side)
    uint8_t  max_lod    = MAX_LOD;    // finest a leaf may be (under the camera)
    //Triangle budget expressed as a leaf cap (max_triangles / MAX_CHUNK_TRIS).
    //select() refines toward this; the cut never exceeds it.
    uint32_t max_leaves = 8192;
};

//====================================
//RESTRICTED QUADTREE
//====================================
class RestrictedQuadtree {
public:
    //balance-preserving budget-greedy refinement for this camera; replaces the
    //previous cut. The result is always restricted (<=1 LOD-delta, crack-free)
    //and within the max_leaves budget.
    //
    //'heightmap' (optional) displaces each node's centre/corners by the terrain
    //elevation when computing screen error, so LOD tracks the ACTUAL displaced
    //surface rather than the analytical sphere - the ground under a camera
    //standing on a mountain refines correctly instead of reading as `height`
    //metres away. Null = analytical sphere (unit tests).
    void select(const QuadtreeParams& params, const CameraView& cam,
                const IHeightmapSource* heightmap = nullptr);

    //the balanced leaf set (packed node_ids), ascending.
    const std::vector<uint64_t>& leaves() const { return m_leaves; }
    uint32_t leaf_count() const { return (uint32_t)m_leaves.size(); }

    //is 'node_id' (packed) one of the cut's leaves?
    bool is_leaf(uint64_t node_id) const { return m_set.contains(node_id); }

    //LOD of the leaf across one edge of 'leaf': leaf.lod for a 1:1 neighbour,
    //leaf.lod-1 if it is coarser, leaf.lod+1 if finer. After select() the
    //result is always within 1 of leaf.lod (the restricted-quadtree invariant).
    uint8_t neighbor_lod(const QuadNode& leaf, QuadEdge edge) const;

    //Leaves across one edge of 'leaf': 1 for a same-LOD or coarser neighbour,
    //2 for a finer one. Fills terrain[] with packed node_ids and returns the count.
    //Used by the generation diff to find cells a changed leaf's seam affects.
    int adjacent_leaves(const QuadNode& leaf, QuadEdge edge, uint64_t out[2]) const;

    //LOD of the leaf at the diagonal of one CORNER of 'leaf'. corner_idx
    //encoding matches child quadrant order: bit 0 = +s, bit 1 = +t. The
    //diagonal is reached by composing two neighbor_node steps (handles cube-
    //face seam crossings since neighbor_node does). Returns the lod of the
    //covering leaf at that diagonal position, or leaf.lod+1 if the diagonal
    //has been subdivided into finer leaves. Edge balance only enforces 1-LOD
    //diffs between EDGE neighbours; the diagonal can be coarser without any
    //edge bit being set, so corner-only LOD differences need their own seam
    //bookkeeping to avoid sub-metre cracks at the shared corner vertex.
    uint8_t corner_lod(const QuadNode& leaf, int corner_idx) const;

    //Leaf at the diagonal corner of 'leaf'. Returns 1 with `out` set to the
    //covering leaf's packed id, or 1 with `out` set to the finer leaf that
    //actually touches the corner POINT (the (3 ^ corner_idx)-quadrant chain
    //of the diagonal node, since the diagonal chunk's corner touching us is
    //the OPPOSITE quadrant of its own subdivision). Returns 0 only at cube
    //vertices where the topology breaks (3 faces meet, not 4) - callers
    //should treat that case as "no extra cell to mark".
    int corner_leaf(const QuadNode& leaf, int corner_idx, uint64_t& out) const;

private:
    void balance();   //safety net only - select()'s greedy keeps the invariant
    //leaf covering node 'm' (walking m upward); false if m's region is
    //subdivided into finer leaves, i.e. no covering leaf exists at or above m.
    bool covering_leaf(const QuadNode& m, QuadNode& out) const;

    //balance-preserving refinement used by select(): do_split records a split
    //for rollback; ensure_lod refines a region's covering leaf up to a target
    //lod, cascading into coarser neighbours; balanced_split does one greedy
    //split plus the cascade the <=1 LOD-delta invariant demands.
    void do_split      (const QuadNode& n, std::vector<uint64_t>& record, uint32_t& leaf_count);
    void ensure_lod    (const QuadNode& region, uint8_t target,
                        std::vector<uint64_t>& record, uint32_t& leaf_count);
    void balanced_split(const QuadNode& N, std::vector<uint64_t>& record, uint32_t& leaf_count);

    std::unordered_set<uint64_t> m_set;       // membership; mutated during refinement
    std::vector<uint64_t>        m_leaves;    // finalised, sorted leaf list
};

} // namespace planet
