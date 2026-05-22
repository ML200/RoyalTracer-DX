//====================================
//PLANET - RESTRICTED QUADTREE
//====================================

#include "restricted_quadtree.h"
#include <algorithm>
#include <queue>

namespace planet {

//====================================
//BALANCE-PRESERVING REFINEMENT PRIMITIVES
//====================================
//split one leaf into its 4 children. Records the parent (so a whole step can
//be rolled back) and bumps the running leaf count (1 leaf -> 4 = +3).
void RestrictedQuadtree::do_split(const QuadNode& n, std::vector<uint64_t>& record,
                                  uint32_t& leaf_count) {
    m_set.erase(pack_node_id(n));
    for (int q = 0; q < 4; ++q)
        m_set.insert(pack_node_id(child_node(n, q)));
    record.push_back(pack_node_id(n));
    leaf_count += 3u;
}

//refine the leaf covering 'region' until it reaches lod 'target', splitting
//any too-coarse neighbour FIRST so the <=1 LOD-delta invariant holds at every
//step. 'region' is always a node whose own lod equals 'target', so this never
//has to refine past it. Terminates: the recursion's target strictly decreases
//going outward, and every split only ever refines.
void RestrictedQuadtree::ensure_lod(const QuadNode& region, uint8_t target,
                                    std::vector<uint64_t>& record, uint32_t& leaf_count) {
    while (true) {
        QuadNode leaf;
        if (!covering_leaf(region, leaf)) return;   // region already finer than target
        if (leaf.lod >= target)           return;   // coarse-enough already
        //to split 'leaf', its edge-neighbours must be no more than one level
        //coarser - pull them up to leaf.lod first.
        for (int e = 0; e < 4; ++e)
            ensure_lod(neighbor_node(leaf, (QuadEdge)e), leaf.lod, record, leaf_count);
        do_split(leaf, record, leaf_count);
    }
}

//split N plus every cascade split the restricted invariant demands. The tree
//must be balanced on entry; it is balanced again on return.
void RestrictedQuadtree::balanced_split(const QuadNode& N, std::vector<uint64_t>& record,
                                        uint32_t& leaf_count) {
    //N's children land at N.lod+1, so N's edge-neighbours must be >= N.lod.
    for (int e = 0; e < 4; ++e)
        ensure_lod(neighbor_node(N, (QuadEdge)e), N.lod, record, leaf_count);
    do_split(N, record, leaf_count);
}

//====================================
//SELECT - balance-preserving greedy triangle-budget refinement
//====================================
void RestrictedQuadtree::select(const QuadtreeParams& params, const CameraView& cam) {
    QuadtreeParams p = params;
    if (p.max_lod > MAX_LOD)   p.max_lod = MAX_LOD;
    if (p.min_lod > p.max_lod) p.min_lod = p.max_lod;

    m_set.clear();

    //screen-error of a node: longest world edge over distance to the camera's
    //near point. LOD is camera-POSITION only (no view direction): a path
    //tracer's GI / reflection rays leave the frustum, so a coarse off-screen
    //patch would still be hit.
    auto screen_error = [&](const QuadNode& n) -> double {
        const NodeGeometry g = compute_node_geometry(n, p.planet);
        const Vec3f  rel       = to_camera_relative(g.center_world, cam.position_world);
        const double dist      = double(length(rel));
        const double near_dist = dist > g.bounding_radius ? dist - g.bounding_radius : 1e-3;
        return g.edge_length / near_dist;
    };

    //--- seed: 6 cube faces, uniformly refined down to the min_lod floor ---
    for (uint8_t f = 0; f < CUBE_FACES; ++f)
        m_set.insert(pack_node_id(QuadNode{ f, 0, 0, 0 }));
    for (uint8_t L = 0; L < p.min_lod; ++L) {
        const std::vector<uint64_t> cur(m_set.begin(), m_set.end());
        m_set.clear();
        for (uint64_t id : cur) {
            const QuadNode n = unpack_node_id(id);
            for (int q = 0; q < 4; ++q)
                m_set.insert(pack_node_id(child_node(n, q)));
        }
    }
    uint32_t leaf_count = (uint32_t)m_set.size();

    //--- greedy budget refinement, balance-preserving ---
    //Pop the highest screen-error leaf and BALANCED-split it: the split plus
    //every coarser-neighbour split the <=1 LOD-delta invariant demands, all
    //counted against the budget. A step that would exceed max_leaves is rolled
    //back and refinement stops - so the cut is always restricted (crack-free)
    //AND within budget. The far side keeps a low error and is never reached,
    //so the closed sphere is preserved without a horizon cull.
    struct Item { double err; QuadNode node; };
    struct ErrLess { bool operator()(const Item& a, const Item& b) const { return a.err < b.err; } };
    std::priority_queue<Item, std::vector<Item>, ErrLess> pq;
    for (uint64_t id : m_set) {
        const QuadNode n = unpack_node_id(id);
        pq.push({ screen_error(n), n });
    }

    std::vector<uint64_t> record;   // nodes split in the current step (rollback log)
    while (!pq.empty()) {
        const QuadNode N = pq.top().node;
        pq.pop();
        if (!is_leaf(pack_node_id(N))) continue;   // already split by a cascade
        if (N.lod >= p.max_lod)        continue;   // at the depth cap - a final leaf

        const uint32_t leaf_count_before = leaf_count;
        record.clear();
        balanced_split(N, record, leaf_count);

        if (leaf_count <= p.max_leaves) {
            //commit: enqueue every leaf this step newly created.
            for (uint64_t pid : record) {
                const QuadNode parent = unpack_node_id(pid);
                for (int q = 0; q < 4; ++q) {
                    const QuadNode c = child_node(parent, q);
                    if (is_leaf(pack_node_id(c)))
                        pq.push({ screen_error(c), c });
                }
            }
        } else {
            //over budget: undo the whole step (erase children, restore parent,
            //in reverse order) and stop - the cut so far is balanced and
            //within budget.
            for (size_t i = record.size(); i-- > 0; ) {
                const QuadNode parent = unpack_node_id(record[i]);
                for (int q = 0; q < 4; ++q)
                    m_set.erase(pack_node_id(child_node(parent, q)));
                m_set.insert(record[i]);
            }
            leaf_count = leaf_count_before;
            break;
        }
    }

    //safety net: the greedy keeps the cut balanced, so this is normally a
    //no-op. If it ever splits anything, the greedy left an imbalance - the
    //terrain still comes out crack-free, and triangle_count creeping past
    //max_triangles is the signal that something regressed.
    balance();

    m_leaves.assign(m_set.begin(), m_set.end());
    std::sort(m_leaves.begin(), m_leaves.end());
}

//====================================
//BALANCE - refine until edge-adjacent leaves differ by <= 1 LOD
//====================================
bool RestrictedQuadtree::covering_leaf(const QuadNode& m, QuadNode& out) const {
    QuadNode c = m;
    while (true) {
        if (is_leaf(pack_node_id(c))) { out = c; return true; }
        if (c.lod == 0) return false;                     // m's region is subdivided finer
        c = parent_node(c);
    }
}

void RestrictedQuadtree::balance() {
    //Worklist of leaves. A leaf demands every edge-neighbour be no more than
    //one level coarser; splitting a too-coarse neighbour enqueues its four
    //children, which may in turn split THEIR neighbours. Splits only ever
    //refine and lod is capped at max_lod, so this terminates.
    std::vector<uint64_t> work(m_set.begin(), m_set.end());

    while (!work.empty()) {
        const uint64_t id = work.back();
        work.pop_back();
        if (!is_leaf(id)) continue;                       // split already since enqueued

        const QuadNode L = unpack_node_id(id);
        for (int e = 0; e < 4; ++e) {
            while (true) {
                QuadNode c;
                if (!covering_leaf(neighbor_node(L, (QuadEdge)e), c)) break;  // neighbour finer
                if (c.lod + 1 >= L.lod) break;            // within one level - done

                m_set.erase(pack_node_id(c));
                for (int q = 0; q < 4; ++q) {
                    const uint64_t child = pack_node_id(child_node(c, q));
                    m_set.insert(child);
                    work.push_back(child);
                }
            }
        }
    }
}

//====================================
//NEIGHBOUR LOD QUERY
//====================================
uint8_t RestrictedQuadtree::neighbor_lod(const QuadNode& leaf, QuadEdge edge) const {
    QuadNode c;
    if (covering_leaf(neighbor_node(leaf, edge), c))
        return c.lod;                                     // 1:1, or neighbour coarser
    return uint8_t(leaf.lod + 1);                         // neighbour subdivided - finer
}

//====================================
//ADJACENT LEAVES (for the generation diff)
//====================================
namespace {
//the two child quadrants of a node touching each edge (quadrant order:
//0=(0,0) 1=(1,0) 2=(0,1) 3=(1,1)).
constexpr int EDGE_CHILDREN[4][2] = {
    { 0, 2 },   // EDGE_NEG_S : x-local 0
    { 1, 3 },   // EDGE_POS_S : x-local 1
    { 0, 1 },   // EDGE_NEG_T : y-local 0
    { 2, 3 },   // EDGE_POS_T : y-local 1
};
} // namespace

int RestrictedQuadtree::adjacent_leaves(const QuadNode& leaf, QuadEdge edge,
                                        uint64_t out[2]) const {
    QuadNode c;
    if (covering_leaf(neighbor_node(leaf, edge), c)) {
        out[0] = pack_node_id(c);                         // 1:1 or coarser - a single leaf
        return 1;
    }
    //neighbour is finer: one leaf across the edge from each of 'leaf's two
    //edge-children.
    const QuadNode a0 = neighbor_node(child_node(leaf, EDGE_CHILDREN[edge][0]), edge);
    const QuadNode a1 = neighbor_node(child_node(leaf, EDGE_CHILDREN[edge][1]), edge);
    out[0] = pack_node_id(a0);
    out[1] = pack_node_id(a1);
    return 2;
}

} // namespace planet
