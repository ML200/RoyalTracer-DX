//====================================
//PLANET - RESTRICTED QUADTREE
//====================================

#include "restricted_quadtree.h"
#include "heightmap_source.h"
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
void RestrictedQuadtree::select(const QuadtreeParams& params, const CameraView& cam,
                                const IHeightmapSource* heightmap) {
    QuadtreeParams p = params;
    if (p.max_lod > MAX_LOD)   p.max_lod = MAX_LOD;
    if (p.min_lod > p.max_lod) p.min_lod = p.max_lod;

    m_set.clear();

    //screen-error of a node: longest world edge over distance to the camera's
    //near point. LOD is camera-POSITION only (no view direction): a path
    //tracer's GI / reflection rays leave the frustum, so a coarse off-screen
    //patch would still be hit.
    //
    //The node geometry is placed on the DISPLACED (heightmap) surface, not the
    //analytical sphere - otherwise a camera standing on relief sees the ground
    //under it as `height` metres away and never refines it. We sample the
    //heightmap at the node centre + 4 corners and rebuild the bounding sphere
    //from the displaced corners so high-relief nodes (cliffs) also gain detail.
    auto screen_error = [&](const QuadNode& n) -> double {
        NodeGeometry g = compute_node_geometry(n, p.planet);
        if (heightmap) {
            const double R  = p.planet.radius;
            const double hC = (double)heightmap->sample(g.center_dir, n.lod);
            g.center_world  = p.planet.center + g.center_dir * (R + hC);
            double max_r = 0.0;
            for (int i = 0; i < 4; ++i) {
                const DVec3  cdir = normalize(g.corners[i] - p.planet.center);
                const double h    = (double)heightmap->sample(cdir, n.lod);
                const DVec3  cw   = p.planet.center + cdir * (R + h);
                const double d    = length(cw - g.center_world);
                if (d > max_r) max_r = d;
            }
            g.bounding_radius = max_r * 1.0001;
        }
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
    //counted against the budget. A step that fits commits; one that overshoots
    //gets rolled back; the next-highest-error leaf is tried. The cut stays
    //restricted (crack-free) AND within budget. The far side keeps a low
    //error and is never reached, so the closed sphere is preserved without
    //a horizon cull.
    //
    //Cascade-overshoot slack: a face-seam chunk's split forces the cross-face
    //neighbour to split too, and that cascade can add ~15-20 leaves vs the
    //usual 3. When the soft budget is close to full, a seam chunk's cascade
    //ALWAYS overshoots and gets rolled back - so the chunks ALONG the seam
    //stay permanently stuck at coarse LOD even when the camera is directly
    //above them, producing the "vertical strip of low-detail cells at a
    //face seam" symptom (the rest of the face refines to max LOD because
    //its interior chunks have no cross-face cascade). The fix is a hard
    //budget above the soft one: commit splits up to `hard`, rollback only
    //past that. 25% slack covers the worst-case seam cascade comfortably
    //while keeping triangle-count bounded (~1.25x soft -> ~5M tris at the
    //4M soft budget the renderer uses).
    const uint32_t soft_budget = p.max_leaves;
    const uint32_t hard_budget = soft_budget + soft_budget / 4 + 32;
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

        if (leaf_count <= hard_budget) {
            //commit: enqueue every leaf this step newly created. We use the
            //hard budget here so a face-seam cascade (15-20 leaves) can land
            //even when soft is almost full - otherwise the seam stays stuck.
            for (uint64_t pid : record) {
                const QuadNode parent = unpack_node_id(pid);
                for (int q = 0; q < 4; ++q) {
                    const QuadNode c = child_node(parent, q);
                    if (is_leaf(pack_node_id(c)))
                        pq.push({ screen_error(c), c });
                }
            }
        } else {
            //Past hard: roll back this step (erase children, restore parent,
            //reverse order). The tree stays balanced - balanced_split records
            //every split it does so the rollback is exact.
            for (size_t i = record.size(); i-- > 0; ) {
                const QuadNode parent = unpack_node_id(record[i]);
                for (int q = 0; q < 4; ++q)
                    m_set.erase(pack_node_id(child_node(parent, q)));
                m_set.insert(record[i]);
            }
            leaf_count = leaf_count_before;
            //...then STOP. Leaves are popped in strict screen-error order, so
            //the node that just overshot has the HIGHEST error of everything
            //still unrefined. The old code did `continue` here to pack in later
            //smaller-cascade leaves - but every such leaf has a LOWER error
            //(it is farther from the camera), so refining it while this nearer
            //leaf stays coarse puts finer LOD on the more distant patch: the
            //"close cell stuck coarse next to a detailed farther one" inversion
            //(and the seam chunks that carry the biggest cascades were the ones
            //it stranded). Breaking keeps LOD monotonic in distance. The only
            //cost is leaving one cascade's worth of budget (~20 leaves out of
            //the ~1.25x soft cap) unused - a fraction of a percent.
            break;
        }
    }

    //safety net: the greedy keeps the cut balanced, so this is normally a
    //no-op. If it ever splits anything, the greedy left an imbalance - the
    //terrain still comes terrain crack-free, and triangle_count creeping past
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
//CORNER (DIAGONAL) LOD / LEAF QUERY
//====================================
//Compose two neighbor_node steps (across the two edges meeting at the corner)
//to reach the diagonal node at the same lod. neighbor_node handles cube-face
//seam crossings and along-edge flips, so this composition is correct everywhere
//except the 8 cube vertices (where 3 faces meet at a topological singularity)
//- there, the two possible compose orders may disagree; we pick one and accept
//that a cube-vertex corner with a coarser neighbour on the "wrong" path can
//slip through. Acceptable: cube vertices are 8 points total, far from any
//camera by construction.
namespace {
inline QuadNode corner_diag_node(const QuadNode& leaf, int corner_idx) {
    const QuadEdge e_s = (corner_idx & 1) ? EDGE_POS_S : EDGE_NEG_S;
    const QuadEdge e_t = (corner_idx & 2) ? EDGE_POS_T : EDGE_NEG_T;
    return neighbor_node(neighbor_node(leaf, e_s), e_t);
}
} // namespace

uint8_t RestrictedQuadtree::corner_lod(const QuadNode& leaf, int corner_idx) const {
    QuadNode c;
    if (covering_leaf(corner_diag_node(leaf, corner_idx), c))
        return c.lod;
    return uint8_t(leaf.lod + 1);                         // diagonal subdivided - finer
}

int RestrictedQuadtree::corner_leaf(const QuadNode& leaf, int corner_idx,
                                    uint64_t& out) const {
    QuadNode diag = corner_diag_node(leaf, corner_idx);
    QuadNode c;
    if (covering_leaf(diag, c)) { out = pack_node_id(c); return 1; }
    //Diagonal subdivided. The finer leaf touching the shared corner POINT sits
    //in the OPPOSITE quadrant of `diag` from us - corner_idx 0 (-s,-t from our
    //POV) means diag's (+s,+t) corner touches us, which is quadrant 3. In
    //general the matching quadrant is (3 ^ corner_idx).
    QuadNode walk = diag;
    while (walk.lod < MAX_LOD) {
        const int q = 3 ^ corner_idx;
        walk = child_node(walk, q);
        if (is_leaf(pack_node_id(walk))) { out = pack_node_id(walk); return 1; }
    }
    return 0;
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
