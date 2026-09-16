#include "restricted_quadtree.h"
#include "heightmap_source.h"
#include <algorithm>
#include <queue>

namespace planet {

// Replaces one leaf with its four contiguous children.
void RestrictedQuadtree::do_split(const QuadNode& n, std::vector<uint64_t>& record,
                                  uint32_t& leaf_count) {
    m_set.erase(pack_node_id(n));
    for (int q = 0; q < 4; ++q)
        m_set.insert(pack_node_id(child_node(n, q)));
    record.push_back(pack_node_id(n));
    leaf_count += 3u;
}

void RestrictedQuadtree::ensure_lod(const QuadNode& region, uint8_t target,
                                    std::vector<uint64_t>& record, uint32_t& leaf_count) {
    while (true) {
        QuadNode leaf;
        if (!covering_leaf(region, leaf)) return;
        if (leaf.lod >= target)           return;
        for (int e = 0; e < 4; ++e)
            ensure_lod(neighbor_node(leaf, (QuadEdge)e), leaf.lod, record, leaf_count);
        do_split(leaf, record, leaf_count);
    }
}

// Refines neighboring leaves before splitting to preserve balance.
void RestrictedQuadtree::balanced_split(const QuadNode& N, std::vector<uint64_t>& record,
                                        uint32_t& leaf_count) {
    for (int e = 0; e < 4; ++e)
        ensure_lod(neighbor_node(N, (QuadEdge)e), N.lod, record, leaf_count);
    do_split(N, record, leaf_count);
}

// Greedily refines visible leaves, then balances adjacent detail levels.
void RestrictedQuadtree::select(const QuadtreeParams& params, const CameraView& cam,
                                const IHeightmapSource* heightmap) {
    QuadtreeParams p = params;
    if (p.max_lod > MAX_LOD)   p.max_lod = MAX_LOD;
    if (p.min_lod > p.max_lod) p.min_lod = p.max_lod;

    m_set.clear();

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

    const uint32_t soft_budget = p.max_leaves;
    const uint32_t hard_budget = soft_budget + soft_budget / 4 + 32;
    struct Item { double err; QuadNode node; };
    struct ErrLess { bool operator()(const Item& a, const Item& b) const { return a.err < b.err; } };
    std::priority_queue<Item, std::vector<Item>, ErrLess> pq;
    for (uint64_t id : m_set) {
        const QuadNode n = unpack_node_id(id);
        pq.push({ screen_error(n), n });
    }

    std::vector<uint64_t> record;
    while (!pq.empty()) {
        const QuadNode N = pq.top().node;
        pq.pop();
        if (!is_leaf(pack_node_id(N))) continue;
        if (N.lod >= p.max_lod)        continue;

        const uint32_t leaf_count_before = leaf_count;
        record.clear();
        balanced_split(N, record, leaf_count);

        if (leaf_count <= hard_budget) {
            for (uint64_t pid : record) {
                const QuadNode parent = unpack_node_id(pid);
                for (int q = 0; q < 4; ++q) {
                    const QuadNode c = child_node(parent, q);
                    if (is_leaf(pack_node_id(c)))
                        pq.push({ screen_error(c), c });
                }
            }
        } else {
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

    balance();

    m_leaves.assign(m_set.begin(), m_set.end());
    std::sort(m_leaves.begin(), m_leaves.end());
}

bool RestrictedQuadtree::covering_leaf(const QuadNode& m, QuadNode& out) const {
    QuadNode c = m;
    while (true) {
        if (is_leaf(pack_node_id(c))) { out = c; return true; }
        if (c.lod == 0) return false;
        c = parent_node(c);
    }
}

// Refines coarse neighbors until adjacent levels differ by at most one.
void RestrictedQuadtree::balance() {
    std::vector<uint64_t> work(m_set.begin(), m_set.end());

    while (!work.empty()) {
        const uint64_t id = work.back();
        work.pop_back();
        if (!is_leaf(id)) continue;

        const QuadNode L = unpack_node_id(id);
        for (int e = 0; e < 4; ++e) {
            while (true) {
                QuadNode c;
                if (!covering_leaf(neighbor_node(L, (QuadEdge)e), c)) break;
                if (c.lod + 1 >= L.lod) break;

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

uint8_t RestrictedQuadtree::neighbor_lod(const QuadNode& leaf, QuadEdge edge) const {
    QuadNode c;
    if (covering_leaf(neighbor_node(leaf, edge), c))
        return c.lod;
    return uint8_t(leaf.lod + 1);
}

namespace {
inline QuadNode corner_diag_node(const QuadNode& leaf, int corner_idx) {
    const QuadEdge e_s = (corner_idx & 1) ? EDGE_POS_S : EDGE_NEG_S;
    const QuadEdge e_t = (corner_idx & 2) ? EDGE_POS_T : EDGE_NEG_T;
    return neighbor_node(neighbor_node(leaf, e_s), e_t);
}
}

uint8_t RestrictedQuadtree::corner_lod(const QuadNode& leaf, int corner_idx) const {
    QuadNode c;
    if (covering_leaf(corner_diag_node(leaf, corner_idx), c))
        return c.lod;
    return uint8_t(leaf.lod + 1);
}

int RestrictedQuadtree::corner_leaf(const QuadNode& leaf, int corner_idx,
                                    uint64_t& out) const {
    QuadNode diag = corner_diag_node(leaf, corner_idx);
    QuadNode c;
    if (covering_leaf(diag, c)) { out = pack_node_id(c); return 1; }
    QuadNode walk = diag;
    while (walk.lod < MAX_LOD) {
        const int q = 3 ^ corner_idx;
        walk = child_node(walk, q);
        if (is_leaf(pack_node_id(walk))) { out = pack_node_id(walk); return 1; }
    }
    return 0;
}

namespace {
constexpr int EDGE_CHILDREN[4][2] = {
    { 0, 2 },
    { 1, 3 },
    { 0, 1 },
    { 2, 3 },
};
}

int RestrictedQuadtree::adjacent_leaves(const QuadNode& leaf, QuadEdge edge,
                                        uint64_t out[2]) const {
    QuadNode c;
    if (covering_leaf(neighbor_node(leaf, edge), c)) {
        out[0] = pack_node_id(c);
        return 1;
    }
    const QuadNode a0 = neighbor_node(child_node(leaf, EDGE_CHILDREN[edge][0]), edge);
    const QuadNode a1 = neighbor_node(child_node(leaf, EDGE_CHILDREN[edge][1]), edge);
    out[0] = pack_node_id(a0);
    out[1] = pack_node_id(a1);
    return 2;
}

}
