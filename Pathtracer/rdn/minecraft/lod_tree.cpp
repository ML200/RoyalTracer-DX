#include "lod_tree.h"
#include "voxel_store.h"
#include "../planet/worker_pool.h"
#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace mc {

void LodTree::configure(const VoxelStore& store) {
    m_store = &store;
    m_roots.clear();
    {
        const std::vector<uint64_t> keys = store.occupied_chunk_keys();
        std::unique_lock<std::shared_mutex> lk(m_occupiedLock);
        m_occupied.clear();
        m_occupied.reserve(keys.size() * 2);
        m_occupied.insert(keys.begin(), keys.end());
    }
    int minSx, maxSx, minSz, maxSz;
    if (!store.column_bounds(0, minSx, maxSx, minSz, maxSz)) { m_rootLevel = 0; return; }
    const int minCx = floor_shift(minSx, 1), maxCx = floor_shift(maxSx, 1);
    const int minCz = floor_shift(minSz, 1), maxCz = floor_shift(maxSz, 1);
    const int minCy = floor_shift(store.min_section_y(0), 1), maxCy = floor_shift(store.max_section_y(0), 1);
    int extent = std::max({ maxCx - minCx + 1, maxCz - minCz + 1, maxCy - minCy + 1 });
    int level = 0;
    while ((1 << level) < extent && level < MAX_LOD_LEVELS - 1) ++level;
    level = std::min(level, store.levels() - 1);
    m_rootLevel = std::max(level, 0);
    for (int z = floor_shift(minCz, m_rootLevel); z <= floor_shift(maxCz, m_rootLevel); ++z)
    for (int y = floor_shift(minCy, m_rootLevel); y <= floor_shift(maxCy, m_rootLevel); ++y)
    for (int x = floor_shift(minCx, m_rootLevel); x <= floor_shift(maxCx, m_rootLevel); ++x)
        m_roots.push_back(pack_node(NodeKey{ (uint8_t)m_rootLevel, x, y, z }));
}

double LodTree::node_distance(const NodeKey& k, const double cam[3]) {
    int64_t o[3];
    node_origin_blocks(k, o);
    const double s = (double)node_size_blocks(k.level);
    double d2 = 0.0;
    for (int a = 0; a < 3; ++a) {
        const double lo = (double)o[a], hi = lo + s;
        const double d = cam[a] < lo ? lo - cam[a] : (cam[a] > hi ? cam[a] - hi : 0.0);
        d2 += d * d;
    }
    return std::sqrt(d2);
}

// Projected voxel size above threshold.
bool LodTree::refines(const NodeKey& k, const double cam[3], float f) const {
    if (k.level == 0) return false;
    return node_distance(k, cam) < (double)f * (double)(1 << k.level);
}

void LodTree::expand_node(uint32_t nodeIndex, LodCut& out) const {
    const NodeKey k = unpack_node(out.nodes[nodeIndex].key);
    const uint32_t first = (uint32_t)out.nodes.size();
    uint32_t count = 0;
    for (int i = 0; i < 8; ++i) {
        const NodeKey c = child_key(k, i);
        if (!chunk_occupied(c)) continue;
        LodCut::Node n;
        n.key = pack_node(c); n.level = (uint8_t)c.level;
        out.nodes.push_back(n);
        ++count;
    }
    out.nodes[nodeIndex].firstChild = first;
    out.nodes[nodeIndex].childCount = (uint8_t)count;
}

void LodTree::select_rec(uint32_t nodeIndex, const double cam[3], float f, LodCut& out) const {
    const uint64_t packed = out.nodes[nodeIndex].key;
    if (!refines(unpack_node(packed), cam, f)) {
        out.leaves.insert(packed);
        out.leafList.push_back(packed);
        return;
    }
    out.interior.insert(packed);
    expand_node(nodeIndex, out);
    const uint32_t first = out.nodes[nodeIndex].firstChild, count = out.nodes[nodeIndex].childCount;
    for (uint32_t j = 0; j < count; ++j) select_rec(first + j, cam, f, out);
}

void LodTree::select(const double cam[3], float lodFactor, LodCut& out, planet::WorkerPool* pool) const {
    out.clear();
    if (!m_store) return;
    std::shared_lock<std::shared_mutex> lk(m_occupiedLock);
    for (uint64_t r : m_roots) {
        const NodeKey k = unpack_node(r);
        if (!chunk_occupied(k)) continue;
        LodCut::Node n;
        n.key = r; n.level = (uint8_t)k.level;
        out.nodes.push_back(n);
        ++out.rootCount;
    }
    if (pool && pool->thread_count() >= 2) select_parallel(cam, lodFactor, out, *pool);
    else select_serial(cam, lodFactor, out);
}

void LodTree::select_serial(const double cam[3], float f, LodCut& out) const {
    for (uint32_t i = 0; i < out.rootCount; ++i) select_rec(i, cam, f, out);
}

void LodTree::select_parallel(const double cam[3], float f, LodCut& out, planet::WorkerPool& pool) const {
    const size_t target = (size_t)pool.thread_count() * 4;
    std::vector<uint32_t> frontier, next;
    for (uint32_t i = 0; i < out.rootCount; ++i) frontier.push_back(i);
    while (frontier.size() < target) {
        next.clear();
        bool expanded = false;
        for (uint32_t idx : frontier) {
            const uint64_t packed = out.nodes[idx].key;
            if (!refines(unpack_node(packed), cam, f)) {
                out.leaves.insert(packed);
                out.leafList.push_back(packed);
                continue;
            }
            out.interior.insert(packed);
            expand_node(idx, out);
            const uint32_t first = out.nodes[idx].firstChild, count = out.nodes[idx].childCount;
            for (uint32_t j = 0; j < count; ++j) next.push_back(first + j);
            expanded = true;
        }
        frontier.swap(next);
        if (!expanded) break;
    }
    if (frontier.empty()) return;
    std::vector<LodCut> pieces(frontier.size());
    pool.parallel_for((uint32_t)frontier.size(), [&](uint32_t p) {
        LodCut& pc = pieces[p];
        LodCut::Node root = out.nodes[frontier[p]];
        root.firstChild = 0; root.childCount = 0;
        pc.nodes.push_back(root);
        pc.rootCount = 1;
        select_rec(0, cam, f, pc);
    });
    for (size_t p = 0; p < pieces.size(); ++p) {
        const LodCut& pc = pieces[p];
        const LodCut::Node& root = pc.nodes[0];
        const uint32_t base = (uint32_t)out.nodes.size();
        LodCut::Node& top = out.nodes[frontier[p]];
        top.childCount = root.childCount;
        top.firstChild = root.childCount ? base + root.firstChild - 1 : 0;
        for (size_t i = 1; i < pc.nodes.size(); ++i) {
            LodCut::Node n = pc.nodes[i];
            if (n.childCount) n.firstChild = base + n.firstChild - 1;
            out.nodes.push_back(n);
        }
        out.leafList.insert(out.leafList.end(), pc.leafList.begin(), pc.leafList.end());
        out.leaves.insert(pc.leaves.begin(), pc.leaves.end());
        out.interior.insert(pc.interior.begin(), pc.interior.end());
    }
}

void LodTree::fallback_rec(const NodeKey& k, const ReadyFn& ready, const ResidentBelowFn& residentBelow,
                           std::vector<RenderItem>& out) const {
    if (k.level == 0) return;
    if (residentBelow && !residentBelow(pack_node(k))) return;
    for (int i = 0; i < 8; ++i) {
        const NodeKey c = child_key(k, i);
        if (!chunk_occupied(c)) continue;
        const uint64_t packed = pack_node(c);
        void* chunk = nullptr;
        if (ready(packed, chunk)) out.push_back(RenderItem{ packed, chunk });
        else fallback_rec(c, ready, residentBelow, out);
    }
}

bool LodTree::render_rec(LodCut& cut, uint32_t nodeIndex, const ReadyFn& ready, const ResidentBelowFn& residentBelow,
                         std::vector<RenderItem>& out) const {
    LodCut::Node& n = cut.nodes[nodeIndex];
    if (n.leaf()) {
        if (ready(n.key, n.chunk)) { out.push_back(RenderItem{ n.key, n.chunk }); return true; }
        fallback_rec(unpack_node(n.key), ready, residentBelow, out);
        return false;
    }
    const size_t mark = out.size();
    bool complete = true;
    const uint32_t first = n.firstChild, count = n.childCount;
    for (uint32_t j = 0; j < count; ++j)
        if (!render_rec(cut, first + j, ready, residentBelow, out)) complete = false;
    if (complete) return true;
    LodCut::Node& again = cut.nodes[nodeIndex];
    if (ready(again.key, again.chunk)) { out.resize(mark); out.push_back(RenderItem{ again.key, again.chunk }); return true; }
    return false;
}

void LodTree::render_list(LodCut& cut, const ReadyFn& ready, std::vector<RenderItem>& out,
                          const ResidentBelowFn& residentBelow, planet::WorkerPool* pool) const {
    out.clear();
    if (!m_store) return;
    std::shared_lock<std::shared_mutex> lk(m_occupiedLock);
    if (pool && pool->thread_count() >= 2 && cut.nodes.size() >= 1024) {
        render_parallel(cut, ready, residentBelow, out, *pool);
        return;
    }
    for (uint32_t i = 0; i < cut.rootCount; ++i) render_rec(cut, i, ready, residentBelow, out);
}

void LodTree::render_parallel(LodCut& cut, const ReadyFn& ready, const ResidentBelowFn& residentBelow,
                              std::vector<RenderItem>& out, planet::WorkerPool& pool) const {
    const size_t target = (size_t)pool.thread_count() * 4;
    std::vector<uint32_t> pieces, queue, next;
    for (uint32_t i = 0; i < cut.rootCount; ++i) queue.push_back(i);
    while (!queue.empty() && pieces.size() + queue.size() < target) {
        next.clear();
        bool expanded = false;
        for (uint32_t idx : queue) {
            const LodCut::Node& n = cut.nodes[idx];
            if (n.leaf()) { pieces.push_back(idx); continue; }
            for (uint32_t j = 0; j < n.childCount; ++j) next.push_back(n.firstChild + j);
            expanded = true;
        }
        queue.swap(next);
        if (!expanded) break;
    }
    pieces.insert(pieces.end(), queue.begin(), queue.end());
    if (pieces.empty()) return;
    struct Piece { std::vector<RenderItem> items; bool complete = false; };
    std::vector<Piece> results(pieces.size());
    pool.parallel_for((uint32_t)pieces.size(), [&](uint32_t p) {
        results[p].complete = render_rec(cut, pieces[p], ready, residentBelow, results[p].items);
    });
    std::unordered_map<uint32_t, uint32_t> pieceOf;
    pieceOf.reserve(pieces.size() * 2);
    for (uint32_t p = 0; p < (uint32_t)pieces.size(); ++p) pieceOf[pieces[p]] = p;
    std::function<bool(uint32_t)> top = [&](uint32_t idx) -> bool {
        const auto it = pieceOf.find(idx);
        if (it != pieceOf.end()) {
            const Piece& r = results[it->second];
            out.insert(out.end(), r.items.begin(), r.items.end());
            return r.complete;
        }
        const size_t mark = out.size();
        bool complete = true;
        const uint32_t first = cut.nodes[idx].firstChild, count = cut.nodes[idx].childCount;
        for (uint32_t j = 0; j < count; ++j)
            if (!top(first + j)) complete = false;
        if (complete) return true;
        LodCut::Node& n = cut.nodes[idx];
        if (ready(n.key, n.chunk)) { out.resize(mark); out.push_back(RenderItem{ n.key, n.chunk }); return true; }
        return false;
    };
    for (uint32_t i = 0; i < cut.rootCount; ++i) top(i);
}

}
