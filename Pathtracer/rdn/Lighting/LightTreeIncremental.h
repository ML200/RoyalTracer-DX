#pragma once

#include "../LightTree.h"
#include "LightTreeRefit.h"
#include <cmath>
#include <functional>
#include <vector>

namespace lt {
class IncrementalTLAS {
  public:
    using Leaf = TLASExtraLeaf;
    static constexpr uint32_t NONE = 0xFFFFFFFFu;

    void rebuild(const std::vector<Leaf>& leaves, uint32_t slotCount, uint32_t bins = 64u) {
        m_nodes.clear();
        m_liveLeaves = m_tombstones = m_maxDepth = 0;
        TLASRebuilder builder;
        const TLASRefitResult r = builder.Build({}, {}, {}, bins, {}, leaves, slotCount, 0u);
        m_nodes.resize(r.nodes.size());
        for (size_t i = 0; i < r.nodes.size(); ++i) {
            const LightTLASNodeGpu& g = r.nodes[i];
            Node& n = m_nodes[i];
            n.sg = sgFromGpu(g);
            n.aabb = boxOf(n.sg);
            n.power = g.power;
            n.firstChild = g.firstChild;
            n.childCount = g.childCount;
            n.capacity = g.childCount;
            n.slot = g.childCount ? NONE : g.slot;
        }
        for (uint32_t i = 0; i < (uint32_t)m_nodes.size(); ++i)
            for (uint32_t c = 0; c < m_nodes[i].childCount; ++c)
                m_nodes[m_nodes[i].firstChild + c].parent = i;
        m_leafOfSlot.assign(slotCount, NONE);
        for (uint32_t i = 0; i < (uint32_t)m_nodes.size(); ++i) {
            const Node& n = m_nodes[i];
            if (n.leaf() && n.slot != NONE && n.slot < slotCount) {
                m_leafOfSlot[n.slot] = i;
                ++m_liveLeaves;
            }
        }
        finalize(slotCount);
    }

    void update(const std::vector<Leaf>& leaves, uint32_t slotCount) {
        // Reuse leaf slots and refit parents; rebuild when fragmentation grows.
        if (m_nodes.empty()) {
            rebuild(leaves, slotCount);
            return;
        }
        if (m_leafOfSlot.size() < slotCount)
            m_leafOfSlot.resize(slotCount, NONE);
        std::vector<uint8_t> present(m_leafOfSlot.size(), 0);
        for (const Leaf& l : leaves)
            if (l.slot < present.size() && l.power > 0.f)
                present[l.slot] = 1;
        for (uint32_t s = 0; s < (uint32_t)m_leafOfSlot.size(); ++s)
            if (m_leafOfSlot[s] != NONE && !present[s])
                remove(s);
        for (const Leaf& l : leaves) {
            if (l.slot >= m_leafOfSlot.size() || !(l.power > 0.f))
                continue;
            const uint32_t n = m_leafOfSlot[l.slot];
            if (n != NONE) {
                Node& nd = m_nodes[n];
                nd.aabb = l.aabb;
                nd.power = l.power;
                nd.sg = l.sg;
            } else {
                insert(l);
            }
        }
        refit_all();
        finalize(slotCount);
    }

    bool empty() const { return m_nodes.empty(); }

    bool degraded() const {
        return m_tombstones > m_liveLeaves / 4u + 16u || m_maxDepth >= 24u ||
               m_nodes.size() > (size_t)m_liveLeaves * 8u + 256u;
    }
    const std::vector<LightTLASNodeGpu>& nodes() const { return m_gpu; }
    const std::vector<LightTreeTrail>& trails() const { return m_trails; }
    uint32_t live_leaves() const { return m_liveLeaves; }
    uint32_t tombstones() const { return m_tombstones; }
    uint32_t max_depth() const { return m_maxDepth; }

  private:
    struct Node {
        Aabb aabb{};
        float power = 0.f;
        SgCluster sg{};
        uint32_t parent = NONE;
        uint32_t firstChild = 0, childCount = 0, capacity = 0;
        uint32_t slot = NONE;
        bool leaf() const { return childCount == 0; }
    };
    std::vector<Node> m_nodes;
    std::vector<uint32_t> m_leafOfSlot;
    std::vector<LightTLASNodeGpu> m_gpu;
    std::vector<LightTreeTrail> m_trails;
    uint32_t m_liveLeaves = 0, m_tombstones = 0, m_maxDepth = 0;

    static Aabb boxOf(const SgCluster& c) {
        const XMFLOAT3 r{c.radius, c.radius, c.radius};
        return {sub3(c.mean, r), add3(c.mean, r)};
    }
    static Node make_leaf(const Leaf& l, uint32_t parent) {
        Node n;
        n.aabb = l.aabb;
        n.power = l.power;
        n.sg = l.sg;
        n.parent = parent;
        n.slot = l.slot;
        return n;
    }
    uint32_t alloc_block() {
        const uint32_t b = (uint32_t)m_nodes.size();
        m_nodes.resize((size_t)b + 4u);
        return b;
    }
    void remove(uint32_t slot) {
        // Tombstones keep slot trails stable until the next rebuild.
        const uint32_t n = m_leafOfSlot[slot];
        Node& nd = m_nodes[n];
        const XMFLOAT3 c = aabbCenter(nd.aabb);
        nd.slot = NONE;
        nd.power = 0.f;
        nd.aabb = {c, c};
        nd.sg = SgCluster{};
        nd.sg.mean = c;
        m_leafOfSlot[slot] = NONE;
        --m_liveLeaves;
        ++m_tombstones;
    }

    void insert(const Leaf& l) {
        if (m_nodes.empty()) {
            m_nodes.push_back(make_leaf(l, NONE));
            m_leafOfSlot[l.slot] = 0;
            ++m_liveLeaves;
            return;
        }
        uint32_t node = 0;
        for (;;) {
            if (m_nodes[node].leaf()) {
                if (m_nodes[node].slot == NONE) {
                    const uint32_t parent = m_nodes[node].parent;
                    m_nodes[node] = make_leaf(l, parent);
                    m_leafOfSlot[l.slot] = node;
                    ++m_liveLeaves;
                    if (m_tombstones)
                        --m_tombstones;
                    return;
                }
                const uint32_t block = alloc_block();
                Node& p = m_nodes[node];
                Node old = p;
                old.parent = node;
                old.childCount = 0;
                old.capacity = 0;
                m_nodes[block] = old;
                m_nodes[block + 1] = make_leaf(l, node);
                p.firstChild = block;
                p.childCount = 2;
                p.capacity = 4;
                p.slot = NONE;
                m_leafOfSlot[old.slot] = block;
                m_leafOfSlot[l.slot] = block + 1;
                ++m_liveLeaves;
                return;
            }
            const Node& n = m_nodes[node];
            uint32_t best = 0;
            float bestCost = std::numeric_limits<float>::infinity();
            for (uint32_t c = 0; c < n.childCount; ++c) {
                const Node& ch = m_nodes[n.firstChild + c];
                if (ch.leaf() && ch.slot == NONE) {
                    best = c;
                    bestCost = 0.f;
                    break;
                }
                const float cost = aabbSurfaceArea(unionAabb(ch.aabb, l.aabb)) - aabbSurfaceArea(ch.aabb);
                if (cost < bestCost) {
                    bestCost = cost;
                    best = c;
                }
            }
            if (n.childCount < n.capacity && aabbSurfaceArea(l.aabb) <= bestCost) {
                const uint32_t idx = n.firstChild + n.childCount;
                m_nodes[idx] = make_leaf(l, node);
                ++m_nodes[node].childCount;
                m_leafOfSlot[l.slot] = idx;
                ++m_liveLeaves;
                return;
            }
            node = n.firstChild + best;
        }
    }

    void refit_all() {
        std::vector<SgCluster> members;
        for (uint32_t i = (uint32_t)m_nodes.size(); i-- > 0;) {
            Node& n = m_nodes[i];
            if (n.leaf())
                continue;
            bool first = true;
            Aabb box{};
            float power = 0.f;
            members.clear();
            for (uint32_t c = 0; c < n.childCount; ++c) {
                const Node& ch = m_nodes[n.firstChild + c];
                if (!(ch.power > 0.f))
                    continue;
                if (first) {
                    box = ch.aabb;
                    first = false;
                } else
                    box = unionAabb(box, ch.aabb);
                power += ch.power;
                members.push_back(ch.sg);
            }
            if (first) {
                const XMFLOAT3 c = aabbCenter(n.aabb);
                box = {c, c};
                n.sg = SgCluster{};
                n.sg.mean = c;
            } else
                n.sg = sgMerge(members);
            n.aabb = box;
            n.power = power;
        }
    }

    void finalize(uint32_t slotCount) {
        m_gpu.assign(m_nodes.size(), LightTLASNodeGpu{});
        for (LightTLASNodeGpu& g : m_gpu)
            g.slot = UINT32_MAX;
        m_trails.assign((std::max)((size_t)slotCount, m_leafOfSlot.size()), 0u);
        m_maxDepth = 0;
        if (m_nodes.empty())
            return;
        struct Item {
            uint32_t node;
            LightTreeTrail trail;
            uint32_t depth;
        };
        std::vector<Item> stack;
        stack.push_back({0u, 0u, 0u});
        while (!stack.empty()) {
            const Item it = stack.back();
            stack.pop_back();
            const Node& n = m_nodes[it.node];
            LightTLASNodeGpu& g = m_gpu[it.node];
            sgToGpu(g, n.sg);
            g.power = n.power;
            g.firstChild = n.childCount ? n.firstChild : 0xFFFFFFFFu;
            g.childCount = n.childCount;
            g.slot = (n.leaf() && n.slot != NONE) ? n.slot : UINT32_MAX;
            g._pad = 0;
            g._reserved[0] = g._reserved[1] = 0;
            m_maxDepth = (std::max)(m_maxDepth, it.depth);
            if (n.leaf()) {
                if (n.slot != NONE && n.slot < m_trails.size())
                    m_trails[n.slot] = it.trail;
                continue;
            }
            if (it.depth >= LT_TRAIL_MAX_DEPTH)
                continue;
            for (uint32_t c = 0; c < n.childCount; ++c)
                stack.push_back({n.firstChild + c, AppendLightTreeTrail(it.trail, c, it.depth), it.depth + 1u});
        }
    }
};

inline void RequestIncrementalRefit(LightTreeRefitManager& manager, std::vector<BLASRootLocal> blasRoots,
                                    std::vector<LightInstanceRef> slots, std::vector<LightSlotGpu> slotRecords,
                                    std::vector<InstanceXformCPU> xforms, std::vector<TLASExtraLeaf> extra,
                                    uint32_t slotCount, uint32_t extraVersion, IncrementalTLAS* tree,
                                    bool forceRebuild, bool compactGpuNodes = true) {
    manager.RequestCustom([roots = std::move(blasRoots), sl = std::move(slots), rec = std::move(slotRecords),
                           xf = std::move(xforms), ex = std::move(extra), slotCount, extraVersion, tree,
                           forceRebuild]() {
        std::vector<TLASExtraLeaf> leaves;
        TLASRebuilder::SceneLeaves(roots, sl, xf, rec, leaves);
        const uint32_t total = (std::max)(slotCount, (uint32_t)sl.size());
        uint32_t extraCount = 0;
        for (const TLASExtraLeaf& e : ex)
            if (e.slot < total && e.power > 0.f) {
                leaves.push_back(e);
                ++extraCount;
            }
        const bool rebuilt = forceRebuild || tree->empty() || tree->degraded();
        if (rebuilt)
            tree->rebuild(leaves, total);
        else
            tree->update(leaves, total);
        TLASRefitResult r;
        r.nodes = tree->nodes();
        r.blasBitTrails = tree->trails();
        r.slots.assign(total, LightSlotGpu{});
        for (const TLASExtraLeaf& l : leaves)
            if (l.slot < total)
                r.slots[l.slot] = l.record;
        r.extraVersion = extraVersion;
        r.extraLeafCount = extraCount;
        r.incremental = !rebuilt;
        return r;
    }, compactGpuNodes);
}
}
