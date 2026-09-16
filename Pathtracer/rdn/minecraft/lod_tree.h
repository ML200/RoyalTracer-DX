#pragma once

#include <cstdint>
#include <functional>
#include <shared_mutex>
#include <unordered_set>
#include <vector>
#include "mc_types.h"

namespace planet { class WorkerPool; }

namespace mc {

class VoxelStore;

struct LodCut {
    struct Node {
        uint64_t key = 0;
        uint32_t firstChild = 0;
        uint8_t  childCount = 0;
        uint8_t  level = 0;
        void*    chunk = nullptr;
        bool leaf() const { return childCount == 0; }
    };
    std::vector<Node>     nodes;
    uint32_t              rootCount = 0;
    std::vector<uint64_t> leafList;
    std::unordered_set<uint64_t, U64Hash> leaves;
    std::unordered_set<uint64_t, U64Hash> interior;
    void clear() { nodes.clear(); rootCount = 0; leafList.clear(); leaves.clear(); interior.clear(); }
};

struct RenderItem { uint64_t key; void* chunk; };

using ReadyFn = std::function<bool(uint64_t key, void*& chunk)>;
using ResidentBelowFn = std::function<bool(uint64_t key)>;

class LodTree {
public:
    // Builds occupancy metadata and roots from the voxel store.
    void configure(const VoxelStore& store);

    int root_level() const { return m_rootLevel; }
    const VoxelStore* store() const { return m_store; }

    static double node_distance(const NodeKey& k, const double cam[3]);

    // Selects a distance-driven cut, optionally splitting work across workers.
    void select(const double cam[3], float lodFactor, LodCut& out, planet::WorkerPool* pool = nullptr) const;

    // Resolves resident geometry with coarser stand-ins for missing chunks.
    void render_list(LodCut& cut, const ReadyFn& ready, std::vector<RenderItem>& out,
                     const ResidentBelowFn& residentBelow = {}, planet::WorkerPool* pool = nullptr) const;

    const std::vector<uint64_t>& roots() const { return m_roots; }

    bool chunk_occupied(const NodeKey& k) const { return m_occupied.count(pack_node(k)) != 0; }
    void add_occupied(uint64_t packedKey) { std::unique_lock<std::shared_mutex> lk(m_occupiedLock); m_occupied.insert(packedKey); }

private:
    bool refines(const NodeKey& k, const double cam[3], float f) const;
    void expand_node(uint32_t nodeIndex, LodCut& out) const;
    void select_rec(uint32_t nodeIndex, const double cam[3], float f, LodCut& out) const;
    void select_serial(const double cam[3], float f, LodCut& out) const;
    void select_parallel(const double cam[3], float f, LodCut& out, planet::WorkerPool& pool) const;
    bool render_rec(LodCut& cut, uint32_t nodeIndex, const ReadyFn& ready, const ResidentBelowFn& residentBelow,
                    std::vector<RenderItem>& out) const;
    void fallback_rec(const NodeKey& k, const ReadyFn& ready, const ResidentBelowFn& residentBelow,
                      std::vector<RenderItem>& out) const;
    void render_parallel(LodCut& cut, const ReadyFn& ready, const ResidentBelowFn& residentBelow,
                         std::vector<RenderItem>& out, planet::WorkerPool& pool) const;

    const VoxelStore*     m_store = nullptr;
    int                   m_rootLevel = 0;
    std::vector<uint64_t> m_roots;
    std::unordered_set<uint64_t, U64Hash> m_occupied;
    mutable std::shared_mutex m_occupiedLock;
};

}
