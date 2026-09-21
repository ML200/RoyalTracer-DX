#pragma once

#include "OceanCommon.h"
#include "../planet/coordinate_system.h"
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ocean {

// Camera-driven quadtree over the world XZ plane. The root is pinned to absolute world space
// rather than to the camera, so a tile covers the same square of ocean from frame to frame: its
// vertices keep their buffer slot, and its acceleration structure can be refitted as the waves
// move instead of rebuilt.
//
// The tree is kept 2:1 balanced so that any two adjacent tiles differ by at most one level, which
// is what makes a single per-edge collapse enough to close the cracks between them.

struct TileDesc {
    uint8_t level = 0;
    uint32_t ix = 0;
    uint32_t iy = 0;
    double minX = 0.0; // absolute world coordinates of the tile's low corner
    double minZ = 0.0;
    double size = 0.0;
    uint32_t stitch = 0;
    uint32_t slot = 0; // persistent geometry and acceleration-structure slot
};

inline uint64_t TileKey(uint8_t level, uint32_t ix, uint32_t iy) {
    return ((uint64_t)level << 56) | ((uint64_t)ix << 28) | (uint64_t)iy;
}

class Quadtree {
  public:
    // Chooses the visible leaves for this camera and assigns each a persistent slot.
    void Select(const Params& p, const planet::CameraView& cam, uint32_t maxTiles);

    const std::vector<TileDesc>& Tiles() const { return m_tiles; }
    uint32_t SelectedLeafCount() const { return m_leafCount; }
    uint32_t DroppedCount() const { return m_dropped; }
    // Slots whose tile left the view this frame; their acceleration structures must be rebuilt
    // rather than refitted when the slot is handed to a different tile.
    bool SlotIsNew(uint32_t slot) const { return slot < m_slotNew.size() && m_slotNew[slot] != 0; }

    void Reset() {
        m_slotOf.clear();
        m_freeSlots.clear();
        m_slotNew.clear();
        m_tiles.clear();
    }

  private:
    uint32_t AcquireSlot(uint64_t key, uint32_t maxTiles, bool& isNew);

    std::unordered_set<uint64_t> m_set;
    std::vector<uint64_t> m_leaves;
    std::unordered_map<uint64_t, uint32_t> m_slotOf;
    std::unordered_map<uint64_t, uint32_t> m_slotOfPrev;
    std::vector<uint32_t> m_freeSlots;
    std::vector<uint8_t> m_slotNew;
    std::vector<TileDesc> m_tiles;
    uint32_t m_leafCount = 0;
    uint32_t m_dropped = 0;
    uint32_t m_nextSlot = 0;
};

namespace detail {

// Distance from the camera to the nearest point of a tile's footprint. Using the footprint rather
// than its centre keeps the level from oscillating when the camera sits over a tile edge.
inline double TileDistance(double minX, double minZ, double size, const planet::DVec3& cam, double seaLevel) {
    const double dx = std::max({minX - cam.x, 0.0, cam.x - (minX + size)});
    const double dz = std::max({minZ - cam.z, 0.0, cam.z - (minZ + size)});
    const double dy = cam.y - seaLevel;
    return std::sqrt(dx * dx + dz * dz + dy * dy);
}

} // namespace detail

inline uint32_t Quadtree::AcquireSlot(uint64_t key, uint32_t maxTiles, bool& isNew) {
    const auto prev = m_slotOfPrev.find(key);
    if (prev != m_slotOfPrev.end()) {
        isNew = false;
        return prev->second;
    }
    isNew = true;
    if (!m_freeSlots.empty()) {
        const uint32_t s = m_freeSlots.back();
        m_freeSlots.pop_back();
        return s;
    }
    if (m_nextSlot < maxTiles)
        return m_nextSlot++;
    return UINT32_MAX;
}

inline void Quadtree::Select(const Params& p, const planet::CameraView& cam, uint32_t maxTiles) {
    m_set.clear();
    m_leaves.clear();
    m_tiles.clear();
    m_dropped = 0;
    if (maxTiles == 0) {
        m_leafCount = 0;
        return;
    }

    const double halfExtent = std::max(64.0, (double)p.extent);
    const double rootSize = halfExtent * 2.0;
    const double minTile = std::max(1.0, (double)p.minTileSize);
    int maxLevel = 0;
    while (rootSize / std::pow(2.0, (double)maxLevel) > minTile && maxLevel < 24)
        ++maxLevel;

    const double seaLevel = (double)p.seaLevelY;

    auto nodeMin = [&](uint8_t level, uint32_t ix, uint32_t iy, double& x, double& z, double& s) {
        s = rootSize / std::pow(2.0, (double)level);
        x = -halfExtent + (double)ix * s;
        z = -halfExtent + (double)iy * s;
    };

    // Keep the complete finite ocean in the ray-visible scene, including behind the camera.
    // Quality selection may coarsen geometry; visibility is never restricted to primary rays.
    auto isVisible = [](double, double, double, double) { return true; };

    // Apply the same LOD rule around the full finite square, independent of view direction.
    struct Node {
        uint8_t level;
        uint32_t ix, iy;
    };
    std::vector<Node> stack;

    // A camera high above the water sees far more ocean than one at deck level. Rather than drop
    // tiles when the budget runs out - which would leave holes along the horizon, where the sky
    // would show straight through - back the detail off and select again. Coarser distant tiles
    // cost nothing visually, because those waves are already carried by the BRDF rather than by
    // geometry.
    double lodFactor = std::max(0.02, (double)p.lodFactor);
    uint32_t visibleCount = 0;
    for (int attempt = 0;; ++attempt) {
        m_set.clear();
        stack.clear();
        stack.push_back({0, 0, 0});
        while (!stack.empty()) {
            const Node n = stack.back();
            stack.pop_back();

            double x, z, s;
            nodeMin(n.level, n.ix, n.iy, x, z, s);
            const double dist = detail::TileDistance(x, z, s, cam.position_world, seaLevel);

            const bool canSplit = (n.level < maxLevel) && (s > minTile * 1.5);
            if (canSplit && s > lodFactor * std::max(dist, 1.0)) {
                stack.push_back({(uint8_t)(n.level + 1), n.ix * 2u + 0u, n.iy * 2u + 0u});
                stack.push_back({(uint8_t)(n.level + 1), n.ix * 2u + 1u, n.iy * 2u + 0u});
                stack.push_back({(uint8_t)(n.level + 1), n.ix * 2u + 0u, n.iy * 2u + 1u});
                stack.push_back({(uint8_t)(n.level + 1), n.ix * 2u + 1u, n.iy * 2u + 1u});
                continue;
            }
            m_set.insert(TileKey(n.level, n.ix, n.iy));
        }

        // Count what would actually be emitted. Balancing only ever adds leaves, so leave headroom
        // for the tiles it will split.
        visibleCount = 0;
        for (uint64_t key : m_set) {
            double x, z, s;
            nodeMin((uint8_t)(key >> 56), (uint32_t)((key >> 28) & 0xFFFFFFFull), (uint32_t)(key & 0xFFFFFFFull), x, z,
                    s);
            if (isVisible(x, z, s, detail::TileDistance(x, z, s, cam.position_world, seaLevel)))
                ++visibleCount;
        }
        if (visibleCount <= (maxTiles * 3u) / 4u || attempt >= 24)
            break;
        lodFactor *= 1.4;
    }

    // Returns the selected leaf covering a cell, which is that cell or one of its ancestors.
    // A miss means the region is subdivided further than the query level.
    auto coveringLeaf = [&](uint8_t level, int64_t nx, int64_t ny, uint8_t& outLevel) -> bool {
        if (nx < 0 || ny < 0)
            return false;
        uint64_t span = (uint64_t)1 << level;
        if ((uint64_t)nx >= span || (uint64_t)ny >= span)
            return false;
        for (int l = level; l >= 0; --l) {
            const uint32_t cx = (uint32_t)(nx >> (level - l));
            const uint32_t cy = (uint32_t)(ny >> (level - l));
            if (m_set.count(TileKey((uint8_t)l, cx, cy))) {
                outLevel = (uint8_t)l;
                return true;
            }
        }
        return false;
    };

    // 2:1 balance. A leaf whose neighbour is two or more levels coarser forces that neighbour to
    // split; splitting can expose new imbalances, so this runs to a fixed point.
    const int kEdgeDX[4] = {-1, 1, 0, 0};
    const int kEdgeDY[4] = {0, 0, -1, 1};
    for (int pass = 0; pass < 32; ++pass) {
        std::vector<uint64_t> toSplit;
        for (uint64_t key : m_set) {
            const uint8_t level = (uint8_t)(key >> 56);
            const uint32_t ix = (uint32_t)((key >> 28) & 0xFFFFFFFull);
            const uint32_t iy = (uint32_t)(key & 0xFFFFFFFull);
            for (int e = 0; e < 4; ++e) {
                uint8_t nl = 0;
                if (!coveringLeaf(level, (int64_t)ix + kEdgeDX[e], (int64_t)iy + kEdgeDY[e], nl))
                    continue;
                if (level > nl + 1)
                    toSplit.push_back(TileKey(nl, (uint32_t)(((int64_t)ix + kEdgeDX[e]) >> (level - nl)),
                                              (uint32_t)(((int64_t)iy + kEdgeDY[e]) >> (level - nl))));
            }
        }
        if (toSplit.empty())
            break;
        for (uint64_t key : toSplit) {
            if (!m_set.erase(key))
                continue;
            const uint8_t level = (uint8_t)(key >> 56);
            const uint32_t ix = (uint32_t)((key >> 28) & 0xFFFFFFFull);
            const uint32_t iy = (uint32_t)(key & 0xFFFFFFFull);
            for (int q = 0; q < 4; ++q)
                m_set.insert(TileKey((uint8_t)(level + 1), ix * 2u + (uint32_t)(q & 1), iy * 2u + (uint32_t)(q >> 1)));
        }
    }

    // Balancing may consume more than the reserved headroom. Retry the complete selection
    // at a coarser LOD before assigning slots instead of dropping tiles and opening holes.
    // Increasing the LOD factor eventually selects just the root, which fits any positive budget.
    if (m_set.size() > maxTiles) {
        Params coarser = p;
        coarser.lodFactor = (float)(lodFactor * 1.4);
        Select(coarser, cam, maxTiles);
        return;
    }
    m_leafCount = (uint32_t)m_set.size();

    // Assign nearby tiles first. The complete balanced selection now fits the tile budget.
    struct Candidate {
        uint64_t key;
        double dist;
        double x, z, s;
    };
    std::vector<Candidate> visible;
    visible.reserve(m_set.size());
    for (uint64_t key : m_set) {
        const uint8_t level = (uint8_t)(key >> 56);
        const uint32_t ix = (uint32_t)((key >> 28) & 0xFFFFFFFull);
        const uint32_t iy = (uint32_t)(key & 0xFFFFFFFull);
        double x, z, s;
        nodeMin(level, ix, iy, x, z, s);
        const double dist = detail::TileDistance(x, z, s, cam.position_world, seaLevel);
        if (!isVisible(x, z, s, dist))
            continue;
        visible.push_back({key, dist, x, z, s});
    }
    std::sort(visible.begin(), visible.end(),
              [](const Candidate& a, const Candidate& b) { return a.dist < b.dist; });

    // Retire disappeared tiles BEFORE allocating their replacements. Waiting until the end
    // can exhaust the pool on a camera jump even when the new selection fits the budget.
    std::unordered_set<uint64_t> retained;
    for (const auto& c : visible) retained.insert(c.key);
    for (const auto& old : m_slotOf)
        if (!retained.count(old.first)) m_freeSlots.push_back(old.second);
    m_slotOfPrev = std::move(m_slotOf);
    m_slotOf.clear();
    m_slotNew.assign(maxTiles, 0);

    m_tiles.reserve(std::min<size_t>(visible.size(), maxTiles));
    for (const Candidate& c : visible) {
        if (m_tiles.size() >= maxTiles) {
            ++m_dropped;
            continue;
        }
        const uint8_t level = (uint8_t)(c.key >> 56);
        const uint32_t ix = (uint32_t)((c.key >> 28) & 0xFFFFFFFull);
        const uint32_t iy = (uint32_t)(c.key & 0xFFFFFFFull);

        TileDesc t;
        t.level = level;
        t.ix = ix;
        t.iy = iy;
        t.minX = c.x;
        t.minZ = c.z;
        t.size = c.s;
        t.stitch = 0;
        for (int e = 0; e < 4; ++e) {
            uint8_t nl = 0;
            if (!coveringLeaf(level, (int64_t)ix + kEdgeDX[e], (int64_t)iy + kEdgeDY[e], nl))
                continue;
            if (nl + 1 == level) {
                static const uint32_t kBit[4] = {OCEAN_EDGE_NEG_X, OCEAN_EDGE_POS_X, OCEAN_EDGE_NEG_Z,
                                                 OCEAN_EDGE_POS_Z};
                t.stitch |= kBit[e];
            }
        }

        bool isNew = false;
        const uint32_t slot = AcquireSlot(c.key, maxTiles, isNew);
        if (slot == UINT32_MAX) {
            ++m_dropped;
            continue;
        }
        t.slot = slot;
        m_slotOf[c.key] = slot;
        if (isNew)
            m_slotNew[slot] = 1;
        m_tiles.push_back(t);
    }

}

} // namespace ocean


