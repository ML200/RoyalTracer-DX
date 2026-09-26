#pragma once

#include "OceanCommon.h"
#include "../planet/coordinate_system.h"
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ocean {

struct TileDesc {
    uint8_t level = 0;
    uint32_t ix = 0;
    uint32_t iy = 0;
    double minX = 0.0; // absolute world low corner
    double minZ = 0.0;
    double size = 0.0;
    uint32_t stitch = 0; // OCEAN_STITCH_BITS per edge: levels to the coarser neighbour
    uint32_t slot = 0;   // persistent geometry/BLAS slot
};

inline uint64_t TileKey(uint8_t level, uint32_t ix, uint32_t iy) {
    return ((uint64_t)level << 56) | ((uint64_t)ix << 28) | (uint64_t)iy;
}

namespace detail {

// To the nearest footprint point, not the centre: no LOD flicker on tile edges.
inline double TileDistance(double minX, double minZ, double size, const planet::DVec3& cam, double seaLevel) {
    const double dx = std::max({minX - cam.x, 0.0, cam.x - (minX + size)});
    const double dz = std::max({minZ - cam.z, 0.0, cam.z - (minZ + size)});
    const double dy = cam.y - seaLevel;
    return std::sqrt(dx * dx + dz * dz + dy * dy);
}

// Four side planes, half-angle tangents scaled by margin; no near/far.
struct ViewCone {
    double F[3] = {0.0, 0.0, 1.0};
    double R[3] = {1.0, 0.0, 0.0};
    double U[3] = {0.0, 1.0, 0.0};
    double tanH = 1.0, tanV = 1.0;
    double n[4][3] = {};

    ViewCone(const planet::CameraView& cam, double margin) {
        auto normalize = [](double* v) {
            const double l = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
            if (l > 1e-12)
                for (int i = 0; i < 3; ++i) v[i] /= l;
        };
        auto cross = [](const double* a, const double* b, double* o) {
            o[0] = a[1] * b[2] - a[2] * b[1];
            o[1] = a[2] * b[0] - a[0] * b[2];
            o[2] = a[0] * b[1] - a[1] * b[0];
        };
        F[0] = cam.forward.x; F[1] = cam.forward.y; F[2] = cam.forward.z;
        U[0] = cam.up.x; U[1] = cam.up.y; U[2] = cam.up.z;
        normalize(F);
        cross(F, U, R);
        if (R[0] * R[0] + R[1] * R[1] + R[2] * R[2] < 1e-12) {
            const double X[3] = {1.0, 0.0, 0.0};
            cross(F, X, R);
        }
        normalize(R);
        cross(R, F, U);
        tanV = std::tan(0.5 * (double)cam.fov_y) * margin;
        tanH = std::tan(0.5 * (double)cam.fov_y) * (double)cam.aspect * margin;
        const double side[4][2] = {{1.0, tanH}, {-1.0, tanH}, {1.0, tanV}, {-1.0, tanV}};
        for (int p = 0; p < 4; ++p) {
            const double* axis = p < 2 ? R : U;
            for (int i = 0; i < 3; ++i) n[p][i] = side[p][0] * -axis[i] + F[i] * side[p][1];
            normalize(n[p]);
        }
    }

    bool Intersects(double x, double y, double z, double radius) const {
        for (const auto& p : n)
            if (p[0] * x + p[1] * y + p[2] * z < -radius)
                return false;
        return true;
    }
};

} // namespace detail

// Tile edge over distance; shared by tile selection and the tessellator's filter width.
struct LodRule {
    detail::ViewCone view;
    double ratio = 0.5;     // in view
    double offscreen = 4.0; // multiplier outside the view
    double nearKeep = 24.0; // in-view detail in every direction within this
    double silhouette = 0.0; // waves sub-pixel past this; ratio grows with distance, 0 = off

    LodRule(const Params& p, const planet::CameraView& cam, double silhouetteDistance)
        : view(cam, 1.2), ratio(std::max(0.02, (double)p.lodFactor)),
          offscreen(std::clamp((double)p.offscreenLodScale, 1.0, 16.0)),
          nearKeep(std::max(0.0, (double)p.nearKeepRadius)), silhouette(std::max(0.0, silhouetteDistance)) {}

    double Ratio(double dist, bool inView) const {
        double r = ratio;
        if (!inView && dist > nearKeep)
            r *= offscreen;
        if (silhouette > 0.0 && dist > silhouette)
            r *= dist / silhouette;
        return r;
    }
};

// Root pinned to world XZ, so tiles keep their slot and BLAS across frames.
class Quadtree {
  public:
    // Max level step between neighbours; edges stitched by OceanStitchSource.
    static constexpr int kMaxLevelStep = 2;
    static_assert((OCEAN_TILE_GRID >> kMaxLevelStep) << kMaxLevelStep == OCEAN_TILE_GRID,
                  "A coarser neighbour's vertices must land on the tile's own grid");
    static_assert(kMaxLevelStep <= (int)OCEAN_STITCH_MASK, "Stitch field too narrow");

    // silhouetteDistance: LodRule::silhouette.
    void Select(const Params& p, const planet::CameraView& cam, uint32_t maxTiles,
                const ICoverage* coverage = nullptr, double silhouetteDistance = 0.0);

    const std::vector<TileDesc>& Tiles() const { return m_tiles; }
    uint32_t SelectedLeafCount() const { return m_leafCount; }
    uint32_t DroppedCount() const { return m_dropped; }
    // Above Params::lodFactor when the budget forced coarser tiles.
    double EffectiveLodFactor() const { return m_lodFactor; }
    // Slot changed tile this frame: rebuild, don't refit.
    bool SlotIsNew(uint32_t slot) const { return slot < m_slotNew.size() && m_slotNew[slot] != 0; }

  private:
    uint32_t AcquireSlot(uint64_t key, uint32_t maxTiles, bool& isNew);

    std::unordered_set<uint64_t> m_set;
    std::unordered_map<uint64_t, uint32_t> m_slotOf;
    std::unordered_map<uint64_t, uint32_t> m_slotOfPrev;
    std::vector<uint32_t> m_freeSlots;
    std::vector<uint8_t> m_slotNew;
    std::vector<TileDesc> m_tiles;
    uint32_t m_leafCount = 0;
    uint32_t m_dropped = 0;
    uint32_t m_nextSlot = 0;
    double m_lodFactor = 0.0;
};

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

inline void Quadtree::Select(const Params& p, const planet::CameraView& cam, uint32_t maxTiles,
                             const ICoverage* coverage, double silhouetteDistance) {
    m_set.clear();
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
    auto decode = [](uint64_t key, uint8_t& level, uint32_t& ix, uint32_t& iy) {
        level = (uint8_t)(key >> 56);
        ix = (uint32_t)((key >> 28) & 0xFFFFFFFull);
        iy = (uint32_t)(key & 0xFFFFFFFull);
    };

    LodRule rule(p, cam, silhouetteDistance);
    auto inView = [&](double x, double z, double s) {
        // Tile bounding sphere, padded 10 m for wave height.
        return rule.view.Intersects(x + 0.5 * s - cam.position_world.x, seaLevel - cam.position_world.y,
                                    z + 0.5 * s - cam.position_world.z, s * 0.70710678 + 10.0);
    };

    // Only coverage removes tiles, never the view; catches balancing splits.
    auto isVisible = [&](double x, double z, double s) {
        return coverage == nullptr || coverage->Test(x, z, s) != Coverage::None;
    };

    struct Node {
        uint8_t level;
        uint32_t ix, iy;
    };
    std::vector<Node> stack;

    // Split edge for partially covered tiles; coarsens with the LOD.
    double coverageSplit = std::max((double)minTile * 2.0, 256.0);
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

            const Coverage cov = coverage ? coverage->Test(x, z, s) : Coverage::Full;
            if (cov == Coverage::None)
                continue;

            const bool canSplit = (n.level < maxLevel) && (s > minTile * 1.5);
            // Split shoreline tiles until their boxes hug the water.
            const bool splitForCoverage = cov == Coverage::Partial && s > coverageSplit;
            if (canSplit && (splitForCoverage || s > rule.Ratio(dist, inView(x, z, s)) * std::max(dist, 1.0))) {
                stack.push_back({(uint8_t)(n.level + 1), n.ix * 2u + 0u, n.iy * 2u + 0u});
                stack.push_back({(uint8_t)(n.level + 1), n.ix * 2u + 1u, n.iy * 2u + 0u});
                stack.push_back({(uint8_t)(n.level + 1), n.ix * 2u + 0u, n.iy * 2u + 1u});
                stack.push_back({(uint8_t)(n.level + 1), n.ix * 2u + 1u, n.iy * 2u + 1u});
                continue;
            }
            m_set.insert(TileKey(n.level, n.ix, n.iy));
        }

        // Leave headroom: balancing only adds leaves.
        uint32_t visibleCount = 0;
        for (uint64_t key : m_set) {
            uint8_t level;
            uint32_t ix, iy;
            decode(key, level, ix, iy);
            double x, z, s;
            nodeMin(level, ix, iy, x, z, s);
            if (isVisible(x, z, s))
                ++visibleCount;
        }
        if (visibleCount <= (maxTiles * 3u) / 4u || attempt >= 24)
            break;
        rule.ratio *= 1.4;
        coverageSplit *= 1.4;
    }

    // Selected leaf at or above a cell; misses where the region is finer.
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

    // Balance to kMaxLevelStep, iterated to a fixed point.
    const int kEdgeDX[4] = {-1, 1, 0, 0};
    const int kEdgeDY[4] = {0, 0, -1, 1};
    for (int pass = 0; pass < 64; ++pass) {
        std::vector<uint64_t> toSplit;
        for (uint64_t key : m_set) {
            uint8_t level;
            uint32_t ix, iy;
            decode(key, level, ix, iy);
            for (int e = 0; e < 4; ++e) {
                uint8_t nl = 0;
                if (!coveringLeaf(level, (int64_t)ix + kEdgeDX[e], (int64_t)iy + kEdgeDY[e], nl))
                    continue;
                if ((int)level > (int)nl + kMaxLevelStep)
                    toSplit.push_back(TileKey(nl, (uint32_t)(((int64_t)ix + kEdgeDX[e]) >> (level - nl)),
                                              (uint32_t)(((int64_t)iy + kEdgeDY[e]) >> (level - nl))));
            }
        }
        if (toSplit.empty())
            break;
        for (uint64_t key : toSplit) {
            if (!m_set.erase(key))
                continue;
            uint8_t level;
            uint32_t ix, iy;
            decode(key, level, ix, iy);
            for (int q = 0; q < 4; ++q)
                m_set.insert(TileKey((uint8_t)(level + 1), ix * 2u + (uint32_t)(q & 1), iy * 2u + (uint32_t)(q >> 1)));
        }
    }

    // Over budget: retry coarser rather than drop tiles; the root always fits.
    if (m_set.size() > maxTiles) {
        Params coarser = p;
        coarser.lodFactor = (float)(rule.ratio * 1.4);
        Select(coarser, cam, maxTiles, coverage, silhouetteDistance);
        return;
    }
    m_leafCount = (uint32_t)m_set.size();
    m_lodFactor = rule.ratio;

    struct Candidate {
        uint64_t key;
        double dist;
        double x, z, s;
    };
    std::vector<Candidate> visible;
    visible.reserve(m_set.size());
    for (uint64_t key : m_set) {
        uint8_t level;
        uint32_t ix, iy;
        decode(key, level, ix, iy);
        double x, z, s;
        nodeMin(level, ix, iy, x, z, s);
        if (!isVisible(x, z, s))
            continue;
        visible.push_back({key, detail::TileDistance(x, z, s, cam.position_world, seaLevel), x, z, s});
    }
    std::sort(visible.begin(), visible.end(),
              [](const Candidate& a, const Candidate& b) { return a.dist < b.dist; });

    // Free vanished tiles' slots first, or a camera jump can starve the pool.
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
        uint8_t level;
        uint32_t ix, iy;
        decode(c.key, level, ix, iy);

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
            const uint32_t step = (uint32_t)std::min<int>((int)level - (int)nl, kMaxLevelStep);
            t.stitch |= step << (OCEAN_STITCH_BITS * (uint32_t)e);
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
