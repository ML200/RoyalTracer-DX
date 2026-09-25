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
// Adjacent tiles may differ by up to kMaxLevelStep levels. The finer side interpolates its edge
// onto the coarser one's (see OceanStitchSource), so no crack opens, and the looser the balance
// the more the detail can fall away where nothing looks closely - a 2:1 tree has to step down one
// level per tile width from everything the camera sees, which spreads the in-view detail around
// most of the circle.

struct TileDesc {
    uint8_t level = 0;
    uint32_t ix = 0;
    uint32_t iy = 0;
    double minX = 0.0; // absolute world coordinates of the tile's low corner
    double minZ = 0.0;
    double size = 0.0;
    uint32_t stitch = 0; // OCEAN_STITCH_BITS per edge: levels to the coarser neighbour
    uint32_t slot = 0;   // persistent geometry and acceleration-structure slot
};

inline uint64_t TileKey(uint8_t level, uint32_t ix, uint32_t iy) {
    return ((uint64_t)level << 56) | ((uint64_t)ix << 28) | (uint64_t)iy;
}

namespace detail {

// Distance from the camera to the nearest point of a tile's footprint. Using the footprint rather
// than its centre keeps the level from oscillating when the camera sits over a tile edge.
inline double TileDistance(double minX, double minZ, double size, const planet::DVec3& cam, double seaLevel) {
    const double dx = std::max({minX - cam.x, 0.0, cam.x - (minX + size)});
    const double dz = std::max({minZ - cam.z, 0.0, cam.z - (minZ + size)});
    const double dy = cam.y - seaLevel;
    return std::sqrt(dx * dx + dz * dz + dy * dy);
}

// The camera's four side planes, widened by `margin` on the tangent of each half-angle, with no
// near or far limit. Tested against spheres in camera-relative double coordinates.
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

// How much geometry a stretch of sea gets, as tile edge over distance. Shared between tile
// selection and the per-point filter width the tessellator uses, so the two describe one rule.
struct LodRule {
    detail::ViewCone view;
    double ratio = 0.5;     // in view
    double offscreen = 4.0; // multiplier outside the view
    double nearKeep = 24.0; // in-view detail regardless of direction nearer than this
    // Past this distance even the tallest waves stand less than a pixel above their troughs, so
    // their silhouettes can no longer be told from a smoother surface and the ratio grows with
    // distance. Zero keeps it constant out to the horizon.
    double silhouette = 0.0;

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

class Quadtree {
  public:
    // Adjacent tiles may differ by this many levels. Two already lets the detail fall away where
    // nothing looks closely; more buys almost nothing further on an open sea.
    static constexpr int kMaxLevelStep = 2;
    static_assert((OCEAN_TILE_GRID >> kMaxLevelStep) << kMaxLevelStep == OCEAN_TILE_GRID,
                  "A coarser neighbour's vertices must land on the tile's own grid");
    static_assert(kMaxLevelStep <= (int)OCEAN_STITCH_MASK, "Stitch field too narrow");

    // Chooses the visible leaves for this camera and assigns each a persistent slot.
    // `coverage`, when given, decides which tiles become geometry at all; see ocean::ICoverage.
    // `silhouetteDistance` is LodRule::silhouette.
    void Select(const Params& p, const planet::CameraView& cam, uint32_t maxTiles,
                const ICoverage* coverage = nullptr, double silhouetteDistance = 0.0);

    const std::vector<TileDesc>& Tiles() const { return m_tiles; }
    uint32_t SelectedLeafCount() const { return m_leafCount; }
    uint32_t DroppedCount() const { return m_dropped; }
    // Tile edge over distance the selection settled on; above Params::lodFactor when the budget
    // made it back the detail off.
    double EffectiveLodFactor() const { return m_lodFactor; }
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

    // Detail follows the view. The camera sees the sea along silhouettes and at the resolution of
    // its pixels; everything else is seen only by secondary rays, through reflections, shadows
    // and refraction that are blurred and noisy by construction and never graze a silhouette. A
    // widened frustum, so a tile is refined before it turns into view, picks the in-view ratio;
    // the rest takes the coarser one. The whole sea stays in the scene either way.
    LodRule rule(p, cam, silhouetteDistance);
    auto inView = [&](double x, double z, double s) {
        // The tile's square at sea level, fattened by the tallest a wave can plausibly stand.
        return rule.view.Intersects(x + 0.5 * s - cam.position_world.x, seaLevel - cam.position_world.y,
                                    z + 0.5 * s - cam.position_world.z, s * 0.70710678 + 10.0);
    };

    // Keep the complete finite ocean in the ray-visible scene, including behind the camera.
    // Quality selection may coarsen geometry; visibility is never restricted to primary rays.
    // A coverage test is the one thing that does remove a tile: where the world holds no water,
    // the sea has nothing to stand for and every ray would pay to traverse it regardless.
    // Coverage has already pruned the empty regions during the descent; this is the safety net
    // for a tile that reached the emission list some other way, such as through balancing.
    auto isVisible = [&](double x, double z, double s) {
        return coverage == nullptr || coverage->Test(x, z, s) != Coverage::None;
    };

    struct Node {
        uint8_t level;
        uint32_t ix, iy;
    };
    std::vector<Node> stack;

    // A camera high above the water sees far more ocean than one at deck level. Rather than drop
    // tiles when the budget runs out - which would leave holes along the horizon, where the sky
    // would show straight through - back the detail off and select again.
    //
    // Edge a partially covered tile is split down to. Small enough that a tile's box no longer
    // reaches across the scene, large enough that a coastline does not cost hundreds of them. It
    // coarsens alongside the level of detail when the budget is tight, so the two back off
    // together rather than this one fighting the budget on its own.
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

            // Where the world has no water at all, the whole subtree goes: no geometry, no
            // instance, and nothing for a ray crossing that ground to step into.
            const Coverage cov = coverage ? coverage->Test(x, z, s) : Coverage::Full;
            if (cov == Coverage::None)
                continue;

            const bool canSplit = (n.level < maxLevel) && (s > minTile * 1.5);
            // A tile straddling a shoreline is split past what distance alone would ask for,
            // until it is small enough that its bounding box hugs the water instead of reaching
            // across the land beside it. Distance may still split it further.
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

        // Count what would actually be emitted. Balancing only ever adds leaves, so leave headroom
        // for the tiles it will split.
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

    // Balance: a leaf whose neighbour is more than kMaxLevelStep levels coarser forces that
    // neighbour to split; splitting can expose new imbalances, so this runs to a fixed point.
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

    // Balancing may consume more than the reserved headroom. Retry the complete selection
    // at a coarser LOD before assigning slots instead of dropping tiles and opening holes.
    // Increasing the LOD factor eventually selects just the root, which fits any positive budget.
    if (m_set.size() > maxTiles) {
        Params coarser = p;
        coarser.lodFactor = (float)(rule.ratio * 1.4);
        Select(coarser, cam, maxTiles, coverage, silhouetteDistance);
        return;
    }
    m_leafCount = (uint32_t)m_set.size();
    m_lodFactor = rule.ratio;

    // Assign nearby tiles first. The complete balanced selection now fits the tile budget.
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
