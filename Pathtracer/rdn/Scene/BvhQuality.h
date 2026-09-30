#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace bvh {

struct Bounds {
    std::array<float, 3> lo{ INFINITY, INFINITY, INFINITY };
    std::array<float, 3> hi{ -INFINITY, -INFINITY, -INFINITY };
    void point(float x, float y, float z) {
        const float p[3]{ x, y, z };
        for (int a = 0; a < 3; ++a) { lo[a] = std::min(lo[a], p[a]); hi[a] = std::max(hi[a], p[a]); }
    }
    bool valid() const { return lo[0] <= hi[0] && lo[1] <= hi[1] && lo[2] <= hi[2]; }
    void include(const Bounds& b) {
        if (!b.valid()) return;
        for (int a = 0; a < 3; ++a) { lo[a] = std::min(lo[a], b.lo[a]); hi[a] = std::max(hi[a], b.hi[a]); }
    }
    float center(int a) const { return (lo[a] + hi[a]) * .5f; }
    double area() const {
        if (!valid()) return 0;
        const double x = hi[0] - lo[0], y = hi[1] - lo[1], z = hi[2] - lo[2];
        return 2 * (x*y + x*z + y*z);
    }
    bool overlaps(const Bounds& b) const {
        for (int a = 0; a < 3; ++a) if (std::min(hi[a], b.hi[a]) <= std::max(lo[a], b.lo[a])) return false;
        return true;
    }
};

struct Split {
    int axis = -1;
    float position = 0;
    double ratio = 1;
};

// Cost from triangle bounds, not centroid bounds.
template<class Reader> Split choose_split(uint32_t count, Reader&& read) {
    constexpr int N = 16;
    Bounds root, centers;
    for (uint32_t i = 0; i < count; ++i) {
        const Bounds b = read(i);
        root.include(b); centers.point(b.center(0), b.center(1), b.center(2));
    }
    Split best;
    const double unsplit = root.area() * count;
    if (!(unsplit > 0)) return best;
    for (int axis = 0; axis < 3; ++axis) {
        const float extent = centers.hi[axis] - centers.lo[axis];
        if (!(extent > 0)) continue;
        struct Bin { Bounds bounds; uint32_t count = 0; };
        std::array<Bin, N> bins;
        for (uint32_t i = 0; i < count; ++i) {
            const Bounds b = read(i);
            const int j = std::clamp((int)((b.center(axis) - centers.lo[axis]) / extent * N), 0, N-1);
            bins[j].bounds.include(b); ++bins[j].count;
        }
        std::array<Bounds, N> suffix;
        std::array<uint32_t, N> suffixCount{};
        Bounds right;
        uint32_t nr = 0;
        for (int j = N-1; j >= 0; --j) {
            right.include(bins[j].bounds); nr += bins[j].count;
            suffix[j] = right; suffixCount[j] = nr;
        }
        Bounds left;
        uint32_t nl = 0;
        for (int j = 0; j < N-1; ++j) {
            left.include(bins[j].bounds); nl += bins[j].count;
            if (!nl || !suffixCount[j+1]) continue;
            // Fixed traversal cost deters tiny splits.
            const double ratio = (root.area() * 32 + left.area() * nl + suffix[j+1].area() * suffixCount[j+1]) / unsplit;
            if (ratio < best.ratio) best = { axis, centers.lo[axis] + extent * ((j+1.0f)/N), ratio };
        }
    }
    return best;
}

struct Survey {
    uint64_t triangles = 0, overlapPairs = 0;
    uint32_t instances = 0;
    double weightedArea = 0;
};

inline Survey survey(const std::vector<Bounds>& bounds, const std::vector<uint32_t>& triangles) {
    Survey out;
    out.instances = (uint32_t)bounds.size();
    std::vector<uint32_t> order(bounds.size());
    for (uint32_t i = 0; i < order.size(); ++i) {
        order[i] = i; out.triangles += triangles[i]; out.weightedArea += bounds[i].area() * triangles[i];
    }
    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return bounds[a].lo[0] < bounds[b].lo[0]; });
    for (size_t i = 0; i < order.size(); ++i)
        for (size_t j = i+1; j < order.size() && bounds[order[j]].lo[0] < bounds[order[i]].hi[0]; ++j)
            if (bounds[order[i]].overlaps(bounds[order[j]])) ++out.overlapPairs;
    return out;
}

}
