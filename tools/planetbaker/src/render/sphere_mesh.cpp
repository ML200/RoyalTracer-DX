#include "render/sphere_mesh.h"

#include <cmath>
#include <unordered_map>
#include <utility>

namespace pb {

static glm::vec3 unit(const glm::vec3& v) {
    return v / glm::length(v);
}

SphereMesh make_icosphere(int subdivisions) {
    const float t = (1.0f + std::sqrt(5.0f)) * 0.5f;

    std::vector<glm::vec3> verts = {
        unit({-1,  t,  0}), unit({ 1,  t,  0}), unit({-1, -t,  0}), unit({ 1, -t,  0}),
        unit({ 0, -1,  t}), unit({ 0,  1,  t}), unit({ 0, -1, -t}), unit({ 0,  1, -t}),
        unit({ t,  0, -1}), unit({ t,  0,  1}), unit({-t,  0, -1}), unit({-t,  0,  1})
    };

    std::vector<uint32_t> faces = {
        0, 11,  5,   0,  5,  1,   0,  1,  7,   0,  7, 10,   0, 10, 11,
        1,  5,  9,   5, 11,  4,  11, 10,  2,  10,  7,  6,   7,  1,  8,
        3,  9,  4,   3,  4,  2,   3,  2,  6,   3,  6,  8,   3,  8,  9,
        4,  9,  5,   2,  4, 11,   6,  2, 10,   8,  6,  7,   9,  8,  1
    };

    auto edge_key = [](uint32_t a, uint32_t b) -> uint64_t {
        if (a > b) std::swap(a, b);
        return (static_cast<uint64_t>(a) << 32) | static_cast<uint64_t>(b);
    };

    for (int s = 0; s < subdivisions; ++s) {
        std::unordered_map<uint64_t, uint32_t> cache;
        std::vector<uint32_t> next;
        next.reserve(faces.size() * 4);

        auto midpoint = [&](uint32_t a, uint32_t b) -> uint32_t {
            uint64_t k = edge_key(a, b);
            auto it = cache.find(k);
            if (it != cache.end()) return it->second;
            uint32_t idx = static_cast<uint32_t>(verts.size());
            verts.push_back(unit((verts[a] + verts[b]) * 0.5f));
            cache.emplace(k, idx);
            return idx;
        };

        for (size_t i = 0; i < faces.size(); i += 3) {
            uint32_t a = faces[i], b = faces[i + 1], c = faces[i + 2];
            uint32_t ab = midpoint(a, b);
            uint32_t bc = midpoint(b, c);
            uint32_t ca = midpoint(c, a);
            next.insert(next.end(), { a, ab, ca,  b, bc, ab,  c, ca, bc,  ab, bc, ca });
        }
        faces = std::move(next);
    }

    return SphereMesh{ std::move(verts), std::move(faces) };
}

}
