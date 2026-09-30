#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>
#include "mc_types.h"

namespace mc {

class BlockRegistry;

constexpr int OMM_MAX_LEVEL    = 2;
constexpr int OMM_MERGE_CAP_L0 = 4;
inline int omm_merge_cap(int level) { return level == 0 ? OMM_MERGE_CAP_L0 : 1; }

inline uint64_t omm_key_face(uint32_t tex, int face, int w, int h, int level, int tri) {
    return 1ull | ((uint64_t)tex << 2) | ((uint64_t)face << 22) | ((uint64_t)w << 25) | ((uint64_t)h << 31)
         | ((uint64_t)level << 37) | ((uint64_t)tri << 41);
}
inline uint64_t omm_key_quad(uint32_t quadIndex, int tri) {
    return 2ull | ((uint64_t)quadIndex << 2) | ((uint64_t)tri << 26);
}

struct OmmTable {
    std::unordered_map<uint64_t, int32_t, U64Hash> index;
    int32_t lookup(uint64_t key, int32_t fallback) const {
        const auto it = index.find(key);
        return it == index.end() ? fallback : it->second;
    }
    bool empty() const { return index.empty(); }
};

struct OmmBakeTri {
    uint64_t key;
    uint32_t tex;
    uint8_t  level;
    float    uv[3][2];
};

void enumerate_omm_triangles(const BlockRegistry& reg, std::vector<OmmBakeTri>& out);

}
