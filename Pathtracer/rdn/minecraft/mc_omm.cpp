#include "mc_omm.h"
#include <algorithm>
#include <cmath>
#include <unordered_set>
#include "block_registry.h"
#include "voxel_mesher.h"

namespace mc {

namespace {

int cutout_texture(const BlockRegistry& reg, uint16_t material) {
    return material < reg.materialCutoutTexture.size() ? reg.materialCutoutTexture[material] : -1;
}

// Chooses subdivision from texture resolution, capped at twelve levels.
uint8_t level_for_texels(float texels) {
    int k = 0;
    while (k < 12 && (float)(1 << k) < texels - 1e-4f) ++k;
    return (uint8_t)k;
}

}

// Enumerates cutout faces and model triangles for micromap baking.
void enumerate_omm_triangles(const BlockRegistry& reg, std::vector<OmmBakeTri>& out) {
    out.clear();
    auto texels_per_tile = [&](int tex) -> float {
        return (size_t)tex < reg.textureAlpha.size() && reg.textureAlpha[(size_t)tex].width > 0
             ? (float)reg.textureAlpha[(size_t)tex].width : 16.0f;
    };
    static const int TRI[2][3] = { { 0, 1, 2 }, { 0, 2, 3 } };

    std::unordered_set<uint64_t> pairs;
    for (size_t i = 1; i < reg.count(); ++i) {
        const BlockInfo& bi = reg.info((BlockId)i);
        if (bi.isCube)
            for (int f = 0; f < 6; ++f) {
                const int t = cutout_texture(reg, bi.faceMaterial[f]);
                if (t >= 0) pairs.insert((uint64_t)t);
            }
        for (int L = 1; L <= OMM_MAX_LEVEL; ++L)
            for (int f = 0; f < 6; ++f) {
                const int t = cutout_texture(reg, bi.lodFaceMaterial[f]);
                if (t >= 0) pairs.insert((uint64_t)t | ((uint64_t)L << 32));
            }
    }
    std::vector<uint64_t> sorted(pairs.begin(), pairs.end());
    std::sort(sorted.begin(), sorted.end());
    for (uint64_t p : sorted) {
        const uint32_t tex = (uint32_t)(p & 0xFFFFFFFFu);
        const int L = (int)(p >> 32);
        const int cap = omm_merge_cap(L);
        const float uvScale = (float)(1 << L);
        for (int f = 0; f < 6; ++f)
        for (int h = 1; h <= cap; ++h)
        for (int w = 1; w <= cap; ++w) {
            float uv[4][2];
            ChunkMesher::face_quad_uvs(f, w, h, uvScale, uv);
            const uint8_t level = level_for_texels((float)std::max(w, h) * uvScale * texels_per_tile((int)tex));
            for (int t = 0; t < 2; ++t) {
                OmmBakeTri b;
                b.key = omm_key_face(tex, f, w, h, L, t);
                b.tex = tex;
                b.level = level;
                for (int k = 0; k < 3; ++k) { b.uv[k][0] = uv[TRI[t][k]][0]; b.uv[k][1] = uv[TRI[t][k]][1]; }
                out.push_back(b);
            }
        }
    }

    for (size_t i = 1; i < reg.count(); ++i) {
        const BlockInfo& bi = reg.info((BlockId)i);
        if (!bi.hasQuads || bi.isCube) continue;
        for (uint32_t q = 0; q < bi.quadCount; ++q) {
            const uint32_t qi = bi.quadBegin + q;
            const BlockQuad& bq = reg.quads[qi];
            const int t = cutout_texture(reg, bq.material);
            if (t < 0) continue;
            float u0 = 1e9f, u1 = -1e9f, v0 = 1e9f, v1 = -1e9f;
            for (int k = 0; k < 4; ++k) {
                u0 = std::min(u0, bq.uv[k][0]); u1 = std::max(u1, bq.uv[k][0]);
                v0 = std::min(v0, bq.uv[k][1]); v1 = std::max(v1, bq.uv[k][1]);
            }
            const uint8_t level = level_for_texels(std::max(u1 - u0, v1 - v0) * texels_per_tile(t));
            for (int tt = 0; tt < 2; ++tt) {
                OmmBakeTri b;
                b.key = omm_key_quad(qi, tt);
                b.tex = (uint32_t)t;
                b.level = level;
                for (int k = 0; k < 3; ++k) { b.uv[k][0] = bq.uv[TRI[tt][k]][0]; b.uv[k][1] = bq.uv[TRI[tt][k]][1]; }
                out.push_back(b);
            }
        }
    }
}

}
