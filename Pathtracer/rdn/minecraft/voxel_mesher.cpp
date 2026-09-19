#include "voxel_mesher.h"
#include "mc_omm.h"
#include "block_registry.h"
#include "voxel_store.h"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace mc {

ChunkMesher::ChunkMesher(const BlockRegistry& reg, const VoxelStore& store)
    : m_reg(reg), m_store(store) {
    m_window.resize((size_t)W * W * W);
    m_child.resize((size_t)CW * CW * CW);
    m_mask.resize((size_t)CHUNK_SIZE * CHUNK_SIZE);
    m_cornerVertex.assign((size_t)6 * CORNERS * CORNERS * CORNERS, 0u);
    m_cornerGen.assign(m_cornerVertex.size(), 0u);
}

bool ChunkMesher::lamp_voxel(Voxel v, int level, const BlockInfo& info, float& sideBlocks) {
    if (level == 0 || !info.emissive || (v & VOX_ANY) == 0) return false;
    const float voxel = (float)(1 << level);
    const uint32_t n = voxel_emissive(v, info.water != 0);
    float side = std::sqrt((float)std::max(n, 1u));
    if (side >= 0.75f * voxel) return false;
    sideBlocks = std::min(side, voxel);
    return true;
}

bool ChunkMesher::renderable(Voxel v, int level, const BlockInfo*& info) const {
    const BlockId id = voxel_id(v);
    if (id == AIR_ID) return false;
    info = &m_reg.info(id);
    if (level == 0) return info->isCube;
    if ((v & VOX_ANY) == 0) return false;
    float side;
    return !lamp_voxel(v, level, *info, side);
}

// Applies opacity and same-state rules to one neighboring voxel.
bool ChunkMesher::occludes(Voxel neighbour, Voxel self, int level, bool selfCullSame) const {
    const BlockId nid = voxel_id(neighbour);
    if (nid == AIR_ID) return false;
    const BlockInfo& ni = m_reg.info(nid);
    const bool same = nid == voxel_id(self) || (ni.water && m_reg.info(voxel_id(self)).water);
    if (level == 0) {
        if (ni.fullOpaque) return true;
        return selfCullSame && same && ni.isCube;
    }
    if (neighbour & VOX_ALL) return true;
    return selfCullSame && same && (neighbour & VOX_ANY);
}

void ChunkMesher::push_triangles(uint32_t i0, uint32_t i1, uint32_t i2, uint32_t i3, uint16_t material, ChunkMesh& out, uint64_t omm0, uint64_t omm1) {
    const bool alpha = material < m_reg.materialAlpha.size() && m_reg.materialAlpha[material] != 0;
    bool lit = false;
    if (material < m_reg.materialEmission.size()) {
        const Vec3f& e = m_reg.materialEmission[material];
        lit = e.x + e.y + e.z > 0.0f;
    }
    Bucket& b = alpha ? (lit ? m_alphaLit : m_alpha) : (lit ? m_opaqueLit : m_opaque);
    b.idx.push_back(i0); b.idx.push_back(i1); b.idx.push_back(i2);
    b.idx.push_back(i0); b.idx.push_back(i2); b.idx.push_back(i3);
    b.mat.push_back(material); b.mat.push_back(material);
    b.omm.push_back(omm0); b.omm.push_back(omm1);
    ++out.quadCount;
}

void ChunkMesher::emit_quad(const Vec3f p[4], const float uv[4][2], const Vec3f& n, uint16_t material, ChunkMesh& out, uint64_t omm0, uint64_t omm1) {
    const uint32_t base = (uint32_t)out.vertices.size();
    const uint32_t pn = pack_normal_oct16(n);
    for (int k = 0; k < 4; ++k) {
        MeshVertex v;
        v.px = p[k].x; v.py = p[k].y; v.pz = p[k].z;
        v.packedNormal = pn;
        v.u = float_to_half(uv[k][0]);
        v.v = float_to_half(uv[k][1]);
        out.vertices.push_back(v);
    }
    push_triangles(base, base + 1, base + 2, base + 3, material, out, omm0, omm1);
}

namespace {
using FaceAxes = FaceProjection;
constexpr const FaceProjection* FACE_AXES = FACE_PROJECTION;
}

namespace {
constexpr float VOLUME_INSET = 0.02f;
constexpr uint32_t MATERIAL_KEY_MASK = 0xFFFFu;
constexpr int WATER_HEIGHT_KEY_SHIFT = 16;
constexpr uint32_t WATER_HEIGHT_KEY_MASK = 0xFFFu;
constexpr uint32_t INSET_KEY_BIT = 1u << 28;
constexpr int WATER_BOTTOM_KEY_SHIFT = 32;
}

void ChunkMesher::face_quad_uvs(int face, int w, int h, float uvScale, float uv[4][2]) {
    const FaceAxes& ax = FACE_AXES[face];
    const int dir = FACE_DIR[face][ax.na];
    const int plane = dir > 0 ? 1 : 0;
    const int as[4] = { 0, w, w, 0 };
    const int bs[4] = { 0, 0, h, h };
    int corner[4][3];
    for (int k = 0; k < 4; ++k) { corner[k][ax.na] = plane; corner[k][ax.ua] = as[k]; corner[k][ax.va] = bs[k]; }
    const Vec3f p0{ (float)corner[0][0], (float)corner[0][1], (float)corner[0][2] };
    const Vec3f p1{ (float)corner[1][0], (float)corner[1][1], (float)corner[1][2] };
    const Vec3f p2{ (float)corner[2][0], (float)corner[2][1], (float)corner[2][2] };
    const Vec3f n{ (float)FACE_DIR[face][0], (float)FACE_DIR[face][1], (float)FACE_DIR[face][2] };
    if (dot(cross(p1 - p0, p2 - p0), n) < 0.0f)
        for (int c = 0; c < 3; ++c) { const int t = corner[1][c]; corner[1][c] = corner[3][c]; corner[3][c] = t; }
    for (int k = 0; k < 4; ++k) {
        uv[k][0] = ax.su * (float)corner[k][ax.ua] * uvScale;
        uv[k][1] = ax.sv * (float)corner[k][ax.va] * uvScale;
    }
}

void ChunkMesher::emit_face_quad(int face, const int corner[4][3], float s, float uvScale, float insetVoxels,
                                 uint16_t material, ChunkMesh& out, uint64_t omm0, uint64_t omm1, float waterTopOffset, float waterBottomOffset) {
    const FaceAxes& ax = FACE_AXES[face];
    const int dir = FACE_DIR[face][ax.na];
    const Vec3f n{ (float)FACE_DIR[face][0], (float)FACE_DIR[face][1], (float)FACE_DIR[face][2] };
    const uint32_t pn = pack_normal_oct16(n);
    const int top = std::max({ corner[0][1], corner[1][1], corner[2][1], corner[3][1] });
    const int bottom = std::min({ corner[0][1], corner[1][1], corner[2][1], corner[3][1] });
    uint32_t idx[4];
    for (int k = 0; k < 4; ++k) {
        const int cx = corner[k][0], cy = corner[k][1], cz = corner[k][2];
        const bool shareable = insetVoxels == 0.0f && waterTopOffset == 0.0f && waterBottomOffset == 0.0f && cx >= 0 && cx < CORNERS && cy >= 0 && cy < CORNERS && cz >= 0 && cz < CORNERS;
        const size_t slot = shareable ? (((size_t)face * CORNERS + (size_t)cz) * CORNERS + (size_t)cy) * CORNERS + (size_t)cx : 0;
        if (shareable && m_cornerGen[slot] == m_gen) { idx[k] = m_cornerVertex[slot]; continue; }
        MeshVertex v;
        float pos[3] = { (float)cx * s, (float)cy * s, (float)cz * s };
        pos[ax.na] -= (float)dir * insetVoxels * s;
        if (cy == top) pos[1] -= waterTopOffset;
        if (cy == bottom) pos[1] += waterBottomOffset;
        v.px = pos[0]; v.py = pos[1]; v.pz = pos[2];
        v.packedNormal = pn;
        v.u = float_to_half(ax.su * (float)corner[k][ax.ua] * uvScale);
        v.v = float_to_half(ax.sv * (float)corner[k][ax.va] * uvScale);
        if (ax.va == 1) {
            const float waterOffset = (cy == bottom ? waterBottomOffset : 0.0f) - (cy == top ? waterTopOffset : 0.0f);
            v.v = float_to_half(ax.sv * ((float)cy + waterOffset / s) * uvScale);
        }
        idx[k] = (uint32_t)out.vertices.size();
        out.vertices.push_back(v);
        if (shareable) { m_cornerGen[slot] = m_gen; m_cornerVertex[slot] = idx[k]; }
    }
    push_triangles(idx[0], idx[1], idx[2], idx[3], material, out, omm0, omm1);
}

// Greedily merges coplanar cube faces with matching materials.
void ChunkMesher::greedy_faces(int level, const MeshParams& p, ChunkMesh& out) {
    const float s = (float)(1 << level);
    const float uvScale = (32.0f * s <= 16384.0f) ? s : 1.0f;
    for (int f = 0; f < 6; ++f) {
        const FaceAxes& ax = FACE_AXES[f];
        const int dir = FACE_DIR[f][ax.na];
        for (int slice = 0; slice < CHUNK_SIZE; ++slice) {
            bool anyFace = false;
            for (int b = 0; b < CHUNK_SIZE; ++b)
            for (int a = 0; a < CHUNK_SIZE; ++a) {
                int c[3];
                c[ax.na] = slice; c[ax.ua] = a; c[ax.va] = b;
                const Voxel v = at(c[0], c[1], c[2]);
                uint64_t key = 0;
                const BlockInfo* bi = nullptr;
                if (renderable(v, level, bi)) {
                    int nb[3] = { c[0], c[1], c[2] };
                    nb[ax.na] += dir;
                    const Voxel nv = at(nb[0], nb[1], nb[2]);
                    bool hidden = occludes(nv, v, level, bi->cullSameId);
                    bool inset = false;
                    uint32_t waterBottom = 0;
                    if (bi->water && f >= FACE_NORTH && m_reg.info(voxel_id(nv)).water && (level == 0 || (nv & VOX_ANY))) {
                        // Close only the exposed step between neighboring water tops.
                        waterBottom = voxel_water_height(nv, level);
                        hidden = waterBottom >= voxel_water_height(v, level);
                    }
                    if (hidden && bi->volume && voxel_id(nv) != voxel_id(v) && !(bi->water && m_reg.info(voxel_id(nv)).water)) { hidden = false; inset = true; }
                    if (!hidden) {
                        const uint16_t m = face_material(level, c, f, *bi, p.flatMaterials);
                        if (m != NO_MATERIAL) {
                            const uint32_t height = bi->water && f != FACE_DOWN ? voxel_water_height(v, level) : 0u;
                            key = ((uint32_t)m + 1u) | (height << WATER_HEIGHT_KEY_SHIFT) | (inset ? INSET_KEY_BIT : 0u)
                                | ((uint64_t)waterBottom << WATER_BOTTOM_KEY_SHIFT);
                            anyFace = true;
                        }
                    }
                }
                m_mask[(size_t)b * CHUNK_SIZE + a] = key;
            }
            if (!anyFace) continue;
            for (int b = 0; b < CHUNK_SIZE; ++b)
            for (int a = 0; a < CHUNK_SIZE; ) {
                const uint64_t key = m_mask[(size_t)b * CHUNK_SIZE + a];
                if (key == 0) { ++a; continue; }
                const uint16_t mat = (uint16_t)((key & MATERIAL_KEY_MASK) - 1u);
                const int cutoutTex = (level <= OMM_MAX_LEVEL && mat < m_reg.materialCutoutTexture.size()) ? m_reg.materialCutoutTexture[mat] : -1;
                const int cap = cutoutTex >= 0 ? omm_merge_cap(level) : CHUNK_SIZE;
                const uint32_t waterBottom = (uint32_t)(key >> WATER_BOTTOM_KEY_SHIFT);
                int w = 1;
                while (a + w < CHUNK_SIZE && w < cap && m_mask[(size_t)b * CHUNK_SIZE + a + w] == key) ++w;
                int h = 1;
                for (; b + h < CHUNK_SIZE && h < (waterBottom ? 1 : cap); ++h) {
                    bool ok = true;
                    for (int k = 0; k < w; ++k)
                        if (m_mask[(size_t)(b + h) * CHUNK_SIZE + a + k] != key) { ok = false; break; }
                    if (!ok) break;
                }
                for (int hh = 0; hh < h; ++hh)
                    for (int k = 0; k < w; ++k) m_mask[(size_t)(b + hh) * CHUNK_SIZE + a + k] = 0;

                const bool inset = (key & INSET_KEY_BIT) != 0u;
                const int plane = dir > 0 ? slice + 1 : slice;
                const int as[4] = { a, a + w, a + w, a };
                const int bs[4] = { b, b, b + h, b + h };
                int corner[4][3];
                for (int k = 0; k < 4; ++k) {
                    corner[k][ax.na] = plane;
                    corner[k][ax.ua] = as[k];
                    corner[k][ax.va] = bs[k];
                }
                const Vec3f p0{ (float)corner[0][0], (float)corner[0][1], (float)corner[0][2] };
                const Vec3f p1{ (float)corner[1][0], (float)corner[1][1], (float)corner[1][2] };
                const Vec3f p2{ (float)corner[2][0], (float)corner[2][1], (float)corner[2][2] };
                const Vec3f n{ (float)FACE_DIR[f][0], (float)FACE_DIR[f][1], (float)FACE_DIR[f][2] };
                if (dot(cross(p1 - p0, p2 - p0), n) < 0.0f) {
                    for (int c = 0; c < 3; ++c) { const int t = corner[1][c]; corner[1][c] = corner[3][c]; corner[3][c] = t; }
                }
                const bool ommable = cutoutTex >= 0 && !inset && uvScale == s;
                const uint32_t waterHeight = (key >> WATER_HEIGHT_KEY_SHIFT) & WATER_HEIGHT_KEY_MASK;
                const float waterTopOffset = waterHeight ? s - (float)waterHeight : 0.0f;
                emit_face_quad(f, corner, s, uvScale, inset ? VOLUME_INSET / s : 0.0f, mat, out,
                               ommable ? omm_key_face((uint32_t)cutoutTex, f, w, h, level, 0) : 0ull,
                               ommable ? omm_key_face((uint32_t)cutoutTex, f, w, h, level, 1) : 0ull, waterTopOffset, (float)waterBottom);
                a += w;
            }
        }
    }
}

uint16_t ChunkMesher::face_material(int level, const int c[3], int f, const BlockInfo& parent, bool flat) const {
    if (level == 0) return parent.faceMaterial[f];
    auto mat_of = [&](const BlockInfo& bi) -> uint16_t { return flat ? bi.flatFaceMaterial[f] : bi.lodFaceMaterial[f]; };
    const int childLevel = level - 1;
    const FaceProjection& pr = FACE_PROJECTION[f];
    const int dir = FACE_DIR[f][pr.na];
    struct Cand { BlockId id; uint16_t mat; int near; int row; int sig; int lit; int count; bool solid; };
    Cand cands[8];
    int n = 0;
    for (int i = 0; i < 8; ++i) {
        const int cc[3] = { 2 * c[0] + (i & 1), 2 * c[1] + ((i >> 1) & 1), 2 * c[2] + ((i >> 2) & 1) };
        const Voxel cv = child_at(cc[0], cc[1], cc[2]);
        const BlockId id = voxel_id(cv);
        if (id == AIR_ID) continue;
        if (childLevel > 0 && (cv & VOX_ANY) == 0) continue;
        const BlockInfo& bi = m_reg.info(id);
        const Voxel nv = child_at(cc[0] + FACE_DIR[f][0], cc[1] + FACE_DIR[f][1], cc[2] + FACE_DIR[f][2]);
        if (occludes(nv, cv, childLevel, bi.cullSameId)) continue;
        const uint16_t m = mat_of(bi);
        if (m == NO_MATERIAL) continue;
        const int near = ((cc[pr.na] & 1) == (dir > 0 ? 1 : 0)) ? 1 : 0;
        const int row  = pr.na == 1 ? 0 : (cc[1] & 1);
        bool merged = false;
        for (int k = 0; k < n; ++k) {
            if (cands[k].id != id) continue;
            ++cands[k].count;
            cands[k].near = std::max(cands[k].near, near);
            cands[k].row  = std::min(cands[k].row, row);
            merged = true;
            break;
        }
        if (!merged) cands[n++] = Cand{ id, m, near, row, (int)bi.sig, bi.emissive ? 1 : 0, 1, ((bi.lodFaceSolid >> f) & 1) != 0 };
    }
    if (n == 0) return mat_of(parent);
    int best = 0;
    for (int k = 1; k < n; ++k) {
        const Cand& a = cands[k];
        const Cand& b = cands[best];
        const bool aReal = a.sig > 0 || a.solid, bReal = b.sig > 0 || b.solid;
        const bool better = aReal != bReal ? aReal
            : a.near  != b.near  ? a.near > b.near
            : a.count != b.count ? a.count > b.count
            : a.row   != b.row   ? a.row < b.row
            : a.lit   != b.lit   ? a.lit < b.lit
            : a.sig   != b.sig   ? a.sig > b.sig
            : a.id < b.id;
        if (better) best = k;
    }
    return cands[best].mat;
}

void ChunkMesher::lamp_cubes(int level, const MeshParams& p, ChunkMesh& out) {
    const float s = (float)(1 << level);
    for (int z = 0; z < CHUNK_SIZE; ++z)
    for (int y = 0; y < CHUNK_SIZE; ++y)
    for (int x = 0; x < CHUNK_SIZE; ++x) {
        const Voxel v = at(x, y, z);
        const BlockId id = voxel_id(v);
        if (id == AIR_ID) continue;
        const BlockInfo& bi = m_reg.info(id);
        float side;
        if (!lamp_voxel(v, level, bi, side)) continue;
        const Vec3f c{ ((float)x + 0.5f) * s, ((float)y + 0.5f) * s, ((float)z + 0.5f) * s };
        const float h = 0.5f * side;
        for (int f = 0; f < 6; ++f) {
            const uint16_t m = p.flatMaterials ? bi.flatFaceMaterial[f] : bi.lodFaceMaterial[f];
            if (m == NO_MATERIAL) continue;
            const FaceAxes& ax = FACE_AXES[f];
            const int dir = FACE_DIR[f][ax.na];
            const float as[4] = { -h, h, h, -h };
            const float bs[4] = { -h, -h, h, h };
            Vec3f pos[4];
            float uv[4][2];
            for (int k = 0; k < 4; ++k) {
                float q[3];
                q[ax.na] = (float)dir * h;
                q[ax.ua] = as[k];
                q[ax.va] = bs[k];
                pos[k] = Vec3f{ c.x + q[0], c.y + q[1], c.z + q[2] };
                uv[k][0] = ax.su * (as[k] + h);
                uv[k][1] = ax.sv * (bs[k] + h);
            }
            const Vec3f n{ (float)FACE_DIR[f][0], (float)FACE_DIR[f][1], (float)FACE_DIR[f][2] };
            if (dot(cross(pos[1] - pos[0], pos[2] - pos[0]), n) < 0.0f) {
                Vec3f tp = pos[1]; pos[1] = pos[3]; pos[3] = tp;
                float tu0 = uv[1][0], tu1 = uv[1][1];
                uv[1][0] = uv[3][0]; uv[1][1] = uv[3][1];
                uv[3][0] = tu0; uv[3][1] = tu1;
            }
            emit_quad(pos, uv, n, m, out);
        }
    }
}

// Emits non-cube model quads and preserves their culling metadata.
void ChunkMesher::model_quads(const MeshParams& p, ChunkMesh& out) {
    (void)p;
    for (int z = 0; z < CHUNK_SIZE; ++z)
    for (int y = 0; y < CHUNK_SIZE; ++y)
    for (int x = 0; x < CHUNK_SIZE; ++x) {
        const Voxel v = at(x, y, z);
        const BlockId id = voxel_id(v);
        if (id == AIR_ID) continue;
        const BlockInfo& bi = m_reg.info(id);
        if (!bi.hasQuads || bi.isCube) continue;
        const Vec3f off{ (float)x, (float)y, (float)z };
        for (uint32_t q = 0; q < bi.quadCount; ++q) {
            const BlockQuad& bq = m_reg.quads[bi.quadBegin + q];
            if (bq.material == NO_MATERIAL) continue;
            if (bq.cullFace != FACE_NONE) {
                const int* d = FACE_DIR[bq.cullFace];
                const Voxel nv = at(x + d[0], y + d[1], z + d[2]);
                if (occludes(nv, v, 0, bi.cullSameId)) continue;
            }
            Vec3f pos[4];
            for (int k = 0; k < 4; ++k) pos[k] = bq.pos[k] + off;
            const int cutoutTex = bq.material < m_reg.materialCutoutTexture.size() ? m_reg.materialCutoutTexture[bq.material] : -1;
            const uint32_t qi = bi.quadBegin + q;
            emit_quad(pos, bq.uv, bq.normal, bq.material, out,
                      cutoutTex >= 0 ? omm_key_quad(qi, 0) : 0ull, cutoutTex >= 0 ? omm_key_quad(qi, 1) : 0ull);
        }
    }
}

// Builds a chunk mesh from a padded neighbor window.
void ChunkMesher::mesh(const NodeKey& key, const MeshParams& params, ChunkMesh& out) {
    out.clear();
    m_opaqueLit.clear(); m_opaque.clear(); m_alphaLit.clear(); m_alpha.clear();
    if (++m_gen == 0u) { std::fill(m_cornerGen.begin(), m_cornerGen.end(), 0u); m_gen = 1u; }
    const int level = key.level;
    m_store.fill_window(level, key.x * CHUNK_SIZE - 1, key.y * CHUNK_SIZE - 1, key.z * CHUNK_SIZE - 1, W, m_window.data());
    if (level > 0)
        m_store.fill_window(level - 1, key.x * 2 * CHUNK_SIZE - 1, key.y * 2 * CHUNK_SIZE - 1, key.z * 2 * CHUNK_SIZE - 1, CW, m_child.data());

    greedy_faces(level, params, out);
    if (level == 0) model_quads(params, out);
    else lamp_cubes(level, params, out);

    out.opaqueLightTriCount = (uint32_t)(m_opaqueLit.idx.size() / 3);
    out.alphaLightTriCount  = (uint32_t)(m_alphaLit.idx.size() / 3);
    out.opaqueTriCount = out.opaqueLightTriCount + (uint32_t)(m_opaque.idx.size() / 3);
    out.alphaTriCount  = out.alphaLightTriCount  + (uint32_t)(m_alpha.idx.size() / 3);
    const Bucket* order[4] = { &m_opaqueLit, &m_opaque, &m_alphaLit, &m_alpha };
    size_t idxTotal = 0, matTotal = 0;
    for (const Bucket* b : order) { idxTotal += b->idx.size(); matTotal += b->mat.size(); }
    out.indices.reserve(idxTotal);
    out.materials.reserve(matTotal);
    out.ommKeys.reserve(matTotal);
    for (const Bucket* b : order) {
        out.indices.insert(out.indices.end(), b->idx.begin(), b->idx.end());
        out.materials.insert(out.materials.end(), b->mat.begin(), b->mat.end());
        out.ommKeys.insert(out.ommKeys.end(), b->omm.begin(), b->omm.end());
    }
}

}
