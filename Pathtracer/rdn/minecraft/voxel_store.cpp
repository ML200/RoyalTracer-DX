#include "voxel_store.h"
#include "block_registry.h"
#include "../planet/worker_pool.h"
#include <algorithm>
#include <cstring>

namespace mc {

void VoxelStore::configure(int minSectionY, int maxSectionY) {
    m_minSy0 = minSectionY;
    m_maxSy0 = maxSectionY;
    m_levels.clear();
    m_levels.emplace_back(std::make_unique<ColumnMap>());
}

void VoxelStore::put_section(int level, int sx, int sy, int sz, Section&& s) {
    while ((int)m_levels.size() <= level) m_levels.emplace_back(std::make_unique<ColumnMap>());
    const int minSy = min_section_y(level), maxSy = max_section_y(level);
    if (sy < minSy || sy > maxSy) return;
    Column& col = (*m_levels[level])[pack_xz(sx, sz)];
    if (col.sections.empty()) col.sections.resize((size_t)(maxSy - minSy + 1));
    auto& slot = col.sections[(size_t)(sy - minSy)];
    if (s.all_air()) { slot.reset(); return; }
    slot = std::make_unique<Section>(std::move(s));
}

const Section* VoxelStore::section(int level, int sx, int sy, int sz) const {
    if (level < 0 || level >= (int)m_levels.size()) return nullptr;
    const int minSy = min_section_y(level), maxSy = max_section_y(level);
    if (sy < minSy || sy > maxSy) return nullptr;
    const ColumnMap& m = *m_levels[level];
    const auto it = m.find(pack_xz(sx, sz));
    if (it == m.end()) return nullptr;
    const Column& col = it->second;
    if (col.sections.empty()) return nullptr;
    return col.sections[(size_t)(sy - minSy)].get();
}

Section& VoxelStore::ensure_section(int level, int sx, int sy, int sz) {
    while ((int)m_levels.size() <= level) m_levels.emplace_back(std::make_unique<ColumnMap>());
    const int minSy = min_section_y(level), maxSy = max_section_y(level);
    Column& col = (*m_levels[level])[pack_xz(sx, sz)];
    if (col.sections.empty()) col.sections.resize((size_t)(maxSy - minSy + 1));
    auto& slot = col.sections[(size_t)(sy - minSy)];
    if (!slot) slot = std::make_unique<Section>(Section::uniform(0));
    return *slot;
}

Voxel VoxelStore::get(int level, int x, int y, int z) const {
    const Section* s = section(level, floor_shift(x, 4), floor_shift(y, 4), floor_shift(z, 4));
    if (!s) return 0;
    return s->get(x & 15, y & 15, z & 15);
}

void VoxelStore::fill_window(int level, int x0, int y0, int z0, int n, Voxel* out) const {
    std::shared_lock<std::shared_mutex> lk(m_lock);
    const int x1 = x0 + n - 1, y1 = y0 + n - 1, z1 = z0 + n - 1;
    std::memset(out, 0, sizeof(Voxel) * (size_t)n * n * n);
    for (int sz = floor_shift(z0, 4); sz <= floor_shift(z1, 4); ++sz)
    for (int sy = floor_shift(y0, 4); sy <= floor_shift(y1, 4); ++sy)
    for (int sx = floor_shift(x0, 4); sx <= floor_shift(x1, 4); ++sx) {
        const Section* s = section(level, sx, sy, sz);
        if (!s) continue;
        const int bx0 = std::max(x0, sx * 16), bx1 = std::min(x1, sx * 16 + 15);
        const int by0 = std::max(y0, sy * 16), by1 = std::min(y1, sy * 16 + 15);
        const int bz0 = std::max(z0, sz * 16), bz1 = std::min(z1, sz * 16 + 15);
        if (s->is_uniform()) {
            const Voxel v = s->uniform_value();
            if (v == 0) continue;
            for (int z = bz0; z <= bz1; ++z)
            for (int y = by0; y <= by1; ++y) {
                Voxel* row = out + ((size_t)(z - z0) * n + (size_t)(y - y0)) * n + (bx0 - x0);
                for (int x = bx0; x <= bx1; ++x) *row++ = v;
            }
            continue;
        }
        for (int z = bz0; z <= bz1; ++z)
        for (int y = by0; y <= by1; ++y) {
            Voxel* row = out + ((size_t)(z - z0) * n + (size_t)(y - y0)) * n + (bx0 - x0);
            for (int x = bx0; x <= bx1; ++x) *row++ = s->get(x & 15, y & 15, z & 15);
        }
    }
}

bool VoxelStore::chunk_occupied(const NodeKey& k) const {
    if (k.level < 0 || k.level >= (int)m_levels.size()) return false;
    for (int dz = 0; dz < CHUNK_SECTIONS; ++dz)
    for (int dy = 0; dy < CHUNK_SECTIONS; ++dy)
    for (int dx = 0; dx < CHUNK_SECTIONS; ++dx)
        if (section(k.level, k.x * CHUNK_SECTIONS + dx, k.y * CHUNK_SECTIONS + dy, k.z * CHUNK_SECTIONS + dz)) return true;
    return false;
}

std::vector<uint64_t> VoxelStore::occupied_chunk_keys() const {
    std::vector<uint64_t> keys;
    for (int L = 0; L < (int)m_levels.size(); ++L) {
        const int minSy = min_section_y(L);
        for (const auto& kv : *m_levels[L]) {
            const int sx = unpack_x(kv.first), sz = unpack_z(kv.first);
            const std::vector<std::unique_ptr<Section>>& secs = kv.second.sections;
            for (size_t i = 0; i < secs.size(); ++i) {
                if (!secs[i]) continue;
                const int sy = minSy + (int)i;
                keys.push_back(pack_node(NodeKey{ (uint8_t)L, floor_shift(sx, 1), floor_shift(sy, 1), floor_shift(sz, 1) }));
            }
        }
    }
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    return keys;
}

bool VoxelStore::column_bounds(int level, int& minSx, int& maxSx, int& minSz, int& maxSz) const {
    if (level < 0 || level >= (int)m_levels.size() || m_levels[level]->empty()) return false;
    minSx = minSz = INT32_MAX; maxSx = maxSz = INT32_MIN;
    for (const auto& kv : *m_levels[level]) {
        const int x = unpack_x(kv.first), z = unpack_z(kv.first);
        minSx = std::min(minSx, x); maxSx = std::max(maxSx, x);
        minSz = std::min(minSz, z); maxSz = std::max(maxSz, z);
    }
    return true;
}

namespace {
inline bool occluding(Voxel n, int level, const BlockRegistry& reg) {
    const BlockId id = voxel_id(n);
    if (id == AIR_ID) return false;
    if (level == 0) return reg.info(id).fullOpaque;
    return (n & VOX_ALL) != 0;
}
}

Voxel VoxelStore::downsample(const BlockRegistry& reg, int childLevel, const Voxel v[8], uint8_t exposed, uint8_t exposedUp) const {
    bool occ = false, all = true, any = false;
    BlockId bestId = AIR_ID;
    int bestUp = -1, bestExposed = -1, bestSignificant = -1, bestCount = 0, bestLit = 1, bestSig = -1;
    uint32_t emissive = 0;
    BlockId  lampId = AIR_ID;
    const bool decorAllowed = childLevel + 1 <= m_decorMaxLevel;
    for (int i = 0; i < 8; ++i) {
        const Voxel c = v[i];
        const BlockId id = voxel_id(c);
        const bool nonAir = (id != AIR_ID) || (c & VOX_OCC);
        occ = occ || nonAir;
        bool cAny, cAll;
        int  cSig;
        const BlockInfo& bi = reg.info(id);
        if (childLevel == 0) {
            cSig = (id == AIR_ID) ? 0 : (int)bi.sig;
            cAny = id != AIR_ID && (cSig > 0 || (decorAllowed && (bi.hasQuads || bi.isCube)));
            cAll = (id != AIR_ID) && bi.fullOpaque;
            if (id != AIR_ID && bi.emissive) { ++emissive; if (!cAny && lampId == AIR_ID) lampId = id; }
        } else {
            cSig = (c & VOX_ANY) ? (int)bi.sig : 0;
            cAny = (c & VOX_ANY) != 0 && (cSig > 0 || decorAllowed);
            cAll = (c & VOX_ALL) != 0;
            const uint32_t lightCount = voxel_emissive(c, bi.water != 0);
            emissive += lightCount;
            if (!cAny && lightCount && lampId == AIR_ID && id != AIR_ID) lampId = id;
        }
        any = any || cAny;
        all = all && cAll;
        if (!cAny) continue;
        const int cExposed = (exposed >> i) & 1;
        const bool solidTop = ((bi.lodFaceSolid >> FACE_UP) & 1) != 0;
        const int cUp = ((exposedUp >> i) & 1) ? ((((i >> 1) & 1) && (cSig > 0 || solidTop)) ? 2 : 1) : 0;
        const int cSignificant = cSig > 0 ? 1 : 0;
        const int cLit = bi.emissive ? 1 : 0;
        int count = 0;
        for (int j = 0; j < 8; ++j) if (voxel_id(v[j]) == id && (childLevel == 0 || (v[j] & VOX_ANY))) ++count;
        const bool better = cUp > bestUp
            || (cUp == bestUp && (cExposed > bestExposed
            || (cExposed == bestExposed && (cSignificant > bestSignificant
            || (cSignificant == bestSignificant && (count > bestCount
            || (count == bestCount && (cLit < bestLit
            || (cLit == bestLit && (cSig > bestSig
            || (cSig == bestSig && id < bestId)))))))))));
        if (better) { bestUp = cUp; bestExposed = cExposed; bestSignificant = cSignificant; bestCount = count; bestLit = cLit; bestSig = cSig; bestId = id; }
    }
    if (emissive > VOX_EMIT_MAX) emissive = VOX_EMIT_MAX;
    if (!any) {
        if (lampId != AIR_ID) return make_voxel(lampId, true, false, occ, emissive);
        return make_voxel(AIR_ID, false, false, occ, emissive);
    }
    if (reg.info(bestId).water) {
        // Highest top over all water states, not the parent roof.
        uint32_t height = 0;
        for (int i = 0; i < 8; ++i) {
            if (!reg.info(voxel_id(v[i])).water || (childLevel > 0 && (v[i] & VOX_ANY) == 0)) continue;
            const uint32_t top = (((i >> 1) & 1u) << childLevel) + voxel_water_height(v[i], childLevel);
            height = std::max(height, top);
        }
        return make_voxel(bestId, true, all, occ, height);
    }
    return make_voxel(bestId, true, all, occ, emissive);
}

void VoxelStore::build_lod(const BlockRegistry& reg, int levelCount, planet::WorkerPool* pool, int decorMaxLevel) {
    if (levelCount < 1) levelCount = 1;
    if (levelCount > MAX_LOD_LEVELS) levelCount = MAX_LOD_LEVELS;
    m_decorMaxLevel = decorMaxLevel;
    m_levels.resize(1);
    for (int L = 1; L < levelCount; ++L) {
        m_levels.emplace_back(std::make_unique<ColumnMap>());
        std::vector<uint64_t> parents;
        parents.reserve(m_levels[L - 1]->size() / 2 + 1);
        for (const auto& kv : *m_levels[L - 1])
            parents.push_back(pack_xz(floor_shift(unpack_x(kv.first), 1), floor_shift(unpack_z(kv.first), 1)));
        std::sort(parents.begin(), parents.end());
        parents.erase(std::unique(parents.begin(), parents.end()), parents.end());
        const int minSy = min_section_y(L), maxSy = max_section_y(L);
        for (uint64_t k : parents) {
            Column& col = (*m_levels[L])[k];
            col.sections.resize((size_t)(maxSy - minSy + 1));
        }
        auto body = [&](uint32_t i) {
            const uint64_t k = parents[i];
            const int sx = unpack_x(k), sz = unpack_z(k);
            const int cl = L - 1;
            constexpr int W = 34;
            std::vector<Voxel> win((size_t)W * W * W);
            std::vector<Voxel> vals(SECTION_VOXELS);
            auto at = [&](int x, int y, int z) -> Voxel {
                return win[((size_t)(z + 1) * W + (size_t)(y + 1)) * W + (size_t)(x + 1)];
            };
            for (int sy = minSy; sy <= maxSy; ++sy) {
                bool anyChild = false;
                for (int c = 0; c < 8 && !anyChild; ++c)
                    anyChild = section(cl, sx * 2 + (c & 1), sy * 2 + ((c >> 1) & 1), sz * 2 + ((c >> 2) & 1)) != nullptr;
                if (!anyChild) continue;
                fill_window(cl, sx * 32 - 1, sy * 32 - 1, sz * 32 - 1, W, win.data());
                for (int y = 0; y < 16; ++y)
                for (int z = 0; z < 16; ++z)
                for (int x = 0; x < 16; ++x) {
                    Voxel v[8];
                    int kk = 0;
                    bool uniform = true;
                    for (int dz = 0; dz < 2; ++dz)
                    for (int dy = 0; dy < 2; ++dy)
                    for (int dx = 0; dx < 2; ++dx) {
                        v[kk] = at(x * 2 + dx, y * 2 + dy, z * 2 + dz);
                        uniform = uniform && (v[kk] == v[0]);
                        ++kk;
                    }
                    uint8_t exposed = 0xFF, exposedUp = 0xFF;
                    if (!uniform) {
                        exposed = 0; exposedUp = 0;
                        kk = 0;
                        for (int dz = 0; dz < 2; ++dz)
                        for (int dy = 0; dy < 2; ++dy)
                        for (int dx = 0; dx < 2; ++dx, ++kk) {
                            if (voxel_id(v[kk]) == AIR_ID) continue;
                            const int cx = x * 2 + dx, cy = y * 2 + dy, cz = z * 2 + dz;
                            const bool up = !occluding(at(cx, cy + 1, cz), cl, reg);
                            bool open = up;
                            for (int f = 0; f < 6 && !open; ++f)
                                open = !occluding(at(cx + FACE_DIR[f][0], cy + FACE_DIR[f][1], cz + FACE_DIR[f][2]), cl, reg);
                            if (open) exposed |= (uint8_t)(1u << kk);
                            if (up) exposedUp |= (uint8_t)(1u << kk);
                        }
                    }
                    vals[section_index(x, y, z)] = downsample(reg, cl, v, exposed, exposedUp);
                }
                Column& col = m_levels[L]->find(k)->second;
                Section sec = Section::from_values(vals.data());
                col.sections[(size_t)(sy - minSy)] = sec.all_air() ? nullptr : std::make_unique<Section>(std::move(sec));
            }
        };
        if (pool && parents.size() > 64) pool->parallel_for((uint32_t)parents.size(), body);
        else for (uint32_t i = 0; i < (uint32_t)parents.size(); ++i) body(i);
    }
}

void VoxelStore::set_block(int x, int y, int z, BlockId id, const BlockRegistry& reg, std::vector<uint64_t>& stale) {
    std::unique_lock<std::shared_mutex> lk(m_lock);
    stale.clear();
    if (m_levels.empty()) m_levels.emplace_back(std::make_unique<ColumnMap>());
    const int sy0 = floor_shift(y, 4);
    if (sy0 < m_minSy0 || sy0 > m_maxSy0) return;
    {
        Section& s = ensure_section(0, floor_shift(x, 4), sy0, floor_shift(z, 4));
        s.set(section_index(x & 15, y & 15, z & 15), (Voxel)id);
    }
    for (int L = 1; L < (int)m_levels.size(); ++L) {
        const int px = floor_shift(x, L), py = floor_shift(y, L), pz = floor_shift(z, L);
        const int cl = L - 1;
        Voxel v[8];
        uint8_t exposed = 0, exposedUp = 0;
        int k = 0;
        for (int dz = 0; dz < 2; ++dz)
        for (int dy = 0; dy < 2; ++dy)
        for (int dx = 0; dx < 2; ++dx, ++k) {
            const int cx = px * 2 + dx, cy = py * 2 + dy, cz = pz * 2 + dz;
            v[k] = get(cl, cx, cy, cz);
            const bool up = !occluding(get(cl, cx, cy + 1, cz), cl, reg);
            bool open = up;
            for (int f = 0; f < 6 && !open; ++f)
                open = !occluding(get(cl, cx + FACE_DIR[f][0], cy + FACE_DIR[f][1], cz + FACE_DIR[f][2]), cl, reg);
            if (open) exposed |= (uint8_t)(1u << k);
            if (up) exposedUp |= (uint8_t)(1u << k);
        }
        const Voxel nv = downsample(reg, cl, v, exposed, exposedUp);
        Section& s = ensure_section(L, floor_shift(px, 4), floor_shift(py, 4), floor_shift(pz, 4));
        s.set(section_index(px & 15, py & 15, pz & 15), nv);
    }
    for (int L = 0; L < (int)m_levels.size(); ++L) {
        const int vx = floor_shift(x, L), vy = floor_shift(y, L), vz = floor_shift(z, L);
        const NodeKey k{ (uint8_t)L, floor_shift(vx, CHUNK_SHIFT), floor_shift(vy, CHUNK_SHIFT), floor_shift(vz, CHUNK_SHIFT) };
        stale.push_back(pack_node(k));
        const int lx = vx & (CHUNK_SIZE - 1), ly = vy & (CHUNK_SIZE - 1), lz = vz & (CHUNK_SIZE - 1);
        if (lx == 0)              stale.push_back(pack_node(NodeKey{ k.level, k.x - 1, k.y, k.z }));
        if (lx == CHUNK_SIZE - 1) stale.push_back(pack_node(NodeKey{ k.level, k.x + 1, k.y, k.z }));
        if (ly == 0)              stale.push_back(pack_node(NodeKey{ k.level, k.x, k.y - 1, k.z }));
        if (ly == CHUNK_SIZE - 1) stale.push_back(pack_node(NodeKey{ k.level, k.x, k.y + 1, k.z }));
        if (lz == 0)              stale.push_back(pack_node(NodeKey{ k.level, k.x, k.y, k.z - 1 }));
        if (lz == CHUNK_SIZE - 1) stale.push_back(pack_node(NodeKey{ k.level, k.x, k.y, k.z + 1 }));
    }
}

VoxelStore::Stats VoxelStore::stats() const {
    Stats st;
    for (const auto& mp : m_levels) { const ColumnMap& m = *mp;
        st.columns += m.size();
        for (const auto& kv : m) {
            for (const auto& s : kv.second.sections) {
                if (!s) continue;
                ++st.sections;
                if (s->is_uniform()) ++st.uniform;
                st.bytes += s->memory_bytes();
            }
            st.bytes += kv.second.sections.capacity() * sizeof(std::unique_ptr<Section>) + 48;
        }
    }
    return st;
}

}
