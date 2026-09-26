#include "anvil.h"
#include "nbt.h"
#include "block_registry.h"
#include "mc_inflate.h"
#include <cstdio>
#include <cstring>

namespace mc {

bool RegionFile::parse_name(const std::string& fileName, int32_t& rx, int32_t& rz) {
    size_t s = fileName.find_last_of("/\\");
    const std::string base = (s == std::string::npos) ? fileName : fileName.substr(s + 1);
    int x = 0, z = 0;
    char tail[8] = {};
    if (std::sscanf(base.c_str(), "r.%d.%d.%7s", &x, &z, tail) != 3) return false;
    if (std::strcmp(tail, "mca") != 0) return false;
    rx = x; rz = z;
    return true;
}

bool RegionFile::open(const std::string& path, std::string* err) {
    m_data.clear();
    FILE* f = nullptr;
#ifdef _WIN32
    fopen_s(&f, path.c_str(), "rb");
#else
    f = std::fopen(path.c_str(), "rb");
#endif
    if (!f) { if (err) *err = "cannot open " + path; return false; }
    std::fseek(f, 0, SEEK_END);
    const long sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (sz < 8192) { std::fclose(f); parse_name(path, rx, rz); return true; }
    m_data.resize((size_t)sz);
    const size_t got = std::fread(m_data.data(), 1, (size_t)sz, f);
    std::fclose(f);
    if (got != (size_t)sz) { m_data.clear(); if (err) *err = "short read: " + path; return false; }
    parse_name(path, rx, rz);
    return true;
}

bool RegionFile::chunk_present(int lx, int lz) const {
    if (m_data.size() < 8192 || lx < 0 || lx > 31 || lz < 0 || lz > 31) return false;
    const size_t h = (size_t)(lx + lz * 32) * 4;
    const uint32_t entry = ((uint32_t)m_data[h] << 24) | ((uint32_t)m_data[h + 1] << 16) | ((uint32_t)m_data[h + 2] << 8) | m_data[h + 3];
    return entry != 0;
}

bool RegionFile::read_chunk(int lx, int lz, std::vector<uint8_t>& nbt, std::string* err) const {
    nbt.clear();
    if (!chunk_present(lx, lz)) { if (err) *err = "chunk absent"; return false; }
    const size_t h = (size_t)(lx + lz * 32) * 4;
    const uint32_t offSectors = ((uint32_t)m_data[h] << 16) | ((uint32_t)m_data[h + 1] << 8) | m_data[h + 2];
    const uint64_t off = (uint64_t)offSectors * 4096ull;
    if (off + 5 > m_data.size()) { if (err) *err = "chunk offset out of range"; return false; }
    const uint8_t* p = m_data.data() + off;
    const uint32_t len = ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | p[3];
    if (len < 1 || off + 4 + len > m_data.size()) { if (err) *err = "chunk length out of range"; return false; }
    const uint8_t kind = p[4];
    const uint8_t* payload = p + 5;
    const size_t payloadLen = len - 1;
    Compression c;
    switch (kind & 0x7F) {
    case 1: c = Compression::Gzip; break;
    case 2: c = Compression::Zlib; break;
    case 3: c = Compression::None; break;
    default: if (err) *err = "unsupported chunk compression"; return false;
    }
    if (kind & 0x80) { if (err) *err = "external .mcc chunks are not supported"; return false; }
    if (!inflate_buffer(payload, payloadLen, c, nbt, payloadLen * 6)) {
        if (err) *err = "chunk inflate failed";
        return false;
    }
    return true;
}

void unpack_block_states(const int64_t* longs, size_t longCount, uint32_t bits,
                         bool padded, uint32_t* out) {
    const uint64_t mask = (bits >= 64) ? ~0ull : ((1ull << bits) - 1ull);
    if (padded) {
        const uint32_t per = 64u / bits;
        for (uint32_t i = 0; i < (uint32_t)SECTION_VOXELS; ++i) {
            const size_t   w  = i / per;
            const uint32_t sh = (i % per) * bits;
            out[i] = (w < longCount) ? (uint32_t)(((uint64_t)longs[w] >> sh) & mask) : 0u;
        }
    } else {
        for (uint32_t i = 0; i < (uint32_t)SECTION_VOXELS; ++i) {
            const uint64_t bit = (uint64_t)i * bits;
            const size_t   w   = (size_t)(bit >> 6);
            const uint32_t sh  = (uint32_t)(bit & 63u);
            if (w >= longCount) { out[i] = 0; continue; }
            uint64_t v = (uint64_t)longs[w] >> sh;
            if (sh + bits > 64 && w + 1 < longCount)
                v |= (uint64_t)longs[w + 1] << (64u - sh);
            out[i] = (uint32_t)(v & mask);
        }
    }
}

namespace {

uint32_t ceil_log2(size_t n) {
    uint32_t b = 0;
    while (((size_t)1 << b) < n) ++b;
    return b;
}

bool decode_section(const NbtValue* paletteList, const NbtValue* states, bool padded,
                    InternCache& interner, Section& out, bool& allAir) {
    allAir = true;
    if (!paletteList || !paletteList->is_list() || paletteList->children.empty()) return true;
    std::vector<BlockId> pal;
    pal.reserve(paletteList->children.size());
    for (const NbtValue& e : paletteList->children) {
        if (!e.is_compound()) { pal.push_back(AIR_ID); continue; }
        pal.push_back(interner.intern(e.get_string("Name"), e.get_compound("Properties")));
    }
    for (BlockId id : pal) if (id != AIR_ID) { allAir = false; break; }
    if (allAir) return true;

    if (pal.size() == 1 || !states || states->longs.empty()) {
        out = Section::uniform((Voxel)pal[0]);
        return true;
    }
    uint32_t bits = ceil_log2(pal.size());
    if (bits < 4) bits = 4;
    static thread_local std::vector<uint32_t> idx;
    static thread_local std::vector<Voxel>    vals;
    idx.resize(SECTION_VOXELS);
    vals.resize(SECTION_VOXELS);
    unpack_block_states(states->longs.data(), states->longs.size(), bits, padded, idx.data());
    const uint32_t n = (uint32_t)pal.size();
    for (uint32_t i = 0; i < (uint32_t)SECTION_VOXELS; ++i) {
        const uint32_t k = idx[i];
        vals[i] = (Voxel)(k < n ? pal[k] : AIR_ID);
    }
    out = Section::from_values(vals.data());
    return true;
}

}

// Handles both legacy and modern section layouts.
bool decode_chunk(const NbtValue& root, InternCache& interner, DecodedChunk& out, std::string* err) {
    out.sections.clear();
    if (!root.is_compound()) { if (err) *err = "root is not a compound"; return false; }
    out.dataVersion = (int32_t)root.get_int("DataVersion", 0);
    const bool padded = out.dataVersion >= 2529;

    const NbtValue* level = root.get_compound("Level");
    const NbtValue* sections = nullptr;
    bool modern = false;
    if (level) {
        out.cx = (int32_t)level->get_int("xPos");
        out.cz = (int32_t)level->get_int("zPos");
        sections = level->get_list("Sections");
    } else {
        out.cx = (int32_t)root.get_int("xPos");
        out.cz = (int32_t)root.get_int("zPos");
        sections = root.get_list("sections");
        modern = true;
    }
    if (!sections) return true;

    for (const NbtValue& sec : sections->children) {
        if (!sec.is_compound()) continue;
        const int32_t y = (int32_t)sec.get_int("Y", 0);
        const NbtValue* pal = nullptr;
        const NbtValue* states = nullptr;
        if (modern) {
            const NbtValue* bs = sec.get_compound("block_states");
            if (!bs) continue;
            pal    = bs->get_list("palette");
            states = bs->find("data");
        } else {
            pal    = sec.get_list("Palette");
            states = sec.find("BlockStates");
            if (!pal) {
                if (sec.find("Blocks")) { if (err) *err = "pre-1.13 chunk format is not supported"; return false; }
                continue;
            }
        }
        if (states && states->type != NbtType::LongArray) states = nullptr;
        DecodedSection ds;
        ds.y = y;
        bool allAir = true;
        if (!decode_section(pal, states, padded, interner, ds.blocks, allAir)) return false;
        if (allAir) continue;
        out.sections.push_back(std::move(ds));
    }
    return true;
}

bool read_level_dat(const std::string& path, LevelInfo& out, std::string* err) {
    FILE* f = nullptr;
#ifdef _WIN32
    fopen_s(&f, path.c_str(), "rb");
#else
    f = std::fopen(path.c_str(), "rb");
#endif
    if (!f) { if (err) *err = "cannot open " + path; return false; }
    std::vector<uint8_t> raw;
    std::fseek(f, 0, SEEK_END);
    const long sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (sz <= 0) { std::fclose(f); if (err) *err = "empty level.dat"; return false; }
    raw.resize((size_t)sz);
    const size_t got = std::fread(raw.data(), 1, (size_t)sz, f);
    std::fclose(f);
    if (got != (size_t)sz) { if (err) *err = "short read"; return false; }

    std::vector<uint8_t> nbt;
    if (raw.size() > 2 && raw[0] == 0x1F && raw[1] == 0x8B) {
        if (!inflate_buffer(raw.data(), raw.size(), Compression::Gzip, nbt)) { if (err) *err = "gzip inflate failed"; return false; }
    } else {
        nbt = raw;
    }
    NbtValue root;
    if (!nbt_parse(nbt.data(), nbt.size(), root, err)) return false;
    const NbtValue* data = root.get_compound("Data");
    if (!data) { if (err) *err = "level.dat has no Data compound"; return false; }
    out.name        = std::string(data->get_string("LevelName"));
    out.dataVersion = (int32_t)data->get_int("DataVersion", 0);
    if (data->find("SpawnX")) {
        out.spawnX = (int32_t)data->get_int("SpawnX");
        out.spawnY = (int32_t)data->get_int("SpawnY", 64);
        out.spawnZ = (int32_t)data->get_int("SpawnZ");
        out.hasSpawn = true;
    }
    return true;
}

}
