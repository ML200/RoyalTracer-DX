#include "mc_zip.h"
#include "mc_inflate.h"
#include <cstdio>
#include <cstring>

namespace mc {

namespace {
inline uint16_t rd16(const uint8_t* p) { return (uint16_t)(p[0] | (p[1] << 8)); }
inline uint32_t rd32(const uint8_t* p) { return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24); }
inline void wr16(std::vector<uint8_t>& v, uint16_t x) { v.push_back((uint8_t)x); v.push_back((uint8_t)(x >> 8)); }
inline void wr32(std::vector<uint8_t>& v, uint32_t x) { wr16(v, (uint16_t)x); wr16(v, (uint16_t)(x >> 16)); }

uint32_t crc32_of(const uint8_t* p, size_t n) {
    static uint32_t table[256];
    static bool init = false;
    if (!init) {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            table[i] = c;
        }
        init = true;
    }
    uint32_t c = 0xFFFFFFFFu;
    for (size_t i = 0; i < n; ++i) c = table[(c ^ p[i]) & 0xFFu] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}
}

bool ZipArchive::open(const std::string& path, std::string* err) {
    m_path = path;
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
    if (sz <= 0) { std::fclose(f); if (err) *err = "empty archive " + path; return false; }
    m_data.resize((size_t)sz);
    const size_t got = std::fread(m_data.data(), 1, (size_t)sz, f);
    std::fclose(f);
    if (got != (size_t)sz) { if (err) *err = "short read " + path; return false; }
    return index(err);
}

bool ZipArchive::open_memory(std::vector<uint8_t> bytes, std::string* err) {
    m_path = "<memory>";
    m_data = std::move(bytes);
    return index(err);
}

// Indexes central-directory entries without inflating their payloads.
bool ZipArchive::index(std::string* err) {
    m_entries.clear();
    const size_t n = m_data.size();
    if (n < 22) { if (err) *err = "archive too small"; return false; }
    size_t eocd = (size_t)-1;
    const size_t stop = (n > 22 + 65535) ? n - 22 - 65535 : 0;
    for (size_t i = n - 22; ; --i) {
        if (rd32(&m_data[i]) == 0x06054b50u) { eocd = i; break; }
        if (i == stop) break;
    }
    if (eocd == (size_t)-1) { if (err) *err = "no end-of-central-directory record"; return false; }
    const uint16_t count  = rd16(&m_data[eocd + 10]);
    const uint32_t cdSize = rd32(&m_data[eocd + 12]);
    const uint32_t cdOff  = rd32(&m_data[eocd + 16]);
    if ((uint64_t)cdOff + cdSize > n) { if (err) *err = "central directory out of range"; return false; }
    size_t p = cdOff;
    m_entries.reserve(count);
    for (uint32_t k = 0; k < count; ++k) {
        if (p + 46 > n || rd32(&m_data[p]) != 0x02014b50u) { if (err) *err = "bad central directory entry"; return false; }
        Entry e;
        e.method      = rd16(&m_data[p + 10]);
        e.compSize    = rd32(&m_data[p + 20]);
        e.uncompSize  = rd32(&m_data[p + 24]);
        const uint16_t nameLen  = rd16(&m_data[p + 28]);
        const uint16_t extraLen = rd16(&m_data[p + 30]);
        const uint16_t commLen  = rd16(&m_data[p + 32]);
        e.localOffset = rd32(&m_data[p + 42]);
        if (p + 46 + nameLen > n) { if (err) *err = "bad entry name"; return false; }
        std::string name((const char*)&m_data[p + 46], nameLen);
        m_entries.emplace(std::move(name), e);
        p += 46 + nameLen + extraLen + commLen;
    }
    return true;
}

// Reads and decompresses one stored or deflated entry.
bool ZipArchive::read(const std::string& name, std::vector<uint8_t>& out) {
    out.clear();
    const auto it = m_entries.find(name);
    if (it == m_entries.end()) return false;
    const Entry& e = it->second;
    const size_t n = m_data.size();
    const size_t lh = e.localOffset;
    if (lh + 30 > n || rd32(&m_data[lh]) != 0x04034b50u) return false;
    const uint16_t nameLen  = rd16(&m_data[lh + 26]);
    const uint16_t extraLen = rd16(&m_data[lh + 28]);
    const size_t dataOff = lh + 30 + nameLen + extraLen;
    if (dataOff + e.compSize > n) return false;
    const uint8_t* src = &m_data[dataOff];
    if (e.method == 0) {
        out.assign(src, src + e.compSize);
        return true;
    }
    if (e.method == 8)
        return inflate_buffer(src, e.compSize, Compression::RawDeflate, out, e.uncompSize);
    return false;
}

void ZipArchive::list(const std::string& prefix, std::vector<std::string>& out) const {
    for (const auto& kv : m_entries)
        if (kv.first.compare(0, prefix.size(), prefix) == 0) out.push_back(kv.first);
}

std::vector<uint8_t> make_stored_zip(const std::vector<std::pair<std::string, std::string>>& files) {
    std::vector<uint8_t> z;
    struct Rec { std::string name; uint32_t off; uint32_t size; uint32_t crc; };
    std::vector<Rec> recs;
    for (const auto& f : files) {
        Rec r{ f.first, (uint32_t)z.size(), (uint32_t)f.second.size(), crc32_of((const uint8_t*)f.second.data(), f.second.size()) };
        wr32(z, 0x04034b50u); wr16(z, 20); wr16(z, 0); wr16(z, 0); wr16(z, 0); wr16(z, 0);
        wr32(z, r.crc); wr32(z, r.size); wr32(z, r.size); wr16(z, (uint16_t)r.name.size()); wr16(z, 0);
        z.insert(z.end(), r.name.begin(), r.name.end());
        z.insert(z.end(), f.second.begin(), f.second.end());
        recs.push_back(r);
    }
    const uint32_t cdStart = (uint32_t)z.size();
    for (const Rec& r : recs) {
        wr32(z, 0x02014b50u); wr16(z, 20); wr16(z, 20); wr16(z, 0); wr16(z, 0); wr16(z, 0); wr16(z, 0);
        wr32(z, r.crc); wr32(z, r.size); wr32(z, r.size); wr16(z, (uint16_t)r.name.size());
        wr16(z, 0); wr16(z, 0); wr16(z, 0); wr16(z, 0); wr32(z, 0); wr32(z, r.off);
        z.insert(z.end(), r.name.begin(), r.name.end());
    }
    const uint32_t cdSize = (uint32_t)z.size() - cdStart;
    wr32(z, 0x06054b50u); wr16(z, 0); wr16(z, 0); wr16(z, (uint16_t)recs.size()); wr16(z, (uint16_t)recs.size());
    wr32(z, cdSize); wr32(z, cdStart); wr16(z, 0);
    return z;
}

}
