#pragma once

#include <cstdint>
#include <vector>
#include "mc_types.h"

namespace mc {

class Section {
public:
    static Section uniform(Voxel v) {
        Section s;
        s.m_palette.push_back(v);
        return s;
    }
    static Section from_values(const Voxel* v);

    bool  is_uniform() const { return m_bits == 0; }
    Voxel uniform_value() const { return m_palette[0]; }
    bool  all_air() const { return m_bits == 0 && m_palette[0] == 0; }
    uint8_t bits() const { return m_bits; }
    const std::vector<Voxel>& palette() const { return m_palette; }

    Voxel get(uint32_t i) const {
        if (m_bits == 0) return m_palette[0];
        const uint32_t per   = 32u / m_bits;
        const uint32_t word  = m_packed[i / per];
        const uint32_t shift = (i % per) * m_bits;
        return m_palette[(word >> shift) & ((1u << m_bits) - 1u)];
    }
    Voxel get(int x, int y, int z) const { return get(section_index(x, y, z)); }

    // Grows the palette when the value is new.
    void set(uint32_t i, Voxel v);

    void unpack(Voxel* out) const;

    size_t memory_bytes() const {
        return sizeof(Section) + m_palette.capacity() * sizeof(Voxel) + m_packed.capacity() * sizeof(uint32_t);
    }

private:
    static uint8_t width_for(size_t paletteSize) {
        if (paletteSize <= 1)   return 0;
        if (paletteSize <= 2)   return 1;
        if (paletteSize <= 4)   return 2;
        if (paletteSize <= 16)  return 4;
        if (paletteSize <= 256) return 8;
        return 16;
    }
    void repack(uint8_t newBits);
    uint32_t raw_index(uint32_t i) const {
        const uint32_t per = 32u / m_bits;
        return (m_packed[i / per] >> ((i % per) * m_bits)) & ((1u << m_bits) - 1u);
    }
    void write_index(uint32_t i, uint32_t idx) {
        const uint32_t per   = 32u / m_bits;
        const uint32_t shift = (i % per) * m_bits;
        uint32_t& w = m_packed[i / per];
        w = (w & ~(((1u << m_bits) - 1u) << shift)) | (idx << shift);
    }

    std::vector<Voxel>    m_palette;
    std::vector<uint32_t> m_packed;
    uint8_t               m_bits = 0;
};

inline Section Section::from_values(const Voxel* v) {
    Section s;
    std::vector<uint32_t> idx(SECTION_VOXELS);
    s.m_palette.reserve(16);
    Voxel last = v[0];
    uint32_t lastIdx = 0;
    s.m_palette.push_back(last);
    for (uint32_t i = 0; i < (uint32_t)SECTION_VOXELS; ++i) {
        const Voxel val = v[i];
        if (val == last) { idx[i] = lastIdx; continue; }
        uint32_t k = 0;
        const uint32_t n = (uint32_t)s.m_palette.size();
        for (; k < n; ++k) if (s.m_palette[k] == val) break;
        if (k == n) s.m_palette.push_back(val);
        idx[i] = k; last = val; lastIdx = k;
    }
    s.m_bits = width_for(s.m_palette.size());
    if (s.m_bits == 0) return s;
    const uint32_t per = 32u / s.m_bits;
    s.m_packed.assign((SECTION_VOXELS + per - 1) / per, 0u);
    for (uint32_t i = 0; i < (uint32_t)SECTION_VOXELS; ++i) s.write_index(i, idx[i]);
    return s;
}

inline void Section::repack(uint8_t newBits) {
    const uint32_t per = 32u / newBits;
    std::vector<uint32_t> np((SECTION_VOXELS + per - 1) / per, 0u);
    if (m_bits != 0) {
        for (uint32_t i = 0; i < (uint32_t)SECTION_VOXELS; ++i) {
            const uint32_t v = raw_index(i);
            np[i / per] |= v << ((i % per) * newBits);
        }
    }
    m_packed = std::move(np);
    m_bits = newBits;
}

inline void Section::set(uint32_t i, Voxel v) {
    uint32_t k = 0;
    const uint32_t n = (uint32_t)m_palette.size();
    for (; k < n; ++k) if (m_palette[k] == v) break;
    if (k == n) {
        m_palette.push_back(v);
        const uint8_t need = width_for(m_palette.size());
        if (need != m_bits) repack(need);
    }
    if (m_bits == 0) return;
    write_index(i, k);
}

inline void Section::unpack(Voxel* out) const {
    if (m_bits == 0) {
        const Voxel v = m_palette[0];
        for (int i = 0; i < SECTION_VOXELS; ++i) out[i] = v;
        return;
    }
    for (uint32_t i = 0; i < (uint32_t)SECTION_VOXELS; ++i) out[i] = m_palette[raw_index(i)];
}

}
