#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include "mc_types.h"
#include "voxel_section.h"

namespace mc {

struct NbtValue;
class  InternCache;

class RegionFile {
public:
    static bool parse_name(const std::string& fileName, int32_t& rx, int32_t& rz);

    bool open(const std::string& path, std::string* err = nullptr);
    void close() { m_data.clear(); m_data.shrink_to_fit(); }

    bool chunk_present(int lx, int lz) const;
    // Output is decompressed NBT.
    bool read_chunk(int lx, int lz, std::vector<uint8_t>& nbt, std::string* err = nullptr) const;

    int32_t rx = 0, rz = 0;

private:
    std::vector<uint8_t> m_data;
};

struct DecodedSection {
    int32_t y = 0;
    Section blocks; // Palette-compressed 16^3 block states.
};

struct DecodedChunk {
    int32_t cx = 0, cz = 0;
    int32_t dataVersion = 0;
    std::vector<DecodedSection> sections;
};

bool decode_chunk(const NbtValue& root, InternCache& interner, DecodedChunk& out, std::string* err = nullptr);

void unpack_block_states(const int64_t* longs, size_t longCount, uint32_t bitsPerEntry,
                         bool padded, uint32_t* outIndices4096);

struct LevelInfo {
    std::string name;
    int32_t dataVersion = 0;
    int32_t spawnX = 0, spawnY = 64, spawnZ = 0;
    bool    hasSpawn = false;
};
bool read_level_dat(const std::string& path, LevelInfo& out, std::string* err = nullptr);

}
