#pragma once

#include <cstdint>
#include <memory>
#include <shared_mutex>
#include <unordered_map>
#include <vector>
#include "mc_types.h"
#include "voxel_section.h"

namespace planet { class WorkerPool; }

namespace mc {

class BlockRegistry;

class VoxelStore {
public:
    // Stores palette sections at level zero and derived LOD levels.
    struct Column {
        std::vector<std::unique_ptr<Section>> sections;
        Column() = default;
        Column(Column&&) noexcept = default;
        Column& operator=(Column&&) noexcept = default;
        Column(const Column&) = delete;
        Column& operator=(const Column&) = delete;
    };
    struct Stats {
        size_t sections    = 0;
        size_t uniform     = 0;
        size_t bytes       = 0;
        size_t columns     = 0;
    };

    void configure(int minSectionY, int maxSectionY);

    int min_section_y(int level) const { return floor_shift(m_minSy0, level); }
    int max_section_y(int level) const { return floor_shift(m_maxSy0, level); }
    int levels() const { return (int)m_levels.size(); }

    void put_section(int level, int sx, int sy, int sz, Section&& s);
    const Section* section(int level, int sx, int sy, int sz) const;

    Voxel get(int level, int x, int y, int z) const;

    void fill_window(int level, int x0, int y0, int z0, int n, Voxel* out) const;

    bool chunk_occupied(const NodeKey& k) const;
    std::vector<uint64_t> occupied_chunk_keys() const;

    bool column_bounds(int level, int& minSx, int& maxSx, int& minSz, int& maxSz) const;
    size_t column_count(int level) const;

    // Builds coarser representatives from eight child voxels at a time.
    void build_lod(const BlockRegistry& reg, int levelCount, planet::WorkerPool* pool, int decorMaxLevel = 3);

    Voxel downsample(const BlockRegistry& reg, int childLevel, const Voxel v[8], uint8_t exposed, uint8_t exposedUp) const;

    void set_block(int x, int y, int z, BlockId id, const BlockRegistry& reg, std::vector<uint64_t>& staleChunks);

    Stats stats() const;
    std::shared_mutex& lock() const { return m_lock; }

private:
    using ColumnMap = std::unordered_map<uint64_t, Column, U64Hash>;
    Section*       section_mut(int level, int sx, int sy, int sz);
    Section&       ensure_section(int level, int sx, int sy, int sz);

    std::vector<std::unique_ptr<ColumnMap>> m_levels;
    int m_minSy0 = 0, m_maxSy0 = 15;
    int m_decorMaxLevel = 3;
    mutable std::shared_mutex m_lock;
};

}
