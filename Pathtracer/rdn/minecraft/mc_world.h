#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include "mc_types.h"
#include "anvil.h"
#include "block_registry.h"
#include "voxel_store.h"
#include "lod_tree.h"

namespace planet { class WorkerPool; }

namespace mc {

struct WorldLoadConfig {
    std::string worldDir;
    int32_t chunkMinX = 1, chunkMaxX = 0, chunkMinZ = 1, chunkMaxZ = 0;
    bool logProgress = true;
};

struct WorldLoadStats {
    uint32_t regionFiles   = 0;
    uint32_t regionsFailed = 0;
    uint32_t chunks        = 0;
    uint32_t chunksFailed  = 0;
    uint32_t sections      = 0;
    uint32_t blockStates   = 0;
    double   loadSeconds   = 0.0;
    double   lodSeconds    = 0.0;
    size_t   storeBytes    = 0;
    int32_t  minChunkX = 0, maxChunkX = 0, minChunkZ = 0, maxChunkZ = 0;
    int32_t  minSectionY = 0, maxSectionY = 0;
};

class World {
public:
    // Loads region data before model and material resolution.
    bool load(const WorldLoadConfig& cfg, planet::WorkerPool* pool, std::string* err = nullptr);

    void build_lod(planet::WorkerPool* pool, int decorMaxLevel = 3);

    BlockRegistry&        registry()       { return m_registry; }
    const BlockRegistry&  registry() const { return m_registry; }
    VoxelStore&           store()          { return m_store; }
    const VoxelStore&     store() const    { return m_store; }
    const LodTree&        lod_tree() const { return m_tree; }
    LodTree&              lod_tree_mut()   { return m_tree; }
    const LevelInfo&      level() const    { return m_level; }
    const WorldLoadStats& stats() const    { return m_stats; }
    int                   lod_levels() const { return m_lodLevels; }

private:
    BlockRegistry  m_registry;
    VoxelStore     m_store;
    LodTree        m_tree;
    LevelInfo      m_level;
    WorldLoadStats m_stats;
    int            m_lodLevels = 1;
};

std::string find_minecraft_jar();

}
