#pragma once

#include <string>
#include <vector>
#include "voxel_streamer.h"

namespace mc {

struct MinecraftWorldConfig {
    // Paths and budgets controlling world loading and streaming.
    std::string worldDir;
    std::string jarPath;
    std::vector<std::string> resourcePacks;
    StreamerConfig streamer;
    int   lodDecorMaxLevel = 3;
    float lodOpaqueCoverage = 0.5f;
    bool  opaqueLeaves  = false;
    bool  opacityMicromaps = true;
    bool  spawnCamera   = true;
    float cameraHeight  = 40.0f;
    bool  pointFilter   = true;
    float flySpeed      = 40.0f;
};

}
