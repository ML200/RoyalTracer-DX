#pragma once

#include <cstdint>
#include <functional>
#include <vector>
#include "mc_types.h"
#include "mc_models.h"

namespace mc {

struct BakeTexture { const uint8_t* rgba = nullptr; int width = 0; int height = 0; size_t pitch = 0; };
using BakeTextureLookup = std::function<BakeTexture(const RawQuad&)>;

struct BakedFace {
    std::vector<uint8_t> rgba;
    int   size = 0;
    float coverage = 0.0f;
    float minDepth = 1.0f;
    bool  empty() const { return coverage <= 0.0f; }
};

// Rasterizes model quads into six coarse face textures.
void bake_block_faces(const std::vector<RawQuad>& quads, int size, const BakeTextureLookup& lookup, BakedFace out[6]);

// Fills transparent texels from wrapped opaque neighbors.
double fill_holes(std::vector<uint8_t>& rgba, int w, int h);

}
