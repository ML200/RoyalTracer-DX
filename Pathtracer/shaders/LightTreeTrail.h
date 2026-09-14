#ifndef LIGHT_TREE_TRAIL_H
#define LIGHT_TREE_TRAIL_H

#define LT_TRAIL_MAX_DEPTH 32u

#ifdef __cplusplus
#include <cstdint>
#include <stdexcept>

namespace lt {
using LightTreeTrail = uint64_t;
static_assert(sizeof(LightTreeTrail) == 8);

inline LightTreeTrail AppendLightTreeTrail(LightTreeTrail trail, uint32_t child, uint32_t depth)
{
    if (depth >= LT_TRAIL_MAX_DEPTH || child >= 4u)
        throw std::logic_error("Light-tree trail capacity exceeded");
    return trail | (LightTreeTrail(child) << (2u * depth));
}

inline bool LightTreeNeedsBalancedSplit(uint32_t count, uint32_t depth)
{
    return depth >= LT_TRAIL_MAX_DEPTH ||
        uint64_t(count) > (uint64_t{1} << (LT_TRAIL_MAX_DEPTH - depth - 1u));
}
}
#else
uint LT_TrailChild(uint2 trail, uint depth)
{
    const uint word = depth < 16u ? trail.x : trail.y;
    return (word >> (2u * (depth & 15u))) & 3u;
}
#endif

#endif
