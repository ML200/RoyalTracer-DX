#pragma once
//====================================
//PLANET - HEIGHTMAP SOURCE
//====================================
//Abstract terrain-elevation source. The tessellator queries it for the surface
//displacement at a direction on the unit sphere. Implementations MUST be
//thread-safe and lock-free: many worker threads sample concurrently.

#include <cstdint>
#include "coordinate_system.h"   // DVec3

namespace planet {

class IHeightmapSource {
public:
    virtual ~IHeightmapSource() = default;

    //surface displacement (metres) along a unit direction, at a given LOD.
    virtual float sample(const DVec3& dir_normalized, uint8_t lod) const = 0;

    //fill terrain[0 .. n*n) with samples over an n x n grid of directions spanning
    //the patch  normalize(corner + s*du + t*dv)  for s,t in [0,1]. The default
    //loops sample(); override for a faster batched path. Must stay thread-safe.
    virtual void sample_grid(const DVec3& corner, const DVec3& du, const DVec3& dv,
                             uint32_t n, uint8_t lod, float* out) const;
};

} // namespace planet
