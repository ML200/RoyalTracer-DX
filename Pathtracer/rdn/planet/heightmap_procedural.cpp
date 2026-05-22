//====================================
//PLANET - PROCEDURAL HEIGHTMAP
//====================================

#include "heightmap_procedural.h"
#include <cmath>

namespace planet {

float HeightmapProcedural::sample(const DVec3& dir, uint8_t /*lod*/) const {
    //the Phase 3 placeholder pattern. lod is ignored here - a real source would
    //add lod-dependent detail octaves.
    const double h = std::sin(dir.x * frequency)
                   * std::cos(dir.z * frequency)
                   * amplitude;
    return float(h);
}

} // namespace planet
