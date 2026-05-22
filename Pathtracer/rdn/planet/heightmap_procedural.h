#pragma once
//====================================
//PLANET - PROCEDURAL HEIGHTMAP
//====================================
//Placeholder elevation source for the Phase 1-5 test target: a cheap analytic
//sin/cos bump pattern. Stateless during sampling, so trivially thread-safe and
//lock-free. A real heightmap (sparse tiled resource + CPU mirror) replaces this
//later behind the same IHeightmapSource interface.

#include "heightmap_source.h"

namespace planet {

class HeightmapProcedural : public IHeightmapSource {
public:
    //displacement = sin(dir.x * frequency) * cos(dir.z * frequency) * amplitude
    //amplitude is in the same units as the planet radius; tune to the test planet.
    float amplitude = 0.001f;
    float frequency = 8.0f;

    float sample(const DVec3& dir_normalized, uint8_t lod) const override;
};

} // namespace planet
