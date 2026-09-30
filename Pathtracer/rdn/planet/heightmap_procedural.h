#pragma once

#include "heightmap_source.h"

namespace planet {

class HeightmapProcedural : public IHeightmapSource {
public:
    float amplitude = 0.001f;
    float frequency = 8.0f;

    float sample(const DVec3& dir_normalized, uint8_t lod) const override;
};

}
