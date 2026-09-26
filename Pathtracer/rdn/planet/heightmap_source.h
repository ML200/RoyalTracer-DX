#pragma once

#include <cstdint>
#include "coordinate_system.h"

namespace planet {

class IHeightmapSource {
public:
    virtual ~IHeightmapSource() = default;

    // Elevation in metres.
    virtual float sample(const DVec3& dir_normalized, uint8_t lod) const = 0;
};

}
