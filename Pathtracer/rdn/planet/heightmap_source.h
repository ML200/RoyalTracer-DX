#pragma once

#include <cstdint>
#include "coordinate_system.h"

namespace planet {

class IHeightmapSource {
public:
    virtual ~IHeightmapSource() = default;

    // Returns elevation in metres along a normalized planet direction.
    virtual float sample(const DVec3& dir_normalized, uint8_t lod) const = 0;

    // Fills an n-by-n grid using the scalar sampling contract.
    virtual void sample_grid(const DVec3& corner, const DVec3& du, const DVec3& dv,
                             uint32_t n, uint8_t lod, float* out) const;
};

}
