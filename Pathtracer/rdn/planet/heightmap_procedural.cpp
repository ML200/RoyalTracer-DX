#include "heightmap_procedural.h"
#include <cmath>

namespace planet {

float HeightmapProcedural::sample(const DVec3& dir, uint8_t) const {
    const double h = std::sin(dir.x * frequency)
                   * std::cos(dir.z * frequency)
                   * amplitude;
    return float(h);
}

}
