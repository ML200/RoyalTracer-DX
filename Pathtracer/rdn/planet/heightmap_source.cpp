//====================================
//PLANET - HEIGHTMAP SOURCE
//====================================

#include "heightmap_source.h"

namespace planet {

//default batched path: just loop the scalar sample(). A real (expensive)
//source overrides this to amortise tile fetches across the grid.
void IHeightmapSource::sample_grid(const DVec3& corner, const DVec3& du, const DVec3& dv,
                                   uint32_t n, uint8_t lod, float* out) const {
    if (n == 0) return;
    const double inv = (n > 1) ? 1.0 / double(n - 1) : 0.0;
    for (uint32_t j = 0; j < n; ++j) {
        const double t = double(j) * inv;
        for (uint32_t i = 0; i < n; ++i) {
            const double s   = double(i) * inv;
            const DVec3  dir = normalize(corner + du * s + dv * t);
            out[j * n + i]   = sample(dir, lod);
        }
    }
}

} // namespace planet
