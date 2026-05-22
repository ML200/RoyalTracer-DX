//====================================
//PLANET - TESSELLATOR
//====================================

#include "tessellator.h"
#include <cmath>
#include <cstring>

namespace planet {

//====================================
//OCTAHEDRAL NORMAL ENCODING (16:16)
//====================================
uint32_t oct_encode(const Vec3f& n) {
    const float inv = 1.0f / (std::fabs(n.x) + std::fabs(n.y) + std::fabs(n.z) + 1e-20f);
    float ox = n.x * inv;
    float oy = n.y * inv;
    if (n.z < 0.0f) {
        const float x = (1.0f - std::fabs(oy)) * (ox >= 0.0f ? 1.0f : -1.0f);
        const float y = (1.0f - std::fabs(ox)) * (oy >= 0.0f ? 1.0f : -1.0f);
        ox = x; oy = y;
    }
    auto q = [](float v) -> uint32_t {                 // [-1,1] -> [0,65535]
        v = v * 0.5f + 0.5f;
        v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
        return (uint32_t)(v * 65535.0f + 0.5f);
    };
    return q(ox) | (q(oy) << 16);
}

Vec3f oct_decode(uint32_t e) {
    auto dq = [](uint32_t u) -> float { return float(u) / 65535.0f * 2.0f - 1.0f; };
    float ox = dq(e & 0xFFFFu);
    float oy = dq((e >> 16) & 0xFFFFu);
    const float oz = 1.0f - (std::fabs(ox) + std::fabs(oy));
    if (oz < 0.0f) {
        const float x = (1.0f - std::fabs(oy)) * (ox >= 0.0f ? 1.0f : -1.0f);
        const float y = (1.0f - std::fabs(ox)) * (oy >= 0.0f ? 1.0f : -1.0f);
        ox = x; oy = y;
    }
    return normalize(Vec3f{ ox, oy, oz });
}

//====================================
//HALF FLOAT (truncating - adequate for uv)
//====================================
uint16_t float_to_half(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const int32_t  exp  = (int32_t)((x >> 23) & 0xFFu) - 127 + 15;
    const uint32_t mant = x & 0x7FFFFFu;
    if (exp <= 0)    return (uint16_t)sign;                       // underflow -> +/-0
    if (exp >= 0x1F) return (uint16_t)(sign | 0x7C00u);           // overflow  -> +/-inf
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}

//====================================
//TESSELLATE
//====================================
namespace {
//surface point for a cube-face coordinate (s,t): cube->sphere direction,
//displaced outward by the heightmap.
inline DVec3 surface_point(uint8_t face, double s, double t,
                           const PlanetGeometry& planet,
                           const IHeightmapSource& hm, uint8_t lod) {
    const DVec3  dir = cube_to_sphere_dir(face, s, t);
    const double h   = hm.sample(dir, lod);
    return planet.center + dir * (planet.radius + h);
}
} // namespace

TessResult tessellate_chunk(const TessJob& job, const IHeightmapSource& heightmap) {
    TessResult res;

    const uint32_t G       = job.grid;             // quads per edge
    const uint32_t edge    = G + 1;                // vertices per edge
    const uint32_t vcount  = edge * edge;
    const uint32_t icount  = G * G * 6;            // 2 triangles * 3 indices per quad

    if (!job.vertex_dest || !job.index_dest)                            return res;
    if (edge > 65536)                                                   return res;  // uint16 range
    if (job.stitch_mask && (G & 1u))                                    return res;  // stitch needs even G
    if ((uint64_t)vcount * CHUNK_VERTEX_STRIDE > job.vertex_capacity)    return res;
    if ((uint64_t)icount * CHUNK_INDEX_STRIDE  > job.index_capacity)     return res;

    //node footprint in face coordinates [-1,1], split into 2^lod cells.
    const double cells = double(1u << job.node.lod);
    const double s0 = -1.0 + 2.0 *  double(job.node.x)        / cells;
    const double s1 = -1.0 + 2.0 * (double(job.node.x) + 1.0) / cells;
    const double t0 = -1.0 + 2.0 *  double(job.node.y)        / cells;
    const double t1 = -1.0 + 2.0 * (double(job.node.y) + 1.0) / cells;
    const double ds = (s1 - s0) / double(G);
    const double dt = (t1 - t0) / double(G);
    const uint8_t lod  = job.node.lod;
    const uint8_t face = job.node.face;

    ChunkVertex* vout = static_cast<ChunkVertex*>(job.vertex_dest);

    //--- vertices ---
    for (uint32_t j = 0; j < edge; ++j) {
        const double t = t0 + dt * double(j);
        for (uint32_t i = 0; i < edge; ++i) {
            const double s = s0 + ds * double(i);

            const DVec3 p = surface_point(face, s, t, job.planet, heightmap, lod);

            const DVec3 local = p - job.anchor_world;
            ChunkVertex& v = vout[j * edge + i];
            v.px = float(local.x);
            v.py = float(local.y);
            v.pz = float(local.z);
            v.normal_oct = 0;
            v.u = float_to_half(float(i) / float(G));
            v.v = float_to_half(float(j) / float(G));
        }
    }

    //--- seam stitch: on each edge facing a COARSER neighbour, snap that edge's
    //odd boundary vertices onto the chord between their even neighbours. The
    //even vertices already coincide with the coarse neighbour's vertices (same
    //(s,t) -> same surface_point, true across a cube-face seam too), so the
    //snapped fine edge becomes the coarse neighbour's polyline exactly - a
    //crack-free 1:2 boundary. uv is left as-is (an odd vertex's uv is already
    //the midpoint); the normal too - terrain shading re-derives it from the hit
    //position, so vertex normals are unused at shade time.
    if (job.stitch_mask) {
        auto snap_edge = [vout, G](uint32_t base, uint32_t step) {
            for (uint32_t k = 1; k < G; k += 2) {
                ChunkVertex&       v = vout[base + k * step];
                const ChunkVertex& a = vout[base + (k - 1) * step];
                const ChunkVertex& b = vout[base + (k + 1) * step];
                v.px = 0.5f * (a.px + b.px);
                v.py = 0.5f * (a.py + b.py);
                v.pz = 0.5f * (a.pz + b.pz);
            }
        };
        if (job.stitch_mask & (1u << EDGE_NEG_S)) snap_edge(0,        edge);  // column i=0
        if (job.stitch_mask & (1u << EDGE_POS_S)) snap_edge(G,        edge);  // column i=G
        if (job.stitch_mask & (1u << EDGE_NEG_T)) snap_edge(0,        1);     // row j=0
        if (job.stitch_mask & (1u << EDGE_POS_T)) snap_edge(G * edge, 1);     // row j=G
    }

    //--- indices: 2 triangles per quad, outward-CCW winding ---
    uint16_t* iout = static_cast<uint16_t*>(job.index_dest);
    uint32_t  w = 0;
    for (uint32_t j = 0; j < G; ++j) {
        for (uint32_t i = 0; i < G; ++i) {
            const uint16_t v00 = (uint16_t)( j        * edge + i);
            const uint16_t v10 = (uint16_t)( j        * edge + i + 1);
            const uint16_t v01 = (uint16_t)((j + 1)   * edge + i);
            const uint16_t v11 = (uint16_t)((j + 1)   * edge + i + 1);
            iout[w++] = v00; iout[w++] = v10; iout[w++] = v11;
            iout[w++] = v00; iout[w++] = v11; iout[w++] = v01;
        }
    }

    res.vertex_count = vcount;
    res.index_count  = icount;
    res.ok           = true;
    return res;
}

} // namespace planet
