#include "tessellator.h"
#include "cube_sphere.h"
#include "../../include/procedural_terrain.h"
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace planet {

// Encodes a unit normal into two signed 16-bit octahedral components.
uint32_t oct_encode(const Vec3f& n) {
    const float inv = 1.0f / (std::fabs(n.x) + std::fabs(n.y) + std::fabs(n.z) + 1e-20f);
    float ox = n.x * inv;
    float oy = n.y * inv;
    if (n.z < 0.0f) {
        const float x = (1.0f - std::fabs(oy)) * (ox >= 0.0f ? 1.0f : -1.0f);
        const float y = (1.0f - std::fabs(ox)) * (oy >= 0.0f ? 1.0f : -1.0f);
        ox = x; oy = y;
    }
    auto q = [](float v) -> uint32_t {
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

uint16_t float_to_half(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const int32_t  exp  = (int32_t)((x >> 23) & 0xFFu) - 127 + 15;
    const uint32_t mant = x & 0x7FFFFFu;
    if (exp <= 0)    return (uint16_t)sign;
    if (exp >= 0x1F) return (uint16_t)(sign | 0x7C00u);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}

namespace {
// Evaluates displaced terrain in the chunk's local coordinate frame.
inline DVec3 surface_point(uint8_t face, double s, double t,
                           const PlanetGeometry& planet,
                           const IHeightmapSource& hm, uint8_t lod,
                           double footprintM, DVec3& dirOut, DVec3& noiseGradOut) {
    const DVec3  dir    = cube_to_sphere_dir(face, s, t);
    const double h_bake = hm.sample(dir, lod);
    const double h_noise = pt_fbm<double>(dir * planet.radius, footprintM, noiseGradOut);
    dirOut = dir;
    return planet.center + dir * (planet.radius + h_bake + h_noise);
}

inline DVec3 cross3(const DVec3& a, const DVec3& b) {
    return DVec3{ a.y * b.z - a.z * b.y,
                 a.z * b.x - a.x * b.z,
                 a.x * b.y - a.y * b.x };
}

inline void face_equiangular_uv01(uint8_t face, const DVec3& dir,
                                  float& u01, float& v01) {
    const double ax = std::fabs(dir.x);
    const double ay = std::fabs(dir.y);
    const double az = std::fabs(dir.z);
    double ut = 0.0, vt = 0.0;
    switch (face) {
        case 0: ut = -dir.z / ax; vt = -dir.y / ax; break;
        case 1: ut =  dir.z / ax; vt = -dir.y / ax; break;
        case 2: ut =  dir.x / ay; vt =  dir.z / ay; break;
        case 3: ut =  dir.x / ay; vt = -dir.z / ay; break;
        case 4: ut =  dir.x / az; vt = -dir.y / az; break;
        default:ut = -dir.x / az; vt = -dir.y / az; break;
    }
    const double k = 4.0 / 3.14159265358979323846;
    u01 = (float)(std::atan(ut) * k * 0.5 + 0.5);
    v01 = (float)(std::atan(vt) * k * 0.5 + 0.5);
}

inline Vec3f terrain_vertex_normal(const DVec3& dir, const IHeightmapSource& hm,
                                   uint8_t lod, double R, const DVec3& noiseGrad) {
    const DVec3  up    = (std::fabs(dir.y) < 0.99) ? DVec3{ 0,1,0 } : DVec3{ 1,0,0 };
    const DVec3  tA    = normalize(cross3(dir, up));
    const DVec3  tB    = cross3(dir, tA);
    const double e     = 1.0e-3;
    const double inv2e = 1.0 / (2.0 * e);
    const double hC    = hm.sample(dir, lod);
    const double dhA   = hm.sample(normalize(dir + tA * e), lod)
                       - hm.sample(normalize(dir - tA * e), lod);
    const double dhB   = hm.sample(normalize(dir + tB * e), lod)
                       - hm.sample(normalize(dir - tB * e), lod);
    const double dhA_noise = R * dot(noiseGrad, tA);
    const double dhB_noise = R * dot(noiseGrad, tB);
    DVec3 n = dir * (R + hC)
            - tA * (dhA * inv2e + dhA_noise)
            - tB * (dhB * inv2e + dhB_noise);
    n = normalize(n);
    return Vec3f{ (float)n.x, (float)n.y, (float)n.z };
}

inline uint32_t pack_normal_int16(const Vec3f& n) {
    const float inv = 1.0f / (std::fabs(n.x) + std::fabs(n.y) + std::fabs(n.z) + 1e-20f);
    float ox = n.x * inv;
    float oy = n.y * inv;
    if (n.z < 0.0f) {
        const float x = (1.0f - std::fabs(oy)) * (ox >= 0.0f ? 1.0f : -1.0f);
        const float y = (1.0f - std::fabs(ox)) * (oy >= 0.0f ? 1.0f : -1.0f);
        ox = x; oy = y;
    }
    auto q = [](float v) -> uint32_t {
        v = v < -1.0f ? -1.0f : (v > 1.0f ? 1.0f : v);
        return (uint32_t)(uint16_t)(int16_t)std::lround(v * 32767.0f);
    };
    return q(ox) | (q(oy) << 16);
}
}

// Writes a stitched grid directly into caller-provided mesh buffers.
TessResult tessellate_chunk(const TessJob& job, const IHeightmapSource& heightmap) {
    TessResult res;

    const uint32_t G       = job.grid;
    const uint32_t edge    = G + 1;
    const uint32_t vcount  = edge * edge;
    const uint32_t icount  = G * G * 6;

    if (!job.vertex_dest || !job.index_dest)                            return res;
    if (edge > 65536)                                                   return res;
    if (job.stitch_mask && (G & 1u))                                    return res;
    if ((uint64_t)vcount * CHUNK_VERTEX_STRIDE > job.vertex_capacity)    return res;
    if ((uint64_t)icount * CHUNK_INDEX_STRIDE  > job.index_capacity)     return res;

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

    {
        static std::atomic<int> s_dbg_count{0};
        if (s_dbg_count.fetch_add(1) < 5) {
            const double cs[5] = { s0, s1, s0, s1, 0.5 * (s0 + s1) };
            const double ct[5] = { t0, t0, t1, t1, 0.5 * (t0 + t1) };
            double hmin = +1e30, hmax = -1e30;
            for (int k = 0; k < 5; ++k) {
                const DVec3 dir = cube_to_sphere_dir(face, cs[k], ct[k]);
                const double h = heightmap.sample(dir, lod);
                if (h < hmin) hmin = h;
                if (h > hmax) hmax = h;
            }
            std::fprintf(stdout,
                "[planet] tess leaf face=%d lod=%u (s=%.3f..%.3f t=%.3f..%.3f): "
                "h range [%.2f, %.2f] m\n",
                (int)face, (unsigned)lod, s0, s1, t0, t1, hmin, hmax);
            std::fflush(stdout);
        }
    }

    const double base_footprint_m =
        pt_chunk_vertex_spacing_m<double>(job.planet.radius, lod);
    const double stitched_footprint_m = base_footprint_m * 2.0;

    const uint8_t corner_mask = (uint8_t)(job.stitch_mask >> 4) & 0x0Fu;

    const DVec3 anchor_delta_fp64 = job.anchor_world - job.scene_origin;
    const DVec3 anchor_delta_fp32 = {
        (double)(float)anchor_delta_fp64.x,
        (double)(float)anchor_delta_fp64.y,
        (double)(float)anchor_delta_fp64.z,
    };
    const DVec3 e = {
        anchor_delta_fp64.x - anchor_delta_fp32.x,
        anchor_delta_fp64.y - anchor_delta_fp32.y,
        anchor_delta_fp64.z - anchor_delta_fp32.z,
    };

    for (uint32_t j = 0; j < edge; ++j) {
        const double t = t0 + dt * double(j);
        const bool on_neg_t_edge = (j == 0u) && (job.stitch_mask & (1u << EDGE_NEG_T));
        const bool on_pos_t_edge = (j == G)  && (job.stitch_mask & (1u << EDGE_POS_T));

        for (uint32_t i = 0; i < edge; ++i) {
            const double s = s0 + ds * double(i);
            const bool on_neg_s_edge = (i == 0u) && (job.stitch_mask & (1u << EDGE_NEG_S));
            const bool on_pos_s_edge = (i == G)  && (job.stitch_mask & (1u << EDGE_POS_S));
            const uint32_t corner_bit = (uint32_t)((i == G) ? 1u : 0u)
                                       | (uint32_t)((j == G) ? 2u : 0u);
            const bool at_a_corner   = (i == 0u || i == G) && (j == 0u || j == G);
            const bool on_corner_stitch = at_a_corner
                                       && (corner_mask & (1u << corner_bit));
            const bool on_stitched   = on_neg_s_edge || on_pos_s_edge
                                    || on_neg_t_edge || on_pos_t_edge
                                    || on_corner_stitch;
            const double footprint_m = on_stitched ? stitched_footprint_m
                                                   : base_footprint_m;

            DVec3 dir, noiseGrad;
            const DVec3 p = surface_point(face, s, t, job.planet, heightmap, lod,
                                          footprint_m, dir, noiseGrad);

            const DVec3 local = p - job.anchor_world;
            const DVec3 corrected = { local.x + e.x, local.y + e.y, local.z + e.z };
            const Vec3f nrm = terrain_vertex_normal(dir, heightmap, lod,
                                                    job.planet.radius, noiseGrad);
            float u01, v01;
            face_equiangular_uv01(face, dir, u01, v01);
            ChunkVertex& v = vout[j * edge + i];
            v.px = float(corrected.x);
            v.py = float(corrected.y);
            v.pz = float(corrected.z);
            v.normal_oct = pack_normal_int16(nrm);
            v.u = float_to_half(u01);
            v.v = float_to_half(v01);
        }
    }

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
        if (job.stitch_mask & (1u << EDGE_NEG_S)) snap_edge(0,        edge);
        if (job.stitch_mask & (1u << EDGE_POS_S)) snap_edge(G,        edge);
        if (job.stitch_mask & (1u << EDGE_NEG_T)) snap_edge(0,        1);
        if (job.stitch_mask & (1u << EDGE_POS_T)) snap_edge(G * edge, 1);
    }

    int32_t* iout = static_cast<int32_t*>(job.index_dest);
    const int32_t base = (int32_t)job.index_vertex_base;
    uint32_t  w = 0;
    for (uint32_t j = 0; j < G; ++j) {
        for (uint32_t i = 0; i < G; ++i) {
            const int32_t v00 = base + (int32_t)( j        * edge + i);
            const int32_t v10 = base + (int32_t)( j        * edge + i + 1);
            const int32_t v01 = base + (int32_t)((j + 1)   * edge + i);
            const int32_t v11 = base + (int32_t)((j + 1)   * edge + i + 1);
            iout[w++] = v00; iout[w++] = v10; iout[w++] = v11;
            iout[w++] = v00; iout[w++] = v11; iout[w++] = v01;
        }
    }

    res.vertex_count = vcount;
    res.index_count  = icount;
    res.ok           = true;
    return res;
}

}
