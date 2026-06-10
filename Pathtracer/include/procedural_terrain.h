#pragma once
//====================================
//PROCEDURAL TERRAIN NOISE (CPU side)
//====================================
//3D Worley FBM that the CPU tessellator adds to the heightmap displacement
//and the GPU shader evaluates analytically for the shading normal. Same
//math on both sides (byte-identical hash + IEEE-754 float arithmetic) so
//the mesh that shadow rays trace and the normal that the BRDF reads stay
//in lockstep.
//
//Sample domain is 3D Cartesian on the sphere (P = dir * R_planet), so no
//cube-face seams. Output is in metres of relief.
//
//The HLSL twin lives at Pathtracer/shaders/procedural_terrain_v8.hlsli;
//every constant and every arithmetic line below MUST stay in sync with
//that file.

#include "../rdn/planet/coordinate_system.h"
#include <cmath>
#include <cstdint>

namespace planet {

//====================================
//Tuning - mirror in procedural_terrain_v8.hlsli
//====================================
//Top octave period (~2x a mountain). Sets the largest noise feature size.
//Power-of-2 so every octave's period is a power of 2 (lacunarity = 2) and
//every frequency is an exact fp32 exponent - matters for the GPU's
//precision-recovery scheme in procedural_terrain_v8.hlsli, where
//`int_metres * freq` must be exact.
constexpr float    PROCTERRAIN_TOP_PERIOD_M    = 16384.0f;
//Smallest octave period. Mars sandy-plains tuning: raised 1 -> 16 m to cut
//the 1..8 m Worley octaves that made the whole surface read as fractured
//rock. Finest relief is now ~16-64 m gentle hummocks, not gravel. (Mirror
//PT_BOTTOM_PERIOD_M in procedural_terrain_v8.hlsli.)
constexpr float    PROCTERRAIN_BOTTOM_PERIOD_M = 16.0f;
//Top octave amplitude in metres of relief; decays each octave by GAIN.
//Mars sandy-plains tuning: 120 -> 60 m for a flatter, low-relief silhouette.
//(Mirror PT_TOP_AMP_M in procedural_terrain_v8.hlsli.)
constexpr float    PROCTERRAIN_TOP_AMP_M       = 60.0f;
constexpr float    PROCTERRAIN_LACUNARITY      = 2.0f;
//GAIN controls how much amplitude smaller octaves keep. 0.5 is the
//classical "constant amplitude/period" FBM; higher values keep small
//octaves taller relative to their wavelength, producing rock-like sharpness
//instead of smooth lumps. Mars sandy-plains tuning: 0.70 -> 0.55 (near-
//classical) so fine octaves stay gentle rather than reading as cobbles.
//Raise back toward 0.70 for a rockier surface. (Mirror PT_GAIN in
//procedural_terrain_v8.hlsli.)
constexpr float    PROCTERRAIN_GAIN            = 0.55f;
//log2(top / bottom) rounded up: 16384 / 1 ~= 14 -> 15 octaves.
constexpr int      PROCTERRAIN_MAX_OCTAVES     = 15;
//Smooth-min sharpness for Voronoi cell merging. Larger -> crisper crack
//edges between cells; smaller -> rounder cells. Mars sandy-plains tuning:
//48 -> 24 so cell boundaries read as rounded dune-like hummocks instead of
//fractured rock seams. Raise back toward 48 for a rockier, cracked surface.
//Going much higher (>~80) makes the blend band narrower than the Nyquist
//footprint at fine scales, which would alias as the camera approaches.
//(Mirror PT_SMOOTH_K in procedural_terrain_v8.hlsli.)
constexpr float    PROCTERRAIN_SMOOTH_K        = 24.0f;
//Per-planet seed; fixed for now, expose via cbuffer later if needed.
constexpr uint32_t PROCTERRAIN_SEED            = 0xA5C9E1B7u;

//====================================
//Integer hash (PCG-derived). Deterministic on CPU and GPU - identical
//arithmetic, no sin-based shortcuts (those drift between platforms).
//====================================
inline uint32_t pt_hash_u32(uint32_t x) {
    x ^= x >> 17;
    x *= 0xed5ad4bbu;
    x ^= x >> 11;
    x *= 0xac4c1b51u;
    x ^= x >> 15;
    x *= 0x31848babu;
    x ^= x >> 14;
    return x;
}

//Hash a 3D integer cell -> feature point offset in [0, 1)^3.
template <typename T>
inline Vec3<T> pt_cell_feature(int32_t ix, int32_t iy, int32_t iz) {
    uint32_t h = pt_hash_u32(
        pt_hash_u32(pt_hash_u32(static_cast<uint32_t>(ix) + PROCTERRAIN_SEED)
                              + static_cast<uint32_t>(iy))
                              + static_cast<uint32_t>(iz));
    const T inv = T(1) / T(0x01000000);
    Vec3<T> o;
    o.x = T(h & 0xFFFFFFu) * inv;
    h *= 1664525u;
    o.y = T(h & 0xFFFFFFu) * inv;
    h *= 1664525u;
    o.z = T(h & 0xFFFFFFu) * inv;
    return o;
}

//====================================
//Smooth Worley F1 over a 3x3x3 cell neighbourhood.
//p is in cell-space (one unit per cell). Returns smooth-min distance to
//the nearest feature point; gradOut is the analytic gradient w.r.t. p.
//
//Smooth-min: F1_smooth = -log(sum_i exp(-k*d_i)) / k. Gradient is
//sum_i (w_i * dd_i/dp) / sum_j w_j with w_i = exp(-k*d_i). This stays
//differentiable across Voronoi cell boundaries (the standard hard-min
//F1 is C^0 only; the gradient discontinuity at the boundary aliases
//visibly in normal mapping).
//====================================
template <typename T>
inline T pt_worley_smooth(const Vec3<T>& p, Vec3<T>& gradOut) {
    const int32_t ix = static_cast<int32_t>(std::floor(p.x));
    const int32_t iy = static_cast<int32_t>(std::floor(p.y));
    const int32_t iz = static_cast<int32_t>(std::floor(p.z));
    const Vec3<T> pf = { p.x - T(ix), p.y - T(iy), p.z - T(iz) };

    T sumExp = T(0);
    Vec3<T> sumWeightedDir{};
    const T k = T(PROCTERRAIN_SMOOTH_K);

    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                const Vec3<T> o = pt_cell_feature<T>(ix + dx, iy + dy, iz + dz);
                //dv = (cell + featureOffset) - p, in cell units
                const Vec3<T> dv = { T(dx) + o.x - pf.x,
                                     T(dy) + o.y - pf.y,
                                     T(dz) + o.z - pf.z };
                const T d = std::sqrt(dot(dv, dv) + T(1e-12));
                const T w = std::exp(-k * d);
                sumExp += w;
                //gradient of d w.r.t. p is (p - feature)/d = -dv/d
                const Vec3<T> gd = dv * (T(-1) / d);
                sumWeightedDir = sumWeightedDir + gd * w;
            }
        }
    }

    const T denom = sumExp + T(1e-30);
    //Smooth-min distance is mathematically -log(sumExp)/k. When several
    //feature points happen to cluster near the sample p, sumExp can climb
    //above 1 and the formula drifts NEGATIVE - a non-physical "distance
    //below zero" that displaces the mesh INWARD at deterministic positions
    //(the cell-feature hash is fixed per-world-coord, so the artifact is
    //pinned to a specific spot regardless of LOD or camera). True nearest-
    //feature distance is by definition >= 0, so clamp here. Zero the
    //gradient on the clamped branch too so the shader's analytic normal
    //agrees with the now-flat surface there - otherwise the shading
    //gradient would point INTO the surface where the geometry has been
    //flattened, flipping the normal.
    const T raw = -std::log(denom) / k;
    if (raw <= T(0)) {
        gradOut = Vec3<T>{};
        return T(0);
    }
    gradOut = sumWeightedDir * (T(1) / denom);
    return raw;
}

//====================================
//Worley FBM with Nyquist gate.
//P is the sample position in metres (Cartesian, sampled at dir * R_planet
//on the unit sphere). footprintM is the per-sample world-space footprint:
//vertex spacing on CPU, screen-pixel footprint on GPU. Octaves whose
//period <= 2*footprint are gated out so the same point evaluated at a
//coarser footprint doesn't see fine-octave content that would alias.
//
//Returns FBM value in metres; gradOut is the 3D gradient (dimensionless,
//rise / run).
//====================================
template <typename T>
inline T pt_fbm(const Vec3<T>& P, T footprintM, Vec3<T>& gradOut) {
    T value = T(0);
    gradOut = Vec3<T>{};

    T period = T(PROCTERRAIN_TOP_PERIOD_M);
    T amp    = T(PROCTERRAIN_TOP_AMP_M);
    T freq   = T(1) / period;
    const T lac           = T(PROCTERRAIN_LACUNARITY);
    const T gain          = T(PROCTERRAIN_GAIN);
    const T bottomPeriod  = T(PROCTERRAIN_BOTTOM_PERIOD_M);
    const T nyquistThresh = T(2) * footprintM;

    for (int i = 0; i < PROCTERRAIN_MAX_OCTAVES; ++i) {
        if (period < nyquistThresh) break;
        if (period < bottomPeriod)  break;

        Vec3<T> octGrad;
        const Vec3<T> octP = P * freq;   // cell-space for this octave
        const T octVal     = pt_worley_smooth<T>(octP, octGrad);

        value     += amp * octVal;
        gradOut.x += amp * freq * octGrad.x;
        gradOut.y += amp * freq * octGrad.y;
        gradOut.z += amp * freq * octGrad.z;

        period /= lac;
        amp    *= gain;
        freq   *= lac;
    }
    return value;
}

//====================================
//Vertex spacing on the surface for a chunk at LOD `lod`, in metres.
//Approximation: each face spans ~PI/2 of arc, subdivided into 2^lod
//chunks per axis, then 32 quads per chunk -> equiangular vertex spacing
//~ (PI/2) * R / (32 * 2^lod). The cube-sphere distorts a little near
//face corners (~sqrt(2) stretch) so this is the face-centre estimate;
//the Nyquist gate has tolerance for it.
//Used by the tessellator to choose the noise footprint per vertex.
//====================================
template <typename T>
inline T pt_chunk_vertex_spacing_m(T planetRadiusM, uint8_t lod) {
    const T face_arc = T(1.5707963267948966);  // PI / 2
    const T denom    = T(32) * T(uint64_t(1) << lod);
    return face_arc * planetRadiusM / denom;
}

} // namespace planet
