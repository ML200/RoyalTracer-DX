#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>

#ifdef __CUDACC__
#define PB_HOSTDEVICE __host__ __device__
#else
#define PB_HOSTDEVICE
#endif

namespace pb {

struct Vec3f {
    float x, y, z;
};

PB_HOSTDEVICE inline Vec3f vec3f_normalize(Vec3f v) {
    float inv = 1.0f / sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    return Vec3f{v.x * inv, v.y * inv, v.z * inv};
}

//====================================
//Equiangular cubed sphere parameterization
//Face order: 0=+X, 1=-X, 2=+Y, 3=-Y, 4=+Z, 5=-Z
//Face local coords u,v in [-1, +1]
//Texture coords s,t in [0, 1] map as s=(u+1)/2, t=(v+1)/2
//Equiangular: angles are linear in u,v so cell solid angles are nearly equal
//====================================

PB_HOSTDEVICE inline Vec3f face_uv_to_sphere(int face, float u, float v) {
    const float PI = 3.14159265358979323846f;
    float ut = tanf(u * PI * 0.25f);
    float vt = tanf(v * PI * 0.25f);
    Vec3f p;
    switch (face) {
        case 0:  p = Vec3f{ 1.0f, -vt, -ut}; break;
        case 1:  p = Vec3f{-1.0f, -vt,  ut}; break;
        case 2:  p = Vec3f{ ut,  1.0f,  vt}; break;
        case 3:  p = Vec3f{ ut, -1.0f, -vt}; break;
        case 4:  p = Vec3f{ ut, -vt,  1.0f}; break;
        default: p = Vec3f{-ut, -vt, -1.0f}; break;
    }
    return vec3f_normalize(p);
}

PB_HOSTDEVICE inline void sphere_to_face_uv(Vec3f p, int& face, float& u, float& v) {
    const float PI = 3.14159265358979323846f;
    float ax = fabsf(p.x), ay = fabsf(p.y), az = fabsf(p.z);
    float ut = 0.0f, vt = 0.0f;
    if (ax >= ay && ax >= az) {
        if (p.x > 0.0f) { face = 0; ut = -p.z / ax; vt = -p.y / ax; }
        else            { face = 1; ut =  p.z / ax; vt = -p.y / ax; }
    } else if (ay >= ax && ay >= az) {
        if (p.y > 0.0f) { face = 2; ut =  p.x / ay; vt =  p.z / ay; }
        else            { face = 3; ut =  p.x / ay; vt = -p.z / ay; }
    } else {
        if (p.z > 0.0f) { face = 4; ut =  p.x / az; vt = -p.y / az; }
        else            { face = 5; ut = -p.x / az; vt = -p.y / az; }
    }
    u = atanf(ut) * 4.0f / PI;
    v = atanf(vt) * 4.0f / PI;
}

//====================================
//CubedSphereGrid describes a single resolution (face_resolution N) for the
//six-face simulation domain. All Field<T> instances and the NeighborTable
//are sized by this grid.
//
//Storage layout per face: (N + 2*HALO) x (N + 2*HALO) cells, row major,
//j-major. Interior cells have i,j in [0, N). Ghost (halo) cells have
//i or j in [-HALO, -1] or [N, N+HALO-1].
//
//A cell center in local face coords is u(i) = (i + 0.5)/N * 2 - 1, same for v.
//====================================

struct CubedSphereGrid {
    static constexpr int HALO = 2;

    int n = 0;

    CubedSphereGrid() = default;
    explicit CubedSphereGrid(int face_resolution) : n(face_resolution) {}

    PB_HOSTDEVICE int stride()           const { return n + 2 * HALO; }
    PB_HOSTDEVICE int cells_per_face()   const { return stride() * stride(); }
    PB_HOSTDEVICE int total_cells()      const { return 6 * cells_per_face(); }
    PB_HOSTDEVICE int interior_per_face()const { return n * n; }
    PB_HOSTDEVICE int total_interior()   const { return 6 * interior_per_face(); }

    //Linear index of cell (i, j) on a face. i, j may be in [-HALO, N+HALO).
    PB_HOSTDEVICE int cell_index(int i, int j) const {
        return (j + HALO) * stride() + (i + HALO);
    }

    //Linear index of cell (face, i, j) in the full 6-face buffer.
    PB_HOSTDEVICE int global_index(int face, int i, int j) const {
        return face * cells_per_face() + cell_index(i, j);
    }

    PB_HOSTDEVICE float u_of_i(int i) const {
        return ((static_cast<float>(i) + 0.5f) / static_cast<float>(n)) * 2.0f - 1.0f;
    }
    PB_HOSTDEVICE float v_of_j(int j) const { return u_of_i(j); }

    //Nearest integer cell index for a u in [-1, +1]. Result clamped to [0, N-1].
    PB_HOSTDEVICE int i_of_u(float u) const {
        float fi = (u + 1.0f) * 0.5f * static_cast<float>(n) - 0.5f;
        int   i  = static_cast<int>(fi + (fi >= 0.0f ? 0.5f : -0.5f));
        if (i < 0)       i = 0;
        if (i >= n)      i = n - 1;
        return i;
    }
    PB_HOSTDEVICE int j_of_v(float v) const { return i_of_u(v); }
};

//====================================
//Per-cell 4-neighbor lookup. The four (face, i, j) triplets are the neighbor
//cells in the +i, -i, +j, -j directions. Interior cells return neighbors on
//the same face. Edge cells return neighbors on the adjacent face. Corner
//cells pick whichever face the diagonal step lands on.
//====================================

struct Neighbor {
    int face;
    int i;
    int j;
};

struct Neighbors4 {
    Neighbor plus_i;
    Neighbor minus_i;
    Neighbor plus_j;
    Neighbor minus_j;
};

}
