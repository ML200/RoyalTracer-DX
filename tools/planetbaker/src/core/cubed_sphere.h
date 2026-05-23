#pragma once

#include <cmath>

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

}
