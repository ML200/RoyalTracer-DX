#pragma once
//====================================
//PLANET - COORDINATE SYSTEM
//====================================
//FP64 world space + camera-relative (floating-origin) conversion.
//
//World space is right-handed, metres, FP64. The planet centre is configurable
//(PlanetGeometry, cube_sphere.h); the test planet sits at the origin. Chunks are
//anchored in FP64 world space and converted to FP32 camera-relative space (camera
//at the origin) for all culling / LOD / rendering math. That conversion is the
//floating-origin trick that keeps FP32 precision usable at planetary scale.
//
//CPU only, no GPU types. See rdn/planet/INTEGRATION_NOTES.md for how this
//reconciles with the renderer's existing FP32 Scene::sceneOriginWorld at Phase 5.

#include <cstdint>
#include <cmath>

namespace planet {

//====================================
//VEC3
//====================================
template <typename T>
struct Vec3 {
    T x = T(0), y = T(0), z = T(0);

    constexpr Vec3() = default;
    constexpr Vec3(T x_, T y_, T z_) : x(x_), y(y_), z(z_) {}

    constexpr Vec3 operator+(const Vec3& r) const { return { x + r.x, y + r.y, z + r.z }; }
    constexpr Vec3 operator-(const Vec3& r) const { return { x - r.x, y - r.y, z - r.z }; }
    constexpr Vec3 operator*(T s)           const { return { x * s, y * s, z * s }; }
    constexpr Vec3 operator/(T s)           const { return { x / s, y / s, z / s }; }
    constexpr Vec3 operator-()              const { return { -x, -y, -z }; }
};

template <typename T> constexpr T dot(const Vec3<T>& a, const Vec3<T>& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
template <typename T> constexpr Vec3<T> cross(const Vec3<T>& a, const Vec3<T>& b) {
    return { a.y * b.z - a.z * b.y,
             a.z * b.x - a.x * b.z,
             a.x * b.y - a.y * b.x };
}
template <typename T> T length_sq(const Vec3<T>& v) { return dot(v, v); }
template <typename T> T length   (const Vec3<T>& v) { return std::sqrt(dot(v, v)); }

//normalize; returns {0,0,0} for a (near-)zero vector so callers never divide by 0
template <typename T> Vec3<T> normalize(const Vec3<T>& v) {
    const T len2 = dot(v, v);
    if (len2 <= T(0)) return {};
    return v * (T(1) / std::sqrt(len2));
}

using DVec3 = Vec3<double>;  // FP64 world space
using Vec3f = Vec3<float>;   // FP32 camera-relative space

//====================================
//CAMERA-RELATIVE CONVERSION
//====================================
//Subtract the camera origin in FP64, THEN narrow to FP32. The order matters: the
//subtraction must happen in FP64 or planetary-scale magnitudes destroy precision.
Vec3f to_camera_relative(const DVec3& world, const DVec3& camera_origin);

//====================================
//CAMERA VIEW
//====================================
//Lightweight, test-constructible camera description. The engine Camera is adapted
//into this at Phase 5; Phase 1 stays decoupled so the visibility pass is
//unit-testable in isolation. (The plan sketched priority_score(Chunk, Camera) -
//that Camera is this CameraView.)
struct CameraView {
    DVec3 position_world{};            // FP64 world-space eye position
    //FP64 floating-origin used as the unified-TLAS origin (Phase 5). Terrain and
    //scene instances are both placed relative to this so they share one frame.
    DVec3 scene_origin{};
    Vec3f forward{ 0.f, 0.f, 1.f };    // view direction (re-normalized on use)
    Vec3f up     { 0.f, 1.f, 0.f };    // approximate up (re-orthonormalized on use)
    float fov_y      = 1.0f;           // vertical field of view, radians
    float aspect     = 16.0f / 9.0f;   // width / height
    float near_plane = 1.0f;           // metres
    float far_plane  = 1.0e9f;         // metres - planetary scale
};

//====================================
//FRUSTUM (camera-relative, FP32)
//====================================
struct Plane {
    Vec3f n{};      // unit normal, points toward the inside of the frustum
    float d = 0.f;  // a point p is inside this plane iff dot(n, p) >= d
};

struct Frustum {
    Plane planes[6];  // near, far, left, right, bottom, top

    static Frustum from_camera(const CameraView& cam);

    //conservative test - 'center' is in camera-relative space (camera at origin)
    bool intersects_sphere(const Vec3f& center, float radius) const;
};

} // namespace planet
