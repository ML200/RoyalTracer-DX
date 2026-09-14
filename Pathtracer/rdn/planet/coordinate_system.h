#pragma once

#include <cstdint>
#include <cmath>

namespace planet {

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

template <typename T> Vec3<T> normalize(const Vec3<T>& v) {
    const T len2 = dot(v, v);
    if (len2 <= T(0)) return {};
    return v * (T(1) / std::sqrt(len2));
}

using DVec3 = Vec3<double>;
using Vec3f = Vec3<float>;

// Converts absolute doubles into renderer-safe camera-relative floats.
Vec3f to_camera_relative(const DVec3& world, const DVec3& camera_origin);

struct CameraView {
    DVec3 position_world{};
    DVec3 scene_origin{};
    Vec3f forward{ 0.f, 0.f, 1.f };
    Vec3f up     { 0.f, 1.f, 0.f };
    float fov_y      = 1.0f;
    float aspect     = 16.0f / 9.0f;
    float near_plane = 1.0f;
    float far_plane  = 1.0e9f;
};

struct Plane {
    Vec3f n{};
    float d = 0.f;
};

struct Frustum {
    Plane planes[6];

    // Builds inward-facing planes from the camera basis.
    static Frustum from_camera(const CameraView& cam);

    // Tests a sphere against all six frustum planes.
    bool intersects_sphere(const Vec3f& center, float radius) const;
};

}
