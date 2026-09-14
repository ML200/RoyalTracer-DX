#include "coordinate_system.h"

namespace planet {

// Subtracts the camera origin before narrowing to float precision.
Vec3f to_camera_relative(const DVec3& world, const DVec3& camera_origin) {
    const DVec3 rel = world - camera_origin;
    return Vec3f{ static_cast<float>(rel.x),
                  static_cast<float>(rel.y),
                  static_cast<float>(rel.z) };
}

// Constructs near, far, and side planes in camera-relative space.
Frustum Frustum::from_camera(const CameraView& cam) {
    const Vec3f F = normalize(cam.forward);
    Vec3f R = cross(F, cam.up);
    if (length_sq(R) < 1e-12f)
        R = cross(F, Vec3f{ 1.f, 0.f, 0.f });
    R = normalize(R);
    const Vec3f U = cross(R, F);

    const float tanV = std::tan(cam.fov_y * 0.5f);
    const float tanH = tanV * cam.aspect;

    Frustum f;
    f.planes[0] = { F,   cam.near_plane };
    f.planes[1] = { -F, -cam.far_plane  };
    f.planes[2] = { normalize( R + F * tanH), 0.f };
    f.planes[3] = { normalize(-R + F * tanH), 0.f };
    f.planes[4] = { normalize( U + F * tanV), 0.f };
    f.planes[5] = { normalize(-U + F * tanV), 0.f };
    return f;
}

bool Frustum::intersects_sphere(const Vec3f& center, float radius) const {
    for (const Plane& p : planes) {
        if (dot(p.n, center) < p.d - radius)
            return false;
    }
    return true;
}

}
