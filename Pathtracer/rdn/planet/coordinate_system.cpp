//====================================
//PLANET - COORDINATE SYSTEM
//====================================

#include "coordinate_system.h"

namespace planet {

Vec3f to_camera_relative(const DVec3& world, const DVec3& camera_origin) {
    const DVec3 rel = world - camera_origin;          // FP64 subtraction first
    return Vec3f{ static_cast<float>(rel.x),
                  static_cast<float>(rel.y),
                  static_cast<float>(rel.z) };        // narrow to FP32 afterwards
}

//====================================
//FRUSTUM
//====================================
Frustum Frustum::from_camera(const CameraView& cam) {
    //orthonormal camera basis
    const Vec3f F = normalize(cam.forward);
    Vec3f R = cross(F, cam.up);
    if (length_sq(R) < 1e-12f)                        // forward ~parallel to up
        R = cross(F, Vec3f{ 1.f, 0.f, 0.f });
    R = normalize(R);
    const Vec3f U = cross(R, F);                      // exact up

    const float tanV = std::tan(cam.fov_y * 0.5f);
    const float tanH = tanV * cam.aspect;

    Frustum f;
    //near / far: offset along the view axis
    f.planes[0] = { F,   cam.near_plane };
    f.planes[1] = { -F, -cam.far_plane  };
    //side planes pass through the camera apex (origin); normals point inward
    f.planes[2] = { normalize( R + F * tanH), 0.f };  // left
    f.planes[3] = { normalize(-R + F * tanH), 0.f };  // right
    f.planes[4] = { normalize( U + F * tanV), 0.f };  // bottom
    f.planes[5] = { normalize(-U + F * tanV), 0.f };  // top
    return f;
}

bool Frustum::intersects_sphere(const Vec3f& center, float radius) const {
    for (const Plane& p : planes) {
        if (dot(p.n, center) < p.d - radius)
            return false;                             // sphere fully outside one plane
    }
    return true;
}

} // namespace planet
