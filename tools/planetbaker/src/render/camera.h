#pragma once

#include <glm/glm.hpp>

namespace pb {

class OrbitCamera {
public:
    glm::vec3 eye() const;
    glm::mat4 view() const;
    glm::mat4 proj(float aspect) const;

    void rotate(float dx_px, float dy_px);
    void zoom(float wheel_delta);
    void reset();

    float fov_deg  = 50.0f;
    float near_p   = 0.01f;
    float far_p    = 100.0f;
    float pitch    = 0.25f;
    float yaw      = 0.0f;
    float distance = 3.2f;
};

}
