#include "render/camera.h"

#include <algorithm>
#include <cmath>

#include <glm/gtc/matrix_transform.hpp>

namespace pb {

glm::vec3 OrbitCamera::eye() const {
    float cp = std::cos(pitch), sp = std::sin(pitch);
    float cy = std::cos(yaw),   sy = std::sin(yaw);
    return glm::vec3(cy * cp, sp, sy * cp) * distance;
}

glm::mat4 OrbitCamera::view() const {
    return glm::lookAt(eye(), glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
}

glm::mat4 OrbitCamera::proj(float aspect) const {
    return glm::perspective(glm::radians(fov_deg), aspect, near_p, far_p);
}

void OrbitCamera::rotate(float dx_px, float dy_px) {
    const float s = 0.005f;
    yaw   -= dx_px * s;
    pitch += dy_px * s;
    const float lim = 1.55f;
    pitch = std::clamp(pitch, -lim, lim);
}

void OrbitCamera::zoom(float wheel_delta) {
    distance *= std::pow(1.15f, -wheel_delta);
    distance = std::clamp(distance, 1.05f, 30.0f);
}

void OrbitCamera::reset() {
    pitch = 0.25f;
    yaw = 0.0f;
    distance = 3.2f;
}

}
