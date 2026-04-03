#include "FlyCamController.h"
#include "../Input/InputManager.h"
#include "../../rdn/manipulator.h"
#include <cmath>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void FlyCamController::InitFromManipulator() {
    glm::vec3 eye, center, up;
    nv_helpers_dx12::CameraManip.getLookat(eye, center, up);
    glm::vec3 fwd = glm::normalize(center - eye);
    m_pitch = -asinf(glm::clamp(fwd.y, -1.0f, 1.0f)) * (180.0f / (float)M_PI);
    m_yaw   = atan2f(fwd.x, fwd.z) * (180.0f / (float)M_PI);
    m_initialized = true;
}

void FlyCamController::Update(float dt) {
    if (!m_initialized) InitFromManipulator();

    if (InputManager::GetMouseButton(0)) {
        XMFLOAT2 delta = InputManager::GetMouseDelta();
        m_yaw   -= delta.x * mouseSensitivity;
        m_pitch += delta.y * mouseSensitivity;
        m_pitch  = std::clamp(m_pitch, -89.0f, 89.0f);
    }

    float yawRad   = m_yaw   * ((float)M_PI / 180.0f);
    float pitchRad = m_pitch * ((float)M_PI / 180.0f);

    glm::vec3 fwd = glm::normalize(glm::vec3(
        cosf(pitchRad) * sinf(yawRad), -sinf(pitchRad), cosf(pitchRad) * cosf(yawRad)));
    glm::vec3 worldUp(0, 1, 0);
    glm::vec3 right = glm::normalize(glm::cross(fwd, worldUp));

    glm::vec3 move(0.0f);
    if (InputManager::GetKey('W'))        move += fwd;
    if (InputManager::GetKey('S'))        move -= fwd;
    if (InputManager::GetKey('D'))        move += right;
    if (InputManager::GetKey('A'))        move -= right;
    if (InputManager::GetKey(VK_SPACE))   move += worldUp;
    if (InputManager::GetKey(VK_CONTROL)) move -= worldUp;

    glm::vec3 eye, center, up;
    nv_helpers_dx12::CameraManip.getLookat(eye, center, up);
    if (glm::length(move) > 0.0f) eye += glm::normalize(move) * (moveSpeed * dt);
    center = eye + fwd;
    nv_helpers_dx12::CameraManip.setLookat(eye, center, worldUp);
}
