#include "FlyCamController.h"
#include "../Input/InputManager.h"
#include "../../rdn/manipulator.h"
#include "../../rdn/Camera/Camera.h"
#include "../../rdn/glm/gtc/matrix_transform.hpp"
#include <cmath>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static constexpr float kPlanetRadiusM = 6360.0f * 1000.0f;

void FlyCamController::InitFromManipulator() {
    glm::vec3 eye, center, up;
    nv_helpers_dx12::CameraManip.getLookat(eye, center, up);
    m_fwd = glm::normalize(center - eye);
    m_initialized = true;
}

void FlyCamController::Update(float dt) {
    if (!m_initialized)
        InitFromManipulator();

    glm::vec3 eye, center, manipUp;
    nv_helpers_dx12::CameraManip.getLookat(eye, center, manipUp);

    // Motion in the shifted frame; up from world space.
    const glm::vec3 sceneOrigin = m_camera ? m_camera->getSceneOriginWorld() : glm::vec3(0.0f);
    const glm::vec3 eyeAbs = eye + sceneOrigin;
    const glm::vec3 planetCenter(0.0f, -kPlanetRadiusM, 0.0f);
    const glm::vec3 up = glm::normalize(eyeAbs - planetCenter);

    if (InputManager::GetMouseButton(0)) {
        const XMFLOAT2 d = InputManager::GetMouseDelta();
        const float toRad = (float)(M_PI / 180.0);
        float dYaw = -d.x * mouseSensitivity * toRad;
        float dPitch = -d.y * mouseSensitivity * toRad;

        const float kMaxPitch = 89.0f * toRad;
        const float curPitch = asinf(glm::clamp(glm::dot(m_fwd, up), -1.0f, 1.0f));
        const float newPitch = glm::clamp(curPitch + dPitch, -kMaxPitch, kMaxPitch);
        dPitch = newPitch - curPitch;

        const glm::mat3 Ryaw = glm::mat3(glm::rotate(glm::mat4(1.0f), dYaw, up));
        const glm::vec3 fwdY = Ryaw * m_fwd;
        const glm::vec3 right = glm::normalize(glm::cross(fwdY, up));
        const glm::mat3 Rpit = glm::mat3(glm::rotate(glm::mat4(1.0f), dPitch, right));
        m_fwd = glm::normalize(Rpit * fwdY);
    }

    const glm::vec3 right = glm::normalize(glm::cross(m_fwd, up));

    glm::vec3 move(0.0f);
    if (InputManager::GetKey('W'))
        move += m_fwd;
    if (InputManager::GetKey('S'))
        move -= m_fwd;
    if (InputManager::GetKey('D'))
        move += right;
    if (InputManager::GetKey('A'))
        move -= right;
    if (InputManager::GetKey(VK_SPACE))
        move += up;
    if (InputManager::GetKey(VK_CONTROL))
        move -= up;

    if (glm::length(move) > 0.0f)
        eye += glm::normalize(move) * (moveSpeed * dt);

    center = eye + m_fwd;
    nv_helpers_dx12::CameraManip.setLookat(eye, center, up);
}
