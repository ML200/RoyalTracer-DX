#pragma once
#include "../../rdn/glm/glm.hpp"

class Camera;

class FlyCamController {
  public:
    float moveSpeed = 5.0f;
    float mouseSensitivity = 0.15f;

    void SetCamera(Camera* c) { m_camera = c; }

    void Update(float dt);

    void Reset() { m_initialized = false; }

  private:
    glm::vec3 m_fwd = glm::vec3(0, 0, -1);
    bool m_initialized = false;
    Camera* m_camera = nullptr;
    void InitFromManipulator();
};
