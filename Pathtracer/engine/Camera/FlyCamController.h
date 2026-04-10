#pragma once

class FlyCamController {
public:
    float moveSpeed        = 5.0f;
    float mouseSensitivity = 0.15f;
    void Update(float dt);
private:
    float m_yaw = 0, m_pitch = 0;
    bool  m_initialized = false;
    void InitFromManipulator();
};
