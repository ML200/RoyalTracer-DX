#pragma once
#include "../../rdn/glm/glm.hpp"

class Camera;

class FlyCamController {
public:
    float moveSpeed        = 5.0f;
    float mouseSensitivity = 0.15f;

    //Camera ref is read each frame for the floating-origin offset; without it
    //the planet-radial up calculation goes stale once the origin snaps.
    void  SetCamera(Camera* c) { m_camera = c; }

    void  Update(float dt);

    //Drop the cached forward so the next Update re-syncs from the manipulator.
    //Call after Camera::ResetView() — otherwise the stale m_fwd snaps the look
    //direction back to the pre-reset orientation on the very next frame.
    void  Reset() { m_initialized = false; }
private:
    //Forward as a world vector instead of (yaw, pitch). Up tilts as the camera
    //flies away from the world tangent point (planet curvature), so Euler
    //angles measured against a fixed world axis stop matching the local frame.
    glm::vec3 m_fwd = glm::vec3(0, 0, -1);
    bool      m_initialized = false;
    Camera*   m_camera      = nullptr;
    void      InitFromManipulator();
};
