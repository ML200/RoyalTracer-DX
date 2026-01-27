#pragma once
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include "glm/glm.hpp"
#include "manipulator.h" // For accessing the camera

struct SimKeyframe {
    glm::vec3 eye;
    glm::vec3 center;
    glm::vec3 up;
    float distanceToNext = 0.0f;
};

class CameraPathSimulator {
public:
    CameraPathSimulator();

    // Prompts user in console (Blocking I/O) using std::wcin/wcout
    void PromptUserConfiguration();

    // Returns true if simulation mode is active
    bool IsActive() const { return m_isActive; }

    // Call this every frame in Renderer::OnUpdate
    // Returns true if the app should close (simulation finished)
    bool Update(float deltaTime, nv_helpers_dx12::Manipulator& camera);

private:
    void LoadKeyframes(const std::wstring& filename);
    void GeneratePathPoints();

    // Configuration
    bool m_isActive = false;
    int m_configMaxSteps = 100;
    float m_configWaitTime = 1.0f; // Seconds to wait at each step

    // Runtime State
    std::vector<SimKeyframe> m_keyframes;
    std::vector<SimKeyframe> m_interpolatedPath; // The full list of steps

    size_t m_currentStepIndex = 0;
    float m_currentWaitTimer = 0.0f;
    bool m_isWaitingForConvergence = true;
};