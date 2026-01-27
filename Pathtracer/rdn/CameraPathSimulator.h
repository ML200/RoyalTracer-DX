#pragma once
#include <string>
#include <vector>
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

    // Prompts user in console or loads from file
    void PromptUserConfiguration();

    // Returns true if simulation mode is active
    bool IsActive() const { return m_isActive; }
    size_t GetCurrentStepIndex() const { return m_currentStepIndex; }

    // Call this every frame in Renderer::OnUpdate
    // Returns true if the app should close (simulation finished)
    bool Update(float deltaTime, nv_helpers_dx12::Manipulator& camera, bool& outShouldCapture);

private:
    void LoadKeyframes(const std::wstring& filename);
    void GeneratePathPoints();

    // --- State Flags ---
    bool m_isActive = false;

    // --- Configuration Variables ---
    int m_configMaxSteps = 100;    // Total steps along the path
    float m_configWaitTime = 0.5f; // Time to wait for accumulation

    // New Rotational Configs
    int m_configRollSteps = 1;     // Number of roll steps (360 degrees)
    float m_configYawAngle = 0.0f; // Max angle for up-vector rotation
    int m_configYawSteps = 1;      // Number of steps for up-vector rotation

    // --- Runtime Data ---
    std::vector<SimKeyframe> m_keyframes;
    std::vector<SimKeyframe> m_interpolatedPath; // The expanded list including rotations

    size_t m_currentStepIndex = 0;
    float m_currentWaitTimer = 0.0f;
    bool m_isWaitingForConvergence = true;
};