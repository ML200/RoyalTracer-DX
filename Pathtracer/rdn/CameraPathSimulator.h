#pragma once
#include <string>
#include <vector>
#include <filesystem>
#include <random>

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

    // Index of the NEXT step to be processed (moved-to or waited/captured depending on state)
    size_t GetCurrentStepIndex() const { return m_currentStepIndex; }

    // Index of the frame that was most recently requested for capture (valid when outShouldCapture==true)
    // Returns disk-relative index (internal index + file offset)
    size_t GetLastCaptureIndex() const { return m_lastCaptureIndex + m_fileIndexOffset; }

    // Expose the configured output directory so Renderer can save to the correct place
    std::wstring GetOutputDir() const { return m_outputDir.wstring(); }

    // Call this every frame in Renderer::OnUpdate
    // Returns true if the app should close (simulation finished)
    bool Update(float deltaTime, nv_helpers_dx12::Manipulator& camera, bool& outShouldCapture);

private:
    void LoadKeyframes(const std::wstring& filename);
    void GeneratePathPoints();

    // Resume by scanning output directory for numerically indexed files; starts at max+1
    size_t InferNextIndexFromOutputDir() const;

private:
    // --- State Flags ---
    bool m_isActive = false;

    // --- Configuration Variables ---
    int   m_configMaxSteps = 100;    // Total steps along the path
    float m_configWaitTime = 0.5f;   // Time to wait for accumulation

    // Rotational Configs
    int   m_configRollSteps = 1;     // Number of roll steps (360 degrees)
    float m_configYawAngle  = 0.0f;  // Max angle for forward rotation around up-axis
    int   m_configYawSteps  = 1;     // Number of steps for forward rotation around up-axis

    // --- Output/Resume ---
    std::filesystem::path m_outputDir = L"output";
    size_t m_fileIndexOffset = 0;    // The offset found on disk (e.g., if files 0-99 exist, this is 100)

    // --- Runtime Data ---
    std::vector<SimKeyframe> m_keyframes;
    std::vector<SimKeyframe> m_interpolatedPath; // The expanded list including rotations

    size_t m_currentStepIndex = 0;   // Internal step index (0 to N for THIS path)
    size_t m_lastCaptureIndex = 0;

    float m_currentWaitTimer = 0.0f;
    bool  m_isWaitingForConvergence = false; // false => next Update will move camera to currentStepIndex
    std::mt19937 m_rng{ std::random_device{}() };
    std::uniform_real_distribution<float> m_rollDist{ 0.0f, 360.0f };

};