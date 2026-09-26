#pragma once
#include <string>
#include <vector>
#include <filesystem>
#include <random>

#include "glm/glm.hpp"
#include "manipulator.h"

struct SimKeyframe {
    glm::vec3 eye;
    glm::vec3 center;
    glm::vec3 up;
    float distanceToNext = 0.0f;
};

class CameraPathSimulator {
  public:
    CameraPathSimulator();

    void PromptUserConfiguration();

    bool IsActive() const { return m_isActive; }

    size_t GetLastCaptureIndex() const { return m_lastCaptureIndex + m_fileIndexOffset; }

    bool Update(float deltaTime, nv_helpers_dx12::Manipulator& camera, bool& outShouldCapture);

  private:
    void LoadKeyframes(const std::wstring& filename);
    void GeneratePathPoints();

    size_t InferNextIndexFromOutputDir() const;

  private:
    bool m_isActive = false;

    int m_configMaxSteps = 100;
    float m_configWaitTime = 0.5f;

    int m_configRollSteps = 1;
    float m_configYawAngle = 0.0f;
    int m_configYawSteps = 1;

    std::filesystem::path m_outputDir = L"output";
    size_t m_fileIndexOffset = 0;

    std::vector<SimKeyframe> m_keyframes;
    std::vector<SimKeyframe> m_interpolatedPath;

    size_t m_currentStepIndex = 0;
    size_t m_lastCaptureIndex = 0;

    float m_currentWaitTimer = 0.0f;
    bool m_isWaitingForConvergence = false;
    std::mt19937 m_rng{std::random_device{}()};
    std::uniform_real_distribution<float> m_rollDist{0.0f, 360.0f};
};
