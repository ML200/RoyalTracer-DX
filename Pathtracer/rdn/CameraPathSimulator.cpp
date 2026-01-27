#include "CameraPathSimulator.h"
#include <algorithm>
#include <iomanip>
#include <Windows.h>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

// GLM Extensions for vector rotation
#define GLM_ENABLE_EXPERIMENTAL
#include "glm/gtx/rotate_vector.hpp"
#include "glm/gtc/constants.hpp"

// Helper to convert wstring to string
std::string ToString(const std::wstring& wstr) {
    if (wstr.empty()) return std::string();
    int size_needed = WideCharToMultiByte(CP_UTF8, 0, &wstr[0], (int)wstr.size(), NULL, 0, NULL, NULL);
    std::string strTo(size_needed, 0);
    WideCharToMultiByte(CP_UTF8, 0, &wstr[0], (int)wstr.size(), &strTo[0], size_needed, NULL, NULL);
    return strTo;
}

// Helper to convert string to wstring
std::wstring ToWString(const std::string& str) {
    if (str.empty()) return std::wstring();
    int size_needed = MultiByteToWideChar(CP_UTF8, 0, &str[0], (int)str.size(), NULL, 0);
    std::wstring wstrTo(size_needed, 0);
    MultiByteToWideChar(CP_UTF8, 0, &str[0], (int)str.size(), &wstrTo[0], size_needed);
    return wstrTo;
}

CameraPathSimulator::CameraPathSimulator()
    : m_configMaxSteps(100)
    , m_configWaitTime(0.5f)
    , m_configRollSteps(1)
    , m_configYawAngle(0.0f)
    , m_configYawSteps(1)
{
}

void CameraPathSimulator::PromptUserConfiguration() {
    // ---------------------------------------------------------
    // 1. OPEN NEW CONSOLE WINDOW (For Status Output)
    // ---------------------------------------------------------
    if (AllocConsole()) {
        FILE* fpDummy;
        freopen_s(&fpDummy, "CONIN$", "r", stdin);
        freopen_s(&fpDummy, "CONOUT$", "w", stdout);
        freopen_s(&fpDummy, "CONOUT$", "w", stderr);

        std::wcin.clear();
        std::wcout.clear();
        std::wcerr.clear();
        std::cin.clear();
        std::cout.clear();
        std::cerr.clear();

        HWND consoleWindow = GetConsoleWindow();
        SetForegroundWindow(consoleWindow);
    }

    std::wcout << L"\n============================================\n";
    std::wcout << L"       CAMERA SIMULATOR CONFIGURATION       \n";
    std::wcout << L"============================================\n";

    // ---------------------------------------------------------
    // 2. LOAD CONFIGURATION FROM FILE
    // ---------------------------------------------------------
    const std::string configFileName = "sim_config.txt";
    std::ifstream configFile(configFileName);

    if (!configFile.is_open()) {
        std::wcout << L"[Info] 'sim_config.txt' not found. Defaulting to Realtime Mode.\n";
        m_isActive = false;
        std::wcout << L">> Closing Console in 2 seconds...\n";
        Sleep(2000);
        FreeConsole();
        return;
    }

    std::string keyframeFileStr;
    std::string line;
    std::string key;
    bool runSim = false;

    // Default values before parsing
    m_configMaxSteps = 100;
    m_configWaitTime = 0.5f;

    while (std::getline(configFile, line)) {
        if (line.empty() || line[0] == '#') continue; // Skip comments
        std::stringstream ss(line);
        ss >> key;

        if (key == "Mode") {
            std::string modeVal;
            ss >> modeVal;
            if (modeVal == "Simulation") runSim = true;
        }
        else if (key == "KeyframeFile") ss >> keyframeFileStr;
        else if (key == "TotalPathSteps") ss >> m_configMaxSteps;
        else if (key == "WaitTime") ss >> m_configWaitTime;
        else if (key == "ForwardRollSteps") ss >> m_configRollSteps; // 360 rotation
        else if (key == "UpRotationAngle") ss >> m_configYawAngle;   // e.g., 30 degrees
        else if (key == "UpRotationSteps") ss >> m_configYawSteps;   // e.g., 3
    }
    configFile.close();

    // ---------------------------------------------------------
    // 3. APPLY CONFIGURATION
    // ---------------------------------------------------------
    if (runSim) {
        m_isActive = true;
        std::wstring wFilename = ToWString(keyframeFileStr);

        std::wcout << L"Mode: SIMULATION\n";
        std::wcout << L"File: " << wFilename << L"\n";
        std::wcout << L"Base Steps: " << m_configMaxSteps << L"\n";
        std::wcout << L"Variations: Roll=" << m_configRollSteps
                   << L", Yaw=" << m_configYawSteps << L" (+/- " << m_configYawAngle << L" deg)\n";

        LoadKeyframes(wFilename);



        GeneratePathPoints();

        std::wcout << L">> Data Generation Ready. " << m_interpolatedPath.size() << L" total frames.\n";
        std::wcout << L">> Closing Console...\n";
        Sleep(2000);
    } else {
        m_isActive = false;
        std::wcout << L"Mode: REALTIME (Interactive)\n";
        std::wcout << L">> Closing Console...\n";
        Sleep(1000);
    }

    // Clean up console so it doesn't hang around empty
    FreeConsole();
}

void CameraPathSimulator::LoadKeyframes(const std::wstring& filename) {
    std::ifstream file(ToString(filename));

    if (!file.is_open()) {
        std::wcout << L"[Error] Could not open keyframe file. Aborting.\n";
        m_isActive = false;
        return;
    }

    m_keyframes.clear();
    std::string line;
    std::getline(file, line); // Skip header

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        float t;
        SimKeyframe k;
        ss >> t
           >> k.eye.x >> k.eye.y >> k.eye.z
           >> k.center.x >> k.center.y >> k.center.z
           >> k.up.x >> k.up.y >> k.up.z;
        m_keyframes.push_back(k);
    }

    // Ensure we have at least 2 keyframes to interpolate
    if (m_keyframes.size() < 2 && !m_keyframes.empty()) {
        // duplicate single keyframe to allow loop to run once
        m_keyframes.push_back(m_keyframes[0]);
    }
}

void CameraPathSimulator::GeneratePathPoints() {
    if (m_keyframes.empty()) return;

    m_interpolatedPath.clear();

    // 1. Calculate total distance for normalizing speed
    float totalDistance = 0.0f;
    for (size_t i = 0; i < m_keyframes.size() - 1; ++i) {
        float d = glm::distance(m_keyframes[i].eye, m_keyframes[i+1].eye);
        m_keyframes[i].distanceToNext = d;
        totalDistance += d;
    }

    // 2. Generate Base Points (Linear Interpolation)
    std::vector<SimKeyframe> basePoints;

    for (size_t i = 0; i < m_keyframes.size() - 1; ++i) {
        SimKeyframe& start = m_keyframes[i];
        SimKeyframe& end   = m_keyframes[i+1];

        // Determine how many steps this segment gets based on physical distance
        float segmentFraction = (totalDistance > 0.0001f) ? (start.distanceToNext / totalDistance) : 0.0f;
        int segmentSteps = static_cast<int>(std::round(segmentFraction * m_configMaxSteps));
        if (segmentSteps < 1) segmentSteps = 1;

        for (int s = 0; s < segmentSteps; ++s) {
            float t = (float)s / (float)segmentSteps;
            SimKeyframe point;
            point.eye    = glm::mix(start.eye, end.eye, t);
            point.center = glm::mix(start.center, end.center, t);
            point.up     = glm::normalize(glm::mix(start.up, end.up, t));
            basePoints.push_back(point);
        }
    }
    basePoints.push_back(m_keyframes.back());

    // 3. Generate Rotational Variations for each Base Point
    for (const auto& base : basePoints) {

        glm::vec3 forward = glm::normalize(base.center - base.eye);
        float lookDist = glm::distance(base.center, base.eye);
        glm::vec3 originalUp = glm::normalize(base.up);

        // --- Variation Loop 1: Yaw (Rotation around UP vector) ---
        // Range: [-m_configYawAngle, +m_configYawAngle]

        int ySteps = std::fmax(1, m_configYawSteps);

        for (int y = 0; y < ySteps; ++y) {
            float yawAngleDeg = 0.0f;

            // Only calculate angle offset if we have more than 1 step requested
            if (ySteps > 1) {
                float t = (float)y / (float)(ySteps - 1); // 0.0 to 1.0
                yawAngleDeg = glm::mix(-m_configYawAngle, m_configYawAngle, t);
            }

            // Rotate the Forward vector around the Up axis
            // Note: glm::rotate takes angle in radians
            glm::vec3 yawedForward = glm::rotate(forward, glm::radians(yawAngleDeg), originalUp);
            glm::vec3 yawedCenter  = base.eye + (yawedForward * lookDist);

            // --- Variation Loop 2: Roll (Rotation around FORWARD vector) ---
            // Range: [0, 360]

            int rSteps = std::fmax(1, m_configRollSteps);

            for (int r = 0; r < rSteps; ++r) {
                float rollAngleDeg = 0.0f;

                if (rSteps > 1) {
                     // 0 to 360, but exclusive of 360 if it wraps perfectly?
                     // Usually 0 and 360 are same, so we do 0 to 360 * (N-1)/N
                     rollAngleDeg = 360.0f * ((float)r / (float)rSteps);
                }

                // Rotate the Up vector around the *new* Forward axis
                glm::vec3 finalUp = glm::rotate(originalUp, glm::radians(rollAngleDeg), yawedForward);

                // Create Final Keyframe
                SimKeyframe finalFrame;
                finalFrame.eye = base.eye;
                finalFrame.center = yawedCenter;
                finalFrame.up = finalUp;
                finalFrame.distanceToNext = 0; // Not used for playback

                m_interpolatedPath.push_back(finalFrame);
            }
        }
    }
}

// In CameraPathSimulator.cpp

// UPDATE the function signature and logic
bool CameraPathSimulator::Update(float deltaTime, nv_helpers_dx12::Manipulator& camera, bool& outShouldCapture) {
    outShouldCapture = false; // Reset flag by default

    if (!m_isActive) return false;

    // Check if finished
    if (m_currentStepIndex >= m_interpolatedPath.size()) {
        return true;
    }

    if (m_isWaitingForConvergence) {
        m_currentWaitTimer += deltaTime;

        // Check if convergence time has passed
        if (m_currentWaitTimer >= m_configWaitTime) {
            // SIGNAL CAPTURE NOW
            outShouldCapture = true;

            // Advance to next step
            m_currentWaitTimer = 0.0f;
            m_currentStepIndex++;
            m_isWaitingForConvergence = false;
        }
    }
    else {
        // Move camera to next position
        if (m_currentStepIndex < m_interpolatedPath.size()) {
            const SimKeyframe& target = m_interpolatedPath[m_currentStepIndex];
            camera.setLookat(target.eye, target.center, target.up);
            m_isWaitingForConvergence = true;
        }
    }
    return false;
}