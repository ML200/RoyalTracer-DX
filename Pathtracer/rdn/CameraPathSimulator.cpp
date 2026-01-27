#include "CameraPathSimulator.h"
#include <algorithm>
#include <iomanip>
#include <Windows.h> // Essential for AllocConsole/FreeConsole
#include <cstdio>    // Essential for freopen_s
#include <iostream>

// Helper to convert wstring to string
std::string ToString(const std::wstring& wstr) {
    if (wstr.empty()) return std::string();
    int size_needed = WideCharToMultiByte(CP_UTF8, 0, &wstr[0], (int)wstr.size(), NULL, 0, NULL, NULL);
    std::string strTo(size_needed, 0);
    WideCharToMultiByte(CP_UTF8, 0, &wstr[0], (int)wstr.size(), &strTo[0], size_needed, NULL, NULL);
    return strTo;
}

CameraPathSimulator::CameraPathSimulator() {}

void CameraPathSimulator::PromptUserConfiguration() {
    // ---------------------------------------------------------
    // 1. OPEN NEW CONSOLE WINDOW
    // ---------------------------------------------------------
    AllocConsole();

    // Redirect standard streams (stdin, stdout, stderr) to the new console
    FILE* fpDummy;
    freopen_s(&fpDummy, "CONIN$", "r", stdin);
    freopen_s(&fpDummy, "CONOUT$", "w", stdout);
    freopen_s(&fpDummy, "CONOUT$", "w", stderr);

    // Synchronize C++ streams with the new console handles
    std::wcin.clear();
    std::wcout.clear();
    std::wcerr.clear();
    std::cin.clear();
    std::cout.clear();
    std::cerr.clear();

    // Bring the new console to the front
    HWND consoleWindow = GetConsoleWindow();
    SetForegroundWindow(consoleWindow);

    // ---------------------------------------------------------
    // 2. RUN DIALOGUE
    // ---------------------------------------------------------
    std::wcout << L"\n============================================\n";
    std::wcout << L"       RENDERER RUN MODE SELECTION          \n";
    std::wcout << L"============================================\n";
    std::wcout << L"1. Realtime (Interactive)\n";
    std::wcout << L"2. Keyframe Simulation (Auto Data Gen)\n";
    std::wcout << L"Select option [1-2]: ";

    int choice;
    while (!(std::wcin >> choice)) {
        std::wcin.clear();
        std::wcin.ignore(10000, L'\n');
        std::wcout << L"Invalid input. Please enter 1 or 2: ";
    }

    if (choice == 2) {
        m_isActive = true;

        std::wstring filename;
        std::wcout << L"Enter Keyframe Filename (e.g. camera_path.txt): ";
        std::wcin >> filename;

        std::wcout << L"Total Number of Steps (frames) to generate: ";
        std::wcin >> m_configMaxSteps;

        std::wcout << L"Convergence wait time per step (seconds): ";
        std::wcin >> m_configWaitTime;

        LoadKeyframes(filename);
        GeneratePathPoints();

        std::wcout << L"\n>> Configuration Complete. Closing Console...\n";
        Sleep(1000); // Give user a second to read "Complete"
    } else {
        m_isActive = false;
        std::wcout << L"\n>> Starting Realtime Mode. Closing Console...\n";
        Sleep(500);
    }

    // ---------------------------------------------------------
    // 3. CLOSE CONSOLE WINDOW
    // ---------------------------------------------------------
    FreeConsole();
}

void CameraPathSimulator::LoadKeyframes(const std::wstring& filename) {
    std::ifstream file(ToString(filename));

    // Note: Since we are inside the console session here, wcout works.
    if (!file.is_open()) {
        std::wcout << L"[Error] Could not open file '" << filename << L"'. Aborting Sim.\n";
        m_isActive = false;
        system("pause"); // Let user see the error
        return;
    }

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

    if (m_keyframes.size() < 2) {
        std::wcout << L"[Error] Need at least 2 keyframes. Found " << m_keyframes.size() << L".\n";
        m_isActive = false;
        system("pause");
    }
}

void CameraPathSimulator::GeneratePathPoints() {
    if (m_keyframes.empty()) return;

    float totalDistance = 0.0f;
    for (size_t i = 0; i < m_keyframes.size() - 1; ++i) {
        float d = glm::distance(m_keyframes[i].eye, m_keyframes[i+1].eye);
        m_keyframes[i].distanceToNext = d;
        totalDistance += d;
    }

    m_interpolatedPath.clear();

    for (size_t i = 0; i < m_keyframes.size() - 1; ++i) {
        SimKeyframe& start = m_keyframes[i];
        SimKeyframe& end   = m_keyframes[i+1];

        float segmentFraction = (totalDistance > 0.0001f) ? (start.distanceToNext / totalDistance) : 0.0f;
        int segmentSteps = static_cast<int>(std::round(segmentFraction * m_configMaxSteps));

        if (segmentSteps < 1) segmentSteps = 1;

        for (int s = 0; s < segmentSteps; ++s) {
            float t = (float)s / (float)segmentSteps;

            SimKeyframe point;
            point.eye    = glm::mix(start.eye, end.eye, t);
            point.center = glm::mix(start.center, end.center, t);
            point.up     = glm::normalize(glm::mix(start.up, end.up, t));

            m_interpolatedPath.push_back(point);
        }
    }
    m_interpolatedPath.push_back(m_keyframes.back());
}

bool CameraPathSimulator::Update(float deltaTime, nv_helpers_dx12::Manipulator& camera) {
    if (!m_isActive) return false;
    if (m_currentStepIndex >= m_interpolatedPath.size()) return true;

    if (m_isWaitingForConvergence) {
        m_currentWaitTimer += deltaTime;

        if (m_currentWaitTimer >= m_configWaitTime) {
            m_currentWaitTimer = 0.0f;
            m_currentStepIndex++;
            m_isWaitingForConvergence = false;
        }
    }
    else {
        if (m_currentStepIndex < m_interpolatedPath.size()) {
            const SimKeyframe& target = m_interpolatedPath[m_currentStepIndex];
            camera.setLookat(target.eye, target.center, target.up);
            m_isWaitingForConvergence = true;

            // Note: Since the console is closed now, this wcout goes nowhere,
            // which is fine. The visual update is the rendering itself.
            // std::wcout << ...
        } else {
            return true;
        }
    }
    return false;
}