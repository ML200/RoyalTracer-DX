#include "CameraPathSimulator.h"
#include <algorithm>
#include <iomanip>
#include <Windows.h>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <filesystem>
#include <cctype>
#include <cstdint>

//GLM extensions for vector rotation
#define GLM_ENABLE_EXPERIMENTAL
#include "glm/gtx/rotate_vector.hpp"
#include "glm/gtc/constants.hpp"

static std::string ToString(const std::wstring& wstr) {
    if (wstr.empty()) return std::string();
    int size_needed = WideCharToMultiByte(CP_UTF8, 0, &wstr[0], (int)wstr.size(), NULL, 0, NULL, NULL);
    std::string strTo(size_needed, 0);
    WideCharToMultiByte(CP_UTF8, 0, &wstr[0], (int)wstr.size(), &strTo[0], size_needed, NULL, NULL);
    return strTo;
}

static std::wstring ToWString(const std::string& str) {
    if (str.empty()) return std::wstring();
    int size_needed = MultiByteToWideChar(CP_UTF8, 0, &str[0], (int)str.size(), NULL, 0);
    std::wstring wstrTo(size_needed, 0);
    MultiByteToWideChar(CP_UTF8, 0, &str[0], (int)str.size(), &wstrTo[0], size_needed);
    return wstrTo;
}

//parses trailing digit run from filename stem, safe on non-numeric files
static bool TryParseIndexFromFilename(const std::filesystem::path& p, uint64_t& outIdx)
{
    std::wstring stem = p.stem().wstring();
    if (stem.empty()) return false;

    int end = (int)stem.size() - 1;
    while (end >= 0 && !iswdigit((wint_t)stem[end])) --end;
    if (end < 0) return false;

    int start = end;
    while (start >= 0 && iswdigit((wint_t)stem[start])) --start;
    ++start;

    try {
        std::wstring numberPart = stem.substr((size_t)start, (size_t)(end - start + 1));
        outIdx = std::stoull(numberPart);
        return true;
    } catch (...) {
        return false;
    }
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
    //console for status output
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

    //load config from file
    const std::string configFileName = "sim_config.txt";
    std::ifstream configFile(configFileName);

    if (!configFile.is_open()) {
        std::wcout << L"[Info] 'sim_config.txt' not found. Defaulting to Realtime Mode.\n";
        m_isActive = false;
        std::wcout << L">> Closing Console in 2 seconds...\n";
        Sleep(2000);
        return;
    }

    std::string keyframeFileStr;
    std::string outputDirStr = "output";
    std::string line;
    std::string key;
    bool runSim = false;

    m_configMaxSteps   = 100;
    m_configWaitTime   = 0.5f;
    m_configRollSteps  = 1;
    m_configYawAngle   = 0.0f;
    m_configYawSteps   = 1;

    while (std::getline(configFile, line)) {
        if (line.empty() || line[0] == '#') continue;
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
        else if (key == "ForwardRollSteps") ss >> m_configRollSteps;
        else if (key == "UpRotationAngle") ss >> m_configYawAngle;
        else if (key == "UpRotationSteps") ss >> m_configYawSteps;
        else if (key == "OutputDir") ss >> outputDirStr;
    }
    configFile.close();

    m_outputDir = ToWString(outputDirStr);

    //apply config
    if (runSim) {
        m_isActive = true;
        std::wstring wFilename = ToWString(keyframeFileStr);

        std::wcout << L"Mode: SIMULATION\n";
        std::wcout << L"File: " << wFilename << L"\n";
        std::wcout << L"Base Steps: " << m_configMaxSteps << L"\n";
        std::wcout << L"Variations: Roll=" << m_configRollSteps
                   << L", Yaw=" << m_configYawSteps << L" (+/- " << m_configYawAngle << L" deg)\n";
        std::wcout << L"OutputDir: " << m_outputDir.wstring() << L"\n";

        LoadKeyframes(wFilename);
        GeneratePathPoints();

        //dataset extension, pick up past end of prev batch
        size_t nextDiskIndex = InferNextIndexFromOutputDir();
        m_fileIndexOffset = nextDiskIndex;

        //new scene/path, start at beginning
        m_currentStepIndex = 0;

        m_isWaitingForConvergence = false;
        m_currentWaitTimer = 0.0f;

        std::wcout << L"------------------------------------------------\n";
        std::wcout << L"[Dataset Extension Mode]\n";
        std::wcout << L"   Existing frames found on disk: " << m_fileIndexOffset << L"\n";
        std::wcout << L"   New frames to generate: " << m_interpolatedPath.size() << L"\n";
        std::wcout << L"   Filenames will range from: "
                   << m_fileIndexOffset << L" to "
                   << (m_fileIndexOffset + m_interpolatedPath.size() - 1) << L"\n";
        std::wcout << L"------------------------------------------------\n";

        std::wcout << L">> Data Generation Ready.\n";
        std::wcout << L">> Closing Console...\n";
        Sleep(2000);
    } else {
        m_isActive = false;
        std::wcout << L"Mode: REALTIME (Interactive)\n";
        std::wcout << L">> Closing Console...\n";
        Sleep(1000);
    }
}

void CameraPathSimulator::LoadKeyframes(const std::wstring& filename) {
    std::ifstream file(ToString(filename));

    if (!file.is_open()) {
        std::wcout << L"[Error] Could not open keyframe file: " << filename << L" Aborting.\n";
        m_isActive = false;
        return;
    }

    m_keyframes.clear();
    std::string line;
    std::getline(file, line);

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

    //need >=2 for interpolation
    if (m_keyframes.size() < 2 && !m_keyframes.empty()) {
        m_keyframes.push_back(m_keyframes[0]);
    }
}

void CameraPathSimulator::GeneratePathPoints() {
    if (m_keyframes.size() < 2) {
        m_interpolatedPath.clear();
        return;
    }

    m_interpolatedPath.clear();

    //total distance to normalize speed
    float totalDistance = 0.0f;
    for (size_t i = 0; i < m_keyframes.size() - 1; ++i) {
        float d = glm::distance(m_keyframes[i].eye, m_keyframes[i + 1].eye);
        m_keyframes[i].distanceToNext = d;
        totalDistance += d;
    }

    //linear interp base points
    std::vector<SimKeyframe> basePoints;

    for (size_t i = 0; i < m_keyframes.size() - 1; ++i) {
        SimKeyframe& start = m_keyframes[i];
        SimKeyframe& end = m_keyframes[i + 1];

        float segmentFraction = (totalDistance > 0.0001f) ? (start.distanceToNext / totalDistance) : 0.0f;
        int segmentSteps = static_cast<int>(std::round(segmentFraction * m_configMaxSteps));
        if (segmentSteps < 1) segmentSteps = 1;

        for (int s = 0; s < segmentSteps; ++s) {
            float t = (float)s / (float)segmentSteps;
            SimKeyframe point;
            point.eye = glm::mix(start.eye, end.eye, t);
            point.center = glm::mix(start.center, end.center, t);
            point.up = glm::normalize(glm::mix(start.up, end.up, t));
            basePoints.push_back(point);
        }
    }
    basePoints.push_back(m_keyframes.back());

    //rotational variations per base point
    for (const auto& base : basePoints) {

        glm::vec3 forward = glm::normalize(base.center - base.eye);
        float lookDist = glm::distance(base.center, base.eye);
        glm::vec3 originalUp = glm::normalize(base.up);

        //yaw, rotate FORWARD around UP
        int ySteps = (int)std::fmax(1.0f, (float)m_configYawSteps);

        for (int y = 0; y < ySteps; ++y) {
            float yawAngleDeg = 0.0f;

            if (ySteps > 1) {
                float t = (float)y / (float)(ySteps - 1);
                yawAngleDeg = glm::mix(-m_configYawAngle, m_configYawAngle, t);
            }

            glm::vec3 yawedForward = glm::rotate(forward, glm::radians(yawAngleDeg), originalUp);
            glm::vec3 yawedCenter = base.eye + (yawedForward * lookDist);

            //roll, rotate UP around yawed FORWARD
            int rSteps = (int)std::fmax(1.0f, (float)m_configRollSteps);

            for (int r = 0; r < rSteps; ++r) {
                //random roll per variation
                const float rollAngleDeg = m_rollDist(m_rng);

                glm::vec3 finalUp = glm::rotate(originalUp, glm::radians(rollAngleDeg), yawedForward);

                SimKeyframe finalFrame;
                finalFrame.eye = base.eye;
                finalFrame.center = yawedCenter;
                finalFrame.up = finalUp;
                finalFrame.distanceToNext = 0;

                m_interpolatedPath.push_back(finalFrame);
            }
        }
    }
}

size_t CameraPathSimulator::InferNextIndexFromOutputDir() const
{
    namespace fs = std::filesystem;
    std::error_code ec;

    //ensure dir, never throw
    fs::create_directories(m_outputDir, ec);
    if (ec) return 0;

    if (!fs::exists(m_outputDir, ec) || ec) return 0;
    if (!fs::is_directory(m_outputDir, ec) || ec) return 0;

    uint64_t maxIdx = 0;
    bool found = false;

    const fs::directory_options opts = fs::directory_options::skip_permission_denied;

    for (fs::directory_iterator it(m_outputDir, opts, ec), end; it != end && !ec; it.increment(ec)) {
        if (ec) break;

        if (!it->is_regular_file(ec) || ec) { ec.clear(); continue; }

        uint64_t idx = 0;
        if (!TryParseIndexFromFilename(it->path(), idx)) continue;

        if (!found || idx > maxIdx) {
            maxIdx = idx;
            found = true;
        }
    }

    return found ? (size_t)(maxIdx + 1) : 0;
}

bool CameraPathSimulator::Update(float deltaTime, nv_helpers_dx12::Manipulator& camera, bool& outShouldCapture) {
    outShouldCapture = false;

    if (!m_isActive) return false;

    //finished
    if (m_currentStepIndex >= m_interpolatedPath.size()) {
        return true;
    }

    //state machine, move -> wait -> capture -> advance
    if (!m_isWaitingForConvergence) {
        const SimKeyframe& target = m_interpolatedPath[m_currentStepIndex];
        camera.setLookat(target.eye, target.center, target.up);
        m_isWaitingForConvergence = true;
        m_currentWaitTimer = 0.0f;
        return false;
    }

    m_currentWaitTimer += deltaTime;

    if (m_currentWaitTimer >= m_configWaitTime) {
        outShouldCapture = true;
        m_lastCaptureIndex = m_currentStepIndex;

        m_currentWaitTimer = 0.0f;
        m_currentStepIndex++;
        m_isWaitingForConvergence = false;
    }

    return false;
}
