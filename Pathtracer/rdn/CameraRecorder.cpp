#include "CameraRecorder.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <ctime>
#include <cstdio>

CameraRecorder::CameraRecorder() {
}

CameraRecorder::~CameraRecorder() {
    if (!m_hasRecorded && !m_filePath.empty()) {
        // Try to delete the file
        if (std::remove(m_filePath.c_str()) == 0) {
            std::cout << "[CameraRecorder] Deleted empty recording file: " << m_filePath << std::endl;
        }
    }
}

void CameraRecorder::Initialize() {
    m_hasRecorded = false;
    m_startTime = std::chrono::high_resolution_clock::now();

    // Generate unique filename
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::tm now_tm;
    localtime_s(&now_tm, &now_c);

    std::stringstream ss;
    ss << "camera_path_"
       << std::put_time(&now_tm, "%Y-%m-%d_%H-%M-%S")
       << ".txt";

    m_filePath = ss.str();

    std::ofstream file(m_filePath);
    if (file.is_open()) {
        file << "# Timestamp | Eye(x,y,z) | Center(x,y,z) | Up(x,y,z)\n";
        file.close();
        std::cout << "[CameraRecorder] Initialized unique file: " << m_filePath << std::endl;
    }
}

void CameraRecorder::CaptureKeyframe(nv_helpers_dx12::Manipulator& camera) {
    m_hasRecorded = true;

    auto now = std::chrono::high_resolution_clock::now();
    float timestamp = std::chrono::duration<float>(now - m_startTime).count();

    glm::vec3 eye, center, up;
    camera.getLookat(eye, center, up);

    std::ofstream file(m_filePath, std::ios::app);
    if (file.is_open()) {
        file << std::fixed << std::setprecision(4)
             << timestamp << " "
             << eye.x << " " << eye.y << " " << eye.z << " "
             << center.x << " " << center.y << " " << center.z << " "
             << up.x << " " << up.y << " " << up.z << "\n";

        std::cout << "[CameraRecorder] Keyframe saved at " << timestamp << "s" << std::endl;
    }
}