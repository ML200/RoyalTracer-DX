#pragma once
#include <string>
#include <vector>
#include <chrono>
#include <fstream>
#include "manipulator.h"

class CameraRecorder {
public:
    CameraRecorder();
    ~CameraRecorder();

    void Initialize();
    void CaptureKeyframe(nv_helpers_dx12::Manipulator& camera);

private:
    std::string m_filePath;
    std::chrono::high_resolution_clock::time_point m_startTime;
    bool m_hasRecorded = false;
};
