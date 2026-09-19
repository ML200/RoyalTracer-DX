//*********************************************************
//
// Copyright (c) Microsoft. All rights reserved.
// This code is licensed under the MIT License (MIT).
// THIS CODE IS PROVIDED *AS IS* WITHOUT WARRANTY OF
// ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING ANY
// IMPLIED WARRANTIES OF FITNESS FOR A PARTICULAR
// PURPOSE, MERCHANTABILITY, OR NON-INFRINGEMENT.
//
//*********************************************************

#include <iostream>
#include "stdafx.h"
#include "../engine/EngineApp.h"
#include "../engine/Scene/EmissiveCubes.h"
#define ENABLE_D3D12_DIAGNOSTICS 1
#include "Diagnostics.h"
#include <comdef.h>

class EnvTestHooks {
  public:
    void Init(Renderer& r) {
        char buf[64] = {};
        float hour = 0.0f;
        if (GetEnvironmentVariableA("RT_SUN_HOUR", buf, sizeof(buf)) > 0 && sscanf_s(buf, "%f", &hour) == 1) {
            r.GetCamera().sunSettings.startUTCHours = hour;
            r.GetCamera().sunSettings.simSpeed = 0.0f;
        }
        char fb[128] = {};
        if (GetEnvironmentVariableA("RT_AUTOFLY", fb, sizeof(fb)) > 0) {
            const int n = sscanf_s(fb, "%f %f %f %f", &m_flyVel.x, &m_flyVel.y, &m_flyVel.z, &m_flySeconds);
            m_fly = n >= 3;
            if (n < 4)
                m_flySeconds = 1e30f;
        }
        char jb[128] = {};
        if (GetEnvironmentVariableA("RT_JUMP", jb, sizeof(jb)) > 0)
            m_jump = sscanf_s(jb, "%f %f %f %f", &m_jumpOffset.x, &m_jumpOffset.y, &m_jumpOffset.z, &m_jumpDelay) == 4;
    }
    void Update(float dt) {
        m_elapsed += dt;
        if (m_fly && m_elapsed <= m_flySeconds) {
            glm::vec3 eye, center, up;
            nv_helpers_dx12::CameraManip.getLookat(eye, center, up);
            const glm::vec3 step = m_flyVel * std::min(dt, 0.1f);
            nv_helpers_dx12::CameraManip.setLookat(eye + step, center + step, up);
        }
        if (m_jump && m_elapsed >= m_jumpDelay) {
            m_jump = false;
            glm::vec3 eye, center, up;
            nv_helpers_dx12::CameraManip.getLookat(eye, center, up);
            nv_helpers_dx12::CameraManip.setLookat(eye + m_jumpOffset, center + m_jumpOffset, up);
            std::printf("[test hook] camera jumped by (%.0f, %.0f, %.0f)\n", m_jumpOffset.x, m_jumpOffset.y,
                        m_jumpOffset.z);
        }
    }

  private:
    bool m_fly = false;
    float m_flySeconds = 1e30f;
    glm::vec3 m_flyVel = glm::vec3(0.0f);
    bool m_jump = false;
    glm::vec3 m_jumpOffset{0.0f};
    float m_jumpDelay = 0.0f;
    float m_elapsed = 0.0f;
};

class MainScene : public SceneDefinition {
  public:
    std::vector<MeshDefinition> GetMeshes() override {
        return {
            /*{"veach-ajar.glb",
             XMMatrixIdentity(), "Modern Tank Garage"},*/
            MinecraftWorld("C:/Users/Malte/Downloads/Grand Teton National Park/Grand Teton National Park",
                           {"C:/Users/Malte/Downloads/Greenfield v0.5.4/Greenfield.Texture.Pack.1.17.zip"},
                           XMMatrixIdentity(), "Night City"),
        };
    }
    void Init(SceneManager& sm, Renderer& r) override {
        if (GetEnvironmentVariableA("RT_MC_CAMERA", nullptr, 0) == 0)
            nv_helpers_dx12::CameraManip.setLookat({0.0f, 5.0f, 12.5f}, {0.0f, 1.7f, -3.0f}, {0.0f, 1.0f, 0.0f});
        r.GetCamera().fovDegrees = 60.0f;
        m_hooks.Init(r);

        EmissiveCubes::Params cubes;
        cubes.count = 1000;
        cubes.cubeSize = 2.0f;
        cubes.emissiveFraction = 1.0f;
        cubes.emissionMin = 25.0f;
        cubes.emissionMax = 50.0f;
        cubes.speedMin = 5.3f;
        cubes.speedMax = 20.0f;
        cubes.spawnMin = {-0000.0f, 50.0f, -0000.0f};
        cubes.spawnMax = {6000.0f, 200.0f, 6000.0f};
        cubes.seed = 42;
        m_emissiveCubes.Init(cubes, sm, r);
    }
    void Update(float dt, SceneManager& sm, FlyCamController& flyCam) override {
        flyCam.Update(dt);
        m_hooks.Update(dt);
        m_emissiveCubes.Update(dt, sm);
    }

  private:
    EnvTestHooks m_hooks;
    EmissiveCubes m_emissiveCubes;
};

class ScopedComInitializer {
  public:
    ScopedComInitializer() { m_hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED); }
    ~ScopedComInitializer() {
        if (SUCCEEDED(m_hr))
            CoUninitialize();
    }
    operator HRESULT() const { return m_hr; }

  private:
    HRESULT m_hr;
};

_Use_decl_annotations_ int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, int nCmdShow) {
    dxdiag::InstallCrashHandler();
    ScopedComInitializer comInitializer;
    if (FAILED(comInitializer)) {
        MessageBoxW(nullptr, L"Failed to initialize COM.", L"Error", MB_OK | MB_ICONERROR);
        return 1;
    }
    if (AllocConsole()) {
        if (GetEnvironmentVariableA("RT_BENCHMARK_FRAMES", nullptr, 0) > 0)
            ShowWindow(GetConsoleWindow(), SW_HIDE);
        char logPath[MAX_PATH] = {};
        const DWORD logLen = GetEnvironmentVariableA("RT_LOG_FILE", logPath, MAX_PATH);
        if (logLen > 0 && logLen < MAX_PATH) {
            freopen(logPath, "w", stdout);
            freopen(logPath, "w", stderr);
            setvbuf(stdout, nullptr, _IONBF, 0);
            setvbuf(stderr, nullptr, _IONBF, 0);
        } else {
            freopen("CONOUT$", "w", stdout);
            freopen("CONOUT$", "w", stderr);
        }
        std::wcout << L"Console initialized" << std::endl;
    }
    auto scene = std::make_unique<MainScene>();
    EngineApp app(1920, 1080, L"DXR Pathtracer - Engine Layer", std::move(scene));
    return Win32Application::Run(&app, hInstance, nCmdShow);
}
