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
#include <comdef.h>

class BistroScene : public SceneDefinition {
public:
    std::vector<MeshDefinition> GetMeshes() override {
        return {
            { "./bistro2/bistro2.obj", XMMatrixIdentity() },
            //{ "./car/car.obj", XMMatrixIdentity() }
        };
    }
    void Init(SceneManager& sm, Renderer& r) override {
        /*EmissiveCubes::Params p;
        p.count = 5000; p.emissiveFraction = 0.4f; p.cubeSize = 0.03f;
        p.emissionMin = 0.5; p.emissionMax = 100;
        p.speedMin = 0.5f; p.speedMax = 2.5f;
        p.spawnMin = {-12, 0.3f, -8}; p.spawnMax = {12, 8, 8};
        m_cubes.Init(p, sm, r);*/
    }
    void Update(float dt, SceneManager& sm, FlyCamController& flyCam) override {
        flyCam.Update(dt);
        //m_cubes.Update(dt, sm);
    }
private:
    EmissiveCubes m_cubes;
};

class ScopedComInitializer {
public:
    ScopedComInitializer()  { m_hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED); }
    ~ScopedComInitializer() { if (SUCCEEDED(m_hr)) CoUninitialize(); }
    operator HRESULT() const { return m_hr; }
private:
    HRESULT m_hr;
};

_Use_decl_annotations_
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, int nCmdShow)
{
    ScopedComInitializer comInitializer;
    if (FAILED(comInitializer)) {
        MessageBox(nullptr, reinterpret_cast<LPCSTR>(L"Failed to initialize COM."),
                   reinterpret_cast<LPCSTR>(L"Error"), MB_OK | MB_ICONERROR);
        return 1;
    }
    if (AllocConsole()) {
        freopen("CONOUT$", "w", stdout);
        freopen("CONOUT$", "w", stderr);
        std::wcout << L"Console initialized" << std::endl;
    }
    auto scene = std::make_unique<BistroScene>();
    EngineApp app(1920, 1080, L"DXR Pathtracer - Engine Layer", std::move(scene));
    return Win32Application::Run(&app, hInstance, nCmdShow);
}
