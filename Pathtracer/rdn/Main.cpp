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
            { "./bistro2/bistro2.obj", XMMatrixIdentity()*XMMatrixScaling(1,1,1) },
            //{ "./car/car.obj", XMMatrixIdentity()*XMMatrixScaling(1,1,1) },
        };
    }
    void Init(SceneManager& sm, Renderer& r) override {
        /*EmissiveCubes::Params p;
        p.count            = 200000;
        p.cubeSize         = 1.0f;
        p.emissiveFraction = 0.02f;
        p.emissionMin      = 2.0f;
        p.emissionMax      = 15.0f;
        p.speedMin         = 3.3f;
        p.speedMax         = 20.0f;
        p.spawnMin         = { -1500.0f, 0.2f, -1500.0f };
        p.spawnMax         = {  1500.0f, 500.0f,  1500.0f };
        p.seed             = 42u;
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
    EngineApp app(1280, 720, L"DXR Pathtracer - Engine Layer", std::move(scene));
    return Win32Application::Run(&app, hInstance, nCmdShow);
}
