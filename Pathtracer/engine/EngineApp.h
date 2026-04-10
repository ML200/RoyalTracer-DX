#pragma once
#include "../rdn/DXSample.h"
#include "../rdn/Renderer.h"
#include "Input/InputManager.h"
#include "Camera/FlyCamController.h"
#include "Scene/SceneManager.h"
#include "Scene/SceneDefinition.h"
#include <memory>

class EngineApp : public DXSample {
public:
    EngineApp(UINT width, UINT height, std::wstring name,
              std::unique_ptr<SceneDefinition> scene);

    void OnInit()    override;
    void OnUpdate()  override;
    void OnRender()  override;
    void OnDestroy() override;
    void OnResize(UINT width, UINT height) override;
    void OnKeyDown(UINT8 key) override;
    void OnKeyUp(UINT8 key)   override;
    void OnButtonDown(UINT32 lParam) override;
    void OnButtonUp(UINT message, UINT32 lParam) override;
    void OnMouseMove(UINT8 wParam, UINT32 lParam) override;

    Renderer&         GetRenderer()     { return m_renderer; }
    SceneManager&     GetSceneManager() { return m_sceneManager; }
    FlyCamController& GetFlyCam()       { return m_flyCam; }

private:
    std::unique_ptr<SceneDefinition> m_sceneDef;
    Renderer         m_renderer;
    SceneManager     m_sceneManager;
    FlyCamController m_flyCam;
    std::chrono::high_resolution_clock::time_point m_prevTime;
};
