#include "../rdn/stdafx.h"
#include "EngineApp.h"
#include "../rdn/Win32Application.h"
#include "Windowsx.h"

EngineApp::EngineApp(UINT width, UINT height, std::wstring name,
                     std::unique_ptr<SceneDefinition> scene)
    : DXSample(width, height, name), m_sceneDef(std::move(scene)), m_renderer(width, height) {}

void EngineApp::OnInit() {
    m_renderer.InitDevice();

    auto meshes = m_sceneDef->GetMeshes();
    std::vector<ModelEntry> models;
    for (auto& m : meshes) models.push_back({ m.path, m.transform, m.name });
    m_renderer.LoadScene(models);

    m_sceneDef->Init(m_sceneManager, m_renderer);
    m_sceneManager.SyncToRendererInitial(m_renderer.GetScene());
    m_renderer.InitSceneGPU();

    m_renderer.SetFlyCam(&m_flyCam);
    m_flyCam.SetCamera(&m_renderer.GetCamera());

    ThrowIfFailed(m_renderer.GetContext().CmdList()->Close());
    m_prevTime = std::chrono::high_resolution_clock::now();
}

void EngineApp::OnUpdate() {
    auto now = std::chrono::high_resolution_clock::now();
    float dt = std::chrono::duration<float>(now - m_prevTime).count();
    m_prevTime = now;

    InputManager::BeginFrame();
    m_sceneDef->Update(dt, m_sceneManager, m_flyCam);
    m_sceneManager.SyncToRenderer(m_renderer);
    m_renderer.UpdateRenderer(dt);
}

void EngineApp::OnRender()  { m_renderer.RenderFrame(); }
void EngineApp::OnDestroy() { m_renderer.DestroyRenderer(); }

void EngineApp::OnResize(UINT width, UINT height) {
    m_width       = width;
    m_height      = height;
    m_aspectRatio = static_cast<float>(width) / static_cast<float>(height);
    m_renderer.OnResize(width, height);
}

void EngineApp::OnKeyDown(UINT8 key) {
    if (!m_renderer.WantsKeyboard()) InputManager::OnKeyDown(key);
}
void EngineApp::OnKeyUp(UINT8 key) {
    InputManager::OnKeyUp(key);
    m_renderer.HandleKeyUp(key);
}
void EngineApp::OnButtonDown(UINT32 lParam) {
    if (m_renderer.WantsMouse()) return;
    if (GetAsyncKeyState(VK_LBUTTON) & 0x8000) InputManager::OnMouseButtonDown(0);
    if (GetAsyncKeyState(VK_RBUTTON) & 0x8000) InputManager::OnMouseButtonDown(1);
    if (GetAsyncKeyState(VK_MBUTTON) & 0x8000) InputManager::OnMouseButtonDown(2);
}
void EngineApp::OnButtonUp(UINT message, UINT32 lParam) {
    if (message == WM_LBUTTONUP) InputManager::OnMouseButtonUp(0);
    if (message == WM_RBUTTONUP) InputManager::OnMouseButtonUp(1);
    if (message == WM_MBUTTONUP) InputManager::OnMouseButtonUp(2);
}
void EngineApp::OnMouseMove(UINT8 wParam, UINT32 lParam) {
    int x = GET_X_LPARAM(lParam), y = GET_Y_LPARAM(lParam);
    InputManager::OnMouseMove(x, y);
    if (m_renderer.WantsMouse()) return;
    bool lmb = wParam & MK_LBUTTON, rmb = wParam & MK_RBUTTON, mmb = wParam & MK_MBUTTON;
    if (lmb) InputManager::OnMouseButtonDown(0); else InputManager::OnMouseButtonUp(0);
    if (rmb) InputManager::OnMouseButtonDown(1); else InputManager::OnMouseButtonUp(1);
    if (mmb) InputManager::OnMouseButtonDown(2); else InputManager::OnMouseButtonUp(2);
}
