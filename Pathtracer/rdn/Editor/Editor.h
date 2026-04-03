#pragma once
// ═══════════════════════════════════════════════════════════════════
// Editor/Editor.h
// ═══════════════════════════════════════════════════════════════════

#include "../Common.h"
#include "../Scene/Scene.h"
#include "../Camera/Camera.h"
#include "../Raytracing/PassSystem.h"
#include "../PostProcess/DLSSManager.h"

#include "../lib/imgui/imgui.h"
#include "../lib/imgui/imgui_impl_dx12.h"
#include "../lib/imgui/imgui_impl_win32.h"

class Editor {
public:
    void Init(HWND hwnd, ID3D12Device* device, UINT numFramesInFlight,
              ID3D12DescriptorHeap* srvHeap,
              D3D12_CPU_DESCRIPTOR_HANDLE fontCpuHandle,
              D3D12_GPU_DESCRIPTOR_HANDLE fontGpuHandle);
    void Shutdown();

    void Draw(Scene& scene, Camera& camera, PassSystem& passes,
              DLSSManager& dlss, ReSTIRSettings& restir, float fps);
    void Render(ID3D12GraphicsCommandList* cmdList);

    bool IsVisible() const { return m_visible; }
    void ToggleVisibility()  { m_visible = !m_visible; }

private:
    void DrawScenePanel(Scene& scene);
    void DrawCameraPanel(Camera& camera);
    void DrawPassPipelinePanel(PassSystem& passes);
    void DrawDLSSPanel(DLSSManager& dlss);
    void DrawMaterialInspector(Scene& scene);
    void DrawReSTIRPanel(ReSTIRSettings& restir);
    void DrawSunPanel(Camera& camera);

    bool m_visible        = true;
    bool m_showSun        = false;
    int  m_selectedModel  = -1;
    int  m_selectedMat    = -1;
    bool m_showMaterials  = false;

    // Cached per-model unique materials (recomputed only on selection change)
    int m_cachedMatModel = -1;
    std::vector<UINT> m_cachedUniqueMats;
};
