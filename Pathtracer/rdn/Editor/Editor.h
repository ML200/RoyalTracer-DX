#pragma once
//====================================
//EDITOR
//====================================

#include "../Common.h"
#include "../Scene/Scene.h"
#include "../Camera/Camera.h"
#include "../Raytracing/PassSystem.h"
#include "../PostProcess/DLSSManager.h"
#include "../NRC/NrcLayout.h"
#include "../../engine/Camera/FlyCamController.h"

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

    void Draw(Scene& scene, Camera& camera, FlyCamController& flyCam,
              PassSystem& passes, DLSSManager& dlss, DLSSGSettings& dlssG,
              ReSTIRSettings& restir, nrc::Settings& nrc,
              float fps, const FrameStats& stats);
    void Render(ID3D12GraphicsCommandList* cmdList);

    bool IsVisible() const { return m_visible; }
    void ToggleVisibility()  { m_visible = !m_visible; }

private:
    void DrawScenePanel(Scene& scene);
    void DrawCameraPanel(Camera& camera, FlyCamController& flyCam);
    void DrawPassPipelinePanel(PassSystem& passes);
    void DrawDLSSPanel(DLSSManager& dlss, DLSSGSettings& dlssG);
    void DrawMaterialInspector(Scene& scene, Camera& camera);
    void DrawReSTIRPanel(ReSTIRSettings& restir);
    void DrawNRCPanel(nrc::Settings& nrc);
    void DrawSunPanel(Camera& camera);

    bool m_visible        = true;
    bool m_showScene      = false;
    bool m_showCamera     = false;
    bool m_showPipeline   = false;
    bool m_showDLSS       = false;
    bool m_showReSTIR     = false;
    bool m_showNRC        = false;
    bool m_showSun        = false;
    bool m_showMaterials  = false;
    int  m_selectedModel  = -1;
    int  m_selectedMat    = -1;

    //cached per-model unique materials, recomputed on selection change
    int m_cachedMatModel = -1;
    std::vector<UINT> m_cachedUniqueMats;

    //persisted name filter, case-insensitive substring, digits match index
    char m_matFilter[128] = {0};
};
