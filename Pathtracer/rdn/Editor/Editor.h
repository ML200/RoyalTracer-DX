#pragma once

#include "../Common.h"
#include "../Scene/Scene.h"
#include "../Camera/Camera.h"
#include "../Raytracing/PassSystem.h"
#include "../PostProcess/DLSSManager.h"
#include "../PostProcess/DLSSNRManager.h"
#include "../planet/stream_orchestrator.h"
#include "../ocean/OceanSystem.h"
#include "../minecraft/voxel_streamer.h"
#include "../minecraft/mc_world.h"
#include "../../engine/Camera/FlyCamController.h"

#include "../lib/imgui/imgui.h"
#include "../lib/imgui/imgui_impl_dx12.h"
#include "../lib/imgui/imgui_impl_win32.h"

class Editor {
  public:
    void Init(HWND hwnd, ID3D12Device* device, UINT numFramesInFlight, ID3D12DescriptorHeap* srvHeap,
              D3D12_CPU_DESCRIPTOR_HANDLE fontCpuHandle, D3D12_GPU_DESCRIPTOR_HANDLE fontGpuHandle);
    void Shutdown();

    void Draw(Scene& scene, Camera& camera, FlyCamController& flyCam, PassSystem& passes, DLSSManager& dlss,
              DLSSNRManager& dlssNR, DLSSGSettings& dlssG, IntegratorSettings& integrator, float fps,
              const FrameStats& stats, const planet::StreamOrchestrator::Stats& planetStats,
              mc::VoxelStreamer* voxels = nullptr, ocean::OceanSystem* ocean = nullptr);
    void Render(ID3D12GraphicsCommandList* cmdList);
    void RenderPlatformWindows();

    bool IsVisible() const { return m_visible; }
    void ToggleVisibility() { m_visible = !m_visible; }

  private:
    void DrawScenePanel(Scene& scene);
    void DrawCameraPanel(Camera& camera, FlyCamController& flyCam);
    void DrawPassPipelinePanel(PassSystem& passes);

    void DrawDLSSPanel(Camera& camera, DLSSManager& dlss, DLSSGSettings& dlssG);

    void DrawDLSSNRPanel(DLSSNRManager& nr);

    void DrawDlssInputsPanel(IntegratorSettings& restir, DLSSManager& dlss);
    void DrawMaterialInspector(Scene& scene, Camera& camera, IntegratorSettings& restir, mc::VoxelStreamer* voxels);
    void DrawIntegratorPanel(IntegratorSettings& restir, const FrameStats& stats);
    void DrawSunPanel(Scene& scene, Camera& camera, const FrameStats& stats, mc::VoxelStreamer* voxels);
    void DrawPerformancePanel(const planet::StreamOrchestrator::Stats& ps, const FrameStats& fs, float fps,
                              const mc::StreamerStats* minecraft);

    void DrawMinecraftPanel(mc::VoxelStreamer& voxels);
    void DrawWaterPanel(ocean::OceanSystem& ocean, Scene& scene);

    bool m_visible = true;
    bool m_showMinecraft = false;
    bool m_showWater = false;
    bool m_showScene = false;
    bool m_showCamera = false;
    bool m_showPipeline = false;
    bool m_showDLSS = false;
    bool m_showDLSSNR = false;
    bool m_showDlssInputs = false;
    bool m_showIntegrator = false;
    bool m_showInactivePasses = false;
    bool m_showSun = false;
    bool m_showMaterials = false;
    bool m_showPerformance = false;
    int m_selectedModel = -1;
    int m_selectedMat = -1;

    int m_cachedMatModel = -1;
    std::vector<UINT> m_cachedUniqueMats;

    // Staged sea state; applied on widget release.
    ocean::Params m_waterParams;
    bool m_waterRespecPending = false;

    char m_matFilter[128] = {0};

  public:
    struct PerformanceHistory {
        static constexpr int N = 256;
        int write = 0;
        int filled = 0;
        bool paused = false;

        float frameCpuMs[N] = {};
        float frameGpuMs[N] = {};
        float streamingCpuMs[N] = {};

        void push(const FrameStats& fs);
    };

  private:
    struct PerformanceFrame {
        FrameStats frame;
        planet::StreamOrchestrator::Stats stream;
        mc::StreamerStats minecraft;
        float fps = 0.0f;
        bool hasMinecraft = false;
    };

    PerformanceHistory m_performanceHistory;
    PerformanceFrame m_performanceFrame;
};
