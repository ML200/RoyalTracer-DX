#pragma once

#include "../Common.h"
#include "../Scene/Scene.h"
#include "../Camera/Camera.h"
#include "../Raytracing/PassSystem.h"
#include "../PostProcess/DLSSManager.h"
#include "../PostProcess/DLSSNRManager.h"
#include "../planet/stream_orchestrator.h"
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
              mc::VoxelStreamer* voxels = nullptr);
    void Render(ID3D12GraphicsCommandList* cmdList);

    bool IsVisible() const { return m_visible; }
    void ToggleVisibility() { m_visible = !m_visible; }

  private:
    void DrawScenePanel(Scene& scene);
    void DrawCameraPanel(Camera& camera, FlyCamController& flyCam);
    void DrawPassPipelinePanel(PassSystem& passes);

    void DrawDLSSPanel(Camera& camera, DLSSManager& dlss, DLSSGSettings& dlssG);

    void DrawDLSSNRPanel(DLSSNRManager& nr);

    void DrawDlssInputsPanel(IntegratorSettings& restir, DLSSManager& dlss);
    void DrawMaterialInspector(Scene& scene, Camera& camera, IntegratorSettings& restir);
    void DrawIntegratorPanel(IntegratorSettings& restir, const FrameStats& stats);
    void DrawSunPanel(Scene& scene, Camera& camera, const FrameStats& stats, mc::VoxelStreamer* voxels);
    void DrawPlanetPerfPanel(const planet::StreamOrchestrator::Stats& ps, const FrameStats& fs, float fps);

    void DrawMinecraftPanel(mc::VoxelStreamer& voxels);

    bool m_visible = true;
    bool m_showMinecraft = false;
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
    bool m_showPlanetPerf = false;
    int m_selectedModel = -1;
    int m_selectedMat = -1;

    int m_cachedMatModel = -1;
    std::vector<UINT> m_cachedUniqueMats;

    char m_matFilter[128] = {0};

  public:
    struct PlanetPerfHistory {
        static constexpr int N = 256;
        int write = 0;
        int filled = 0;
        bool paused = false;

        float frame_total_ms[N] = {};
        float frame_gpu_wait_ms[N] = {};
        float planet_cpu_ms[N] = {};
        float planet_plan_ms[N] = {};
        float planet_blas_gpu_ms[N] = {};
        float planet_tlas_gpu_ms[N] = {};
        float cells_recorded[N] = {};
        float pipe_pending[N] = {};
        float pipe_ready[N] = {};
        float pipe_blas_pending[N] = {};
        float pipe_built[N] = {};

        void push(const planet::StreamOrchestrator::Stats& ps, const FrameStats& fs);
    };

  private:
    PlanetPerfHistory m_planetHist;
};
