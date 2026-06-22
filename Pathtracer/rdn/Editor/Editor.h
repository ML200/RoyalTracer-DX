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
#include "../planet/stream_orchestrator.h"
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
              float fps, const FrameStats& stats,
              const planet::StreamOrchestrator::Stats& planetStats);
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
    void DrawInitialSamplingPanel(ReSTIRSettings& restir);
    void DrawNRCPanel(nrc::Settings& nrc);
    void DrawSunPanel(Camera& camera);
    void DrawCloudPanel(Camera& camera);
    void DrawPlanetPerfPanel(const planet::StreamOrchestrator::Stats& ps,
                             const FrameStats& fs, float fps);

    bool m_visible        = true;
    bool m_showScene      = false;
    bool m_showCamera     = false;
    bool m_showPipeline   = false;
    bool m_showDLSS       = false;
    bool m_showReSTIR     = false;
    bool m_showNRC        = false;
    bool m_showInitialSampling = false;
    bool m_showSun        = false;
    bool m_showClouds     = false;
    bool m_showMaterials  = false;
    bool m_showPlanetPerf = false;
    int  m_selectedModel  = -1;
    int  m_selectedMat    = -1;

    //cached per-model unique materials, recomputed on selection change
    int m_cachedMatModel = -1;
    std::vector<UINT> m_cachedUniqueMats;

    //persisted name filter, case-insensitive substring, digits match index
    char m_matFilter[128] = {0};

public:
    //====================================
    //PLANET PERF HISTORY
    //====================================
    //Ring buffer of per-frame samples for the planet performance panel.
    //ImGui::PlotLines takes (values, count, offset) - 'offset' is the index
    //of the OLDEST sample, so the plot wraps naturally. Sized for a few
    //seconds at 60-240 fps (= 256 frames, ~1-4 seconds).
    //
    //Made public so the free plot helpers in Editor.cpp can reference 'N';
    //the member m_planetHist below stays private.
    struct PlanetPerfHistory {
        static constexpr int N = 256;
        int      write  = 0;                 // next slot to overwrite
        int      filled = 0;                 // samples pushed so far (<= N)
        bool     paused = false;              // freeze the rings for inspection

        float frame_total_ms     [N] = {};   // FrameStats::cpuFrameMs
        float frame_gpu_ms       [N] = {};   // FrameStats::gpuMs
        float planet_cpu_ms      [N] = {};   // Stats::blas_record_cpu_ms
        float planet_plan_ms     [N] = {};   // Stats::plan_ms (one shot per rebuild)
        float planet_blas_gpu_ms [N] = {};   // Stats::blas_gpu_ms
        float planet_tlas_gpu_ms [N] = {};   // Stats::tlas_gpu_ms
        float cells_recorded     [N] = {};   // Stats::cells_recorded per frame
        float pipe_pending       [N] = {};   // Stats::cells_pending
        float pipe_ready         [N] = {};   // Stats::cells_ready
        float pipe_blas_pending  [N] = {};   // Stats::cells_recorded_total
        float pipe_built         [N] = {};   // Stats::dirty_built

        void push(const planet::StreamOrchestrator::Stats& ps, const FrameStats& fs);
    };

private:
    PlanetPerfHistory m_planetHist;
};
