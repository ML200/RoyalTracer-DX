#pragma once
//====================================
//LOW-LEVEL DXR RENDERER
//====================================
//owned by EngineApp

#include "Common.h"
#include "DXSample.h"

#include "Core/DeviceContext.h"
#include "Core/ResourceFactory.h"
#include "Interop/CudaInterop.h"
#include "NRC/NrcNetwork.h"
#include "Scene/Scene.h"
#include "Scene/AssetLoader.h"
#include "Camera/Camera.h"
#include "Raytracing/PassSystem.h"
#include "PostProcess/DLSSManager.h"
#include <sl_reflex.h>
#include "Editor/Editor.h"
#include "../engine/Camera/FlyCamController.h"
#include "LightTree.h"
#include "Lighting/LightTreeRefit.h"
#include "CameraRecorder.h"
#include "CameraPathSimulator.h"

#include "nv_helpers_dx12/ShaderBindingTableGenerator.h"
#include "nv_helpers_dx12/TopLevelASGenerator.h"

//CPU-side sizing stubs matching shader SoA sizes
// Sizes mirror the HLSL SoA strides exactly. Reservoir comes from
// PLANE_WSUM(60) + SZ_4(4) in Reservoir_v8.hlsli. SampleData is BYTES_SD
// in Sample_Data_v8.hlsli.
struct Reservoir_GI  { uint8_t pad[64]; };
struct SampleData    { uint8_t pad[24]; };

class Renderer {
public:
    Renderer(UINT width, UINT height);

    //====================================
    //PUBLIC API
    //====================================
    void InitDevice();
    void LoadScene(const std::vector<ModelEntry>& models);
    void InitSceneGPU();
    void UpdateRenderer(float dt);
    void RenderFrame();
    void DestroyRenderer();
    void OnResize(UINT newWidth, UINT newHeight);

    Scene&         GetScene()       { return m_scene; }
    Camera&        GetCamera()      { return m_camera; }
    DeviceContext& GetContext()      { return m_ctx; }
    void           SetFlyCam(FlyCamController* fc) { m_flyCam = fc; }
    UINT           GetWidth() const { return m_width; }
    UINT           GetHeight() const{ return m_height; }
    float          GetAspectRatio() const { return m_aspectRatio; }

    //new procedural mesh with its own BLAS, call between LoadScene and InitSceneGPU
    UINT CreateProceduralMesh(const std::vector<Vertex>& vertices,
                              const std::vector<UINT>& indices,
                              const Material& material);

    //mesh sharing geometry with existing mesh, own material, cheap
    UINT CreateMeshInstance(UINT sourceMeshIndex, const Material& material);
    void HandleSceneStructuralChange();

    bool WantsKeyboard() const;
    bool WantsMouse() const;
    void HandleKeyUp(UINT8 key);

    //====================================
    //CUDA INTEROP
    //====================================
    //callback for L"cuda:<name>" pass entries, runs on CudaInterop stream, fence-gated
    //shouldRun is checked by the dispatcher BEFORE the D3D12 cmd list close/execute/
    //fence/wait/reopen cycle -- when it returns false the round-trip is skipped
    //entirely, saving the WDDM cross-context overhead even when the op would no-op.
    using CudaOpFn   = std::function<void()>;
    using CudaOpPred = std::function<bool()>;
    struct CudaOp {
        CudaOpFn   fn;
        CudaOpPred shouldRun = [] { return true; };
    };
    void RegisterCudaOp(const std::wstring& name, CudaOpFn fn) {
        m_cudaOps[name] = CudaOp{ std::move(fn), [] { return true; } };
    }
    void RegisterCudaOp(const std::wstring& name, CudaOpFn fn, CudaOpPred shouldRun) {
        m_cudaOps[name] = CudaOp{ std::move(fn), std::move(shouldRun) };
    }
    CudaInterop& GetCudaInterop() { return m_cudaInterop; }

    //NRC runtime toggles, read by raygen/debug via push constants
    nrc::Settings& GetNrcSettings() { return m_nrcSettings; }
    bool           IsNrcReady() const { return m_nrcReady; }

private:
    UINT  m_width;
    UINT  m_height;
    float m_aspectRatio;

    //====================================
    //MODULES
    //====================================
    DeviceContext       m_ctx;
    Scene               m_scene;
    Camera              m_camera;
    PassSystem          m_passes;
    DLSSManager         m_dlss;
    Editor              m_editor;
    FlyCamController*   m_flyCam = nullptr;
    ReSTIRSettings      m_restirSettings;
    lt::LightTreeBuilder m_lightTree;
    lt::LightTreeRefitManager m_lightTreeRefit;
    std::vector<lt::BLASRootLocal> m_blasLocalRoots;
    std::vector<lt::LightTLASNodeGpu> m_pendingTLASUpload;
    std::vector<uint32_t> m_pendingBLASBitTrail;
    std::vector<lt::BlasRangeGpu> m_pendingBLASRanges;
    ComPtr<ID3D12Resource> m_tlasUploadStaging;
    ComPtr<ID3D12Resource> m_blasBitTrailUploadStaging;
    ComPtr<ID3D12Resource> m_rangesUploadStaging;
    ComPtr<ID3D12Resource> m_ltTlasGpu;
    ComPtr<ID3D12Resource> m_ltBlasBitTrailGpu;
    ComPtr<ID3D12Resource> m_ltRangesGpu;
    UINT m_ltTlasGpuCapacity         = 0;
    UINT m_ltBlasBitTrailGpuCapacity = 0;
    UINT m_ltRangesGpuCapacity       = 0;
    static constexpr UINT LT_TLAS_SRV_SLOT          = 20;
    static constexpr UINT LT_BLASRANGES_SRV_SLOT    = 22;
    static constexpr UINT LT_BLASBITTRAIL_SRV_SLOT  = 27;

    bool m_emissiveGpuDirty = false;
    ComPtr<ID3D12Resource> m_emissiveUploadStaging;
    ComPtr<ID3D12Resource> m_triToLightIdUploadStaging;
    ComPtr<ID3D12Resource> m_ownedEmissiveGpu;
    ComPtr<ID3D12Resource> m_ownedTriToLightIdGpu;
    UINT m_emissiveGpuCapacity     = 0;
    UINT m_triToLightIdGpuCapacity = 0;
    static constexpr UINT EMISSIVE_TRI_SRV_SLOT    = 9;
    static constexpr UINT TRI_TO_LIGHTID_SRV_SLOT  = 24;
    static constexpr UINT INSTANCE_PROPS_SRV_SLOT  = 6;
    CameraRecorder      m_recorder;
    CameraPathSimulator m_simulator;

    //====================================
    //ACCELERATION STRUCTURES
    //====================================
    struct AccelerationStructureBuffers {
        ComPtr<ID3D12Resource> pScratch, pResult, pResultUncompacted, pInstanceDesc;
    };
    nv_helpers_dx12::TopLevelASGenerator m_topLevelASGenerator;
    AccelerationStructureBuffers m_topLevelASBuffers;

    AccelerationStructureBuffers CreateBottomLevelAS(
        std::vector<std::pair<ComPtr<ID3D12Resource>, uint32_t>> vb,
        std::vector<std::pair<ComPtr<ID3D12Resource>, uint32_t>> ib,
        UINT opaqueTriCount, UINT alphaTriCount,
        MeshGPU* meshOmm = nullptr);
    void CreateTopLevelAS(
        const std::vector<Scene::TLASInstance>& instances,
        bool updateOnly = false);
    void CreateAccelerationStructures();

    ComPtr<ID3D12RootSignature>       m_rayGenSignature, m_computeSignature;
    ComPtr<ID3D12RootSignature>       m_hitSignature, m_missSignature;
    ComPtr<ID3D12StateObject>         m_rtStateObject;
    ComPtr<ID3D12StateObjectProperties> m_rtStateObjectProps;
    std::vector<ComPtr<ID3D12PipelineState>> m_csPSOs;
    std::vector<std::wstring>         m_callableShaderNames;
    nv_helpers_dx12::ShaderBindingTableGenerator m_sbtHelper;
    ComPtr<ID3D12Resource>            m_sbtStorage;

    struct WgRuntimeData { D3D12_PROGRAM_IDENTIFIER id; D3D12_GPU_VIRTUAL_ADDRESS_RANGE backing; ComPtr<ID3D12Resource> backingRes; };
    std::vector<WgRuntimeData>        m_wgRuntime;
    std::vector<ComPtr<ID3D12StateObject>> m_wgStateObjects;
    std::vector<ComPtr<ID3D12WorkGraphProperties>> m_wgProps;

    ComPtr<ID3D12RootSignature> CreateRayGenSignature();
    ComPtr<ID3D12RootSignature> CreateComputeSignature();
    ComPtr<ID3D12RootSignature> CreateHitSignature();
    ComPtr<ID3D12RootSignature> CreateMissSignature();
    void CreateRaytracingPipeline();
    void CreateShaderBindingTable();

    ComPtr<ID3D12Resource>       m_outputResource;
    ComPtr<ID3D12Resource>       m_permanentDataTexture;
    ComPtr<ID3D12Resource>       m_scratchPing;
    ComPtr<ID3D12Resource>       m_pathStateBuffer;
    ComPtr<ID3D12Resource>       m_autoExposeBuffer;  // 32B persistent: sumLog2LumFixed, smoothedLog2Lum, isInitialized, tileCount, prevTime, _pad
    // Heap slots 10/11 (root-sig u2/u3) had two extra reservoir buffers from
    // the old DI/GI split. The unified pipeline only touches the GI pair, so
    // the DI ones are gone. Heap slots stay alive via null UAV bindings to
    // preserve the contiguous u2..u7 descriptor range.
    ComPtr<ID3D12Resource>       m_reservoirBuffer_3, m_reservoirBuffer_4;
    ComPtr<ID3D12Resource>       m_sampleBuffer_current, m_sampleBuffer_last;
    ComPtr<ID3D12DescriptorHeap> m_srvUavHeap;
    ComPtr<ID3D12DescriptorHeap> m_stagingUavHeap;

    void CreateRaytracingOutputBuffer();
    void CreateShaderResourceHeap();
    void CreatePathStateBuffer();

    ComPtr<ID3D12Resource>         m_stackBuffers[MAX_STACKS];
    ComPtr<ID3D12Resource>         m_globalCounterBuffer;
    ComPtr<ID3D12Resource>         m_indirectArgsBuffer;
    ComPtr<ID3D12Resource>         m_zeroBuffer;
    ComPtr<ID3D12CommandSignature> m_commandSignature;
    ComPtr<ID3D12PipelineState>    m_psoSetupIndirect, m_psoSetupIndirectNoClear;
    ComPtr<ID3D12RootSignature>    m_rsSetupIndirect;

    ComPtr<ID3D12Resource>         m_sortCountBuffer, m_sortOffsetBuffer, m_sortBoundsBuffer;
    ComPtr<ID3D12Resource>         m_sortBoundsResetBuffer;
    D3D12_GPU_DESCRIPTOR_HANDLE    m_sortCountGpuHandle{}, m_sortOffsetGpuHandle{}, m_sortBoundsGpuHandle{};
    D3D12_CPU_DESCRIPTOR_HANDLE    m_sortCountCpuHandle{}, m_sortOffsetCpuHandle{}, m_sortBoundsCpuHandle{};

    void CreateStreamingCompactionBuffers();
    void CreateIndirectCommandSignature();
    void CompileSetupIndirectShader();
    void ClearSortBuffers(ID3D12GraphicsCommandList* cmdList);

    ComPtr<ID3D12Resource> m_lutTextureArray;
    std::vector<ComPtr<ID3D12Resource>> m_lutUploadHeaps;
    void GenerateLutTextures();
    void CreateAndUploadLutArray(const std::vector<std::vector<float>>& data,
                                ComPtr<ID3D12Resource>& target, const std::wstring& name);

    //paired spatial reuse textures (Lin et al. 2026)
    ComPtr<ID3D12Resource> m_reuseTexture[3];
    std::vector<ComPtr<ID3D12Resource>> m_reuseTextureUploadHeaps;
    void InitReuseTextures();

    UINT m_currentDisplayLevel = 0;
    std::vector<UINT> m_displayLevels = { 0, 1, 2, 3, 4,5 };

    static constexpr UINT IMGUI_FONT_HEAP_SLOT = 999999;
    static constexpr UINT DLSS_UAV_HEAP_START  = 39;
    float m_fps = 0.0f;
    int m_dlssModeChangedFrames = 0;
    bool m_reflexAvailable = false;
    DLSSGSettings m_dlssG;
    FrameStats m_frameStats;

    uint32_t m_time = 0;
    void PopulateCommandList();
    void UploadLightTreeTLAS(ID3D12GraphicsCommandList* cmdList);
    void UploadEmissiveBuffers(ID3D12GraphicsCommandList* cmdList);
    void KickLightTreeRefit();
    std::vector<InstanceXformCPU> BuildXformsFromScene() const;
    void RebuildDLSSDescriptors();
    void RebuildResolutionDependentDescriptors();
    //rebinds NRC UAV descriptors 58-60 after reallocating InferenceIn/Out/PendingGI on resize
    void RebuildNrcDescriptors();

    ComPtr<ID3D12Resource> m_readbackBuffer;
    void CreateReadbackBuffer();
    void SaveSimulationData(uint32_t stepIndex);

    //====================================
    //CUDA INTEROP STATE
    //====================================
    CudaInterop                              m_cudaInterop;
    CudaInterop::Fence                       m_cudaFence;
    UINT64                                   m_cudaFenceValue = 0;
    std::unordered_map<std::wstring, CudaOp> m_cudaOps;

    //====================================
    //NRC STATE
    //====================================
    //shared D3D12/CUDA buffers, layout in rdn/NRC/NrcLayout.h
    //m_nrcReady gates every NRC pass, false on interop/tcnn init failure
    nrc::Network                             m_nrcNetwork;
    CudaInterop::Buffer                      m_nrcInferenceIn;
    CudaInterop::Buffer                      m_nrcInferenceOut;
    CudaInterop::Buffer                      m_nrcPendingGI;
    CudaInterop::Buffer                      m_nrcTrainRecords;
    CudaInterop::Buffer                      m_nrcCounters;
    uint32_t                                 m_nrcInferenceCapacity = 0;
    //dynamic per-frame cap on inference slots. Each frame the renderer reads
    //the prior frame's actual counter via async readback and shrinks this to
    //AlignBatch(prev*5/4 + safetyFloor), capped at the static buffer capacity.
    //Pushed to raygen as nrc_inference_capacity (slot 27) and used to size the
    //CUDA inference dispatch. Initialised to and reset on resize/reinit to the
    //full buffer capacity so the first frame after each event has headroom.
    uint32_t                                 m_nrcDynamicInferenceCap = 0;
    bool                                     m_nrcReady = false;
    nrc::Settings                            m_nrcSettings{};
    //adaptive training tile, updated each frame from LastValidVertexCount, packed into nrc_flags bits 8..15
    uint32_t                                 m_nrcTrainTileSide = nrc::kInitialTrainingTileSide;
    //resolution-scaled training records target. Recomputed each frame from screen
    //size so per-cell sample density stays consistent across resolutions. Fed to
    //both the adaptive tile feedback (as target) and the fill kernel (as cap).
    //Initialised to the legacy 1080p fixed target so a trainer lambda that ever
    //fires before the first frame tick has a sane cap to use.
    uint32_t                                 m_nrcTrainRecordsTarget =
        nrc::kTrainingBatchSize * nrc::kTrainingBatchesPerFrame;
    //weight-collapse auto-reinit state. Counter ticks up while LastInferenceOutMagnitudeMean
    //stays below threshold; cooldown gates the detector for a short window after a reinit
    //so the freshly seeded network has time to learn before the canary can fire again.
    uint32_t                                 m_nrcCollapseConsecutive = 0u;
    uint32_t                                 m_nrcReinitCooldown      = 0u;
};
