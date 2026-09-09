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
#include "PostProcess/DLSSNRManager.h"
#include <sl_reflex.h>
#include "Editor/Editor.h"
#include "../engine/Camera/FlyCamController.h"
#include "LightTree.h"
#include "Lighting/LightTreeRefit.h"
#include "CameraRecorder.h"
#include "CameraPathSimulator.h"
#include <random>
#include <array>

#include "nv_helpers_dx12/ShaderBindingTableGenerator.h"
#include "nv_helpers_dx12/TopLevelASGenerator.h"

#include "planet/stream_orchestrator.h"   // Phase 4 BVH stream pipeline

//CPU-side sizing stubs matching shader SoA sizes
// Sizes mirror the HLSL SoA strides exactly. Reservoir comes from
// PLANE_HYB(68) + 12 in Reservoir_v8.hlsli (48B RECON record + 16B FW plane
// + 4B solo V2 plane + 12B hybrid-shift HYB plane: seed | cachedJac | gBase;
// the write-only WSUM plane was removed). SampleData is BYTES_SD in
// Sample_Data_v8.hlsli.
struct Reservoir_GI  { uint8_t pad[80]; };
struct SampleData    { uint8_t pad[36]; };

// Pixel count for the per-pixel SoA buffers (reservoirs, sample data, path
// state). MapPixelID (Common_v8.hlsli) swizzles into 8-wide x 4-tall tiles,
// so linear indices run to ceil(W/8)*ceil(H/4)*32 - MORE than W*H whenever
// the resolution is not tile-divisible (e.g. 1707x960). Must stay identical
// to numPx() / ps_numPx() in Reservoir_v8.hlsli / Path_State_v8.hlsli, which
// use the same count as their SoA plane stride.
inline UINT TileAlignedPx(UINT w, UINT h) {
    return ((w + 7u) / 8u) * ((h + 3u) / 4u) * 32u;
}

// Backed bytes per pixel of the path-state buffer. The HLSL plane layout
// (Path_State_v8.hlsli) spans 176 B/px, but only the raygen-live prefix
// (PACK1..CLAS2, 72 B/px) plus the SPMIS split-pass scratch that aliases the
// buffer from offset 0 need backing: 8B header + SPMIS_TOTAL_ROLES(5) * 32B
// generic job-slot planes (startPx/resPx/prob/J8/c — see the UNIFIED SHIFT
// JOB layout in HashGridHash_v8.hlsli) = 168 B/px. Bump if SPMIS_SPLIT_MAXDRAWS
// grows (>= 8 + (SPMIS_SPLIT_MAXDRAWS+1)*32).
constexpr UINT kPathStateBytesPerPx = 168;

// Mirrors HLSL's SPMIS_SPLIT_MAXDRAWS (HashGridHash_v8.hlsli) — the host needs
// it to size the dynamic "spatial_shift" loop count (Ntn+1) each frame.
constexpr UINT kSpmisSplitMaxDraws = 4;

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
    //DLSS-NR: optional NGX neural post-process on the final composited frame.
    //Fully isolated from the Streamline DLSS-RR path above.
    DLSSNRManager       m_dlssNR;
    // Actual pre-UI display color, including NR when active, for frame generation.
    ComPtr<ID3D12Resource> m_dlssHudlessColor;
    Editor              m_editor;
    planet::StreamOrchestrator m_planet;   // Phase 4/5 BVH stream pipeline
    //MONOTONIC frame counter for the planet streaming system. m_ctx.FrameIndex()
    //is the cycling swapchain back-buffer index (0..bufferCount-1) - the chunk
    //manager ages chunks by frame number for retire hysteresis and MUST get a
    //true monotonic count, not a value that wraps every 2-3 frames.
    uint32_t m_planetFrame = 0;
    //scratch for the per-frame unified TLAS build - m_scene.tlasInstances
    //converted into the planet module's D3D12-only SceneInstanceDesc layout.
    std::vector<planet::SceneInstanceDesc> m_planetSceneInstances;
    //PLANET ROCKS: scene-mesh slots of the generated boulder variants (geometry
    //in the combined buffers + one BLAS each) and the camera-following streamer
    //that emits their per-frame instance set.
    planet::RockScatter   m_rockScatter;
    std::vector<UINT>     m_rockMeshIndices;
    FlyCamController*   m_flyCam = nullptr;
    IntegratorSettings      m_integratorSettings;
    IntegratorSettings      m_previousIntegratorSettings;
    bool                    m_integratorHistoryValid = false;
    int                m_previousIntegratorMode = -1;
    // ReSTIR lite (shaders/RestirLite_v8.hlsli): reuse-table upload into the
    // SHaRC allocation, per-frame table transforms, history validity.
    bool                   m_liteReusePending = false;
    float                  m_liteReuseSigma = 0.0f;
    ComPtr<ID3D12Resource> m_liteReuseUpload;
    std::vector<ComPtr<ID3D12Resource>> m_liteReuseRetired; // uploads possibly still in flight
    std::mt19937           m_liteRng{ 0x4c495445u };
    void BuildLiteReuseTables(float sigma);
    lt::LightTreeBuilder m_lightTree;
    lt::LightTreeRefitManager m_lightTreeRefit;
    std::vector<lt::BLASRootLocal> m_blasLocalRoots;
    std::vector<lt::LightTLASNodeGpu> m_pendingTLASUpload;
    std::vector<lt::LightTreeTrail> m_pendingBLASBitTrail;
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
    ComPtr<ID3D12Resource>       m_spmisBuffer;       // SPMIS global hash grid (root UAV u25)
    ComPtr<ID3D12Resource>       m_autoExposeBuffer;  // 128B persistent: AE state (0..19) + DLSS guide sentinel (32..63), see Includes_v8.hlsli

    //DLSS guide sentinel readback: 3-deep ring of 32 B persistently-mapped
    //readback buffers. Slot n%3 receives frame n's copy of gAutoExpose bytes
    //32..63 (recorded at Stage::DLSS); slot (n+1)%3 = frame n-2, guaranteed
    //GPU-complete under the frame fence, is decoded into m_dlss.sentinel.
    ComPtr<ID3D12Resource>       m_sentinelReadback[3];
    const uint32_t*              m_sentinelMapped[3] = { nullptr, nullptr, nullptr };
    uint64_t                     m_sentinelFrame = 0;
    // Heap slots 10/11 (root-sig u2/u3) had two extra reservoir buffers from
    // the old DI/GI split. The unified pipeline only touches the GI pair, so
    // the DI ones are gone. Heap slots stay alive via null UAV bindings to
    // preserve the contiguous u2..u7 descriptor range.
    ComPtr<ID3D12Resource>       m_reservoirBuffer_3, m_reservoirBuffer_4;
    //2-slot shader-visible sampler heap for SamplerDescriptorHeap[] in
    //SampleMaterialTex (slot 0 = s0 clone, slot 1 = s3 clone); created in
    //CreateShaderResourceHeap, bound alongside m_srvUavHeap.
    ComPtr<ID3D12DescriptorHeap> m_samplerHeap;
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
    //compacted raygen indirect dispatch: survivor queue (root UAV u26),
    //104B DISPATCH_RAYS args (default heap) + template (upload, rewritten on
    //SBT rebuild), and the DISPATCH_RAYS command signature.
    ComPtr<ID3D12Resource>         m_raygenQueueBuffer;
    ComPtr<ID3D12Resource>         m_sharcBuffer;
    ComPtr<ID3D12QueryHeap>        m_sharcTimingHeap;
    ComPtr<ID3D12Resource>         m_sharcTimingReadback;
    UINT64                       m_sharcTimestampFrequency = 0;
    UINT                         m_sharcTimingMask = 0;
    bool                         m_sharcResetPending = true;
    bool                         m_sharcWasEnabled = false;
    uint32_t                     m_sharcFrame = 0;
    int                          m_sharcCellExponent = -3;
    int                          m_sharcGuideLevel = 2;
    int                          m_sharcBounceLimit = 16;
    int                          m_sharcTextureFilter = 0;
    bool                         m_sharcLightingValid = false;
    SunSettings                  m_sharcSunSettings{};
    struct SharcInstanceState { XMMATRIX transform; UINT meshIndex; };
    std::vector<SharcInstanceState> m_sharcInstanceState;
    ComPtr<ID3D12Resource>         m_raysIndirectArgs;
    //(No hybrid-shift replay args buffer: Pass_temp_replay and
    //Pass_spmis_shift are both plain full-screen DispatchRays calls, walking
    //in place as role threads — neither uses an indirect dispatch / queue.)
    ComPtr<ID3D12Resource>         m_raysArgsTemplate;
    ComPtr<ID3D12CommandSignature> m_raysCommandSignature;
    void WriteRaysIndirectTemplate();
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

    //star / Milky Way skybox (NASA SVS 4851 Deep Star Maps EXR), sampled in
    //EvaluateStars via the celestial-frame ray direction. Loaded from
    //SKY_STARS_EXR_PATH at startup. Upload heap is retained until the post
    //init flush, then freed alongside m_lutUploadHeaps.
    ComPtr<ID3D12Resource> m_skyStarsTexture;
    ComPtr<ID3D12Resource> m_skyStarsUploadHeap;
    void InitSkyStarsTexture();

    ComPtr<ID3D12Resource> m_terrainHeightmapTexture;
    ComPtr<ID3D12Resource> m_terrainHeightmapUploadHeap;
    void InitTerrainHeightmapTexture();

    ComPtr<ID3D12Resource> m_terrainSurfaceColorTexture;
    ComPtr<ID3D12Resource> m_terrainNormalTexture;
    ComPtr<ID3D12Resource> m_terrainCompanionUploadHeap;
    void InitTerrainSurfaceColorTexture();
    void InitTerrainNormalTexture();

    //Atmospheric transmittance and multiple-scattering LUTs, baked each frame.
    ComPtr<ID3D12Resource>       m_skyTransmittanceLUT;
    ComPtr<ID3D12Resource>       m_skyMultiScatterLUT;
    ComPtr<ID3D12DescriptorHeap> m_skyLutBakeHeap;
    ComPtr<ID3D12RootSignature>  m_skyLutBakeSig;
    ComPtr<ID3D12PipelineState>  m_skyLutTransmittancePSO;
    ComPtr<ID3D12PipelineState>  m_skyLutMultiScatterPSO;
    bool m_skyLutsReady = false;
    float m_skyLutTurbidity = -1.0f;
    void InitSkyLUTBake();
    void RecordSkyLUTBake(ID3D12GraphicsCommandList4* cmd);
    ComPtr<ID3D12Resource> m_cumulusNoise, m_cumulusLight, m_cumulusEnvironment, m_cumulusAmbient, m_cumulusQueries;
    ComPtr<ID3D12Resource> m_cumulusNoiseBA; // B/A plane; R/G stays at t52.
    ComPtr<ID3D12Resource> m_cumulusDensity, m_cumulusDensityTags;
    bool m_previousDensityCacheEnabled = true;
    bool m_cumulusNoiseReady = false;
    bool m_cumulusAmbientReady = false;
    std::array<float,5> m_cumulusAmbientKey{};
    bool m_cumulusSettingsValid = false;
    CumulusSettings m_previousCumulusSettings;
    void InitCumulusResources();

    UINT m_currentDisplayLevel = 0;
    std::vector<UINT> m_displayLevels = { 0, 1, 2, 3, 4,5 };

    static constexpr UINT IMGUI_FONT_HEAP_SLOT = 999999;
    static constexpr UINT DLSS_UAV_HEAP_START  = 39;
    float m_fps = 0.0f;
    int m_dlssModeChangedFrames = 0;
    //tracks DLSSManager::clampEmitterSpikes so toggling it can drop DLSS temporal
    //history (the emitter input encoding changes, which would otherwise ghost).
    bool m_dlssClampEmitterSpikesPrev = false;
    bool m_reflexAvailable = false;
    DLSSGSettings m_dlssG;
    FrameStats m_frameStats;

    uint32_t m_time = 0;
    void PopulateCommandList();
    //adapt the engine Camera into a planet::CameraView (Phase 4/5)
    planet::CameraView MakePlanetCamera() const;
    //convert m_scene.tlasInstances into m_planetSceneInstances (Phase 5 TLAS)
    void BuildPlanetSceneInstances();
    void UploadLightTreeTLAS(ID3D12GraphicsCommandList* cmdList);
    void UploadEmissiveBuffers(ID3D12GraphicsCommandList* cmdList);
    void KickLightTreeRefit();
    std::vector<InstanceXformCPU> BuildXformsFromScene() const;
    void RebuildDLSSDescriptors();
    void RebuildResolutionDependentDescriptors();
    //rebinds NRC UAV descriptors 58-60 after reallocating InferenceIn/Out/PendingGI on resize
    void RebuildNrcDescriptors();
    //per-frame current/last sample-buffer ping-pong (replaces copySampleData)
    void SwapSampleBuffers();

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
