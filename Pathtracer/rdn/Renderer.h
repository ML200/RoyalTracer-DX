#pragma once
// ═══════════════════════════════════════════════════════════════════
// Renderer.h — Low-level DXR rendering module. Owned by EngineApp.
// ═══════════════════════════════════════════════════════════════════

#include "Common.h"
#include "DXSample.h"

#include "Core/DeviceContext.h"
#include "Core/ResourceFactory.h"
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

// CPU-side sizing stubs matching shader-side SoA layout sizes
struct Reservoir_DI  { uint8_t pad[100]; };
struct Reservoir_GI  { uint8_t pad[100]; };
struct SampleData    { uint8_t pad[100]; };
struct InitialBSDFRay{ uint8_t pad[100]; };

class Renderer {
public:
    Renderer(UINT width, UINT height);

    // ── Public API (called by EngineApp) ────────────────────────
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

    // Create a new procedural mesh with its own BLAS. Call between LoadScene/InitSceneGPU.
    UINT CreateProceduralMesh(const std::vector<Vertex>& vertices,
                              const std::vector<UINT>& indices,
                              const Material& material);

    // Create a mesh entry that shares geometry (BLAS/VB/IB) with an existing mesh
    // but has its own material. Much cheaper than CreateProceduralMesh.
    UINT CreateMeshInstance(UINT sourceMeshIndex, const Material& material);
    void HandleSceneStructuralChange();

    bool WantsKeyboard() const;
    bool WantsMouse() const;
    void HandleKeyUp(UINT8 key);

private:
    UINT  m_width;
    UINT  m_height;
    float m_aspectRatio;

    // ── Modules ──────────────────────────────────────────────────
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
    std::vector<uint32_t> m_pendingBLASToItem;
    std::vector<lt::BlasRangeGpu> m_pendingBLASRanges;
    ComPtr<ID3D12Resource> m_tlasUploadStaging;
    ComPtr<ID3D12Resource> m_blasToItemUploadStaging;
    ComPtr<ID3D12Resource> m_rangesUploadStaging;
    ComPtr<ID3D12Resource> m_ltTlasGpu;
    ComPtr<ID3D12Resource> m_ltBtIGpu;
    ComPtr<ID3D12Resource> m_ltRangesGpu;
    UINT m_ltTlasGpuCapacity   = 0;
    UINT m_ltBtIGpuCapacity    = 0;
    UINT m_ltRangesGpuCapacity = 0;
    static constexpr UINT LT_TLAS_SRV_SLOT        = 20;
    static constexpr UINT LT_BLASRANGES_SRV_SLOT   = 22;
    static constexpr UINT LT_BLASTOITEM_SRV_SLOT   = 27;

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

    // AS
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
        const std::vector<std::pair<ComPtr<ID3D12Resource>, XMMATRIX>>& instances,
        bool updateOnly = false);
    void CreateAccelerationStructures();

    ComPtr<ID3D12RootSignature>       m_rayGenSignature, m_computeSignature;
    ComPtr<ID3D12RootSignature>       m_hitSignature, m_missSignature;
    ComPtr<ID3D12StateObject>         m_rtStateObject;
    ComPtr<ID3D12StateObjectProperties> m_rtStateObjectProps;
    std::vector<ComPtr<ID3D12PipelineState>> m_csPSOs;
    std::vector<std::wstring>         m_callableShaderNames;
    std::vector<ComPtr<IDxcBlob>>     m_rayGenLibs;
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
    ComPtr<ID3D12Resource>       m_reservoirBuffer, m_reservoirBuffer_2;
    ComPtr<ID3D12Resource>       m_reservoirBuffer_3, m_reservoirBuffer_4;
    ComPtr<ID3D12Resource>       m_sampleBuffer_current, m_sampleBuffer_last;
    ComPtr<ID3D12Resource>       m_initialBSDFRayBuffer;
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

    // Neural Path Guiding (NPG/NASG) buffers — u24-u28
    ComPtr<ID3D12Resource>         m_npgWeights;      // u24: float[3269] - NASG MLP weights+bias (13076 bytes)
    ComPtr<ID3D12Resource>         m_npgGradients;    // u25: float[3269] - accumulated gradients
    ComPtr<ID3D12Resource>         m_npgAdamM;        // u26: float[3269] - Adam first moment
    ComPtr<ID3D12Resource>         m_npgAdamV;        // u27: float[3269] - Adam second moment
    ComPtr<ID3D12Resource>         m_npgCounters;     // u28: uint[8]     - sample count + adam_t
    bool                           m_npgInitialized = false;

    void CreateStreamingCompactionBuffers();
    void CreateIndirectCommandSignature();
    void CompileSetupIndirectShader();
    void ClearSortBuffers(ID3D12GraphicsCommandList* cmdList);

    ComPtr<ID3D12Resource> m_lutTextureArray;
    std::vector<ComPtr<ID3D12Resource>> m_lutUploadHeaps;
    void GenerateLutTextures();
    void CreateAndUploadLutArray(const std::vector<std::vector<float>>& data,
                                ComPtr<ID3D12Resource>& target, const std::wstring& name);

    UINT m_currentDisplayLevel = 0;
    std::vector<UINT> m_displayLevels = { 0, 1, 2, 3 }; // noisy, dlss, accumulated, debug

    static constexpr UINT IMGUI_FONT_HEAP_SLOT = 999999;
    static constexpr UINT DLSS_UAV_HEAP_START  = 39;
    float m_fps = 0.0f;
    int m_dlssModeChangedFrames = 0;  // >0 = skip temporal reuse for this many frames
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

    ComPtr<ID3D12Resource> m_readbackBuffer;
    void CreateReadbackBuffer();
    void SaveSimulationData(uint32_t stepIndex);
};
