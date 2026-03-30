#pragma once
// ═══════════════════════════════════════════════════════════════════
// Renderer.h — Slim orchestrator. Owns the modules, wires them
//              together, and drives the frame loop. No business
//              logic lives here — it's all delegated.
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
#include "Editor/Editor.h"
#include "LightTree.h"
#include "Lighting/LightTreeRefit.h"
#include "CameraRecorder.h"
#include "CameraPathSimulator.h"

#include "nv_helpers_dx12/ShaderBindingTableGenerator.h"
#include "nv_helpers_dx12/TopLevelASGenerator.h"

class Renderer : public DXSample {
public:
    Renderer(UINT width, UINT height, std::wstring name);
    void OnInit()    override;
    void OnUpdate()  override;
    void OnRender()  override;
    void OnDestroy() override;

private:
    // ── Modules ──────────────────────────────────────────────────
    DeviceContext       m_ctx;
    Scene               m_scene;
    Camera              m_camera;
    PassSystem          m_passes;
    DLSSManager         m_dlss;
    Editor              m_editor;
    lt::LightTreeBuilder m_lightTree;
    lt::LightTreeRefitManager m_lightTreeRefit;
    std::vector<lt::BLASRootLocal> m_blasLocalRoots;
    std::vector<lt::LightTLASNodeGpu> m_pendingTLASUpload;
    std::vector<uint32_t> m_pendingBLASToItem;
    ComPtr<ID3D12Resource> m_tlasUploadStaging;
    ComPtr<ID3D12Resource> m_blasToItemUploadStaging;
    ComPtr<ID3D12Resource> m_ltTlasGpu;
    ComPtr<ID3D12Resource> m_ltBtIGpu;
    UINT m_ltTlasGpuCapacity  = 0;
    UINT m_ltBtIGpuCapacity   = 0;
    static constexpr UINT LT_TLAS_SRV_SLOT        = 20;
    static constexpr UINT LT_BLASTOITEM_SRV_SLOT   = 27;

    // Emissive triangle GPU re-upload (when emission values change)
    bool m_emissiveGpuDirty = false;
    ComPtr<ID3D12Resource> m_emissiveUploadStaging;
    ComPtr<ID3D12Resource> m_triToLightIdUploadStaging;
    ComPtr<ID3D12Resource> m_ownedEmissiveGpu;       // replaces scene's after first re-upload
    ComPtr<ID3D12Resource> m_ownedTriToLightIdGpu;
    UINT m_emissiveGpuCapacity     = 0;
    UINT m_triToLightIdGpuCapacity = 0;
    static constexpr UINT EMISSIVE_TRI_SRV_SLOT    = 9;
    static constexpr UINT TRI_TO_LIGHTID_SRV_SLOT  = 24;
    CameraRecorder      m_recorder;
    CameraPathSimulator m_simulator;

    // ── Input ────────────────────────────────────────────────────
    bool g_keys[256] = {};
    void OnKeyDown(UINT8 key) override;
    void OnKeyUp(UINT8 key)   override;
    void OnButtonDown(UINT32 lParam);
    void OnMouseMove(UINT8 wParam, UINT32 lParam);

    // ── Acceleration structures ──────────────────────────────────
    struct AccelerationStructureBuffers {
        ComPtr<ID3D12Resource> pScratch, pResult, pResultUncompacted, pInstanceDesc;
    };
    nv_helpers_dx12::TopLevelASGenerator m_topLevelASGenerator;
    AccelerationStructureBuffers m_topLevelASBuffers;

    AccelerationStructureBuffers CreateBottomLevelAS(
        std::vector<std::pair<ComPtr<ID3D12Resource>, uint32_t>> vb,
        std::vector<std::pair<ComPtr<ID3D12Resource>, uint32_t>> ib,
        UINT opaqueTriCount, UINT alphaTriCount);
    void CreateTopLevelAS(
        const std::vector<std::pair<ComPtr<ID3D12Resource>, XMMATRIX>>& instances,
        bool updateOnly = false);
    void CreateAccelerationStructures();

    // ── Pipeline ─────────────────────────────────────────────────
    ComPtr<ID3D12RootSignature>       m_rayGenSignature, m_computeSignature;
    ComPtr<ID3D12RootSignature>       m_hitSignature, m_missSignature;
    ComPtr<ID3D12StateObject>         m_rtStateObject;
    ComPtr<ID3D12StateObjectProperties> m_rtStateObjectProps;
    std::vector<ComPtr<ID3D12PipelineState>> m_csPSOs;
    std::vector<std::wstring>         m_callableShaderNames;
    std::vector<ComPtr<IDxcBlob>>     m_rayGenLibs;
    nv_helpers_dx12::ShaderBindingTableGenerator m_sbtHelper;
    ComPtr<ID3D12Resource>            m_sbtStorage;

    // Work graphs
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

    // ── Render targets & UAV heap ────────────────────────────────
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

    // ── Streaming compaction / indirect dispatch ─────────────────
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

    // ── LUT textures ─────────────────────────────────────────────
    ComPtr<ID3D12Resource> m_lutTextureArray;
    std::vector<ComPtr<ID3D12Resource>> m_lutUploadHeaps;
    void GenerateLutTextures();
    void CreateAndUploadLutArray(const std::vector<std::vector<float>>& data,
                                ComPtr<ID3D12Resource>& target, const std::wstring& name);

    // ── Display ──────────────────────────────────────────────────
    UINT m_currentDisplayLevel = 0;
    std::vector<UINT> m_displayLevels = { 0, 1, 2, 3 };

    // ── Editor ───────────────────────────────────────────────────
    static constexpr UINT IMGUI_FONT_HEAP_SLOT = 999999;  // last slot in the 1M heap
    float m_fps = 0.0f;

    // ── Frame dispatch ───────────────────────────────────────────
    uint32_t m_time = 0;
    void PopulateCommandList();
    void UploadLightTreeTLAS(ID3D12GraphicsCommandList* cmdList);
    void UploadEmissiveBuffers(ID3D12GraphicsCommandList* cmdList);
    void KickLightTreeRefit();
    std::vector<InstanceXformCPU> BuildXformsFromScene() const;

    // ── Readback / simulation ────────────────────────────────────
    ComPtr<ID3D12Resource> m_readbackBuffer;
    void CreateReadbackBuffer();
    void SaveSimulationData(uint32_t stepIndex);
};
