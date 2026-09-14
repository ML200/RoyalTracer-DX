#pragma once

#include "Common.h"
#include "DXSample.h"

#include "Core/DeviceContext.h"
#include "Core/ResourceFactory.h"
#include "Core/GpuProfiler.h"
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
#include "Lighting/LightTreeIncremental.h"
#include "CameraRecorder.h"
#include "CameraPathSimulator.h"
#include <random>
#include <array>

#include "nv_helpers_dx12/ShaderBindingTableGenerator.h"
#include "nv_helpers_dx12/TopLevelASGenerator.h"

#include "planet/stream_orchestrator.h"
#include "minecraft/mc_config.h"
#include "minecraft/mc_omm.h"
#include "Scene/OmmBuilder.h"
#include "minecraft/mc_world.h"
#include "minecraft/voxel_streamer.h"

// Byte strides shared with the shader SoA layouts.
struct Reservoir_GI {
    uint8_t pad[80];
};
struct SampleData {
    uint8_t pad[36];
};

// Includes padding for the shader's 8x4 pixel tiles.
inline UINT TileAlignedPx(UINT w, UINT h) {
    return ((w + 7u) / 8u) * ((h + 3u) / 4u) * 32u;
}

constexpr UINT kSpmisSplitMaxDraws = 4;
// Path-state storage also backs the SPMIS job planes.
constexpr UINT kPathStateBytesPerPx = 8 + (kSpmisSplitMaxDraws + 1) * 32;

class Renderer {
  public:
    Renderer(UINT width, UINT height);

    void InitDevice();
    void LoadScene(const std::vector<ModelEntry>& models);

    bool LoadMinecraftWorld(const mc::MinecraftWorldConfig& cfg, const DirectX::XMMATRIX& placement);
    void InitSceneGPU();
    void UpdateRenderer(float dt);
    void RenderFrame();
    void DestroyRenderer();
    void OnResize(UINT newWidth, UINT newHeight);

    Scene& GetScene() { return m_scene; }
    Camera& GetCamera() { return m_camera; }
    DeviceContext& GetContext() { return m_ctx; }
    void SetFlyCam(FlyCamController* fc) { m_flyCam = fc; }
    UINT GetWidth() const { return m_width; }
    UINT GetHeight() const { return m_height; }
    float GetAspectRatio() const { return m_aspectRatio; }

    UINT CreateProceduralMesh(const std::vector<Vertex>& vertices, const std::vector<UINT>& indices,
                              const Material& material);

    UINT CreateMeshInstance(UINT sourceMeshIndex, const Material& material);
    void HandleSceneStructuralChange();

    bool WantsKeyboard() const;
    bool WantsMouse() const;
    void HandleKeyUp(UINT8 key);

  private:
    UINT m_width;
    UINT m_height;
    float m_aspectRatio;

    DeviceContext m_ctx;
    Scene m_scene;
    Camera m_camera;
    PassSystem m_passes;
    DLSSManager m_dlss;

    DLSSNRManager m_dlssNR;

    ComPtr<ID3D12Resource> m_dlssHudlessColor;
    Editor m_editor;
    planet::StreamOrchestrator m_planet;

    uint32_t m_planetFrame = 0; // Monotonic, independent of the swapchain index.

    std::vector<planet::SceneInstanceDesc> m_planetSceneInstances;

    planet::RockScatter m_rockScatter;
    std::vector<UINT> m_rockMeshIndices;

    // Workers must stop before their world is destroyed.
    std::unique_ptr<mc::World> m_mcWorld;
    mc::VoxelStreamer m_voxels;
    mc::MinecraftWorldConfig m_mcConfig;
    FlyCamController* m_flyCam = nullptr;
    IntegratorSettings m_integratorSettings;
    IntegratorSettings m_previousIntegratorSettings;
    bool m_integratorHistoryValid = false;
    int m_previousIntegratorMode = -1;

    bool m_liteReusePending = false;
    float m_liteReuseSigma = 0.0f;
    ComPtr<ID3D12Resource> m_liteReuseUpload;
    std::vector<ComPtr<ID3D12Resource>> m_liteReuseRetired; // uploads possibly still in flight
    std::mt19937 m_liteRng{0x4c495445u};
    void BuildLiteReuseTables(float sigma);
    bool m_lightLearningResetPending = true;

    bool m_lightLearningRevalidatePending = false;
    lt::IncrementalTLAS m_liveLightTlas;
    bool m_lightTlasForceRebuild = true; // after a scene light-tree rebuild: start the persistent tree over
    bool m_pendingTlasIncremental = false;
    bool m_lightLearningWasEnabled = false;
    int m_lightLearningCellExponent = 0;
    float m_lightLearningLodScale = 0.05f;
    lt::LightTreeBuilder m_lightTree;
    lt::LightTreeRefitManager m_lightTreeRefit;
    std::vector<lt::BLASRootLocal> m_blasLocalRoots;
    std::vector<lt::LightTLASNodeGpu> m_pendingTLASUpload;
    std::vector<lt::LightTLASNodeGpu> m_publishedLightTLAS;

    std::vector<lt::LightTreeTrail> m_pendingBLASBitTrail;
    std::vector<lt::LightSlotGpu> m_pendingSlotRecords;
    ComPtr<ID3D12Resource> m_tlasUploadStaging;
    ComPtr<ID3D12Resource> m_blasBitTrailUploadStaging;
    ComPtr<ID3D12Resource> m_slotUploadStaging;
    ComPtr<ID3D12Resource> m_ltTlasGpu;
    ComPtr<ID3D12Resource> m_ltBlasBitTrailGpu;
    ComPtr<ID3D12Resource> m_ltSlotGpu;
    UINT m_ltTlasGpuCapacity = 0;
    UINT m_ltBlasBitTrailGpuCapacity = 0;
    UINT m_ltSlotGpuCapacity = 0;
    static constexpr UINT LT_TLAS_SRV_SLOT = 20;
    static constexpr UINT LT_SLOT_SRV_SLOT = 16;
    static constexpr UINT LT_BLASBITTRAIL_SRV_SLOT = 27;

    void BindVoxelLights(ID3D12GraphicsCommandList* cmdList);
    bool MeshLightsActive() const { return m_scene.HasMeshLights() || m_liveVoxelLightLeaves > 0; }
    ComPtr<ID3D12Resource> m_vxLightRecords, m_vxLightNodes, m_vxLightLeaf, m_vxLightTrails;

    OmmGpuData m_vxOmm;
    mc::OmmTable m_vxOmmTable;
    ComPtr<ID3D12Resource> m_vxOmmIndices;

    bool m_lightClassApplied[Scene::LightClassCount] = {true, true};
    struct RetiredResource {
        ComPtr<ID3D12Resource> res;
        uint64_t frame;
    };
    std::vector<RetiredResource> m_vxLightRetired;
    UINT m_vxSceneRecordCap = 0, m_vxSceneNodeCap = 0;
    uint32_t m_voxelLightKickedVersion = 0xFFFFFFFFu;
    uint32_t m_pendingVoxelLightVersion = 0, m_pendingVoxelLeafCount = 0;
    uint32_t m_liveVoxelLightLeaves = 0;
    double m_lastVoxelLightKick = 0.0;

    bool m_emissiveGpuDirty = false;
    ComPtr<ID3D12Resource> m_emissiveUploadStaging;
    ComPtr<ID3D12Resource> m_triToLightIdUploadStaging;
    ComPtr<ID3D12Resource> m_ownedEmissiveGpu;
    ComPtr<ID3D12Resource> m_ownedTriToLightIdGpu;
    UINT m_emissiveGpuCapacity = 0;
    UINT m_triToLightIdGpuCapacity = 0;
    static constexpr UINT EMISSIVE_TRI_SRV_SLOT = 9;
    static constexpr UINT TRI_TO_LIGHTID_SRV_SLOT = 24;
    static constexpr UINT INSTANCE_PROPS_SRV_SLOT = 6;
    CameraRecorder m_recorder;
    CameraPathSimulator m_simulator;

    struct AccelerationStructureBuffers {
        ComPtr<ID3D12Resource> pScratch, pResult, pResultUncompacted, pInstanceDesc;
    };
    nv_helpers_dx12::TopLevelASGenerator m_topLevelASGenerator;
    AccelerationStructureBuffers m_topLevelASBuffers;

    AccelerationStructureBuffers CreateBottomLevelAS(const std::vector<std::pair<ComPtr<ID3D12Resource>, uint32_t>>& vb,
                                                     const std::vector<std::pair<ComPtr<ID3D12Resource>, uint32_t>>& ib,
                                                     UINT opaqueTriCount, UINT alphaTriCount,
                                                     MeshGPU* meshOmm = nullptr);
    void CreateTopLevelAS(const std::vector<Scene::TLASInstance>& instances, bool updateOnly = false);
    void CreateAccelerationStructures();

    ComPtr<ID3D12RootSignature> m_rayGenSignature, m_computeSignature;
    ComPtr<ID3D12RootSignature> m_hitSignature, m_missSignature;
    ComPtr<ID3D12StateObject> m_rtStateObject;
    ComPtr<ID3D12StateObjectProperties> m_rtStateObjectProps;
    std::vector<ComPtr<ID3D12PipelineState>> m_csPSOs;
    std::vector<std::wstring> m_callableShaderNames;
    nv_helpers_dx12::ShaderBindingTableGenerator m_sbtHelper;
    ComPtr<ID3D12Resource> m_sbtStorage;

    struct WgRuntimeData {
        D3D12_PROGRAM_IDENTIFIER id;
        D3D12_GPU_VIRTUAL_ADDRESS_RANGE backing;
        ComPtr<ID3D12Resource> backingRes;
    };
    std::vector<WgRuntimeData> m_wgRuntime;
    std::vector<ComPtr<ID3D12StateObject>> m_wgStateObjects;
    std::vector<ComPtr<ID3D12WorkGraphProperties>> m_wgProps;

    ComPtr<ID3D12RootSignature> CreateRayGenSignature();
    ComPtr<ID3D12RootSignature> CreateComputeSignature();
    ComPtr<ID3D12RootSignature> CreateHitSignature();
    ComPtr<ID3D12RootSignature> CreateMissSignature();
    void CreateRaytracingPipeline();
    void CreateShaderBindingTable();

    ComPtr<ID3D12Resource> m_outputResource;
    ComPtr<ID3D12Resource> m_permanentDataTexture;
    ComPtr<ID3D12Resource> m_scratchPing;
    ComPtr<ID3D12Resource> m_pathStateBuffer;
    ComPtr<ID3D12Resource> m_spmisBuffer;
    ComPtr<ID3D12Resource> m_autoExposeBuffer;

    ComPtr<ID3D12Resource> m_sentinelReadback[3];
    const uint32_t* m_sentinelMapped[3] = {nullptr, nullptr, nullptr};
    uint64_t m_sentinelFrame = 0;

    ComPtr<ID3D12Resource> m_reservoirBuffer_3, m_reservoirBuffer_4;

    ComPtr<ID3D12DescriptorHeap> m_samplerHeap;
    ComPtr<ID3D12Resource> m_sampleBuffer_current, m_sampleBuffer_last;
    ComPtr<ID3D12DescriptorHeap> m_srvUavHeap;
    ComPtr<ID3D12DescriptorHeap> m_stagingUavHeap;

    void CreateRaytracingOutputBuffer();
    void CreateShaderResourceHeap();
    void CreatePathStateBuffer();

    ComPtr<ID3D12Resource> m_stackBuffers[MAX_STACKS];
    ComPtr<ID3D12Resource> m_globalCounterBuffer;
    ComPtr<ID3D12Resource> m_indirectArgsBuffer;
    ComPtr<ID3D12Resource> m_zeroBuffer;
    ComPtr<ID3D12CommandSignature> m_commandSignature;

    ComPtr<ID3D12Resource> m_raygenQueueBuffer;
    ComPtr<ID3D12Resource> m_sharcBuffer;
    GpuProfiler m_gpuProfiler;
    bool m_sharcResetPending = true;
    bool m_sharcWasEnabled = false;
    uint32_t m_sharcFrame = 0;
    int m_sharcCellExponent = -3;
    int m_sharcGuideLevel = 2;
    int m_sharcBounceLimit = 16;
    int m_sharcTextureFilter = 0;
    bool m_sharcLightingValid = false;
    SunSettings m_sharcSunSettings{};
    struct SharcInstanceState {
        XMMATRIX transform;
        UINT meshIndex;
    };
    std::vector<SharcInstanceState> m_sharcInstanceState;
    ComPtr<ID3D12Resource> m_raysIndirectArgs;

    ComPtr<ID3D12Resource> m_raysArgsTemplate;
    ComPtr<ID3D12CommandSignature> m_raysCommandSignature;
    void WriteRaysIndirectTemplate();
    ComPtr<ID3D12PipelineState> m_psoSetupIndirect, m_psoSetupIndirectNoClear;
    ComPtr<ID3D12RootSignature> m_rsSetupIndirect;

    ComPtr<ID3D12Resource> m_sortCountBuffer, m_sortOffsetBuffer, m_sortBoundsBuffer;
    ComPtr<ID3D12Resource> m_sortBoundsResetBuffer;
    D3D12_GPU_DESCRIPTOR_HANDLE m_sortCountGpuHandle{}, m_sortOffsetGpuHandle{}, m_sortBoundsGpuHandle{};
    D3D12_CPU_DESCRIPTOR_HANDLE m_sortCountCpuHandle{}, m_sortOffsetCpuHandle{}, m_sortBoundsCpuHandle{};

    void CreateStreamingCompactionBuffers();
    void CreateIndirectCommandSignature();
    void CompileSetupIndirectShader();
    void ClearSortBuffers(ID3D12GraphicsCommandList* cmdList);

    ComPtr<ID3D12Resource> m_lutTextureArray;
    std::vector<ComPtr<ID3D12Resource>> m_lutUploadHeaps;
    void GenerateLutTextures();
    void CreateAndUploadLutArray(const std::vector<std::vector<float>>& data, ComPtr<ID3D12Resource>& target,
                                 const std::wstring& name);

    ComPtr<ID3D12Resource> m_skyStarsTexture;
    ComPtr<ID3D12Resource> m_skyStarsUploadHeap;
    void InitSkyStarsTexture();

    ComPtr<ID3D12Resource> m_blueNoiseTexture;
    ComPtr<ID3D12Resource> m_blueNoiseUploadHeap;
    void InitBlueNoiseTexture();

    ComPtr<ID3D12Resource> m_terrainHeightmapTexture;
    ComPtr<ID3D12Resource> m_terrainHeightmapUploadHeap;
    void InitTerrainHeightmapTexture();

    ComPtr<ID3D12Resource> m_terrainSurfaceColorTexture;
    ComPtr<ID3D12Resource> m_terrainNormalTexture;
    ComPtr<ID3D12Resource> m_terrainCompanionUploadHeap;
    void InitTerrainSurfaceColorTexture();
    void InitTerrainNormalTexture();

    ComPtr<ID3D12Resource> m_skyTransmittanceLUT;
    ComPtr<ID3D12Resource> m_skyMultiScatterLUT;
    ComPtr<ID3D12DescriptorHeap> m_skyLutBakeHeap;
    ComPtr<ID3D12RootSignature> m_skyLutBakeSig;
    ComPtr<ID3D12PipelineState> m_skyLutTransmittancePSO;
    ComPtr<ID3D12PipelineState> m_skyLutMultiScatterPSO;
    bool m_skyLutsReady = false;
    float m_skyLutTurbidity = -1.0f;
    void InitSkyLUTBake();
    void RecordSkyLUTBake(ID3D12GraphicsCommandList4* cmd);
    ComPtr<ID3D12Resource> m_cumulusNoise, m_cumulusLight, m_cumulusEnvironment, m_cumulusAmbient, m_cumulusQueries;
    ComPtr<ID3D12Resource> m_cumulusNoiseBA;
    ComPtr<ID3D12Resource> m_cumulusDensity, m_cumulusDensityTags;
    bool m_previousDensityCacheEnabled = true;
    bool m_cumulusNoiseReady = false;
    bool m_cumulusAmbientReady = false;
    std::array<float, 5> m_cumulusAmbientKey{};
    bool m_cumulusSettingsValid = false;
    CumulusSettings m_previousCumulusSettings;
    void InitCumulusResources();

    UINT m_currentDisplayLevel = 0;
    std::vector<UINT> m_displayLevels = {0, 1, 2, 4, 5};

    static constexpr UINT IMGUI_FONT_HEAP_SLOT = 999999;
    static constexpr UINT DLSS_UAV_HEAP_START = 39;
    float m_fps = 0.0f;
    int m_dlssModeChangedFrames = 0;

    bool m_dlssClampEmitterSpikesPrev = false;
    bool m_reflexAvailable = false;
    DLSSGSettings m_dlssG;
    FrameStats m_frameStats;

    uint32_t m_time = 0;
    void PopulateCommandList();

    planet::CameraView MakePlanetCamera() const;

    void BuildPlanetSceneInstances();
    void UploadLightTreeTLAS(ID3D12GraphicsCommandList* cmdList);
    void UploadEmissiveBuffers(ID3D12GraphicsCommandList* cmdList);
    void KickLightTreeRefit();
    std::vector<InstanceXformCPU> BuildXformsFromScene() const;
    void RebuildDLSSDescriptors();
    void RebuildResolutionDependentDescriptors();

    void SwapSampleBuffers();

    ComPtr<ID3D12Resource> m_readbackBuffer;
    void CreateReadbackBuffer();
    void SaveSimulationData(uint32_t stepIndex);
};
