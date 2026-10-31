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
#include "../shaders/PathStateLayout.h"
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
#include <chrono>
#include <random>
#include <array>

#include "nv_helpers_dx12/ShaderBindingTableGenerator.h"
#include "nv_helpers_dx12/TopLevelASGenerator.h"

#include "planet/stream_orchestrator.h"
#include "ocean/OceanSystem.h"
#include "minecraft/mc_config.h"
#include "minecraft/mc_omm.h"
#include "Scene/OmmBuilder.h"
#include "Scene/HostStream.h"
#include "minecraft/mc_world.h"
#include "minecraft/voxel_streamer.h"
#include "Clusters/ClusterLod.h"
#include "Clusters/LodGpu.h"
#include "Clusters/InstanceSets.h"

// Byte strides shared with the shader SoA layouts.
struct SampleData {
    uint8_t pad[36];
};

// Includes padding for the shader's 8x4 pixel tiles.
inline UINT TileAlignedPx(UINT w, UINT h) {
    return ((w + 7u) / 8u) * ((h + 3u) / 4u) * 32u;
}

constexpr UINT kPathStateBytesPerPx = PS_PATH_STATE_BYTES;

// How the renderer is hosted. The standalone app owns a window and lets the renderer create the device; a host
// process (the Unity plugin) creates the DeviceContext headless, keeps it across renderers and reads HostOutputs.
struct RendererConfig {
    UINT width = 0;
    UINT height = 0;
    bool headless = false;
};

class Renderer {
  public:
    Renderer(DeviceContext& ctx, const RendererConfig& config);

    void InitDevice(HWND hwnd = nullptr);
    void LoadScene(const std::vector<ModelEntry>& models);

    bool LoadMinecraftWorld(const mc::MinecraftWorldConfig& cfg, const DirectX::XMMATRIX& placement);
    void InitSceneGPU();
    void UpdateRenderer(float dt);
    void RenderFrame();
    void DestroyRenderer();
    void OnResize(UINT newWidth, UINT newHeight);

    Scene& GetScene() { return m_scene; }
    Camera& GetCamera() { return m_camera; }
    ocean::OceanSystem& GetOcean() { return m_ocean; }
    DeviceContext& GetContext() { return m_ctx; }
    const DLSSManager& GetDLSS() const { return m_dlss; }
    void SetFlyCam(FlyCamController* fc) { m_flyCam = fc; }
    UINT GetWidth() const { return m_width; }
    UINT GetHeight() const { return m_height; }

    UINT CreateProceduralMesh(const std::vector<Vertex>& vertices, const std::vector<UINT>& indices,
                              const Material& material);

    UINT CreateMeshInstance(UINT sourceMeshIndex, const Material& material);
    void HandleSceneStructuralChange();

    // Headless: where this frame's results go (shaders/HostOutputLayout.h). RGBA16F color and R32F depth at display
    // size, UAV-capable with simultaneous access; null discards. Call between UpdateRenderer and RenderFrame.
    void SetHostOutputs(ID3D12Resource* color, ID3D12Resource* depth);
    // The host's sky map (HOST_SKY_SRV_SLOT), read while Camera::hostSkyScale > 0; null clears it.
    void SetHostSky(ID3D12Resource* sky);
    void SetDlssMode(sl::DLSSMode mode) { m_dlss.mode = mode; } // applied by the next UpdateRenderer
    // Before InitSceneGPU; takes the voxel world's slot, so not together with a Minecraft world.
    void AttachHostStream(IHostStream* stream) { m_host = stream; }

    bool WantsKeyboard() const;
    bool WantsMouse() const;
    void HandleKeyUp(UINT8 key);

  private:
    UINT m_width;
    UINT m_height;
    float m_aspectRatio;
    bool m_headless;
    HWND m_hwnd = nullptr;

    DeviceContext& m_ctx;
    Scene m_scene;
    Camera m_camera;
    PassSystem m_passes;
    DLSSManager m_dlss;

    DLSSNRManager m_dlssNR;

    ComPtr<ID3D12Resource> m_dlssHudlessColor;
    Editor m_editor;
    // Before its clients (the streams below), which return their clusters on destruction.
    clusters::ClusterGeometry m_clusters;
    planet::StreamOrchestrator m_planet;
    ocean::OceanSystem m_ocean;
    float m_lastDt = 0.0f;

    uint32_t m_planetFrame = 0; // Monotonic, independent of the swapchain index.

    std::vector<planet::SceneInstanceDesc> m_planetSceneInstances;

#if RDN_CLUSTER_API
    // Per-instance cuts of the scene meshes' cluster LOD hierarchies, chosen and built first in each frame's cluster
    // window; an instance with a cut traces its cluster BLAS in place of its mesh's. Paged meshes' cuts are chosen on the
    // GPU after the TLAS build when the TLAS is partitioned (RDN_LOD_GPU=0: on the CPU).
    struct LodClient final : planet::IClusterClient {
        Renderer* renderer = nullptr;
        void before_cluster_work() override {
            if (renderer->m_pager.active()) renderer->m_pager.before_begin();
            if (renderer->m_lodGpuMode) {
                renderer->m_lodFeedback = {};
                renderer->m_lodGpu.collect(renderer->m_lodFeedback);
            }
            renderer->m_instanceSets.collect();
        }
        void queue_cluster_work(ID3D12GraphicsCommandList4*) override { renderer->UpdateSceneLod(); }
        void after_tlas(ID3D12GraphicsCommandList4* cl, planet::TlasBuilder& tlas) override {
            renderer->RecordSceneLodGpu(cl, tlas);
            renderer->RecordInstanceSets(cl, tlas);
        }
        void on_submitted(uint64_t computeFence) override {
            if (renderer->m_pager.active()) renderer->m_pager.on_submitted(computeFence);
            if (renderer->m_lodGpuMode) renderer->m_lodGpu.on_submitted(computeFence);
            renderer->m_instanceSets.on_submitted(computeFence);
        }
    };
    // Mass instancing: sets of one mesh at many transforms, which the GPU writes into the TLAS. RDN_SCATTER builds a
    // test forest from a scene object.
    clusters::InstanceSets m_instanceSets;
    std::vector<UINT> m_scatterSources;              // its scene instances
    std::vector<std::pair<UINT, UINT>> m_setRecords; // set slot, mesh
    const void* m_setRecordBuffer = nullptr;          // the instance-properties buffer the records are in
    std::chrono::steady_clock::time_point m_setStart; // the wind's clock
    void CreateScatterSets(uint32_t count);
    void RecordInstanceSets(ID3D12GraphicsCommandList4* cl, planet::TlasBuilder& tlas);
    clusters::LodInstances m_lod;
    clusters::LodGpu m_lodGpu;
    bool m_lodGpuMode = false;
    clusters::LodGpu::Feedback m_lodFeedback;
    std::vector<UINT> m_lodGpuInstance;  // scene instance per GPU LOD instance
    std::vector<UINT> m_lodGpuHitGroup;  // their hit groups as of the last reset
    std::vector<float> m_lodGpuViews;    // per GPU LOD instance: camera in mesh space, near distance
    float m_lodErrorScale = 1.0f;        // the adaptive error budget's factor
    float m_lodPixelsPerUnit = 0.0f, m_lodThreshold = 1.0f;
    // Streams the paged meshes' hierarchy pages the cuts want.
    clusters::Pager m_pager;
    std::vector<UINT> m_pagedMeshes; // scene mesh per pager mesh
    std::vector<clusters::LodRequest> m_lodRequests;
    void LoadPagedRoots();
    LodClient m_lodClient;
    std::vector<UINT> m_lodMeshOf;        // per instance, as of the last reset
    const MeshGPU* m_lodMeshes = nullptr; // the mesh array then
    std::vector<clusters::LodView> m_lodViews;
    void UpdateSceneLod();
    void RecordSceneLodGpu(ID3D12GraphicsCommandList4* cl, planet::TlasBuilder& tlas);
#endif

    planet::RockScatter m_rockScatter;
    std::vector<UINT> m_rockMeshIndices;

    // Workers must stop before their world is destroyed.
    std::unique_ptr<mc::World> m_mcWorld;
    mc::VoxelStreamer m_voxels;
    IHostStream* m_host = nullptr;
    // Streamed emissive geometry: the voxel world's or the host's (they share the external slot).
    IDynamicLights* DynamicLights() { return m_voxels.enabled() ? static_cast<IDynamicLights*>(&m_voxels) : m_host; }
    mc::MinecraftWorldConfig m_mcConfig;
    FlyCamController* m_flyCam = nullptr;
    IntegratorSettings m_integratorSettings;
    IntegratorSettings m_previousIntegratorSettings;
    bool m_integratorHistoryValid = false;

    bool m_liteReusePending = false;
    float m_liteReuseSigma = 0.0f;
    ComPtr<ID3D12Resource> m_liteReuseUpload;
    std::vector<ComPtr<ID3D12Resource>> m_liteReuseRetired; // uploads possibly still in flight
    std::mt19937 m_liteRng{0x4c495445u};
    void BuildLiteReuseTables(float sigma);
    bool m_lightLearningResetPending = true;

    bool m_lightLearningRevalidatePending = false;
    lt::IncrementalTLAS m_liveLightTlas;
    bool m_lightTlasForceRebuild = true; // restart after a light-tree rebuild
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
    // Cluster BLAS for meshes with BLAS build inputs (and micromaps, if any), synchronously.
    void BuildClusterBlas(const std::vector<MeshGPU*>& meshes);
    void CreateTopLevelAS(const std::vector<Scene::TLASInstance>& instances, bool updateOnly = false);
    void CreateAccelerationStructures();

    ComPtr<ID3D12RootSignature> m_rayGenSignature, m_computeSignature;
    ComPtr<ID3D12RootSignature> m_hitSignature, m_missSignature;
    ComPtr<ID3D12StateObject> m_rtStateObject;
    ComPtr<ID3D12StateObjectProperties> m_rtStateObjectProps;
    std::vector<ComPtr<ID3D12PipelineState>> m_csPSOs;
    nv_helpers_dx12::ShaderBindingTableGenerator m_sbtHelper;
    ComPtr<ID3D12Resource> m_sbtStorage;


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
    ComPtr<ID3D12Resource> m_skyBakeBuffer;
    ComPtr<ID3D12Resource> m_autoExposeBuffer;

    ComPtr<ID3D12Resource> m_sentinelReadback[3];
    const uint32_t* m_sentinelMapped[3] = {nullptr, nullptr, nullptr};
    uint64_t m_sentinelFrame = 0;

    ComPtr<ID3D12Resource> m_liteReservoirs;

    ComPtr<ID3D12DescriptorHeap> m_samplerHeap;
    ComPtr<ID3D12Resource> m_sampleBuffer_current, m_sampleBuffer_last;
    ComPtr<ID3D12DescriptorHeap> m_srvUavHeap;
    ComPtr<ID3D12DescriptorHeap> m_stagingUavHeap;

    void CreateRaytracingOutputBuffer();
    void CreateShaderResourceHeap();
    void CreatePathStateBuffer();


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

    // t60 placeholder (one unused record) without clustered geometry.
    ComPtr<ID3D12Resource> m_clusterRecords;
    ComPtr<ID3D12Resource> m_setMotionNone;
    uint32_t m_setFrame = 0; // stamps the instance sets' motion records, uploaded with the camera

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
    bool m_ptWaterShaders = false; // the pipeline holds the path tracer's water shaders (scene has a sea)
    float m_skyLutTurbidity = -1.0f;
    void InitSkyLUTBake();
    void RecordSkyLUTBake(ID3D12GraphicsCommandList4* cmd);

    UINT m_currentDisplayLevel = 0;
    std::vector<UINT> m_displayLevels = {0, 1, 2, 4, 5};

    static constexpr UINT IMGUI_FONT_HEAP_SLOT = 999999;
    static constexpr UINT DLSS_UAV_HEAP_START = 33;
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
