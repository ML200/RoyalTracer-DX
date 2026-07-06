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

#include "planet/stream_orchestrator.h"   // Phase 4 BVH stream pipeline

//CPU-side sizing stubs matching shader SoA sizes
// Sizes mirror the HLSL SoA strides exactly. Reservoir comes from
// PLANE_HYB(72) + 12 in Reservoir_v8.hlsli (48B RECON record + 16B FW plane
// + 4B solo V2 plane + 4B wsum plane + 12B hybrid-shift HYB plane:
// seed | cachedJac | gBase). SampleData is BYTES_SD in Sample_Data_v8.hlsli.
struct Reservoir_GI  { uint8_t pad[84]; };
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
// buffer from offset 0 (8B header + SPMIS_SPLIT_MAXDRAWS*28 = 120 B/px; the
// hybrid shift added the 8B J8 sub-slot {cachedNew, Jn} per draw — see
// HashGridHash_v8.hlsli) need backing. Bump if SPMIS_SPLIT_MAXDRAWS grows.
constexpr UINT kPathStateBytesPerPx = 120;

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
    ComPtr<ID3D12Resource>       m_spmisBuffer;       // SPMIS global hash grid (root UAV u25)
    ComPtr<ID3D12Resource>       m_autoExposeBuffer;  // 32B persistent: sumLog2LumFixed, smoothedLog2Lum, isInitialized, tileCount, prevTime, _pad
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
    ComPtr<ID3D12Resource>         m_raysIndirectArgs;
    //hybrid-shift replay dispatches ([0] temporal, [1] spatial): each owns its
    //args buffer so the COMMON->COPY_DEST->INDIRECT_ARGUMENT->decay cycle never
    //collides with the raygen args buffer parked in INDIRECT_ARGUMENT.
    ComPtr<ID3D12Resource>         m_raysIndirectArgsReplay[2];
    ComPtr<ID3D12Resource>         m_raysArgsTemplate;
    ComPtr<ID3D12CommandSignature> m_raysCommandSignature;
    //SBT slot of the lite raygen variant (cloud surface shadows compiled out),
    //appended after the token-derived raygen records in CreateShaderBindingTable.
    uint32_t                       m_raygenLiteSbtSlot = UINT32_MAX;
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

    //Volumetric cloud noise — 256³ RGBA8 3D texture baked once at startup by
    //Pass_cloudnoise_bake_v8.hlsl (compute pass). The runtime cloud shader
    //(Clouds_v8.hlsli) samples this instead of evaluating Perlin/Worley
    //analytically per density tap, dropping the per-sample cost from
    //~80-200 ALU ops to a single Texture3D.SampleLevel(). NSight showed
    //the analytical path as the dominant cloud-frame cost (compute-bound,
    //not bandwidth-bound), so this is the single biggest perf lever.
    //Channel layout: R=Perlin-Worley, G=WorleyFBM, B=value (HF), A=Worley.
    //
    //The bake uses a private 1-UAV heap, root signature, and PSO because
    //it runs ONCE before CreateShaderResourceHeap; allocating a UAV slot
    //in m_srvUavHeap (which is built later, and only carries an SRV view
    //of this texture at runtime) would have required reordering init.
    ComPtr<ID3D12Resource>       m_cloudNoiseTexture;
    ComPtr<ID3D12DescriptorHeap> m_cloudNoiseBakeHeap;
    ComPtr<ID3D12RootSignature>  m_cloudNoiseBakeSig;
    ComPtr<ID3D12PipelineState>  m_cloudNoiseBakePSO;
    void BakeCloudNoiseTexture();

    //Planet-scale cloud coverage map — NASA Blue Marble "cloud_combined_8192.tif"
    //(8192×4096 equirectangular). Loaded once at startup from
    //include/cloud_coverage.tif (downloaded by CMake), converted to single-
    //channel R8 luminance, and uploaded as DXGI_FORMAT_R8_UNORM. Sampled by
    //Clouds_v8.hlsli (g_cloudCoverage at register t43) to gate the procedural
    //noise body with real-world climatology — gives continental-scale weather
    //fronts instead of uniform global coverage. Hardware bilinear filtering
    //produces the gradient between map pixels (no hard coverage edges).
    ComPtr<ID3D12Resource> m_cloudCoverageTexture;
    ComPtr<ID3D12Resource> m_cloudCoverageUploadHeap;
    void InitCloudCoverageTexture();

    //Terrain heightmap cubemap — 6-layer Texture2DArray<R32F> downsampled
    //from the baker output (CPU side keeps full bake resolution; this is
    //the GPU-friendly version the shader samples for shadows + normal
    //finite-diff + cloud bottom). Uploaded once at startup from the
    //planet::StreamOrchestrator's HeightmapCubemap (which loads from
    //./terrain/). Sampled by Includes_v8.hlsli's TerrainHeight at register
    //t45 via equiangular cubed-sphere projection.
    ComPtr<ID3D12Resource> m_terrainHeightmapTexture;
    ComPtr<ID3D12Resource> m_terrainHeightmapUploadHeap;
    void InitTerrainHeightmapTexture();

    //Baker v8 companion textures. Each is a Texture2DArray with 6 layers
    //(one per cube face). Surface_color (RGBA8) is the Mars-tint albedo
    //at t46; normal (RGBA8) is the tangent-space normal map at t47; both
    //at the same resolution as the heightmap (downsampled to GPU res from
    //the CPU bake). Cloud_offset (R32F km, 256^2) at t48 is a heavily-
    //smoothed elevation reference the cloud renderer uses to set local
    //cloud base. Uploaded once at startup from the planet::HeightmapCubemap
    //companion arrays. A missing layer leaves the resource null and the
    //SRV becomes a null fallback, keeping the legacy look intact.
    ComPtr<ID3D12Resource> m_terrainSurfaceColorTexture;
    ComPtr<ID3D12Resource> m_terrainNormalTexture;
    ComPtr<ID3D12Resource> m_terrainCloudOffsetTexture;
    ComPtr<ID3D12Resource> m_terrainCompanionUploadHeap;
    void InitTerrainSurfaceColorTexture();
    void InitTerrainNormalTexture();
    void InitTerrainCloudOffsetTexture();

    //Spatiotemporal blue noise array — 128x128x64 RGBA8 Texture2DArray
    //filled once by Pass_stbn_bake_v8.hlsl. The cloud shader's CloudRand4
    //samples this instead of evaluating a white noise hash, so the cone
    //shadow taps and per pixel step jitter carry a blue noise spatial
    //spectrum that DLSS RR's spatial filter cleanly removes. Wired up the
    //same way as the cloud noise bake (private 1 UAV heap, root sig, PSO)
    //because it runs before CreateShaderResourceHeap.
    ComPtr<ID3D12Resource>       m_cloudSTBNTexture;
    ComPtr<ID3D12DescriptorHeap> m_cloudSTBNBakeHeap;
    ComPtr<ID3D12RootSignature>  m_cloudSTBNBakeSig;
    ComPtr<ID3D12PipelineState>  m_cloudSTBNBakePSO;
    void BakeCloudSTBNTexture();

    //Per-frame sky LUTs (Pass_skylut_bake_v8.hlsl):
    //  - sun transmittance over (r, mu), 256x64 RGBA16F → t49. Replaces the
    //    per-call inner march in TransmittanceToSun.
    //  - cloud ambient probe scatter over sun-zenith cosine, 128x2 RGBA16F
    //    → t50. Replaces the two per-pixel IntegrateScattering probes in
    //    EvaluateAtmosphereAndClouds.
    //  - Hillaire 2020 multiple-scattering transfer Psi_ms over (sun-zenith
    //    cosine, altitude), 32x32 RGBA16F → t51. Real 2nd+ order air
    //    scattering; the ambient bake reads its UAV, so RecordSkyLUTBake
    //    dispatches it before mainAmbient with a UAV barrier between.
    //InitSkyLUTBake creates the textures + private root sig/heap/PSOs at
    //startup; RecordSkyLUTBake records the three dispatches at the top of
    //every PopulateCommandList, before any consumer pass. Rebaked per frame
    //because the integrals depend on live cbuffer params (turbidity, cloud
    //layer sliders); the whole bake is ~18K texels.
    ComPtr<ID3D12Resource>       m_skyTransmittanceLUT;
    ComPtr<ID3D12Resource>       m_cloudAmbientLUT;
    ComPtr<ID3D12Resource>       m_skyMultiScatterLUT;
    //Cloud->sun optical-depth shell map (Texture2DArray<R16F>, 384x384x6).
    //Baked by the mainCloudSunOD kernel; read at t52 (g_cloudSunOD) to collapse
    //the per-sample cloud self-shadow march. Unlike the sky LUTs the bake also
    //samples the cloud noise/coverage SRVs (see InitSkyLUTBake).
    ComPtr<ID3D12Resource>       m_cloudSunODLUT;
    ComPtr<ID3D12DescriptorHeap> m_skyLutBakeHeap;
    ComPtr<ID3D12RootSignature>  m_skyLutBakeSig;
    ComPtr<ID3D12PipelineState>  m_skyLutTransmittancePSO;
    ComPtr<ID3D12PipelineState>  m_skyLutAmbientPSO;
    ComPtr<ID3D12PipelineState>  m_skyLutMultiScatterPSO;
    ComPtr<ID3D12PipelineState>  m_skyLutCloudSunODPSO;
    void InitSkyLUTBake();
    void RecordSkyLUTBake(ID3D12GraphicsCommandList4* cmd);
    //Bind a 1×1 R8 fallback when the TIFF is missing or fails to load —
    //the shader formula `saturate(base * map * 2)` collapses to `base`
    //when `map = 0.5`, so a grey fallback restores the pre-coverage-map
    //behaviour visually while keeping the descriptor table populated.
    void CreateCloudCoverageFallback(uint8_t value);

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
