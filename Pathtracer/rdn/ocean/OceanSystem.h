#pragma once

#include "OceanCommon.h"
#include "OceanQuadtree.h"
#include <fstream>
#include "../planet/stream_orchestrator.h"

struct DeviceContext;

namespace ocean {

// Recorded on the streaming compute queue, which graphics waits on before tracing.
class OceanSystem : public planet::IExternalStream {
  public:
    ~OceanSystem() override = default;

    void Configure(const Params& p);

    // Optional water mask; null covers the whole extent. Must outlive the ocean.
    void SetCoverage(const ICoverage* coverage) { m_coverage = coverage; }
    const Params& GetParams() const { return m_params; }
    bool Enabled() const { return m_params.enabled; }

    // A host's simulation in place of the FFT (Params::external): per band a texture on this device (RGBA16F, height
    // in x, readable without transitions) and its mapping, one band per cascade. No bands hides the ocean. Call before
    // BeginFrame.
    struct ExternalBand {
        ID3D12Resource* texture = nullptr;
        float invPatch = 0.0f;              // 1 / patch length, m
        float offsetU = 0.0f, offsetV = 0.0f; // texture offset at the frame's time
        float amplitude = 0.0f;
        float fadeA = 0.0f, fadeB = 1.0f;     // distance fade, saturate(d * fadeA + fadeB)
    };
    struct External {
        uint32_t bands = 0;
        uint32_t resolution = 0; // texture size
        double seaLevel = 0.0;   // world y of the still surface
        float maxHeight = 0.0f;  // bound on |height|
        ExternalBand band[OCEAN_CASCADES];
    };
    void SetExternal(const External& e);

    // A host's overlays on the surface, from two atlases on this device (readable without transitions): extra height
    // (x, times heightScale) and foam (x whitecaps, y bubbles, as optical thickness, times foamScale) over host-world
    // rectangles. Call before BeginFrame.
    struct Overlay {
        float originX = 0.0f, originZ = 0.0f; // the rectangle's corner at u = v = 0, host world (the renderer's -z)
        float sizeX = 0.0f, sizeZ = 0.0f;
        float height[4] = {}; // texel region (x, y, w, h) spanning the rectangle; w 0: no height
        float foam[4] = {};
        float heightScale = 0.0f, foamScale = 0.0f;
        float maxHeight = 0.0f; // bound on |height| after the scale
        float foamMax = 0.0f;   // cap on the whitecap cover, 0: none
    };
    struct Overlays {
        ID3D12Resource* height = nullptr;
        ID3D12Resource* foam = nullptr;
        uint32_t count = 0;
        Overlay overlay[OCEAN_OVERLAYS];
    };
    void SetOverlays(const Overlays& o);

    bool Visible() const { return m_params.enabled && (!m_params.external || m_external.bands > 0); }

    // Space needed in the scene's shared buffers; query before creating them.
    struct Reservation {
        uint32_t vertexElems = 0;
        uint32_t indexElems = 0;
        uint32_t matIDElems = 0;
        uint32_t instanceSlots = 0;
    };
    Reservation GetReservation() const;

    // Clustered tiles (CLAS rebuilt every frame, cluster BLAS over them) when geometry is enabled.
    void Init(ID3D12Device5* device, DeviceContext* ctx, clusters::ClusterGeometry* geometry);

    // propsBase: first instance-property record.
    void BindScene(ID3D12Resource* globalVertex, ID3D12Resource* globalIndex, uint32_t vertexBase,
                   uint32_t indexBase, uint32_t materialBase, uint32_t propsBase, uint32_t materialIndex);

    void CreateDescriptors(ID3D12Device* device, ID3D12DescriptorHeap* heap);

    // Selects tiles and stages uploads; call before submit_work.
    void BeginFrame(float dt, const planet::CameraView& cam, uint32_t frameIndex);

    // planet::IExternalStream
    uint32_t instance_capacity() const override;
    void record_gpu_work(ID3D12GraphicsCommandList* copyList, ID3D12GraphicsCommandList4* computeList) override;
    void append_instances(planet::TlasBuilder& tlas, InstanceProperties* props, const planet::DVec3& sceneOrigin,
                          uint32_t hitGroup, bool& forceRebuild, bool& forceRefit) override;
    void on_submitted(uint64_t copyFence, uint64_t computeFence) override;

    // Clear dielectric with volume absorption and single scattering.
    static Material MakeMaterial(const Params& p);

    static constexpr uint32_t kGpuStages = 4;
    struct Stats {
        uint32_t tiles = 0;
        uint32_t leaves = 0;
        uint32_t dropped = 0;
        uint32_t builds = 0;
        uint32_t refits = 0;
        uint64_t triangles = 0;
        uint64_t blasBytes = 0;
        uint64_t resourceBytes = 0;
        // Short-wave horizontal gain (ShortWaveChop) and the resulting Jacobian spread.
        float horizontalGain = 1.0f;
        float crestStrain = 0.0f;
        float bakeMs = 0.0f;
        double slopeVarSpectrum = 0.0;
        double slopeVarCoxMunk = 0.0;
        double significantWaveHeight = 0.0;
        double surfaceY = 0.0;
        double crestSteepness = 0.0; // max a*sigma over cascades, <= kMaxSkewSteepness
        // Displacement bounds about the mean level, m.
        double crestHeight = 0.0;
        double troughDepth = 0.0;

        // GPU ms; streaming queue, so absent from per-pass profiles.
        float gpuStageMs[kGpuStages] = {}; // spectrum, mips, tessellation, structures
        float gpuTotalMs = 0.0f;

        // Last measured whitecaps (OceanFoamState), per foam level.
        float foamCoverage[OCEAN_FOAM_LEVELS] = {};
        float foamWhite = 0.0f; // white share of the reference level; steered to foamTarget
        float foamBreaking = 0.0f;
        float foamThreshold[OCEAN_FOAM_LEVELS] = {}; // breaking Jacobian
        float foamMatch[OCEAN_FOAM_LEVELS] = {};
        float foamTarget = 0.0f; // ocean::WhitecapCover
    };
    static constexpr const wchar_t* kGpuStageNames[kGpuStages] = {L"spectrum", L"mips", L"tessellation",
                                                                  L"structures"};
    const Stats& GetStats() const { return m_stats; }

  private:
    static constexpr uint32_t kTimestamps = kGpuStages + 1;

    // Size every buffer from this, so it matches GetReservation.
    uint32_t TileBudget() const {
        return m_params.maxTiles == 0u ? (uint32_t)OCEAN_MAX_TILES
                                       : std::min(m_params.maxTiles, (uint32_t)OCEAN_MAX_TILES);
    }

    // Indices are absolute, so every BLAS declares the whole span.
    uint32_t VertexSpan() const { return m_vertexBase + TileBudget() * OCEAN_TILE_VERTS; }

    void Bake();
    void BakeTurbulence();
    void UploadBaked(ID3D12GraphicsCommandList* copyList);
    void UploadTurbulence(ID3D12GraphicsCommandList* copyList);
    void RecordSimulation(ID3D12GraphicsCommandList4* cl);
    void RecordFoam(ID3D12GraphicsCommandList4* cl);
    void RecordTessellation(ID3D12GraphicsCommandList4* cl);
    void RecordAccelerationStructures(ID3D12GraphicsCommandList4* cl);
    void Transition(ID3D12GraphicsCommandList* cl, ID3D12Resource* r, D3D12_RESOURCE_STATES& state,
                    D3D12_RESOURCE_STATES next);
    void Timestamp(ID3D12GraphicsCommandList* cl, uint32_t point) {
        if (m_timestamps)
            cl->EndQuery(m_timestamps.Get(), D3D12_QUERY_TYPE_TIMESTAMP, m_frameIndex * kTimestamps + point);
    }

    Params m_params;
    const ICoverage* m_coverage = nullptr;
    bool m_initialised = false;

    ID3D12Device5* m_device = nullptr;
    ID3D12DescriptorHeap* m_srvHeap = nullptr;
    DeviceContext* m_ctx = nullptr;

    Quadtree m_quadtree;
    Spectrum m_spectrum;

    ID3D12Resource* m_globalVertex = nullptr;
    ID3D12Resource* m_globalIndex = nullptr;
    uint32_t m_vertexBase = 0;
    uint32_t m_indexBase = 0;
    uint32_t m_materialBase = 0;
    uint32_t m_propsBase = 0;
    uint32_t m_materialIndex = 0;

    ComPtr<ID3D12Resource> m_h0;
    ComPtr<ID3D12Resource> m_fft;
    ComPtr<ID3D12Resource> m_disp[2]; // ping-pong; last frame's feeds motion vectors
    ComPtr<ID3D12Resource> m_deriv;
    D3D12_RESOURCE_STATES m_dispState[2] = {D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                            D3D12_RESOURCE_STATE_UNORDERED_ACCESS};
    D3D12_RESOURCE_STATES m_derivState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    // Whitecap foam, OCEAN_SRV_FOAM0/1; ping-pong.
    ComPtr<ID3D12Resource> m_foam[2];
    D3D12_RESOURCE_STATES m_foamState[2] = {D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                            D3D12_RESOURCE_STATE_UNORDERED_ACCESS};
    uint32_t m_foamParity = 0;
    int64_t m_foamCell[OCEAN_FOAM_LEVELS][2] = {}; // last frame's origin, whole world texels
    bool m_foamHistory = false;
    // Breaking histograms and state (OCEAN_UAV_FOAM_STATS); readback per frame in flight.
    ComPtr<ID3D12Resource> m_foamStats;
    D3D12_RESOURCE_STATES m_foamStatsState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    ComPtr<ID3D12Resource> m_foamStatsReadback;
    ComPtr<ID3D12Resource> m_bakeUpload;
    ComPtr<ID3D12Resource> m_turbulence;
    ComPtr<ID3D12Resource> m_turbulenceUpload;

    ComPtr<ID3D12Resource> m_paramsBuffer;
    ComPtr<ID3D12Resource> m_tilesBuffer;
    ComPtr<ID3D12Resource> m_paramsUpload[FRAME_COUNT];
    ComPtr<ID3D12Resource> m_tilesUpload[FRAME_COUNT];
    uint64_t m_timestampFence[FRAME_COUNT] = {};

    ComPtr<ID3D12Resource> m_blasBuffer;
    ComPtr<ID3D12Resource> m_blasScratch;
    uint64_t m_blasSlotSize = 0;
    uint64_t m_blasScratchSize = 0; // per tile; one range per budget slot
    // Clustered: each budget slot's CLAS region and cluster IDs; every tile shares the topology, so the spans too.
    clusters::ClusterGeometry* m_clusters = nullptr;
    bool m_clustered = false;
    ComPtr<ID3D12Resource> m_clasBuffer;
    uint64_t m_clasSlotSize = 0, m_clasScratchSize = 0;
    std::vector<clusters::Span> m_tileSpans;
    std::vector<uint32_t> m_tileClusterBase;
#if RDN_CLUSTER_API
    D3D12_RTAS_CLUSTER_LIMITS m_tileLimits{};
#endif

    ComPtr<ID3D12RootSignature> m_rootSig;
    ComPtr<ID3D12PipelineState> m_psoFftH;
    ComPtr<ID3D12PipelineState> m_psoFftV;
    ComPtr<ID3D12PipelineState> m_psoMip;
    ComPtr<ID3D12PipelineState> m_psoImport;
    External m_external;
    ID3D12Resource* m_externalViews[OCEAN_CASCADES] = {}; // textures the OCEAN_SRV_EXTERNAL views point at
    Overlays m_overlays;
    ID3D12Resource* m_overlayViews[2] = {}; // the atlases the OCEAN_SRV_OVERLAY_* views point at
    ComPtr<ID3D12PipelineState> m_psoTiles;
    ComPtr<ID3D12PipelineState> m_psoFoam;
    ComPtr<ID3D12PipelineState> m_psoFoamMip;
    ComPtr<ID3D12PipelineState> m_psoFoamStats;

    // Mean sea level used; deepest trough below it.
    double m_surfaceY = 0.0;
    double m_waveDepth = 0.0;

    // Per cascade: elevation variance, energy-weighted wavenumber.
    double m_cascadeVariance[OCEAN_CASCADES] = {};
    double m_cascadeMeanK[OCEAN_CASCADES] = {};
    double m_strainRms = 0.0; // horizontal strain norm RMS per unit choppiness
    ChopBand m_chopBand; // OceanParamsGPU.chopBand
    // OceanParamsGPU.residualSlope, solved with the spectrum.
    float m_residualSlope[OCEAN_ROUGHNESS_ENTRIES] = {};
    uint32_t m_frameCounter = 0;

    std::vector<XMFLOAT4> m_h0Data; // baked spectrum, CPU copy
    std::vector<XMFLOAT4> m_turbulenceData; // gain, d/dx, d/dz
    bool m_turbulenceBakePending = true;
    bool m_turbulenceDirty = true;
    OceanParamsGPU m_gpuParams{};
    std::vector<OceanTileGPU> m_gpuTiles;
    std::vector<uint8_t> m_blasBuilt; // per slot: refittable structure exists
    std::vector<double> m_blasRebuiltAt; // simulated time of the last full build
    static constexpr double kBlasRebuildSeconds = 1.0; // refit lifetime

    double m_time = 0.0;
    double m_phaseEpoch = 0.0;
    planet::DVec3 m_previousOrigin{};
    float m_dt = 0.0f;
    uint32_t m_frameIndex = 0;
    uint32_t m_historyFrames = 0; // frames since the last re-bake
    uint32_t m_parity = 0;
    bool m_bakePending = true;
    bool m_h0Dirty = true;
    planet::DVec3 m_sceneOrigin{};

    Stats m_stats;
    ComPtr<ID3D12QueryHeap> m_timestamps;
    ComPtr<ID3D12Resource> m_timestampReadback;
    uint64_t m_timestampFrequency = 0;
    uint64_t m_profileFrame = 0;
    std::ofstream m_profile;
};

} // namespace ocean
