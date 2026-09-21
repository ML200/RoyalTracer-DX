#pragma once

#include "OceanCommon.h"
#include "OceanQuadtree.h"
#include <fstream>
#include "../planet/stream_orchestrator.h"

struct DeviceContext;

namespace ocean {

// The ocean plugs into the streaming orchestrator's external-geometry slot: its simulation and
// tessellation are recorded onto the streaming compute queue, its tiles are appended to the same
// top-level acceleration structure as the rest of the scene, and the graphics queue waits on that
// queue before tracing. The geometry lives in the renderer's global vertex and index buffers, so
// the hit evaluator reaches ocean triangles exactly as it reaches any other mesh.
class OceanSystem : public planet::IExternalStream {
  public:
    ~OceanSystem() override = default;

    void Configure(const Params& p);
    const Params& GetParams() const { return m_params; }
    bool Enabled() const { return m_params.enabled; }

    // Vertex, index, material-ID and instance-property space the ocean needs inside the scene's
    // shared buffers. Queried before those buffers are created.
    struct Reservation {
        uint32_t vertexElems = 0;
        uint32_t indexElems = 0;
        uint32_t matIDElems = 0;
        uint32_t instanceSlots = 0;
    };
    Reservation GetReservation() const;

    void Init(ID3D12Device5* device, DeviceContext* ctx);

    // Binds the scene ranges the ocean owns. `propsBase` is the first instance-property record.
    void BindScene(ID3D12Resource* globalVertex, ID3D12Resource* globalIndex, uint32_t vertexBase,
                   uint32_t indexBase, uint32_t materialBase, uint32_t propsBase, uint32_t materialIndex);

    // Writes the ocean's descriptors into the renderer's shader-visible heap.
    void CreateDescriptors(ID3D12Device* device, ID3D12DescriptorHeap* heap);

    // Selects tiles and stages this frame's uploads. Call before submit_work.
    void BeginFrame(float dt, const planet::CameraView& cam, uint32_t frameIndex);

    // planet::IExternalStream
    uint32_t instance_capacity() const override;
    void record_gpu_work(ID3D12GraphicsCommandList* copyList, ID3D12GraphicsCommandList4* computeList) override;
    void append_instances(planet::TlasBuilder& tlas, InstanceProperties* props, const planet::DVec3& sceneOrigin,
                          uint32_t hitGroup, bool& forceRebuild) override;
    void on_submitted(uint64_t copyFence, uint64_t computeFence) override;

    // Clear dielectric surface with separate absorption and single-scattering volume controls.
    static Material MakeMaterial(const Params& p);

    struct Stats {
        uint32_t tiles = 0;
        uint32_t leaves = 0;
        uint32_t dropped = 0;
        uint32_t builds = 0;
        uint32_t refits = 0;
        uint64_t triangles = 0;
        uint64_t blasBytes = 0;
        uint64_t resourceBytes = 0;
        float conditioningGain = 1.0f;
        float bakeMs = 0.0f;
        double slopeVarSpectrum = 0.0;
        double slopeVarCoxMunk = 0.0;
        double significantWaveHeight = 0.0;
        double whitecapCoverage = 0.0;
        double whitecapMeasured = 0.0;
        double surfaceY = 0.0;
        // Steepest crest sharpening any cascade carries, as a*sigma. Saturated at
        // ocean::kMaxSkewSteepness the band is as peaked as it can get without its troughs
        // turning back up.
        double crestSteepness = 0.0;
    };
    const Stats& GetStats() const { return m_stats; }

  private:
    // Vertices the acceleration structures may address. The indices stored for a tile are absolute
    // positions in the global buffer, so every build has to declare the whole span.
    uint32_t VertexSpan() const { return m_vertexBase + OCEAN_MAX_TILES * OCEAN_TILE_VERTS; }

    void Bake();
    void UploadBaked(ID3D12GraphicsCommandList* copyList);
    void RecordSimulation(ID3D12GraphicsCommandList4* cl);
    void RecordTessellation(ID3D12GraphicsCommandList4* cl);
    void RecordAccelerationStructures(ID3D12GraphicsCommandList4* cl);
    void Timestamp(ID3D12GraphicsCommandList* cl, uint32_t point) {
        if (m_timestamps) cl->EndQuery(m_timestamps.Get(), D3D12_QUERY_TYPE_TIMESTAMP, m_frameIndex * 7 + point);
    }

    Params m_params;
    bool m_paramsDirty = true;
    bool m_initialised = false;

    ID3D12Device5* m_device = nullptr;
    ID3D12DescriptorHeap* m_srvHeap = nullptr;
    DeviceContext* m_ctx = nullptr;

    Quadtree m_quadtree;
    Spectrum m_spectrum;

    // Scene ranges
    ID3D12Resource* m_globalVertex = nullptr;
    ID3D12Resource* m_globalIndex = nullptr;
    uint32_t m_vertexBase = 0;
    uint32_t m_indexBase = 0;
    uint32_t m_materialBase = 0;
    uint32_t m_propsBase = 0;
    uint32_t m_materialIndex = 0;

    // Simulation resources
    ComPtr<ID3D12Resource> m_h0;
    ComPtr<ID3D12Resource> m_wave;
    ComPtr<ID3D12Resource> m_fft;
    ComPtr<ID3D12Resource> m_disp;
    ComPtr<ID3D12Resource> m_deriv;
    ComPtr<ID3D12Resource> m_moments;
    ComPtr<ID3D12Resource> m_surface;
    ComPtr<ID3D12Resource> m_previousDisp;
    ComPtr<ID3D12Resource> m_foam[2];
    ComPtr<ID3D12Resource> m_bakeUpload;

    ComPtr<ID3D12Resource> m_paramsBuffer;
    ComPtr<ID3D12Resource> m_tilesBuffer;
    ComPtr<ID3D12Resource> m_statsBuffer;
    ComPtr<ID3D12Resource> m_statsReadback[FRAME_COUNT];
    uint64_t m_statsFence[FRAME_COUNT] = {};
    ComPtr<ID3D12Resource> m_paramsUpload[FRAME_COUNT];
    ComPtr<ID3D12Resource> m_tilesUpload[FRAME_COUNT];

    ComPtr<ID3D12Resource> m_blasBuffer;
    ComPtr<ID3D12Resource> m_blasScratch;
    uint64_t m_blasSlotSize = 0;
    uint64_t m_blasScratchSize = 0;
    uint64_t m_blasUpdateScratchSize = 0;
    static constexpr uint32_t kScratchSlots = 24;

    ComPtr<ID3D12RootSignature> m_rootSig;
    ComPtr<ID3D12PipelineState> m_psoEvolve;
    ComPtr<ID3D12PipelineState> m_psoFftH;
    ComPtr<ID3D12PipelineState> m_psoFftV;
    ComPtr<ID3D12PipelineState> m_psoAssemble;
    ComPtr<ID3D12PipelineState> m_psoCondition;
    ComPtr<ID3D12PipelineState> m_psoFoam;
    ComPtr<ID3D12PipelineState> m_psoMip;
    ComPtr<ID3D12PipelineState> m_psoTiles;

    // Folding threshold control. The baked z-score sets the instantaneous folding fraction, but
    // foam persists and drifts for seconds afterwards, so the coverage that actually reaches the
    // image is several times larger and depends on the decay and drift settings. Rather than fold
    // a fudge factor into the calibration, the measured coverage steers a per-cascade offset.
    double m_foamZBase = 0.0;
    double m_foamZOffset[OCEAN_CASCADES] = {};
    double m_foamSigmaJ[OCEAN_CASCADES] = {};
    double m_foamTargetPerCascade = 0.0;
    double m_foamMeasured[OCEAN_CASCADES] = {};

    // Mean sea level actually used, and how far the deepest trough reaches below it.
    double m_surfaceY = 0.0;
    double m_waveDepth = 0.0;

    // Elevation variance each cascade carries and its energy-weighted wavenumber. The crest
    // sharpening is solved from these every frame, so it needs no re-bake of its own.
    double m_cascadeVariance[OCEAN_CASCADES] = {};
    double m_cascadeMeanK[OCEAN_CASCADES] = {};

    // Baked spectrum, CPU side. The same samples produce the textures and the slope variances.
    std::vector<XMFLOAT4> m_h0Data;
    std::vector<XMFLOAT4> m_waveData;
    OceanParamsGPU m_gpuParams{};
    std::vector<OceanTileGPU> m_gpuTiles;
    std::vector<uint8_t> m_blasBuilt; // per slot: has a valid structure that may be refitted

    double m_time = 0.0;
    double m_phaseEpoch = 0.0;
    float m_effectiveChoppiness = 0.0f;
    bool m_resetFoam = true;
    planet::DVec3 m_previousOrigin{};
    float m_dt = 0.0f;
    uint32_t m_frameIndex = 0;
    uint32_t m_frameCounter = 0;
    uint32_t m_foamParity = 0;
    bool m_bakePending = true;
    bool m_h0Dirty = true;
    bool m_firstRecord = true;
    planet::DVec3 m_sceneOrigin{};

    Stats m_stats;
    ComPtr<ID3D12QueryHeap> m_timestamps;
    ComPtr<ID3D12Resource> m_timestampReadback;
    uint64_t m_timestampFrequency = 0;
    uint64_t m_profileFrame = 0;
    std::ofstream m_profile;
};

} // namespace ocean

