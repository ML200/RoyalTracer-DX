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
//
// That wait is on the critical path of every frame - the CPU has already waited for the previous
// frame's graphics work before this is submitted - so everything recorded here is paid for in
// full, before a single ray is traced.
class OceanSystem : public planet::IExternalStream {
  public:
    ~OceanSystem() override = default;

    void Configure(const Params& p);

    // Where the scene actually has water. Without one the sea covers its whole extent; with one
    // it is only built where the world holds water, which is what keeps a block world from paying
    // to traverse a plane through every street. Must outlive the ocean.
    void SetCoverage(const ICoverage* coverage) { m_coverage = coverage; }
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
                          uint32_t hitGroup, bool& forceRebuild, bool& forceRefit) override;
    void on_submitted(uint64_t copyFence, uint64_t computeFence) override;

    // Clear dielectric surface with separate absorption and single-scattering volume controls.
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
        // Horizontal gain the short waves get (ocean::ShortWaveChop, on top of the choppiness), and
        // the spread of the surface's Jacobian that results: how pointed the crests are, and so how
        // many of them break.
        float horizontalGain = 1.0f;
        float crestStrain = 0.0f;
        float bakeMs = 0.0f;
        double slopeVarSpectrum = 0.0;
        double slopeVarCoxMunk = 0.0;
        double significantWaveHeight = 0.0;
        double surfaceY = 0.0;
        // Steepest crest sharpening any cascade carries, as a*sigma. Saturated at
        // ocean::kMaxSkewSteepness the band is as peaked as it can get without its troughs
        // turning back up.
        double crestSteepness = 0.0;
        // Bounds on the displaced surface about the mean level, metres.
        double crestHeight = 0.0;
        double troughDepth = 0.0;

        // What the ocean costs the GPU each frame, by stage, in milliseconds. This work is
        // recorded on the streaming compute queue, which the graphics queue waits on before it
        // traces - so it lands in the frame's wait rather than in any of the render passes, and
        // is invisible to a per-pass profiler even while it dominates.
        float gpuStageMs[kGpuStages] = {}; // spectrum, mips, tessellation, structures
        float gpuTotalMs = 0.0f;

        // Whitecaps as the simulation last measured them (OceanFoamState): the share of the sea
        // under foam over each foam level's own ground - OCEAN_FOAM_REFERENCE_LEVEL's is the one
        // steered to foamTarget - the share breaking, and the Jacobian each level breaks below.
        float foamCoverage[OCEAN_FOAM_LEVELS] = {};
        float foamWhite = 0.0f; // share of the reference level plainly white: what foamTarget is met in
        float foamBreaking = 0.0f;
        float foamThreshold[OCEAN_FOAM_LEVELS] = {};
        float foamMatch[OCEAN_FOAM_LEVELS] = {};
        float foamTarget = 0.0f; // the cover asked for (ocean::WhitecapCover)
    };
    static constexpr const wchar_t* kGpuStageNames[kGpuStages] = {L"spectrum", L"mips", L"tessellation",
                                                                  L"structures"};
    const Stats& GetStats() const { return m_stats; }

  private:
    static constexpr uint32_t kTimestamps = kGpuStages + 1;

    // Tiles this ocean may keep resident, which is what its per-frame acceleration-structure work
    // and its memory both scale with. Read wherever a buffer is sized, so it stays consistent with
    // what GetReservation asked the scene for.
    uint32_t TileBudget() const {
        return m_params.maxTiles == 0u ? (uint32_t)OCEAN_MAX_TILES
                                       : std::min(m_params.maxTiles, (uint32_t)OCEAN_MAX_TILES);
    }

    // Vertices the acceleration structures may address. The indices stored for a tile are absolute
    // positions in the global buffer, so every build has to declare the whole span.
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

    // Scene ranges
    ID3D12Resource* m_globalVertex = nullptr;
    ID3D12Resource* m_globalIndex = nullptr;
    uint32_t m_vertexBase = 0;
    uint32_t m_indexBase = 0;
    uint32_t m_materialBase = 0;
    uint32_t m_propsBase = 0;
    uint32_t m_materialIndex = 0;

    // Simulation resources. The displacement pair ping-pongs: the array written this frame is
    // the current surface, the other one last frame's, which is all motion vectors need.
    ComPtr<ID3D12Resource> m_h0;
    ComPtr<ID3D12Resource> m_fft;
    ComPtr<ID3D12Resource> m_disp[2];
    ComPtr<ID3D12Resource> m_deriv;
    D3D12_RESOURCE_STATES m_dispState[2] = {D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                            D3D12_RESOURCE_STATE_UNORDERED_ACCESS};
    D3D12_RESOURCE_STATES m_derivState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    // World-space whitecap foam (OCEAN_SRV_FOAM0/1): two camera-centred levels per texture, and two
    // textures that ping-pong so last frame's foam is read while this frame's is written.
    ComPtr<ID3D12Resource> m_foam[2];
    D3D12_RESOURCE_STATES m_foamState[2] = {D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                            D3D12_RESOURCE_STATE_UNORDERED_ACCESS};
    uint32_t m_foamParity = 0;
    // Where each level sat last frame, in whole texels of absolute world XZ, and whether that
    // frame's foam can be carried over at all.
    int64_t m_foamCell[OCEAN_FOAM_LEVELS][2] = {};
    bool m_foamHistory = false;
    // Breaking-point histograms and the state set from them (OCEAN_UAV_FOAM_STATS), and a copy of
    // the state per frame in flight for the readouts.
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
    uint64_t m_blasScratchSize = 0; // per tile; the pool holds one range for every slot of the budget

    ComPtr<ID3D12RootSignature> m_rootSig;
    ComPtr<ID3D12PipelineState> m_psoFftH;
    ComPtr<ID3D12PipelineState> m_psoFftV;
    ComPtr<ID3D12PipelineState> m_psoMip;
    ComPtr<ID3D12PipelineState> m_psoTiles;
    ComPtr<ID3D12PipelineState> m_psoFoam;
    ComPtr<ID3D12PipelineState> m_psoFoamMip;
    ComPtr<ID3D12PipelineState> m_psoFoamStats;

    // Mean sea level actually used, and how far the deepest trough reaches below it.
    double m_surfaceY = 0.0;
    double m_waveDepth = 0.0;

    // Elevation variance each cascade carries and its energy-weighted wavenumber. The crest
    // sharpening is solved from these every frame, so it needs no re-bake of its own.
    double m_cascadeVariance[OCEAN_CASCADES] = {};
    double m_cascadeMeanK[OCEAN_CASCADES] = {};
    // RMS of the horizontal strain tensor's norm per unit choppiness, summed over every cascade:
    // the spread of the surface's Jacobian, which is how pointed its crests are.
    double m_strainRms = 0.0;
    // Which waves take the short-wave chop (OceanParamsGPU.chopBand); follows the spectral peak.
    ChopBand m_chopBand;
    // OceanParamsGPU.residualSlope, solved with the spectrum.
    float m_residualSlope[OCEAN_ROUGHNESS_ENTRIES] = {};
    uint32_t m_frameCounter = 0;

    // Baked spectrum, CPU side. The same samples produce the texture and the statistics.
    std::vector<XMFLOAT4> m_h0Data;
    // Kilometre-scale sea-state field, (gain, d/dx, d/dz). Independent of the spectrum, so it is
    // rebuilt only when its own controls move rather than on every re-bake.
    std::vector<XMFLOAT4> m_turbulenceData;
    bool m_turbulenceBakePending = true;
    bool m_turbulenceDirty = true;
    OceanParamsGPU m_gpuParams{};
    std::vector<OceanTileGPU> m_gpuTiles;
    std::vector<uint8_t> m_blasBuilt; // per slot: has a valid structure that may be refitted
    // Simulated time each slot was last built rather than refitted. Paused water never goes
    // stale, because its vertices are not moving.
    std::vector<double> m_blasRebuiltAt;
    // How long a refitted structure is allowed to drift before it is built again.
    static constexpr double kBlasRebuildSeconds = 1.0;

    double m_time = 0.0;
    double m_phaseEpoch = 0.0;
    planet::DVec3 m_previousOrigin{};
    float m_dt = 0.0f;
    uint32_t m_frameIndex = 0;
    // Frames simulated since the field last changed underneath its history (a re-bake). Motion
    // vectors need one completed frame to difference against.
    uint32_t m_historyFrames = 0;
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
