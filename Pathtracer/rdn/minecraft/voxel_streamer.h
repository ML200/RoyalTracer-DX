#pragma once

#include <atomic>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "mc_types.h"
#include "lod_tree.h"
#include "voxel_mesher.h"
#include "../Common.h"
#include "../planet/stream_orchestrator.h"
#include "../Lighting/LightTreeRefit.h"

struct DeviceContext;

#include "mc_placement.h"
#include "mc_omm.h"
namespace mc {

class World;

struct StreamerConfig {
    float    lodFactor          = 512.0f;
    float    lodFactorMin       = 48.0f;
    bool     adaptiveLod        = true;
    uint64_t triangleBudget     = 100000000ull;
    int      flatColorLevel     = 6;
    uint32_t vertexCapacity     = 200u << 20;
    uint32_t indexCapacity      = 360u << 20;
    uint32_t matIdCapacity      = 128u << 20;
    uint32_t ommIndexCapacity   = 48u << 20;
    uint64_t blasPoolBytes      = 4096ull << 20;
    uint64_t blasBuildPoolBytes = 768ull << 20;
    bool     blasCompaction     = true;
    uint32_t maxInstances       = 32768;
    uint32_t stagingSlots       = 128;
    uint32_t stagingSlotBytes   = 3u << 20;
    uint64_t scratchBytesPerFrame = 128ull << 20;
    uint32_t buildBudget        = 24;
    uint32_t buildBudgetBurst   = 64;
    uint32_t meshJobsInFlight   = 64;
    uint32_t meshJobsInFlightBurst = 128;
    uint32_t retireFrames       = 8;
    uint32_t evictFrames        = 300;
    bool     lights             = true;
    uint32_t maxLightTris       = 2u << 20;
    uint32_t lightRecordCapacity= 3u << 20;
    uint32_t lightNodeCapacity  = 6u << 20;
    uint32_t maxLightSlots      = 16384;
    int      lightMaxLevel      = 8;
    bool     freezeLod          = false;
    bool     warmUp             = true;
    double   warmUpSeconds      = 45.0;
};

struct StreamerStats {
    uint32_t desired = 0, rendered = 0, ready = 0, empty = 0, pending = 0, meshing = 0, meshed = 0, uploading = 0;
    uint32_t chunksTracked = 0;
    uint64_t trianglesRendered = 0, trianglesResident = 0, trianglesEstimated = 0, triangleBudget = 0;
    float    lodFactorNow = 0.0f;
    float    cutReadyFraction = 1.0f;
    uint64_t vertexUsed = 0, indexUsed = 0, matIdUsed = 0, blasUsed = 0, blasBuildUsed = 0;
    uint32_t buildsThisFrame = 0, copiesThisFrame = 0, compactionsThisFrame = 0;
    uint64_t uploadBytesThisFrame = 0;
    uint32_t allocFailures = 0;
    uint32_t evicted = 0;
    float    meshMsAvg = 0.0f;
    float    selectMs  = 0.0f;
    float    adoptMs   = 0.0f;
    float    resolveMs = 0.0f;
    float    adaptMs   = 0.0f;
    uint32_t cutNodes  = 0;
    float    recordMs  = 0.0f;
    double   warmUpSeconds = 0.0;
    bool     warmUpComplete = false;
    uint64_t lightTrisResident = 0, lightTrisInTree = 0, lightRecUsed = 0, lightNodeUsed = 0;
    uint32_t lightChunksInTree = 0, lightSlotsActive = 0, lightChunksDropped = 0;
    uint32_t lightVersion = 0, lightLiveVersion = 0;
};

class SpanAllocator {
public:
    void init(uint64_t capacity, uint64_t granularity);
    bool allocate(uint64_t count, uint64_t& offset);
    void free(uint64_t offset, uint64_t count);
    uint64_t used() const { return m_used; }
    uint64_t capacity() const { return m_capacity; }
    size_t   fragments() const { return m_free.size(); }
private:
    struct Span { uint64_t first, count; };
    std::vector<Span> m_free;
    uint64_t m_capacity = 0, m_granularity = 1, m_used = 0;
};

struct LightBinding {
    ID3D12Resource* records   = nullptr;
    ID3D12Resource* nodes     = nullptr;
    ID3D12Resource* leafIndex = nullptr;
    ID3D12Resource* trails    = nullptr;
    uint32_t recordBase = 0, recordCapacity = 0;
    uint32_t nodeBase   = 0, nodeCapacity   = 0;
    uint32_t slotBase   = 0;
};

class VoxelStreamer final : public planet::IExternalStream {
public:
    VoxelStreamer();
    ~VoxelStreamer() override;

    // Initializes GPU pools and worker state for asynchronous chunk streaming.
    void init(ID3D12Device5* device, DeviceContext* ctx, World* world, const StreamerConfig& cfg);
    bool enabled() const { return m_world != nullptr; }

    void set_placement(const Placement& p) { m_placement = p; }
    const Placement& placement() const { return m_placement; }

    void bind_geometry(ID3D12Resource* vertexGlobal, ID3D12Resource* indexGlobal, uint32_t combinedVertexCount,
                       uint32_t vertexBaseElems, uint32_t indexBaseElems,
                       ID3D12Resource* materialIds, uint32_t matIdBaseElems,
                       uint32_t instancePropsBase);

    void bind_omm(ID3D12Resource* indexBuffer, D3D12_GPU_VIRTUAL_ADDRESS ommArray, const OmmTable* table);

    void bind_lights(const LightBinding& b);
    void unbind_lights();
    bool lights_bound() const { return m_lightsBound; }
    void set_light_slot_base(uint32_t base);

    uint32_t light_version() const { return m_lightVersion; }
    void light_snapshot(const planet::DVec3& sceneOrigin, std::vector<lt::TLASExtraLeaf>& out,
                        uint32_t& version, uint32_t& slotCount) const;
    void on_light_tlas_published(uint32_t version, bool hasVoxelLeaves);
    bool has_live_lights() const { return m_lightTlasHasVoxels; }

    // Selects the LOD cut and advances uploads, builds, and retirements.
    void begin_frame(const double camWorld[3]);

    void warm_up(const double camWorld[3]);

    uint32_t instance_capacity() const override { return m_cfg.maxInstances; }
    void record_gpu_work(ID3D12GraphicsCommandList* copyList, ID3D12GraphicsCommandList4* computeList) override;
    void append_instances(planet::TlasBuilder& tlas, InstanceProperties* props, const planet::DVec3& sceneOrigin,
                          uint32_t hitGroup, bool& forceRebuild) override;
    // Keeps staging and BLAS resources alive until both fences retire.
    void on_submitted(uint64_t copyFence, uint64_t computeFence) override;

    void set_block(int x, int y, int z, BlockId id);

    StreamerConfig&       config()       { return m_cfg; }
    const StreamerConfig& config() const { return m_cfg; }
    const StreamerStats&  stats()  const { return m_stats; }
    const World*          world()  const { return m_world; }

private:
    enum class State : uint8_t { Pending, Meshing, Meshed, Uploading, Compacting, Ready, Empty };
    static constexpr uint32_t NONE = 0xFFFFFFFFu;

    struct LightData {
        uint64_t recOff = 0, recCount = 0;
        uint64_t nodeOff = 0, nodeCount = 0;
        uint32_t litOpaque = 0, litAlpha = 0;
        float bmin[3] = {}, bmax[3] = {};
        float power = 0.0f;
        float axis[3] = { 0, 0, 1 };
        float cosTheta = -1.0f, sinTheta = 0.0f;
        uint32_t gen = 0;
        bool valid() const { return recCount > 0; }
    };
    struct GpuChunk {
        uint64_t vtxOff = 0, vtxCount = 0;
        uint64_t idxOff = 0, idxCount = 0;
        uint64_t matOff = 0, matCount = 0;
        uint64_t blasOff = 0, blasSize = 0;
        D3D12_GPU_VIRTUAL_ADDRESS blasVa = 0;
        uint64_t buildOff = 0, buildSize = 0;
        D3D12_GPU_VIRTUAL_ADDRESS buildVa = 0;
        uint32_t triCount = 0, opaqueTriCount = 0;
        uint64_t ommOff = 0, ommCount = 0;
        LightData light;
        bool valid() const { return triCount > 0; }
    };
    struct MeshJob {
        NodeKey    key;
        uint64_t   packed = 0;
        uint32_t   version = 0;
        MeshParams params;
        bool       wantLights = false;
        uint32_t   lightGen = 0;
        std::atomic<int> state{ 0 };
        GpuChunk   gpu;
        bool       lightsDropped = false;
        bool       buildPoolFull = false;
        uint64_t   scratchSize = 0;
        int        stagingSlot = -1;
        ComPtr<ID3D12Resource> privateUpload;
        ID3D12Resource* srcBuffer = nullptr;
        uint8_t*   srcMapped = nullptr;
        uint64_t   srcVtxOff = 0, srcIdxOff = 0;
        uint64_t   srcRecOff = 0, srcNodeOff = 0, srcLeafOff = 0, srcTrailOff = 0;
        uint32_t   vtxBytes = 0, idxBytes = 0;
        uint32_t   recBytes = 0, nodeBytes = 0, leafBytes = 0, trailBytes = 0;
        float      meshMs = 0.0f;
    };
    struct Chunk {
        NodeKey  key;
        State    state = State::Pending;
        uint32_t version = 0;
        bool     rebuilding = false;
        bool     remeshAfterUpload = false;
        bool     lightsDropped = false;
        bool     resident = false;
        GpuChunk gpu;
        GpuChunk next;
        std::shared_ptr<MeshJob> job;
        uint64_t uploadFence = 0;
        uint64_t compactFence = 0;
        uint32_t compactSlot = NONE, compactIndex = 0;
        bool     compactQueued = false;
        uint32_t instanceSlot = NONE;
        uint32_t cutNode = NONE;
        uint32_t lightSlot = NONE;
        uint32_t propsLightSlot = NONE;
        uint32_t lightIncludedFrame = 0;
        uint32_t lastRenderedFrame = 0, lastDesiredFrame = 0, createdFrame = 0, retryFrame = 0;
        float    priority = 0.0f;
        bool     propsWritten = false;
    };
    struct LightSlot {
        uint64_t  chunk = INVALID_NODE_KEY;
        uint32_t  instanceID = NONE;
        uint32_t  includedAt = 0, excludedAt = 0;
        int64_t   origin[3] = { 0, 0, 0 };
        LightData light;
        bool active() const { return excludedAt == NONE; }
    };
    struct SelectJob {
        double cam[3] = { 0, 0, 0 };
        float  factor = 0.0f;
        LodCut cut;
        std::atomic<int> state{ 0 };
    };
    struct RenderEntry { uint64_t key; Chunk* chunk; };
    struct StagingSlot { uint64_t offset = 0; uint8_t* mapped = nullptr; bool inUse = false; uint64_t copyFence = 0; bool pendingFence = false; };
    struct Retired { GpuChunk gpu; uint32_t frame; uint32_t lightExcludedAt; };
    struct RetiredLight { LightData light; uint32_t frame; uint32_t excludedAt; };
    struct RetiredSlot { uint32_t slot; uint32_t frame; };
    struct RetiredLightSlot { uint32_t slot; uint32_t frame; uint32_t excludedAt; };
    struct RetiredUpload { ComPtr<ID3D12Resource> res; uint64_t copyFence; };

    void run_job(const std::shared_ptr<MeshJob>& job);
    ChunkMesher& thread_mesher();
    void calibrate_estimates();
    uint32_t piece_count(uint32_t count, uint32_t minPerPiece) const;
    template <class F> void parallel_ranges(uint32_t count, uint32_t minPerPiece, F&& body);
    template <class F> void parallel_buckets(F&& body);
    bool build_lights(const ChunkMesh& mesh, MeshJob& job, std::vector<LightTriangle>& records,
                      lt::LightTreeBuilder::SingleBLAS& blas);
    bool allocate_gpu(GpuChunk& g, uint32_t vtxCount, uint32_t idxCount, uint32_t triCount, uint64_t blasSize,
                      uint32_t lightRecs, uint32_t lightNodes, bool* buildPoolFull = nullptr);
    void free_gpu(const GpuChunk& g);
    void free_light(const LightData& l);
    void retire_gpu(const GpuChunk& g, uint32_t lightExcludedAt);
    void retire_light(const LightData& l, uint32_t excludedAt);
    bool light_region_free(uint32_t excludedAt) const;
    void reclaim();
    void process_jobs();
    void select_and_schedule();
    void adopt_cut(LodCut& cut, const double cam[3], float factor);
    void set_resident(Chunk& c, bool resident);
    void adapt_lod();
    uint64_t estimate_triangles();
    uint64_t effective_budget() const;
    void dispatch_jobs();
    void evict(bool immediate);
    void update_light_set(uint32_t liveBefore);
    void drop_far_lights();
    uint32_t exclude_light_slot(Chunk& c);
    bool light_slot_live(uint32_t slot) const;
    int  acquire_staging();
    void release_staging_after(int slot, uint64_t copyFence);
    uint32_t acquire_instance_slot();
    void release_instance_slot(uint32_t slot);
    float priority_of(const NodeKey& k) const;
    void update_stats();

    ID3D12Device5* m_device = nullptr;
    DeviceContext* m_ctx = nullptr;
    World*         m_world = nullptr;
    StreamerConfig m_cfg;
    StreamerStats  m_stats;

    ID3D12Resource* m_vertexGlobal = nullptr;
    ID3D12Resource* m_indexGlobal = nullptr;
    ID3D12Resource* m_materialIds = nullptr;
    uint8_t*        m_materialIdsMapped = nullptr;
    uint32_t        m_combinedVertexCount = 0;
    uint32_t        m_vertexBase = 0, m_indexBase = 0, m_matIdBase = 0, m_propsBase = 0;
    D3D12_GPU_VIRTUAL_ADDRESS m_vertexVa = 0, m_indexVa = 0;

    LightBinding          m_lights;
    std::atomic<bool>     m_lightsBound{ false };
    std::atomic<uint32_t> m_lightGen{ 0 };

    std::mutex     m_poolMutex;
    SpanAllocator  m_vtxAlloc, m_idxAlloc, m_matAlloc, m_blasAlloc, m_lightRecAlloc, m_lightNodeAlloc;
    ComPtr<ID3D12Resource> m_blasPool;
    D3D12_GPU_VIRTUAL_ADDRESS m_blasPoolVa = 0;
    SpanAllocator             m_buildAlloc;
    ComPtr<ID3D12Resource>    m_buildPool;
    D3D12_GPU_VIRTUAL_ADDRESS m_buildPoolVa = 0;
    std::vector<ComPtr<ID3D12Resource>> m_compactInfo;
    std::vector<ComPtr<ID3D12Resource>> m_compactReadback;
    std::vector<const uint64_t*>        m_compactMapped;
    std::vector<uint32_t>               m_compactCount;
    std::vector<uint8_t>                m_compactInfoCopyState;
    std::vector<uint64_t>  m_toCompact;
    std::vector<uint64_t>  m_compactingThisFrame;
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS build_flags() const;
    SpanAllocator             m_ommAlloc;
    ID3D12Resource*           m_ommIndexBuffer = nullptr;
    uint8_t*                  m_ommIndicesMapped = nullptr;
    D3D12_GPU_VIRTUAL_ADDRESS m_ommIndexVa = 0, m_ommArrayVa = 0;
    const OmmTable*           m_ommTable = nullptr;

    ComPtr<ID3D12Resource>    m_staging;
    uint8_t*                  m_stagingMapped = nullptr;
    std::vector<StagingSlot>  m_stagingSlots;
    std::vector<ComPtr<ID3D12Resource>> m_scratch;
    std::vector<uint64_t>     m_scratchUsed;
    std::vector<RetiredUpload> m_retiredUploads;

    std::unordered_map<uint64_t, Chunk, U64Hash> m_chunks;
    std::unordered_map<uint64_t, uint32_t, U64Hash> m_residentBelow;
    std::vector<std::shared_ptr<MeshJob>> m_jobs;
    std::vector<RenderItem>  m_renderItems;
    std::vector<uint64_t>    m_prevRenderKeys;
    std::vector<RenderEntry> m_render;
    uint32_t               m_renderListFrame = 0;
    std::vector<uint64_t>  m_uploading;
    std::vector<uint64_t>  m_uploadingThisFrame;
    std::shared_ptr<SelectJob> m_selectJob;
    std::vector<Retired>   m_retired;
    std::vector<RetiredLight> m_retiredLights;
    std::vector<RetiredSlot> m_retiredSlots;
    std::vector<uint32_t>  m_freeSlots;
    uint32_t               m_nextSlot = 0;
    LodCut                 m_cut;
    bool                   m_cutValid = false;
    bool                   m_renderListChanged = true;
    double                 m_cam[3] = { 0, 0, 0 };
    Placement              m_placement;
    DirectX::XMMATRIX placement_matrix(float tx, float ty, float tz) const;
    double                 m_cutCam[3] = { 1e30, 1e30, 1e30 };
    float                  m_cutLodFactor = -1.0f;
    uint32_t               m_cutChangedFrame = 0;
    uint64_t               m_trianglesRendered = 0;
    float                  m_lodFactorNow = 0.0f;
    double                 m_cutReadyFraction = 1.0;
    bool                   m_jumpPending = false;
    bool                   m_evictNow = false;
    int                    m_cutFlatLevel = -1;
    int                    m_lightsApplied = -1;
    double                 m_levelTriEst[MAX_LOD_LEVELS] = {};
    planet::DVec3          m_lastOrigin{ 1e30, 1e30, 1e30 };
    uint32_t               m_frame = 0;
    uint64_t               m_lastCopyFence = 0, m_lastComputeFence = 0;
    bool                   m_unlimitedBudget = false;

    std::vector<LightSlot>        m_lightSlots;
    std::deque<uint32_t>          m_freeLightSlots;
    std::vector<RetiredLightSlot> m_retiredLightSlots;
    uint32_t               m_lightVersion = 0;
    uint32_t               m_lightLiveVersion = 0;
    bool                   m_lightTlasHasVoxels = false;
    bool                   m_lightSetDirty = false;

    std::unique_ptr<planet::WorkerPool> m_lodPool;
    std::unique_ptr<planet::WorkerPool> m_workers;
};

template <class F> void VoxelStreamer::parallel_ranges(uint32_t count, uint32_t minPerPiece, F&& body) {
    const uint32_t pieces = piece_count(count, minPerPiece);
    if (pieces <= 1) { if (count) body(0u, count, 0u); return; }
    m_lodPool->parallel_for(pieces, [&](uint32_t p) {
        const uint32_t b = (uint32_t)((uint64_t)count * p / pieces), e = (uint32_t)((uint64_t)count * (p + 1) / pieces);
        if (e > b) body(b, e, p);
    });
}

template <class F> void VoxelStreamer::parallel_buckets(F&& body) {
    const uint32_t buckets = (uint32_t)m_chunks.bucket_count();
    parallel_ranges(buckets, 512, [&](uint32_t b0, uint32_t b1, uint32_t piece) {
        for (uint32_t b = b0; b < b1; ++b)
            for (auto it = m_chunks.begin(b); it != m_chunks.end(b); ++it) body(it->first, it->second, piece);
    });
}

}
