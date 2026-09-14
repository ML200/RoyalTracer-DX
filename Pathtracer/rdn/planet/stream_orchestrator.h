#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include "coordinate_system.h"
#include "cube_sphere.h"
#include "heightmap_cubemap.h"
#include "worker_pool.h"
#include "tlas_builder.h"
#include "generation.h"
#include "rock_scatter.h"

struct DeviceContext;
struct InstanceProperties;

namespace planet {

struct IExternalStream {
    virtual ~IExternalStream() = default;
    virtual uint32_t instance_capacity() const = 0;
    virtual void record_gpu_work(ID3D12GraphicsCommandList* copyList, ID3D12GraphicsCommandList4* computeList) = 0;
    virtual void append_instances(TlasBuilder& tlas, InstanceProperties* props, const DVec3& sceneOrigin,
                                  uint32_t hitGroup, bool& forceRebuild) = 0;
    virtual void on_submitted(uint64_t copyFence, uint64_t computeFence) = 0;
};

constexpr uint32_t TERRAIN_INSTANCE_BASE = 1u << 20;

constexpr uint32_t MAX_TERRAIN_CELLS = 4096;

constexpr uint32_t MAX_ROCK_INSTANCES = 4096;

constexpr uint32_t TERRAIN_GEO_LEAF_SLOTS = 6144;

struct StreamConfig {
    bool     enabled = true;
    PlanetGeometry planet{};
    uint8_t  min_lod = 0;
    uint8_t  max_lod = MAX_LOD;
    uint32_t max_triangles = 3000000u;
    uint32_t max_leaves_per_cell = 128;
    double   max_cell_radius_m = 1e30;
    uint32_t max_scene_instances = 4096;
    std::string heightmap_dir   = "./terrain";
    float    rebuild_trigger_m = 1.0f;
    uint32_t build_budget = 1;
    bool     predict = true;
};

struct SceneInstanceDesc {
    D3D12_GPU_VIRTUAL_ADDRESS blas = 0;
    float    transform[12]   = {};
    uint32_t instance_id     = 0;
    uint32_t hit_group_index = 0;
    uint32_t flags           = 0;
};

struct TerrainSlotGPU {
    uint32_t node_lo;
    uint32_t node_hi;
    uint32_t changed;
};

class StableIdMap {
public:
    void clear() { m_id.clear(); m_free.clear(); m_next = 0; }

    uint32_t get(uint64_t node_id) {
        const auto it = m_id.find(node_id);
        if (it != m_id.end()) return it->second;
        uint32_t id;
        if (!m_free.empty()) { id = m_free.back(); m_free.pop_back(); }
        else                   id = m_next++;
        m_id.emplace(node_id, id);
        return id;
    }
    uint32_t peak() const { return m_next; }

    template <typename Fn> void retain(Fn&& keep) {
        for (auto it = m_id.begin(); it != m_id.end(); ) {
            if (!keep(it->first)) { m_free.push_back(it->second); it = m_id.erase(it); }
            else ++it;
        }
    }

private:
    std::unordered_map<uint64_t, uint32_t> m_id;
    std::vector<uint32_t>                  m_free;
    uint32_t                               m_next = 0;
};

struct ThroughputEstimator {
    float    rebuild_frames      = 20.0f;
    float    step_ms             = 0.0f;
    uint32_t last_rebuild_frames = 0;

    void on_rebuild_done(uint32_t frames) {
        last_rebuild_frames = frames;
        rebuild_frames += ((float)frames - rebuild_frames) * 0.25f;
    }
    void on_step(float ms) {
        step_ms = (step_ms <= 0.0f) ? ms : step_ms + (ms - step_ms) * 0.25f;
    }
};

class StreamOrchestrator {
public:
    // Initializes terrain pools, workers, and asynchronous generation state.
    void init(ID3D12Device5* device, DeviceContext* ctx, const StreamConfig& cfg);

    void bind_geometry(ID3D12Resource* combinedVtx, uint8_t* vtxMapped,
                       ID3D12Resource* combinedIdx, uint8_t* idxMapped,
                       ID3D12Resource* instanceProps,
                       uint32_t sceneVertexCount, uint32_t sceneIndexCount,
                       uint32_t combinedVertexCount, uint32_t terrainPropsBase,
                       uint32_t terrainLeafSlots, uint32_t terrainMatIDBase,
                       uint32_t terrainTriLightBase);

    // Selects visible terrain, schedules work, and prepares frame uploads.
    void begin_frame(uint32_t frame_index, const CameraView& cam);
    void submit_work(const SceneInstanceDesc* scene, uint32_t scene_count,
                     uint32_t terrain_hit_group, uint32_t external_hit_group = 0);
    void end_frame();

    void set_external(IExternalStream* s) { m_external = s; }
    void bind_instance_properties(ID3D12Resource* props) { m_instanceProps = props; }

    struct RockVariantGPU {
        D3D12_GPU_VIRTUAL_ADDRESS blas_va = 0;
        uint32_t vertexBase = 0;
        uint32_t indexBase  = 0;
        uint32_t triCount   = 0;
    };
    void set_rock_variants(uint32_t propsBase, const std::vector<RockVariantGPU>& variants);
    void set_rock_instances(const RockInstance* insts, uint32_t count);

    bool enabled() const { return m_cfg.enabled; }
    PlanetGeometry planet_geometry() const { return m_cfg.planet; }

    struct TerrainReservation {
        uint32_t vertexElems       = 0;
        uint32_t indexElems        = 0;
        uint32_t matIDElems        = 0;
        uint32_t triLightElems     = 0;
        uint32_t instanceSlots     = 0;
        uint32_t leafSlots         = 0;
        uint32_t propsBase         = 0; // Fixed base for terrain instance records.
    };
    TerrainReservation terrain_reservation() const {
        TerrainReservation r{};
        if (!m_cfg.enabled) return r;
        r.leafSlots     = TERRAIN_GEO_LEAF_SLOTS;
        r.vertexElems   = r.leafSlots * MAX_CHUNK_VERTS;
        r.indexElems    = r.leafSlots * MAX_CHUNK_TRIS * 3u;
        r.matIDElems    = m_cfg.max_leaves_per_cell * MAX_CHUNK_TRIS;
        r.triLightElems = r.matIDElems;
        r.instanceSlots = MAX_TERRAIN_CELLS;
        r.propsBase     = m_cfg.max_scene_instances;
        return r;
    }

    void reserve_scene_instances(uint32_t count);

    ID3D12Resource*           tlas_result()  const { return m_tlas.result(); }
    D3D12_GPU_VIRTUAL_ADDRESS tlas_address() const { return m_tlas.tlas_address(); }
    ID3D12Resource* terrain_table() const { return m_terrainTable.Get(); }
    static constexpr uint32_t terrain_table_count() { return MAX_TERRAIN_CELLS; }

    const HeightmapCubemap& heightmap() const { return m_heightmap; }

    struct Stats {
        bool     built          = false;
        bool     rebuilding     = false;
        uint32_t leaf_count     = 0;
        uint32_t cell_count     = 0;
        uint32_t tlas_instances = 0;
        uint64_t triangle_count = 0;
        uint32_t dirty_total    = 0;
        uint32_t dirty_built    = 0;
        float    rebuild_frames_est  = 0.0f;
        float    step_ms             = 0.0f;
        uint32_t last_rebuild_frames = 0;

        float    blas_record_cpu_ms = 0.0f;
        float    plan_ms            = 0.0f;
        float    blas_gpu_ms        = 0.0f;
        float    tlas_gpu_ms        = 0.0f;

        uint32_t cells_pending        = 0;
        uint32_t cells_ready          = 0;
        uint32_t cells_recorded       = 0;
        uint32_t cells_recorded_total = 0;

        uint32_t cells_dropped        = 0;
        uint32_t geo_free_leaves      = 0;
        uint32_t stable_id_peak       = 0;
    };
    const Stats& stats() const { return m_stats; }

private:
    GenerationParams make_params() const;
    void assign_stable_ids(Generation& g);
    void record_tlas(const SceneInstanceDesc* scene, uint32_t scene_count,
                     uint32_t terrain_hit_group, uint32_t external_hit_group,
                     ID3D12GraphicsCommandList4* compute_cl);
    uint32_t external_capacity() const { return m_external ? m_external->instance_capacity() : 0u; }

    IExternalStream* m_external = nullptr;
    DeviceContext*  m_ctx    = nullptr;
    ID3D12Device5*  m_device = nullptr;
    StreamConfig    m_cfg;
    uint32_t        m_frame  = 0;
    DVec3           m_sceneOrigin{};

    TerrainGeoPool  m_geoPool;
    ID3D12Resource* m_instanceProps       = nullptr;
    uint32_t        m_terrainPropsBase    = 0;
    uint32_t        m_terrainMatIDBase    = 0;
    uint32_t        m_terrainTriLightBase = 0;

    uint32_t                     m_rockPropsBase = 0;
    std::vector<RockVariantGPU>  m_rockVariants;
    std::vector<RockInstance>    m_rockInstances;

    Generation        m_live;
    GenerationBuilder m_builder;
    bool              m_haveLive = false;
    DVec3             m_liveCamPos{};
    DVec3             m_rebuildTargetPos{};
    StableIdMap       m_ids;

    ThroughputEstimator m_throughput;
    DVec3               m_camVel{};
    DVec3               m_prevCamPos{};
    bool                m_havePrev     = false;
    uint32_t            m_triggerFrame = 0;

    struct Retiring { Generation gen; uint32_t dropFrame = 0; };
    std::vector<Retiring> m_retiring;

    TlasBuilder         m_tlas;
    HeightmapCubemap    m_heightmap;

    ComPtr<ID3D12Resource> m_terrainTable;
    TerrainSlotGPU*        m_terrainTableMapped = nullptr;
    uint64_t               m_terrainNodePrev[MAX_TERRAIN_CELLS] = {};
    uint64_t               m_curNode[MAX_TERRAIN_CELLS] = {};

    static constexpr uint32_t TS_RING     = 4;
    static constexpr uint32_t TS_PER_SLOT = 3;
    ComPtr<ID3D12QueryHeap>   m_queryHeap;
    ComPtr<ID3D12Resource>    m_tsReadback;
    uint64_t*                 m_tsReadbackMapped = nullptr;
    uint64_t                  m_tsFreq = 0;
    struct TsSlot { uint64_t fence = 0; bool pending = false; };
    TsSlot                    m_tsRing[TS_RING]{};
    uint32_t                  m_tsWrite = 0;

    Stats m_stats;

    WorkerPool m_workers;
};

}
