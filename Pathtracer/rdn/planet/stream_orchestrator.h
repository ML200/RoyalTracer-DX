#pragma once
//====================================
//PLANET - STREAM ORCHESTRATOR
//====================================
//The planet's per-frame driver and a Renderer member (m_planet). Owns the LIVE
//Generation (rendered), a GenerationBuilder (an in-flight ping-pong rebuild),
//and the unified per-frame TLAS.
//
//Per-frame contract (Renderer::RenderFrame):
//  begin_frame  - first call: build the LIVE generation synchronously (init
//                 stall). Later calls: reclaim, swap a finished rebuild in as
//                 the new LIVE generation, and trigger a rebuild when the
//                 camera has drifted far enough.
//  submit_work  - advance an in-flight rebuild by one per-frame budget of dirty
//                 cells, then rebuild + submit the unified TLAS (which always
//                 renders the LIVE generation) on the planet compute queue.
//  end_frame    - nothing.
//
//Ping-pong: a rebuild diffs the target cut against LIVE, rebuilds only dirty
//cells, reuses clean cells' BLASes (shared ComPtr), and swaps atomically when
//every dirty cell is built. The retired generation is held a few frames so no
//in-flight TLAS dangles. Terrain InstanceID is a stable per-node id, so a cell
//reused across a swap keeps its id and ReSTIR/DLSS history survives.

#include <cstdint>
#include <vector>
#include <unordered_map>
#include "coordinate_system.h"
#include "cube_sphere.h"
#include "heightmap_procedural.h"
#include "worker_pool.h"
#include "tlas_builder.h"
#include "generation.h"

struct DeviceContext;

namespace planet {

//Instance-ID base for terrain in the unified TLAS. Scene mesh instances keep
//0..N; anything >= this is a terrain cell. The shader branches on it.
constexpr uint32_t TERRAIN_INSTANCE_BASE = 1u << 20;

//Upper bound on the simultaneously-live terrain cells - sizes the terrain
//table and the terrain portion of the TLAS.
constexpr uint32_t MAX_TERRAIN_CELLS = 4096;

struct StreamConfig {
    //master on/off. false: begin_frame is a no-op (no generation is built),
    //and record_tlas emits a scene-only unified TLAS with no terrain
    //instances - the renderer runs exactly as if there were no planet.
    bool     enabled = true;
    PlanetGeometry planet{};                     // FP64 world centre + surface radius
    uint8_t  min_lod = 0;                        // coarsest leaf (far side)
    uint8_t  max_lod = MAX_LOD;                  // finest leaf (under the camera)
    uint32_t max_triangles = 3000000u;           // planet-wide triangle budget; the
                                                 // quadtree auto-tunes its leaf cut
                                                 // to the nearest fit at or under it
    uint32_t max_leaves_per_cell = 64;            // BLAS triangle budget (leaves per cell)
    double   max_cell_radius_m = 1e30;           // FP32-precision cap on a cell's extent
    uint32_t max_scene_instances = 4096;         // unified-TLAS scene-instance allowance
    float    heightmap_amplitude = 1000.0f;       // procedural fBm peak height (metres)
    float    heightmap_frequency = 1000.0f;        // fBm base frequency on the unit sphere
    float    rebuild_trigger_m = 10.0f;        // camera drift that triggers a ping-pong rebuild
    uint32_t build_budget = 2;                   // dirty cells built per frame during a rebuild
    bool     predict = true;                     // aim a rebuild at the predicted swap-time camera
};

//One scene mesh instance handed to the orchestrator for the unified TLAS.
//'transform' is a row-major 3x4 object->world matrix, scene-origin-relative.
struct SceneInstanceDesc {
    D3D12_GPU_VIRTUAL_ADDRESS blas = 0;
    float    transform[12]   = {};
    uint32_t instance_id     = 0;
    uint32_t hit_group_index = 0;
    uint32_t flags           = 0;   // D3D12_RAYTRACING_INSTANCE_FLAGS
};

//One terrain-table entry, indexed by (terrain InstanceID - TERRAIN_INSTANCE_BASE)
//= the cell's stable id. node_lo/node_hi are the 64-bit packed quadtree node;
//'changed' is 1 the frame a different cell takes that id (a ping-pong swap
//reshuffled it), so the GI passes drop that cell's stale temporal samples.
struct TerrainSlotGPU {
    uint32_t node_lo;   // packed quadtree node_id, low  32 bits (face | lod | x)
    uint32_t node_hi;   // packed quadtree node_id, high 32 bits (y)
    uint32_t changed;
};

//Assigns each terrain cell (keyed by its quadtree node) a stable index used as
//its terrain-table slot / terrain InstanceID. A cell reused across a ping-pong
//swap keeps its index; vanished cells' indices are recycled.
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
    //recycle the id of every node for which keep(node_id) returns false.
    template <typename Fn> void retain(Fn&& keep) {
        for (auto it = m_id.begin(); it != m_id.end(); ) {
            if (!keep(it->first)) { m_free.push_back(it->second); it = m_id.erase(it); }
            else ++it;
        }
    }

private:
    std::unordered_map<uint64_t, uint32_t> m_id;     // node_id -> stable id
    std::vector<uint32_t>                  m_free;   // recycled ids
    uint32_t                               m_next = 0;
};

//Tracks how long ping-pong rebuilds take so the orchestrator can predict where
//the camera will be when a rebuild completes and aim the rebuilt LOD there.
struct ThroughputEstimator {
    float    rebuild_frames      = 20.0f;   // EWMA: a rebuild's trigger->swap frame span
    float    step_ms             = 0.0f;    // EWMA: per-frame CPU build (step) cost - diagnostic
    uint32_t last_rebuild_frames = 0;       // the most recent observed span

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
    void init(ID3D12Device5* device, DeviceContext* ctx, const StreamConfig& cfg);

    void begin_frame(uint32_t frame_index, const CameraView& cam);
    void submit_work(const SceneInstanceDesc* scene, uint32_t scene_count,
                     uint32_t terrain_hit_group);
    void end_frame();

    //unified TLAS - the renderer points the SceneBVH SRV here once, after init.
    ID3D12Resource*           tlas_result()  const { return m_tlas.result(); }
    D3D12_GPU_VIRTUAL_ADDRESS tlas_address() const { return m_tlas.tlas_address(); }
    //terrain table - the renderer creates one SRV over this, after init.
    ID3D12Resource* terrain_table() const { return m_terrainTable.Get(); }
    static constexpr uint32_t terrain_table_count() { return MAX_TERRAIN_CELLS; }

    struct Stats {
        bool     built          = false;
        bool     rebuilding     = false;
        uint32_t leaf_count     = 0;
        uint32_t cell_count     = 0;
        uint32_t tlas_instances = 0;
        uint64_t triangle_count = 0;
        uint32_t dirty_total    = 0;   // dirty cells in the in-flight rebuild
        uint32_t dirty_built    = 0;
        float    rebuild_frames_est  = 0.0f;   // EWMA estimate of a rebuild's frame span
        float    step_ms             = 0.0f;   // EWMA per-frame CPU build cost
        uint32_t last_rebuild_frames = 0;

        //Per-frame timing breakdown (ms). With async tessellation the render
        //thread's per-frame rebuild cost IS blas_record_cpu_ms - tess + alloc
        //moved to the worker pool. plan_ms is the one-off plan-job duration
        //(also off-thread). blas_gpu / tlas_gpu are GPU timestamps on the
        //planet compute queue, fence-gated readback (lag a few frames). Raw
        //last-frame values so hitches stay visible.
        float    blas_record_cpu_ms = 0.0f;
        float    plan_ms            = 0.0f;
        float    blas_gpu_ms        = 0.0f;
        float    tlas_gpu_ms        = 0.0f;

        //Per-cell async-pipeline counts. cells_pending + cells_ready +
        //cells_recorded_total + cells_built = dirty_total in steady state.
        uint32_t cells_pending        = 0;   // tess in flight (workers or queued)
        uint32_t cells_ready          = 0;   // tess done, BLAS not yet recorded
        uint32_t cells_recorded       = 0;   // BLAS recorded THIS frame
        uint32_t cells_recorded_total = 0;   // BLAS recorded, fence pending
    };
    const Stats& stats() const { return m_stats; }

private:
    GenerationParams make_params() const;
    void assign_stable_ids(Generation& g);
    void record_tlas(const SceneInstanceDesc* scene, uint32_t scene_count,
                     uint32_t terrain_hit_group, ID3D12GraphicsCommandList4* compute_cl);

    DeviceContext*  m_ctx    = nullptr;
    ID3D12Device5*  m_device = nullptr;
    StreamConfig    m_cfg;
    uint32_t        m_frame  = 0;
    DVec3           m_sceneOrigin{};                 // unified-TLAS origin, set each begin_frame

    Generation        m_live;                        // the rendered generation
    GenerationBuilder m_builder;                      // an in-flight ping-pong rebuild
    bool              m_haveLive = false;
    DVec3             m_liveCamPos{};                 // camera the LIVE generation was built for
    DVec3             m_rebuildTargetPos{};           // camera the in-flight rebuild targets
    StableIdMap       m_ids;

    //throughput estimate + camera prediction: aim a rebuild's LOD at where the
    //camera will be when the rebuild completes, not where it is at trigger time.
    ThroughputEstimator m_throughput;
    DVec3               m_camVel{};                   // EWMA of the per-frame camera world delta
    DVec3               m_prevCamPos{};
    bool                m_havePrev     = false;
    uint32_t            m_triggerFrame = 0;            // frame the in-flight rebuild was triggered

    //a retired generation, kept a few frames so no in-flight TLAS dangles.
    struct Retiring { Generation gen; uint32_t dropFrame = 0; };
    std::vector<Retiring> m_retiring;

    TlasBuilder         m_tlas;
    HeightmapProcedural m_heightmap;

    //terrain table: one TerrainSlotGPU per stable id, persistently-mapped upload heap.
    ComPtr<ID3D12Resource> m_terrainTable;
    TerrainSlotGPU*        m_terrainTableMapped = nullptr;
    uint64_t               m_terrainNodePrev[MAX_TERRAIN_CELLS] = {};   // last frame's per-slot node
    uint64_t               m_curNode[MAX_TERRAIN_CELLS] = {};           // this frame's, scratch

    //GPU timestamp queries on the planet compute queue (BLAS + TLAS GPU times).
    //Ring of TS_RING slots, each holds TS_PER_SLOT timestamps; each frame writes
    //the next slot and reads back the slot it is about to overwrite (gated on
    //the planet-compute fence so we never read GPU-pending data). TS_RING >
    //frames-in-flight, so a slot's fence has retired by the time we re-use it.
    static constexpr uint32_t TS_RING     = 4;
    static constexpr uint32_t TS_PER_SLOT = 3;       // T_start, after-BLAS, after-TLAS
    ComPtr<ID3D12QueryHeap>   m_queryHeap;
    ComPtr<ID3D12Resource>    m_tsReadback;
    uint64_t*                 m_tsReadbackMapped = nullptr;
    uint64_t                  m_tsFreq = 0;          // compute-queue timestamp ticks per second
    struct TsSlot { uint64_t fence = 0; bool pending = false; };
    TsSlot                    m_tsRing[TS_RING]{};
    uint32_t                  m_tsWrite = 0;

    Stats m_stats;

    //declared LAST: ~WorkerPool joins its threads before the members the
    //generation build references are destroyed.
    WorkerPool m_workers;
};

} // namespace planet
