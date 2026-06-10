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

namespace planet {

//Instance-ID base for terrain in the unified TLAS. Scene mesh instances keep
//0..N; anything >= this is a terrain cell. The shader branches on it.
constexpr uint32_t TERRAIN_INSTANCE_BASE = 1u << 20;

//Upper bound on the simultaneously-live terrain cells - sizes the terrain
//table and the terrain portion of the TLAS / instanceProps.
constexpr uint32_t MAX_TERRAIN_CELLS = 4096;

//Upper bound on simultaneously-live camera-streamed scatter rocks - sizes the
//rock portion of the unified TLAS and the rock instanceProps range. Must match
//Scene::ReserveRocks's count and RockScatterConfig::max_rocks.
constexpr uint32_t MAX_ROCK_INSTANCES = 4096;

//Leaf-slot capacity of the terrain geometry pool in the unified buffers. Sized
//for ~live (max_triangles / MAX_CHUNK_TRIS) plus ping-pong + retire headroom.
//At 6144 slots: ~134 MB vertices + ~151 MB indices in the combined UPLOAD
//buffer. Lower it if VRAM is tight (or once the scene region moves to DEFAULT).
constexpr uint32_t TERRAIN_GEO_LEAF_SLOTS = 6144;

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
    //Leaves per BLAS cell. Large = few BLASes (good trace/build perf). A cell
    //needs a CONTIGUOUS run of geometry-pool leaf slots; if the pool is too
    //fragmented to fit one, the builder falls back to splitting that cell into
    //smaller standalone sub-cells that DO fit (see GenerationBuilder::poll) -
    //so big cells stay the fast path and fragmentation never punches a hole.
    uint32_t max_leaves_per_cell = 128;            // BLAS triangle budget (leaves per cell)
    double   max_cell_radius_m = 1e30;           // FP32-precision cap on a cell's extent
    uint32_t max_scene_instances = 4096;         // unified-TLAS scene-instance allowance
    //Directory containing the baker output: manifest.json + 6
    //elevation_face<N>.r32 cube-face files. Resolved relative to the runtime
    //working directory; CMake copies include/terrain/ here on build.
    std::string heightmap_dir   = "./terrain";
    float    rebuild_trigger_m = 1.0f;        // camera drift that triggers a ping-pong rebuild
    //BLAS builds recorded per frame during a rebuild. 2 halves rebuild
    //latency vs 1 with a barely-perceptible per-frame GPU cost bump,
    //which lets the LOD keep up with normal camera motion.
    uint32_t build_budget = 1;
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
    //high-water mark of simultaneously-assigned ids (== the largest id+1 ever
    //needed). If this reaches MAX_TERRAIN_CELLS, cells with that id are dropped.
    uint32_t peak() const { return m_next; }

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

    //Bind the unified geometry/instance buffers AFTER scene load (the scene
    //owns them and reserves a terrain region). Terrain now shades through the
    //scene path: chunk meshes are tessellated into the combined vertex/index
    //buffers and each live cell gets an InstanceProperties entry at
    //(terrainPropsBase + stable_id), which is also its TLAS InstanceID.
    //terrainPropsBase is the FIXED scene-instance capacity, so terrain indices
    //never move when scene instances are added/removed.
    void bind_geometry(ID3D12Resource* combinedVtx, uint8_t* vtxMapped,
                       ID3D12Resource* combinedIdx, uint8_t* idxMapped,
                       ID3D12Resource* instanceProps,
                       uint32_t sceneVertexCount, uint32_t sceneIndexCount,
                       uint32_t combinedVertexCount, uint32_t terrainPropsBase,
                       uint32_t terrainLeafSlots, uint32_t terrainMatIDBase,
                       uint32_t terrainTriLightBase);

    void begin_frame(uint32_t frame_index, const CameraView& cam);
    void submit_work(const SceneInstanceDesc* scene, uint32_t scene_count,
                     uint32_t terrain_hit_group);
    void end_frame();

    //PLANET ROCKS: camera-streamed scatter rocks (rdn/planet/rock_scatter.h).
    //set_rock_variants is called once after scene load with each boulder
    //variant's BLAS + combined-buffer base offsets and the reserved
    //instanceProps base; set_rock_instances is called every frame (before
    //submit_work) with the current live set. record_tlas emits one TLAS
    //instance + InstanceProperties per live rock at (rockPropsBase + stable_id).
    struct RockVariantGPU {
        D3D12_GPU_VIRTUAL_ADDRESS blas_va = 0;
        uint32_t vertexBase = 0;   // element offset of the variant's verts in the combined buffer
        uint32_t indexBase  = 0;   // element offset of the variant's indices
        uint32_t triCount   = 0;   // opaque triangle count
    };
    void set_rock_variants(uint32_t propsBase, const std::vector<RockVariantGPU>& variants);
    void set_rock_instances(const RockInstance* insts, uint32_t count);

    //planet master on/off (StreamConfig::enabled), so the renderer can skip the
    //terrain buffer reservation / bind when terrain is disabled.
    bool enabled() const { return m_cfg.enabled; }
    //PLANET ROCKS: planet centre + radius, for the rock-scatter placement.
    PlanetGeometry planet_geometry() const { return m_cfg.planet; }

    //Terrain region the renderer must reserve in the combined scene buffers (all
    //zero when terrain is disabled). The renderer passes these to
    //Scene::ReserveTerrain and the matching counts back via bind_geometry.
    struct TerrainReservation {
        uint32_t vertexElems       = 0;   // leafSlots * MAX_CHUNK_VERTS
        uint32_t indexElems        = 0;   // leafSlots * MAX_CHUNK_TRIS*3
        uint32_t matIDElems        = 0;   // max_leaves_per_cell * MAX_CHUNK_TRIS (shared)
        uint32_t triLightElems     = 0;   // == matIDElems
        uint32_t instanceSlots     = 0;   // MAX_TERRAIN_CELLS
        uint32_t leafSlots         = 0;   // geometry-pool capacity
        //FIXED base where terrain InstanceProperties live (= the scene-instance
        //capacity). Terrain InstanceID = propsBase + stable_id, INDEPENDENT of
        //the runtime scene-instance count, so adding/removing scene instances
        //can never shift terrain into (or out of) the scene region or past the
        //SRV bound. Scene instances are capped at this value by the TLAS, so
        //[0, propsBase) holds the scene and [propsBase, propsBase+instanceSlots)
        //holds terrain - disjoint by construction.
        uint32_t propsBase         = 0;   // = max_scene_instances
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

    //unified TLAS - the renderer points the SceneBVH SRV here once, after init.
    ID3D12Resource*           tlas_result()  const { return m_tlas.result(); }
    D3D12_GPU_VIRTUAL_ADDRESS tlas_address() const { return m_tlas.tlas_address(); }
    //terrain table - the renderer creates one SRV over this, after init.
    ID3D12Resource* terrain_table() const { return m_terrainTable.Get(); }
    static constexpr uint32_t terrain_table_count() { return MAX_TERRAIN_CELLS; }

    //CPU heightmap. The renderer reads this once after init() to upload a
    //downsampled copy onto a GPU texture for the shader sampler. Read-only
    //after init; safe to call concurrently with begin_frame / submit_work.
    const HeightmapCubemap& heightmap() const { return m_heightmap; }

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

        //HOLE diagnostics. cells_dropped = LIVE cells skipped from the TLAS this
        //frame (no geometry slot, or stable_id overflow) -> visible holes.
        //geo_free_leaves = free slots in the terrain geometry pool (0 = pool
        //exhausted). stable_id_peak = StableIdMap high-water (>= MAX_TERRAIN_CELLS
        //means id overflow). Compare tlas_instances to its capacity for the third
        //drop path (TLAS instance-desc overflow).
        uint32_t cells_dropped        = 0;
        uint32_t geo_free_leaves      = 0;
        uint32_t stable_id_peak       = 0;
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

    //Unified geometry/instance binding (set by bind_geometry). m_geoPool is
    //declared HERE - before the generation-holding members below - so it is
    //destroyed AFTER them: a CellInstance's GeoSlot frees its slot back to the
    //pool in its destructor, and those run as the generations tear down.
    TerrainGeoPool  m_geoPool;
    ID3D12Resource* m_instanceProps       = nullptr; // scene instanceProperties (UPLOAD); terrain region written per frame
    uint32_t        m_terrainPropsBase    = 0;       // FIXED terrain InstanceID / props base (= scene capacity)
    uint32_t        m_terrainMatIDBase    = 0;       // shared terrain materialID region (element offset)
    uint32_t        m_terrainTriLightBase = 0;       // shared terrain triToLightId region (element offset)

    //PLANET ROCKS: variant descriptors (set once after scene load) + the current
    //live instance set (set per frame) + the reserved instanceProps base.
    uint32_t                     m_rockPropsBase = 0;
    std::vector<RockVariantGPU>  m_rockVariants;
    std::vector<RockInstance>    m_rockInstances;

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
    HeightmapCubemap    m_heightmap;

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
