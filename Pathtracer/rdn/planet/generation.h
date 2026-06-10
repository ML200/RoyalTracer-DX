#pragma once
//====================================
//PLANET - GENERATION
//====================================
//A Generation is one complete, renderable planet state: a restricted-quadtree
//leaf cut, its BLAS cell cut, and one built BLAS per cell.
//
//GenerationBuilder builds a Generation FULLY ASYNC w.r.t. the render thread:
//
//  begin()        - enqueue a 'plan' job (LOD select + cell cut + diff against
//                   LIVE) onto the worker pool. Returns IMMEDIATELY.
//  poll()         - render-thread tick: when the plan job finishes, enqueues
//                   one tessellation job per dirty cell (each allocates its
//                   own upload buffers + tessellates + sets state=Ready).
//  record_ready_blas() - render-thread call inside submit_work: for every
//                   dirty cell with state==Ready, records the BLAS build onto
//                   the planet compute list. This is the only synchronous
//                   render-thread CPU work for a rebuild.
//  on_submitted() - the just-recorded batch executes under a compute fence.
//  reclaim()      - cells whose fence retired transition Ready->Built;
//                   recorded-batch transient buffers are freed.
//  done() -> take() - every dirty cell built, full generation yields.
//
//The render thread never tessellates and never allocates upload buffers for
//terrain - those move to the worker pool. The render thread only records BLAS
//commands and the unified TLAS, which is sub-ms even with hundreds of cells.
//
//Memory-ordering note: tess jobs write to UPLOAD-heap (write-combining) memory
//and signal completion via an atomic state flag. An _mm_sfence() before the
//state store flushes the WC writes so the render thread (which sees Ready
//via acquire-load) is guaranteed they reached the heap before recording a
//BLAS build that references them. The earlier async-tess attempt failed
//without this fence (GPU corruption).

#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>
#include "coordinate_system.h"
#include "cube_sphere.h"
#include "restricted_quadtree.h"
#include "blas_cells.h"
#include "heightmap_source.h"
#include "worker_pool.h"
#include "blas_pool.h"        // create_buffer, HEAP_*, PLANET_BLAS_BUILD_FLAGS, ComPtr, <d3d12.h>
#include "terrain_geo_pool.h"

namespace planet {

struct GenerationParams {
    QuadtreeParams quadtree{};
    CellCutParams  cells{};
};

//RAII owner of a terrain geometry-pool slot. Frees the K-leaf range back to the
//pool when the LAST CellInstance referencing it is destroyed. A clean cell
//reused across a ping-pong swap shares this via shared_ptr, so the slot (and
//its vertex/index bytes in the combined buffer) stays valid as long as ANY
//generation still renders the cell. All frees happen on the render thread,
//where generations are torn down - the pool is not thread-safe.
struct GeoSlot {
    TerrainGeoPool* pool       = nullptr;
    uint32_t        leaf_off   = TERRAIN_GEO_INVALID;
    uint32_t        leaf_count = 0;
    GeoSlot(TerrainGeoPool* p, uint32_t off, uint32_t k)
        : pool(p), leaf_off(off), leaf_count(k) {}
    ~GeoSlot() { if (pool && leaf_off != TERRAIN_GEO_INVALID) pool->free(leaf_off, leaf_count); }
    //non-copyable: the slot is owned once and shared via shared_ptr, so the
    //free() in the destructor must run exactly once (a copy would double-free).
    GeoSlot(const GeoSlot&)            = delete;
    GeoSlot& operator=(const GeoSlot&) = delete;
};

//one renderable terrain cell: its BLAS, world anchor, quadtree node (the
//world-stable identity), the stable id assigned to that node (= its terrain
//InstanceProperties slot / TLAS InstanceID offset), and its slot in the unified
//geometry buffer.
struct CellInstance {
    ComPtr<ID3D12Resource>    blas;
    D3D12_GPU_VIRTUAL_ADDRESS blas_va   = 0;
    DVec3                     anchor_world{};
    uint64_t                  node_id   = INVALID_NODE;
    uint32_t                  stable_id = 0;
    //unified-buffer geometry slot (set when the cell's geometry is built;
    //copied along with the shared BLAS when a clean cell is reused).
    std::shared_ptr<GeoSlot>  geo;
    uint32_t                  vtx_base_elems = 0;   // first vertex element in the combined buffer
    uint32_t                  idx_base_elems = 0;   // first index  element in the combined buffer
    uint32_t                  geo_tri_count  = 0;   // leaf_count * MAX_CHUNK_TRIS
    //The cell's leaves, as a range into BlasCellSet::cell_leaves(). Usually the
    //whole cellSet cell; a SPLIT fallback sub-cell (when the pool couldn't fit
    //the whole cell contiguously) carries a sub-range. Carrying the range here
    //decouples a CellInstance from the 1:1 cellSet mapping so sub-cells can be
    //appended past the cellSet cell count.
    uint32_t                  leaf_begin     = 0;
    uint32_t                  leaf_count     = 0;
};

//a complete, renderable planet state. Produced by GenerationBuilder; the
//orchestrator holds one as the LIVE generation.
struct Generation {
    RestrictedQuadtree        qt;
    BlasCellSet               cellSet;
    std::vector<CellInstance> cells;             // parallel to cellSet.cells()
    uint32_t                  leaf_count     = 0;
    uint64_t                  triangle_count = 0;
    bool valid() const { return !cells.empty(); }
};

class GenerationBuilder {
public:
    //Per-rebuild state. Streaming = plan finished, tess jobs in flight, BLAS
    //builds being recorded as cells complete; the rebuild is 'done' when every
    //dirty cell has retired its BLAS fence (state == Built).
    enum class State : uint8_t { Idle, Planning, Streaming };

    //Bind the unified geometry buffers once (after scene load). The builder
    //allocates each dirty cell a slot from `pool`, tessellates straight into
    //the cell's region of the mapped combined vertex/index buffers, and builds
    //a single-geometry BLAS over it (vtxVA = combined vertex buffer base, so the
    //tessellator's absolute indices resolve). Persists across begin() calls.
    void set_geometry(TerrainGeoPool* pool, uint8_t* vtxMapped, uint8_t* idxMapped,
                      D3D12_GPU_VIRTUAL_ADDRESS vtxVA, D3D12_GPU_VIRTUAL_ADDRESS idxVA,
                      uint32_t combinedVertexCount) {
        m_pool = pool; m_vtxMapped = vtxMapped; m_idxMapped = idxMapped;
        m_vtxVA = vtxVA; m_idxVA = idxVA; m_combinedVertexCount = combinedVertexCount;
    }

    //Begin an async build. Enqueues the plan job on 'workers'; returns
    //immediately. 'live' is captured by pointer and must outlive the build
    //(the orchestrator only swaps LIVE on done()). 'heightmap' is captured
    //similarly for the tess jobs.
    void begin(ID3D12Device5* device, const GenerationParams& params,
               const CameraView& cam, const Generation* live,
               const IHeightmapSource& heightmap, WorkerPool& workers);

    //Render-thread tick: when the plan job finishes, transitions Planning ->
    //Streaming and enqueues one tess job per dirty cell. Cheap; safe to call
    //every frame.
    void poll();

    //Record BLAS builds onto compute_cl for every dirty cell with state==Ready
    //(tess complete, BLAS not yet recorded), up to 'budget' cells. Returns
    //cells recorded. Uses one shared scratch buffer per call; UAV barriers
    //serialise it. The call's transient buffers are sealed into a Batch by
    //on_submitted(). budget=0 means "no cap" (drain everything Ready).
    //
    //Rate-limiting is GPU-side throughput control: tess jobs all finish
    //roughly together (parallel workers), so without a cap the render thread
    //records dozens of BLAS builds at once and the planet compute submission
    //balloons - the graphics queue waits that whole submission via the planet
    //compute fence. A small budget paces the BLAS work across frames.
    uint32_t record_ready_blas(ID3D12GraphicsCommandList4* compute_cl,
                               uint32_t budget);

    //The just-recorded batch executes under this compute-fence value.
    void on_submitted(uint64_t compute_fence);
    //Cells whose batch fence retired transition Ready -> Built; the batch's
    //transient upload/scratch buffers are freed (the GPU is done reading them).
    void reclaim(uint64_t completed_fence);

    State    state()        const { return m_state; }
    bool     active()       const { return m_state != State::Idle; }
    bool     all_recorded() const;
    bool     done()         const;
    uint32_t dirty_total()  const { return (uint32_t)m_dirty.size(); }
    uint32_t dirty_built()  const;
    uint32_t dirty_ready()  const;               // tess done, BLAS not yet recorded
    uint32_t dirty_recorded() const;             // BLAS recorded, fence pending
    uint32_t dirty_tessellating() const;         // workers still tessellating

    //Last record_ready_blas() duration (ms); render-thread CPU cost per frame
    //during a rebuild. plan_ms() is the most recent plan-job duration (one
    //per rebuild). Tessellation runs on workers in parallel with the render
    //thread, so there is no per-frame render-thread tess cost to report.
    float    blas_record_ms() const { return m_lastBlasRecordMs; }
    float    plan_ms()        const { return m_planMs.load(std::memory_order_acquire); }

    Generation take();                           // ends the build; yields the generation

private:
    void plan_job_(GenerationParams params, CameraView cam,
                   const Generation* live);
    void tess_job_(uint32_t dirty_idx);

    State            m_state = State::Idle;
    Generation       m_gen;
    GenerationParams m_params;
    //Scene origin captured at begin() for the lifetime of this rebuild. The
    //tessellator pre-compensates each vertex by FP32(anchor - scene_origin)
    //quantisation error against THIS value; the renderer's TLAS instance
    //transform also uses (anchor - sceneOrigin) with the current scene
    //origin. Both must agree (within an FP32-exact 1 km snap delta) for the
    //compensation to be valid - they do, because cam.scene_origin only
    //changes on the renderer's 1 km grid snaps and the snap delta is
    //FP32-exact, leaving the per-vertex correction `e` invariant.
    DVec3            m_sceneOrigin{};
    ID3D12Device5*           m_device    = nullptr;
    const IHeightmapSource*  m_heightmap = nullptr;
    WorkerPool*              m_workers   = nullptr;

    std::atomic<bool>  m_planDone{false};         // plan job -> render thread
    std::atomic<float> m_planMs{0.0f};            // most recent plan duration

    std::vector<uint32_t> m_dirty;                // indices into m_gen.cells
    std::vector<uint64_t> m_resultSize;           // BLAS result size per geometry count K
    uint64_t              m_maxScratch = 0;

    //Per-dirty-cell async-build state. Lifetime: one rebuild. Heap-allocated
    //in poll() once the plan completes so the atomic states aren't moved. The
    //tessellator writes straight into the cell's slot of the mapped combined
    //buffers (no per-cell upload buffer), so only the state flag is needed.
    struct CellBuild {
        std::atomic<uint8_t>   state{0};          // 0=Pending 1=Ready 2=Recorded 3=Built
    };
    std::unique_ptr<CellBuild[]> m_cellBuilds;
    uint32_t                     m_cellBuildCount = 0;

    //Unified geometry binding (set_geometry, persists across rebuilds).
    TerrainGeoPool*           m_pool       = nullptr;
    uint8_t*                  m_vtxMapped  = nullptr;   // combined vertex buffer base (mapped)
    uint8_t*                  m_idxMapped  = nullptr;   // combined index  buffer base (mapped)
    D3D12_GPU_VIRTUAL_ADDRESS m_vtxVA      = 0;         // combined vertex buffer GPU VA
    D3D12_GPU_VIRTUAL_ADDRESS m_idxVA      = 0;         // combined index  buffer GPU VA
    uint32_t                  m_combinedVertexCount = 0;

    //A Batch is one frame's record_ready_blas() output: the cells recorded
    //together and their transient (vtx/idx/scratch) buffers. The buffers stay
    //alive until the batch's fence retires; the cells transition to Built
    //then too.
    struct Batch {
        uint64_t fence = 0;
        std::vector<uint32_t> recordedDirty;      // dirty_idx values
        std::vector<ComPtr<ID3D12Resource>> transients;
    };
    std::vector<Batch> m_batches;                 // recorded, fence pending
    Batch              m_pending;                 // accumulating; sealed by on_submitted()

    float              m_lastBlasRecordMs = 0.0f;
};

} // namespace planet
