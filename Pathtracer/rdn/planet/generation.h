#pragma once

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
#include "blas_pool.h"
#include "terrain_geo_pool.h"

namespace planet {

struct GenerationParams {
    QuadtreeParams quadtree{};
    CellCutParams  cells{};
};

// Shared ownership keeps geometry alive across generation swaps.
struct GeoSlot {
    TerrainGeoPool* pool       = nullptr;
    uint32_t        leaf_off   = TERRAIN_GEO_INVALID;
    uint32_t        leaf_count = 0;
    GeoSlot(TerrainGeoPool* p, uint32_t off, uint32_t k)
        : pool(p), leaf_off(off), leaf_count(k) {}
    ~GeoSlot() { if (pool && leaf_off != TERRAIN_GEO_INVALID) pool->free(leaf_off, leaf_count); }
    GeoSlot(const GeoSlot&)            = delete;
    GeoSlot& operator=(const GeoSlot&) = delete;
};

struct CellInstance {
    ComPtr<ID3D12Resource>    blas;
    D3D12_GPU_VIRTUAL_ADDRESS blas_va   = 0;
    DVec3                     anchor_world{};
    uint64_t                  node_id   = INVALID_NODE;
    uint32_t                  stable_id = 0;
    std::shared_ptr<GeoSlot>  geo;
    uint32_t                  vtx_base_elems = 0;
    uint32_t                  idx_base_elems = 0;
    uint32_t                  geo_tri_count  = 0;
    uint32_t                  leaf_begin     = 0;
    uint32_t                  leaf_count     = 0;
};

struct Generation {
    RestrictedQuadtree        qt;
    BlasCellSet               cellSet;
    std::vector<CellInstance> cells;
    uint32_t                  leaf_count     = 0;
    uint64_t                  triangle_count = 0;
    // A generation is drawable once at least one cell has geometry.
    bool valid() const { return !cells.empty(); }
};

class GenerationBuilder {
public:
    enum class State : uint8_t { Idle, Planning, Streaming };

    void set_geometry(TerrainGeoPool* pool, uint8_t* vtxMapped, uint8_t* idxMapped,
                      D3D12_GPU_VIRTUAL_ADDRESS vtxVA, D3D12_GPU_VIRTUAL_ADDRESS idxVA,
                      uint32_t combinedVertexCount) {
        m_pool = pool; m_vtxMapped = vtxMapped; m_idxMapped = idxMapped;
        m_vtxVA = vtxVA; m_idxVA = idxVA; m_combinedVertexCount = combinedVertexCount;
    }

    // Starts planning asynchronously against the currently live generation.
    void begin(ID3D12Device5* device, const GenerationParams& params,
               const CameraView& cam, const Generation* live,
               const IHeightmapSource& heightmap, WorkerPool& workers);

    // Advances planning, tessellation, and completed resource state.
    void poll();

    // Records at most budget builds and seals their resources on submission.
    uint32_t record_ready_blas(ID3D12GraphicsCommandList4* compute_cl,
                               uint32_t budget);

    void on_submitted(uint64_t compute_fence);
    void reclaim(uint64_t completed_fence);

    State    state()        const { return m_state; }
    bool     active()       const { return m_state != State::Idle; }
    bool     all_recorded() const;
    bool     done()         const;
    uint32_t dirty_total()  const { return (uint32_t)m_dirty.size(); }
    uint32_t dirty_built()  const;
    uint32_t dirty_ready()  const;
    uint32_t dirty_recorded() const;
    uint32_t dirty_tessellating() const;

    float    blas_record_ms() const { return m_lastBlasRecordMs; }
    float    plan_ms()        const { return m_planMs.load(std::memory_order_acquire); }

    Generation take();

private:
    void plan_job_(GenerationParams params, CameraView cam,
                   const Generation* live);
    void tess_job_(uint32_t dirty_idx);

    State            m_state = State::Idle;
    Generation       m_gen;
    GenerationParams m_params;
    DVec3            m_sceneOrigin{};
    ID3D12Device5*           m_device    = nullptr;
    const IHeightmapSource*  m_heightmap = nullptr;
    WorkerPool*              m_workers   = nullptr;

    std::atomic<bool>  m_planDone{false};
    std::atomic<float> m_planMs{0.0f};

    std::vector<uint32_t> m_dirty;
    std::vector<uint64_t> m_resultSize;
    uint64_t              m_maxScratch = 0;

    struct CellBuild {
        std::atomic<uint8_t>   state{0};
    };
    std::unique_ptr<CellBuild[]> m_cellBuilds;
    uint32_t                     m_cellBuildCount = 0;

    TerrainGeoPool*           m_pool       = nullptr;
    uint8_t*                  m_vtxMapped  = nullptr;
    uint8_t*                  m_idxMapped  = nullptr;
    D3D12_GPU_VIRTUAL_ADDRESS m_vtxVA      = 0;
    D3D12_GPU_VIRTUAL_ADDRESS m_idxVA      = 0;
    uint32_t                  m_combinedVertexCount = 0;

    struct Batch {
        uint64_t fence = 0;
        std::vector<uint32_t> recordedDirty;
        std::vector<ComPtr<ID3D12Resource>> transients;
    };
    std::vector<Batch> m_batches;
    Batch              m_pending;

    float              m_lastBlasRecordMs = 0.0f;
};

}
