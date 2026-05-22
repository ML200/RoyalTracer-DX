#pragma once
//====================================
//PLANET - GENERATION
//====================================
//A Generation is one complete, renderable planet state: a restricted-quadtree
//leaf cut, its BLAS cell cut, and one built BLAS per cell.
//
//GenerationBuilder builds a Generation incrementally:
//  begin()        - compute the target leaf + cell cut; against a LIVE
//                   generation, diff it (clean cells reuse the LIVE BLAS,
//                   dirty cells are queued).
//  step()         - tessellate + record the BLAS builds for a budget of dirty
//                   cells onto a compute command list (parallel CPU tessellation).
//  on_submitted() - the just-recorded batch executes under a compute fence.
//  reclaim()      - free a batch's transient buffers once its fence retires.
//  done() -> take() - yields the finished Generation.
//
//This drives both the first generation (synchronous: step to completion with
//CPU waits) and ping-pong rebuilds (one step per frame, fence-polled).

#include <cstdint>
#include <vector>
#include "coordinate_system.h"
#include "cube_sphere.h"
#include "restricted_quadtree.h"
#include "blas_cells.h"
#include "heightmap_source.h"
#include "worker_pool.h"
#include "blas_pool.h"        // create_buffer, HEAP_*, PLANET_BLAS_BUILD_FLAGS, ComPtr, <d3d12.h>

namespace planet {

struct GenerationParams {
    QuadtreeParams quadtree{};
    CellCutParams  cells{};
};

//one renderable terrain cell: its BLAS, world anchor, quadtree node (the
//world-stable identity), and the stable terrain-table id assigned to that node.
struct CellInstance {
    ComPtr<ID3D12Resource>    blas;
    D3D12_GPU_VIRTUAL_ADDRESS blas_va   = 0;
    DVec3                     anchor_world{};
    uint64_t                  node_id   = INVALID_NODE;
    uint32_t                  stable_id = 0;
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
    //begin a build for this camera. live==nullptr -> first generation (every
    //cell dirty); else diff against live and reuse its clean cells.
    void begin(ID3D12Device5* device, const GenerationParams& params,
               const CameraView& cam, const Generation* live);

    //tessellate (parallel) + record BLAS builds for up to 'budget' not-yet-built
    //dirty cells onto compute_cl. Returns the number of cells recorded.
    uint32_t step(ID3D12Device5* device, const IHeightmapSource& heightmap,
                  WorkerPool& workers, ID3D12GraphicsCommandList4* compute_cl,
                  uint32_t budget);

    //the batch step() just recorded executes under this compute-fence value.
    void on_submitted(uint64_t compute_fence);
    //free transient buffers of batches whose build fence has retired.
    void reclaim(uint64_t completed_fence);

    bool     active()       const { return m_active; }
    bool     all_recorded() const { return m_cursor >= (uint32_t)m_dirty.size(); }
    bool     done()         const { return m_active && m_built >= (uint32_t)m_dirty.size(); }
    uint32_t dirty_total()  const { return (uint32_t)m_dirty.size(); }
    uint32_t dirty_built()  const { return m_built; }

    //Last step()'s CPU timing breakdown (ms). 'tess' covers per-cell upload-buffer
    //allocate/map + scratch alloc + parallel tessellation; 'blas_record' covers
    //the BLAS-descriptor build + BuildRaytracingAccelerationStructure record loop.
    float    tess_ms()        const { return m_lastTessMs; }
    float    blas_record_ms() const { return m_lastBlasRecordMs; }

    Generation take();                           // move out the finished generation; ends the build

private:
    bool             m_active = false;
    Generation       m_gen;
    GenerationParams m_params;
    std::vector<uint32_t> m_dirty;               // cell indices still to build
    uint32_t         m_cursor = 0;               // next m_dirty[] entry to record
    uint32_t         m_built  = 0;               // dirty cells whose build fence retired

    std::vector<uint64_t> m_resultSize;          // BLAS result size per geometry count K
    uint64_t              m_maxScratch = 0;

    struct Batch {
        uint64_t fence = 0;
        uint32_t count = 0;                      // dirty cells this batch built
        std::vector<ComPtr<ID3D12Resource>> transient;   // upload + scratch, freed on fence
    };
    std::vector<Batch> m_batches;                // recorded, fence pending
    Batch              m_pending;                // accumulated by step(), sealed by on_submitted()

    //per-step CPU timing breakdown (ms); published via tess_ms / blas_record_ms.
    float              m_lastTessMs       = 0.0f;
    float              m_lastBlasRecordMs = 0.0f;
};

} // namespace planet
