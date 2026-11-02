#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>
#include "ClusterLod.h"
#include "ClusterPages.h"

namespace planet {
class TlasBuilder;
}

namespace clusters {

#if RDN_CLUSTER_API
// Cluster LOD on the GPU (Lod_v8.hlsl): every frame, per instance, the cut of its mesh's hierarchy for its view; a cluster
// BLAS over a changed cut, batched per size class with counts the GPU writes; partitioned TLAS updates for the instances
// whose BLAS changed, applied in a second in-place build after the CPU's; and feedback for the pager (requested pages,
// when traced cuts last used each group, how far over budget each group is), read back a few frames later. An instance at
// its mesh's coarsest cut traces the mesh's own BLAS. Needs a partitioned TLAS and paged hierarchies.
// An instance's cluster BLAS live in a ring of entries sized for its cut, allocated from one pool: the CPU hands an
// instance a ring of its cut's size class (a larger one as the cut grows, a smaller one as it shrinks) and takes it back
// after the instance has traced its mesh's own BLAS for a while; the GPU switches rings at a build and releases the last
// one once no frame in flight traces it. Until its first ring an instance stays on its mesh's BLAS.
class LodGpu {
  public:
    struct Instance {
        Lod* lod = nullptr;                // paged; null: not this system's
        D3D12_GPU_VIRTUAL_ADDRESS baseBlas = 0;
        uint32_t tlasIndex = 0, hitGroup = 0;
        // A cell of a celled hierarchy (its index among the hierarchy's cells, cells): the hierarchy clusters of that cell
        // (by cells, per cluster), none of the fixed ones, and its BLAS for an empty share of the cut (baseBlas). The cells
        // of a hierarchy share one view and one pass over its clusters, which hands each to its cell.
        uint32_t cell = ~0u;
        const std::vector<uint32_t>* cells = nullptr;
    };
    struct Feedback {
        std::vector<LodRequest> requests;
        uint32_t builds = 0, tracedClusters = 0, tlasUpdates = 0;
        uint64_t buildRefs = 0;
        float gpuMs = 0.0f; // the frame's LOD work: cuts, BLAS builds, TLAS updates
        bool valid = false;
    };

    void init(DeviceContext* ctx, ID3D12DeviceRaytracing2* rtas, uint32_t frameSlots);
    bool active() const { return m_instances != 0; }
    // A new instance set; its static tables are uploaded on cl. Every instance starts on its mesh's own BLAS.
    void reset(const std::vector<Instance>& instances, ID3D12GraphicsCommandList4* cl, uint64_t frame);
    // Before ClusterGeometry::begin: the feedback of finished frames (requests; lastUsed and want into the hierarchies).
    void collect(Feedback& out);
    // After the CPU's TLAS build, in the frame's compute list; slot: the cluster frame slot. views: per instance of
    // reset, the camera in mesh space and its near distance; changes: residency since the last frame, in order.
    void record(ID3D12GraphicsCommandList4* cl, uint32_t slot, planet::TlasBuilder& tlas, const std::vector<float>& views,
                const std::vector<ResidencyChange>& changes, uint64_t frame, float pixelsPerUnit, float threshold);
    void on_submitted(uint64_t computeFence);
    uint64_t ring_bytes() const { return m_ringBytes; }
    uint64_t ring_used() const { return m_ringAlloc.used() * RING_UNIT; }
    uint32_t ring_failures() const { return m_ringFailures; }
    // The latest feedback's counts (no requests), and the clusters the instances trace at their meshes' own BLAS.
    const Feedback& last() const { return m_last; }
    uint64_t base_clusters() const { return m_baseClusters; }

  private:
    struct MeshEntry {
        Lod* lod;
        uint32_t groupBase, clusterBase, clasBase;
    };
    static constexpr uint64_t RING_UNIT = 256; // ring pool allocation unit, the BLAS alignment
    struct Ring {
        uint64_t offset = 0, units = 0;
        uint32_t cls = 0;
    };
    // The CPU's view of an instance's ring: the one it traces, the one handed over and not yet taken, how long the cut
    // has been much smaller than the ring or at the mesh's own BLAS, and whether the ring is asked back.
    struct InstanceRing {
        uint32_t id = ~0u, cls = 0, pending = ~0u, shrinking = 0, coarsest = 0;
        bool drop = false;
    };
    struct StateUpdate {
        uint32_t index, offset, value;
    };
    uint32_t alloc_ring(uint32_t cls);
    void manage_rings(const uint8_t* state, const uint32_t* counters);
    struct Slot {
        ComPtr<ID3D12Resource> upload, readback;
        uint64_t uploadBytes = 0;
        uint8_t* uploadMapped = nullptr;
        const uint8_t* readbackMapped = nullptr;
        uint64_t fence = 0;
        bool pending = false, recorded = false;
    };
    ComPtr<ID3D12Resource> buffer(uint64_t bytes, bool uav, const wchar_t* name);
    template <class T> ComPtr<ID3D12Resource> upload_static(const std::vector<T>& data, ID3D12GraphicsCommandList4* cl,
                                                            const wchar_t* name, bool uav = false);

    DeviceContext* m_ctx = nullptr;
    ComPtr<ID3D12DeviceRaytracing2> m_rtas;
    ComPtr<ID3D12RootSignature> m_root;
    ComPtr<ID3D12PipelineState> m_apply, m_clear, m_groups, m_clusters, m_plan, m_mark, m_finalize;

    uint32_t m_instances = 0, m_groupCount = 0, m_groupWork = 0, m_clusterWork = 0, m_classCount = 0;
    std::vector<MeshEntry> m_meshes;
    std::vector<uint32_t> m_instanceMesh;            // mesh entry per instance
    std::unordered_map<const Lod*, uint32_t> m_meshOf;
    std::vector<uint32_t> m_tlasToInstance;
    ComPtr<ID3D12Resource> m_meshBuf, m_instanceBuf, m_groupBuf, m_clusterBuf, m_depBuf, m_groupWorkBuf,
        m_clusterWorkBuf, m_classInfoBuf, m_targetBuf;
    ComPtr<ID3D12Resource> m_state, m_finePrime, m_cutIdx, m_counters, m_work, m_feedback, m_lastUsed, m_want,
        m_requested, m_residency, m_clasAddr;
    ComPtr<ID3D12Resource> m_ring, m_scratch;
    ComPtr<ID3D12QueryHeap> m_timeHeap; // per frame slot, the LOD work's start and end
    uint64_t m_timeFrequency = 1;
    struct Class {
        uint32_t maxPerArg = 0, capacity = 0, budget = 0, argBase = 0;
        uint64_t entryBytes = 0, scratchOff = 0;
    };
    std::vector<Class> m_classes;
    std::vector<uint32_t> m_bound; // per instance: the cluster count its cuts stay within
    std::vector<uint32_t> m_fixed; // per instance: the fixed clusters leading its cuts
    uint64_t m_ringBytes = 0;
    RangeAllocator m_ringAlloc;
    std::vector<Ring> m_rings;
    std::vector<uint32_t> m_freeRings;
    std::vector<InstanceRing> m_instanceRings;
    std::vector<std::pair<uint32_t, uint64_t>> m_ringFrees; // rings the GPU released, and the frame of their last use
    std::vector<StateUpdate> m_ringUpdates;
    uint64_t m_frame = 0;
    uint32_t m_ringFailures = 0;
    uint32_t m_classArgs = 0, m_classDests = 0, m_classCounts = 0, m_classBudgets = 0, m_tlasArgs = 0;
    uint32_t m_maxRequests = 4096;
    uint64_t m_readbackBytes = 0;
    Feedback m_last;
    uint64_t m_baseClusters = 0;
    std::vector<Slot> m_slots;
    uint32_t m_slot = 0;
    // The last reset's uploads and the set before it, until the frames that use them have run.
    std::vector<ComPtr<ID3D12Resource>> m_keep;
    uint64_t m_keepFence = 0;
    bool m_keepPending = false;
    bool m_rewriteAll = false; // the TLAS may still hold the last set's BLAS
};
#endif

} // namespace clusters
