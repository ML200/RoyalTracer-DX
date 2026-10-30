#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>
#include "ClusterLod.h"

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
class LodGpu {
  public:
    struct Instance {
        Lod* lod = nullptr;                // paged; null: not this system's
        D3D12_GPU_VIRTUAL_ADDRESS baseBlas = 0;
        uint32_t tlasIndex = 0, hitGroup = 0;
    };
    struct Feedback {
        std::vector<LodRequest> requests;
        uint32_t builds = 0, tracedClusters = 0, tlasUpdates = 0;
        uint64_t buildRefs = 0;
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
    // The latest feedback's counts (no requests), and the clusters the instances trace at their meshes' own BLAS.
    const Feedback& last() const { return m_last; }
    uint64_t base_clusters() const { return m_baseClusters; }

  private:
    struct MeshEntry {
        Lod* lod;
        uint32_t groupBase, clusterBase, clasBase;
    };
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
        m_clusterWorkBuf, m_ringAddrBuf, m_classInfoBuf;
    ComPtr<ID3D12Resource> m_state, m_finePrime, m_cutIdx, m_counters, m_work, m_feedback, m_lastUsed, m_want,
        m_requested, m_residency, m_clasAddr;
    ComPtr<ID3D12Resource> m_ring, m_scratch;
    struct Class {
        uint32_t maxPerArg = 0, capacity = 0, budget = 0, argBase = 0;
        uint64_t scratchOff = 0;
    };
    std::vector<Class> m_classes;
    uint64_t m_ringBytes = 0;
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
