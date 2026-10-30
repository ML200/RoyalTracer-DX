#pragma once

#include <cstdint>
#include <mutex>
#include <vector>
#include "../Common.h"
#include "../../shaders/DlssGuideLayout.h"

struct DeviceContext;

namespace clusters {

constexpr uint32_t MAX_TRIS = 256, MAX_VERTS = 256;

inline uint32_t cluster_info(uint32_t kind, uint32_t level) {
    return (kind << CLUSTER_INFO_KIND_SHIFT) | (level & CLUSTER_INFO_LEVEL_MASK);
}

// A run of a mesh's triangle order: its cluster-local triangle i is the mesh's triangle firstTri + i, which is what
// ClusterRecord.triBase gives the shaders.
struct Span {
    uint32_t firstTri;
    uint16_t triCount, uniqueVerts;
};

// Orders units (runs of unitTris consecutive triangles; 2 keeps a quad together) along a Morton curve over their
// centroids (xyz per unit) and cuts that order at octree cells of at most maxTris triangles. order receives the units in
// their new order, cells each cell's first position in it.
void morton_cells(const float* centroids, uint32_t unitCount, uint32_t unitTris, uint32_t maxTris,
                  std::vector<uint32_t>& order, std::vector<uint32_t>& cells);

// Splits triangles [cuts[0], end) into spans within the CLAS limits, starting a new span at every cut. indices are the
// mesh's, below vertexCount.
void split(const uint32_t* indices, uint32_t vertexCount, const std::vector<uint32_t>& cuts, uint32_t end,
           std::vector<Span>& out);

#if RDN_CLUSTER_API
// The limits of one build over spans, with geometry indices up to maxGeometryIndex.
D3D12_RTAS_CLUSTER_LIMITS limits(const Span* spans, uint32_t count, uint32_t maxGeometryIndex, bool omm);

// A mesh's clusters, as one CLAS build batch sees them.
struct Mesh {
    D3D12_GPU_VIRTUAL_ADDRESS vertices = 0; // float3 positions at vertexStride
    uint32_t vertexStride = 0;
    D3D12_GPU_VIRTUAL_ADDRESS indices = 0;  // uint32, addressed from vertices; the mesh's triangle 0
    const Span* spans = nullptr;
    uint32_t clusterCount = 0, clusterBase = 0;
    // Triangles below opaqueTriCount are opaque (geometry 0); the rest alpha, as geometry 1 when there are opaque ones
    // unless alphaGeometry says which.
    uint32_t opaqueTriCount = 0;
    uint32_t alphaGeometry = ~0u;
    D3D12_GPU_VIRTUAL_ADDRESS ommArray = 0, ommIndices = 0; // uint32 per alpha triangle, from the first; 0: none
    D3D12_RTAS_CLUSTER_LIMITS limits{};
    bool omm() const { return ommIndices != 0; }
};
#endif

// The engine's clustered geometry: cluster IDs and their records (t60), and the CLAS / cluster BLAS operations clients
// queue. A client builds a mesh's CLAS into a region sized by the prebuild query, reads back what they really took, moves
// them into a tight region and builds its cluster BLAS over the moved ones (the spec's replacement for compaction).
class ClusterGeometry {
  public:
    struct Config {
        uint32_t clusterCapacity = 4u << 20;
        uint64_t argBytesPerSlot = 8u << 20;
        uint32_t sizeEntriesPerSlot = 1u << 16;
        uint64_t syncArgBytes = 64u << 20;
        uint32_t syncSizeEntries = 1u << 20;
    };
    struct Sizes {
        uint64_t clas = 0, clasScratch = 0, blas = 0;
    };
    // A queued operation's built sizes; valid until a later begin() on its slot.
    struct Ticket {
        uint32_t slot = ~0u, index = 0;
        uint64_t gen = 0;
        bool valid() const { return slot != ~0u; }
    };

    // Off without device support (or without the API in the headers); nothing else may be called then. slots serve the
    // frames in flight; one more, sync_slot(), serves builds that wait for their list (loading), so they never reuse
    // arguments a frame still in flight reads.
    void init(DeviceContext* ctx, uint32_t slots, const Config& cfg);
    bool enabled() const { return m_enabled; }
    uint32_t sync_slot() const { return (uint32_t)m_gen.size() - 1; }
    ID3D12Resource* records() const { return m_records.Get(); }
    uint64_t ids_used() const { return m_ids.used; }

    // Thread-safe.
    bool allocate_ids(uint32_t count, uint32_t& base);
    void free_ids(uint32_t base, uint32_t count);
    // Records with CLUSTER_KIND_STATIC; write_info then marks hierarchy clusters.
    void write_records(uint32_t base, const Span* spans, uint32_t count);
    void write_info(uint32_t base, const uint32_t* info, uint32_t count);

#if RDN_CLUSTER_API
    ID3D12DeviceRaytracing2* rtas_device() const { return m_device.Get(); }
    // Thread-safe. prebuild's blas is for a cluster BLAS over limits.MaxArgCount clusters.
    bool prebuild(const D3D12_RTAS_CLUSTER_LIMITS& limits, bool omm, Sizes& out) const;
    void blas_prebuild(uint32_t clusterCount, bool omm, uint64_t& size, uint64_t& scratch) const;
    uint64_t move_scratch(uint32_t clusterCount, uint64_t clasBytes, bool omm, bool explicitDest = false) const;
    uint64_t finish_scratch(uint32_t clusterCount, uint64_t clasBytes, bool omm) const;
#endif

    // A frame's (or a load batch's) list: begin once, then queue and record; every client of the frame queues between
    // begin and its last record, which may run more than once.
    uint32_t frame_slots() const { return (uint32_t)m_gen.size() - 1; }
    void begin(uint32_t slot, ID3D12GraphicsCommandList4* cl);
    uint32_t slot() const { return m_slot; }
    bool can_queue(uint32_t clusterCount, bool finish) const;
    // Room for queue_blas_from over clasCount CLAS, or for queue_move_to of clasCount CLAS.
    bool can_queue_blas(uint32_t clasCount) const;
#if RDN_CLUSTER_API
    // Builds the mesh's CLAS into result (implicit destinations, prebuild-sized); the ticket gives their built sizes.
    Ticket queue_build(const Mesh& mesh, D3D12_GPU_VIRTUAL_ADDRESS result, D3D12_GPU_VIRTUAL_ADDRESS scratch);
    // Moves built CLAS into clasDest, packed (clasBytes: their built sizes' sum).
    void queue_move(uint32_t clusterBase, uint32_t clusterCount, bool omm, D3D12_GPU_VIRTUAL_ADDRESS clasDest,
                    uint64_t clasBytes, D3D12_GPU_VIRTUAL_ADDRESS scratch);
    // Builds a cluster BLAS at blasDest over moved CLAS (after this list's moves), or with moved false over CLAS built
    // in place and never moved (after this list's builds); the ticket gives its built size.
    Ticket queue_blas(uint32_t clusterBase, uint32_t clusterCount, bool omm, D3D12_GPU_VIRTUAL_ADDRESS blasDest,
                      D3D12_GPU_VIRTUAL_ADDRESS scratch, bool moved = true);
    // queue_move, then queue_blas over the same clusters; scratch is finish_scratch().
    Ticket queue_finish(uint32_t clusterBase, uint32_t clusterCount, bool omm, D3D12_GPU_VIRTUAL_ADDRESS clasDest,
                        uint64_t clasBytes, D3D12_GPU_VIRTUAL_ADDRESS blasDest, D3D12_GPU_VIRTUAL_ADDRESS scratch);
    // Builds a cluster BLAS at blasDest over the CLAS at the count addresses in clas (built earlier, not moved since),
    // with the inputs of blas_prebuild(maxCount), which sized blasDest and scratch.
    Ticket queue_blas_from(const uint64_t* clas, uint32_t count, uint32_t maxCount, bool omm,
                           D3D12_GPU_VIRTUAL_ADDRESS blasDest, D3D12_GPU_VIRTUAL_ADDRESS scratch);
    // Moves built CLAS (by ID) to the destinations in dests, one per cluster, each at least its built size; scratch is
    // move_scratch(clusterCount, clasBytes, omm, true).
    void queue_move_to(uint32_t clusterBase, uint32_t clusterCount, bool omm, const uint64_t* dests, uint64_t clasBytes,
                       D3D12_GPU_VIRTUAL_ADDRESS scratch);
#endif
    // Copies the moved CLAS addresses of cluster IDs [base, base + count) into dest (in COPY_DEST) at offset.
    void copy_addresses(ID3D12GraphicsCommandList* cl, uint32_t base, uint32_t count, ID3D12Resource* dest, uint64_t offset);
    void record(ID3D12GraphicsCommandList4* cl);
    // Once the recording list executed; false when a later begin() on the slot overwrote them.
    bool sizes(const Ticket& t, uint32_t count, uint64_t* out) const;

  private:
    // First fit over ID ranges.
    struct IdRanges {
        std::vector<std::pair<uint32_t, uint32_t>> free; // first, count
        uint64_t used = 0;
    };

    bool m_enabled = false;
    Config m_cfg;
    uint32_t m_slot = 0;
    mutable std::mutex m_idMutex;
    IdRanges m_ids;
    ComPtr<ID3D12Resource> m_records; // ClusterRecord per cluster ID, CPU-written
    ClusterRecord* m_recordsMapped = nullptr;
    // CLAS address per cluster ID: where the build put it, and where the move put it.
    ComPtr<ID3D12Resource> m_built, m_final;
    D3D12_GPU_VIRTUAL_ADDRESS m_builtVa = 0, m_finalVa = 0;
    std::vector<ComPtr<ID3D12Resource>> m_args; // per slot
    std::vector<uint8_t*> m_argsMapped;
    std::vector<uint64_t> m_argCapacity;
    std::vector<uint32_t> m_sizeCapacity;
    std::vector<ComPtr<ID3D12Resource>> m_sizes, m_sizesReadback;
    std::vector<const uint64_t*> m_sizesMapped;
    std::vector<uint8_t> m_sizesCopied;
    std::vector<uint64_t> m_gen; // per slot, bumped by begin()
    uint64_t m_argsUsed = 0;
    uint32_t m_sizesUsed = 0;
#if RDN_CLUSTER_API
    template <class Inputs> struct Op {
        Inputs in;
        D3D12_RTAS_BATCHED_OPERATION_DATA data;
    };
    ComPtr<ID3D12DeviceRaytracing2> m_device;
    std::vector<Op<D3D12_RTAS_CLUSTER_TRIANGLES_INPUTS_DESC>> m_builds;
    std::vector<Op<D3D12_RTAS_CLUSTER_MOVES_DESC>> m_moves;
    std::vector<Op<D3D12_RTAS_CLAS_INPUTS_DESC>> m_blas, m_builtBlas;
#endif
};

} // namespace clusters
