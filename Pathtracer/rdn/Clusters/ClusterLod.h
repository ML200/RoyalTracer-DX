#pragma once

#include <cstdint>
#include <functional>
#include <vector>
#include "ClusterGeometry.h"

namespace clusters {

constexpr uint32_t LOD_NONE = ~0u, NO_REGION = ~0u;
// Every region gets a hierarchy, down to a single cluster.
constexpr uint32_t MIN_LOD_TRIS = 1;

// Hierarchies for scene meshes: with clustered geometry, unless RDN_LOD=0.
bool lod_enabled();

// A group's simplified bounds: replacing the group's clusters by their simplification moves the surface by up to error,
// within the sphere. FLT_MAX for a group never replaced (a root, or where simplification got stuck).
struct LodGroup {
    float center[3], radius, error;
};
struct LodCluster {
    uint32_t group;   // the group the cluster is part of
    uint32_t refined; // the group whose simplification made it; LOD_NONE for original triangles
};

// A mesh's cluster LOD hierarchy (Nanite style: clusters grouped, simplified with the group borders locked and clustered
// again, up to roots), every level resident. By the mesh's cluster ID offset: [0, fixed) are outside the hierarchy and in
// every cut, then the hierarchy's clusters, its original ones first, so [0, base) is the full-detail cut.
struct Lod {
    uint32_t fixed = 0, base = 0;
    // The mesh triangles only coarser levels use: no light sources, for instance.
    uint32_t coarseTriBase = 0, coarseTriCount = 0;
    std::vector<LodCluster> clusters; // from offset fixed
    std::vector<LodGroup> groups;
    std::vector<uint64_t> clas;       // CLAS address per offset while built (0 otherwise)

    // Paged: the hierarchy lives on disk, a page per group (its clusters), and only some groups are resident. A group
    // may be resident only while the groups holding its simplification (its deps) are, so the roots, which have none,
    // stay; base then ends the roots' clusters, the coarsest cut. The hierarchy's cluster IDs start at idBase.
    bool paged = false;
    uint32_t idBase = 0;
    uint32_t pager = LOD_NONE;      // the pager's index for the mesh
    std::vector<uint8_t> resident;  // per group
    std::vector<uint32_t> depFirst; // per group, and one past the last
    std::vector<uint32_t> deps;
    std::vector<uint64_t> lastUsed; // per group: the last frame a traced cut held one of its clusters
    std::vector<float> want;        // per group, this frame: how far over budget its simplification is (0: fine)

    bool empty() const { return clusters.empty(); }
    uint32_t count() const { return fixed + (uint32_t)clusters.size(); }
    bool is_resident(uint32_t group) const { return !paged || resident[group]; }
};

// Each hierarchy cluster's level: 0 for original triangles, one above the deepest cluster of the group simplified
// into it otherwise.
void lod_levels(const Lod& lod, std::vector<uint8_t>& levels);

// A page entering a hierarchy's cuts (its clusters' CLAS addresses with it) or leaving them, for whoever mirrors
// residency elsewhere.
struct ResidencyChange {
    Lod* lod;
    uint32_t group, firstCluster, clusterCount;
    bool resident;
};

// A page a cut would refine into: not resident, with its deps resident. priority: how far over the error budget the
// coarser level it replaces is.
struct LodRequest {
    uint32_t mesh, group;
    float priority;
};

// A mesh for build_lod. Triangles with a region go into the hierarchy and simplify only with their region's (a region
// per material keeps materials apart); those with NO_REGION stay outside.
struct LodSource {
    const uint8_t* vertices = nullptr; // float3 position, float3 normal and float2 uv at the offsets, per stride
    uint32_t stride = 0, normalOffset = 0, uvOffset = 0, vertexCount = 0;
    const uint32_t* indices = nullptr;
    const uint32_t* regions = nullptr; // per triangle
    uint32_t triCount = 0;
    // Foliage regions (alpha-tested cards): their simplification dilates open borders against area loss and keeps the
    // folds of double-sided sheets.
    std::function<bool(uint32_t region)> foliage;
};
struct LodBuild {
    std::vector<uint32_t> indices; // the hierarchy's triangles, cluster after cluster: cluster k is spans[k]
    std::vector<float> positions;  // per entry of indices: its vertex's position as that cluster has it (dilation moves them)
    std::vector<uint32_t> regions; // per triangle of indices
    std::vector<Span> spans;       // firstTri counts from indices' first triangle
    std::vector<LodCluster> clusters;
    std::vector<LodGroup> groups;
    uint32_t originalClusters = 0; // the first ones, the full-detail cut
};
// Vertices a region shares with another one or with triangles outside the hierarchy stay where they are, so no cut opens
// a crack against them. Parallel over the regions.
void build_lod(const LodSource& src, LodBuild& out);

// The view a cut is chosen for, with the camera in mesh space. A group is fine enough when its error projects to at most
// threshold pixels.
struct LodView {
    float camera[3] = {};
    float pixelsPerUnit = 0; // 0.5 * screen height / tan(fov_y / 2): the projection of error 1 at distance 1
    float threshold = 1;     // pixels
    float znear = 0.01f;     // mesh units
};
// The hierarchy clusters (offsets from Lod::fixed) of the view's cut: those whose own triangles are fine enough while
// their group's simplification is not. Crack-free, as group errors and bounds only grow towards the roots; a group not
// resident counts as fine enough, which residency closed towards the roots keeps crack-free. requests receives the
// pages the view wants next.
void select_cut(const Lod& lod, const LodView& view, std::vector<uint32_t>& cut, std::vector<LodRequest>* requests = nullptr);

#if RDN_CLUSTER_API
// Per-instance cuts and the cluster BLAS over them. Every frame each instance gets the cut for its view; one whose cut
// changed gets a cluster BLAS over the new one, in a ring per instance that frames in flight still trace. Until its first,
// an instance traces its mesh's full-detail BLAS.
class LodInstances {
  public:
    struct Instance {
        Lod* lod = nullptr; // null: an instance without a hierarchy
        bool omm = false;
    };
    struct Budget {
        uint32_t builds = 128;          // cluster BLAS per frame
        uint32_t clusters = 1u << 18;   // their CLAS references per frame
    };
    struct Stats {
        uint32_t instances = 0, builds = 0, deferred = 0;
        uint64_t cutClusters = 0, fullClusters = 0, ringBytes = 0;
    };

    void init(ClusterGeometry* geometry, ID3D12Device* device, const Budget& budget);
    // A new instance set (scene load, instances added or removed): every instance restarts on its full-detail BLAS.
    void reset(const std::vector<Instance>& instances, uint64_t frame);
    bool active() const { return m_active; }
    // In the frame's cluster window (between ClusterGeometry::begin and record): picks each instance's cut for views[i],
    // and queues the cluster BLAS of the cuts that changed, as many as the budget, the arguments and the ring allow; the
    // rest follow in later frames. Paged hierarchies get their groups' lastUsed and the pages wanted in requests.
    void update(const LodView* views, uint64_t frame, std::vector<LodRequest>* requests = nullptr);
    // 0: the mesh's own BLAS.
    D3D12_GPU_VIRTUAL_ADDRESS blas(uint32_t instance) const;
    const Stats& stats() const { return m_stats; }

  private:
    static constexpr uint32_t MAX_RING = 8;
    struct State {
        Lod* lod = nullptr;
        bool omm = false;
        uint64_t ringOff = 0, blasSize = 0, scratch = 0;
        uint32_t current = LOD_NONE;      // ring entry traced; LOD_NONE: the mesh's BLAS
        uint64_t retired[MAX_RING] = {};  // frame from which an entry is no longer traced
        std::vector<uint32_t> cut;        // of the current BLAS
    };
    ClusterGeometry* m_geometry = nullptr;
    ID3D12Device* m_device = nullptr;
    Budget m_budget;
    bool m_active = false;
    uint32_t m_ring = 0, m_next = 0;
    std::vector<State> m_states;
    ComPtr<ID3D12Resource> m_pool;                 // every instance's ring
    std::vector<ComPtr<ID3D12Resource>> m_scratch; // per frame slot
    uint64_t m_scratchCapacity = 0;
    struct Retired {
        ComPtr<ID3D12Resource> buffer;
        uint64_t frame;
    };
    std::vector<Retired> m_retired; // a replaced pool, until no frame in flight traces it
    std::vector<uint32_t> m_cut;
    std::vector<uint64_t> m_addresses;
    Stats m_stats;
};
#endif

} // namespace clusters
