#pragma once

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>
#include <vector>
#include "ClusterLod.h"

namespace clusters {

// First fit over [0, capacity) in units, coalescing on free. Not thread-safe.
class RangeAllocator {
  public:
    void init(uint64_t capacity);
    bool allocate(uint64_t count, uint64_t& offset);
    void free(uint64_t offset, uint64_t count);
    uint64_t used() const { return m_used; }
    uint64_t capacity() const { return m_capacity; }
    // More room at the end.
    void grow(uint64_t capacity);

  private:
    std::vector<std::pair<uint64_t, uint64_t>> m_free; // offset, count; sorted
    uint64_t m_capacity = 0, m_used = 0;
};

// Cluster pages streamed from disk: with cluster hierarchies, unless RDN_PAGING=0 (then every level stays resident).
bool paging_enabled();
// Where baked page files go: RDN_GEOMETRY_CACHE, else %LOCALAPPDATA%\RoyalTracer\GeometryCache.
std::string cache_dir();
struct PoolSizes {
    uint32_t vertices = 0, tris = 0; // the page pool at the front of the global vertex, index and material-ID arrays
    uint64_t clasBytes = 0, ommBytes = 0;
    bool fixed = false;              // set by the environment: the pool does not grow to the scene
};
// RDN_PAGE_POOL_VERTS, RDN_PAGE_POOL_TRIS, RDN_PAGE_POOL_CLAS_MB and RDN_PAGE_POOL_OMM_MB override the defaults.
PoolSizes pool_sizes();

// A hierarchy's pages as stored: per group its clusters' vertices (BTriVertex) and page-local 16-bit indices; an alpha
// page (alpha-tested or transmissive material) then an index per triangle into its mesh's micromaps.
struct PageEntry {
    uint64_t offset = 0;
    uint32_t bytes = 0, firstCluster = 0, clusterCount = 0, vertexCount = 0, triCount = 0;
    bool alpha = false;
    uint16_t file = 0; // into PagedSource::files, for a page index
};
// Opacity micromaps: per triangle an OMM index (or a special index), and the micromaps they refer to.
struct PageOmm {
    std::vector<int32_t> indices;
    std::vector<D3D12_RAYTRACING_OPACITY_MICROMAP_DESC> descs;
    std::vector<D3D12_RAYTRACING_OPACITY_MICROMAP_HISTOGRAM_ENTRY> histogram;
    std::vector<uint8_t> raw;
};
struct PagedSource {
    std::string path;
    uint32_t materialBase = 0;        // the model's first material, which the file's material IDs count from
    std::vector<PageEntry> pages;     // per group
    std::vector<Span> spans;          // per hierarchy cluster; firstTri within its page
    std::vector<uint32_t> materials;  // per hierarchy cluster, relative to the model's first material
    PageOmm omm;                      // the alpha pages' micromaps (no indices; those are in the pages)
    // Coded pages: a uint16 material per triangle (relative to materialBase) after the indices, in place of the
    // clusters' materials; an alpha page's micromap indices then refer to the array at ommArray, the loader's (0: none,
    // so micromapped triangles fall back to any-hit).
    bool coded = false;
    D3D12_GPU_VIRTUAL_ADDRESS ommArray = 0;
    std::vector<uint8_t> extra;       // what the bake kept with the file for its loader
    // A page index (write_page_index): the files the pages are in, and per hierarchy cluster the cell (spatial owner) it
    // belongs to. Roots do not lead its clusters.
    std::vector<std::string> files;
    std::vector<uint32_t> cells;
};

// The bake's input mesh: Vertex-layout vertices (float3 position, float4 normal, float2 uv).
struct BakeMesh {
    const uint8_t* vertices = nullptr;
    uint32_t stride = 0, normalOffset = 0, uvOffset = 0;
    uint32_t materialBase = 0; // subtracted from LodBuild::regions (material IDs) on disk
    // A region's opacity: false for an opaque one; else out gets micromaps of at most maxLevel subdivisions over the
    // triangles (indices into uvs, two floats per vertex), shared by its pages. Called for its full-detail triangles and
    // for its coarser ones apart.
    std::function<bool(uint32_t region, int maxLevel, const float* uvs, uint32_t vertexCount, const uint32_t* indices,
                       uint32_t triCount, PageOmm& out)> opacity;
    // With LodBuild::codes, pages are coded (see PagedSource): per code its material (relative; the code itself when
    // unset). Then with opacity, a code is a material and micromaps are baked per material (opacity is asked with
    // materialBase + code), else per code whether it needs any-hit (which makes its page alpha) and its micromap index
    // there (or a special index).
    std::function<uint16_t(uint32_t code)> codeMaterial;
    std::function<bool(uint32_t code)> codeAlpha;
    std::function<int32_t(uint32_t code)> codeOmm;
    std::vector<uint8_t> extra; // kept with the file, for its loader
};
// Writes the hierarchy's page file; the roots' pages first, and their clusters first by offset.
bool write_pages(const std::string& path, const LodBuild& build, const BakeMesh& mesh);

// A coded page's bytes (see PagedSource) from its clusters' triangles (vertex indices into mesh's vertices) and codes; layout
// gets its counts, whether it is alpha and per cluster its span in the page.
struct PageCluster {
    const uint32_t* indices;
    const uint32_t* codes;
    uint32_t triCount;
};
struct PageLayout {
    uint32_t vertexCount = 0, triCount = 0;
    bool alpha = false;
    std::vector<Span> spans;
};
bool serialize_page(const PageCluster* clusters, uint32_t count, const BakeMesh& mesh, std::vector<uint8_t>& out,
                    PageLayout& layout);
// A page index over pages in several files, for a hierarchy built in stages (build_lod_stage): per group its bounds and
// page; per cluster its group, refined group, span in its page and cell.
struct IndexPage {
    uint64_t offset = 0;
    uint32_t file = 0, bytes = 0, vertexCount = 0, triCount = 0;
    bool alpha = false;
};
bool write_page_index(const std::string& path, const std::vector<LodGroup>& groups, const std::vector<IndexPage>& pages,
                      const std::vector<LodCluster>& clusters, const std::vector<Span>& spans,
                      const std::vector<std::string>& files, const std::vector<uint32_t>& cells,
                      const std::vector<uint8_t>& extra);
// Reads a page file's tables into lod (groups, clusters, deps, base; marked paged) and source.
bool read_pages(const std::string& path, Lod& lod, PagedSource& source);

#if RDN_CLUSTER_API
// Keeps the pages the instances' cuts want resident, within the pools: requested pages are read on I/O threads, copied
// into the page pool of the global geometry arrays and their CLAS built into an arena; once the build's sizes are back,
// the CLAS move into the CLAS pool, tightly, and the page becomes resident. Least recently used pages that no traced cut
// and no resident page depends on make room.
class Pager {
  public:
    // The page pool's home: ranges of the global vertex, index (3 per triangle) and material-ID arrays at the same
    // triangle positions, the allocators over them, and the arrays.
    struct Geometry {
        ID3D12Resource* vertices = nullptr; // BTriVertex
        ID3D12Resource* indices = nullptr;  // absolute vertex indices
        D3D12_RESOURCE_STATES readState = D3D12_RESOURCE_STATE_GENERIC_READ;
        RangeAllocator* vertexAlloc = nullptr;
        RangeAllocator* triAlloc = nullptr;
        std::vector<uint32_t>* materialIds = nullptr;        // CPU copy
        ComPtr<ID3D12Resource>* materialIdBuffer = nullptr; // upload heap, mirrors the CPU copy
    };
    struct Config {
        uint64_t clasPoolBytes = 512ull << 20;
        uint64_t ommPoolBytes = 256ull << 20;
        uint64_t arenaBytes = 32ull << 20;   // per frame slot: CLAS builds, and a quarter as much scratch
        uint64_t stagingBytes = 16ull << 20; // per frame slot: page uploads
        uint32_t installsPerFrame = 96;
        uint32_t retiresPerFrame = 32;
        uint32_t readsInFlight = 128;
        uint32_t ioThreads = 2;
    };
    struct Stats {
        uint32_t meshes = 0, pages = 0, resident = 0, pinned = 0, reading = 0, building = 0;
        uint32_t installed = 0, evicted = 0, dropped = 0; // this frame
        uint64_t installedTotal = 0, evictedTotal = 0, droppedTotal = 0, retiredTotal = 0;
        uint64_t residentClusters = 0, residentTris = 0, readBytes = 0;
        uint64_t clasUsed = 0, clasCapacity = 0, trisUsed = 0, trisCapacity = 0, vertsUsed = 0, vertsCapacity = 0;
        // The cuts' need as a share of the tightest pool: pages they used lately, bring in or ask for (the rest of what
        // the pools hold is evictable).
        float pressure = 0.0f;
    };

    ~Pager();
    void init(DeviceContext* ctx, ClusterGeometry* geometry, const Geometry& g, const Config& cfg);
    bool active() const { return !m_meshes.empty(); }
    // A paged mesh: its hierarchy (from read_pages) and its own range of the pool at triBase (its flat triangle IDs
    // count from there); omm: its fixed clusters use micromaps (as its alpha pages do). Allocates the hierarchy's cluster
    // IDs.
    void add_mesh(Lod* lod, PagedSource&& source, uint32_t triBase, bool omm);
    // The mesh's hierarchy moved (the scene's mesh array grew).
    void rebind(uint32_t mesh, Lod* lod) { m_meshes[mesh].lod = lod; }
    // Before load_roots: every page of the mesh loads with the roots and stays (mass-instanced meshes, whose instances
    // share uniform cuts of the whole hierarchy).
    void pin(uint32_t mesh) {
        for (Page& p : m_meshes[mesh].pages) p.pinned = true;
    }
    // Loads every root page synchronously (cl, then flush) and calls blas for each mesh with a cluster BLAS over its
    // fixed clusters and roots: its coarsest cut. The fixed clusters' CLAS addresses must be in lod->clas.
    void load_roots(ID3D12GraphicsCommandList4* cl, const std::function<void()>& flush,
                    const std::function<void(uint32_t mesh, ComPtr<ID3D12Resource> blas)>& blas);

    // A frame, inside the orchestrator's cluster window: before_begin before ClusterGeometry::begin (collects finished
    // builds' sizes while their tickets hold), moves after begin (finished pages become resident, before the cuts),
    // installs after the cuts (reads what they requested, installs what was read), on_submitted with the frame's fence.
    void before_begin();
    void moves(uint64_t frame);
    void installs(std::vector<LodRequest>& requests, ID3D12GraphicsCommandList* copy, ID3D12GraphicsCommandList4* compute);
    void on_submitted(uint64_t computeFence);
    const Stats& stats() const { return m_stats; }
    // Cuts made on the GPU: lastUsed and want arrive with its feedback, frames late, so a resident page leaves the cuts
    // before its memory is freed, and the residency changes are logged for the GPU's tables.
    void set_external_feedback(bool on) { m_external = on; }
    std::vector<ResidencyChange> take_changes() {
        std::vector<ResidencyChange> out;
        out.swap(m_changes);
        return out;
    }

  private:
    // Retiring: out of the cuts, its memory held until no frame in flight traces it.
    enum class State : uint8_t { Absent, Reading, Read, Building, Built, Resident, Retiring };
    struct Page {
        State state = State::Absent;
        bool pinned = false;
        uint32_t dependents = 0; // resident pages it is a dep of
        uint32_t slot = 0;
        uint64_t fence = 0;
        uint64_t retiredAt = 0; // the frame it left the cuts, while Retiring
        ClusterGeometry::Ticket ticket;
        uint64_t vertex = 0, tri = 0;       // pool ranges while installed
        uint64_t clas = 0, clasUnits = 0;   // CLAS pool range (128-byte units) while resident
        std::vector<uint8_t> data;          // read, until built
        std::vector<uint64_t> sizes;        // built CLAS sizes, until moved
    };
    struct Mesh {
        Lod* lod = nullptr;
        PagedSource src;
        std::vector<HANDLE> files; // per PagedSource::files, or the one page file
        uint32_t triBase = 0, materialBase = 0;
        bool omm = false; // its fixed clusters use micromaps
        std::vector<uint8_t> levels; // per hierarchy cluster
        uint64_t ommArray = 0, ommUnits = 0; // its alpha pages' micromap array in the OMM pool (128-byte units)
        std::vector<Page> pages;
    };
    struct Read {
        uint32_t mesh, group;
        HANDLE file;
        uint64_t offset;
        uint32_t bytes;
        std::vector<uint8_t> data;
        bool ok = false;
    };
    struct Arena {
        ComPtr<ID3D12Resource> buffer, scratch, staging;
        uint8_t* stagingMapped = nullptr;
        uint64_t used = 0, scratchUsed = 0, stagingUsed = 0;
    };

    bool alloc_geometry(Mesh& m, uint32_t group, bool evict);
    void free_geometry(Mesh& m, uint32_t group);
    // Writes the page's vertices and absolute indices into staging and copies them into the pool; its material IDs and
    // cluster records directly.
    void upload(Mesh& m, uint32_t group, const uint8_t* data, ID3D12GraphicsCommandList* copy, ID3D12Resource* staging,
                uint8_t* stagingMapped, uint64_t stagingOff, uint32_t* gpuMaterials);
    clusters::Mesh build_desc(const Mesh& m, uint32_t group) const;
    // An alpha page's OMM indices, as stored after its geometry (and its materials, when coded).
    static const int32_t* omm_indices(const PageEntry& e, const uint8_t* data, bool coded);
    // The OMM array build's result and scratch sizes (128- and 256-byte aligned).
    void omm_prebuild(const PageOmm& o, uint64_t& result, uint64_t& scratch) const;
    // Builds the micromaps from raw and descs already in GPU memory (rawVa 128-byte aligned) into dest.
    void omm_build(ID3D12GraphicsCommandList4* cl, const PageOmm& o, D3D12_GPU_VIRTUAL_ADDRESS rawVa,
                   D3D12_GPU_VIRTUAL_ADDRESS descsVa, D3D12_GPU_VIRTUAL_ADDRESS dest, D3D12_GPU_VIRTUAL_ADDRESS scratch);
    // Places a read page and queues its CLAS build; false when the pools or this frame's room are short.
    bool install(uint32_t mesh, uint32_t group, ID3D12GraphicsCommandList* copy, ID3D12GraphicsCommandList4* compute,
                 Arena& a, uint32_t* gpuMaterials);
    // Moves a built page's CLAS into the CLAS pool, tightly by their built sizes, and makes it resident; false when
    // there is no room.
    bool place_clas(uint32_t mesh, uint32_t group, D3D12_GPU_VIRTUAL_ADDRESS scratch, bool evict);
    bool move(uint32_t mesh, uint32_t group, Arena& a);
    void make_resident(uint32_t mesh, uint32_t group);
    // Takes a resident page out of the cuts (its memory stays until evict).
    void retire(uint32_t mesh, uint32_t group);
    void evict(uint32_t mesh, uint32_t group);
    // The pools are full while pages wanted at pressure wait: retires the leaves wanted least.
    void rebalance(float pressure);
    // Evicts the least recently used pages nothing needs any more until test() passes; false if it never does.
    bool make_room(const std::function<bool()>& test);
    void io_loop();

    DeviceContext* m_ctx = nullptr;
    ClusterGeometry* m_geometry = nullptr;
    Geometry m_geo;
    Config m_cfg;
    std::vector<Mesh> m_meshes;
    ComPtr<ID3D12Resource> m_clasPool;
    RangeAllocator m_clasAlloc;
    ComPtr<ID3D12Resource> m_ommPool;      // the meshes' micromap arrays, resident from load
    RangeAllocator m_ommAlloc;
    ComPtr<ID3D12Resource> m_ommIndexPool; // OMM index per pool triangle, CPU-written
    int32_t* m_ommIndexMapped = nullptr;
    std::vector<Arena> m_arenas; // per frame slot
    uint64_t m_frame = 0;
    std::vector<std::pair<uint32_t, uint32_t>> m_building; // built this frame, for on_submitted
    std::vector<std::pair<uint32_t, uint32_t>> m_inFlight; // Building and Built pages
    std::vector<std::pair<uint32_t, uint32_t>> m_read;     // Read pages, oldest first
    uint32_t m_reading = 0;
    std::vector<std::tuple<uint64_t, uint32_t, uint32_t>> m_lru; // eviction candidates this frame: last used, mesh, group
    size_t m_lruNext = 0;
    bool m_lruBuilt = false;
    bool m_starved = false; // this frame, a page found no room in the pools
    bool m_external = false;
    uint32_t m_retiresLeft = 0; // this frame, for make_room with external feedback
    float m_reviveBar = 0.0f;   // while starved: retiring pages wanted less than half this stay out
    std::vector<ResidencyChange> m_changes;

    std::mutex m_ioMutex;
    std::condition_variable m_ioWake;
    std::deque<Read> m_ioQueue, m_ioDone;
    std::vector<std::thread> m_ioThreads;
    bool m_ioStop = false;
    Stats m_stats;
};
#endif

} // namespace clusters
