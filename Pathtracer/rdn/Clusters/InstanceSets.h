#pragma once

#include <cstdint>
#include <functional>
#include <vector>
#include "ClusterLod.h"

namespace planet {
class TlasBuilder;
}

namespace clusters {

#if RDN_CLUSTER_API
// Mass instancing (InstanceSets_v8.hlsl): sets of one object at very many transforms, which the GPU writes into the
// partitioned TLAS. The object's meshes are resident in full (Pager::pin); a few uniform cuts of each, at doubling
// object-space error, are built once as cluster BLAS (buckets) that all its instances share. Every frame each instance
// takes the coarsest bucket within the error budget at its distance, a looser budget off screen, and never none, since
// rays leave the view. Instances are sorted into cells of at most CELL_INSTANCES (a TLAS partition each), whose bounds
// let a cell skip its instances while its bucket holds. The records carry the meshes in world space, so the hit's
// transform places them (IS_SET_INSTANCE).
//
// An object may be an assembly (foliage: a tree of branch and twig parts). Far off it is its whole, one mesh, down to
// a stand-in of three textured planes beyond its coarsest cut; an instance whose whole would need full detail expands
// into its parts, each a TLAS instance of its own from a pool of part slots, and sways in the wind near the camera.
class InstanceSets {
  public:
    static constexpr uint32_t CELL_INSTANCES = 1024; // the TLAS's instances per partition
    static constexpr uint32_t MAX_BUCKETS = 16;      // per mesh
    static constexpr uint32_t MAX_PARTS = 256;       // per assembly
    static constexpr uint32_t ASSEMBLY_BLOCKS = 4096; // instances of a set expanded at once

    // Placed at position (relative to the scene origin), rotated by the unit quaternion rotation (x, y, z, w) and
    // uniformly scaled, in the set's space.
    struct Instance {
        float position[3];
        float scale;
        float rotation[4];
    };
    // A mesh in a set's object: resident in full, placed in set space, whose origin an instance's position places.
    struct Mesh {
        const Lod* lod = nullptr;
        bool omm = false;         // its CLAS carry micromaps
        uint32_t record = 0;      // instance ID: its set record
        uint32_t hitGroup = 0;
        uint32_t flags = 0;       // D3D12_RAYTRACING_INSTANCE_FLAGS
        float base[12] = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0 }; // mesh to set space (3x4, rows)
    };
    struct Part {
        Mesh mesh;
        float pivot[3] = {};     // set space: where it hangs from
        float windWeight = 0.0f; // how far it sways
    };
    struct Desc {
        Mesh whole;
        std::vector<Part> parts; // an assembly's: near instances expand into them
        bool proxy = false;      // farther than the whole's coarsest cut: the stand-in
        Mesh stand;
        float center[3] = {}, radius = 0.0f; // the object's bounding sphere in set space
    };
    // The scene origin, and the camera relative to it; planes: points p in view have dot(xyz, p) + w >= 0 (unit
    // normals).
    struct View {
        float origin[3];
        float camera[3];
        float planes[5][4];
        float pixelsPerUnit, threshold, znear;
        float time; // seconds, for the wind
    };
    struct Wind {
        float distance = 60.0f;  // parts sway nearer than this, and hold still beyond
        float amplitude = 0.05f; // radians at full weight
        float frequency = 1.6f;  // radians per second
    };
    struct Stats {
        uint32_t sets = 0, cells = 0, buckets = 0, slots = 0;
        uint64_t instances = 0, blasBytes = 0, gpuBytes = 0;
        // The latest frame read back: TLAS instance writes (single and of whole expansions) and updates, cells whose
        // instances were visited, instances expanded into their parts.
        uint32_t writes = 0, blockWrites = 0, updates = 0, activeCells = 0, expanded = 0;
        uint64_t written = 0; // instances in the TLAS
        bool valid = false;
    };

    void init(DeviceContext* ctx, ClusterGeometry* geometry, ID3D12DeviceRaytracing2* rtas, uint32_t frameSlots);
    // Builds the set's buckets on cl, then flush; it joins the GPU tables at commit. Returns its index.
    uint32_t add(const Desc& d, std::vector<Instance> instances, ID3D12GraphicsCommandList4* cl,
                 const std::function<void()>& flush);
    // After the adds: the GPU tables, and the TLAS range (a new structure, which the next frame fills). origin: the
    // scene origin the instances' positions are relative to.
    void commit(planet::TlasBuilder& tlas, const float origin[3], ID3D12GraphicsCommandList4* cl,
                const std::function<void()>& flush);
    bool active() const { return m_cellCount != 0; }
    // Before the frame's cluster work: the stats of finished frames.
    void collect();
    // After the CPU's TLAS build, in the frame's compute list; slot: the cluster frame slot.
    void record(ID3D12GraphicsCommandList4* cl, uint32_t slot, planet::TlasBuilder& tlas, const View& view,
                float offscreenScale, const Wind& wind);
    void on_submitted(uint64_t computeFence);
    const Stats& stats() const { return m_stats; }

  private:
    struct Slot {
        ComPtr<ID3D12Resource> constants, readback;
        uint8_t* constantsMapped = nullptr;
        const uint32_t* readbackMapped = nullptr;
        uint64_t fence = 0;
        bool pending = false, recorded = false;
    };
    ComPtr<ID3D12Resource> buffer(uint64_t bytes, bool uav, const wchar_t* name);
    ComPtr<ID3D12Resource> upload(const void* data, uint64_t bytes, bool uav, ID3D12GraphicsCommandList4* cl,
                                  std::vector<ComPtr<ID3D12Resource>>& keep, const wchar_t* name);
    // The mesh's buckets, their BLAS built now (cl, then flush), appended to the table with their errors in set space;
    // standIn > 0 makes it a single bucket of full detail at that error. Returns the first; count, and the coarsest's
    // error and the finest's and coarsest's cluster counts, through the rest.
    uint32_t add_buckets(const Mesh& m, float standIn, ID3D12GraphicsCommandList4* cl,
                         const std::function<void()>& flush, uint32_t& count, float& lastError, uint32_t& fine,
                         uint32_t& coarse);

    DeviceContext* m_ctx = nullptr;
    ClusterGeometry* m_geometry = nullptr;
    ComPtr<ID3D12DeviceRaytracing2> m_rtas;
    ComPtr<ID3D12RootSignature> m_root;
    ComPtr<ID3D12PipelineState> m_reset, m_cells, m_collapse, m_alloc, m_instances, m_finalize;

    std::vector<uint8_t> m_instanceData; // packed instances, until commit
    std::vector<uint32_t> m_cellData;    // packed cells, until commit
    std::vector<uint32_t> m_bucketData, m_partData, m_setData;
    std::vector<ComPtr<ID3D12Resource>> m_blas;
    uint32_t m_setCount = 0, m_instanceCount = 0, m_cellCount = 0;
    uint32_t m_blockCount = 0, m_slotCount = 0, m_poolPartitions = 0; // assemblies' part slots
    ComPtr<ID3D12Resource> m_setBuf, m_bucketBuf, m_cellBuf, m_instanceBuf, m_partBuf, m_instanceState, m_cellState,
        m_work, m_pool;
    uint32_t m_maxWrites = 0, m_maxUpdates = 0, m_maxBlockWrites = 0;
    uint32_t m_writeArgs = 0, m_updateArgs = 0, m_translateArgs = 0, m_blockArgs = 0, m_freeLists = 0, m_partStates = 0;
    uint32_t m_tlasGeneration = ~0u;
    float m_origin[3] = {}, m_translated[3] = {}; // the sets' origin, and their partitions' translation
    std::vector<Slot> m_slots;
    uint32_t m_slot = 0;
    Stats m_stats;
};
#endif

} // namespace clusters
