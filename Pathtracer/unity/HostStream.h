#pragma once
#include "../rdn/Scene/HostStream.h"
#include "../rdn/minecraft/voxel_streamer.h"
#include "../rdn/LightTree.h"
#include "RoyalTracerApi.h"
#include <DirectXPackedVector.h>
#include <deque>
#include <functional>
#include <string>
#include <vector>

// Everything a Unity scene contributes, streamed into one running renderer: textures, materials, meshes and instances
// come and go at any time. Data is staged on the CPU and reaches the GPU in record_gpu_work on the planet compute
// queue (texture and buffer copies, BLAS builds), which the frame's graphics work waits for. GPU memory is freed a
// few frames after its release, once no queued frame can read it. Not thread-safe: the plugin serialises calls.
class HostStream final : public IHostStream {
  public:
    struct Config {
        uint32_t vertices = 16u << 20;
        uint32_t indices = 48u << 20;
        uint32_t triangles = 16u << 20;
        uint32_t instances = 65536;
        uint32_t materials = 16384; // slot 0 is the fallback for unknown material ids
        uint32_t lightRecords = 1u << 18; // emissive triangles
        uint32_t lightNodes = 1u << 19;
        uint32_t lightSlots = 4096;       // emissive instances in the light tree
    };

    explicit HostStream(const Config& config);
    ~HostStream() override;

    // IHostStream
    Reservation reservation() const override;
    void bind(const Binding& b) override;
    uint32_t instance_capacity() const override { return m_cfg.instances; }
    void record_gpu_work(ID3D12GraphicsCommandList* copyList, ID3D12GraphicsCommandList4* computeList) override;
    void append_instances(planet::TlasBuilder& tlas, InstanceProperties* props, const planet::DVec3& sceneOrigin,
                          uint32_t hitGroup, bool& forceRebuild, bool& forceRefit) override;
    void on_submitted(uint64_t, uint64_t) override {}

    // IDynamicLights: emissive triangles of streamed meshes, one light slot per visible emissive instance.
    bool lights_wanted() const override { return true; }
    uint32_t light_record_capacity() const override { return m_cfg.lightRecords; }
    uint32_t light_node_capacity() const override { return m_cfg.lightNodes; }
    void bind_lights(const DynamicLightBinding& b) override;
    bool lights_bound() const override { return m_lightsBound; }
    void set_light_slot_base(uint32_t base) override;
    uint32_t light_version() const override { return m_lightVersion; }
    void light_snapshot(const planet::DVec3& sceneOrigin, std::vector<lt::TLASExtraLeaf>& out, uint32_t& version,
                        uint32_t& slotCount) const override;
    void on_light_tlas_published(uint32_t version, bool hasLeaves) override;

    // Host API; ids are non-negative, -1 means failure (GetLastError-style reason in m_error).
    int32_t CreateTexture(const RT_TextureDesc& desc);
    void ReleaseTexture(int32_t id);
    int32_t CreateMaterial(const RT_MaterialDesc& desc);
    bool UpdateMaterial(int32_t id, const RT_MaterialDesc& desc);
    void ReleaseMaterial(int32_t id);
    int32_t CreateMesh(const RT_MeshDesc& desc);
    void ReleaseMesh(int32_t id); // instances keep it alive
    bool UpdateVertices(int32_t id, const float* positions, const float* normals); // deforming meshes
    int32_t CreateInstance(int32_t mesh, const float rowMajor3x4[12]);
    void ReleaseInstance(int32_t id);
    void SetTransform(int32_t instance, const float rowMajor3x4[12]);
    void SetVisible(int32_t instance, bool visible);
    void SetFlags(int32_t instance, uint32_t flags); // RT_InstanceFlags
    // Emitter sets: an octahedron per emitter in one emitter instance, geometry and light records rewritten whole on
    // every update. Empty sets are hidden.
    int32_t CreateEmitterSet(uint32_t capacity);
    bool UpdateEmitterSet(int32_t id, const RT_Emitter* emitters, uint32_t count);
    void ReleaseEmitterSet(int32_t id);

    const std::string& Error() const { return m_error; }
    std::string Stats() const;

  private:
    struct Texture {
        ComPtr<ID3D12Resource> resource, staging;
        std::vector<D3D12_PLACED_SUBRESOURCE_FOOTPRINT> footprints;
        bool live = false, uploaded = false;
    };
    // A mesh's emissive triangles: dense records in object space and their light BLAS. The mesh orders its
    // triangles [opaque emissive, opaque, alpha emissive, alpha], as the dense lookup (LightRecordOf) requires.
    struct MeshLights {
        std::vector<LightTriangle> records;
        std::vector<UINT> materials; // per record, so emission updates reach it
        lt::LightTreeBuilder::SingleBLAS blas;
        uint32_t litOpaque = 0, litAlpha = 0;
        uint64_t recOff = 0, nodeOff = 0;
        bool allocated = false, uploaded = false;
    };
    struct LightSlot {
        int32_t instance = -1;
        uint32_t includedAt = 0, excludedAt = 0xFFFFFFFFu;
        bool active = false;
    };

    struct Mesh {
        uint64_t vtxOff = 0, vtxCount = 0, idxOff = 0, triCount = 0, matOff = 0; // offsets: own regions
        uint32_t opaqueTris = 0;
        ComPtr<ID3D12Resource> staging, blas;
        uint64_t stagingIndexOffset = 0, scratchBytes = 0, updateScratchBytes = 0;
        uint32_t users = 0;
        bool live = false, released = false, built = false, deforming = false;
        bool rebuild = false; // emitter sets: vertices jump between updates, so updates rebuild instead of refit
        std::vector<DirectX::PackedVector::XMHALF2> uvs; // deforming: kept for vertex updates
        ComPtr<ID3D12Resource> update;                    // deforming: vertices for the next refit
        std::shared_ptr<MeshLights> lights;               // null without emissive triangles
    };
    struct EmitterSet {
        int32_t material = -1, mesh = -1, instance = -1;
        uint32_t capacity = 0;
        bool visible = false;
        std::vector<float> positions, normals;
        std::vector<uint64_t> order; // Morton code << 32 | emitter
    };
    struct Instance {
        int32_t mesh = -1;
        float xform[12] = {}, lastFrame[12] = {}, writtenPrev[12] = {};
        bool live = false, visible = true, propsValid = false, seen = false;
        uint32_t lightSlot = 0xFFFFFFFFu, writtenLightSlot = 0xFFFFFFFFu, writtenLightRecord = 0xFFFFFFFFu;
        uint32_t flags = 0;
    };

    int32_t Allocate(std::vector<uint32_t>& freeList, uint32_t& next, uint32_t capacity);
    void Retire(std::function<void()> action);
    Material ToMaterial(const RT_MaterialDesc& d) const;
    void ApplyCone(LightTriangle& r, UINT material) const;
    void WriteMaterial(uint32_t slot, const Material& m);
    void DropMesh(int32_t id);
    void ExcludeLightSlot(Instance& i);
    bool LightSlotLive(uint32_t slot) const;
    void RecordLightUploads(ID3D12GraphicsCommandList4* cl);
    static bool BuildLightTree(MeshLights& l);
    void BuildGeometryDescs(const Mesh& m, D3D12_RAYTRACING_GEOMETRY_DESC out[2], uint32_t& count) const;
    static void WriteVertices(BTriVertex* v, uint32_t count, const float* positions, const float* normals,
                              const float* uvs, const DirectX::PackedVector::XMHALF2* packedUvs);

    Config m_cfg;
    Binding m_b;
    bool m_bound = false;
    std::string m_error;
    uint64_t m_frame = 0;
    std::deque<std::pair<uint64_t, std::function<void()>>> m_retired;

    mc::SpanAllocator m_vtxAlloc, m_idxAlloc, m_matAlloc;
    uint8_t* m_materialIdsMapped = nullptr;
    D3D12_GPU_VIRTUAL_ADDRESS m_vertexVa = 0, m_indexVa = 0;
    UINT m_descriptorSize = 0;

    std::vector<Texture> m_textures;
    std::vector<uint32_t> m_freeTextures;
    uint32_t m_nextTexture = 0;
    std::vector<uint32_t> m_pendingTextures;

    std::vector<uint8_t> m_materialLive;
    std::vector<DirectX::XMFLOAT2> m_materialCone; // per material: the records' coneScale, coneOffset (0 = Lambertian)
    std::vector<uint32_t> m_freeMaterials;
    uint32_t m_nextMaterial = 1;
    std::vector<uint32_t> m_pendingMaterials;

    std::vector<Mesh> m_meshes;
    std::vector<uint32_t> m_freeMeshes;
    uint32_t m_nextMesh = 0;
    std::vector<uint32_t> m_pendingMeshes, m_pendingUpdates;

    std::vector<EmitterSet> m_emitterSets;
    std::vector<uint32_t> m_freeEmitterSets;

    std::vector<Instance> m_instances;
    std::vector<uint32_t> m_freeInstances;
    uint32_t m_nextInstance = 0;
    bool m_instancesChanged = false;
    planet::DVec3 m_lastOrigin{};

    DynamicLightBinding m_lights;
    bool m_lightsBound = false, m_lightTlasHasHost = false, m_lightSetDirty = false, m_warnedLightSpace = false;
    mc::SpanAllocator m_recAlloc, m_nodeAlloc;
    std::vector<uint32_t> m_pendingLights; // meshes whose light data waits for upload
    std::vector<LightSlot> m_lightSlots;
    std::vector<uint32_t> m_freeLightSlots;
    struct RetiredSlot {
        uint32_t slot;
        uint64_t frame;
        uint32_t excludedAt;
    };
    std::vector<RetiredSlot> m_retiredLightSlots;
    uint32_t m_lightVersion = 0, m_lightLive = 0;
};
