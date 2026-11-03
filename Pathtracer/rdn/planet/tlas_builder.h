#pragma once

#include <cstdint>
#include <vector>
#include <unordered_map>
#include "../Common.h"
#include "blas_pool.h"
#include "coordinate_system.h"

namespace planet {

class TlasBuilder {
public:
    void init(ID3D12Device5* device, uint32_t max_instances);

    void reserve(uint32_t required_instances);
    void begin(uint32_t required_instances = 0);

#if RDN_CLUSTER_API
    // Partitioned TLAS from here on (Agility 1.722 preview). Instances are kept by instance ID, which is unique and
    // stable (each owns an instance-property record); a build writes only the instances that changed, appeared or
    // vanished, so only their partitions are rebuilt. Partitions are cells of world space.
    void enable_partitioned(ID3D12DeviceRaytracing2* device);
#endif
    bool partitioned() const { return m_partitioned; }
    // World position of the camera-relative frame, so partitions stay put when the origin moves.
    void set_scene_origin(const DVec3& origin) { m_origin = origin; }

    // Camera-relative transform; diffed against the last frame. dynamic: its BLAS changes every frame in place, so a
    // partitioned TLAS rewrites it every build, in the global partition.
    void add_instance(D3D12_GPU_VIRTUAL_ADDRESS blas,
                      const float transform[12],
                      uint32_t instance_id,
                      uint32_t hit_group_index,
                      D3D12_RAYTRACING_INSTANCE_FLAGS flags,
                      uint8_t mask = 0xFF,
                      bool dynamic = false);

    // Rebuilds on change or force; refit updates in place. Partitioned: writes what changed; force and refit unused.
    bool build(ID3D12GraphicsCommandList4* cmd, bool force = false, bool refit = false);
    bool last_build_recorded() const { return m_lastBuildRecorded; }

    D3D12_GPU_VIRTUAL_ADDRESS tlas_address() const { return result()->GetGPUVirtualAddress(); }
    ID3D12Resource* result()         const { return m_partitioned ? m_ptlas.Get() : m_result.Get(); }
    uint32_t        instance_count() const { return m_count; }
    uint32_t        max_instances()  const { return m_max; }
    uint64_t        result_bytes()   const { return result() ? result()->GetDesc().Width : 0; }
    uint64_t        scratch_bytes()  const;
    // Partitioned: the last build's instance writes, and partitions in use.
    uint32_t        writes()         const { return m_lastWrites; }
    uint32_t        partitions_used() const { return (uint32_t)m_cellPartition.size(); }
    // Partitioned: the instance IDs the last build wrote.
    const std::vector<uint32_t>& written() const { return m_written; }
    // Partitioned: instances and partitions the GPU writes (build_gpu_ops), after the CPU's own. Changing them starts a
    // new structure, as growth does; generation() then changes, and their writer writes them all again. perPartition:
    // the most instances one of their partitions holds.
    void reserve_external(uint32_t instances, uint32_t partitions, uint32_t perPartition);
    uint32_t external_instance_base() const { return m_cpuCapacity; }
    uint32_t external_partition_base() const;
    uint32_t external_instances() const { return m_extInstances; }
    uint32_t generation() const { return m_generation; }
#if RDN_CLUSTER_API
    // Partitioned, after build: a second in-place build of operations the GPU wrote (opsVa: the operations, countVa:
    // their count), applied over the CPU's writes. Their arguments may only update instances the TLAS holds enabled.
    void build_gpu_ops(ID3D12GraphicsCommandList4* cmd, D3D12_GPU_VIRTUAL_ADDRESS opsVa, D3D12_GPU_VIRTUAL_ADDRESS countVa);
#endif

private:
    // Partitioned TLAS state.
    struct Slot {
        D3D12_RAYTRACING_INSTANCE_DESC desc{};
        uint32_t partition = 0;
        uint32_t seen = 0;
        bool active = false, assigned = false;
    };
    struct Frame {
        D3D12_RAYTRACING_INSTANCE_DESC desc;
        bool dynamic;
    };
    uint32_t partition_for(const D3D12_RAYTRACING_INSTANCE_DESC& d, bool dynamic, const Slot& s);
    void assign(Slot& s, uint32_t partition);
    void unassign(Slot& s);
    void create_partitioned(uint32_t instanceCapacity);
    bool build_partitioned(ID3D12GraphicsCommandList4* cmd);

    bool m_partitioned = false;
    DVec3 m_origin{};
#if RDN_CLUSTER_API
    ComPtr<ID3D12DeviceRaytracing2> m_rtas;
    D3D12_RTAS_PARTITIONED_TLAS_INPUTS_DESC m_ptlasInputs{};
#endif
    ComPtr<ID3D12Resource> m_ptlas, m_ptlasScratch;
    std::vector<ComPtr<ID3D12Resource>> m_ptlasArgs; // a ring; a frame in flight still reads its build's arguments
    std::vector<uint8_t*> m_ptlasArgsMapped;
    uint32_t m_ptlasRing = 0;
    std::vector<ComPtr<ID3D12Resource>> m_retired; // replaced by growth, kept for the frames still using them
    uint32_t m_retiredAge = 0;
    bool m_ptlasBuilt = false;
    uint32_t m_serial = 0;
    std::vector<Slot> m_slots;                   // by instance ID
    std::vector<Frame> m_frame;                  // this build's instances
    std::unordered_map<uint64_t, uint32_t> m_cellPartition;
    std::vector<uint64_t> m_partitionCell;
    std::vector<uint32_t> m_partitionCount, m_freePartitions;
    uint32_t m_globalCount = 0;
    uint32_t m_lastWrites = 0;
    std::vector<uint32_t> m_written;
    uint32_t m_cpuCapacity = 0;                    // instance indices the CPU writes; the external ones follow
    uint32_t m_extInstances = 0, m_extPartitions = 0, m_extPerPartition = 0;
    uint32_t m_generation = 0;

    ComPtr<ID3D12Device5> m_device;
    ComPtr<ID3D12Resource> m_result;
    ComPtr<ID3D12Resource> m_scratch;
    ComPtr<ID3D12Resource> m_instanceDescs;
    D3D12_RAYTRACING_INSTANCE_DESC* m_mapped = nullptr;
    uint32_t m_max   = 0;
    uint32_t m_count = 0;
    uint32_t m_builtCount = 0;
    bool m_built = false;
    bool m_changed = true;
    bool m_lastBuildRecorded = false;
    uint32_t m_refitsSinceBuild = 0;
    std::vector<D3D12_RAYTRACING_INSTANCE_DESC> m_descriptors;
};

}
