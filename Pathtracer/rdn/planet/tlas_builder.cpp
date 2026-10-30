#include "tlas_builder.h"
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <cstdio>

namespace planet {

// ALLOW_UPDATE enables the refit path.
static constexpr D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS TLAS_BUILD_FLAGS =
    (D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS)(
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE |
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE);

// Refit quality decays; rebuild after this many.
static constexpr uint32_t MAX_REFITS_BETWEEN_BUILDS = 64;

// Partitioned TLAS: partitions are world cells of PARTITION_CELL units, filled up to MAX_PER_PARTITION instances (the
// spec suggests 100-1000); the rest, and dynamic instances, go to the global partition, which also parks the disabled.
static constexpr uint32_t PARTITION_COUNT   = 4096;
static constexpr uint32_t MAX_PER_PARTITION = 1024;
static constexpr double   PARTITION_CELL    = 256.0;
static constexpr uint32_t GLOBAL_PARTITION  = 0xFFFFFFFFu; // D3D12_RTAS_PARTITIONED_TLAS_PARTITION_INDEX_GLOBAL_PARTITION
static constexpr uint32_t PTLAS_ARG_RING    = 4;           // more than the frames in flight
static constexpr uint64_t PTLAS_OP_OFFSET   = 8;           // after the op count
static constexpr uint64_t PTLAS_ARGS_OFFSET = 64;

void TlasBuilder::init(ID3D12Device5* device, uint32_t max_instances) {
    if (m_instanceDescs && m_mapped) m_instanceDescs->Unmap(0, nullptr);
    m_mapped = nullptr;
    m_instanceDescs.Reset();
    m_device = device;
    m_max = 0;
    m_count = 0;
    m_builtCount = 0;
    m_built = false;
    m_lastBuildRecorded = false;
    m_descriptors.clear();
    reserve(std::max(max_instances, 1u));
}

void TlasBuilder::reserve(uint32_t required_instances) {
    if (required_instances <= m_max) return;
    if (m_partitioned) {
        m_max = required_instances;
#if RDN_CLUSTER_API
        if (required_instances > m_cpuCapacity) create_partitioned(required_instances);
#endif
        return;
    }
    constexpr uint32_t maxDxrInstances = 0xFFFFFFu;
    if (required_instances > maxDxrInstances)
        throw std::runtime_error("planet::TlasBuilder: DXR instance limit exceeded");
    const uint32_t capacity = std::max(required_instances, (uint32_t)std::min(
        uint64_t(maxDxrInstances), uint64_t(m_max) * 2));

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS inputs = {};
    inputs.Type          = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
    inputs.DescsLayout   = D3D12_ELEMENTS_LAYOUT_ARRAY;
    inputs.Flags         = TLAS_BUILD_FLAGS;
    inputs.NumDescs      = capacity;
    inputs.InstanceDescs = 0;

    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO info = {};
    m_device->GetRaytracingAccelerationStructurePrebuildInfo(&inputs, &info);

    auto result  = create_buffer(m_device.Get(), info.ResultDataMaxSizeInBytes,
                              D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                              D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
                              HEAP_DEFAULT);
    // Shared by build and update.
    auto scratch = create_buffer(m_device.Get(),
                              std::max(info.ScratchDataSizeInBytes, info.UpdateScratchDataSizeInBytes),
                              D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                              D3D12_RESOURCE_STATE_COMMON,
                              HEAP_DEFAULT);

    const uint64_t descBytes = (uint64_t)capacity * sizeof(D3D12_RAYTRACING_INSTANCE_DESC);
    auto descriptors = create_buffer(m_device.Get(), descBytes, D3D12_RESOURCE_FLAG_NONE,
                                    D3D12_RESOURCE_STATE_GENERIC_READ, HEAP_UPLOAD);
    void* mapped = nullptr;
    D3D12_RANGE no_read = { 0, 0 };
    if (FAILED(descriptors->Map(0, &no_read, &mapped)))
        throw std::runtime_error("planet::TlasBuilder: instance-desc Map failed");
    if (m_instanceDescs && m_mapped) m_instanceDescs->Unmap(0, nullptr);
    m_result = std::move(result);
    m_scratch = std::move(scratch);
    m_instanceDescs = std::move(descriptors);
    m_mapped = static_cast<D3D12_RAYTRACING_INSTANCE_DESC*>(mapped);
    if (!m_descriptors.empty())
        std::memcpy(m_mapped, m_descriptors.data(), m_descriptors.size() * sizeof(m_descriptors[0]));
    m_max = capacity;
    m_built = false;
}

void TlasBuilder::begin(uint32_t required_instances) {
    reserve(required_instances);
    m_count = 0;
    m_changed = false;
    m_frame.clear();
}

void TlasBuilder::add_instance(D3D12_GPU_VIRTUAL_ADDRESS blas,
                               const float transform[12],
                               uint32_t instance_id,
                               uint32_t hit_group_index,
                               D3D12_RAYTRACING_INSTANCE_FLAGS flags,
                               uint8_t mask,
                               bool dynamic) {
    if (instance_id > 0xFFFFFFu || hit_group_index > 0xFFFFFFu)
        throw std::runtime_error("planet::TlasBuilder: instance or hit-group ID exceeds 24 bits");
    D3D12_RAYTRACING_INSTANCE_DESC d{};
    std::memcpy(d.Transform, transform, sizeof(float) * 12);
    d.InstanceID                          = instance_id     & 0xFFFFFFu;
    d.InstanceMask                        = mask;
    d.InstanceContributionToHitGroupIndex = hit_group_index & 0xFFFFFFu;
    d.Flags                               = (UINT)flags     & 0xFFu;
    d.AccelerationStructure               = blas;
    if (m_partitioned) {
        m_frame.push_back(Frame{ d, dynamic });
        ++m_count;
        return;
    }
    reserve(m_count + 1);
    if (m_count >= m_descriptors.size()) {
        m_descriptors.push_back(d);
        m_mapped[m_count] = d;
        m_changed = true;
    } else if (std::memcmp(&m_descriptors[m_count], &d, sizeof(d)) != 0) {
        m_descriptors[m_count] = d;
        m_mapped[m_count] = d;
        m_changed = true;
    }
    ++m_count;
}

bool TlasBuilder::build(ID3D12GraphicsCommandList4* cmd, bool force, bool refit) {
    if (m_partitioned) return build_partitioned(cmd);
    const bool rebuild = !m_built || m_changed || m_builtCount != m_count || force ||
                         (refit && m_refitsSinceBuild >= MAX_REFITS_BETWEEN_BUILDS);
    m_lastBuildRecorded = rebuild || refit;
    if (!m_lastBuildRecorded) return false;

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC desc = {};
    desc.Inputs.Type          = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
    desc.Inputs.DescsLayout   = D3D12_ELEMENTS_LAYOUT_ARRAY;
    desc.Inputs.Flags         = TLAS_BUILD_FLAGS;
    desc.Inputs.NumDescs      = m_count;
    desc.Inputs.InstanceDescs = m_instanceDescs->GetGPUVirtualAddress();
    desc.DestAccelerationStructureData    = m_result->GetGPUVirtualAddress();
    desc.ScratchAccelerationStructureData = m_scratch->GetGPUVirtualAddress();
    desc.SourceAccelerationStructureData  = 0;

    if (!rebuild) {
        // In-place update: source == destination.
        desc.Inputs.Flags = (D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS)(
            desc.Inputs.Flags | D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE);
        desc.SourceAccelerationStructureData = m_result->GetGPUVirtualAddress();
        ++m_refitsSinceBuild;
    } else {
        m_refitsSinceBuild = 0;
    }

    cmd->BuildRaytracingAccelerationStructure(&desc, 0, nullptr);

    D3D12_RESOURCE_BARRIER uav = {};
    uav.Type          = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    uav.UAV.pResource = m_result.Get();
    cmd->ResourceBarrier(1, &uav);
    m_built = true;
    m_builtCount = m_count;
    return true;
}

uint64_t TlasBuilder::scratch_bytes() const {
    ID3D12Resource* s = m_partitioned ? m_ptlasScratch.Get() : m_scratch.Get();
    return s ? s->GetDesc().Width : 0;
}

#if RDN_CLUSTER_API
void TlasBuilder::enable_partitioned(ID3D12DeviceRaytracing2* device) {
    m_rtas = device;
    m_partitioned = true;
    if (m_instanceDescs && m_mapped) m_instanceDescs->Unmap(0, nullptr);
    m_mapped = nullptr;
    m_instanceDescs.Reset();
    m_result.Reset();
    m_scratch.Reset();
    m_descriptors.clear();
    create_partitioned(std::max(m_max, 1024u));
}

// Growth starts over: a new structure with every instance written again (the caller rebinds the new address).
void TlasBuilder::create_partitioned(uint32_t instanceCapacity) {
    uint32_t capacity = 1024;
    while (capacity < instanceCapacity) capacity *= 2;
    if (m_ptlas) m_retired.push_back(m_ptlas);
    if (m_ptlasScratch) m_retired.push_back(m_ptlasScratch);
    for (auto& a : m_ptlasArgs) m_retired.push_back(a);

    D3D12_RTAS_PARTITIONED_TLAS_INPUTS_DESC& in = m_ptlasInputs;
    in = {};
    in.Flags = D3D12_RTAS_PARTITIONED_TLAS_FLAG_FAST_TRACE;
    // The GPU's instances follow the scene origin by translating their partitions.
    if (m_extPartitions) in.Flags |= D3D12_RTAS_PARTITIONED_TLAS_FLAG_ENABLE_PARTITION_TRANSLATION;
    in.InstanceCount = capacity + m_extInstances;
    in.MaxInstancePerPartitionCount = MAX_PER_PARTITION;
    in.PartitionCount = PARTITION_COUNT + m_extPartitions;
    in.MaxInstanceInGlobalPartitionCount = capacity;
    m_cpuCapacity = capacity;
    ++m_generation;
    D3D12_RTAS_OPERATION_INPUTS opIn = {};
    opIn.Type = D3D12_RTAS_OPERATION_TYPE_PARTITIONED_TLAS;
    opIn.pPartitionedTLASInputsDesc = &in;
    D3D12_RTAS_OPERATION_PREBUILD_INFO info = {};
    m_rtas->GetRTASOperationPrebuildInfo(&opIn, &info);
    if (!info.ResultDataMaxSizeInBytes) throw std::runtime_error("planet::TlasBuilder: partitioned TLAS prebuild query failed");

    m_ptlas = create_buffer(m_device.Get(), info.ResultDataMaxSizeInBytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                            D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, HEAP_DEFAULT);
    m_ptlas->SetName(L"PartitionedTLAS");
    m_ptlasScratch = create_buffer(m_device.Get(), std::max<uint64_t>(info.ScratchDataSizeInBytes, 256),
                                   D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                   HEAP_DEFAULT);
    const uint64_t argBytes =
        PTLAS_ARGS_OFFSET + (uint64_t)capacity * sizeof(D3D12_RTAS_PARTITIONED_TLAS_OPERATION_WRITE_INSTANCE_ARGS);
    m_ptlasArgs.resize(PTLAS_ARG_RING);
    m_ptlasArgsMapped.assign(PTLAS_ARG_RING, nullptr);
    for (uint32_t i = 0; i < PTLAS_ARG_RING; ++i) {
        m_ptlasArgs[i] = create_buffer(m_device.Get(), argBytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ,
                                       HEAP_UPLOAD);
        void* p = nullptr;
        D3D12_RANGE noRead = { 0, 0 };
        if (FAILED(m_ptlasArgs[i]->Map(0, &noRead, &p)))
            throw std::runtime_error("planet::TlasBuilder: partitioned TLAS argument Map failed");
        m_ptlasArgsMapped[i] = static_cast<uint8_t*>(p);
    }
    m_ptlasBuilt = false;
    m_slots.assign(capacity, Slot{});
    m_cellPartition.clear();
    m_partitionCell.assign(PARTITION_COUNT, 0);
    m_partitionCount.assign(PARTITION_COUNT, 0);
    m_freePartitions.resize(PARTITION_COUNT);
    for (uint32_t p = 0; p < PARTITION_COUNT; ++p) m_freePartitions[p] = PARTITION_COUNT - 1 - p;
    m_globalCount = 0;
    std::printf("[tlas] partitioned: %u instances (%u GPU-written), %u partitions of up to %u; %.1f MB + %.1f MB "
                "scratch\n",
                in.InstanceCount, m_extInstances, in.PartitionCount, MAX_PER_PARTITION,
                (double)info.ResultDataMaxSizeInBytes / 1048576.0, (double)info.ScratchDataSizeInBytes / 1048576.0);
}

void TlasBuilder::reserve_external(uint32_t instances, uint32_t partitions) {
    if (!m_partitioned || (instances == m_extInstances && partitions == m_extPartitions)) return;
    m_extInstances = instances;
    m_extPartitions = partitions;
    create_partitioned(std::max(m_cpuCapacity, 1024u));
}
#endif

uint32_t TlasBuilder::external_partition_base() const { return PARTITION_COUNT; }

// The partition of the world cell holding the instance's origin; an instance stays in a full partition it is already in.
uint32_t TlasBuilder::partition_for(const D3D12_RAYTRACING_INSTANCE_DESC& d, bool dynamic, const Slot& s) {
    if (dynamic) return GLOBAL_PARTITION;
    auto cell = [&](int row, double origin) {
        return (uint64_t)((int64_t)std::floor((d.Transform[row][3] + origin) / PARTITION_CELL) & 0x1FFFFF);
    };
    const uint64_t key = (cell(0, m_origin.x) << 42) | (cell(1, m_origin.y) << 21) | cell(2, m_origin.z);
    auto it = m_cellPartition.find(key);
    if (it != m_cellPartition.end()) {
        if (s.assigned && s.partition == it->second) return it->second;
        return m_partitionCount[it->second] < MAX_PER_PARTITION ? it->second : GLOBAL_PARTITION;
    }
    if (m_freePartitions.empty()) return GLOBAL_PARTITION;
    const uint32_t p = m_freePartitions.back();
    m_freePartitions.pop_back();
    m_cellPartition.emplace(key, p);
    m_partitionCell[p] = key;
    return p;
}

void TlasBuilder::assign(Slot& s, uint32_t partition) {
    if (partition == GLOBAL_PARTITION) ++m_globalCount;
    else ++m_partitionCount[partition];
    s.partition = partition;
    s.assigned = true;
}

void TlasBuilder::unassign(Slot& s) {
    if (!s.assigned) return;
    s.assigned = false;
    if (s.partition == GLOBAL_PARTITION) { --m_globalCount; return; }
    if (--m_partitionCount[s.partition] == 0) {
        m_cellPartition.erase(m_partitionCell[s.partition]);
        m_freePartitions.push_back(s.partition);
    }
}

bool TlasBuilder::build_partitioned(ID3D12GraphicsCommandList4* cmd) {
#if RDN_CLUSTER_API
    uint32_t maxId = 0;
    for (const Frame& f : m_frame) maxId = std::max(maxId, f.desc.InstanceID);
    if (!m_frame.empty() && maxId >= m_cpuCapacity) create_partitioned(maxId + 1);

    using Args = D3D12_RTAS_PARTITIONED_TLAS_OPERATION_WRITE_INSTANCE_ARGS;
    ++m_serial;
    m_ptlasRing = (m_ptlasRing + 1) % PTLAS_ARG_RING;
    uint8_t* base = m_ptlasArgsMapped[m_ptlasRing];
    Args* args = reinterpret_cast<Args*>(base + PTLAS_ARGS_OFFSET);
    uint32_t n = 0;
    m_written.clear();
    auto write = [&](const D3D12_RAYTRACING_INSTANCE_DESC& d, uint32_t index, uint32_t partition, bool enabled) {
        m_written.push_back(index);
        Args a = {};
        std::memcpy(a.Transform, d.Transform, sizeof(a.Transform));
        a.InstanceID = d.InstanceID;
        a.InstanceMask = d.InstanceMask;
        a.InstanceContributionToHitGroupIndex = d.InstanceContributionToHitGroupIndex;
        a.InstanceFlags = d.Flags; // the low instance flags match D3D12_RTAS_PARTITIONED_TLAS_INSTANCE_FLAGS
        a.AccelerationStructure = enabled ? d.AccelerationStructure : 0;
        a.InstanceIndex = index;
        a.PartitionIndex = partition;
        args[n++] = a;
    };
    for (const Frame& f : m_frame) {
        Slot& s = m_slots[f.desc.InstanceID];
        if (s.seen == m_serial) continue; // one instance per ID
        s.seen = m_serial;
        const uint32_t p = partition_for(f.desc, f.dynamic, s);
        if (!f.dynamic && s.active && s.assigned && s.partition == p &&
            std::memcmp(&s.desc, &f.desc, sizeof(f.desc)) == 0)
            continue;
        if (!s.assigned || s.partition != p) {
            unassign(s);
            assign(s, p);
        }
        s.desc = f.desc;
        s.active = true;
        write(f.desc, f.desc.InstanceID, p, true);
    }
    // Gone since the last build: disabled, and parked in the global partition so cells only count live instances.
    for (uint32_t i = 0; i < (uint32_t)m_slots.size(); ++i) {
        Slot& s = m_slots[i];
        if (!s.active || s.seen == m_serial) continue;
        s.active = false;
        if (s.partition != GLOBAL_PARTITION) {
            unassign(s);
            assign(s, GLOBAL_PARTITION);
        }
        write(s.desc, i, GLOBAL_PARTITION, false);
    }
    m_lastWrites = n;
    m_lastBuildRecorded = n > 0 || !m_ptlasBuilt;
    if (!m_lastBuildRecorded) return false;

    const D3D12_GPU_VIRTUAL_ADDRESS argsVa = m_ptlasArgs[m_ptlasRing]->GetGPUVirtualAddress();
    *reinterpret_cast<uint32_t*>(base) = n ? 1u : 0u;
    auto* op = reinterpret_cast<D3D12_RTAS_PARTITIONED_TLAS_OPERATION*>(base + PTLAS_OP_OFFSET);
    op->Type = D3D12_RTAS_PARTITIONED_TLAS_OPERATION_TYPE_WRITE_INSTANCE;
    op->ArgCount = n;
    op->ArgData = { argsVa + PTLAS_ARGS_OFFSET, sizeof(Args) };

    const D3D12_GPU_VIRTUAL_ADDRESS va = m_ptlas->GetGPUVirtualAddress();
    D3D12_RTAS_PARTITIONED_TLAS_OPERATION_DATA data = {};
    data.SourceAccelerationStructureData = m_ptlasBuilt ? va : 0; // in place after the first build
    data.DestAccelerationStructureData = va;
    data.ScratchAccelerationStructureData = m_ptlasScratch->GetGPUVirtualAddress();
    data.IndirectPartitionedTlasOpCount = argsVa;
    data.IndirectPartitionedTlasOps = argsVa + PTLAS_OP_OFFSET;
    D3D12_RTAS_OPERATION_DESC desc = {};
    desc.Inputs.Type = D3D12_RTAS_OPERATION_TYPE_PARTITIONED_TLAS;
    desc.Inputs.pPartitionedTLASInputsDesc = &m_ptlasInputs;
    desc.pPartitionedTlasOperationData = &data;
    ComPtr<ID3D12CommandListRaytracing2> list;
    if (FAILED(cmd->QueryInterface(IID_PPV_ARGS(&list))))
        throw std::runtime_error("planet::TlasBuilder: RTAS command list unavailable");
    list->ExecuteIndirectRTASOperations(1, &desc, D3D12_EXECUTE_INDIRECT_RTAS_OPERATIONS_FLAG_NONE);
    const D3D12_RESOURCE_BARRIER uav = CD3DX12_RESOURCE_BARRIER::UAV(m_ptlas.Get());
    cmd->ResourceBarrier(1, &uav);
    m_ptlasBuilt = true;
    // Replaced buffers outlive the frames that may still read them.
    if (!m_retired.empty() && ++m_retiredAge > PTLAS_ARG_RING) { m_retired.clear(); m_retiredAge = 0; }
    return true;
#else
    (void)cmd;
    return false;
#endif
}

#if RDN_CLUSTER_API
void TlasBuilder::build_gpu_ops(ID3D12GraphicsCommandList4* cmd, D3D12_GPU_VIRTUAL_ADDRESS opsVa,
                                D3D12_GPU_VIRTUAL_ADDRESS countVa) {
    if (!m_partitioned || !m_ptlasBuilt) return;
    const D3D12_RESOURCE_BARRIER all = CD3DX12_RESOURCE_BARRIER::UAV(nullptr); // the TLAS and the shared scratch
    cmd->ResourceBarrier(1, &all);
    const D3D12_RESOURCE_BARRIER uav = CD3DX12_RESOURCE_BARRIER::UAV(m_ptlas.Get());
    const D3D12_GPU_VIRTUAL_ADDRESS va = m_ptlas->GetGPUVirtualAddress();
    D3D12_RTAS_PARTITIONED_TLAS_OPERATION_DATA data = {};
    data.SourceAccelerationStructureData = va;
    data.DestAccelerationStructureData = va;
    data.ScratchAccelerationStructureData = m_ptlasScratch->GetGPUVirtualAddress();
    data.IndirectPartitionedTlasOpCount = countVa;
    data.IndirectPartitionedTlasOps = opsVa;
    D3D12_RTAS_OPERATION_DESC desc = {};
    desc.Inputs.Type = D3D12_RTAS_OPERATION_TYPE_PARTITIONED_TLAS;
    desc.Inputs.pPartitionedTLASInputsDesc = &m_ptlasInputs;
    desc.pPartitionedTlasOperationData = &data;
    ComPtr<ID3D12CommandListRaytracing2> list;
    if (FAILED(cmd->QueryInterface(IID_PPV_ARGS(&list))))
        throw std::runtime_error("planet::TlasBuilder: RTAS command list unavailable");
    list->ExecuteIndirectRTASOperations(1, &desc, D3D12_EXECUTE_INDIRECT_RTAS_OPERATIONS_FLAG_NONE);
    cmd->ResourceBarrier(1, &uav);
}
#endif

}
