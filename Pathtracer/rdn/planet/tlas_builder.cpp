//====================================
//PLANET - TLAS BUILDER
//====================================

#include "tlas_builder.h"
#include <cstring>
#include <stdexcept>
#include <algorithm>

namespace planet {

//Keep the traversal-quality policy: rebuild changed scenes, reuse static ones.
static constexpr D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS TLAS_BUILD_FLAGS =
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;

void TlasBuilder::init(ID3D12Device5* device, uint32_t max_instances) {
    if (m_instanceDescs && m_mapped) m_instanceDescs->Unmap(0, nullptr);
    m_mapped = nullptr;
    m_instanceDescs.Reset();
    m_device = device;
    m_max = 0;
    m_count = 0;
    m_builtCount = 0;
    m_built = false;
    m_descriptors.clear();
    reserve(std::max(max_instances, 1u));
}

void TlasBuilder::reserve(uint32_t required_instances) {
    if (required_instances <= m_max) return;
    constexpr uint32_t maxDxrInstances = 0xFFFFFFu;
    if (required_instances > maxDxrInstances)
        throw std::runtime_error("planet::TlasBuilder: DXR instance limit exceeded");
    const uint32_t capacity = std::max(required_instances, (uint32_t)std::min(
        uint64_t(maxDxrInstances), uint64_t(m_max) * 2));

    //prebuild sized for the worst-case instance count (addresses are ignored)
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
    auto scratch = create_buffer(m_device.Get(), info.ScratchDataSizeInBytes,
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
}

void TlasBuilder::add_instance(D3D12_GPU_VIRTUAL_ADDRESS blas,
                               const float transform[12],
                               uint32_t instance_id,
                               uint32_t hit_group_index,
                               D3D12_RAYTRACING_INSTANCE_FLAGS flags) {
    if (instance_id > 0xFFFFFFu || hit_group_index > 0xFFFFFFu)
        throw std::runtime_error("planet::TlasBuilder: instance or hit-group ID exceeds 24 bits");
    reserve(m_count + 1);
    D3D12_RAYTRACING_INSTANCE_DESC d{};
    std::memcpy(d.Transform, transform, sizeof(float) * 12);
    d.InstanceID                          = instance_id     & 0xFFFFFFu;
    d.InstanceMask                        = 0xFF;
    d.InstanceContributionToHitGroupIndex = hit_group_index & 0xFFFFFFu;
    d.Flags                               = (UINT)flags     & 0xFFu;
    d.AccelerationStructure               = blas;
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

bool TlasBuilder::build(ID3D12GraphicsCommandList4* cmd, bool force) {
    m_lastBuildRecorded = !m_built || m_changed || m_builtCount != m_count || force;
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

    cmd->BuildRaytracingAccelerationStructure(&desc, 0, nullptr);

    D3D12_RESOURCE_BARRIER uav = {};
    uav.Type          = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    uav.UAV.pResource = m_result.Get();
    cmd->ResourceBarrier(1, &uav);
    m_built = true;
    m_builtCount = m_count;
    return true;
}

} // namespace planet
