//====================================
//PLANET - TLAS BUILDER
//====================================

#include "tlas_builder.h"
#include <cstring>
#include <stdexcept>

namespace planet {

//full rebuild every frame; static unified scene -> PREFER_FAST_TRACE, no UPDATE.
static constexpr D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS TLAS_BUILD_FLAGS =
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;

void TlasBuilder::init(ID3D12Device5* device, uint32_t max_instances) {
    m_max   = max_instances ? max_instances : 1;
    m_count = 0;

    //prebuild sized for the worst-case instance count (addresses are ignored)
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS inputs = {};
    inputs.Type          = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
    inputs.DescsLayout   = D3D12_ELEMENTS_LAYOUT_ARRAY;
    inputs.Flags         = TLAS_BUILD_FLAGS;
    inputs.NumDescs      = m_max;
    inputs.InstanceDescs = 0;

    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO info = {};
    device->GetRaytracingAccelerationStructurePrebuildInfo(&inputs, &info);

    m_result  = create_buffer(device, info.ResultDataMaxSizeInBytes,
                              D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                              D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
                              HEAP_DEFAULT);
    m_scratch = create_buffer(device, info.ScratchDataSizeInBytes,
                              D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                              D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                              HEAP_DEFAULT);

    const uint64_t descBytes = (uint64_t)m_max * sizeof(D3D12_RAYTRACING_INSTANCE_DESC);
    m_instanceDescs = create_buffer(device, descBytes, D3D12_RESOURCE_FLAG_NONE,
                                    D3D12_RESOURCE_STATE_GENERIC_READ, HEAP_UPLOAD);
    void* mapped = nullptr;
    D3D12_RANGE no_read = { 0, 0 };
    if (FAILED(m_instanceDescs->Map(0, &no_read, &mapped)))
        throw std::runtime_error("planet::TlasBuilder: instance-desc Map failed");
    m_mapped = static_cast<D3D12_RAYTRACING_INSTANCE_DESC*>(mapped);
}

void TlasBuilder::begin() { m_count = 0; }

void TlasBuilder::add_instance(D3D12_GPU_VIRTUAL_ADDRESS blas,
                               const float transform[12],
                               uint32_t instance_id,
                               uint32_t hit_group_index,
                               D3D12_RAYTRACING_INSTANCE_FLAGS flags) {
    if (m_count >= m_max) return;                         // fail soft on overflow
    D3D12_RAYTRACING_INSTANCE_DESC& d = m_mapped[m_count++];
    std::memcpy(d.Transform, transform, sizeof(float) * 12);
    d.InstanceID                          = instance_id     & 0xFFFFFFu;
    d.InstanceMask                        = 0xFF;
    d.InstanceContributionToHitGroupIndex = hit_group_index & 0xFFFFFFu;
    d.Flags                               = (UINT)flags     & 0xFFu;
    d.AccelerationStructure               = blas;
}

void TlasBuilder::build(ID3D12GraphicsCommandList4* cmd) {
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
}

} // namespace planet
