/*-----------------------------------------------------------------------
Copyright (c) 2014-2018, NVIDIA. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
* Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
* Neither the name of its contributors may be used to endorse
or promote products derived from this software without specific
prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
-----------------------------------------------------------------------*/

#include <stdexcept>
#include "TopLevelASGenerator.h"

#ifndef ROUND_UP
#define ROUND_UP(v, powerOf2Alignment) (((v) + (powerOf2Alignment) - 1) & ~((powerOf2Alignment) - 1))
#endif

namespace nv_helpers_dx12 {

void TopLevelASGenerator::AddInstance(ID3D12Resource* bottomLevelAS,

                                      const DirectX::XMMATRIX& transform,

                                      UINT instanceID,

                                      UINT hitGroupIndex,

                                      D3D12_RAYTRACING_INSTANCE_FLAGS flags) {
    m_instances.emplace_back(Instance(bottomLevelAS, transform, instanceID, hitGroupIndex, flags));
}

void TopLevelASGenerator::ComputeASBufferSizes(ID3D12Device5* device, bool allowUpdate,

                                               UINT64* scratchSizeInBytes,

                                               UINT64* resultSizeInBytes,

                                               UINT64* descriptorsSizeInBytes

) {

    m_flags = allowUpdate ? D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE
                          : D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_NONE;

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS
    prebuildDesc = {};
    prebuildDesc.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
    prebuildDesc.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    prebuildDesc.NumDescs = static_cast<UINT>(m_instances.size());
    prebuildDesc.Flags = m_flags;

    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO info = {};

    device->GetRaytracingAccelerationStructurePrebuildInfo(&prebuildDesc, &info);

    info.ResultDataMaxSizeInBytes =
        ROUND_UP(info.ResultDataMaxSizeInBytes, D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);
    info.ScratchDataSizeInBytes = ROUND_UP(info.ScratchDataSizeInBytes, D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);

    m_resultSizeInBytes = info.ResultDataMaxSizeInBytes;
    m_scratchSizeInBytes = info.ScratchDataSizeInBytes;

    m_instanceDescsSizeInBytes =
        ROUND_UP(sizeof(D3D12_RAYTRACING_INSTANCE_DESC) * static_cast<UINT64>(m_instances.size()),
                 D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);

    *scratchSizeInBytes = m_scratchSizeInBytes;
    *resultSizeInBytes = m_resultSizeInBytes;
    *descriptorsSizeInBytes = m_instanceDescsSizeInBytes;
}

void TopLevelASGenerator::Generate(ID3D12GraphicsCommandList4* commandList, ID3D12Resource* scratchBuffer,

                                   ID3D12Resource* resultBuffer, ID3D12Resource* descriptorsBuffer,

                                   bool updateOnly,

                                   ID3D12Resource* previousResult

) {

    D3D12_RAYTRACING_INSTANCE_DESC* instanceDescs;
    descriptorsBuffer->Map(0, nullptr, reinterpret_cast<void**>(&instanceDescs));
    if (!instanceDescs) {
        throw std::logic_error("Cannot map the instance descriptor buffer - is it "
                               "in the upload heap?");
    }

    auto instanceCount = static_cast<UINT>(m_instances.size());

    if (!updateOnly) {
        ZeroMemory(instanceDescs, m_instanceDescsSizeInBytes);
    }

    for (uint32_t i = 0; i < instanceCount; i++) {

        instanceDescs[i].InstanceID = m_instances[i].instanceID;

        instanceDescs[i].InstanceContributionToHitGroupIndex = m_instances[i].hitGroupIndex;

        instanceDescs[i].Flags = m_instances[i].flags;

        // D3D12 instance descriptors store the transposed 3x4 transform.
        DirectX::XMMATRIX m = XMMatrixTranspose(m_instances[i].transform);
        memcpy(instanceDescs[i].Transform, &m, sizeof(instanceDescs[i].Transform));

        instanceDescs[i].AccelerationStructure = m_instances[i].bottomLevelAS->GetGPUVirtualAddress();

        instanceDescs[i].InstanceMask = 0xFF;
    }

    descriptorsBuffer->Unmap(0, nullptr);

    D3D12_GPU_VIRTUAL_ADDRESS pSourceAS = updateOnly ? previousResult->GetGPUVirtualAddress() : 0;

    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS flags = m_flags;

    if (flags == D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE && updateOnly) {
        flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE;
    }

    if (m_flags != D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE && updateOnly) {
        throw std::logic_error("Cannot update a top-level AS not originally built for updates");
    }
    if (updateOnly && previousResult == nullptr) {
        throw std::logic_error("Top-level hierarchy update requires the previous hierarchy");
    }

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC buildDesc = {};
    buildDesc.Inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
    buildDesc.Inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    buildDesc.Inputs.InstanceDescs = descriptorsBuffer->GetGPUVirtualAddress();
    buildDesc.Inputs.NumDescs = instanceCount;
    buildDesc.DestAccelerationStructureData = {resultBuffer->GetGPUVirtualAddress()};
    buildDesc.ScratchAccelerationStructureData = {scratchBuffer->GetGPUVirtualAddress()};
    buildDesc.SourceAccelerationStructureData = pSourceAS;
    buildDesc.Inputs.Flags = flags;

    commandList->BuildRaytracingAccelerationStructure(&buildDesc, 0, nullptr);

    D3D12_RESOURCE_BARRIER uavBarrier;
    uavBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    uavBarrier.UAV.pResource = resultBuffer;
    uavBarrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    commandList->ResourceBarrier(1, &uavBarrier);
}

// Patch changed descriptors while preserving the existing instance layout.
void TopLevelASGenerator::UpdateAndRefit(ID3D12GraphicsCommandList4* commandList, ID3D12Resource* scratchBuffer,
                                         ID3D12Resource* resultBuffer, ID3D12Resource* descriptorsBuffer,
                                         const std::vector<uint32_t>& dirtyIndices) {
    D3D12_RAYTRACING_INSTANCE_DESC* instanceDescs;
    descriptorsBuffer->Map(0, nullptr, reinterpret_cast<void**>(&instanceDescs));
    if (!instanceDescs)
        throw std::logic_error("Cannot map the instance descriptor buffer");

    for (uint32_t i : dirtyIndices) {
        if (i >= m_instances.size())
            continue;
        DirectX::XMMATRIX m = XMMatrixTranspose(m_instances[i].transform);
        memcpy(instanceDescs[i].Transform, &m, sizeof(instanceDescs[i].Transform));
    }
    descriptorsBuffer->Unmap(0, nullptr);

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC buildDesc = {};
    buildDesc.Inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
    buildDesc.Inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    buildDesc.Inputs.InstanceDescs = descriptorsBuffer->GetGPUVirtualAddress();
    buildDesc.Inputs.NumDescs = static_cast<UINT>(m_instances.size());
    buildDesc.DestAccelerationStructureData = resultBuffer->GetGPUVirtualAddress();
    buildDesc.ScratchAccelerationStructureData = scratchBuffer->GetGPUVirtualAddress();
    buildDesc.SourceAccelerationStructureData = resultBuffer->GetGPUVirtualAddress();
    buildDesc.Inputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE;

    commandList->BuildRaytracingAccelerationStructure(&buildDesc, 0, nullptr);

    D3D12_RESOURCE_BARRIER uavBarrier;
    uavBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    uavBarrier.UAV.pResource = resultBuffer;
    uavBarrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    commandList->ResourceBarrier(1, &uavBarrier);
}

void TopLevelASGenerator::RebuildInPlace(ID3D12GraphicsCommandList4* commandList, ID3D12Resource* scratchBuffer,
                                         ID3D12Resource* resultBuffer, ID3D12Resource* descriptorsBuffer,
                                         const std::vector<uint32_t>& dirtyIndices) {
    D3D12_RAYTRACING_INSTANCE_DESC* instanceDescs;
    descriptorsBuffer->Map(0, nullptr, reinterpret_cast<void**>(&instanceDescs));
    if (!instanceDescs)
        throw std::logic_error("Cannot map the instance descriptor buffer");

    for (uint32_t i : dirtyIndices) {
        if (i >= m_instances.size())
            continue;
        DirectX::XMMATRIX m = XMMatrixTranspose(m_instances[i].transform);
        memcpy(instanceDescs[i].Transform, &m, sizeof(instanceDescs[i].Transform));
    }
    descriptorsBuffer->Unmap(0, nullptr);

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC buildDesc = {};
    buildDesc.Inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
    buildDesc.Inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    buildDesc.Inputs.InstanceDescs = descriptorsBuffer->GetGPUVirtualAddress();
    buildDesc.Inputs.NumDescs = static_cast<UINT>(m_instances.size());
    buildDesc.DestAccelerationStructureData = resultBuffer->GetGPUVirtualAddress();
    buildDesc.ScratchAccelerationStructureData = scratchBuffer->GetGPUVirtualAddress();
    buildDesc.SourceAccelerationStructureData = 0;
    buildDesc.Inputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE |
                             D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD;

    commandList->BuildRaytracingAccelerationStructure(&buildDesc, 0, nullptr);

    D3D12_RESOURCE_BARRIER uavBarrier;
    uavBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    uavBarrier.UAV.pResource = resultBuffer;
    uavBarrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    commandList->ResourceBarrier(1, &uavBarrier);
}

TopLevelASGenerator::Instance::Instance(ID3D12Resource* blAS, const DirectX::XMMATRIX& tr, UINT iID, UINT hgId,
                                        D3D12_RAYTRACING_INSTANCE_FLAGS f)
    : bottomLevelAS(blAS), transform(tr), instanceID(iID), hitGroupIndex(hgId), flags(f) {}
} // namespace nv_helpers_dx12
