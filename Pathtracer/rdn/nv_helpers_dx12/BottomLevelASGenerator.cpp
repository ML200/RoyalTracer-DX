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
#include <memory>
#include "BottomLevelASGenerator.h"

#ifndef ROUND_UP
#define ROUND_UP(v, powerOf2Alignment) (((v) + (powerOf2Alignment) - 1) & ~((powerOf2Alignment) - 1))
#endif

namespace nv_helpers_dx12 {

void BottomLevelASGenerator::AddVertexBuffer(ID3D12Resource* vertexBuffer,

                                             UINT64 vertexOffsetInBytes, uint32_t vertexCount, UINT vertexSizeInBytes,

                                             ID3D12Resource* transformBuffer,

                                             UINT64 transformOffsetInBytes,

                                             bool isOpaque

) {
    AddVertexBuffer(vertexBuffer, vertexOffsetInBytes, vertexCount, vertexSizeInBytes, nullptr, 0, 0, transformBuffer,
                    transformOffsetInBytes, isOpaque);
}

void BottomLevelASGenerator::AddVertexBuffer(ID3D12Resource* vertexBuffer,

                                             UINT64 vertexOffsetInBytes, uint32_t vertexCount, UINT vertexSizeInBytes,

                                             ID3D12Resource* indexBuffer,

                                             UINT64 indexOffsetInBytes, uint32_t indexCount,
                                             ID3D12Resource* transformBuffer,

                                             UINT64 transformOffsetInBytes,

                                             bool isOpaque

) {

    D3D12_RAYTRACING_GEOMETRY_DESC descriptor = {};
    descriptor.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
    descriptor.Triangles.VertexBuffer.StartAddress = vertexBuffer->GetGPUVirtualAddress() + vertexOffsetInBytes;
    descriptor.Triangles.VertexBuffer.StrideInBytes = vertexSizeInBytes;
    descriptor.Triangles.VertexCount = vertexCount;
    descriptor.Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
    descriptor.Triangles.IndexBuffer = indexBuffer ? (indexBuffer->GetGPUVirtualAddress() + indexOffsetInBytes) : 0;
    descriptor.Triangles.IndexFormat = indexBuffer ? DXGI_FORMAT_R32_UINT : DXGI_FORMAT_UNKNOWN;
    descriptor.Triangles.IndexCount = indexCount;
    descriptor.Triangles.Transform3x4 =
        transformBuffer ? (transformBuffer->GetGPUVirtualAddress() + transformOffsetInBytes) : 0;
    descriptor.Flags = isOpaque ? D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE : D3D12_RAYTRACING_GEOMETRY_FLAG_NONE;

    m_vertexBuffers.push_back(descriptor);
}

void BottomLevelASGenerator::AddVertexBufferWithOMM(ID3D12Resource* vertexBuffer, UINT64 vertexOffsetInBytes,
                                                    uint32_t vertexCount, UINT vertexSizeInBytes,
                                                    ID3D12Resource* indexBuffer, UINT64 indexOffsetInBytes,
                                                    uint32_t indexCount, ID3D12Resource* transformBuffer,
                                                    UINT64 transformOffsetInBytes, D3D12_GPU_VIRTUAL_ADDRESS ommArray,
                                                    D3D12_GPU_VIRTUAL_ADDRESS ommIndexBuffer, uint32_t ommIndexCount) {

    auto storage = std::make_unique<OmmLinkageStorage>();

    auto& tri = storage->triangles;
    tri = {};
    tri.VertexBuffer.StartAddress = vertexBuffer->GetGPUVirtualAddress() + vertexOffsetInBytes;
    tri.VertexBuffer.StrideInBytes = vertexSizeInBytes;
    tri.VertexCount = vertexCount;
    tri.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
    tri.IndexBuffer = indexBuffer ? (indexBuffer->GetGPUVirtualAddress() + indexOffsetInBytes) : 0;
    tri.IndexFormat = indexBuffer ? DXGI_FORMAT_R32_UINT : DXGI_FORMAT_UNKNOWN;
    tri.IndexCount = indexCount;
    tri.Transform3x4 = transformBuffer ? (transformBuffer->GetGPUVirtualAddress() + transformOffsetInBytes) : 0;

    auto& link = storage->linkage;
    link = {};
    link.OpacityMicromapIndexBuffer.StartAddress = ommIndexBuffer;
    link.OpacityMicromapIndexBuffer.StrideInBytes = sizeof(int32_t);
    link.OpacityMicromapIndexFormat = DXGI_FORMAT_R32_UINT;
    link.OpacityMicromapBaseLocation = 0;
    link.OpacityMicromapArray = ommArray;

    D3D12_RAYTRACING_GEOMETRY_DESC descriptor = {};
    descriptor.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_OMM_TRIANGLES;
    descriptor.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_NONE;
    descriptor.OmmTriangles.pTriangles = &storage->triangles;
    descriptor.OmmTriangles.pOmmLinkage = &storage->linkage;

    m_ommStorage.push_back(std::move(storage));
    m_vertexBuffers.push_back(descriptor);
}

void BottomLevelASGenerator::ComputeASBufferSizes(ID3D12Device5* device,
                                                  D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS buildFlags,
                                                  UINT64* scratchSizeInBytes, UINT64* resultSizeInBytes) {

    m_flags = buildFlags;

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS prebuildDesc;
    prebuildDesc.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
    prebuildDesc.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    prebuildDesc.NumDescs = static_cast<UINT>(m_vertexBuffers.size());
    prebuildDesc.pGeometryDescs = m_vertexBuffers.data();
    prebuildDesc.Flags = m_flags;

    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO info = {};
    device->GetRaytracingAccelerationStructurePrebuildInfo(&prebuildDesc, &info);

    *scratchSizeInBytes = ROUND_UP(info.ScratchDataSizeInBytes, D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);
    *resultSizeInBytes = ROUND_UP(info.ResultDataMaxSizeInBytes, D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);

    m_scratchSizeInBytes = *scratchSizeInBytes;
    m_resultSizeInBytes = *resultSizeInBytes;
}

void BottomLevelASGenerator::Generate(ID3D12GraphicsCommandList4* commandList, ID3D12Resource* scratchBuffer,

                                      ID3D12Resource* resultBuffer, bool updateOnly,

                                      ID3D12Resource* previousResult

) {

    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS flags = m_flags;

    if (flags == D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE && updateOnly) {
        flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE;
    }

    if (m_flags != D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE && updateOnly) {
        throw std::logic_error("Cannot update a bottom-level AS not originally built for updates");
    }
    if (updateOnly && previousResult == nullptr) {
        throw std::logic_error("Bottom-level hierarchy update requires the previous hierarchy");
    }

    if (m_resultSizeInBytes == 0 || m_scratchSizeInBytes == 0) {
        throw std::logic_error("Invalid scratch and result buffer sizes - ComputeASBufferSizes needs "
                               "to be called before Build");
    }

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC buildDesc;
    buildDesc.Inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
    buildDesc.Inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    buildDesc.Inputs.NumDescs = static_cast<UINT>(m_vertexBuffers.size());
    buildDesc.Inputs.pGeometryDescs = m_vertexBuffers.data();
    buildDesc.DestAccelerationStructureData = {resultBuffer->GetGPUVirtualAddress()};
    buildDesc.ScratchAccelerationStructureData = {scratchBuffer->GetGPUVirtualAddress()};
    buildDesc.SourceAccelerationStructureData = previousResult ? previousResult->GetGPUVirtualAddress() : 0;
    buildDesc.Inputs.Flags = flags;

    commandList->BuildRaytracingAccelerationStructure(&buildDesc, 0, nullptr);

    D3D12_RESOURCE_BARRIER uavBarrier;
    uavBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    uavBarrier.UAV.pResource = resultBuffer;
    uavBarrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    commandList->ResourceBarrier(1, &uavBarrier);
}
} // namespace nv_helpers_dx12
