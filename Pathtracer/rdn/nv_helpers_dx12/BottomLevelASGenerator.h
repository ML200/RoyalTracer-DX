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

#pragma once

#include "d3d12.h"

#include <vector>

namespace nv_helpers_dx12 {

class BottomLevelASGenerator {
  public:
    void AddVertexBuffer(ID3D12Resource* vertexBuffer,

                         UINT64 vertexOffsetInBytes,

                         uint32_t vertexCount,

                         UINT vertexSizeInBytes,

                         ID3D12Resource* transformBuffer,

                         UINT64 transformOffsetInBytes,

                         bool isOpaque = true

    );

    void AddVertexBuffer(ID3D12Resource* vertexBuffer,

                         UINT64 vertexOffsetInBytes,

                         uint32_t vertexCount,

                         UINT vertexSizeInBytes,

                         ID3D12Resource* indexBuffer,

                         UINT64 indexOffsetInBytes,

                         uint32_t indexCount, ID3D12Resource* transformBuffer,

                         UINT64 transformOffsetInBytes,

                         bool isOpaque = true

    );

    void AddVertexBufferWithOMM(ID3D12Resource* vertexBuffer, UINT64 vertexOffsetInBytes, uint32_t vertexCount,
                                UINT vertexSizeInBytes, ID3D12Resource* indexBuffer, UINT64 indexOffsetInBytes,
                                uint32_t indexCount, ID3D12Resource* transformBuffer, UINT64 transformOffsetInBytes,
                                D3D12_GPU_VIRTUAL_ADDRESS ommArray, D3D12_GPU_VIRTUAL_ADDRESS ommIndexBuffer,
                                uint32_t ommIndexCount);

    void ComputeASBufferSizes(ID3D12Device5* device,

                              D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS buildFlags,
                              UINT64* scratchSizeInBytes,

                              UINT64* resultSizeInBytes

    );

    void Generate(ID3D12GraphicsCommandList4* commandList, ID3D12Resource* scratchBuffer,

                  ID3D12Resource* resultBuffer, bool updateOnly = false, ID3D12Resource* previousResult = nullptr

    );

  private:
    std::vector<D3D12_RAYTRACING_GEOMETRY_DESC> m_vertexBuffers = {};

    struct OmmLinkageStorage {
        D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC triangles;
        D3D12_RAYTRACING_GEOMETRY_OMM_LINKAGE_DESC linkage;
    };
    std::vector<std::unique_ptr<OmmLinkageStorage>> m_ommStorage;

    UINT64 m_scratchSizeInBytes = 0;

    UINT64 m_resultSizeInBytes = 0;

    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS m_flags;
};
} // namespace nv_helpers_dx12
