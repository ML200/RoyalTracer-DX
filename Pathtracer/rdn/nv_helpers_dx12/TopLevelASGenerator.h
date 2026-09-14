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

#include <DirectXMath.h>

#include <vector>

namespace nv_helpers_dx12 {

class TopLevelASGenerator {
  public:
    void AddInstance(ID3D12Resource* bottomLevelAS,

                     const DirectX::XMMATRIX& transform,

                     UINT instanceID,

                     UINT hitGroupIndex,

                     D3D12_RAYTRACING_INSTANCE_FLAGS flags = D3D12_RAYTRACING_INSTANCE_FLAG_NONE);

    void ComputeASBufferSizes(ID3D12Device5* device, bool allowUpdate,

                              UINT64* scratchSizeInBytes,

                              UINT64* resultSizeInBytes,

                              UINT64* descriptorsSizeInBytes

    );

    void Generate(ID3D12GraphicsCommandList4* commandList, ID3D12Resource* scratchBuffer,

                  ID3D12Resource* resultBuffer, ID3D12Resource* descriptorsBuffer,

                  bool updateOnly = false, ID3D12Resource* previousResult = nullptr

    );

    void UpdateAndRefit(ID3D12GraphicsCommandList4* commandList, ID3D12Resource* scratchBuffer,
                        ID3D12Resource* resultBuffer, ID3D12Resource* descriptorsBuffer,
                        const std::vector<uint32_t>& dirtyIndices);

    void RebuildInPlace(ID3D12GraphicsCommandList4* commandList, ID3D12Resource* scratchBuffer,
                        ID3D12Resource* resultBuffer, ID3D12Resource* descriptorsBuffer,
                        const std::vector<uint32_t>& dirtyIndices);

  private:
    struct Instance {
        Instance(ID3D12Resource* blAS, const DirectX::XMMATRIX& tr, UINT iID, UINT hgId,
                 D3D12_RAYTRACING_INSTANCE_FLAGS f);

        ID3D12Resource* bottomLevelAS;

        const DirectX::XMMATRIX& transform;

        UINT instanceID;

        UINT hitGroupIndex;

        D3D12_RAYTRACING_INSTANCE_FLAGS flags;
    };

    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS m_flags;

    std::vector<Instance> m_instances;

    UINT64 m_scratchSizeInBytes;

    UINT64 m_instanceDescsSizeInBytes;

    UINT64 m_resultSizeInBytes;
};
} // namespace nv_helpers_dx12
