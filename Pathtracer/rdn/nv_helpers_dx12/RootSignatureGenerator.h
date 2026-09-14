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

#include <wrl/client.h>

#include "d3d12.h"

#include <tuple>
#include <vector>

namespace nv_helpers_dx12 {

class RootSignatureGenerator {
  public:
    void AddHeapRangesParameter(const std::vector<D3D12_DESCRIPTOR_RANGE>& ranges);

    void AddHeapRangesParameter(std::vector<std::tuple<UINT, UINT, UINT, D3D12_DESCRIPTOR_RANGE_TYPE, UINT>> ranges);

    void AddRootParameter(D3D12_ROOT_PARAMETER_TYPE type, UINT shaderRegister = 0, UINT registerSpace = 0,
                          UINT numRootConstants = 1);

    Microsoft::WRL::ComPtr<ID3D12RootSignature> Generate(ID3D12Device* device, bool isLocal);

  private:
    std::vector<std::vector<D3D12_DESCRIPTOR_RANGE>> m_ranges;

    std::vector<D3D12_ROOT_PARAMETER> m_parameters;

    std::vector<UINT> m_rangeLocations;

    enum {
        RSC_BASE_SHADER_REGISTER = 0,
        RSC_NUM_DESCRIPTORS = 1,
        RSC_REGISTER_SPACE = 2,
        RSC_RANGE_TYPE = 3,
        RSC_OFFSET_IN_DESCRIPTORS_FROM_TABLE_START = 4
    };
};
} // namespace nv_helpers_dx12
