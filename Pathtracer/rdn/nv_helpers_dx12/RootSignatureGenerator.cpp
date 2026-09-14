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
#include "RootSignatureGenerator.h"

namespace nv_helpers_dx12 {

void RootSignatureGenerator::AddHeapRangesParameter(const std::vector<D3D12_DESCRIPTOR_RANGE>& ranges) {
    m_ranges.push_back(ranges);

    D3D12_ROOT_PARAMETER param = {};
    param.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    param.DescriptorTable.NumDescriptorRanges = static_cast<UINT>(ranges.size());

    param.DescriptorTable.pDescriptorRanges = nullptr;

    m_parameters.push_back(param);

    m_rangeLocations.push_back(static_cast<UINT>(m_ranges.size() - 1));
}

void RootSignatureGenerator::AddHeapRangesParameter(
    std::vector<std::tuple<UINT, UINT, UINT, D3D12_DESCRIPTOR_RANGE_TYPE, UINT>> ranges) {

    std::vector<D3D12_DESCRIPTOR_RANGE> rangeStorage;
    for (const auto& input : ranges) {
        D3D12_DESCRIPTOR_RANGE r = {};
        r.BaseShaderRegister = std::get<RSC_BASE_SHADER_REGISTER>(input);
        r.NumDescriptors = std::get<RSC_NUM_DESCRIPTORS>(input);
        r.RegisterSpace = std::get<RSC_REGISTER_SPACE>(input);
        r.RangeType = std::get<RSC_RANGE_TYPE>(input);
        r.OffsetInDescriptorsFromTableStart = std::get<RSC_OFFSET_IN_DESCRIPTORS_FROM_TABLE_START>(input);
        rangeStorage.push_back(r);
    }

    AddHeapRangesParameter(rangeStorage);
}

void RootSignatureGenerator::AddRootParameter(D3D12_ROOT_PARAMETER_TYPE type, UINT shaderRegister, UINT registerSpace,
                                              UINT numRootConstants) {
    D3D12_ROOT_PARAMETER param = {};
    param.ParameterType = type;

    if (type == D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS) {
        param.Constants.Num32BitValues = numRootConstants;
        param.Constants.RegisterSpace = registerSpace;
        param.Constants.ShaderRegister = shaderRegister;
    } else {
        param.Descriptor.RegisterSpace = registerSpace;
        param.Descriptor.ShaderRegister = shaderRegister;
    }

    param.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    m_parameters.push_back(param);

    m_rangeLocations.push_back(~0u);
}

Microsoft::WRL::ComPtr<ID3D12RootSignature> RootSignatureGenerator::Generate(ID3D12Device* device, bool isLocal) {

    for (size_t i = 0; i < m_parameters.size(); i++) {
        if (m_parameters[i].ParameterType == D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE) {
            m_parameters[i].DescriptorTable.pDescriptorRanges = m_ranges[m_rangeLocations[i]].data();
        }
    }

    D3D12_ROOT_SIGNATURE_DESC rootDesc = {};
    rootDesc.NumParameters = static_cast<UINT>(m_parameters.size());
    rootDesc.pParameters = m_parameters.data();

    rootDesc.Flags = isLocal ? D3D12_ROOT_SIGNATURE_FLAG_LOCAL_ROOT_SIGNATURE : D3D12_ROOT_SIGNATURE_FLAG_NONE;

    Microsoft::WRL::ComPtr<ID3DBlob> pSigBlob;
    Microsoft::WRL::ComPtr<ID3DBlob> pErrorBlob;
    HRESULT hr = D3D12SerializeRootSignature(&rootDesc, D3D_ROOT_SIGNATURE_VERSION_1_0, &pSigBlob, &pErrorBlob);
    if (FAILED(hr)) {
        throw std::logic_error("Cannot serialize root signature");
    }
    Microsoft::WRL::ComPtr<ID3D12RootSignature> pRootSig;
    hr = device->CreateRootSignature(0, pSigBlob->GetBufferPointer(), pSigBlob->GetBufferSize(),
                                     IID_PPV_ARGS(&pRootSig));
    if (FAILED(hr)) {
        throw std::logic_error("Cannot create root signature");
    }
    return pRootSig;
}

} // namespace nv_helpers_dx12
