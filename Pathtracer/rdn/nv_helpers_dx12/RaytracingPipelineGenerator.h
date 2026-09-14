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

#include <dxcapi.h>
#include <wrl/client.h>

#include <string>
#include <vector>

namespace nv_helpers_dx12 {

class RayTracingPipelineGenerator {
  public:
    RayTracingPipelineGenerator(ID3D12Device5* device);

    void AddLibrary(IDxcBlob* dxilLibrary, const std::vector<std::wstring>& symbolExports);

    void AddHitGroup(const std::wstring& hitGroupName, const std::wstring& closestHitSymbol,
                     const std::wstring& anyHitSymbol = L"", const std::wstring& intersectionSymbol = L"");

    void AddRootSignatureAssociation(ID3D12RootSignature* rootSignature, const std::vector<std::wstring>& symbols);

    void SetMaxPayloadSize(UINT sizeInBytes);

    void SetMaxAttributeSize(UINT sizeInBytes);

    void SetMaxRecursionDepth(UINT maxDepth);

    void SetPipelineFlags(D3D12_RAYTRACING_PIPELINE_FLAGS flags);

    Microsoft::WRL::ComPtr<ID3D12StateObject> Generate();

    void SetGlobalRootSignature(ID3D12RootSignature* rootSig);

  private:
    struct Library {
        Library(IDxcBlob* dxil, const std::vector<std::wstring>& exportedSymbols);

        Library(const Library& source);

        Microsoft::WRL::ComPtr<IDxcBlob> m_dxil;
        const std::vector<std::wstring> m_exportedSymbols;

        std::vector<D3D12_EXPORT_DESC> m_exports;
        D3D12_DXIL_LIBRARY_DESC m_libDesc;
    };

    struct HitGroup {
        HitGroup(std::wstring hitGroupName, std::wstring closestHitSymbol, std::wstring anyHitSymbol = L"",
                 std::wstring intersectionSymbol = L"");

        HitGroup(const HitGroup& source);

        std::wstring m_hitGroupName;
        std::wstring m_closestHitSymbol;
        std::wstring m_anyHitSymbol;
        std::wstring m_intersectionSymbol;
        D3D12_HIT_GROUP_DESC m_desc = {};
    };

    struct RootSignatureAssociation {
        RootSignatureAssociation(ID3D12RootSignature* rootSignature, const std::vector<std::wstring>& symbols);

        RootSignatureAssociation(const RootSignatureAssociation& source);

        ID3D12RootSignature* m_rootSignature;
        ID3D12RootSignature* m_rootSignaturePointer;
        std::vector<std::wstring> m_symbols;
        std::vector<LPCWSTR> m_symbolPointers;
        D3D12_SUBOBJECT_TO_EXPORTS_ASSOCIATION m_association = {};
    };

    void CreateDummyRootSignatures();

    void BuildShaderExportList(std::vector<std::wstring>& exportedSymbols);

    std::vector<Library> m_libraries = {};
    std::vector<HitGroup> m_hitGroups = {};
    std::vector<RootSignatureAssociation> m_rootSignatureAssociations = {};

    UINT m_maxPayLoadSizeInBytes = 0;

    UINT m_maxAttributeSizeInBytes = 2 * sizeof(float);

    UINT m_maxRecursionDepth = 1;

    ID3D12Device5* m_device;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> m_dummyLocalRootSignature;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> m_dummyGlobalRootSignature;

    ID3D12RootSignature* m_globalRootSignature;

    D3D12_RAYTRACING_PIPELINE_FLAGS m_pipelineFlags = D3D12_RAYTRACING_PIPELINE_FLAG_NONE;
};

} // namespace nv_helpers_dx12
