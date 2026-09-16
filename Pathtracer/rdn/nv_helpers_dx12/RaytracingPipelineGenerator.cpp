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

#include "RaytracingPipelineGenerator.h"

#include "dxcapi.h"
#include <unordered_set>
#include <stdexcept>

namespace nv_helpers_dx12 {

RayTracingPipelineGenerator::RayTracingPipelineGenerator(ID3D12Device5* device)
    : m_device(device), m_globalRootSignature(nullptr) {

    CreateDummyRootSignatures();
}

void RayTracingPipelineGenerator::AddLibrary(IDxcBlob* dxilLibrary, const std::vector<std::wstring>& symbolExports) {
    m_libraries.emplace_back(dxilLibrary, symbolExports);
}

void RayTracingPipelineGenerator::AddHitGroup(const std::wstring& hitGroupName, const std::wstring& closestHitSymbol,
                                              const std::wstring& anyHitSymbol,
                                              const std::wstring& intersectionSymbol) {
    m_hitGroups.emplace_back(HitGroup(hitGroupName, closestHitSymbol, anyHitSymbol, intersectionSymbol));
}

void RayTracingPipelineGenerator::AddRootSignatureAssociation(ID3D12RootSignature* rootSignature,
                                                              const std::vector<std::wstring>& symbols) {
    m_rootSignatureAssociations.emplace_back(RootSignatureAssociation(rootSignature, symbols));
}

void RayTracingPipelineGenerator::SetMaxPayloadSize(UINT sizeInBytes) {
    m_maxPayLoadSizeInBytes = sizeInBytes;
}

void RayTracingPipelineGenerator::SetMaxAttributeSize(UINT sizeInBytes) {
    m_maxAttributeSizeInBytes = sizeInBytes;
}

void RayTracingPipelineGenerator::SetMaxRecursionDepth(UINT maxDepth) {
    m_maxRecursionDepth = maxDepth;
}

Microsoft::WRL::ComPtr<ID3D12StateObject> RayTracingPipelineGenerator::Generate() {

    UINT64 subobjectCount =
        m_libraries.size() + m_hitGroups.size() + 1 + 1 + 2 * m_rootSignatureAssociations.size() + 2 + 1;

    // Associations store element pointers, so this allocation must remain stable.
    std::vector<D3D12_STATE_SUBOBJECT> subobjects(subobjectCount);

    UINT currentIndex = 0;

    for (const Library& lib : m_libraries) {
        D3D12_STATE_SUBOBJECT libSubobject = {};
        libSubobject.Type = D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY;
        libSubobject.pDesc = &lib.m_libDesc;

        subobjects[currentIndex++] = libSubobject;
    }

    for (const HitGroup& group : m_hitGroups) {
        D3D12_STATE_SUBOBJECT hitGroup = {};
        hitGroup.Type = D3D12_STATE_SUBOBJECT_TYPE_HIT_GROUP;
        hitGroup.pDesc = &group.m_desc;

        subobjects[currentIndex++] = hitGroup;
    }

    D3D12_RAYTRACING_SHADER_CONFIG shaderDesc = {};
    shaderDesc.MaxPayloadSizeInBytes = m_maxPayLoadSizeInBytes;
    shaderDesc.MaxAttributeSizeInBytes = m_maxAttributeSizeInBytes;

    D3D12_STATE_SUBOBJECT shaderConfigObject = {};
    shaderConfigObject.Type = D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_SHADER_CONFIG;
    shaderConfigObject.pDesc = &shaderDesc;

    subobjects[currentIndex++] = shaderConfigObject;

    std::vector<std::wstring> exportedSymbols = {};
    std::vector<LPCWSTR> exportedSymbolPointers = {};
    BuildShaderExportList(exportedSymbols);

    exportedSymbolPointers.reserve(exportedSymbols.size());
    for (const auto& name : exportedSymbols) {
        exportedSymbolPointers.push_back(name.c_str());
    }
    const WCHAR** shaderExports = exportedSymbolPointers.data();

    D3D12_SUBOBJECT_TO_EXPORTS_ASSOCIATION shaderPayloadAssociation = {};
    shaderPayloadAssociation.NumExports = static_cast<UINT>(exportedSymbols.size());
    shaderPayloadAssociation.pExports = shaderExports;
    shaderPayloadAssociation.pSubobjectToAssociate = &subobjects[(currentIndex - 1)];

    D3D12_STATE_SUBOBJECT shaderPayloadAssociationObject = {};
    shaderPayloadAssociationObject.Type = D3D12_STATE_SUBOBJECT_TYPE_SUBOBJECT_TO_EXPORTS_ASSOCIATION;
    shaderPayloadAssociationObject.pDesc = &shaderPayloadAssociation;
    subobjects[currentIndex++] = shaderPayloadAssociationObject;

    for (RootSignatureAssociation& assoc : m_rootSignatureAssociations) {

        D3D12_STATE_SUBOBJECT rootSigObject = {};
        rootSigObject.Type = D3D12_STATE_SUBOBJECT_TYPE_LOCAL_ROOT_SIGNATURE;
        rootSigObject.pDesc = &assoc.m_rootSignature;

        subobjects[currentIndex++] = rootSigObject;

        assoc.m_association.NumExports = static_cast<UINT>(assoc.m_symbolPointers.size());
        assoc.m_association.pExports = assoc.m_symbolPointers.data();
        assoc.m_association.pSubobjectToAssociate = &subobjects[(currentIndex - 1)];

        D3D12_STATE_SUBOBJECT rootSigAssociationObject = {};
        rootSigAssociationObject.Type = D3D12_STATE_SUBOBJECT_TYPE_SUBOBJECT_TO_EXPORTS_ASSOCIATION;
        rootSigAssociationObject.pDesc = &assoc.m_association;

        subobjects[currentIndex++] = rootSigAssociationObject;
    }

    D3D12_STATE_SUBOBJECT globalRootSig;
    globalRootSig.Type = D3D12_STATE_SUBOBJECT_TYPE_GLOBAL_ROOT_SIGNATURE;

    ID3D12RootSignature* dgSig = m_globalRootSignature ? m_globalRootSignature : m_dummyGlobalRootSignature.Get();

    globalRootSig.pDesc = &dgSig;

    subobjects[currentIndex++] = globalRootSig;

    D3D12_STATE_SUBOBJECT dummyLocalRootSig;
    dummyLocalRootSig.Type = D3D12_STATE_SUBOBJECT_TYPE_LOCAL_ROOT_SIGNATURE;
    ID3D12RootSignature* dlSig = m_dummyLocalRootSignature.Get();
    dummyLocalRootSig.pDesc = &dlSig;
    subobjects[currentIndex++] = dummyLocalRootSig;

    D3D12_RAYTRACING_PIPELINE_CONFIG1 pipelineConfig1 = {};
    pipelineConfig1.MaxTraceRecursionDepth = m_maxRecursionDepth;
    pipelineConfig1.Flags = m_pipelineFlags;

    D3D12_STATE_SUBOBJECT pipelineConfigObject = {};
    pipelineConfigObject.Type = D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_PIPELINE_CONFIG1;
    pipelineConfigObject.pDesc = &pipelineConfig1;

    subobjects[currentIndex++] = pipelineConfigObject;

    D3D12_STATE_OBJECT_DESC pipelineDesc = {};
    pipelineDesc.Type = D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE;
    pipelineDesc.NumSubobjects = currentIndex;
    pipelineDesc.pSubobjects = subobjects.data();

    Microsoft::WRL::ComPtr<ID3D12StateObject> rtStateObject;

    HRESULT hr = m_device->CreateStateObject(&pipelineDesc, IID_PPV_ARGS(&rtStateObject));
    if (FAILED(hr)) {
        char buf[256];
        sprintf_s(buf, "Could not create the raytracing state object (HRESULT 0x%08X)", (unsigned)hr);
        throw std::logic_error(buf);
    }
    return rtStateObject;
}

void RayTracingPipelineGenerator::CreateDummyRootSignatures() {

    D3D12_ROOT_SIGNATURE_DESC rootDesc = {};
    rootDesc.NumParameters = 0;
    rootDesc.pParameters = nullptr;

    rootDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

    HRESULT hr = 0;

    Microsoft::WRL::ComPtr<ID3DBlob> serializedRootSignature;
    Microsoft::WRL::ComPtr<ID3DBlob> error;

    hr = D3D12SerializeRootSignature(&rootDesc, D3D_ROOT_SIGNATURE_VERSION_1, &serializedRootSignature, &error);
    if (FAILED(hr)) {
        throw std::logic_error("Could not serialize the global root signature");
    }
    hr = m_device->CreateRootSignature(0, serializedRootSignature->GetBufferPointer(),
                                       serializedRootSignature->GetBufferSize(),
                                       IID_PPV_ARGS(&m_dummyGlobalRootSignature));

    serializedRootSignature.Reset();
    if (FAILED(hr)) {
        throw std::logic_error("Could not create the global root signature");
    }

    rootDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_LOCAL_ROOT_SIGNATURE;
    hr = D3D12SerializeRootSignature(&rootDesc, D3D_ROOT_SIGNATURE_VERSION_1, &serializedRootSignature, &error);
    if (FAILED(hr)) {
        throw std::logic_error("Could not serialize the local root signature");
    }
    hr = m_device->CreateRootSignature(0, serializedRootSignature->GetBufferPointer(),
                                       serializedRootSignature->GetBufferSize(),
                                       IID_PPV_ARGS(&m_dummyLocalRootSignature));

    serializedRootSignature.Reset();
    if (FAILED(hr)) {
        throw std::logic_error("Could not create the local root signature");
    }
}

void RayTracingPipelineGenerator::BuildShaderExportList(std::vector<std::wstring>& exportedSymbols) {
    std::unordered_set<std::wstring> exports;

    for (const Library& lib : m_libraries) {
        for (const auto& exportName : lib.m_exportedSymbols) {
#ifdef _DEBUG
            if (exports.find(exportName) != exports.end()) {
                throw std::logic_error("Multiple definition of a symbol in the imported DXIL libraries");
            }
#endif
            exports.insert(exportName);
        }
    }

#ifdef _DEBUG
    std::unordered_set<std::wstring> all_exports = exports;

    for (const auto& hitGroup : m_hitGroups) {
        if (!hitGroup.m_anyHitSymbol.empty() && exports.find(hitGroup.m_anyHitSymbol) == exports.end()) {
            throw std::logic_error("Any hit symbol not found in the imported DXIL libraries");
        }

        if (!hitGroup.m_closestHitSymbol.empty() && exports.find(hitGroup.m_closestHitSymbol) == exports.end()) {
            throw std::logic_error("Closest hit symbol not found in the imported DXIL libraries");
        }

        if (!hitGroup.m_intersectionSymbol.empty() && exports.find(hitGroup.m_intersectionSymbol) == exports.end()) {
            throw std::logic_error("Intersection symbol not found in the imported DXIL libraries");
        }

        all_exports.insert(hitGroup.m_hitGroupName);
    }

    for (const auto& assoc : m_rootSignatureAssociations) {
        for (const auto& symb : assoc.m_symbols) {
            if (!symb.empty() && all_exports.find(symb) == all_exports.end()) {
                throw std::logic_error("Root association symbol not found in the "
                                       "imported DXIL libraries and hit group names");
            }
        }
    }
#endif

    for (const auto& hitGroup : m_hitGroups) {
        if (!hitGroup.m_anyHitSymbol.empty()) {
            exports.erase(hitGroup.m_anyHitSymbol);
        }
        if (!hitGroup.m_closestHitSymbol.empty()) {
            exports.erase(hitGroup.m_closestHitSymbol);
        }
        if (!hitGroup.m_intersectionSymbol.empty()) {
            exports.erase(hitGroup.m_intersectionSymbol);
        }
        exports.insert(hitGroup.m_hitGroupName);
    }

    for (const auto& name : exports) {
        exportedSymbols.push_back(name);
    }
}

RayTracingPipelineGenerator::Library::Library(IDxcBlob* dxil, const std::vector<std::wstring>& exportedSymbols)
    : m_dxil(dxil), m_exportedSymbols(exportedSymbols), m_exports(exportedSymbols.size()) {
    for (size_t i = 0; i < m_exportedSymbols.size(); i++) {
        m_exports[i] = {};
        m_exports[i].Name = m_exportedSymbols[i].c_str();
        m_exports[i].ExportToRename = nullptr;
        m_exports[i].Flags = D3D12_EXPORT_FLAG_NONE;
    }

    m_libDesc.DXILLibrary.BytecodeLength = dxil->GetBufferSize();
    m_libDesc.DXILLibrary.pShaderBytecode = dxil->GetBufferPointer();
    m_libDesc.NumExports = static_cast<UINT>(m_exportedSymbols.size());
    m_libDesc.pExports = m_exports.data();
}

RayTracingPipelineGenerator::Library::Library(const Library& source)
    : Library(source.m_dxil.Get(), source.m_exportedSymbols) {}

RayTracingPipelineGenerator::HitGroup::HitGroup(std::wstring hitGroupName, std::wstring closestHitSymbol,
                                                std::wstring anyHitSymbol, std::wstring intersectionSymbol)
    : m_hitGroupName(std::move(hitGroupName)), m_closestHitSymbol(std::move(closestHitSymbol)),
      m_anyHitSymbol(std::move(anyHitSymbol)), m_intersectionSymbol(std::move(intersectionSymbol)) {
    m_desc.HitGroupExport = m_hitGroupName.c_str();
    m_desc.ClosestHitShaderImport = m_closestHitSymbol.empty() ? nullptr : m_closestHitSymbol.c_str();
    m_desc.AnyHitShaderImport = m_anyHitSymbol.empty() ? nullptr : m_anyHitSymbol.c_str();
    m_desc.IntersectionShaderImport = m_intersectionSymbol.empty() ? nullptr : m_intersectionSymbol.c_str();
}

RayTracingPipelineGenerator::HitGroup::HitGroup(const HitGroup& source)
    : HitGroup(source.m_hitGroupName, source.m_closestHitSymbol, source.m_anyHitSymbol, source.m_intersectionSymbol) {}

RayTracingPipelineGenerator::RootSignatureAssociation::RootSignatureAssociation(
    ID3D12RootSignature* rootSignature, const std::vector<std::wstring>& symbols)
    : m_rootSignature(rootSignature), m_symbols(symbols), m_symbolPointers(symbols.size()) {
    for (size_t i = 0; i < m_symbols.size(); i++) {
        m_symbolPointers[i] = m_symbols[i].c_str();
    }
    m_rootSignaturePointer = m_rootSignature;
}

RayTracingPipelineGenerator::RootSignatureAssociation::RootSignatureAssociation(const RootSignatureAssociation& source)
    : RootSignatureAssociation(source.m_rootSignature, source.m_symbols) {}

void RayTracingPipelineGenerator::SetPipelineFlags(D3D12_RAYTRACING_PIPELINE_FLAGS flags) {
    m_pipelineFlags = flags;
}

void RayTracingPipelineGenerator::SetGlobalRootSignature(ID3D12RootSignature* rootSig) {
    m_globalRootSignature = rootSig;
}
} // namespace nv_helpers_dx12
