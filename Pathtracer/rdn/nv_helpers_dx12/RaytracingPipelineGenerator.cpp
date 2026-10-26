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
#include <algorithm>
#include <cstdint>
#include <numeric>
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

static Microsoft::WRL::ComPtr<ID3D12StateObject> CreateOrThrow(ID3D12Device5* device,
                                                                const D3D12_STATE_OBJECT_DESC& desc) {
    Microsoft::WRL::ComPtr<ID3D12StateObject> stateObject;
    HRESULT hr = device->CreateStateObject(&desc, IID_PPV_ARGS(&stateObject));
    if (FAILED(hr)) {
        char buf[256];
        sprintf_s(buf, "Could not create the raytracing %s (HRESULT 0x%08X)",
                  desc.Type == D3D12_STATE_OBJECT_TYPE_COLLECTION ? "collection" : "state object", (unsigned)hr);
        throw std::logic_error(buf);
    }
    return stateObject;
}

Microsoft::WRL::ComPtr<ID3D12StateObject> RayTracingPipelineGenerator::Generate() {
#ifdef _DEBUG
    CheckAssociationSymbols();
#endif
    std::vector<size_t> libraries(m_libraries.size()), hitGroups(m_hitGroups.size());
    std::iota(libraries.begin(), libraries.end(), size_t(0));
    std::iota(hitGroups.begin(), hitGroups.end(), size_t(0));
    return CreateStateObject(D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE, libraries, hitGroups);
}

size_t RayTracingPipelineGenerator::PrepareCollections() {
    // Collections drop association symbols defined elsewhere, so a misspelled one would otherwise go unnoticed.
    CheckAssociationSymbols();

    // A hit group compiles with the libraries of its shaders. Groups sharing a library share a collection;
    // unrelated ones (e.g. two closest-hit variants) compile side by side.
    auto libraryOf = [&](const std::wstring& symbol) -> size_t {
        for (size_t i = 0; i < m_libraries.size(); ++i) {
            const std::vector<std::wstring>& exports = m_libraries[i].m_exportedSymbols;
            if (std::find(exports.begin(), exports.end(), symbol) != exports.end())
                return i;
        }
        return SIZE_MAX;
    };
    std::vector<size_t> parent(m_libraries.size());
    std::iota(parent.begin(), parent.end(), size_t(0));
    auto root = [&](size_t i) -> size_t {
        while (parent[i] != i)
            i = parent[i] = parent[parent[i]];
        return i;
    };
    std::vector<bool> hasHitShader(m_libraries.size(), false);
    std::vector<size_t> groupLibrary(m_hitGroups.size(), SIZE_MAX);
    for (size_t g = 0; g < m_hitGroups.size(); ++g) {
        const HitGroup& group = m_hitGroups[g];
        for (const std::wstring* symbol :
             {&group.m_closestHitSymbol, &group.m_anyHitSymbol, &group.m_intersectionSymbol}) {
            const size_t lib = symbol->empty() ? SIZE_MAX : libraryOf(*symbol);
            if (lib == SIZE_MAX)
                continue;
            hasHitShader[lib] = true;
            if (groupLibrary[g] == SIZE_MAX)
                groupLibrary[g] = lib;
            else
                parent[root(lib)] = root(groupLibrary[g]);
        }
    }

    m_collections.clear();
    std::vector<size_t> collectionOfRoot(m_libraries.size(), SIZE_MAX);
    for (size_t i = 0; i < m_libraries.size(); ++i) {
        if (!hasHitShader[i]) {
            m_collections.push_back({{i}, {}, nullptr});
            continue;
        }
        size_t& c = collectionOfRoot[root(i)];
        if (c == SIZE_MAX) {
            c = m_collections.size();
            m_collections.emplace_back();
        }
        m_collections[c].libraries.push_back(i);
    }
    for (size_t g = 0; g < m_hitGroups.size(); ++g) {
        // A group without a known shader gets its own collection, whose creation reports the missing symbol.
        if (groupLibrary[g] == SIZE_MAX)
            m_collections.push_back({{}, {g}, nullptr});
        else
            m_collections[collectionOfRoot[root(groupLibrary[g])]].hitGroups.push_back(g);
    }
    return m_collections.size();
}

void RayTracingPipelineGenerator::CreateCollection(size_t index) {
    Collection& collection = m_collections[index];
    collection.stateObject =
        CreateStateObject(D3D12_STATE_OBJECT_TYPE_COLLECTION, collection.libraries, collection.hitGroups);
}

std::wstring RayTracingPipelineGenerator::CollectionName(size_t index) const {
    std::wstring name;
    for (size_t i : m_collections[index].libraries)
        for (const std::wstring& e : m_libraries[i].m_exportedSymbols)
            name += (name.empty() ? L"" : L"+") + e;
    return name;
}

Microsoft::WRL::ComPtr<ID3D12StateObject> RayTracingPipelineGenerator::LinkCollections() {
    std::vector<D3D12_EXISTING_COLLECTION_DESC> collections;
    collections.reserve(m_collections.size());
    for (const Collection& collection : m_collections) {
        if (!collection.stateObject)
            throw std::logic_error("Every collection must be created before linking");
        collections.push_back({collection.stateObject.Get(), 0, nullptr});
    }

    std::vector<D3D12_STATE_SUBOBJECT> subobjects;
    for (const D3D12_EXISTING_COLLECTION_DESC& collection : collections)
        subobjects.push_back({D3D12_STATE_SUBOBJECT_TYPE_EXISTING_COLLECTION, &collection});

    // The same configuration every collection carries; all of them must match.
    const D3D12_RAYTRACING_SHADER_CONFIG shaderConfig = {m_maxPayLoadSizeInBytes, m_maxAttributeSizeInBytes};
    subobjects.push_back({D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_SHADER_CONFIG, &shaderConfig});
    ID3D12RootSignature* globalRootSignature =
        m_globalRootSignature ? m_globalRootSignature : m_dummyGlobalRootSignature.Get();
    subobjects.push_back({D3D12_STATE_SUBOBJECT_TYPE_GLOBAL_ROOT_SIGNATURE, &globalRootSignature});
    const D3D12_RAYTRACING_PIPELINE_CONFIG1 pipelineConfig = {m_maxRecursionDepth, m_pipelineFlags};
    subobjects.push_back({D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_PIPELINE_CONFIG1, &pipelineConfig});

    const D3D12_STATE_OBJECT_DESC desc = {D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE, (UINT)subobjects.size(),
                                          subobjects.data()};
    return CreateOrThrow(m_device, desc);
}

Microsoft::WRL::ComPtr<ID3D12StateObject>
RayTracingPipelineGenerator::CreateStateObject(D3D12_STATE_OBJECT_TYPE type, const std::vector<size_t>& libraries,
                                               const std::vector<size_t>& hitGroups) const {
    UINT64 subobjectCount =
        libraries.size() + hitGroups.size() + 1 + 1 + 2 * m_rootSignatureAssociations.size() + 2 + 1;

    // Associations store element pointers, so this allocation must remain stable.
    std::vector<D3D12_STATE_SUBOBJECT> subobjects(subobjectCount);

    UINT currentIndex = 0;

    for (size_t i : libraries) {
        D3D12_STATE_SUBOBJECT libSubobject = {};
        libSubobject.Type = D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY;
        libSubobject.pDesc = &m_libraries[i].m_libDesc;

        subobjects[currentIndex++] = libSubobject;
    }

    for (size_t i : hitGroups) {
        D3D12_STATE_SUBOBJECT hitGroup = {};
        hitGroup.Type = D3D12_STATE_SUBOBJECT_TYPE_HIT_GROUP;
        hitGroup.pDesc = &m_hitGroups[i].m_desc;

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
    BuildShaderExportList(libraries, hitGroups, exportedSymbols);

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

    // A collection may only associate its own symbols; a pipeline keeps every symbol so typos fail creation.
    std::unordered_set<std::wstring> localSymbols;
    for (size_t i : libraries)
        localSymbols.insert(m_libraries[i].m_exportedSymbols.begin(), m_libraries[i].m_exportedSymbols.end());
    for (size_t i : hitGroups)
        localSymbols.insert(m_hitGroups[i].m_hitGroupName);

    // Reserved: the subobjects point into both.
    std::vector<std::vector<LPCWSTR>> associationSymbols;
    std::vector<D3D12_SUBOBJECT_TO_EXPORTS_ASSOCIATION> associations;
    associationSymbols.reserve(m_rootSignatureAssociations.size());
    associations.reserve(m_rootSignatureAssociations.size());

    for (const RootSignatureAssociation& assoc : m_rootSignatureAssociations) {
        std::vector<LPCWSTR>& symbols = associationSymbols.emplace_back();
        for (size_t s = 0; s < assoc.m_symbols.size(); ++s)
            if (type != D3D12_STATE_OBJECT_TYPE_COLLECTION || localSymbols.count(assoc.m_symbols[s]))
                symbols.push_back(assoc.m_symbolPointers[s]);
        if (symbols.empty())
            continue;

        D3D12_STATE_SUBOBJECT rootSigObject = {};
        rootSigObject.Type = D3D12_STATE_SUBOBJECT_TYPE_LOCAL_ROOT_SIGNATURE;
        rootSigObject.pDesc = &assoc.m_rootSignature;

        subobjects[currentIndex++] = rootSigObject;

        D3D12_SUBOBJECT_TO_EXPORTS_ASSOCIATION& association = associations.emplace_back();
        association.NumExports = static_cast<UINT>(symbols.size());
        association.pExports = symbols.data();
        association.pSubobjectToAssociate = &subobjects[(currentIndex - 1)];

        D3D12_STATE_SUBOBJECT rootSigAssociationObject = {};
        rootSigAssociationObject.Type = D3D12_STATE_SUBOBJECT_TYPE_SUBOBJECT_TO_EXPORTS_ASSOCIATION;
        rootSigAssociationObject.pDesc = &association;

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
    pipelineDesc.Type = type;
    pipelineDesc.NumSubobjects = currentIndex;
    pipelineDesc.pSubobjects = subobjects.data();

    return CreateOrThrow(m_device, pipelineDesc);
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

void RayTracingPipelineGenerator::BuildShaderExportList(const std::vector<size_t>& libraries,
                                                        const std::vector<size_t>& hitGroups,
                                                        std::vector<std::wstring>& exportedSymbols) const {
    std::unordered_set<std::wstring> exports;

    for (size_t i : libraries) {
        for (const auto& exportName : m_libraries[i].m_exportedSymbols) {
#ifdef _DEBUG
            if (exports.find(exportName) != exports.end()) {
                throw std::logic_error("Multiple definition of a symbol in the imported DXIL libraries");
            }
#endif
            exports.insert(exportName);
        }
    }

#ifdef _DEBUG
    for (size_t i : hitGroups) {
        const HitGroup& hitGroup = m_hitGroups[i];
        if (!hitGroup.m_anyHitSymbol.empty() && exports.find(hitGroup.m_anyHitSymbol) == exports.end()) {
            throw std::logic_error("Any hit symbol not found in the imported DXIL libraries");
        }

        if (!hitGroup.m_closestHitSymbol.empty() && exports.find(hitGroup.m_closestHitSymbol) == exports.end()) {
            throw std::logic_error("Closest hit symbol not found in the imported DXIL libraries");
        }

        if (!hitGroup.m_intersectionSymbol.empty() && exports.find(hitGroup.m_intersectionSymbol) == exports.end()) {
            throw std::logic_error("Intersection symbol not found in the imported DXIL libraries");
        }
    }
#endif

    for (size_t i : hitGroups) {
        const HitGroup& hitGroup = m_hitGroups[i];
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

void RayTracingPipelineGenerator::CheckAssociationSymbols() const {
    std::unordered_set<std::wstring> all_exports;
    for (const Library& lib : m_libraries)
        all_exports.insert(lib.m_exportedSymbols.begin(), lib.m_exportedSymbols.end());
    for (const HitGroup& hitGroup : m_hitGroups)
        all_exports.insert(hitGroup.m_hitGroupName);

    for (const auto& assoc : m_rootSignatureAssociations) {
        for (const auto& symb : assoc.m_symbols) {
            if (!symb.empty() && all_exports.find(symb) == all_exports.end()) {
                throw std::logic_error("Root association symbol not found in the "
                                       "imported DXIL libraries and hit group names");
            }
        }
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
