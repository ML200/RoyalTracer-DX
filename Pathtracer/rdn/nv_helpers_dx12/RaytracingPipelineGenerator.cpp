/*-----------------------------------------------------------------------
Copyright (c) 2014-2018, NVIDIA. All rights reserved.
... (License Header) ...
-----------------------------------------------------------------------*/

#include "RaytracingPipelineGenerator.h"

#include "dxcapi.h"
#include <unordered_set>
#include <stdexcept>

namespace nv_helpers_dx12
{

//--------------------------------------------------------------------------------------------------
// The pipeline helper requires access to the device, as well as the
// raytracing device prior to Windows 10 RS5.
RayTracingPipelineGenerator::RayTracingPipelineGenerator(ID3D12Device5* device)
    : m_device(device), m_globalRootSignature(nullptr) // <--- Initialize to nullptr
{
  // The pipeline creation requires having at least one empty global and local root signatures, so
  // we systematically create both, as this does not incur any overhead
  CreateDummyRootSignatures();
}

// ... [AddLibrary, AddHitGroup, AddRootSignatureAssociation, SetMaxPayloadSize, SetMaxAttributeSize, SetMaxRecursionDepth implementations remain unchanged] ...
void RayTracingPipelineGenerator::AddLibrary(IDxcBlob* dxilLibrary,
                                             const std::vector<std::wstring>& symbolExports)
{
  m_libraries.emplace_back(Library(dxilLibrary, symbolExports));
}

void RayTracingPipelineGenerator::AddHitGroup(const std::wstring& hitGroupName,
                                              const std::wstring& closestHitSymbol,
                                              const std::wstring& anyHitSymbol /*= L""*/,
                                              const std::wstring& intersectionSymbol /*= L""*/)
{
  m_hitGroups.emplace_back(
      HitGroup(hitGroupName, closestHitSymbol, anyHitSymbol, intersectionSymbol));
}

void RayTracingPipelineGenerator::AddRootSignatureAssociation(
    ID3D12RootSignature* rootSignature, const std::vector<std::wstring>& symbols)
{
  m_rootSignatureAssociations.emplace_back(RootSignatureAssociation(rootSignature, symbols));
}

void RayTracingPipelineGenerator::SetMaxPayloadSize(UINT sizeInBytes)
{
  m_maxPayLoadSizeInBytes = sizeInBytes;
}

void RayTracingPipelineGenerator::SetMaxAttributeSize(UINT sizeInBytes)
{
  m_maxAttributeSizeInBytes = sizeInBytes;
}

void RayTracingPipelineGenerator::SetMaxRecursionDepth(UINT maxDepth)
{
  m_maxRecursionDepth = maxDepth;
}

//--------------------------------------------------------------------------------------------------
//
// Compiles the raytracing state object
ID3D12StateObject* RayTracingPipelineGenerator::Generate()
{
  // The pipeline is made of a set of sub-objects, representing the DXIL libraries, hit group
  // declarations, root signature associations, plus some configuration objects
  UINT64 subobjectCount =
      m_libraries.size() +                     // DXIL libraries
      m_hitGroups.size() +                     // Hit group declarations
      1 +                                      // Shader configuration
      1 +                                      // Shader payload
      2 * m_rootSignatureAssociations.size() + // Root signature declaration + association
      2 +                                      // Global and local root signatures
      1;                                       // Final pipeline subobject

  // Initialize a vector with the target object count.
  std::vector<D3D12_STATE_SUBOBJECT> subobjects(subobjectCount);

  UINT currentIndex = 0;

  // Add all the DXIL libraries
  for (const Library& lib : m_libraries)
  {
    D3D12_STATE_SUBOBJECT libSubobject = {};
    libSubobject.Type = D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY;
    libSubobject.pDesc = &lib.m_libDesc;

    subobjects[currentIndex++] = libSubobject;
  }

  // Add all the hit group declarations
  for (const HitGroup& group : m_hitGroups)
  {
    D3D12_STATE_SUBOBJECT hitGroup = {};
    hitGroup.Type = D3D12_STATE_SUBOBJECT_TYPE_HIT_GROUP;
    hitGroup.pDesc = &group.m_desc;

    subobjects[currentIndex++] = hitGroup;
  }

  // Add a subobject for the shader payload configuration
  D3D12_RAYTRACING_SHADER_CONFIG shaderDesc = {};
  shaderDesc.MaxPayloadSizeInBytes = m_maxPayLoadSizeInBytes;
  shaderDesc.MaxAttributeSizeInBytes = m_maxAttributeSizeInBytes;

  D3D12_STATE_SUBOBJECT shaderConfigObject = {};
  shaderConfigObject.Type = D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_SHADER_CONFIG;
  shaderConfigObject.pDesc = &shaderDesc;

  subobjects[currentIndex++] = shaderConfigObject;

  // Build a list of all the symbols for ray generation, miss and hit groups
  std::vector<std::wstring> exportedSymbols = {};
  std::vector<LPCWSTR> exportedSymbolPointers = {};
  BuildShaderExportList(exportedSymbols);

  // Build an array of the string pointers
  exportedSymbolPointers.reserve(exportedSymbols.size());
  for (const auto& name : exportedSymbols)
  {
    exportedSymbolPointers.push_back(name.c_str());
  }
  const WCHAR** shaderExports = exportedSymbolPointers.data();

  // Add a subobject for the association between shaders and the payload
  D3D12_SUBOBJECT_TO_EXPORTS_ASSOCIATION shaderPayloadAssociation = {};
  shaderPayloadAssociation.NumExports = static_cast<UINT>(exportedSymbols.size());
  shaderPayloadAssociation.pExports = shaderExports;
  shaderPayloadAssociation.pSubobjectToAssociate = &subobjects[(currentIndex - 1)];

  D3D12_STATE_SUBOBJECT shaderPayloadAssociationObject = {};
  shaderPayloadAssociationObject.Type = D3D12_STATE_SUBOBJECT_TYPE_SUBOBJECT_TO_EXPORTS_ASSOCIATION;
  shaderPayloadAssociationObject.pDesc = &shaderPayloadAssociation;
  subobjects[currentIndex++] = shaderPayloadAssociationObject;

  // The root signature association
  for (RootSignatureAssociation& assoc : m_rootSignatureAssociations)
  {
    // Add a subobject to declare the root signature
    D3D12_STATE_SUBOBJECT rootSigObject = {};
    rootSigObject.Type = D3D12_STATE_SUBOBJECT_TYPE_LOCAL_ROOT_SIGNATURE;
    rootSigObject.pDesc = &assoc.m_rootSignature;

    subobjects[currentIndex++] = rootSigObject;

    // Add a subobject for the association
    assoc.m_association.NumExports = static_cast<UINT>(assoc.m_symbolPointers.size());
    assoc.m_association.pExports = assoc.m_symbolPointers.data();
    assoc.m_association.pSubobjectToAssociate = &subobjects[(currentIndex - 1)];

    D3D12_STATE_SUBOBJECT rootSigAssociationObject = {};
    rootSigAssociationObject.Type = D3D12_STATE_SUBOBJECT_TYPE_SUBOBJECT_TO_EXPORTS_ASSOCIATION;
    rootSigAssociationObject.pDesc = &assoc.m_association;

    subobjects[currentIndex++] = rootSigAssociationObject;
  }

  // ----------------------------------------------------------------------------------
  // CHANGE: Select between Custom Global Root Signature and Dummy Empty Root Signature
  // ----------------------------------------------------------------------------------
  D3D12_STATE_SUBOBJECT globalRootSig;
  globalRootSig.Type = D3D12_STATE_SUBOBJECT_TYPE_GLOBAL_ROOT_SIGNATURE;

  // If the user provided a global root signature via SetGlobalRootSignature, use it.
  // Otherwise, fallback to the dummy empty one.
  ID3D12RootSignature* dgSig = m_globalRootSignature ? m_globalRootSignature : m_dummyGlobalRootSignature;

  globalRootSig.pDesc = &dgSig;

  subobjects[currentIndex++] = globalRootSig;
  // ----------------------------------------------------------------------------------

  // The pipeline construction always requires an empty local root signature
  D3D12_STATE_SUBOBJECT dummyLocalRootSig;
  dummyLocalRootSig.Type = D3D12_STATE_SUBOBJECT_TYPE_LOCAL_ROOT_SIGNATURE;
  ID3D12RootSignature* dlSig = m_dummyLocalRootSignature;
  dummyLocalRootSig.pDesc = &dlSig;
  subobjects[currentIndex++] = dummyLocalRootSig;

  // Add a subobject for the ray tracing pipeline configuration
  D3D12_RAYTRACING_PIPELINE_CONFIG pipelineConfig = {};
  pipelineConfig.MaxTraceRecursionDepth = m_maxRecursionDepth;

  D3D12_STATE_SUBOBJECT pipelineConfigObject = {};
  pipelineConfigObject.Type = D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_PIPELINE_CONFIG;
  pipelineConfigObject.pDesc = &pipelineConfig;

  subobjects[currentIndex++] = pipelineConfigObject;

  // Describe the ray tracing pipeline state object
  D3D12_STATE_OBJECT_DESC pipelineDesc = {};
  pipelineDesc.Type = D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE;
  pipelineDesc.NumSubobjects = currentIndex;
  pipelineDesc.pSubobjects = subobjects.data();

  ID3D12StateObject* rtStateObject = nullptr;

  // Create the state object
  HRESULT hr = m_device->CreateStateObject(&pipelineDesc, IID_PPV_ARGS(&rtStateObject));
  if (FAILED(hr))
  {
    char buf[256];
    sprintf_s(buf, "Could not create the raytracing state object (HRESULT 0x%08X)", (unsigned)hr);
    throw std::logic_error(buf);
  }
  return rtStateObject;
}

// ... [CreateDummyRootSignatures, BuildShaderExportList, Class Constructors, etc. remain unchanged] ...
void RayTracingPipelineGenerator::CreateDummyRootSignatures()
{
  // Creation of the global root signature
  D3D12_ROOT_SIGNATURE_DESC rootDesc = {};
  rootDesc.NumParameters = 0;
  rootDesc.pParameters = nullptr;
  // A global root signature is the default, hence this flag
  rootDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

  HRESULT hr = 0;

  ID3DBlob* serializedRootSignature;
  ID3DBlob* error;

  // Create the empty global root signature
  hr = D3D12SerializeRootSignature(&rootDesc, D3D_ROOT_SIGNATURE_VERSION_1,
                                   &serializedRootSignature, &error);
  if (FAILED(hr))
  {
    throw std::logic_error("Could not serialize the global root signature");
  }
  hr = m_device->CreateRootSignature(0, serializedRootSignature->GetBufferPointer(),
                                     serializedRootSignature->GetBufferSize(),
                                     IID_PPV_ARGS(&m_dummyGlobalRootSignature));

  serializedRootSignature->Release();
  if (FAILED(hr))
  {
    throw std::logic_error("Could not create the global root signature");
  }

  // Create the local root signature, reusing the same descriptor but altering the creation flag
  rootDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_LOCAL_ROOT_SIGNATURE;
  hr = D3D12SerializeRootSignature(&rootDesc, D3D_ROOT_SIGNATURE_VERSION_1,
                                   &serializedRootSignature, &error);
  if (FAILED(hr))
  {
    throw std::logic_error("Could not serialize the local root signature");
  }
  hr = m_device->CreateRootSignature(0, serializedRootSignature->GetBufferPointer(),
                                     serializedRootSignature->GetBufferSize(),
                                     IID_PPV_ARGS(&m_dummyLocalRootSignature));

  serializedRootSignature->Release();
  if (FAILED(hr))
  {
    throw std::logic_error("Could not create the local root signature");
  }
}

void RayTracingPipelineGenerator::BuildShaderExportList(std::vector<std::wstring>& exportedSymbols)
{
  std::unordered_set<std::wstring> exports;

  // Add all the symbols exported by the libraries
  for (const Library& lib : m_libraries)
  {
    for (const auto& exportName : lib.m_exportedSymbols)
    {
#ifdef _DEBUG
      if (exports.find(exportName) != exports.end())
      {
        throw std::logic_error("Multiple definition of a symbol in the imported DXIL libraries");
      }
#endif
      exports.insert(exportName);
    }
  }

#ifdef _DEBUG
  std::unordered_set<std::wstring> all_exports = exports;

  for (const auto& hitGroup : m_hitGroups)
  {
    if (!hitGroup.m_anyHitSymbol.empty() && exports.find(hitGroup.m_anyHitSymbol) == exports.end())
    {
      throw std::logic_error("Any hit symbol not found in the imported DXIL libraries");
    }

    if (!hitGroup.m_closestHitSymbol.empty() &&
        exports.find(hitGroup.m_closestHitSymbol) == exports.end())
    {
      throw std::logic_error("Closest hit symbol not found in the imported DXIL libraries");
    }

    if (!hitGroup.m_intersectionSymbol.empty() &&
        exports.find(hitGroup.m_intersectionSymbol) == exports.end())
    {
      throw std::logic_error("Intersection symbol not found in the imported DXIL libraries");
    }

    all_exports.insert(hitGroup.m_hitGroupName);
  }

  for (const auto& assoc : m_rootSignatureAssociations)
  {
    for (const auto& symb : assoc.m_symbols)
    {
      if (!symb.empty() && all_exports.find(symb) == all_exports.end())
      {
        throw std::logic_error("Root association symbol not found in the "
                               "imported DXIL libraries and hit group names");
      }
    }
  }
#endif

  for (const auto& hitGroup : m_hitGroups)
  {
    if (!hitGroup.m_anyHitSymbol.empty())
    {
      exports.erase(hitGroup.m_anyHitSymbol);
    }
    if (!hitGroup.m_closestHitSymbol.empty())
    {
      exports.erase(hitGroup.m_closestHitSymbol);
    }
    if (!hitGroup.m_intersectionSymbol.empty())
    {
      exports.erase(hitGroup.m_intersectionSymbol);
    }
    exports.insert(hitGroup.m_hitGroupName);
  }

  for (const auto& name : exports)
  {
    exportedSymbols.push_back(name);
  }
}

RayTracingPipelineGenerator::Library::Library(IDxcBlob* dxil,
                                              const std::vector<std::wstring>& exportedSymbols)
    : m_dxil(dxil), m_exportedSymbols(exportedSymbols), m_exports(exportedSymbols.size())
{
  for (size_t i = 0; i < m_exportedSymbols.size(); i++)
  {
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
    : Library(source.m_dxil, source.m_exportedSymbols)
{
}

RayTracingPipelineGenerator::HitGroup::HitGroup(std::wstring hitGroupName,
                                                std::wstring closestHitSymbol,
                                                std::wstring anyHitSymbol /*= L""*/,
                                                std::wstring intersectionSymbol /*= L""*/)
    : m_hitGroupName(std::move(hitGroupName)), m_closestHitSymbol(std::move(closestHitSymbol)),
      m_anyHitSymbol(std::move(anyHitSymbol)), m_intersectionSymbol(std::move(intersectionSymbol))
{
  m_desc = {};
  m_desc.HitGroupExport = m_hitGroupName.c_str();
  m_desc.Type = D3D12_HIT_GROUP_TYPE_TRIANGLES;
  m_desc.ClosestHitShaderImport = m_closestHitSymbol.empty() ? nullptr : m_closestHitSymbol.c_str();
  m_desc.AnyHitShaderImport = m_anyHitSymbol.empty() ? nullptr : m_anyHitSymbol.c_str();
  m_desc.IntersectionShaderImport =
      m_intersectionSymbol.empty() ? nullptr : m_intersectionSymbol.c_str();
}

RayTracingPipelineGenerator::HitGroup::HitGroup(const HitGroup& source)
    : HitGroup(source.m_hitGroupName, source.m_closestHitSymbol, source.m_anyHitSymbol,
               source.m_intersectionSymbol)
{
}

RayTracingPipelineGenerator::RootSignatureAssociation::RootSignatureAssociation(
    ID3D12RootSignature* rootSignature, const std::vector<std::wstring>& symbols)
    : m_rootSignature(rootSignature), m_symbols(symbols), m_symbolPointers(symbols.size())
{
  for (size_t i = 0; i < m_symbols.size(); i++)
  {
    m_symbolPointers[i] = m_symbols[i].c_str();
  }
  m_rootSignaturePointer = m_rootSignature;
}

RayTracingPipelineGenerator::RootSignatureAssociation::RootSignatureAssociation(
    const RootSignatureAssociation& source)
    : RootSignatureAssociation(source.m_rootSignature, source.m_symbols)
{
}

void RayTracingPipelineGenerator::SetGlobalRootSignature(ID3D12RootSignature* rootSig)
{
  m_globalRootSignature = rootSig;
}
} // namespace nv_helpers_dx12