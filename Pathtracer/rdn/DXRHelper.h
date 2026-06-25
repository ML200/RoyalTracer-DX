/******************************************************************************
 * Copyright 1998-2018 NVIDIA Corp. All Rights Reserved.
 *****************************************************************************/

#pragma once

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iostream>
#include <chrono>

#include <d3d12.h>
#include <dxcapi.h>
#include <wrl/client.h> // For Microsoft::WRL::ComPtr

#include "DXSampleHelper.h" // Assuming this contains ThrowIfFailed/ThrowIfFailed etc.

using Microsoft::WRL::ComPtr;

namespace nv_helpers_dx12
{

//--------------------------------------------------------------------------------------------------
// Buffer Creation Helper
//--------------------------------------------------------------------------------------------------
inline ID3D12Resource* CreateBuffer(ID3D12Device* m_device, uint64_t size,
                                    D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES initState,
                                    const D3D12_HEAP_PROPERTIES& heapProps)
{
    //D3D12 rejects a zero-width buffer: the call fails and this helper would
    //return null, crashing the caller on the first dereference. Empty
    //geometry buffers can legitimately request size 0 — e.g. the global
    //vertex/index buffers of a scene with no triangle meshes. Round up to a
    //minimal valid allocation that such callers simply never read.
    if (size == 0) size = 256;

    D3D12_RESOURCE_DESC bufDesc = {};
    bufDesc.Alignment = 0;
    bufDesc.DepthOrArraySize = 1;
    bufDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    bufDesc.Flags = flags;
    bufDesc.Format = DXGI_FORMAT_UNKNOWN;
    bufDesc.Height = 1;
    bufDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    bufDesc.MipLevels = 1;
    bufDesc.SampleDesc.Count = 1;
    bufDesc.SampleDesc.Quality = 0;
    bufDesc.Width = size;

    ID3D12Resource* pBuffer = nullptr;
    //Throw on failure instead of returning null. A silent null return just
    //defers the crash to the caller's first dereference, with no HRESULT and
    //no call site — exactly the un-debuggable failure this used to produce.
    //The size==0 guard above already covers the one legitimate empty-buffer
    //case; anything else failing here is a real error and should be loud.
    ThrowIfFailed(m_device->CreateCommittedResource(
        &heapProps, D3D12_HEAP_FLAG_NONE, &bufDesc, initState, nullptr,
        IID_PPV_ARGS(&pBuffer)));
    return pBuffer;
}


#ifndef ROUND_UP
#define ROUND_UP(v, powerOf2Alignment) (((v) + (powerOf2Alignment)-1) & ~((powerOf2Alignment)-1))
#endif

// Specifies a heap used for uploading. This heap type has CPU access optimized
// for uploading to the GPU.
static const D3D12_HEAP_PROPERTIES kUploadHeapProps = {
    D3D12_HEAP_TYPE_UPLOAD, D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_MEMORY_POOL_UNKNOWN, 0, 0};

// Specifies the default heap. This heap type experiences the most bandwidth for
// the GPU, but cannot provide CPU access.
static const D3D12_HEAP_PROPERTIES kDefaultHeapProps = {
    D3D12_HEAP_TYPE_DEFAULT, D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_MEMORY_POOL_UNKNOWN, 0, 0};

static const CD3DX12_HEAP_PROPERTIES kReadbackHeapProps(D3D12_HEAP_TYPE_READBACK);

//--------------------------------------------------------------------------------------------------
// Internal Helper: Compile using IDxcCompiler3 (Modern Interface)
// This is required for SM 6.x signing and validation to work correctly.
//--------------------------------------------------------------------------------------------------
inline IDxcBlob* CompileShaderNew(LPCWSTR fileName, LPCWSTR entryPoint, LPCWSTR targetProfile)
{
    // 1. Initialize DXC Compiler and Utils
    static IDxcCompiler3* pCompiler3 = nullptr;
    static IDxcUtils* pUtils = nullptr;
    static IDxcIncludeHandler* pIncludeHandler = nullptr;

    if (!pCompiler3)
    {
        ThrowIfFailed(DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(&pCompiler3)));
        ThrowIfFailed(DxcCreateInstance(CLSID_DxcUtils, IID_PPV_ARGS(&pUtils)));
        ThrowIfFailed(pUtils->CreateDefaultIncludeHandler(&pIncludeHandler));
    }

    // 2. Load the Shader Source File
    ComPtr<IDxcBlobEncoding> pSourceBlob;
    HRESULT hr = pUtils->LoadFile(fileName, nullptr, &pSourceBlob);
    if (FAILED(hr))
    {
        std::wstring wFileName(fileName);
        std::string sFileName(wFileName.begin(), wFileName.end());
        throw std::runtime_error("Failed to load shader file: " + sFileName);
    }

    DxcBuffer sourceBuffer;
    sourceBuffer.Ptr = pSourceBlob->GetBufferPointer();
    sourceBuffer.Size = pSourceBlob->GetBufferSize();
    sourceBuffer.Encoding = DXC_CP_ACP;

    // 3. Configure Arguments
    std::vector<LPCWSTR> args;

    // File and Entry Point
    args.push_back(fileName);
    args.push_back(L"-E");
    args.push_back(entryPoint);
    args.push_back(L"-T");
    args.push_back(targetProfile);

    // Debug and Optimization flags
    args.push_back(L"-Zi");               // Generate debug info
    args.push_back(L"-Qembed_debug");     // Embed the PDB in the DXIL container.
                                          // Without this, -Zi puts debug into a
                                          // separate blob and Nsight cannot find
                                          // HLSL source for the SASS/DXIL view.
    args.push_back(L"-Zss");              // Source-hash stable across rebuilds,
                                          // lets Nsight's capture re-correlate
                                          // when only line numbers move.
    // -Qstrip_debug intentionally omitted: keeping the debug info embedded in
    // the DXIL blob lets Nsight correlate GPU work back to HLSL source lines.
    // -O3 below stays on so the profile reflects the real optimized shaders.
    args.push_back(L"-Qstrip_reflect");   // Strip reflection from the shipping blob (extractable via DXC_OUT_REFLECTION)
    args.push_back(L"-O3");               // Optimization
    args.push_back(L"-enable-16bit-types");

    // Shader Model specific defines
    args.push_back(L"-D"); args.push_back(L"MAX_REGS=96");
    args.push_back(L"-HV"); args.push_back(L"2021");

    // 4. Compile
    ComPtr<IDxcResult> pResult;
    auto compileT0 = std::chrono::high_resolution_clock::now();
    hr = pCompiler3->Compile(
        &sourceBuffer,
        args.data(), (uint32_t)args.size(),
        pIncludeHandler,
        IID_PPV_ARGS(&pResult)
    );

    // 5. Check Status
    if (SUCCEEDED(hr))
    {
        pResult->GetStatus(&hr);
    }
    auto compileMs = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - compileT0).count();

    std::wstring shaderName(fileName);
    shaderName = shaderName.substr(shaderName.find_last_of(L"/\\") + 1);

    // 6. On failure, surface the compiler diagnostics (critical) and abort. DXC
    // routes warnings + errors through DXC_OUT_ERRORS; we only pull and show that
    // text when the compile actually failed — successful compiles print timing
    // only, no log spam.
    if (FAILED(hr))
    {
        ComPtr<IDxcBlobUtf8> pErrors;
        pResult->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(&pErrors), nullptr);
        std::string errMsg = "Shader compilation failed.";
        if (pErrors && pErrors->GetStringLength() > 0)
        {
            OutputDebugStringA(pErrors->GetStringPointer());
            errMsg.assign(pErrors->GetStringPointer(), pErrors->GetStringLength());
        }
        MessageBoxA(nullptr, errMsg.c_str(), "Shader Compilation Failed", MB_OK | MB_ICONERROR);
        throw std::logic_error("Shader compilation failed.");
    }

    // Per-shader compile time (success path).
    std::wcout << L"[Shader] " << shaderName << L" compiled in "
               << compileMs << L" ms" << std::endl;

    // 7. Retrieve the Compiled Shader Object
    // Using IDxcCompiler3 ensures the blob is properly signed by dxil.dll
    IDxcBlob* pBlob = nullptr;
    ThrowIfFailed(pResult->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(&pBlob), nullptr));

    return pBlob;
}

//--------------------------------------------------------------------------------------------------
// Wrappers to maintain compatibility with your existing Renderer code
//--------------------------------------------------------------------------------------------------

// Compile a HLSL file into a DXIL library (e.g. for Ray Tracing)
inline IDxcBlob* CompileShaderLibrary(LPCWSTR fileName)
{
    return CompileShaderNew(fileName, L"", L"lib_6_9");
}

// Compile a HLSL file as a Compute Shader
inline Microsoft::WRL::ComPtr<IDxcBlob> CompileCS(LPCWSTR fileName, LPCWSTR entryPoint = L"main")
{
    return CompileShaderNew(fileName, entryPoint, L"cs_6_9");
}

// Compile a HLSL file as a Work Graph library
inline Microsoft::WRL::ComPtr<IDxcBlob> CompileWG(LPCWSTR fileName, LPCWSTR entryPoint = L"main")
{
    return CompileShaderNew(fileName, entryPoint, L"lib_6_9");
}

//--------------------------------------------------------------------------------------------------
// Descriptor Heap Helper
//--------------------------------------------------------------------------------------------------
inline ID3D12DescriptorHeap* CreateDescriptorHeap(ID3D12Device* device, uint32_t count,
                                           D3D12_DESCRIPTOR_HEAP_TYPE type, bool shaderVisible)
{
  D3D12_DESCRIPTOR_HEAP_DESC desc = {};
  desc.NumDescriptors = count;
  desc.Type = type;
  desc.Flags =
      shaderVisible ? D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE : D3D12_DESCRIPTOR_HEAP_FLAG_NONE;

  ID3D12DescriptorHeap* pHeap;
  ThrowIfFailed(device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&pHeap)));
  return pHeap;
}

//--------------------------------------------------------------------------------------------------
// Menger Sponge Helper
//--------------------------------------------------------------------------------------------------
template <class Vertex>
void GenerateMengerSponge(int32_t level, float probability, std::vector<Vertex>& outputVertices,
                          std::vector<UINT>& outputIndices)
{
  struct Cube
  {
    Cube(const DirectX::XMVECTOR& tlf, float s) : m_topLeftFront(tlf), m_size(s)
    {
    }
    DirectX::XMVECTOR m_topLeftFront;
    float m_size;

    void enqueueQuad(std::vector<Vertex>& vertices, std::vector<UINT>& indices,
                     const DirectX::XMVECTOR& bottomLeft4, const DirectX::XMVECTOR& dx, const DirectX::XMVECTOR& dy, bool flip)
    {
      UINT currentIndex = static_cast<UINT>(vertices.size());
      DirectX::XMFLOAT3 bottomLeft(bottomLeft4.m128_f32);
      DirectX::XMVECTOR normal = DirectX::XMVector3Cross(DirectX::XMVector3Normalize(dy), DirectX::XMVector3Normalize(dx));
      if (flip)
      {
        normal = -normal;

        indices.push_back(currentIndex + 0);
        indices.push_back(currentIndex + 2);
        indices.push_back(currentIndex + 1);

        indices.push_back(currentIndex + 3);
        indices.push_back(currentIndex + 1);
        indices.push_back(currentIndex + 2);
      }
      else
      {

        indices.push_back(currentIndex + 0);
        indices.push_back(currentIndex + 1);
        indices.push_back(currentIndex + 2);

        indices.push_back(currentIndex + 2);
        indices.push_back(currentIndex + 1);
        indices.push_back(currentIndex + 3);
      }

      const DirectX::XMFLOAT4 n = {normal.m128_f32[0], normal.m128_f32[1], normal.m128_f32[2], 0.f};
      vertices.push_back(
          {{bottomLeft.x, bottomLeft.y, bottomLeft.z, 1.f}, n, {1.f, 0.f, 0.f, 1.f}});
      vertices.push_back({{bottomLeft.x + dx.m128_f32[0], bottomLeft.y + dx.m128_f32[1],
                           bottomLeft.z + dx.m128_f32[2], 1.f},
                          n,
                          {0.5f, 1.f, 0.f, 1.f}});
      vertices.push_back({{bottomLeft.x + dy.m128_f32[0], bottomLeft.y + dy.m128_f32[1],
                           bottomLeft.z + dy.m128_f32[2], 1.f},
                          n,
                          {0.5f, 0.f, 1.f, 1.f}});

      vertices.push_back({{bottomLeft.x + dx.m128_f32[0] + dy.m128_f32[0],
                           bottomLeft.y + dx.m128_f32[1] + dy.m128_f32[1],
                           bottomLeft.z + dx.m128_f32[2] + dy.m128_f32[2], 1.f},
                          n,
                          {0.f, 1.f, 0.f, 1.f}});
    }
    void enqueueVertices(std::vector<Vertex>& vertices, std::vector<UINT>& indices)
    {

      DirectX::XMVECTOR current = m_topLeftFront;
      enqueueQuad(vertices, indices, current, {m_size, 0, 0}, {0, m_size, 0}, false);
      enqueueQuad(vertices, indices, current, {m_size, 0, 0}, {0, 0, m_size}, true);
      enqueueQuad(vertices, indices, current, {0, m_size, 0}, {0, 0, m_size}, false);

      current.m128_f32[0] += m_size;
      current.m128_f32[1] += m_size;
      current.m128_f32[2] += m_size;
      enqueueQuad(vertices, indices, current, {-m_size, 0, 0}, {0, -m_size, 0}, true);
      enqueueQuad(vertices, indices, current, {-m_size, 0, 0}, {0, 0, -m_size}, false);
      enqueueQuad(vertices, indices, current, {0, -m_size, 0}, {0, 0, -m_size}, true);
    }
    void split(std::vector<Cube>& cubes)
    {
      float size = m_size / 3.f;
      DirectX::XMVECTOR topLeftFront = m_topLeftFront;
      for (int x = 0; x < 3; x++)
      {
        topLeftFront.m128_f32[0] = m_topLeftFront.m128_f32[0] + static_cast<float>(x) * size;
        for (int y = 0; y < 3; y++)
        {
          if (x == 1 && y == 1)
            continue;
          topLeftFront.m128_f32[1] = m_topLeftFront.m128_f32[1] + static_cast<float>(y) * size;
          for (int z = 0; z < 3; z++)
          {
            if (x == 1 && z == 1)
              continue;
            if (y == 1 && z == 1)
              continue;

            topLeftFront.m128_f32[2] = m_topLeftFront.m128_f32[2] + static_cast<float>(z) * size;
            cubes.push_back({topLeftFront, size});
          }
        }
      }
    }

    void splitProb(std::vector<Cube>& cubes, float prob)
    {

      float size = m_size / 3.f;
      DirectX::XMVECTOR topLeftFront = m_topLeftFront;
      for (int x = 0; x < 3; x++)
      {
        topLeftFront.m128_f32[0] = m_topLeftFront.m128_f32[0] + static_cast<float>(x) * size;
        for (int y = 0; y < 3; y++)
        {
          topLeftFront.m128_f32[1] = m_topLeftFront.m128_f32[1] + static_cast<float>(y) * size;
          for (int z = 0; z < 3; z++)
          {
            float sample = rand() / static_cast<float>(RAND_MAX);
            if (sample > prob)
              continue;
            topLeftFront.m128_f32[2] = m_topLeftFront.m128_f32[2] + static_cast<float>(z) * size;
            cubes.push_back({topLeftFront, size});
          }
        }
      }
    }
  };

  DirectX::XMVECTOR orig;
  orig.m128_f32[0] = -0.5f;
  orig.m128_f32[1] = -0.5f;
  orig.m128_f32[2] = -0.5f;
  orig.m128_f32[3] = 1.f;

  Cube cube(orig, 1.f);

  std::vector<Cube> cubes1 = {cube};
  std::vector<Cube> cubes2 = {};

  auto previous = &cubes1;
  auto next = &cubes2;

  for (int i = 0; i < level; i++)
  {
    for (Cube& c : *previous)
    {
      if (probability < 0.f)
        c.split(*next);
      else
        c.splitProb(*next, 20.f / 27.f);
    }
    auto temp = previous;
    previous = next;
    next = temp;
    next->clear();
  }

  outputVertices.reserve(24 * previous->size());
  outputIndices.reserve(24 * previous->size());
  for (Cube& c : *previous)
  {
    c.enqueueVertices(outputVertices, outputIndices);
  }
}

} // namespace nv_helpers_dx12