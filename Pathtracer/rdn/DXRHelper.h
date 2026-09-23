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
#include <wrl/client.h>

#include "DXSampleHelper.h"

using Microsoft::WRL::ComPtr;

namespace nv_helpers_dx12 {
inline ComPtr<ID3D12Resource> CreateBuffer(ID3D12Device* m_device, uint64_t size, D3D12_RESOURCE_FLAGS flags,
                                           D3D12_RESOURCE_STATES initState, const D3D12_HEAP_PROPERTIES& heapProps) {
    if (size == 0)
        size = 256;

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

    ComPtr<ID3D12Resource> pBuffer;

    ThrowIfFailed(m_device->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &bufDesc, initState, nullptr,
                                                    IID_PPV_ARGS(&pBuffer)));
    return pBuffer;
}

#ifndef ROUND_UP
#define ROUND_UP(v, powerOf2Alignment) (((v) + (powerOf2Alignment) - 1) & ~((powerOf2Alignment) - 1))
#endif

static const D3D12_HEAP_PROPERTIES kUploadHeapProps = {D3D12_HEAP_TYPE_UPLOAD, D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
                                                       D3D12_MEMORY_POOL_UNKNOWN, 0, 0};

static const D3D12_HEAP_PROPERTIES kDefaultHeapProps = {D3D12_HEAP_TYPE_DEFAULT, D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
                                                        D3D12_MEMORY_POOL_UNKNOWN, 0, 0};

static const CD3DX12_HEAP_PROPERTIES kReadbackHeapProps(D3D12_HEAP_TYPE_READBACK);

// Compile a runtime shader variant while retaining diagnostics and blob ownership.
inline ComPtr<IDxcBlob> CompileShaderNew(LPCWSTR fileName, LPCWSTR entryPoint, LPCWSTR targetProfile,
                                         const std::vector<std::wstring>& extraDefines = {}) {
    struct CompilerContext {
        ComPtr<IDxcCompiler3> compiler;
        ComPtr<IDxcUtils> utils;
        ComPtr<IDxcIncludeHandler> includes;

        CompilerContext() {
            ThrowIfFailed(DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(&compiler)));
            ThrowIfFailed(DxcCreateInstance(CLSID_DxcUtils, IID_PPV_ARGS(&utils)));
            ThrowIfFailed(utils->CreateDefaultIncludeHandler(&includes));
        }
    };
    // Reuse DXC services across all shader variants.
    static const CompilerContext context;

    ComPtr<IDxcBlobEncoding> pSourceBlob;
    HRESULT hr = context.utils->LoadFile(fileName, nullptr, &pSourceBlob);
    if (FAILED(hr)) {
        OutputDebugStringW(fileName);
        ThrowIfFailed(hr);
    }

    DxcBuffer sourceBuffer;
    sourceBuffer.Ptr = pSourceBlob->GetBufferPointer();
    sourceBuffer.Size = pSourceBlob->GetBufferSize();
    sourceBuffer.Encoding = DXC_CP_ACP;

    std::vector<LPCWSTR> args;

    args.push_back(fileName);
    args.push_back(L"-E");
    args.push_back(entryPoint);
    args.push_back(L"-T");
    args.push_back(targetProfile);

    args.push_back(L"-Zi");
    args.push_back(L"-Qembed_debug");

    args.push_back(L"-Zss");

    args.push_back(L"-Qstrip_reflect");
    args.push_back(L"-O3");
    args.push_back(L"-enable-16bit-types");

    args.push_back(L"-D");
    args.push_back(L"MAX_REGS=96");
    args.push_back(L"-HV");
    args.push_back(L"2021");

    for (const auto& d : extraDefines) {
        args.push_back(L"-D");
        args.push_back(d.c_str());
    }

    // Defines for every shader from RDN_SHADER_DEFINES ("A=0;B=1"), so a feature can be compiled
    // out for an A/B measurement without editing the shaders.
    static const std::vector<std::wstring> envDefines = [] {
        std::vector<std::wstring> defines;
        wchar_t buffer[1024];
        const DWORD length = GetEnvironmentVariableW(L"RDN_SHADER_DEFINES", buffer, 1024);
        if (length == 0 || length >= 1024)
            return defines;
        std::wstringstream list(std::wstring(buffer, length));
        for (std::wstring define; std::getline(list, define, L';');)
            if (!define.empty())
                defines.push_back(define);
        return defines;
    }();
    for (const auto& d : envDefines) {
        args.push_back(L"-D");
        args.push_back(d.c_str());
    }

    ComPtr<IDxcResult> pResult;
    auto compileT0 = std::chrono::high_resolution_clock::now();
    hr = context.compiler->Compile(&sourceBuffer, args.data(), (uint32_t)args.size(), context.includes.Get(),
                                   IID_PPV_ARGS(&pResult));

    if (SUCCEEDED(hr)) {
        ThrowIfFailed(pResult->GetStatus(&hr));
    }
    auto compileMs =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - compileT0)
            .count();

    std::wstring shaderName(fileName);
    shaderName = shaderName.substr(shaderName.find_last_of(L"/\\") + 1);

    if (FAILED(hr)) {
        ComPtr<IDxcBlobUtf8> pErrors;
        if (pResult)
            pResult->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(&pErrors), nullptr);
        std::string errMsg = "Shader compilation failed.";
        if (pErrors && pErrors->GetStringLength() > 0) {
            OutputDebugStringA(pErrors->GetStringPointer());
            errMsg.assign(pErrors->GetStringPointer(), pErrors->GetStringLength());
        }
        MessageBoxA(nullptr, errMsg.c_str(), "Shader Compilation Failed", MB_OK | MB_ICONERROR);
        throw std::runtime_error(errMsg);
    }

    std::wcout << L"[Shader] " << shaderName << L" compiled in " << compileMs << L" ms" << std::endl;

    ComPtr<IDxcBlob> pBlob;
    ThrowIfFailed(pResult->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(&pBlob), nullptr));

    // With Aftermath GPU crash dumps on (Diagnostics.h), every shader handed to the driver is kept
    // next to the dump, so the dump's shader hashes can be resolved to the HLSL lines embedded in
    // the binaries (-Zi -Qembed_debug).
    wchar_t keepDir[MAX_PATH] = {};
    if (GetEnvironmentVariableW(L"RT_AFTERMATH_SHADER_DIR", keepDir, MAX_PATH) > 0) {
        uint32_t hash = 2166136261u;
        const auto* bytes = static_cast<const uint8_t*>(pBlob->GetBufferPointer());
        for (size_t i = 0; i < pBlob->GetBufferSize(); ++i)
            hash = (hash ^ bytes[i]) * 16777619u;
        wchar_t suffix[16];
        swprintf_s(suffix, L"-%08X.dxil", hash);
        std::ofstream keep(std::wstring(keepDir) + shaderName + L"-" + entryPoint + suffix,
                           std::ios::binary | std::ios::trunc);
        keep.write(static_cast<const char*>(pBlob->GetBufferPointer()), (std::streamsize)pBlob->GetBufferSize());
    }

    return pBlob;
}

inline ComPtr<IDxcBlob> CompileShaderLibrary(LPCWSTR fileName, const std::vector<std::wstring>& extraDefines = {}) {
    return CompileShaderNew(fileName, L"", L"lib_6_9", extraDefines);
}

inline Microsoft::WRL::ComPtr<IDxcBlob> CompileCS(LPCWSTR fileName, LPCWSTR entryPoint = L"main") {
    return CompileShaderNew(fileName, entryPoint, L"cs_6_9");
}

inline Microsoft::WRL::ComPtr<IDxcBlob> CompileWG(LPCWSTR fileName, LPCWSTR entryPoint = L"main") {
    return CompileShaderNew(fileName, entryPoint, L"lib_6_9");
}

inline ComPtr<ID3D12DescriptorHeap> CreateDescriptorHeap(ID3D12Device* device, uint32_t count,
                                                         D3D12_DESCRIPTOR_HEAP_TYPE type, bool shaderVisible) {
    D3D12_DESCRIPTOR_HEAP_DESC desc = {};
    desc.NumDescriptors = count;
    desc.Type = type;
    desc.Flags = shaderVisible ? D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE : D3D12_DESCRIPTOR_HEAP_FLAG_NONE;

    ComPtr<ID3D12DescriptorHeap> pHeap;
    ThrowIfFailed(device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&pHeap)));
    return pHeap;
}

} // namespace nv_helpers_dx12
