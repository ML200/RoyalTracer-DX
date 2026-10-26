/******************************************************************************
 * Copyright 1998-2018 NVIDIA Corp. All Rights Reserved.
 *****************************************************************************/

#pragma once

#include <atomic>
#include <cstring>
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

// On-disk DXIL cache in ShaderCache/ (next to the exe). An entry is reused only when the compiler version,
// arguments, source and every included file are byte-identical to the compile that produced it.
namespace shader_cache {
inline constexpr uint32_t kMagic = 0x32435344; // "DSC2"; bump when the entry layout changes
inline constexpr wchar_t kDir[] = L"ShaderCache";

inline uint64_t Hash(const void* data, size_t size, uint64_t h = 14695981039346656037ull) {
    const auto* bytes = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < size; ++i)
        h = (h ^ bytes[i]) * 1099511628211ull;
    return h;
}

struct Dependency {
    std::wstring path;
    uint64_t hash;
};

// Wraps the default handler to record each include DXC opens. Stack-allocated; lives only for one Compile().
struct RecordingIncludeHandler final : IDxcIncludeHandler {
    IDxcIncludeHandler* inner;
    std::vector<Dependency> deps;

    explicit RecordingIncludeHandler(IDxcIncludeHandler* handler) : inner(handler) {}

    HRESULT STDMETHODCALLTYPE LoadSource(LPCWSTR fileName, IDxcBlob** source) override {
        HRESULT hr = inner->LoadSource(fileName, source);
        if (SUCCEEDED(hr) && *source) {
            wchar_t full[MAX_PATH];
            const DWORD n = GetFullPathNameW(fileName, MAX_PATH, full, nullptr);
            std::wstring path = (n > 0 && n < MAX_PATH) ? std::wstring(full, n) : std::wstring(fileName);
            bool seen = false;
            for (const auto& d : deps)
                seen |= d.path == path;
            if (!seen)
                deps.push_back({std::move(path), Hash((*source)->GetBufferPointer(), (*source)->GetBufferSize())});
        }
        return hr;
    }
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** object) override {
        if (riid == __uuidof(IDxcIncludeHandler) || riid == __uuidof(IUnknown)) {
            *object = this;
            return S_OK;
        }
        *object = nullptr;
        return E_NOINTERFACE;
    }
    ULONG STDMETHODCALLTYPE AddRef() override { return 1; }
    ULONG STDMETHODCALLTYPE Release() override { return 1; }
};

// Entry layout: magic, source hash, dependency count, {path length, path, hash}..., DXIL size, DXIL.
inline ComPtr<IDxcBlob> Load(const std::wstring& entryPath, uint64_t sourceHash, IDxcUtils* utils) {
    std::ifstream in(entryPath, std::ios::binary | std::ios::ate);
    const std::streamoff fileSize = in ? (std::streamoff)in.tellg() : 0;
    if (fileSize <= 0)
        return nullptr;
    std::vector<char> data((size_t)fileSize);
    in.seekg(0);
    if (!in.read(data.data(), (std::streamsize)data.size()))
        return nullptr;

    size_t offset = 0;
    auto read = [&](void* dst, size_t size) {
        if (size > data.size() - offset)
            return false;
        memcpy(dst, data.data() + offset, size);
        offset += size;
        return true;
    };

    uint32_t magic = 0, depCount = 0;
    uint64_t storedSourceHash = 0;
    if (!read(&magic, 4) || magic != kMagic || !read(&storedSourceHash, 8) || storedSourceHash != sourceHash ||
        !read(&depCount, 4))
        return nullptr;

    for (uint32_t i = 0; i < depCount; ++i) {
        uint32_t pathLength = 0;
        if (!read(&pathLength, 4) || pathLength > 32767)
            return nullptr;
        std::wstring path(pathLength, L'\0');
        uint64_t depHash = 0;
        if (!read(path.data(), pathLength * sizeof(wchar_t)) || !read(&depHash, 8))
            return nullptr;
        ComPtr<IDxcBlobEncoding> dep;
        if (FAILED(utils->LoadFile(path.c_str(), nullptr, &dep)) ||
            Hash(dep->GetBufferPointer(), dep->GetBufferSize()) != depHash)
            return nullptr;
    }

    uint64_t blobSize = 0;
    if (!read(&blobSize, 8) || blobSize == 0 || blobSize != data.size() - offset)
        return nullptr;
    ComPtr<IDxcBlobEncoding> blob;
    if (FAILED(utils->CreateBlob(data.data() + offset, (UINT32)blobSize, DXC_CP_ACP, &blob)))
        return nullptr;
    return blob;
}

inline void Store(const std::wstring& entryPath, uint64_t sourceHash, const std::vector<Dependency>& deps,
                  IDxcBlob* blob) {
    CreateDirectoryW(kDir, nullptr);
    std::ofstream out(entryPath, std::ios::binary | std::ios::trunc);
    if (!out)
        return;
    const uint32_t depCount = (uint32_t)deps.size();
    out.write(reinterpret_cast<const char*>(&kMagic), 4);
    out.write(reinterpret_cast<const char*>(&sourceHash), 8);
    out.write(reinterpret_cast<const char*>(&depCount), 4);
    for (const auto& d : deps) {
        const uint32_t pathLength = (uint32_t)d.path.size();
        out.write(reinterpret_cast<const char*>(&pathLength), 4);
        out.write(reinterpret_cast<const char*>(d.path.data()), pathLength * sizeof(wchar_t));
        out.write(reinterpret_cast<const char*>(&d.hash), 8);
    }
    const uint64_t blobSize = blob->GetBufferSize();
    out.write(reinterpret_cast<const char*>(&blobSize), 8);
    out.write(static_cast<const char*>(blob->GetBufferPointer()), (std::streamsize)blobSize);
}
} // namespace shader_cache

inline ComPtr<IDxcBlob> CompileShaderNew(LPCWSTR fileName, LPCWSTR entryPoint, LPCWSTR targetProfile,
                                         const std::vector<std::wstring>& extraDefines = {}) {
    struct CompilerContext {
        ComPtr<IDxcCompiler3> compiler;
        ComPtr<IDxcUtils> utils;
        ComPtr<IDxcIncludeHandler> includes;
        uint64_t versionHash = 0;

        CompilerContext() {
            ThrowIfFailed(DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(&compiler)));
            ThrowIfFailed(DxcCreateInstance(CLSID_DxcUtils, IID_PPV_ARGS(&utils)));
            ThrowIfFailed(utils->CreateDefaultIncludeHandler(&includes));

            // A different dxcompiler.dll invalidates every cache entry.
            ComPtr<IDxcVersionInfo> version;
            UINT32 v[2] = {};
            if (SUCCEEDED(compiler.As(&version)))
                version->GetVersion(&v[0], &v[1]);
            versionHash = shader_cache::Hash(v, sizeof(v));
            ComPtr<IDxcVersionInfo2> version2;
            UINT32 commitCount = 0;
            char* commit = nullptr;
            if (SUCCEEDED(compiler.As(&version2)) && SUCCEEDED(version2->GetCommitInfo(&commitCount, &commit)) &&
                commit) {
                versionHash = shader_cache::Hash(&commitCount, sizeof(commitCount), versionHash);
                versionHash = shader_cache::Hash(commit, strlen(commit), versionHash);
                CoTaskMemFree(commit);
            }
        }
    };
    // DXC objects are not thread-safe; shaders compile on several threads at once.
    static thread_local const CompilerContext context;

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

    // Defines for all shaders, e.g. RDN_SHADER_DEFINES="A=0;B=1".
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

    std::wstring shaderName(fileName);
    shaderName = shaderName.substr(shaderName.find_last_of(L"/\\") + 1);

    uint64_t argsHash = context.versionHash;
    for (LPCWSTR arg : args)
        argsHash = shader_cache::Hash(arg, (wcslen(arg) + 1) * sizeof(wchar_t), argsHash);
    wchar_t argsHex[17];
    swprintf_s(argsHex, L"%016llX", argsHash);
    const std::wstring cachePath = std::wstring(shader_cache::kDir) + L"/" + shaderName + L"-" +
                                   (*entryPoint ? entryPoint : L"lib") + L"-" + argsHex + L".bin";
    const uint64_t sourceHash = shader_cache::Hash(sourceBuffer.Ptr, sourceBuffer.Size);

    auto compileT0 = std::chrono::high_resolution_clock::now();
    auto elapsedMs = [&] {
        return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() -
                                                                     compileT0)
            .count();
    };

    // One write per line, so lines from concurrent compiles do not interleave.
    auto logLine = [](const std::wstring& line) { std::wcout << line + L"\n" << std::flush; };

    ComPtr<IDxcBlob> pBlob = shader_cache::Load(cachePath, sourceHash, context.utils.Get());
    if (pBlob) {
        logLine(L"[Shader] " + shaderName + L" loaded from cache in " + std::to_wstring(elapsedMs()) + L" ms");
    } else {
        shader_cache::RecordingIncludeHandler includes(context.includes.Get());
        ComPtr<IDxcResult> pResult;
        hr = context.compiler->Compile(&sourceBuffer, args.data(), (uint32_t)args.size(), &includes,
                                       IID_PPV_ARGS(&pResult));

        if (SUCCEEDED(hr)) {
            ThrowIfFailed(pResult->GetStatus(&hr));
        }
        auto compileMs = elapsedMs();

        if (FAILED(hr)) {
            ComPtr<IDxcBlobUtf8> pErrors;
            if (pResult)
                pResult->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(&pErrors), nullptr);
            std::string errMsg = "Shader compilation failed.";
            if (pErrors && pErrors->GetStringLength() > 0) {
                OutputDebugStringA(pErrors->GetStringPointer());
                errMsg.assign(pErrors->GetStringPointer(), pErrors->GetStringLength());
            }
            // An error in a shared include fails every concurrent compile; one dialog is enough.
            static std::atomic<bool> errorShown{false};
            if (!errorShown.exchange(true))
                MessageBoxA(nullptr, errMsg.c_str(), "Shader Compilation Failed", MB_OK | MB_ICONERROR);
            throw std::runtime_error(errMsg);
        }

        logLine(L"[Shader] " + shaderName + L" compiled in " + std::to_wstring(compileMs) + L" ms");

        ThrowIfFailed(pResult->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(&pBlob), nullptr));
        shader_cache::Store(cachePath, sourceHash, includes.deps, pBlob.Get());
    }

    // Shaders kept for Aftermath dump decoding (Diagnostics.h).
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
