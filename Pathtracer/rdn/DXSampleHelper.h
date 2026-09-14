//*********************************************************
//
// Copyright (c) Microsoft. All rights reserved.
// This code is licensed under the MIT License (MIT).
// THIS CODE IS PROVIDED *AS IS* WITHOUT WARRANTY OF
// ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING ANY
// IMPLIED WARRANTIES OF FITNESS FOR A PARTICULAR
// PURPOSE, MERCHANTABILITY, OR NON-INFRINGEMENT.
//
//*********************************************************

#pragma once

#include "Win32Application.h"
#include <d3d12.h>
#include <wrl/client.h>
#include <source_location>
#include <stdexcept>
#include <cstdio>

inline const char* HrName(HRESULT hr) {
    switch (static_cast<unsigned>(hr)) {
    case 0x80070057u:
        return "E_INVALIDARG";
    case 0x8007000Eu:
        return "E_OUTOFMEMORY";
    case 0x80004005u:
        return "E_FAIL";
    case 0x80004001u:
        return "E_NOTIMPL";
    case 0x887A0001u:
        return "DXGI_ERROR_INVALID_CALL";
    case 0x887A0002u:
        return "DXGI_ERROR_NOT_FOUND";
    case 0x887A0005u:
        return "DXGI_ERROR_DEVICE_REMOVED";
    case 0x887A0006u:
        return "DXGI_ERROR_DEVICE_HUNG";
    case 0x887A0007u:
        return "DXGI_ERROR_DEVICE_RESET";
    case 0x887A0020u:
        return "DXGI_ERROR_DRIVER_INTERNAL_ERROR";
    default:
        return "unrecognised HRESULT";
    }
}

inline void ThrowIfFailed(HRESULT hr, const std::source_location loc = std::source_location::current()) {
    if (FAILED(hr)) {
        char buf[1024];
        sprintf_s(buf, "D3D12 call failed: HRESULT 0x%08X (%s)\n%s:%u\n%s", static_cast<unsigned>(hr), HrName(hr),
                  loc.file_name(), loc.line(), loc.function_name());
        throw std::runtime_error(buf);
    }
}

inline void GetAssetsPath(_Out_writes_(pathSize) WCHAR* path, UINT pathSize) {
    if (path == nullptr) {
        throw std::exception();
    }

    DWORD size = GetModuleFileNameW(nullptr, path, pathSize);
    if (size == 0 || size == pathSize) {
        throw std::exception();
    }

    WCHAR* lastSlash = wcsrchr(path, L'\\');
    if (lastSlash) {
        *(lastSlash + 1) = L'\0';
    }
}

#if defined(_DEBUG)
inline void SetName(ID3D12Object* pObject, LPCWSTR name) {
    pObject->SetName(name);
}
inline void SetNameIndexed(ID3D12Object* pObject, LPCWSTR name, UINT index) {
    WCHAR fullName[50];
    if (swprintf_s(fullName, L"%s[%u]", name, index) > 0) {
        pObject->SetName(fullName);
    }
}
#else
inline void SetName(ID3D12Object*, LPCWSTR) {}
inline void SetNameIndexed(ID3D12Object*, LPCWSTR, UINT) {}
#endif

#define NAME_D3D12_OBJECT(x) SetName(x.Get(), L#x)
#define NAME_D3D12_OBJECT_INDEXED(x, n) SetNameIndexed(x[n].Get(), L#x, n)
