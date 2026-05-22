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

//Readable name for the HRESULTs that actually turn up in D3D12/DXR work.
//Anything unrecognised still prints as hex, which is enough to look up.
inline const char* HrName(HRESULT hr)
{
	switch (static_cast<unsigned>(hr))
	{
	case 0x80070057u: return "E_INVALIDARG";
	case 0x8007000Eu: return "E_OUTOFMEMORY";
	case 0x80004005u: return "E_FAIL";
	case 0x80004001u: return "E_NOTIMPL";
	case 0x887A0001u: return "DXGI_ERROR_INVALID_CALL";
	case 0x887A0002u: return "DXGI_ERROR_NOT_FOUND";
	case 0x887A0005u: return "DXGI_ERROR_DEVICE_REMOVED";
	case 0x887A0006u: return "DXGI_ERROR_DEVICE_HUNG";
	case 0x887A0007u: return "DXGI_ERROR_DEVICE_RESET";
	case 0x887A0020u: return "DXGI_ERROR_DRIVER_INTERNAL_ERROR";
	default:          return "unrecognised HRESULT";
	}
}

//Throws on failure with the HRESULT and the call site baked into the
//message. The previous version threw a default-constructed std::exception,
//whose what() is MSVC's useless literal "Unknown exception" — every D3D12
//failure looked identical and the actual error code was discarded.
inline void ThrowIfFailed(HRESULT hr,
	const std::source_location loc = std::source_location::current())
{
	if (FAILED(hr))
	{
		char buf[1024];
		sprintf_s(buf,
			"D3D12 call failed: HRESULT 0x%08X (%s)\n%s:%u\n%s",
			static_cast<unsigned>(hr), HrName(hr),
			loc.file_name(), loc.line(), loc.function_name());
		throw std::runtime_error(buf);
	}
}

inline void GetAssetsPath(_Out_writes_(pathSize) WCHAR* path, UINT pathSize)
{
	if (path == nullptr)
	{
		throw std::exception();
	}

	DWORD size = GetModuleFileName(nullptr, reinterpret_cast<LPSTR>(path), pathSize);
	if (size == 0 || size == pathSize)
	{
		//failed or truncated
		throw std::exception();
	}

	WCHAR* lastSlash = wcsrchr(path, L'\\');
	if (lastSlash)
	{
		*(lastSlash + 1) = L'\0';
	}
}

inline HRESULT ReadDataFromFile(LPCWSTR filename, byte** data, UINT* size)
{
	using namespace Microsoft::WRL;

	CREATEFILE2_EXTENDED_PARAMETERS extendedParams = {};
	extendedParams.dwSize = sizeof(CREATEFILE2_EXTENDED_PARAMETERS);
	extendedParams.dwFileAttributes = FILE_ATTRIBUTE_NORMAL;
	extendedParams.dwFileFlags = FILE_FLAG_SEQUENTIAL_SCAN;
	extendedParams.dwSecurityQosFlags = SECURITY_ANONYMOUS;
	extendedParams.lpSecurityAttributes = nullptr;
	extendedParams.hTemplateFile = nullptr;

	Wrappers::FileHandle file(CreateFile2(filename, GENERIC_READ, FILE_SHARE_READ, OPEN_EXISTING, &extendedParams));
	if (file.Get() == INVALID_HANDLE_VALUE)
	{
		throw std::exception();
	}

	FILE_STANDARD_INFO fileInfo = {};
	if (!GetFileInformationByHandleEx(file.Get(), FileStandardInfo, &fileInfo, sizeof(fileInfo)))
	{
		throw std::exception();
	}

	if (fileInfo.EndOfFile.HighPart != 0)
	{
		throw std::exception();
	}

	*data = reinterpret_cast<byte*>(malloc(fileInfo.EndOfFile.LowPart));
	*size = fileInfo.EndOfFile.LowPart;

	if (!ReadFile(file.Get(), *data, fileInfo.EndOfFile.LowPart, nullptr, nullptr))
	{
		throw std::exception();
	}

	return S_OK;
}

//debug naming helpers
#if defined(_DEBUG)
inline void SetName(ID3D12Object* pObject, LPCWSTR name)
{
	pObject->SetName(name);
}
inline void SetNameIndexed(ID3D12Object* pObject, LPCWSTR name, UINT index)
{
	WCHAR fullName[50];
	if (swprintf_s(fullName, L"%s[%u]", name, index) > 0)
	{
		pObject->SetName(fullName);
	}
}
#else
inline void SetName(ID3D12Object*, LPCWSTR)
{
}
inline void SetNameIndexed(ID3D12Object*, LPCWSTR, UINT)
{
}
#endif

//ComPtr naming helpers, use variable name as object name
#define NAME_D3D12_OBJECT(x) SetName(x.Get(), L#x)
#define NAME_D3D12_OBJECT_INDEXED(x, n) SetNameIndexed(x[n].Get(), L#x, n)
