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

#include <iostream>

#include "stdafx.h"
#include "Renderer.h"

#include <comdef.h>


class ScopedComInitializer
{
public:
    ScopedComInitializer()
    {
        m_hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    }

    ~ScopedComInitializer()
    {
        if (SUCCEEDED(m_hr))
        {
            CoUninitialize();
        }
    }

    // Eine kleine Hilfsfunktion, um den Status zu prüfen
    operator HRESULT() const
    {
        return m_hr;
    }

private:
    HRESULT m_hr;
};

_Use_decl_annotations_
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, int nCmdShow)
{
    // Erstelle das Scoped-Objekt. COM wird im Konstruktor initialisiert.
    ScopedComInitializer comInitializer;
    if (FAILED(comInitializer))
    {
        MessageBox(nullptr, reinterpret_cast<LPCSTR>(L"Failed to initialize COM."), reinterpret_cast<LPCSTR>(L"Error"), MB_OK | MB_ICONERROR);
        return 1;
    }

    if (AllocConsole()) {
        freopen("CONOUT$", "w", stdout);
        freopen("CONOUT$", "w", stderr);
        std::wcout << L"Console initialized" << std::endl;
    }

    Renderer sample(1920, 1080, L"DXR Pathtracer - experimental");
    return Win32Application::Run(&sample, hInstance, nCmdShow);
}
