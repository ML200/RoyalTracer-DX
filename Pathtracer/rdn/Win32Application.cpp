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

#include "stdafx.h"

#include "Win32Application.h"
#include "../lib/imgui/imgui.h"

HWND Win32Application::m_hwnd = nullptr;

extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

int Win32Application::Run(DXSample* pSample, HINSTANCE hInstance, int nCmdShow) {
    int argc = 0;
    LPWSTR* argv = CommandLineToArgvW(GetCommandLineW(), &argc);
    if (!argv)
        ThrowIfFailed(HRESULT_FROM_WIN32(GetLastError()));
    pSample->ParseCommandLineArgs(argv, argc);
    LocalFree(argv);

    WNDCLASSEXW windowClass = {};
    windowClass.cbSize = sizeof(windowClass);
    windowClass.style = CS_HREDRAW | CS_VREDRAW;
    windowClass.lpfnWndProc = WindowProc;
    windowClass.hInstance = hInstance;
    windowClass.hCursor = LoadCursorW(nullptr, MAKEINTRESOURCEW(32512));
    windowClass.lpszClassName = L"DXSampleClass";
    if (!RegisterClassExW(&windowClass))
        ThrowIfFailed(HRESULT_FROM_WIN32(GetLastError()));

    RECT windowRect = {0, 0, static_cast<LONG>(pSample->GetWidth()), static_cast<LONG>(pSample->GetHeight())};
    AdjustWindowRect(&windowRect, WS_OVERLAPPEDWINDOW, FALSE);

    m_hwnd = CreateWindowW(windowClass.lpszClassName, pSample->GetTitle(), WS_OVERLAPPEDWINDOW, CW_USEDEFAULT,
                           CW_USEDEFAULT, windowRect.right - windowRect.left, windowRect.bottom - windowRect.top,
                           nullptr, nullptr, hInstance, pSample);

    if (!m_hwnd)
        ThrowIfFailed(HRESULT_FROM_WIN32(GetLastError()));
    pSample->OnInit();

    char benchmarkFrames[32]{};
    unsigned frameLimit = 0;
    if (GetEnvironmentVariableA("RT_BENCHMARK_FRAMES",benchmarkFrames,sizeof(benchmarkFrames)) > 0)
        sscanf_s(benchmarkFrames,"%u",&frameLimit);
    ShowWindow(m_hwnd, frameLimit ? SW_HIDE : nCmdShow);

    MSG msg = {};
    while (msg.message != WM_QUIT) {
        if (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        } else if (frameLimit) {
            pSample->OnUpdate(); pSample->OnRender();
            if (--frameLimit == 0) PostQuitMessage(0);
        }
    }

    pSample->OnDestroy();

    return static_cast<int>(msg.wParam);
}

LRESULT CALLBACK Win32Application::WindowProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) {
    if (ImGui_ImplWin32_WndProcHandler(hWnd, message, wParam, lParam))
        return true;
    DXSample* pSample = reinterpret_cast<DXSample*>(GetWindowLongPtr(hWnd, GWLP_USERDATA));

    switch (message) {
    case WM_CREATE: {
        auto* pCreateStruct = reinterpret_cast<CREATESTRUCTW*>(lParam);
        SetWindowLongPtr(hWnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(pCreateStruct->lpCreateParams));
    }
        return 0;

    case WM_KEYDOWN:
        if (pSample) {
            pSample->OnKeyDown(static_cast<UINT8>(wParam));
        }
        if (static_cast<UINT8>(wParam) == VK_ESCAPE)
            PostQuitMessage(0);
        return 0;

    case WM_KEYUP:
        if (pSample) {
            pSample->OnKeyUp(static_cast<UINT8>(wParam));
        }
        return 0;

    case WM_PAINT:
        if (pSample) {
            pSample->OnUpdate();
            pSample->OnRender();
        }
        return 0;

    case WM_SIZE:
        if (pSample && wParam == SIZE_MAXIMIZED) {
            UINT w = LOWORD(lParam);
            UINT h = HIWORD(lParam);
            if (w > 0 && h > 0)
                pSample->OnResize(w, h);
        }
        return 0;

    case WM_EXITSIZEMOVE:
        if (pSample) {
            RECT rc;
            GetClientRect(hWnd, &rc);
            UINT w = rc.right - rc.left;
            UINT h = rc.bottom - rc.top;
            if (w > 0 && h > 0)
                pSample->OnResize(w, h);
        }
        return 0;

    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    case WM_LBUTTONDOWN:
    case WM_RBUTTONDOWN:
    case WM_MBUTTONDOWN:
        if (pSample) {
            pSample->OnButtonDown(static_cast<UINT32>(lParam));
        }
        return 0;
    case WM_LBUTTONUP:
    case WM_RBUTTONUP:
    case WM_MBUTTONUP:
        if (pSample) {
            pSample->OnButtonUp(message, static_cast<UINT32>(lParam));
        }
        return 0;
    case WM_MOUSEMOVE:
        if (pSample) {
            pSample->OnMouseMove(static_cast<UINT8>(wParam), static_cast<UINT32>(lParam));
        }
        return 0;
    }

    return DefWindowProcW(hWnd, message, wParam, lParam);
}
