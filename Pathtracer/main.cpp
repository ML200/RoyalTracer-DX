#include <Windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <iostream>

// Link necessary d3d12 libraries
#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")

// Forward declaration of the WindowProc function
LRESULT CALLBACK WindowProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);

int main() {
    // Initialize the window class.
    WNDCLASSEX windowClass = {0};
    windowClass.cbSize = sizeof(WNDCLASSEX);
    windowClass.style = CS_HREDRAW | CS_VREDRAW;
    windowClass.lpfnWndProc = WindowProc;
    windowClass.hInstance = GetModuleHandle(nullptr);
    windowClass.hCursor = LoadCursor(NULL, IDC_ARROW);
    windowClass.lpszClassName = "DirectX12WindowClass";
    RegisterClassEx(&windowClass);

    RECT windowRect = {0, 0, 1280, 720};
    AdjustWindowRect(&windowRect, WS_OVERLAPPEDWINDOW, FALSE);

    // Create the window and store a handle to it.
    HWND windowHandle = CreateWindow(
            windowClass.lpszClassName,
            "DirectX12 Basic Window",
            WS_OVERLAPPEDWINDOW,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            windowRect.right - windowRect.left,
            windowRect.bottom - windowRect.top,
            nullptr,    // We have no parent window.
            nullptr,    // We aren't using menus.
            windowClass.hInstance,
            nullptr);

    if (!windowHandle) {
        return -1;
    }

    ShowWindow(windowHandle, SW_SHOW);

    // Initialize DirectX 12
    // ...

    // Main message loop
    MSG msg = {};
    while (msg.message != WM_QUIT) {
        if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
    }

    // Clean up DirectX 12
    // ...

    return static_cast<int>(msg.wParam);
}

// WindowProc function
LRESULT CALLBACK WindowProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) {
    switch (message) {
        case WM_DESTROY:
            PostQuitMessage(0);
            return 0;
    }

    // Handle any messages the switch statement didn't
    return DefWindowProc(hWnd, message, wParam, lParam);
}