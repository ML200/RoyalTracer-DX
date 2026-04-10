#pragma once
#include <DirectXMath.h>
#include <cstring>
#include <Windows.h>
using namespace DirectX;

class InputManager {
public:
    static bool     GetKey(UINT8 key)           { return s_currentKeys[key]; }
    static bool     GetKeyDown(UINT8 key)       { return s_currentKeys[key] && !s_previousKeys[key]; }
    static bool     GetKeyUp(UINT8 key)         { return !s_currentKeys[key] && s_previousKeys[key]; }
    static bool     GetMouseButton(int btn)     { return btn < 3 && s_currentMouse[btn]; }
    static bool     GetMouseButtonDown(int btn) { return btn < 3 && s_currentMouse[btn] && !s_previousMouse[btn]; }
    static XMFLOAT2 GetMouseDelta()             { return { (float)s_deltaX, (float)s_deltaY }; }
    static XMFLOAT2 GetMousePosition()          { return { (float)s_mouseX, (float)s_mouseY }; }
    static float    GetAxis(const char* name);

    static void BeginFrame();
    static void OnKeyDown(UINT8 key);
    static void OnKeyUp(UINT8 key);
    static void OnMouseMove(int x, int y);
    static void OnMouseButtonDown(int btn);
    static void OnMouseButtonUp(int btn);

private:
    static bool s_currentKeys[256], s_previousKeys[256];
    static bool s_currentMouse[3], s_previousMouse[3];
    static int  s_mouseX, s_mouseY;
    static int  s_deltaX, s_deltaY;
    static int  s_accumDeltaX, s_accumDeltaY;
    static bool s_firstMouse;
};
