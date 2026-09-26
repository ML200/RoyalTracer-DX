#pragma once
#include <DirectXMath.h>
#include <Windows.h>
using namespace DirectX;

class InputManager {
  public:
    static bool GetKey(UINT8 key) { return s_currentKeys[key]; }
    static bool GetMouseButton(int btn) { return btn < 3 && s_currentMouse[btn]; }
    static XMFLOAT2 GetMouseDelta() { return {(float)s_deltaX, (float)s_deltaY}; }

    static void BeginFrame();
    static void OnKeyDown(UINT8 key);
    static void OnKeyUp(UINT8 key);
    static void OnMouseMove(int x, int y);
    static void OnMouseButtonDown(int btn);
    static void OnMouseButtonUp(int btn);

  private:
    static bool s_currentKeys[256];
    static bool s_currentMouse[3];
    static int s_mouseX, s_mouseY;
    static int s_deltaX, s_deltaY;
    static int s_accumDeltaX, s_accumDeltaY;
    static bool s_firstMouse;
};
