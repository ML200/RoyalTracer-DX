#include "InputManager.h"

bool InputManager::s_currentKeys[256] = {};
bool InputManager::s_currentMouse[3] = {};
int InputManager::s_mouseX = 0;
int InputManager::s_mouseY = 0;
int InputManager::s_deltaX = 0;
int InputManager::s_deltaY = 0;
int InputManager::s_accumDeltaX = 0;
int InputManager::s_accumDeltaY = 0;
bool InputManager::s_firstMouse = true;

void InputManager::BeginFrame() {
    s_deltaX = s_accumDeltaX;
    s_deltaY = s_accumDeltaY;
    s_accumDeltaX = 0;
    s_accumDeltaY = 0;
}
void InputManager::OnKeyDown(UINT8 key) {
    s_currentKeys[key] = true;
}
void InputManager::OnKeyUp(UINT8 key) {
    s_currentKeys[key] = false;
}
void InputManager::OnMouseMove(int x, int y) {
    if (s_firstMouse) {
        s_mouseX = x;
        s_mouseY = y;
        s_firstMouse = false;
    }
    s_accumDeltaX += x - s_mouseX;
    s_accumDeltaY += y - s_mouseY;
    s_mouseX = x;
    s_mouseY = y;
}
void InputManager::OnMouseButtonDown(int btn) {
    if (btn >= 0 && btn < 3)
        s_currentMouse[btn] = true;
}
void InputManager::OnMouseButtonUp(int btn) {
    if (btn >= 0 && btn < 3)
        s_currentMouse[btn] = false;
}
