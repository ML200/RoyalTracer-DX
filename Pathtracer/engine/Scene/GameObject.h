#pragma once
#include "../Core/Transform.h"
#include <cstdint>
#include <Windows.h>

struct GameObject {
    uint32_t  id        = 0;
    Transform transform;
    UINT      meshIndex = 0;
    bool      active    = true;
};
