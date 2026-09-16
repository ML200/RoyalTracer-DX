#pragma once

#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <cmath>

inline void ThrowIfFailed(HRESULT hr)
{
    if (FAILED(hr))
    {
        throw std::exception();
    }
}