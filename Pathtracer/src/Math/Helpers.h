//
// Created by m on 30.01.2024.
//

#ifndef PATHTRACER_HELPERS_H
#define PATHTRACER_HELPERS_H

#pragma once

#define WIN32_LEAN_AND_MEAN
#include <Windows.h> // For HRESULT
#include <cmath>

// From DXSampleHelper.h
// Source: https://github.com/Microsoft/DirectX-Graphics-Samples
inline void ThrowIfFailed(HRESULT hr)
{
    if (FAILED(hr))
    {
        throw std::exception();
    }
}
#endif //PATHTRACER_HELPERS_H
