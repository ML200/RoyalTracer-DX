#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

//====================================
//CUDA error check macro
//====================================
#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        cudaError_t _err = (expr);                                             \
        if (_err != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n",                     \
                         __FILE__, __LINE__, cudaGetErrorString(_err));        \
            std::abort();                                                      \
        }                                                                      \
    } while (0)
