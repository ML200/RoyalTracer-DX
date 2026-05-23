#include <cstdio>
#include <cuda_runtime.h>

#include "gpu/cuda_check.h"

namespace pb {

__global__ void hello_kernel() {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("[CUDA] Hello from kernel.\n");
    }
}

void run_hello_kernel() {
    hello_kernel<<<1, 32>>>();
    CUDA_CHECK(cudaDeviceSynchronize());
}

void print_cuda_device_info() {
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    std::printf("[CUDA] %d device(s) detected.\n", device_count);

    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp props{};
        CUDA_CHECK(cudaGetDeviceProperties(&props, i));
        const double gib = static_cast<double>(props.totalGlobalMem)
                         / (1024.0 * 1024.0 * 1024.0);
        std::printf("[CUDA] Device %d: %s, SM %d.%d, %.2f GiB, %d MPs.\n",
                    i, props.name, props.major, props.minor,
                    gib, props.multiProcessorCount);
    }
}

}
