#include "core/field.h"
#include "core/neighbor_table.h"
#include "gpu/cuda_check.h"

#include <cuda_runtime.h>
#include <vector_types.h>

#include <cstdint>

namespace pb {

template <typename T>
__global__ void halo_gather_kernel(T* __restrict__ data,
                                   const GhostCopy* __restrict__ copies,
                                   int count) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= count) return;
    GhostCopy c = copies[k];
    data[c.dst_idx] = data[c.src_idx];
}

template <typename T>
void Field<T>::halo_exchange(const NeighborTable& table) {
    int count = static_cast<int>(table.copy_count());
    if (count == 0) return;

    const int block = 256;
    const int grid  = (count + block - 1) / block;
    halo_gather_kernel<T><<<grid, block>>>(buffer_.data(),
                                           table.device_copies().data(),
                                           count);
    CUDA_CHECK(cudaGetLastError());
}

//====================================
//Explicit instantiations. Add new T as fields are introduced (see Field roster
//in PLAN section 6.3). uint8_t, float, float2, float4 cover everything M1
//needs and most of what later passes will need.
//====================================

template class Field<float>;
template class Field<float2>;
template class Field<float4>;
template class Field<std::uint8_t>;
template class Field<std::uint32_t>;

}
