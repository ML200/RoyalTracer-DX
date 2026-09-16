#pragma once

#include <cstdint>
#include <vector>
#include "blas_pool.h"

namespace planet {

constexpr uint32_t SCRATCH_POOL_SIZE = 8;

class ScratchPool {
public:
    // Initializes fixed scratch slots sized for the worst BLAS build.
    void init(ID3D12Device5* device);

    // Acquires scratch storage for one in-flight BLAS build.
    int  acquire();
    // Defers reuse until the supplied GPU fence retires.
    void release(int index, uint64_t fence_value);
    void reclaim_completed(uint64_t completed_fence_value);

    D3D12_GPU_VIRTUAL_ADDRESS gpu_address(int index) const;

    uint64_t scratch_size() const { return m_slot_stride; }
    uint32_t capacity()     const { return SCRATCH_POOL_SIZE; }
    uint32_t free_count()   const { return (uint32_t)m_free.size(); }
    uint64_t total_bytes()  const { return m_slot_stride * SCRATCH_POOL_SIZE; }

private:
    ComPtr<ID3D12Resource>    m_buffer;
    D3D12_GPU_VIRTUAL_ADDRESS m_base_va = 0;
    uint64_t                  m_slot_stride = 0;

    std::vector<int> m_free;
    struct Pending { int index; uint64_t fence; };
    std::vector<Pending> m_pending;
};

}
