#pragma once
//====================================
//PLANET - SCRATCH POOL
//====================================
//Preallocated pool of BLAS-build scratch buffers, carved from one default-heap
//buffer. A scratch buffer is needed only while its build is in flight; once the
//build's fence retires it is reusable. Same claim / fence-deferred-release
//pattern as BlasPool. CPU bookkeeping only - init() is the only GPU touch.

#include <cstdint>
#include <vector>
#include "blas_pool.h"   // ChunkBlasSizes, query_chunk_blas_sizes, DXR helpers, ROUND_UP

namespace planet {

constexpr uint32_t SCRATCH_POOL_SIZE = 8;   // max concurrent in-flight BLAS builds

class ScratchPool {
public:
    void init(ID3D12Device5* device);

    int  acquire();                                    // scratch index, or -1 if exhausted
    void release(int index, uint64_t fence_value);
    void reclaim_completed(uint64_t completed_fence_value);

    D3D12_GPU_VIRTUAL_ADDRESS gpu_address(int index) const;

    //----- queries -----
    uint64_t scratch_size() const { return m_slot_stride; }   // usable bytes per buffer
    uint32_t capacity()     const { return SCRATCH_POOL_SIZE; }
    uint32_t free_count()   const { return (uint32_t)m_free.size(); }
    uint64_t total_bytes()  const { return m_slot_stride * SCRATCH_POOL_SIZE; }

private:
    ComPtr<ID3D12Resource>    m_buffer;
    D3D12_GPU_VIRTUAL_ADDRESS m_base_va = 0;
    uint64_t                  m_slot_stride = 0;       // ROUND_UP(scratch_size, 256)

    std::vector<int> m_free;
    struct Pending { int index; uint64_t fence; };
    std::vector<Pending> m_pending;
};

} // namespace planet
