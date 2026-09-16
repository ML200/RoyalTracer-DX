#pragma once

#include <cstdint>
#include <vector>
#include "blas_pool.h"

namespace planet {

constexpr uint32_t GEOM_POOL_SIZE = 48;

class GeometryPool {
public:
    // Initializes one shared COMMON-state vertex/index allocation.
    void init(ID3D12Device5* device);

    // Acquires a reusable chunk geometry slot.
    int  acquire();
    // Defers reuse until the supplied GPU fence retires.
    void release(int index, uint64_t fence_value);
    void reclaim_completed(uint64_t completed_fence_value);

    ID3D12Resource*           buffer() const { return m_buffer.Get(); }
    uint64_t                  vertex_offset (int index) const;
    uint64_t                  index_offset  (int index) const;
    D3D12_GPU_VIRTUAL_ADDRESS vertex_address(int index) const;
    D3D12_GPU_VIRTUAL_ADDRESS index_address (int index) const;

    uint32_t capacity()    const { return GEOM_POOL_SIZE; }
    uint32_t free_count()  const { return (uint32_t)m_free.size(); }
    uint64_t total_bytes() const { return m_slot_stride * GEOM_POOL_SIZE; }

private:
    ComPtr<ID3D12Resource>    m_buffer;
    D3D12_GPU_VIRTUAL_ADDRESS m_base_va = 0;
    uint64_t                  m_vblock = 0;
    uint64_t                  m_iblock = 0;
    uint64_t                  m_slot_stride = 0;

    std::vector<int> m_free;
    struct Pending { int index; uint64_t fence; };
    std::vector<Pending> m_pending;
};

}
