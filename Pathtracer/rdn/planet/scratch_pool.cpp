//====================================
//PLANET - SCRATCH POOL
//====================================

#include "scratch_pool.h"
#include <cassert>
#include <iostream>

namespace planet {

void ScratchPool::init(ID3D12Device5* device) {
    const ChunkBlasSizes sizes = query_chunk_blas_sizes(device);

    //BLAS scratch data uses the same 256-byte AS alignment as the result buffer.
    m_slot_stride = align_up(sizes.scratch_size,
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT);

    const uint64_t total = m_slot_stride * SCRATCH_POOL_SIZE;
    m_buffer = create_buffer(
        device, total,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_COMMON,
        HEAP_DEFAULT);
    m_base_va = m_buffer->GetGPUVirtualAddress();

    m_free.clear();
    m_free.reserve(SCRATCH_POOL_SIZE);
    for (int i = (int)SCRATCH_POOL_SIZE; i-- > 0; )
        m_free.push_back(i);
    m_pending.clear();
    m_pending.reserve(SCRATCH_POOL_SIZE);

    std::wcout << L"[planet] ScratchPool: " << SCRATCH_POOL_SIZE << L" buffers x "
               << (m_slot_stride / 1024) << L" KB = "
               << (total / 1024) << L" KB" << std::endl;
}

int ScratchPool::acquire() {
    if (m_free.empty()) {
        assert(false && "ScratchPool exhausted");
        return -1;                       // fail soft in release builds
    }
    const int idx = m_free.back();
    m_free.pop_back();
    return idx;
}

void ScratchPool::release(int index, uint64_t fence_value) {
    if (index < 0 || index >= (int)SCRATCH_POOL_SIZE) {
        assert(false && "ScratchPool: bad index");
        return;
    }
    m_pending.push_back({ index, fence_value });
}

void ScratchPool::reclaim_completed(uint64_t completed_fence_value) {
    size_t w = 0;
    for (size_t r = 0; r < m_pending.size(); ++r) {
        if (m_pending[r].fence <= completed_fence_value)
            m_free.push_back(m_pending[r].index);
        else
            m_pending[w++] = m_pending[r];
    }
    m_pending.resize(w);
}

D3D12_GPU_VIRTUAL_ADDRESS ScratchPool::gpu_address(int index) const {
    return m_base_va + (uint64_t)index * m_slot_stride;
}

} // namespace planet
