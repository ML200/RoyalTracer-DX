//====================================
//PLANET - GEOMETRY POOL
//====================================

#include "geometry_pool.h"
#include <cassert>
#include <iostream>

namespace planet {

void GeometryPool::init(ID3D12Device5* device) {
    m_vblock      = align_up(CHUNK_VERTEX_BYTES, 256);
    m_iblock      = align_up(CHUNK_INDEX_BYTES,  256);
    m_slot_stride = m_vblock + m_iblock;

    const uint64_t total = m_slot_stride * GEOM_POOL_SIZE;
    //COMMON state: this buffer is a copy-queue CopyBufferRegion destination AND
    //a compute-queue BLAS-build input. For BUFFERS, implicit state promotion +
    //the copy->compute fence cover both uses, so no explicit barriers are needed.
    m_buffer = create_buffer(device, total, D3D12_RESOURCE_FLAG_NONE,
                             D3D12_RESOURCE_STATE_COMMON, HEAP_DEFAULT);
    m_base_va = m_buffer->GetGPUVirtualAddress();

    m_free.clear();
    m_free.reserve(GEOM_POOL_SIZE);
    for (int i = (int)GEOM_POOL_SIZE; i-- > 0; ) m_free.push_back(i);
    m_pending.clear();
    m_pending.reserve(GEOM_POOL_SIZE);

    std::wcout << L"[planet] GeometryPool: " << GEOM_POOL_SIZE << L" slots x "
               << (m_slot_stride / 1024) << L" KB = "
               << (total / (1024 * 1024)) << L" MB" << std::endl;
}

int GeometryPool::acquire() {
    if (m_free.empty()) { assert(false && "GeometryPool exhausted"); return -1; }
    const int idx = m_free.back();
    m_free.pop_back();
    return idx;
}

void GeometryPool::release(int index, uint64_t fence_value) {
    if (index < 0 || index >= (int)GEOM_POOL_SIZE) { assert(false && "GeometryPool: bad index"); return; }
    m_pending.push_back({ index, fence_value });
}

void GeometryPool::reclaim_completed(uint64_t completed_fence_value) {
    size_t w = 0;
    for (size_t r = 0; r < m_pending.size(); ++r) {
        if (m_pending[r].fence <= completed_fence_value)
            m_free.push_back(m_pending[r].index);
        else
            m_pending[w++] = m_pending[r];
    }
    m_pending.resize(w);
}

uint64_t GeometryPool::vertex_offset(int index) const {
    return (uint64_t)index * m_slot_stride;
}
uint64_t GeometryPool::index_offset(int index) const {
    return (uint64_t)index * m_slot_stride + m_vblock;
}
D3D12_GPU_VIRTUAL_ADDRESS GeometryPool::vertex_address(int index) const {
    return m_base_va + vertex_offset(index);
}
D3D12_GPU_VIRTUAL_ADDRESS GeometryPool::index_address(int index) const {
    return m_base_va + index_offset(index);
}

} // namespace planet
