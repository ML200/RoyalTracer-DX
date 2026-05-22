#pragma once
//====================================
//PLANET - GEOMETRY POOL
//====================================
//Cycling pool of chunk vertex+index GPU buffers - the copy destination for a
//tessellated mesh and the geometry input for its BLAS build. A static BLAS does
//not reference its source geometry once built, so a slot is reusable as soon as
//its build fence retires. One default-heap buffer in COMMON state, GEOM_POOL_SIZE
//slots (each = a 256-aligned vertex block + index block). Same claim /
//fence-deferred-release pattern as ScratchPool.

#include <cstdint>
#include <vector>
#include "blas_pool.h"   // create_buffer, HEAP_DEFAULT, align_up, ComPtr, chunk_mesh consts

namespace planet {

constexpr uint32_t GEOM_POOL_SIZE = 48;   // ~3x MAX_BUILDS_PER_FRAME of in-flight geometry

class GeometryPool {
public:
    void init(ID3D12Device5* device);

    int  acquire();                                    // slot index, or -1 if exhausted
    void release(int index, uint64_t fence_value);
    void reclaim_completed(uint64_t completed_fence_value);

    ID3D12Resource*           buffer() const { return m_buffer.Get(); }
    uint64_t                  vertex_offset (int index) const;   // byte offset within buffer()
    uint64_t                  index_offset  (int index) const;
    D3D12_GPU_VIRTUAL_ADDRESS vertex_address(int index) const;
    D3D12_GPU_VIRTUAL_ADDRESS index_address (int index) const;

    uint32_t capacity()    const { return GEOM_POOL_SIZE; }
    uint32_t free_count()  const { return (uint32_t)m_free.size(); }
    uint64_t total_bytes() const { return m_slot_stride * GEOM_POOL_SIZE; }

private:
    ComPtr<ID3D12Resource>    m_buffer;
    D3D12_GPU_VIRTUAL_ADDRESS m_base_va = 0;
    uint64_t                  m_vblock = 0;        // 256-aligned CHUNK_VERTEX_BYTES
    uint64_t                  m_iblock = 0;        // 256-aligned CHUNK_INDEX_BYTES
    uint64_t                  m_slot_stride = 0;   // m_vblock + m_iblock

    std::vector<int> m_free;
    struct Pending { int index; uint64_t fence; };
    std::vector<Pending> m_pending;
};

} // namespace planet
