#pragma once

#include <cstdint>
#include <vector>
#include "chunk_mesh.h"
#include <windows.h>
#include <d3d12.h>
#include <wrl/client.h>

namespace planet {

using Microsoft::WRL::ComPtr;

constexpr D3D12_HEAP_PROPERTIES HEAP_DEFAULT = {
    D3D12_HEAP_TYPE_DEFAULT, D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_MEMORY_POOL_UNKNOWN, 0, 0 };
constexpr D3D12_HEAP_PROPERTIES HEAP_UPLOAD = {
    D3D12_HEAP_TYPE_UPLOAD,  D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_MEMORY_POOL_UNKNOWN, 0, 0 };
constexpr D3D12_HEAP_PROPERTIES HEAP_READBACK = {
    D3D12_HEAP_TYPE_READBACK, D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_MEMORY_POOL_UNKNOWN, 0, 0 };

inline uint64_t align_up(uint64_t v, uint64_t a) { return (v + a - 1) & ~(a - 1); }

ComPtr<ID3D12Resource> create_buffer(ID3D12Device* device, uint64_t size,
                                     D3D12_RESOURCE_FLAGS  flags,
                                     D3D12_RESOURCE_STATES state,
                                     const D3D12_HEAP_PROPERTIES& heap);

constexpr uint32_t MAX_BLAS_SLOTS = 512;
constexpr uint16_t INVALID_SLOT   = 0xFFFFu;

constexpr D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS PLANET_BLAS_BUILD_FLAGS =
    (D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS)(
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE |
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_COMPACTION);

struct ChunkBlasSizes {
    uint64_t result_size  = 0;
    uint64_t scratch_size = 0;
};
ChunkBlasSizes query_chunk_blas_sizes(ID3D12Device5* device);

class BlasPool {
public:
    // Initializes fixed, 256-byte-aligned BLAS result slots.
    void init(ID3D12Device5* device);

    // Claims a slot whose previous GPU use completed.
    uint16_t allocate();

    // Returns a slot after its supplied GPU fence retires.
    void release(uint16_t slot, uint64_t fence_value);

    void reclaim_completed(uint64_t completed_fence_value);

    D3D12_GPU_VIRTUAL_ADDRESS gpu_address(uint16_t slot) const;

    uint64_t slot_size()   const { return m_slot_stride; }
    uint32_t capacity()    const { return MAX_BLAS_SLOTS; }
    uint32_t free_count()  const { return (uint32_t)m_free.size(); }
    uint32_t in_use()      const { return MAX_BLAS_SLOTS - free_count() - (uint32_t)m_pending.size(); }
    uint64_t total_bytes() const { return m_slot_stride * MAX_BLAS_SLOTS; }
    ID3D12Resource* buffer() const { return m_buffer.Get(); }

private:
    ComPtr<ID3D12Resource>    m_buffer;
    D3D12_GPU_VIRTUAL_ADDRESS m_base_va = 0;
    uint64_t                  m_slot_stride = 0;

    std::vector<uint16_t> m_free;
    struct Pending { uint16_t slot; uint64_t fence; };
    std::vector<Pending>  m_pending;
};

}
