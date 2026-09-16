#pragma once

#include <cstdint>
#include "blas_pool.h"

namespace planet {

constexpr uint32_t UPLOAD_RING_FRAMES = 3;

class UploadRing {
public:
    struct Region {
        void*                     cpu_ptr = nullptr;
        D3D12_GPU_VIRTUAL_ADDRESS  gpu_va  = 0;
        uint64_t                  offset  = 0;
        uint32_t                  bytes   = 0;
        bool valid() const { return cpu_ptr != nullptr; }
    };

    void init(ID3D12Device5* device);

    // Selects a frame region after its fence was checked by the caller.
    void begin_frame(uint32_t frame_slot);

    // Bump-allocates persistently mapped upload memory with alignment.
    Region sub_allocate(uint32_t bytes, uint32_t align);

    // Rewinds a region after the caller confirms fence completion.
    void reset_frame(uint32_t frame_slot);

    uint32_t frame_capacity() const { return m_frame_capacity; }
    uint64_t total_bytes()    const { return (uint64_t)m_frame_capacity * UPLOAD_RING_FRAMES; }
    ID3D12Resource* buffer()  const { return m_buffer.Get(); }

private:
    ComPtr<ID3D12Resource>    m_buffer;
    uint8_t*                  m_cpu_base = nullptr;
    D3D12_GPU_VIRTUAL_ADDRESS m_gpu_base = 0;

    uint32_t m_frame_capacity = 0;
    uint32_t m_offset[UPLOAD_RING_FRAMES] = {};
    uint32_t m_active_slot = 0;
};

}
