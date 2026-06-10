#pragma once
//====================================
//PLANET - UPLOAD RING
//====================================
//One persistently-mapped upload-heap buffer, partitioned into UPLOAD_RING_FRAMES
//equal regions. Each in-flight frame bump-allocates chunk vertex/index data from
//its own region; reset_frame() rewinds a region once that frame's fence retires.
//Mapped once at startup, never unmapped, never transitioned. Tessellation
//workers (Phase 3) write straight into the Regions handed terrain here.

#include <cstdint>
#include "blas_pool.h"   // chunk byte budget, MAX_BUILDS_PER_FRAME, DXR helpers

namespace planet {

constexpr uint32_t UPLOAD_RING_FRAMES = 3;

class UploadRing {
public:
    struct Region {
        void*                     cpu_ptr = nullptr;   // write target (persistently mapped)
        D3D12_GPU_VIRTUAL_ADDRESS  gpu_va  = 0;         // GPU virtual address
        uint64_t                  offset  = 0;         // byte offset within buffer() - CopyBufferRegion src
        uint32_t                  bytes   = 0;
        bool valid() const { return cpu_ptr != nullptr; }
    };

    void init(ID3D12Device5* device);

    //select the region for the frame about to record (frame_index % UPLOAD_RING_FRAMES)
    void begin_frame(uint32_t frame_slot);

    //bump-allocate inside the active frame region; Region.valid()==false on overflow.
    //'align' must be a power of two.
    Region sub_allocate(uint32_t bytes, uint32_t align);

    //rewind a frame region - call once that frame's GPU fence has retired
    void reset_frame(uint32_t frame_slot);

    //----- queries -----
    uint32_t frame_capacity() const { return m_frame_capacity; }
    uint64_t total_bytes()    const { return (uint64_t)m_frame_capacity * UPLOAD_RING_FRAMES; }
    ID3D12Resource* buffer()  const { return m_buffer.Get(); }

private:
    ComPtr<ID3D12Resource>    m_buffer;
    uint8_t*                  m_cpu_base = nullptr;     // persistent Map() pointer
    D3D12_GPU_VIRTUAL_ADDRESS m_gpu_base = 0;

    uint32_t m_frame_capacity = 0;
    uint32_t m_offset[UPLOAD_RING_FRAMES] = {};         // bump offset within each region
    uint32_t m_active_slot = 0;
};

} // namespace planet
