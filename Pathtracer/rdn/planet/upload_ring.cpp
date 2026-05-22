//====================================
//PLANET - UPLOAD RING
//====================================

#include "upload_ring.h"
#include <cassert>
#include <iostream>
#include <stdexcept>

namespace planet {

void UploadRing::init(ID3D12Device5* device) {
    //one frame region must hold a full frame of chunk uploads: up to
    //MAX_BUILDS_PER_FRAME chunks, each a 256-aligned vertex block + index block.
    const uint32_t per_chunk = (uint32_t)(align_up(CHUNK_VERTEX_BYTES, 256)
                                        + align_up(CHUNK_INDEX_BYTES, 256));
    m_frame_capacity = per_chunk * MAX_BUILDS_PER_FRAME;

    const uint64_t total = (uint64_t)m_frame_capacity * UPLOAD_RING_FRAMES;
    m_buffer = create_buffer(
        device, total,
        D3D12_RESOURCE_FLAG_NONE,
        D3D12_RESOURCE_STATE_GENERIC_READ,
        HEAP_UPLOAD);
    m_gpu_base = m_buffer->GetGPUVirtualAddress();

    //map once and keep it mapped for the buffer's whole lifetime
    void* mapped = nullptr;
    D3D12_RANGE no_read = { 0, 0 };          // CPU writes only, never reads back
    const HRESULT hr = m_buffer->Map(0, &no_read, &mapped);
    if (FAILED(hr))
        throw std::runtime_error("planet::UploadRing: Map failed");
    m_cpu_base = static_cast<uint8_t*>(mapped);

    for (uint32_t& o : m_offset) o = 0;
    m_active_slot = 0;

    std::wcout << L"[planet] UploadRing: " << UPLOAD_RING_FRAMES << L" frames x "
               << (m_frame_capacity / 1024) << L" KB = "
               << (total / (1024 * 1024)) << L" MB" << std::endl;
}

void UploadRing::begin_frame(uint32_t frame_slot) {
    m_active_slot = frame_slot % UPLOAD_RING_FRAMES;
}

UploadRing::Region UploadRing::sub_allocate(uint32_t bytes, uint32_t align) {
    const uint32_t start = (uint32_t)align_up(m_offset[m_active_slot], align);
    Region r;
    if (start + bytes > m_frame_capacity) {
        assert(false && "UploadRing frame region overflow");
        return r;                            // invalid (cpu_ptr == nullptr)
    }
    const uint64_t abs = (uint64_t)m_active_slot * m_frame_capacity + start;
    r.cpu_ptr = m_cpu_base + abs;
    r.gpu_va  = m_gpu_base + abs;
    r.offset  = abs;
    r.bytes   = bytes;
    m_offset[m_active_slot] = start + bytes;
    return r;
}

void UploadRing::reset_frame(uint32_t frame_slot) {
    m_offset[frame_slot % UPLOAD_RING_FRAMES] = 0;
}

} // namespace planet
