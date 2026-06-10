#pragma once
//====================================
//PLANET - BLAS POOL
//====================================
//Preallocated pool of bottom-level acceleration-structure result-buffer slots.
//Every slot is a 256-aligned sub-range of ONE committed buffer that lives
//permanently in the RAYTRACING_ACCELERATION_STRUCTURE state and is never
//transitioned. Slots are handed terrain by allocate(), returned by release() with a
//fence value, and become reusable once reclaim_completed() sees that fence
//retire on the GPU (the GPU may still read an old BLAS via an in-flight TLAS).
//
//This header also carries the Phase 2 chunk-geometry sizing constants and the
//small GPU helpers shared by the scratch pool and upload ring. The planet GPU
//module is self-contained: it pulls only <d3d12.h> + WRL, not the engine's DXR
//helpers. Every method here is CPU bookkeeping only - init() is the sole GPU touch.

#include <cstdint>
#include <vector>
#include "chunk_mesh.h"
#include <windows.h>
#include <d3d12.h>
#include <wrl/client.h>

namespace planet {

using Microsoft::WRL::ComPtr;

//====================================
//GPU HELPERS
//====================================
//heap property blocks (default = GPU-local VRAM, upload = CPU-visible staging)
constexpr D3D12_HEAP_PROPERTIES HEAP_DEFAULT = {
    D3D12_HEAP_TYPE_DEFAULT, D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_MEMORY_POOL_UNKNOWN, 0, 0 };
constexpr D3D12_HEAP_PROPERTIES HEAP_UPLOAD = {
    D3D12_HEAP_TYPE_UPLOAD,  D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_MEMORY_POOL_UNKNOWN, 0, 0 };
constexpr D3D12_HEAP_PROPERTIES HEAP_READBACK = {
    D3D12_HEAP_TYPE_READBACK, D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_MEMORY_POOL_UNKNOWN, 0, 0 };

//round v up to a power-of-two alignment
inline uint64_t align_up(uint64_t v, uint64_t a) { return (v + a - 1) & ~(a - 1); }

//create a committed buffer; throws std::runtime_error on failure (startup use only)
ComPtr<ID3D12Resource> create_buffer(ID3D12Device* device, uint64_t size,
                                     D3D12_RESOURCE_FLAGS  flags,
                                     D3D12_RESOURCE_STATES state,
                                     const D3D12_HEAP_PROPERTIES& heap);

//====================================
//POOL CONSTANTS (Phase 2)
//====================================
//Chunk-geometry constants (MAX_CHUNK_VERTS/TRIS, CHUNK_VERTEX_STRIDE, the byte
//budgets, the ChunkVertex layout) live in chunk_mesh.h - shared with the CPU
//tessellator. Only the BLAS-pool-specific constants remain here.
constexpr uint32_t MAX_BLAS_SLOTS = 512;
constexpr uint16_t INVALID_SLOT   = 0xFFFFu;

//BLAS build flags. The pool sizes its slots for these; Phase 4 MUST build with
//the same flags. PREFER_FAST_TRACE for static terrain; ALLOW_COMPACTION is set
//so compaction can be added later without resizing the pool. No ALLOW_UPDATE.
constexpr D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS PLANET_BLAS_BUILD_FLAGS =
    (D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS)(
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE |
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_COMPACTION);

//====================================
//PREBUILD SIZES
//====================================
//Worst-case BLAS result + scratch sizes for one chunk, queried from the driver.
struct ChunkBlasSizes {
    uint64_t result_size  = 0;   // ResultDataMaxSizeInBytes
    uint64_t scratch_size = 0;   // ScratchDataSizeInBytes
};
ChunkBlasSizes query_chunk_blas_sizes(ID3D12Device5* device);

//====================================
//BLAS POOL
//====================================
class BlasPool {
public:
    //allocate the backing buffer (one CreateCommittedResource). Call once at startup.
    void init(ID3D12Device5* device);

    //claim a free slot; returns INVALID_SLOT if the pool is exhausted.
    uint16_t allocate();

    //return a slot. It is NOT reusable until reclaim_completed() sees fence_value
    //retire - an in-flight TLAS may still reference the old BLAS.
    void release(uint16_t slot, uint64_t fence_value);

    //move every pending slot whose fence has retired back onto the free list.
    void reclaim_completed(uint64_t completed_fence_value);

    //GPU virtual address of a slot's BLAS result buffer (256-aligned).
    D3D12_GPU_VIRTUAL_ADDRESS gpu_address(uint16_t slot) const;

    //----- queries -----
    uint64_t slot_size()   const { return m_slot_stride; }   // usable bytes per slot
    uint32_t capacity()    const { return MAX_BLAS_SLOTS; }
    uint32_t free_count()  const { return (uint32_t)m_free.size(); }
    uint32_t in_use()      const { return MAX_BLAS_SLOTS - free_count() - (uint32_t)m_pending.size(); }
    uint64_t total_bytes() const { return m_slot_stride * MAX_BLAS_SLOTS; }
    ID3D12Resource* buffer() const { return m_buffer.Get(); }

private:
    ComPtr<ID3D12Resource>    m_buffer;             // one buffer, all slots
    D3D12_GPU_VIRTUAL_ADDRESS m_base_va = 0;
    uint64_t                  m_slot_stride = 0;    // align_up(result_size, 256)

    std::vector<uint16_t> m_free;                   // free slot indices
    struct Pending { uint16_t slot; uint64_t fence; };
    std::vector<Pending>  m_pending;                // released, awaiting GPU fence
};

} // namespace planet
