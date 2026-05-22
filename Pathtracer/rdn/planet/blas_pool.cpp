//====================================
//PLANET - BLAS POOL
//====================================

#include "blas_pool.h"
#include <cassert>
#include <iostream>
#include <stdexcept>

namespace planet {

//====================================
//BUFFER CREATION
//====================================
ComPtr<ID3D12Resource> create_buffer(ID3D12Device* device, uint64_t size,
                                     D3D12_RESOURCE_FLAGS  flags,
                                     D3D12_RESOURCE_STATES state,
                                     const D3D12_HEAP_PROPERTIES& heap) {
    if (size == 0) size = 256;          // D3D12 rejects a zero-width buffer

    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension          = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Alignment          = 0;
    desc.Width              = size;
    desc.Height             = 1;
    desc.DepthOrArraySize   = 1;
    desc.MipLevels          = 1;
    desc.Format             = DXGI_FORMAT_UNKNOWN;
    desc.SampleDesc.Count   = 1;
    desc.SampleDesc.Quality = 0;
    desc.Layout             = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags              = flags;

    ComPtr<ID3D12Resource> buffer;
    const HRESULT hr = device->CreateCommittedResource(
        &heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr, IID_PPV_ARGS(&buffer));
    if (FAILED(hr)) {
        std::wcout << L"[planet] create_buffer failed: 0x"
                   << std::hex << hr << std::dec << std::endl;
        throw std::runtime_error("planet::create_buffer: CreateCommittedResource failed");
    }
    return buffer;
}

//====================================
//PREBUILD QUERY
//====================================
ChunkBlasSizes query_chunk_blas_sizes(ID3D12Device5* device) {
    //a worst-case chunk geometry desc. Prebuild only reads counts / formats /
    //flags - the buffer addresses are ignored, so they stay zero.
    D3D12_RAYTRACING_GEOMETRY_DESC geom = {};
    geom.Type  = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
    geom.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
    geom.Triangles.VertexBuffer.StartAddress  = 0;
    geom.Triangles.VertexBuffer.StrideInBytes = CHUNK_VERTEX_STRIDE;
    geom.Triangles.VertexCount  = MAX_CHUNK_VERTS;
    geom.Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
    geom.Triangles.IndexBuffer  = 0;
    geom.Triangles.IndexCount   = MAX_CHUNK_TRIS * 3;
    geom.Triangles.IndexFormat  = DXGI_FORMAT_R16_UINT;
    geom.Triangles.Transform3x4 = 0;

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS inputs = {};
    inputs.Type           = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
    inputs.DescsLayout    = D3D12_ELEMENTS_LAYOUT_ARRAY;
    inputs.Flags          = PLANET_BLAS_BUILD_FLAGS;
    inputs.NumDescs       = 1;
    inputs.pGeometryDescs = &geom;

    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO info = {};
    device->GetRaytracingAccelerationStructurePrebuildInfo(&inputs, &info);

    ChunkBlasSizes s;
    s.result_size  = info.ResultDataMaxSizeInBytes;
    s.scratch_size = info.ScratchDataSizeInBytes;
    return s;
}

//====================================
//INIT
//====================================
void BlasPool::init(ID3D12Device5* device) {
    const ChunkBlasSizes sizes = query_chunk_blas_sizes(device);

    //pad each slot to the AS byte alignment so every slot VA is a valid
    //BuildRaytracingAccelerationStructure destination.
    m_slot_stride = align_up(sizes.result_size,
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT);

    const uint64_t total = m_slot_stride * MAX_BLAS_SLOTS;
    m_buffer = create_buffer(
        device, total,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
        HEAP_DEFAULT);
    m_base_va = m_buffer->GetGPUVirtualAddress();

    m_free.clear();
    m_free.reserve(MAX_BLAS_SLOTS);
    for (uint32_t i = MAX_BLAS_SLOTS; i-- > 0; )
        m_free.push_back((uint16_t)i);
    m_pending.clear();
    m_pending.reserve(MAX_BLAS_SLOTS);

    std::wcout << L"[planet] BlasPool: " << MAX_BLAS_SLOTS << L" slots x "
               << (m_slot_stride / 1024) << L" KB = "
               << (total / (1024 * 1024)) << L" MB" << std::endl;
}

//====================================
//ALLOCATE / RELEASE / RECLAIM
//====================================
uint16_t BlasPool::allocate() {
    if (m_free.empty()) {
        assert(false && "BlasPool exhausted");
        return INVALID_SLOT;            // fail soft in release builds
    }
    const uint16_t slot = m_free.back();
    m_free.pop_back();
    return slot;
}

void BlasPool::release(uint16_t slot, uint64_t fence_value) {
    if (slot >= MAX_BLAS_SLOTS) { assert(false && "BlasPool: bad slot"); return; }
    m_pending.push_back({ slot, fence_value });
}

void BlasPool::reclaim_completed(uint64_t completed_fence_value) {
    //compact m_pending in place: retired slots go back to m_free, the rest stay
    size_t w = 0;
    for (size_t r = 0; r < m_pending.size(); ++r) {
        if (m_pending[r].fence <= completed_fence_value)
            m_free.push_back(m_pending[r].slot);
        else
            m_pending[w++] = m_pending[r];
    }
    m_pending.resize(w);
}

D3D12_GPU_VIRTUAL_ADDRESS BlasPool::gpu_address(uint16_t slot) const {
    return m_base_va + (uint64_t)slot * m_slot_stride;
}

} // namespace planet
