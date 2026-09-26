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

constexpr D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS PLANET_BLAS_BUILD_FLAGS =
    (D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS)(
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE |
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_COMPACTION);

}
