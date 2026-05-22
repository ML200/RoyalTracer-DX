#pragma once
//====================================
//PLANET - TLAS BUILDER
//====================================
//Builds the unified top-level acceleration structure each frame on the async
//compute queue: scene mesh instances + terrain Ready-chunks + the 6-face
//fallback layer, in one camera-relative (scene-origin-relative) TLAS.
//
//Instance descriptors are written into a persistently-mapped upload buffer; the
//result + scratch buffers are preallocated for a max instance count and never
//resized. Per frame: begin() -> add_instance() x N -> build(compute_cl).

#include <cstdint>
#include "blas_pool.h"   // create_buffer, HEAP_*, ComPtr, <d3d12.h>

namespace planet {

class TlasBuilder {
public:
    //preallocate result/scratch sized for max_instances; map the desc buffer.
    void init(ID3D12Device5* device, uint32_t max_instances);

    void begin();        // reset the instance write cursor for a new frame

    //append one instance. 'transform' is a row-major 3x4 object->world matrix
    //(12 floats); 'blas' is the instance's BLAS result-buffer GPU address.
    void add_instance(D3D12_GPU_VIRTUAL_ADDRESS blas,
                      const float transform[12],
                      uint32_t instance_id,
                      uint32_t hit_group_index,
                      D3D12_RAYTRACING_INSTANCE_FLAGS flags);

    //record the full TLAS rebuild on a compute-capable command list
    void build(ID3D12GraphicsCommandList4* cmd);

    D3D12_GPU_VIRTUAL_ADDRESS tlas_address() const { return m_result->GetGPUVirtualAddress(); }
    ID3D12Resource* result()         const { return m_result.Get(); }
    uint32_t        instance_count() const { return m_count; }
    uint32_t        max_instances()  const { return m_max; }

private:
    ComPtr<ID3D12Resource> m_result;          // TLAS, permanently in AS state
    ComPtr<ID3D12Resource> m_scratch;
    ComPtr<ID3D12Resource> m_instanceDescs;   // upload heap, persistently mapped
    D3D12_RAYTRACING_INSTANCE_DESC* m_mapped = nullptr;
    uint32_t m_max   = 0;
    uint32_t m_count = 0;
};

} // namespace planet
