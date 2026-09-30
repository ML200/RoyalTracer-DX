#pragma once

#include <cstdint>
#include <vector>
#include "blas_pool.h"

namespace planet {

class TlasBuilder {
public:
    void init(ID3D12Device5* device, uint32_t max_instances);

    void reserve(uint32_t required_instances);
    void begin(uint32_t required_instances = 0);

    // Camera-relative transform; diffed against the last frame.
    void add_instance(D3D12_GPU_VIRTUAL_ADDRESS blas,
                      const float transform[12],
                      uint32_t instance_id,
                      uint32_t hit_group_index,
                      D3D12_RAYTRACING_INSTANCE_FLAGS flags);

    // Rebuilds on change or force; refit updates in place.
    bool build(ID3D12GraphicsCommandList4* cmd, bool force = false, bool refit = false);
    bool last_build_recorded() const { return m_lastBuildRecorded; }

    D3D12_GPU_VIRTUAL_ADDRESS tlas_address() const { return m_result->GetGPUVirtualAddress(); }
    ID3D12Resource* result()         const { return m_result.Get(); }
    uint32_t        instance_count() const { return m_count; }
    uint32_t        max_instances()  const { return m_max; }
    uint64_t        result_bytes()   const { return m_result ? m_result->GetDesc().Width : 0; }
    uint64_t        scratch_bytes()  const { return m_scratch ? m_scratch->GetDesc().Width : 0; }

private:
    ComPtr<ID3D12Device5> m_device;
    ComPtr<ID3D12Resource> m_result;
    ComPtr<ID3D12Resource> m_scratch;
    ComPtr<ID3D12Resource> m_instanceDescs;
    D3D12_RAYTRACING_INSTANCE_DESC* m_mapped = nullptr;
    uint32_t m_max   = 0;
    uint32_t m_count = 0;
    uint32_t m_builtCount = 0;
    bool m_built = false;
    bool m_changed = true;
    bool m_lastBuildRecorded = false;
    uint32_t m_refitsSinceBuild = 0;
    std::vector<D3D12_RAYTRACING_INSTANCE_DESC> m_descriptors;
};

}
