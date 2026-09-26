#include "blas_pool.h"
#include <iostream>
#include <stdexcept>

namespace planet {

ComPtr<ID3D12Resource> create_buffer(ID3D12Device* device, uint64_t size,
                                     D3D12_RESOURCE_FLAGS  flags,
                                     D3D12_RESOURCE_STATES state,
                                     const D3D12_HEAP_PROPERTIES& heap) {
    if (size == 0) size = 256;

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

}
