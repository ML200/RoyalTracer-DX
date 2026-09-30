#pragma once

#include "../Common.h"
#include "../nv_helpers_dx12/BottomLevelASGenerator.h"
#include "../DXRHelper.h"

struct ResourceFactory {
    explicit ResourceFactory(ID3D12Device* dev) : device(dev) {}

    ComPtr<ID3D12Resource> CreateUAVBuffer(UINT sizeBytes, const std::wstring& name) const {
        auto desc = CD3DX12_RESOURCE_DESC::Buffer(sizeBytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        ComPtr<ID3D12Resource> res;
        ThrowIfFailed(device->CreateCommittedResource(&nv_helpers_dx12::kDefaultHeapProps, D3D12_HEAP_FLAG_NONE, &desc,
                                                      D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                                                      IID_PPV_ARGS(&res)));
        res->SetName(name.c_str());
        return res;
    }

    ComPtr<ID3D12Resource> CreateUploadBuffer(UINT sizeBytes) const {
        return nv_helpers_dx12::CreateBuffer(device, sizeBytes, D3D12_RESOURCE_FLAG_NONE,
                                             D3D12_RESOURCE_STATE_GENERIC_READ, nv_helpers_dx12::kUploadHeapProps);
    }

    ComPtr<ID3D12Resource> CreateTexture2D(UINT w, UINT h, DXGI_FORMAT fmt, UINT arraySize = 1,
                                           D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                           const std::wstring& name = L"") const {
        D3D12_RESOURCE_DESC d = {};
        d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        d.Width = w;
        d.Height = h;
        d.DepthOrArraySize = arraySize;
        d.MipLevels = 1;
        d.Format = fmt;
        d.SampleDesc.Count = 1;
        d.Flags = flags;

        ComPtr<ID3D12Resource> res;
        ThrowIfFailed(device->CreateCommittedResource(&nv_helpers_dx12::kDefaultHeapProps, D3D12_HEAP_FLAG_NONE, &d,
                                                      D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                                                      IID_PPV_ARGS(&res)));
        if (!name.empty())
            res->SetName(name.c_str());
        return res;
    }

    ComPtr<ID3D12Resource> CreateUploadBufferWithData(const void* data, UINT sizeBytes) const {
        auto buf = CreateUploadBuffer(sizeBytes);
        void* p = nullptr;
        buf->Map(0, nullptr, &p);
        memcpy(p, data, sizeBytes);
        buf->Unmap(0, nullptr);
        return buf;
    }

  private:
    ID3D12Device* device;
};
