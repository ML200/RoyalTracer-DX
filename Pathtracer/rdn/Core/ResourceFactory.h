#pragma once
// ═══════════════════════════════════════════════════════════════════
// Core/ResourceFactory.h — Convenience wrappers for buffer/texture
//                          creation, upload, and barrier management.
// ═══════════════════════════════════════════════════════════════════

#include "../Common.h"
#include "../nv_helpers_dx12/BottomLevelASGenerator.h" // for heap props
#include "../DXRHelper.h"

struct ResourceFactory {
    explicit ResourceFactory(ID3D12Device* dev) : device(dev) {}

    // ── Raw buffer (default heap, UAV-capable) ───────────────────
    ComPtr<ID3D12Resource> CreateUAVBuffer(UINT sizeBytes, const std::wstring& name) const {
        auto desc = CD3DX12_RESOURCE_DESC::Buffer(sizeBytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        ComPtr<ID3D12Resource> res;
        ThrowIfFailed(device->CreateCommittedResource(
            &nv_helpers_dx12::kDefaultHeapProps, D3D12_HEAP_FLAG_NONE,
            &desc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&res)));
        res->SetName(name.c_str());
        return res;
    }

    // ── Default-heap buffer (generic read after copy) ────────────
    ComPtr<ID3D12Resource> CreateDefaultBuffer(UINT sizeBytes, const std::wstring& name) const {
        auto res = nv_helpers_dx12::CreateBuffer(device, sizeBytes,
            D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST,
            nv_helpers_dx12::kDefaultHeapProps);
        res->SetName(name.c_str());
        return res;
    }

    // ── Upload-heap buffer ───────────────────────────────────────
    ComPtr<ID3D12Resource> CreateUploadBuffer(UINT sizeBytes) const {
        return nv_helpers_dx12::CreateBuffer(device, sizeBytes,
            D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ,
            nv_helpers_dx12::kUploadHeapProps);
    }

    // ── Upload data to a default-heap buffer via staging ─────────
    //    Returns the upload heap (caller must keep alive until GPU done)
    ComPtr<ID3D12Resource> UploadToBuffer(
        ID3D12GraphicsCommandList* cmdList,
        ID3D12Resource* dst, const void* data, UINT sizeBytes,
        D3D12_RESOURCE_STATES afterState = D3D12_RESOURCE_STATE_GENERIC_READ) const
    {
        auto upload = CreateUploadBuffer(sizeBytes);
        void* p = nullptr;
        CD3DX12_RANGE r(0, 0);
        ThrowIfFailed(upload->Map(0, &r, &p));
        memcpy(p, data, sizeBytes);
        upload->Unmap(0, nullptr);

        cmdList->CopyBufferRegion(dst, 0, upload.Get(), 0, sizeBytes);
        auto barrier = CD3DX12_RESOURCE_BARRIER::Transition(dst,
            D3D12_RESOURCE_STATE_COPY_DEST, afterState);
        cmdList->ResourceBarrier(1, &barrier);
        return upload;
    }

    // ── Create 2D texture (default heap) ─────────────────────────
    ComPtr<ID3D12Resource> CreateTexture2D(
        UINT w, UINT h, DXGI_FORMAT fmt, UINT arraySize = 1,
        D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        const std::wstring& name = L"") const
    {
        D3D12_RESOURCE_DESC d = {};
        d.Dimension        = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        d.Width            = w;
        d.Height           = h;
        d.DepthOrArraySize = arraySize;
        d.MipLevels        = 1;
        d.Format           = fmt;
        d.SampleDesc.Count = 1;
        d.Flags            = flags;

        ComPtr<ID3D12Resource> res;
        ThrowIfFailed(device->CreateCommittedResource(
            &nv_helpers_dx12::kDefaultHeapProps, D3D12_HEAP_FLAG_NONE,
            &d, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&res)));
        if (!name.empty()) res->SetName(name.c_str());
        return res;
    }

    // ── Upload-mapped helper for constant/structured data ────────
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
