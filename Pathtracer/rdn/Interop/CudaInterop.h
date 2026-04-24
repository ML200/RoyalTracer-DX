#pragma once
//====================================
//CUDA D3D12 INTEROP FOR NRC
//====================================
//shared D3D12 buffers as raw CUDA device pointers + shared fences
//callers never see a CUDA header, types hidden behind opaque pimpl
//typical setup, Init -> CreateBuffer+CreateFence, HLSL binds as UAV, tcnn uses cudaPtr

#include <d3d12.h>
#include <wrl/client.h>
#include <cstdint>
#include <memory>

class CudaInterop {
public:
    //====================================
    //PUBLIC CUDA-FREE TYPES
    //====================================
    struct Buffer {
        Microsoft::WRL::ComPtr<ID3D12Resource> resource;
        void*  cudaPtr   = nullptr;
        size_t sizeBytes = 0;
    };

    struct Fence {
        Microsoft::WRL::ComPtr<ID3D12Fence> fence;
        uint64_t id = 0;
    };

    CudaInterop();
    ~CudaInterop();
    CudaInterop(const CudaInterop&)            = delete;
    CudaInterop& operator=(const CudaInterop&) = delete;

    //====================================
    //LIFETIME
    //====================================
    //matches CUDA device to D3D12 adapter via LUID
    //returns false if no matching CUDA device, caller falls back to non-CUDA path
    bool Init(ID3D12Device10* device);
    void Shutdown();
    bool IsReady() const;

    //====================================
    //RESOURCE CREATION
    //====================================
    //shared D3D12 UAV buffer, imported to CUDA, empty Buffer on failure
    Buffer CreateBuffer(size_t bytes, const wchar_t* debugName = nullptr);

    //shared D3D12 fence + CUDA external semaphore
    //signal/wait from CUDA via CudaSignal/CudaWait, from D3D12 via cmdQueue->Signal/Wait
    Fence CreateFence(const wchar_t* debugName = nullptr);

    //====================================
    //CUDA-SIDE SYNC ON INTERNAL STREAM
    //====================================
    void CudaWait  (const Fence& f, uint64_t value);
    void CudaSignal(const Fence& f, uint64_t value);

    //====================================
    //STREAM ACCESSOR
    //====================================
    //cudaStream_t as void*, CUDA side reinterprets, auto s = (cudaStream_t)interop.Stream()
    void* Stream() const;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};
