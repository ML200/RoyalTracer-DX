#pragma once
// ═══════════════════════════════════════════════════════════════════
// Interop/CudaInterop.h — CUDA/D3D12 interop for NRC (tiny-cuda-nn).
//
// Exposes shared D3D12 buffers as raw CUDA device pointers, plus
// shared fences for queue-level synchronization. Designed so a caller
// never sees a single CUDA header — all CUDA types are hidden behind
// an opaque pimpl.
//
// Typical setup:
//     m_cuda.Init(device);
//     auto q = m_cuda.CreateBuffer(W*H*sizeof(QueryRecord), L"NRC_Queries");
//     auto f = m_cuda.CreateFence(L"NRC_Fence");
//     // HLSL:   bind q.resource as UAV
//     // tcnn:   pass q.cudaPtr
//     // D3D12:  cmdQueue->Signal(f.fence.Get(), v);
//     // CUDA:   m_cuda.CudaWait(f, v);
// ═══════════════════════════════════════════════════════════════════

#include <d3d12.h>
#include <wrl/client.h>
#include <cstdint>
#include <memory>

class CudaInterop {
public:
    // ── Public, CUDA-free types ───────────────────────────────────
    struct Buffer {
        Microsoft::WRL::ComPtr<ID3D12Resource> resource;   // bind as UAV/SRV in HLSL
        void*  cudaPtr   = nullptr;                        // pass to tcnn / CUDA kernels
        size_t sizeBytes = 0;
    };

    struct Fence {
        Microsoft::WRL::ComPtr<ID3D12Fence> fence;         // signal/wait from D3D12 queue
        uint64_t id = 0;                                   // opaque handle into Impl
    };

    CudaInterop();
    ~CudaInterop();
    CudaInterop(const CudaInterop&)            = delete;
    CudaInterop& operator=(const CudaInterop&) = delete;

    // ── Lifetime ─────────────────────────────────────────────────
    // Matches the CUDA device to the D3D12 adapter via adapter LUID.
    // Returns false if no matching CUDA device exists — caller should
    // fall back to a non-CUDA code path.
    bool Init(ID3D12Device10* device);
    void Shutdown();
    bool IsReady() const;

    // ── Resource creation ────────────────────────────────────────
    // Allocates a SHARED D3D12 committed buffer (UAV-capable) and
    // imports it to CUDA. Empty Buffer on failure (resource == null).
    Buffer CreateBuffer(size_t bytes, const wchar_t* debugName = nullptr);

    // Allocates a SHARED D3D12 fence and imports it as a CUDA external
    // semaphore. Signal/Wait from CUDA via CudaSignal/CudaWait below;
    // from D3D12 via cmdQueue->Signal(fence, v) or cmdQueue->Wait.
    Fence CreateFence(const wchar_t* debugName = nullptr);

    // ── CUDA-side sync (on internal stream) ──────────────────────
    void CudaWait  (const Fence& f, uint64_t value);
    void CudaSignal(const Fence& f, uint64_t value);

    // ── Stream accessor ──────────────────────────────────────────
    // Returns cudaStream_t as void* to keep CUDA headers out.
    // Reinterpret on the CUDA side: auto s = (cudaStream_t)interop.Stream();
    void* Stream() const;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};
