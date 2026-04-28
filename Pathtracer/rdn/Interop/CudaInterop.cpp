//====================================
//CUDA D3D12 INTEROP IMPL
//====================================

#include "CudaInterop.h"

// CUDA 12.x exposes D3D12 interop (external memory / external semaphore)
// directly from cuda_runtime.h; there is no cuda_d3d12_interop.h.
#include <cuda_runtime.h>

#include <windows.h>
#include <cstdio>
#include <cstring>
#include <vector>

// ── Local logging (Common.h pulls in too much for this isolated TU) ──
namespace {
    void LogErr(const char* msg) {
        OutputDebugStringA(msg);
        std::fputs(msg, stderr);
    }

    #define CUDA_TRY(expr, onfail_return)                                      \
        do {                                                                   \
            cudaError_t _e = (expr);                                           \
            if (_e != cudaSuccess) {                                           \
                char _buf[256];                                                \
                std::snprintf(_buf, sizeof(_buf),                              \
                    "[CudaInterop] %s -> %s\n", #expr, cudaGetErrorString(_e));\
                LogErr(_buf);                                                  \
                onfail_return;                                                 \
            }                                                                  \
        } while (0)
}

// ── Opaque state ───────────────────────────────────────────────────
struct CudaInterop::Impl {
    ID3D12Device10* device       = nullptr;
    bool            ready        = false;
    int             cudaDeviceId = -1;
    cudaStream_t    stream       = nullptr;

    struct BufRec {
        cudaExternalMemory_t ext;
        void*                ptr;
        ID3D12Resource*      res;   // non-owning, for identification
    };
    std::vector<BufRec> buffers;

    struct FenceRec {
        uint64_t                id;
        cudaExternalSemaphore_t sem;
    };
    std::vector<FenceRec> fences;
    uint64_t              nextFenceId = 1;

    cudaExternalSemaphore_t findSem(uint64_t id) const {
        for (const auto& r : fences) if (r.id == id) return r.sem;
        return nullptr;
    }
};

// ── Ctor/Dtor/Accessors ────────────────────────────────────────────
CudaInterop::CudaInterop()
    : m_impl(std::make_unique<Impl>()) {}

CudaInterop::~CudaInterop() { Shutdown(); }

bool  CudaInterop::IsReady() const { return m_impl && m_impl->ready; }
void* CudaInterop::Stream () const { return m_impl ? (void*)m_impl->stream : nullptr; }

// ── Init ───────────────────────────────────────────────────────────
bool CudaInterop::Init(ID3D12Device10* device) {
    if (!device || !m_impl) return false;
    m_impl->device = device;

    // Pick the CUDA device whose LUID matches the D3D12 adapter.
    // cudaDeviceProp::luid is 8 bytes: LowPart (4) followed by HighPart (4).
    LUID luid = device->GetAdapterLuid();
    char wantLuid[8];
    std::memcpy(wantLuid,     &luid.LowPart,  4);
    std::memcpy(wantLuid + 4, &luid.HighPart, 4);

    int count = 0;
    CUDA_TRY(cudaGetDeviceCount(&count), return false);

    int matched = -1;
    for (int i = 0; i < count; ++i) {
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, i) != cudaSuccess) continue;
        if (std::memcmp(prop.luid, wantLuid, 8) == 0) { matched = i; break; }
    }
    if (matched < 0) {
        LogErr("[CudaInterop] No CUDA device matches the D3D12 adapter LUID\n");
        return false;
    }

    CUDA_TRY(cudaSetDevice(matched), return false);
    CUDA_TRY(cudaStreamCreateWithFlags(&m_impl->stream, cudaStreamNonBlocking), return false);

    m_impl->cudaDeviceId = matched;
    m_impl->ready        = true;
    return true;
}

// ── Shutdown ───────────────────────────────────────────────────────
void CudaInterop::Shutdown() {
    if (!m_impl) return;

    if (m_impl->stream) {
        cudaStreamSynchronize(m_impl->stream);
        cudaStreamDestroy(m_impl->stream);
        m_impl->stream = nullptr;
    }
    for (auto& b : m_impl->buffers) {
        if (b.ptr) cudaFree(b.ptr);
        if (b.ext) cudaDestroyExternalMemory(b.ext);
    }
    m_impl->buffers.clear();

    for (auto& f : m_impl->fences) {
        if (f.sem) cudaDestroyExternalSemaphore(f.sem);
    }
    m_impl->fences.clear();

    m_impl->device = nullptr;
    m_impl->ready  = false;
}

// ── Buffer creation ────────────────────────────────────────────────
CudaInterop::Buffer CudaInterop::CreateBuffer(size_t bytes, const wchar_t* debugName) {
    Buffer out;
    if (!IsReady() || bytes == 0) return out;

    // 1. Shared, UAV-capable committed buffer.
    D3D12_HEAP_PROPERTIES heapProps{};
    heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;

    D3D12_RESOURCE_DESC desc{};
    desc.Dimension          = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width              = bytes;
    desc.Height             = 1;
    desc.DepthOrArraySize   = 1;
    desc.MipLevels          = 1;
    desc.Format             = DXGI_FORMAT_UNKNOWN;
    desc.SampleDesc.Count   = 1;
    desc.Layout             = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags              = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    HRESULT hr = m_impl->device->CreateCommittedResource(
        &heapProps,
        D3D12_HEAP_FLAG_SHARED,
        &desc,
        D3D12_RESOURCE_STATE_COMMON,
        nullptr,
        IID_PPV_ARGS(&out.resource));
    if (FAILED(hr)) {
        LogErr("[CudaInterop] CreateCommittedResource (SHARED buffer) failed\n");
        return {};
    }
    if (debugName) out.resource->SetName(debugName);

    // 2. NT shared handle.
    HANDLE sharedH = nullptr;
    hr = m_impl->device->CreateSharedHandle(
        out.resource.Get(), nullptr, GENERIC_ALL, nullptr, &sharedH);
    if (FAILED(hr) || !sharedH) {
        LogErr("[CudaInterop] CreateSharedHandle (buffer) failed\n");
        return {};
    }

    // 3. Import to CUDA as external memory.
    cudaExternalMemoryHandleDesc memDesc{};
    memDesc.type                 = cudaExternalMemoryHandleTypeD3D12Resource;
    memDesc.handle.win32.handle  = sharedH;
    memDesc.size                 = bytes;
    memDesc.flags                = cudaExternalMemoryDedicated;

    cudaExternalMemory_t ext = nullptr;
    cudaError_t e = cudaImportExternalMemory(&ext, &memDesc);
    CloseHandle(sharedH);   // CUDA duplicates the handle on import.
    if (e != cudaSuccess) {
        char buf[256];
        std::snprintf(buf, sizeof(buf),
            "[CudaInterop] cudaImportExternalMemory failed: %s\n", cudaGetErrorString(e));
        LogErr(buf);
        return {};
    }

    // 4. Map to a CUDA device pointer.
    cudaExternalMemoryBufferDesc bufDesc{};
    bufDesc.offset = 0;
    bufDesc.size   = bytes;
    bufDesc.flags  = 0;

    void* ptr = nullptr;
    e = cudaExternalMemoryGetMappedBuffer(&ptr, ext, &bufDesc);
    if (e != cudaSuccess) {
        cudaDestroyExternalMemory(ext);
        char buf[256];
        std::snprintf(buf, sizeof(buf),
            "[CudaInterop] cudaExternalMemoryGetMappedBuffer failed: %s\n",
            cudaGetErrorString(e));
        LogErr(buf);
        return {};
    }

    out.cudaPtr   = ptr;
    out.sizeBytes = bytes;
    m_impl->buffers.push_back({ ext, ptr, out.resource.Get() });
    return out;
}

// ── Fence creation ─────────────────────────────────────────────────
CudaInterop::Fence CudaInterop::CreateFence(const wchar_t* debugName) {
    Fence out;
    if (!IsReady()) return out;

    HRESULT hr = m_impl->device->CreateFence(
        0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&out.fence));
    if (FAILED(hr)) {
        LogErr("[CudaInterop] CreateFence (SHARED) failed\n");
        return {};
    }
    if (debugName) out.fence->SetName(debugName);

    HANDLE sharedH = nullptr;
    hr = m_impl->device->CreateSharedHandle(
        out.fence.Get(), nullptr, GENERIC_ALL, nullptr, &sharedH);
    if (FAILED(hr) || !sharedH) {
        LogErr("[CudaInterop] CreateSharedHandle (fence) failed\n");
        return {};
    }

    cudaExternalSemaphoreHandleDesc semDesc{};
    semDesc.type                = cudaExternalSemaphoreHandleTypeD3D12Fence;
    semDesc.handle.win32.handle = sharedH;

    cudaExternalSemaphore_t sem = nullptr;
    cudaError_t e = cudaImportExternalSemaphore(&sem, &semDesc);
    CloseHandle(sharedH);
    if (e != cudaSuccess) {
        char buf[256];
        std::snprintf(buf, sizeof(buf),
            "[CudaInterop] cudaImportExternalSemaphore failed: %s\n",
            cudaGetErrorString(e));
        LogErr(buf);
        return {};
    }

    out.id = m_impl->nextFenceId++;
    m_impl->fences.push_back({ out.id, sem });
    return out;
}

// ── CUDA-side sync ─────────────────────────────────────────────────
void CudaInterop::CudaWait(const Fence& f, uint64_t value) {
    if (!IsReady()) return;
    cudaExternalSemaphore_t sem = m_impl->findSem(f.id);
    if (!sem) return;

    cudaExternalSemaphoreWaitParams p{};
    p.params.fence.value = value;
    CUDA_TRY(cudaWaitExternalSemaphoresAsync(&sem, &p, 1, m_impl->stream), return);
}

void CudaInterop::CudaSignal(const Fence& f, uint64_t value) {
    if (!IsReady()) return;
    cudaExternalSemaphore_t sem = m_impl->findSem(f.id);
    if (!sem) return;

    cudaExternalSemaphoreSignalParams p{};
    p.params.fence.value = value;
    CUDA_TRY(cudaSignalExternalSemaphoresAsync(&sem, &p, 1, m_impl->stream), return);
}
