#pragma once

#include <cstdint>
#include <windows.h>
#include <d3d12.h>
#include <wrl/client.h>

namespace planet {

class FenceTimeline {
public:
    // Creates the fence and event used by both CPU and GPU waits.
    void init(ID3D12Device* device);
    void shutdown();

    // Signals the queue and returns its monotonic fence value.
    uint64_t signal(ID3D12CommandQueue* queue);
    // Inserts a GPU wait without blocking the calling thread.
    void     queue_wait(ID3D12CommandQueue* queue, uint64_t value);
    void     cpu_wait(uint64_t value);
    uint64_t completed() const;
    uint64_t last_signaled() const { return m_next - 1; }
    ID3D12Fence* fence() const { return m_fence.Get(); }

private:
    Microsoft::WRL::ComPtr<ID3D12Fence> m_fence;
    uint64_t m_next  = 1;
    HANDLE   m_event = nullptr;
};

}
