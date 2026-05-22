#pragma once
//====================================
//PLANET - QUEUE SYNC
//====================================
//FenceTimeline: an ID3D12Fence plus a monotonic value, with signal / cross-queue
//wait / CPU wait / completion query. The planet stream pipeline uses one per
//async queue (copy, compute). Self-contained - pulls only <d3d12.h> + WRL.

#include <cstdint>
#include <windows.h>
#include <d3d12.h>
#include <wrl/client.h>

namespace planet {

class FenceTimeline {
public:
    void init(ID3D12Device* device);
    void shutdown();

    uint64_t signal(ID3D12CommandQueue* queue);                      // signal next value, return it
    void     queue_wait(ID3D12CommandQueue* queue, uint64_t value);  // GPU-side wait on a queue
    void     cpu_wait(uint64_t value);                               // block the CPU until value
    uint64_t completed() const;                                      // fence GetCompletedValue
    uint64_t last_signaled() const { return m_next - 1; }
    ID3D12Fence* fence() const { return m_fence.Get(); }

private:
    Microsoft::WRL::ComPtr<ID3D12Fence> m_fence;
    uint64_t m_next  = 1;
    HANDLE   m_event = nullptr;
};

} // namespace planet
