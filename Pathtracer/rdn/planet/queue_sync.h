#pragma once

#include <cstdint>
#include <windows.h>
#include <d3d12.h>
#include <wrl/client.h>

namespace planet {

class FenceTimeline {
public:
    void init(ID3D12Device* device);
    void shutdown();

    uint64_t signal(ID3D12CommandQueue* queue);
    void     queue_wait(ID3D12CommandQueue* queue, uint64_t value);
    void     cpu_wait(uint64_t value);
    uint64_t completed() const;
    ID3D12Fence* fence() const { return m_fence.Get(); }

private:
    Microsoft::WRL::ComPtr<ID3D12Fence> m_fence;
    uint64_t m_next  = 1;
    HANDLE   m_event = nullptr;
};

}
