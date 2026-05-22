//====================================
//PLANET - QUEUE SYNC
//====================================

#include "queue_sync.h"
#include <stdexcept>

namespace planet {

void FenceTimeline::init(ID3D12Device* device) {
    if (FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_fence))))
        throw std::runtime_error("planet::FenceTimeline: CreateFence failed");
    m_next  = 1;
    m_event = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    if (!m_event)
        throw std::runtime_error("planet::FenceTimeline: CreateEvent failed");
}

void FenceTimeline::shutdown() {
    if (m_event) { CloseHandle(m_event); m_event = nullptr; }
}

uint64_t FenceTimeline::signal(ID3D12CommandQueue* queue) {
    const uint64_t v = m_next++;
    queue->Signal(m_fence.Get(), v);
    return v;
}

void FenceTimeline::queue_wait(ID3D12CommandQueue* queue, uint64_t value) {
    if (value > 0) queue->Wait(m_fence.Get(), value);
}

void FenceTimeline::cpu_wait(uint64_t value) {
    if (value == 0 || m_fence->GetCompletedValue() >= value) return;
    m_fence->SetEventOnCompletion(value, m_event);
    WaitForSingleObject(m_event, INFINITE);
}

uint64_t FenceTimeline::completed() const {
    return m_fence->GetCompletedValue();
}

} // namespace planet
