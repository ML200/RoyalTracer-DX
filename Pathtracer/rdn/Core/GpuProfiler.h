#pragma once

#include "../Common.h"
#include "../DXRHelper.h"

class GpuProfiler {
  public:
    static constexpr UINT MaxPasses = 512;
    static constexpr UINT InvalidPass = UINT_MAX;

    void Init(ID3D12Device* device, ID3D12CommandQueue* queue) {
        if (m_heap)
            return;
        D3D12_QUERY_HEAP_DESC queries{};
        queries.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
        queries.Count = 2 + 2 * MaxPasses;
        ThrowIfFailed(device->CreateQueryHeap(&queries, IID_PPV_ARGS(&m_heap)));
        const auto heap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK);
        const auto desc = CD3DX12_RESOURCE_DESC::Buffer(queries.Count * sizeof(UINT64));
        ThrowIfFailed(device->CreateCommittedResource(
            &heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&m_readback)));
        ThrowIfFailed(queue->GetTimestampFrequency(&m_frequency));
        m_spans.reserve(MaxPasses);
    }

    void BeginFrame(ID3D12GraphicsCommandList* cmd) {
        m_spans.clear();
        m_truncated = false;
        m_pending = false;
        cmd->EndQuery(m_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0);
    }

    UINT BeginPass(ID3D12GraphicsCommandList* cmd, std::string name, int cacheGroup = -1) {
        if (m_spans.size() == MaxPasses) {
            m_truncated = true;
            return InvalidPass;
        }
        const UINT id = static_cast<UINT>(m_spans.size());
        // A PIX-format marker per pass: DRED records it as the breadcrumb context of the
        // following commands, so a device removal dump names the pass the GPU was in.
        {
            const std::wstring wide(name.begin(), name.end());
            cmd->SetMarker(0u /* PIX_EVENT_UNICODE_VERSION */, wide.c_str(),
                           static_cast<UINT>((wide.size() + 1u) * sizeof(wchar_t)));
        }
        m_spans.push_back({std::move(name), cacheGroup});
        cmd->EndQuery(m_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 2 + id * 2);
        return id;
    }

    void EndPass(ID3D12GraphicsCommandList* cmd, UINT id) {
        if (id != InvalidPass)
            cmd->EndQuery(m_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 3 + id * 2);
    }

    void EndFrame(ID3D12GraphicsCommandList* cmd) {
        cmd->EndQuery(m_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 1);
        cmd->ResolveQueryData(m_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, 2 + 2 * static_cast<UINT>(m_spans.size()),
                              m_readback.Get(), 0);
        m_pending = true;
    }

    // The caller completes the render fence before reading or reusing queries.
    void Readback(FrameStats& stats) {
        stats.gpuPasses.clear();
        stats.gpuFrameMs = 0;
        stats.gpuTimingsValid = false;
        stats.gpuTimingsTruncated = false;
        stats.cacheTimingMask = 0;
        std::fill(std::begin(stats.cachePassMs), std::end(stats.cachePassMs), 0.0f);
        if (!m_pending || m_frequency == 0)
            return;

        void* mapped = nullptr;
        const D3D12_RANGE range{0, (2 + 2 * m_spans.size()) * sizeof(UINT64)};
        ThrowIfFailed(m_readback->Map(0, &range, &mapped));
        const auto* ticks = static_cast<const UINT64*>(mapped);
        const double milliseconds = 1000.0 / static_cast<double>(m_frequency);
        stats.gpuTimingsValid = ticks[1] >= ticks[0];
        if (stats.gpuTimingsValid)
            stats.gpuFrameMs = static_cast<float>((ticks[1] - ticks[0]) * milliseconds);
        stats.gpuTimingsTruncated = m_truncated;
        for (size_t i = 0; i < m_spans.size(); ++i) {
            const UINT64 begin = ticks[2 + 2 * i], end = ticks[3 + 2 * i];
            if (end < begin) {
                stats.gpuTimingsValid = false;
                continue;
            }
            const float ms = static_cast<float>((end - begin) * milliseconds);
            const auto& span = m_spans[i];
            auto found = std::find_if(stats.gpuPasses.begin(), stats.gpuPasses.end(),
                                      [&](const GpuPassTiming& p) { return p.name == span.name; });
            if (found == stats.gpuPasses.end())
                stats.gpuPasses.push_back({span.name, ms, 1});
            else {
                found->gpuMs += ms;
                ++found->calls;
            }
            if (span.cacheGroup >= 0 && span.cacheGroup < static_cast<int>(FrameStats::GpuTimingCount)) {
                stats.cachePassMs[span.cacheGroup] += ms;
                stats.cacheTimingMask |= 1u << span.cacheGroup;
            }
        }
        const D3D12_RANGE written{0, 0};
        m_readback->Unmap(0, &written);
        m_pending = false;
    }

  private:
    struct Span {
        std::string name;
        int cacheGroup;
    };
    ComPtr<ID3D12QueryHeap> m_heap;
    ComPtr<ID3D12Resource> m_readback;
    UINT64 m_frequency = 0;
    std::vector<Span> m_spans;
    bool m_pending = false, m_truncated = false;
};
