#include <windows.h>
#include <d3dx12.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../rdn/Core/GpuProfiler.h"

using Microsoft::WRL::ComPtr;

static void Check(HRESULT hr) {
    if (FAILED(hr))
        throw std::runtime_error("D3D12 operation failed");
}

static void Require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

static D3D12_RESOURCE_DESC BufferDesc(UINT64 bytes) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = bytes;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    return desc;
}

struct Runner {
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> commands;
    ComPtr<ID3D12Fence> fence;
    ComPtr<ID3D12Resource> source;
    ComPtr<ID3D12Resource> destination;
    HANDLE fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    UINT64 fenceValue = 0;
    GpuProfiler profiler;
    FrameStats stats;

    Runner() {
        Require(fenceEvent != nullptr, "Fence event creation failed");

        D3D12_COMMAND_QUEUE_DESC queueDesc{};
        Check(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device)));
        Check(device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&queue)));
        Check(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)));
        Check(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr,
                                        IID_PPV_ARGS(&commands)));
        Check(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));

        const auto uploadHeap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
        const auto defaultHeap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
        const auto desc = BufferDesc(4096);
        Check(device->CreateCommittedResource(&uploadHeap, D3D12_HEAP_FLAG_NONE, &desc,
                                              D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                              IID_PPV_ARGS(&source)));
        Check(device->CreateCommittedResource(&defaultHeap, D3D12_HEAP_FLAG_NONE, &desc,
                                              D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                              IID_PPV_ARGS(&destination)));
        void* mapped = nullptr;
        Check(source->Map(0, nullptr, &mapped));
        memset(mapped, 0x5a, 4096);
        source->Unmap(0, nullptr);
        profiler.Init(device.Get(), queue.Get());
    }

    ~Runner() {
        if (fenceEvent)
            CloseHandle(fenceEvent);
    }

    void CopyWork() {
        commands->CopyBufferRegion(destination.Get(), 0, source.Get(), 0, 4096);
    }

    void SubmitAndWait() {
        Check(commands->Close());
        ID3D12CommandList* lists[] = {commands.Get()};
        queue->ExecuteCommandLists(1, lists);
        Check(queue->Signal(fence.Get(), ++fenceValue));
        Check(fence->SetEventOnCompletion(fenceValue, fenceEvent));
        Require(WaitForSingleObject(fenceEvent, 30000) == WAIT_OBJECT_0, "GPU timed out");
        Check(allocator->Reset());
        Check(commands->Reset(allocator.Get(), nullptr));
    }

    void TestTimings() {
        profiler.BeginFrame(commands.Get());
        const UINT copy0 = profiler.BeginPass(commands.Get(), "Copy", 0);
        CopyWork();
        profiler.EndPass(commands.Get(), copy0);
        const UINT copy1 = profiler.BeginPass(commands.Get(), "Copy", 0);
        CopyWork();
        profiler.EndPass(commands.Get(), copy1);
        const UINT dispatchA = profiler.BeginPass(commands.Get(), "Dispatch A");
        CopyWork();
        profiler.EndPass(commands.Get(), dispatchA);
        const UINT dispatchB = profiler.BeginPass(commands.Get(), "Dispatch B");
        CopyWork();
        profiler.EndPass(commands.Get(), dispatchB);
        profiler.EndFrame(commands.Get());
        SubmitAndWait();

        profiler.Readback(stats);
        Require(stats.gpuTimingsValid, "Timestamp queries were invalid");
        Require(std::isfinite(stats.gpuFrameMs) && stats.gpuFrameMs >= 0, "Frame time was invalid");
        Require(stats.gpuPasses.size() == 3, "Pass names were not aggregated correctly");
        const GpuPassTiming* copy = nullptr;
        const GpuPassTiming* dispatchAStats = nullptr;
        const GpuPassTiming* dispatchBStats = nullptr;
        for (const auto& pass : stats.gpuPasses) {
            if (pass.name == "Copy") copy = &pass;
            if (pass.name == "Dispatch A") dispatchAStats = &pass;
            if (pass.name == "Dispatch B") dispatchBStats = &pass;
        }
        Require(copy && copy->calls == 2, "Repeated pass calls were not aggregated");
        Require(dispatchAStats && dispatchAStats->calls == 1, "First dispatch label was lost");
        Require(dispatchBStats && dispatchBStats->calls == 1, "Second dispatch label was lost");
        const float summedMs = copy->gpuMs + dispatchAStats->gpuMs + dispatchBStats->gpuMs;
        const float tolerance = std::max(1e-4f, stats.gpuFrameMs * 0.001f);
        Require(stats.gpuFrameMs + tolerance >= summedMs, "Nested pass times exceeded frame time");
        Require((stats.cacheTimingMask & 1u) != 0 && stats.cachePassMs[0] >= 0,
                "Cache timing was not recorded");
    }

    void TestEmptyFrame() {
        profiler.BeginFrame(commands.Get());
        profiler.EndFrame(commands.Get());
        SubmitAndWait();
        profiler.Readback(stats);
        Require(stats.gpuTimingsValid, "Empty frame timestamps were invalid");
        Require(stats.gpuPasses.empty(), "Empty frame retained old passes");
        Require(stats.cacheTimingMask == 0, "Empty frame retained cache timing masks");
        Require(std::isfinite(stats.gpuFrameMs) && stats.gpuFrameMs >= 0, "Empty frame time was invalid");
    }

    void TestOverflow() {
        profiler.BeginFrame(commands.Get());
        for (UINT i = 0; i < GpuProfiler::MaxPasses + 1; ++i) {
            const UINT id = profiler.BeginPass(commands.Get(), "Overflow");
            if (id != GpuProfiler::InvalidPass)
                CopyWork();
            profiler.EndPass(commands.Get(), id);
        }
        profiler.EndFrame(commands.Get());
        SubmitAndWait();
        profiler.Readback(stats);
        Require(stats.gpuTimingsTruncated, "MaxPasses overflow was not reported");
        Require(stats.gpuPasses.size() == 1 && stats.gpuPasses[0].calls == GpuProfiler::MaxPasses,
                "MaxPasses overflow produced invalid pass queries");
    }
};

static void CheckDebugMessages(ID3D12Device* device, SIZE_T baseline) {
    ComPtr<ID3D12InfoQueue> info;
    if (FAILED(device->QueryInterface(IID_PPV_ARGS(&info))))
        return;
    const SIZE_T count = info->GetNumStoredMessages();
    for (SIZE_T i = baseline; i < count; ++i) {
        SIZE_T size = 0;
        Check(info->GetMessage(i, nullptr, &size));
        std::vector<uint8_t> storage(size);
        auto* message = reinterpret_cast<D3D12_MESSAGE*>(storage.data());
        Check(info->GetMessage(i, message, &size));
        Require(message->Severity != D3D12_MESSAGE_SEVERITY_ERROR &&
                    message->Severity != D3D12_MESSAGE_SEVERITY_CORRUPTION,
                "D3D12 debug layer reported an error");
    }
}

int main() {
    try {
        ComPtr<ID3D12Debug> debug;
        const bool debugEnabled = SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)));
        if (debugEnabled)
            debug->EnableDebugLayer();
        Runner runner;
        ComPtr<ID3D12InfoQueue> info;
        const SIZE_T baseline = SUCCEEDED(runner.device->QueryInterface(IID_PPV_ARGS(&info)))
                                    ? info->GetNumStoredMessages()
                                    : 0;
        runner.TestTimings();
        runner.TestEmptyFrame();
        runner.TestOverflow();
        if (debugEnabled)
            CheckDebugMessages(runner.device.Get(), baseline);
        std::cout << "PASS: GPU profiler timestamps, aggregation, cache reset, and MaxPasses truncation.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
