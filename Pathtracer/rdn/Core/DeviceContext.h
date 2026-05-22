#pragma once
//====================================
//DEVICE CONTEXT
//====================================
//owns D3D12 device, swap chain, command infra, single source of truth

#include "../Common.h"
#include <dxgi1_4.h>
#include <dxgi1_5.h>

#include <sl.h>
#include <sl_consts.h>
#include <sl_helpers.h>
#include <sl_dlss.h>
#include <sl_dlss_g.h>

#include "../planet/queue_sync.h"   // planet::FenceTimeline

struct DeviceContext {
    // PLANET_INTEGRATION: single source of truth for the ID3D12Device, the one
    // DIRECT command queue, the per-frame fence, and frames-in-flight (bufferCount).
    // Planet BVH streaming (Phase 4) adds an async COMPUTE queue, a COPY queue, and
    // their fences here. See rdn/planet/INTEGRATION_NOTES.md.
    //====================================
    //CREATION
    //====================================
    void Init(HWND hwnd, UINT width, UINT height, bool useWarp = false);
    void Shutdown();

    //====================================
    //FRAME MANAGEMENT
    //====================================
    //resets allocator, opens command list
    void BeginFrame();
    //closes list, executes, presents, signals fence
    void ExecuteAndPresent();
    //CPU wait for last submitted frame, call before writing shared buffers
    void WaitForPreviousFrame();
    //CPU stall until GPU idle
    void WaitForGPU();
    //execute current list, wait, reopen
    void FlushAndReset();
    //resize swap chain, RTVs, depth
    void Resize(UINT newWidth, UINT newHeight);

    //split-submission for mid-frame interop, close+submit+signal, no CPU wait
    void CloseExecuteAndSignal(ID3D12Fence* extFence, UINT64 value);
    //queue-side wait, then reopen fresh cmd list on current frame allocator
    void WaitAndReopen(ID3D12Fence* extFence, UINT64 value);

    //====================================
    //PLANET STREAMING (Phase 4)
    //====================================
    //async COMPUTE + COPY command lists for the planet BVH stream pipeline
    ID3D12GraphicsCommandList10* ComputeList() const { return planetComputeList.Get(); }
    ID3D12GraphicsCommandList10* CopyList()    const { return planetCopyList.Get(); }
    //reset the planet compute+copy allocators/lists for the current frame
    void   ResetPlanetLists();
    //close + execute the copy list on the COPY queue; signal + return the copy fence
    UINT64 SubmitPlanetCopy();
    //COMPUTE queue waits the copy fence, then close + execute + signal compute
    UINT64 SubmitPlanetCompute(UINT64 waitCopyValue);
    UINT64 PlanetComputeCompleted()    const;
    UINT64 PlanetComputeLastSignaled() const;
    void   PlanetCopyCpuWait(UINT64 value);
    void   PlanetComputeCpuWait(UINT64 value);
    ID3D12CommandQueue* PlanetComputeQueue() const { return planetComputeQueue.Get(); }

    //====================================
    //ACCESSORS
    //====================================
    ID3D12Device10*                Device()       const { return device.Get(); }
    ID3D12GraphicsCommandList10*   CmdList()      const { return cmdList.Get(); }
    ID3D12CommandQueue*            CmdQueue()     const { return cmdQueue.Get(); }
    IDXGISwapChain3*               SwapChain()    const { return swapChain.Get(); }
    ID3D12Resource*                BackBuffer()   const { return renderTargets[frameIndex].Get(); }
    D3D12_CPU_DESCRIPTOR_HANDLE    CurrentRTV()   const;
    D3D12_CPU_DESCRIPTOR_HANDLE    DSV()          const;
    UINT                           FrameIndex()   const { return frameIndex; }
    UINT                           BufferCount()  const { return bufferCount; }
    UINT                           Width()        const { return width; }
    UINT                           Height()       const { return height; }
    float                          AspectRatio()  const { return (float)width / (float)height; }

    //Streamline
    sl::FrameToken*     frameToken    = nullptr;
    sl::ViewportHandle  viewportHandle = sl::ViewportHandle(0);
    HINSTANCE__*        slModule      = nullptr;

    //====================================
    //PUBLIC RAW MEMBERS
    //====================================
    ComPtr<ID3D12Device10>               device;
    ComPtr<ID3D12GraphicsCommandList10>  cmdList;
    ComPtr<ID3D12CommandQueue>           cmdQueue;

private:
    void CreateDeviceAndSwapChain(HWND hwnd, bool useWarp);
    void CreateRTVsAndDepth();
    void InitStreamline();
    void InitPlanetStreaming();

    UINT  width  = 0;
    UINT  height = 0;

    ComPtr<IDXGISwapChain3>        swapChain;
    ComPtr<ID3D12CommandAllocator> cmdAllocators[MAX_BACK_BUFFERS];
    ComPtr<ID3D12Resource>         renderTargets[MAX_BACK_BUFFERS];

    ComPtr<ID3D12DescriptorHeap>   rtvHeap;
    UINT                           rtvDescriptorSize = 0;

    ComPtr<ID3D12DescriptorHeap>   dsvHeap;
    ComPtr<ID3D12Resource>         depthStencil;

    //sync
    ComPtr<ID3D12Fence>            fence;
    UINT64                         fenceValues[MAX_BACK_BUFFERS] = {};
    UINT64                         nextFenceValue = 1;
    HANDLE                         fenceEvent = nullptr;
    UINT                           frameIndex = 0;
    //buffer count may differ from FRAME_COUNT with DLSS-G
    UINT                           bufferCount = FRAME_COUNT;
    bool                           tearingSupported = false;

    //planet streaming (Phase 4): async COMPUTE + COPY queues, per-frame
    //allocators/lists, and a copy -> compute fence pair.
    ComPtr<ID3D12CommandQueue>          planetComputeQueue;
    ComPtr<ID3D12CommandQueue>          planetCopyQueue;
    ComPtr<ID3D12CommandAllocator>      planetComputeAllocators[MAX_BACK_BUFFERS];
    ComPtr<ID3D12CommandAllocator>      planetCopyAllocators[MAX_BACK_BUFFERS];
    ComPtr<ID3D12GraphicsCommandList10> planetComputeList;
    ComPtr<ID3D12GraphicsCommandList10> planetCopyList;
    planet::FenceTimeline               planetComputeFence;
    planet::FenceTimeline               planetCopyFence;
    UINT64                              planetComputeAtSlot[MAX_BACK_BUFFERS] = {};
    UINT64                              planetCopyAtSlot[MAX_BACK_BUFFERS] = {};
};
