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

struct DeviceContext {
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
};
