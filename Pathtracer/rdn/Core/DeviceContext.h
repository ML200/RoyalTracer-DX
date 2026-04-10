#pragma once
// ═══════════════════════════════════════════════════════════════════
// Core/DeviceContext.h — Owns the D3D12 device, swap chain, and
//                        command infrastructure. Single source of
//                        truth for anything device-level.
// ═══════════════════════════════════════════════════════════════════

#include "../Common.h"
#include <dxgi1_4.h>
#include <dxgi1_5.h>

#include <sl.h>
#include <sl_consts.h>
#include <sl_helpers.h>
#include <sl_dlss.h>
#include <sl_dlss_g.h>

struct DeviceContext {
    // ── Creation ─────────────────────────────────────────────────
    void Init(HWND hwnd, UINT width, UINT height, bool useWarp = false);
    void Shutdown();

    // ── Frame management ─────────────────────────────────────────
    void BeginFrame();                     // resets allocator, opens command list
    void ExecuteAndPresent();              // closes list, executes, presents, signals fence
    void WaitForPreviousFrame();           // wait for last submitted frame (call before writing shared buffers)
    void WaitForGPU();                     // CPU stall until GPU is idle
    void FlushAndReset();                  // execute current list, wait, reopen
    void Resize(UINT newWidth, UINT newHeight);  // resize swap chain, RTVs, depth

    // ── Accessors ────────────────────────────────────────────────
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

    // Streamline
    sl::FrameToken*     frameToken    = nullptr;
    sl::ViewportHandle  viewportHandle = sl::ViewportHandle(0);
    HINSTANCE__*        slModule      = nullptr;

    // ── Public members (for modules that need raw access) ────────
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

    // RTV heap
    ComPtr<ID3D12DescriptorHeap>   rtvHeap;
    UINT                           rtvDescriptorSize = 0;

    // Depth
    ComPtr<ID3D12DescriptorHeap>   dsvHeap;
    ComPtr<ID3D12Resource>         depthStencil;

    // Sync
    ComPtr<ID3D12Fence>            fence;
    UINT64                         fenceValues[MAX_BACK_BUFFERS] = {};  // per-slot: last fence value signaled
    UINT64                         nextFenceValue = 1;                  // monotonically increasing
    HANDLE                         fenceEvent = nullptr;
    UINT                           frameIndex = 0;
    UINT                           bufferCount = FRAME_COUNT;           // actual swap chain buffer count (may differ from FRAME_COUNT with DLSS-G)
    bool                           tearingSupported = false;
};
