#pragma once

#include "../Common.h"
#include <dxgi1_4.h>
#include <dxgi1_5.h>

#include <sl.h>
#include <sl_consts.h>
#include <sl_helpers.h>
#include <sl_dlss.h>
#include <sl_dlss_g.h>

#include "../planet/queue_sync.h"

struct DeviceContext {
    void Init(HWND hwnd, UINT width, UINT height, bool useWarp = false);
    void Shutdown();

    void BeginFrame();

    void ExecuteAndPresent();

    void WaitForPreviousFrame();

    void WaitForGPU(); // Synchronizes all queues.

    void FlushAndReset();

    void Resize(UINT newWidth, UINT newHeight);

    ID3D12GraphicsCommandList10* ComputeList() const { return planetComputeList.Get(); }
    ID3D12GraphicsCommandList10* CopyList() const { return planetCopyList.Get(); }

    void ResetPlanetLists();

    UINT64 SubmitPlanetCopy();

    UINT64 SubmitPlanetCompute(UINT64 waitCopyValue);
    UINT64 PlanetComputeCompleted() const;
    UINT64 PlanetComputeLastSignaled() const;
    void PlanetCopyCpuWait(UINT64 value);
    void PlanetComputeCpuWait(UINT64 value);
    ID3D12CommandQueue* PlanetComputeQueue() const { return planetComputeQueue.Get(); }

    ID3D12Device10* Device() const { return device.Get(); }
    // Streamline hands out a proxy device; removal state and DRED data live on the native one.
    ID3D12Device* NativeDevice() const { return nativeDevice ? nativeDevice.Get() : device.Get(); }
    ID3D12GraphicsCommandList10* CmdList() const { return cmdList.Get(); }
    ID3D12CommandQueue* CmdQueue() const { return cmdQueue.Get(); }
    IDXGISwapChain3* SwapChain() const { return swapChain.Get(); }
    ID3D12Resource* BackBuffer() const { return renderTargets[frameIndex].Get(); }
    D3D12_CPU_DESCRIPTOR_HANDLE CurrentRTV() const;
    D3D12_CPU_DESCRIPTOR_HANDLE DSV() const;
    UINT FrameIndex() const { return frameIndex; }
    UINT BufferCount() const { return bufferCount; }
    UINT Width() const { return width; }
    UINT Height() const { return height; }
    float AspectRatio() const { return (float)width / (float)height; }

    sl::FrameToken* frameToken = nullptr;
    sl::ViewportHandle viewportHandle = sl::ViewportHandle(0);
    HINSTANCE__* slModule = nullptr;

    ComPtr<ID3D12Device10> device;

    ComPtr<ID3D12Device> nativeDevice;
    ComPtr<ID3D12GraphicsCommandList10> cmdList;
    ComPtr<ID3D12CommandQueue> cmdQueue;

  private:
    void CreateDeviceAndSwapChain(HWND hwnd, bool useWarp);
    void CreateRTVsAndDepth();
    void InitStreamline();
    void InitPlanetStreaming();

    UINT width = 0;
    UINT height = 0;

    ComPtr<IDXGISwapChain3> swapChain;
    ComPtr<ID3D12CommandAllocator> cmdAllocators[MAX_BACK_BUFFERS];
    ComPtr<ID3D12Resource> renderTargets[MAX_BACK_BUFFERS];

    ComPtr<ID3D12DescriptorHeap> rtvHeap;
    UINT rtvDescriptorSize = 0;

    ComPtr<ID3D12DescriptorHeap> dsvHeap;
    ComPtr<ID3D12Resource> depthStencil;

    ComPtr<ID3D12Fence> fence;
    UINT64 fenceValues[MAX_BACK_BUFFERS] = {};
    UINT64 nextFenceValue = 1;
    HANDLE fenceEvent = nullptr;
    UINT frameIndex = 0;

    UINT bufferCount = FRAME_COUNT;
    bool tearingSupported = false;

    ComPtr<ID3D12CommandQueue> planetComputeQueue;
    ComPtr<ID3D12CommandQueue> planetCopyQueue;
    ComPtr<ID3D12CommandAllocator> planetComputeAllocators[MAX_BACK_BUFFERS];
    ComPtr<ID3D12CommandAllocator> planetCopyAllocators[MAX_BACK_BUFFERS];
    ComPtr<ID3D12GraphicsCommandList10> planetComputeList;
    ComPtr<ID3D12GraphicsCommandList10> planetCopyList;
    planet::FenceTimeline planetComputeFence;
    planet::FenceTimeline planetCopyFence;
    UINT64 planetComputeAtSlot[MAX_BACK_BUFFERS] = {};
    UINT64 planetCopyAtSlot[MAX_BACK_BUFFERS] = {};
};
