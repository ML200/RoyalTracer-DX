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

// A device for a host process (the Unity plugin): no window or swap chain of its own, its own D3D12Core loaded side by
// side through ID3D12DeviceFactory, Streamline in manual-hooking mode with DLSS-RR, and DLSS frame generation and Reflex
// for when it presents through a window of the host's (AttachWindow).
struct HeadlessDeviceDesc {
    LUID adapterLuid{};      // the host device's adapter; zero picks the first hardware adapter
    UINT sdkVersion = 619;   // Agility SDK version of the D3D12Core.dll in sdkPath
    std::string sdkPath;     // folder holding that D3D12Core.dll, with a trailing separator
    std::wstring pluginDir;  // folder holding sl.interposer.dll and the Streamline plugins
    UINT width = 0;
    UINT height = 0;
    sl::PFun_LogMessageCallback* slLog = nullptr;
};

// What InitHeadless found on the adapter; filled as far as it got, also when it throws.
struct AdapterReport {
    std::wstring name;
    UINT vendorId = 0;
    UINT64 dedicatedVideoMemory = 0;
    D3D_SHADER_MODEL shaderModel = {};
    D3D12_RAYTRACING_TIER raytracingTier = D3D12_RAYTRACING_TIER_NOT_SUPPORTED;
    bool serReorders = false;
    bool dlssRayReconstruction = false;
};

struct DeviceContext {
    void Init(HWND hwnd, UINT width, UINT height, bool useWarp = false);
    void InitHeadless(const HeadlessDeviceDesc& desc);
    bool IsHeadless() const { return headless; }
    // Headless: presents through this window (a child of the host's) instead of the hidden swap chain, so Streamline can
    // generate frames there. Throws on failure.
    void AttachWindow(HWND hwnd, UINT width, UINT height);
    void DetachWindow();
    IDXGISwapChain3* WindowSwapChain() const { return windowSwapChain.Get(); }
    UINT WindowBufferCount() const { return 3; }
    bool WindowTearing() const { return windowTearing; }
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
    void PlanetComputeCpuWait(UINT64 value);
    ID3D12CommandQueue* PlanetComputeQueue() const { return planetComputeQueue.Get(); }

    ID3D12Device10* Device() const { return device.Get(); }
    // Non-proxied device, for removal state and DRED.
    ID3D12Device* NativeDevice() const { return nativeDevice ? nativeDevice.Get() : device.Get(); }
    ID3D12GraphicsCommandList10* CmdList() const { return cmdList.Get(); }
    ID3D12CommandQueue* CmdQueue() const { return cmdQueue.Get(); }
    ID3D12Resource* BackBuffer() const { return renderTargets[frameIndex].Get(); }
    D3D12_CPU_DESCRIPTOR_HANDLE CurrentRTV() const;
    D3D12_CPU_DESCRIPTOR_HANDLE DSV() const;
    UINT FrameIndex() const { return frameIndex; }
    UINT BufferCount() const { return bufferCount; }
    UINT Width() const { return width; }
    UINT Height() const { return height; }
    // Clustered geometry and partitioned TLAS (Agility 1.722 preview, SM 6.10); windowed path only.
    bool ClusteredGeometry() const { return clusteredGeometry; }

    sl::FrameToken* frameToken = nullptr;
    sl::ViewportHandle viewportHandle = sl::ViewportHandle(0);
    HINSTANCE__* slModule = nullptr;
    AdapterReport adapterReport; // headless only

    ComPtr<ID3D12Device10> device;

    ComPtr<ID3D12Device> nativeDevice;
    ComPtr<ID3D12GraphicsCommandList10> cmdList;
    ComPtr<ID3D12CommandQueue> cmdQueue;

  private:
    void CreateDeviceAndSwapChain(HWND hwnd, bool useWarp);
    void CreateRTVsAndDepth();
    void InitStreamline();
    void InitPlanetStreaming();
    void CreateFrameObjects();
    void CreatePresentHook();
    template <class T> ComPtr<T> UpgradeSwapChain(T* native);

    UINT width = 0;
    UINT height = 0;
    bool headless = false;
    bool clusteredGeometry = false;

    ComPtr<IDXGISwapChain3> swapChain;
    // Headless: a tiny composition swap chain that is never shown. Presenting it through Streamline's proxy runs
    // the per-frame bookkeeping Streamline otherwise does in the app's Present.
    ComPtr<IDXGISwapChain1> slPresentHook;
    ComPtr<IDXGIFactory4> headlessFactory;
    ComPtr<IDXGISwapChain3> windowSwapChain; // Streamline's proxy
    bool windowTearing = false;
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
