// ═══════════════════════════════════════════════════════════════════
// Core/DeviceContext.cpp
// ═══════════════════════════════════════════════════════════════════

#include "../stdafx.h"
#include "DeviceContext.h"
#include "../DXRHelper.h"
#include "../DXSample.h"
#include "../nv_helpers_dx12/BottomLevelASGenerator.h"
#include "../Diagnostics.h"

#define ENABLE_D3D12_DIAGNOSTICS 1

extern "C" {
    __declspec(dllexport) extern const UINT  D3D12SDKVersion = 719;
    __declspec(dllexport) extern const char* D3D12SDKPath    = ".\\";
}

#ifndef SL_CHECK
#define SL_CHECK(x) do { sl::Result r = (x); if (r != sl::Result::eOk) { \
    std::wcout << L"[SL] " << L#x << L" failed: " << (int)r << std::endl; } \
} while(0)
#endif

// ─────────────────────────────────────────────────────────────────
void DeviceContext::Init(HWND hwnd, UINT w, UINT h, bool useWarp) {
    width = w; height = h;
    slModule = LoadLibrary("sl.interposer.dll");
    InitStreamline();
    CreateDeviceAndSwapChain(hwnd, useWarp);

    // Query actual buffer count — DLSS-G may have added extra back buffers
    DXGI_SWAP_CHAIN_DESC scDesc{};
    swapChain->GetDesc(&scDesc);
    bufferCount = scDesc.BufferCount;
    std::wcout << L"[DX] Actual swap chain buffer count: " << bufferCount << std::endl;

    CreateRTVsAndDepth();

    for (UINT n = 0; n < bufferCount; ++n)
        ThrowIfFailed(device->CreateCommandAllocator(
            D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&cmdAllocators[n])));

    ThrowIfFailed(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
        cmdAllocators[frameIndex].Get(), nullptr, IID_PPV_ARGS(&cmdList)));

    ThrowIfFailed(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));
    for (UINT n = 0; n < bufferCount; ++n) fenceValues[n] = 0;
    fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    if (!fenceEvent) ThrowIfFailed(HRESULT_FROM_WIN32(GetLastError()));
}

void DeviceContext::Shutdown() {
    WaitForGPU();
    CloseHandle(fenceEvent);
    slShutdown();
}

// ─────────────────────────────────────────────────────────────────
void DeviceContext::WaitForPreviousFrame() {
    const UINT64 prevFence = nextFenceValue - 1;
    if (prevFence > 0 && fence->GetCompletedValue() < prevFence) {
        ThrowIfFailed(fence->SetEventOnCompletion(prevFence, fenceEvent));
        WaitForSingleObject(fenceEvent, INFINITE);
    }
}

void DeviceContext::BeginFrame() {
    auto* allocator = cmdAllocators[frameIndex].Get();
    ThrowIfFailed(allocator->Reset());
    ThrowIfFailed(cmdList->Reset(allocator, nullptr));

    if (frameToken) slGetNewFrameToken(frameToken, nullptr);
}

void DeviceContext::ExecuteAndPresent() {
    ThrowIfFailed(cmdList->Close());
    ID3D12CommandList* lists[] = { cmdList.Get() };
    cmdQueue->ExecuteCommandLists(1, lists);

#if ENABLE_D3D12_DIAGNOSTICS
    // Dump D3D messages before Present so we see errors from command list execution
    // even if DLSS-G's Present hook crashes
    dxdiag::DumpNewMessages();
#endif

    // Present — use ALLOW_TEARING when supported (DLSS-G's Present hook requires it)
    UINT presentFlags = tearingSupported ? DXGI_PRESENT_ALLOW_TEARING : 0;
    HRESULT presentHr = swapChain->Present(0, presentFlags);
    if (FAILED(presentHr)) {
        std::wcout << L"[DX] Present failed: 0x" << std::hex << presentHr << std::dec << std::endl;
#if ENABLE_D3D12_DIAGNOSTICS
        dxdiag::DumpNewMessages();
#endif
        ThrowIfFailed(presentHr);
    }

    // Signal AFTER Present — DLSS-G's Present hook may submit additional GPU work
    const UINT64 fv = nextFenceValue++;
    fenceValues[frameIndex] = fv;
    ThrowIfFailed(cmdQueue->Signal(fence.Get(), fv));

    // Advance to next back buffer (non-blocking)
    frameIndex = swapChain->GetCurrentBackBufferIndex();

#if ENABLE_D3D12_DIAGNOSTICS
    dxdiag::CheckDeviceRemoved(device.Get());
    dxdiag::DumpNewMessages();
#endif
}

void DeviceContext::WaitForGPU() {
    // Drain all in-flight work — used for shutdown and FlushAndReset
    const UINT64 fv = nextFenceValue++;
    ThrowIfFailed(cmdQueue->Signal(fence.Get(), fv));
    if (fence->GetCompletedValue() < fv) {
        ThrowIfFailed(fence->SetEventOnCompletion(fv, fenceEvent));
        WaitForSingleObject(fenceEvent, INFINITE);
    }
    frameIndex = swapChain->GetCurrentBackBufferIndex();
}

void DeviceContext::FlushAndReset() {
    ThrowIfFailed(cmdList->Close());
    ID3D12CommandList* lists[] = { cmdList.Get() };
    cmdQueue->ExecuteCommandLists(1, lists);
    WaitForGPU();
    ThrowIfFailed(cmdAllocators[frameIndex]->Reset());
    ThrowIfFailed(cmdList->Reset(cmdAllocators[frameIndex].Get(), nullptr));
}

void DeviceContext::Resize(UINT newWidth, UINT newHeight) {
    if (newWidth == 0 || newHeight == 0) return;
    if (newWidth == width && newHeight == height) return;

    WaitForGPU();

    // Release back buffer references before ResizeBuffers
    for (UINT n = 0; n < bufferCount; ++n)
        renderTargets[n].Reset();
    depthStencil.Reset();

    width  = newWidth;
    height = newHeight;

    // Pass 0 for buffer count to keep the existing count (DLSS-G may have added extra buffers)
    UINT resizeFlags = tearingSupported ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0;
    ThrowIfFailed(swapChain->ResizeBuffers(
        0, width, height, DXGI_FORMAT_R8G8B8A8_UNORM, resizeFlags));
    frameIndex = swapChain->GetCurrentBackBufferIndex();

    // Re-query actual buffer count in case it changed
    DXGI_SWAP_CHAIN_DESC scDesc{};
    swapChain->GetDesc(&scDesc);
    bufferCount = scDesc.BufferCount;

    // Recreate RTVs
    CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHandle(rtvHeap->GetCPUDescriptorHandleForHeapStart());
    for (UINT n = 0; n < bufferCount; n++) {
        ThrowIfFailed(swapChain->GetBuffer(n, IID_PPV_ARGS(&renderTargets[n])));
        device->CreateRenderTargetView(renderTargets[n].Get(), nullptr, rtvHandle);
        rtvHandle.Offset(1, rtvDescriptorSize);
    }

    // Recreate depth stencil
    auto hp  = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
    auto drd = CD3DX12_RESOURCE_DESC::Tex2D(DXGI_FORMAT_D32_FLOAT, width, height, 1, 1);
    drd.Flags |= D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    CD3DX12_CLEAR_VALUE cv(DXGI_FORMAT_D32_FLOAT, 1.0f, 0);
    ThrowIfFailed(device->CreateCommittedResource(
        &hp, D3D12_HEAP_FLAG_NONE, &drd, D3D12_RESOURCE_STATE_DEPTH_WRITE,
        &cv, IID_PPV_ARGS(&depthStencil)));

    D3D12_DEPTH_STENCIL_VIEW_DESC dsvDesc = {};
    dsvDesc.Format        = DXGI_FORMAT_D32_FLOAT;
    dsvDesc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
    device->CreateDepthStencilView(depthStencil.Get(), &dsvDesc,
        dsvHeap->GetCPUDescriptorHandleForHeapStart());
}

// ─────────────────────────────────────────────────────────────────
D3D12_CPU_DESCRIPTOR_HANDLE DeviceContext::CurrentRTV() const {
    return CD3DX12_CPU_DESCRIPTOR_HANDLE(
        rtvHeap->GetCPUDescriptorHandleForHeapStart(), frameIndex, rtvDescriptorSize);
}

D3D12_CPU_DESCRIPTOR_HANDLE DeviceContext::DSV() const {
    return dsvHeap->GetCPUDescriptorHandleForHeapStart();
}

// ─────────────────────────────────────────────────────────────────
void DeviceContext::InitStreamline() {
    sl::Preferences pref{};
    pref.renderAPI       = sl::RenderAPI::eD3D12;
    pref.engine          = sl::EngineType::eCustom;
    pref.applicationId   = 231313132;
    pref.showConsole     = true;
    pref.logLevel        = sl::LogLevel::eVerbose;
    pref.flags           = sl::PreferenceFlags::eLoadDownloadedPlugins
                         | sl::PreferenceFlags::eUseFrameBasedResourceTagging;

    static sl::Feature featList[] = { sl::kFeatureDLSS, sl::kFeatureDLSS_RR, sl::kFeatureDLSS_G, sl::kFeatureReflex, sl::kFeaturePCL };
    pref.featuresToLoad    = featList;
    pref.numFeaturesToLoad = _countof(featList);

    sl::Result res = slInit(pref, sl::kSDKVersion);
    if (res != sl::Result::eOk)
        std::wcout << L"slInit failed! Error: " << (int)res << std::endl;
}

// ─────────────────────────────────────────────────────────────────
void DeviceContext::CreateDeviceAndSwapChain(HWND hwnd, bool useWarp) {
    typedef HRESULT(WINAPI* PFunCreateDXGIFactory2)(UINT, REFIID, void**);
    typedef HRESULT(WINAPI* PFunD3D12CreateDevice)(IUnknown*, D3D_FEATURE_LEVEL, REFIID, void**);

    auto slCreateFactory = reinterpret_cast<PFunCreateDXGIFactory2>(
        GetProcAddress(slModule, "CreateDXGIFactory2"));
    auto slCreateDevice  = reinterpret_cast<PFunD3D12CreateDevice>(
        GetProcAddress(slModule, "D3D12CreateDevice"));

    ComPtr<IDXGIFactory4> factory;
    ThrowIfFailed(slCreateFactory(0, IID_PPV_ARGS(&factory)));

    // Check tearing support (required by DLSS-G's Present hook)
    {
        ComPtr<IDXGIFactory5> factory5;
        if (SUCCEEDED(factory.As(&factory5))) {
            BOOL allowTearing = FALSE;
            factory5->CheckFeatureSupport(DXGI_FEATURE_PRESENT_ALLOW_TEARING,
                &allowTearing, sizeof(allowTearing));
            tearingSupported = (allowTearing == TRUE);
        }
        std::wcout << L"[DX] Tearing support: " << (tearingSupported ? L"yes" : L"no") << std::endl;
    }

#if ENABLE_D3D12_DIAGNOSTICS
    dxdiag::EnableDebugLayerAndDred();
#endif

    if (useWarp) {
        ComPtr<IDXGIAdapter> warpAdapter;
        ThrowIfFailed(factory->EnumWarpAdapter(IID_PPV_ARGS(&warpAdapter)));
        ThrowIfFailed(slCreateDevice(warpAdapter.Get(),
            D3D_FEATURE_LEVEL_12_1, IID_PPV_ARGS(&device)));
    } else {
        ComPtr<IDXGIAdapter1> hardwareAdapter;
        // Inline adapter selection (equivalent to DXSample::GetHardwareAdapter)
        for (UINT i = 0; factory->EnumAdapters1(i, &hardwareAdapter) != DXGI_ERROR_NOT_FOUND; ++i) {
            DXGI_ADAPTER_DESC1 desc;
            hardwareAdapter->GetDesc1(&desc);
            if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
            if (SUCCEEDED(slCreateDevice(hardwareAdapter.Get(),
                D3D_FEATURE_LEVEL_12_1, _uuidof(ID3D12Device), nullptr)))
                break;
        }
        ThrowIfFailed(slCreateDevice(hardwareAdapter.Get(),
            D3D_FEATURE_LEVEL_12_1, IID_PPV_ARGS(&device)));
#if ENABLE_D3D12_DIAGNOSTICS
        dxdiag::HookDevice(device.Get());
#endif
    }

    sl::Result res = slSetD3DDevice(device.Get());
    if (res != sl::Result::eOk)
        std::wcout << L"slSetD3DDevice failed: " << (int)res << std::endl;

    // Check DLSS-RR support
    LUID luid = device->GetAdapterLuid();
    sl::AdapterInfo ai;
    ai.deviceLUID = (uint8_t*)&luid;
    ai.deviceLUIDSizeInBytes = sizeof(LUID);
    sl::Result sr = slIsFeatureSupported(sl::kFeatureDLSS_RR, ai);
    if (sr != sl::Result::eOk)
        std::wcout << L"[DLSS-RR] not supported: " << (int)sr << std::endl;

    // Command queue
    D3D12_COMMAND_QUEUE_DESC qd = {};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    ThrowIfFailed(device->CreateCommandQueue(&qd, IID_PPV_ARGS(&cmdQueue)));

    // Swap chain
    DXGI_SWAP_CHAIN_DESC1 scd = {};
    scd.BufferCount  = FRAME_COUNT;
    scd.Width        = width;
    scd.Height       = height;
    scd.Format       = DXGI_FORMAT_R8G8B8A8_UNORM;
    scd.BufferUsage  = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    scd.SwapEffect   = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    scd.SampleDesc.Count = 1;
    if (tearingSupported)
        scd.Flags |= DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING;

    ComPtr<IDXGISwapChain1> sc1;
    ThrowIfFailed(factory->CreateSwapChainForHwnd(
        cmdQueue.Get(), hwnd, &scd, nullptr, nullptr, &sc1));
    ThrowIfFailed(factory->MakeWindowAssociation(hwnd, DXGI_MWA_NO_ALT_ENTER));
    ThrowIfFailed(sc1.As(&swapChain));
    frameIndex = swapChain->GetCurrentBackBufferIndex();

    slGetNewFrameToken(frameToken, nullptr);
}

// ─────────────────────────────────────────────────────────────────
void DeviceContext::CreateRTVsAndDepth() {
    // RTV heap
    D3D12_DESCRIPTOR_HEAP_DESC rtvDesc = {};
    rtvDesc.NumDescriptors = bufferCount;
    rtvDesc.Type  = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    ThrowIfFailed(device->CreateDescriptorHeap(&rtvDesc, IID_PPV_ARGS(&rtvHeap)));
    rtvDescriptorSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

    CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHandle(rtvHeap->GetCPUDescriptorHandleForHeapStart());
    for (UINT n = 0; n < bufferCount; n++) {
        ThrowIfFailed(swapChain->GetBuffer(n, IID_PPV_ARGS(&renderTargets[n])));
        device->CreateRenderTargetView(renderTargets[n].Get(), nullptr, rtvHandle);
        rtvHandle.Offset(1, rtvDescriptorSize);
    }

    // Depth
    dsvHeap = nv_helpers_dx12::CreateDescriptorHeap(device.Get(), 1,
        D3D12_DESCRIPTOR_HEAP_TYPE_DSV, false);

    auto hp  = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
    auto drd = CD3DX12_RESOURCE_DESC::Tex2D(DXGI_FORMAT_D32_FLOAT, width, height, 1, 1);
    drd.Flags |= D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    CD3DX12_CLEAR_VALUE cv(DXGI_FORMAT_D32_FLOAT, 1.0f, 0);
    ThrowIfFailed(device->CreateCommittedResource(
        &hp, D3D12_HEAP_FLAG_NONE, &drd, D3D12_RESOURCE_STATE_DEPTH_WRITE,
        &cv, IID_PPV_ARGS(&depthStencil)));

    D3D12_DEPTH_STENCIL_VIEW_DESC dsvDesc = {};
    dsvDesc.Format        = DXGI_FORMAT_D32_FLOAT;
    dsvDesc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
    device->CreateDepthStencilView(depthStencil.Get(), &dsvDesc,
        dsvHeap->GetCPUDescriptorHandleForHeapStart());
}
