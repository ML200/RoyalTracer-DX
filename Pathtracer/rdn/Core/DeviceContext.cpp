#include "../stdafx.h"
#include "DeviceContext.h"
#include "../DXRHelper.h"
#include "../DXSample.h"
#include "../nv_helpers_dx12/BottomLevelASGenerator.h"

#define ENABLE_D3D12_DIAGNOSTICS 1
#include "../Diagnostics.h"

#ifndef RT_HOSTED
extern "C" {
__declspec(dllexport) extern const UINT D3D12SDKVersion = 722;
__declspec(dllexport) extern const char* D3D12SDKPath = ".\\";
}
#endif

#ifndef SL_CHECK
#define SL_CHECK(x)                                                                                                    \
    do {                                                                                                               \
        sl::Result r = (x);                                                                                            \
        if (r != sl::Result::eOk) {                                                                                    \
            std::wcout << L"[SL] " << L#x << L" failed: " << (int)r << std::endl;                                      \
        }                                                                                                              \
    } while (0)
#endif

#ifndef SL_VERBOSE_LOGGING

#define SL_VERBOSE_LOGGING 1
#endif

#ifndef DX_VERBOSE_LOGGING
#define DX_VERBOSE_LOGGING 0
#endif

#if DX_VERBOSE_LOGGING
#define DXLOG(expr)                                                                                                    \
    do {                                                                                                               \
        std::wcout << L"[DX] " << expr << std::endl;                                                                   \
    } while (0)
#else
#define DXLOG(expr)                                                                                                    \
    do {                                                                                                               \
    } while (0)
#endif

void DeviceContext::Init(HWND hwnd, UINT w, UINT h, bool useWarp) {
    width = w;
    height = h;
    slModule = LoadLibrary("sl.interposer.dll");
    InitStreamline();
    CreateDeviceAndSwapChain(hwnd, useWarp);

    DXGI_SWAP_CHAIN_DESC scDesc{};
    swapChain->GetDesc(&scDesc);
    bufferCount = scDesc.BufferCount;
    DXLOG(L"Actual swap chain buffer count: " << bufferCount);

    CreateRTVsAndDepth();
    CreateFrameObjects();
}

void DeviceContext::InitHeadless(const HeadlessDeviceDesc& desc) {
    headless = true;
    width = desc.width;
    height = desc.height;
    bufferCount = FRAME_COUNT;
    frameIndex = 0;

    // CLSID_D3D12SDKConfiguration; the Windows SDK's dxguid.lib may predate it.
    static const GUID kSdkConfigurationClsid = {0x7cda6aca, 0xa03e, 0x49c8, {0x94, 0x58, 0x03, 0x34, 0xd2, 0x0e, 0x07, 0xce}};
    ComPtr<ID3D12SDKConfiguration1> sdkConfig;
    ComPtr<ID3D12DeviceFactory> factory;
    if (FAILED(D3D12GetInterface(kSdkConfigurationClsid, IID_PPV_ARGS(&sdkConfig))) ||
        FAILED(sdkConfig->CreateDeviceFactory(desc.sdkVersion, desc.sdkPath.c_str(), IID_PPV_ARGS(&factory))))
        throw std::runtime_error("The Direct3D 12 Agility SDK could not be loaded (Windows too old, or the plugin "
                                 "data folder is incomplete)");
    // Fail instead of replacing the host's device as the process singleton.
    ThrowIfFailed(factory->SetFlags(D3D12_DEVICE_FACTORY_FLAG_DISALLOW_STORING_NEW_DEVICE_AS_SINGLETON));

    ComPtr<IDXGIFactory4> dxgi;
    ThrowIfFailed(CreateDXGIFactory2(0, IID_PPV_ARGS(&dxgi)));
    headlessFactory = dxgi;
    ComPtr<IDXGIAdapter1> adapter;
    if (desc.adapterLuid.LowPart || desc.adapterLuid.HighPart) {
        ThrowIfFailed(dxgi->EnumAdapterByLuid(desc.adapterLuid, IID_PPV_ARGS(&adapter)));
    } else {
        for (UINT i = 0; dxgi->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
            DXGI_ADAPTER_DESC1 ad;
            adapter->GetDesc1(&ad);
            if (!(ad.Flags & DXGI_ADAPTER_FLAG_SOFTWARE))
                break;
        }
    }
    AdapterReport& report = adapterReport;
    {
        DXGI_ADAPTER_DESC1 ad{};
        adapter->GetDesc1(&ad);
        report.name = ad.Description;
        report.vendorId = ad.VendorId;
        report.dedicatedVideoMemory = ad.DedicatedVideoMemory;
    }
    if (FAILED(factory->CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_1, IID_PPV_ARGS(&device))))
        throw std::runtime_error("The GPU does not support Direct3D 12 feature level 12_1");
    nativeDevice = device;

    // The shaders are SM 6.9 libraries using SER and opacity micromaps (DXR 1.2).
    D3D12_FEATURE_DATA_SHADER_MODEL sm{D3D_HIGHEST_SHADER_MODEL};
    while (FAILED(device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &sm, sizeof(sm))) &&
           sm.HighestShaderModel > D3D_SHADER_MODEL_5_1)
        sm.HighestShaderModel = (D3D_SHADER_MODEL)(sm.HighestShaderModel - 1);
    report.shaderModel = sm.HighestShaderModel;
    D3D12_FEATURE_DATA_D3D12_OPTIONS5 opts5{};
    if (SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &opts5, sizeof(opts5))))
        report.raytracingTier = opts5.RaytracingTier;
    D3D12_FEATURE_DATA_D3D12_OPTIONS22 opts22{};
    if (SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS22, &opts22, sizeof(opts22))))
        report.serReorders = opts22.ShaderExecutionReorderingActuallyReorders;
    if (report.shaderModel < D3D_SHADER_MODEL_6_9)
        throw std::runtime_error("The GPU driver does not support Shader Model 6.9 (update the driver)");
    if (report.raytracingTier < D3D12_RAYTRACING_TIER_1_2)
        throw std::runtime_error("The GPU does not support DirectX Raytracing 1.2");

    slModule = LoadLibraryW((desc.pluginDir + L"\\sl.interposer.dll").c_str());
    if (!slModule)
        throw std::runtime_error("sl.interposer.dll not found next to the plugin");
    static const std::wstring pluginDir = desc.pluginDir;
    static const wchar_t* pluginPaths[] = {pluginDir.c_str()};
    static sl::Feature features[] = {sl::kFeatureDLSS_RR, sl::kFeatureDLSS_G, sl::kFeatureReflex, sl::kFeaturePCL};
    sl::Preferences pref{};
    pref.renderAPI = sl::RenderAPI::eD3D12;
    pref.engine = sl::EngineType::eCustom;
    pref.applicationId = 231313132;
    pref.logLevel = sl::LogLevel::eDefault;
    pref.logMessageCallback = desc.slLog;
    pref.pathsToPlugins = pluginPaths;
    pref.numPathsToPlugins = 1;
    // No proxies: the host owns its swap chain; DLSS-RR needs a device and command lists, frame generation the
    // swap chain of AttachWindow, upgraded by hand.
    pref.flags = sl::PreferenceFlags::eUseManualHooking | sl::PreferenceFlags::eUseFrameBasedResourceTagging;
    pref.featuresToLoad = features;
    pref.numFeaturesToLoad = _countof(features);
    if (slInit(pref, sl::kSDKVersion) != sl::Result::eOk)
        throw std::runtime_error("slInit failed");
    if (slSetD3DDevice(device.Get()) != sl::Result::eOk) {
        slShutdown();
        throw std::runtime_error("slSetD3DDevice failed");
    }
    {
        LUID luid = device->GetAdapterLuid();
        sl::AdapterInfo ai;
        ai.deviceLUID = (uint8_t*)&luid;
        ai.deviceLUIDSizeInBytes = sizeof(LUID);
        const sl::Result sr = slIsFeatureSupported(sl::kFeatureDLSS_RR, ai);
        report.dlssRayReconstruction = sr == sl::Result::eOk;
        if (!report.dlssRayReconstruction) {
            slShutdown();
            if (sr == sl::Result::eErrorDriverOutOfDate)
                throw std::runtime_error("DLSS Ray Reconstruction needs a newer NVIDIA driver");
            if (sr == sl::Result::eErrorOSOutOfDate)
                throw std::runtime_error("DLSS Ray Reconstruction needs a newer Windows version");
            throw std::runtime_error(std::string("DLSS Ray Reconstruction is not supported on this GPU (") +
                                     sl::getResultAsStr(sr) + ")");
        }
    }

    D3D12_COMMAND_QUEUE_DESC qd = {};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    ThrowIfFailed(device->CreateCommandQueue(&qd, IID_PPV_ARGS(&cmdQueue)));

    CreatePresentHook();
    slGetNewFrameToken(frameToken, nullptr);

    CreateFrameObjects();
}

void DeviceContext::CreatePresentHook() {
    DXGI_SWAP_CHAIN_DESC1 scd = {};
    scd.Width = 8;
    scd.Height = 8;
    scd.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    scd.SampleDesc.Count = 1;
    scd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    scd.BufferCount = 2;
    scd.Scaling = DXGI_SCALING_STRETCH;
    scd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;
    scd.AlphaMode = DXGI_ALPHA_MODE_PREMULTIPLIED;
    ComPtr<IDXGISwapChain1> native;
    ThrowIfFailed(headlessFactory->CreateSwapChainForComposition(cmdQueue.Get(), &scd, nullptr, &native));
    slPresentHook = UpgradeSwapChain(native.Get());
}

// Streamline's proxy for a swap chain, in place of the caller's reference (which the proxy takes over).
template <class T> ComPtr<T> DeviceContext::UpgradeSwapChain(T* native) {
    T* hooked = native;
    native->AddRef();
    if (slUpgradeInterface((void**)&hooked) != sl::Result::eOk) {
        std::wcout << L"[SL] slUpgradeInterface(swap chain) failed" << std::endl;
        ComPtr<T> plain;
        plain.Attach(native);
        return plain;
    }
    // The proxy keeps the native swap chain alive; the reference taken above goes unless the proxy took it over,
    // so that releasing the proxy releases the swap chain (Streamline follows one swap chain at a time).
    const ULONG count = native->AddRef() - 1;
    native->Release();
    if (hooked != native && count > 2)
        native->Release();
    ComPtr<T> proxy;
    proxy.Attach(hooked);
    return proxy;
}

void DeviceContext::AttachWindow(HWND hwnd, UINT width, UINT height) {
    DetachWindow();
    // Streamline follows one swap chain: the hidden one goes while the window presents.
    WaitForGPU();
    slPresentHook.Reset();
    ComPtr<IDXGIFactory5> factory5;
    BOOL tearing = FALSE;
    if (SUCCEEDED(headlessFactory.As(&factory5)))
        factory5->CheckFeatureSupport(DXGI_FEATURE_PRESENT_ALLOW_TEARING, &tearing, sizeof(tearing));
    windowTearing = tearing == TRUE;
    DXGI_SWAP_CHAIN_DESC1 scd = {};
    scd.Width = std::max(width, 8u);
    scd.Height = std::max(height, 8u);
    scd.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    scd.SampleDesc.Count = 1;
    scd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    scd.BufferCount = WindowBufferCount();
    scd.Scaling = DXGI_SCALING_STRETCH;
    scd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    scd.AlphaMode = DXGI_ALPHA_MODE_IGNORE;
    scd.Flags = windowTearing ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0;
    // Created through Streamline's factory proxy: frame generation sets itself up as it sees the swap chain created.
    ComPtr<IDXGIFactory4> nativeFactory;
    ThrowIfFailed(CreateDXGIFactory2(0, IID_PPV_ARGS(&nativeFactory)));
    ComPtr<IDXGIFactory4> factory = UpgradeSwapChain(nativeFactory.Get());
    nativeFactory.Reset();
    ComPtr<IDXGISwapChain1> created;
    ThrowIfFailed(factory->CreateSwapChainForHwnd(cmdQueue.Get(), hwnd, &scd, nullptr, nullptr, &created));
    factory->MakeWindowAssociation(hwnd, DXGI_MWA_NO_ALT_ENTER);
    ThrowIfFailed(created.As(&windowSwapChain));
}

void DeviceContext::DetachWindow() {
    if (!windowSwapChain)
        return;
    WaitForGPU();
    windowSwapChain.Reset();
    if (headless && !slPresentHook)
        CreatePresentHook();
}

void DeviceContext::CreateFrameObjects() {
    for (UINT n = 0; n < bufferCount; ++n)
        ThrowIfFailed(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&cmdAllocators[n])));

    ThrowIfFailed(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, cmdAllocators[frameIndex].Get(), nullptr,
                                            IID_PPV_ARGS(&cmdList)));
    cmdList->SetName(L"MainGraphicsList");

    ThrowIfFailed(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));
    for (UINT n = 0; n < bufferCount; ++n)
        fenceValues[n] = 0;
    fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    if (!fenceEvent)
        ThrowIfFailed(HRESULT_FROM_WIN32(GetLastError()));

    InitPlanetStreaming();
}

void DeviceContext::Shutdown() {
    WaitForGPU();
    planetComputeFence.shutdown();
    planetCopyFence.shutdown();
    CloseHandle(fenceEvent);
    slShutdown();
}

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

    if (frameToken)
        slGetNewFrameToken(frameToken, nullptr);
}

void DeviceContext::ExecuteAndPresent() {
    ThrowIfFailed(cmdList->Close());

    cmdQueue->Wait(planetComputeFence.fence(), planetComputeAtSlot[frameIndex]);

    ID3D12CommandList* lists[] = {cmdList.Get()};
    cmdQueue->ExecuteCommandLists(1, lists);

#if ENABLE_D3D12_DIAGNOSTICS

    dxdiag::DumpNewMessages();
#endif

    if (headless) {
        if (slPresentHook && !windowSwapChain) // with a window, its present (the host's) ends the frame
            slPresentHook->Present(0, 0);
        ThrowIfFailed(cmdQueue->Signal(fence.Get(), fenceValues[frameIndex] = nextFenceValue++));
        frameIndex = (frameIndex + 1) % bufferCount;
        return;
    }

    UINT presentFlags = tearingSupported ? DXGI_PRESENT_ALLOW_TEARING : 0;
    HRESULT presentHr = swapChain->Present(0, presentFlags);
    if (FAILED(presentHr)) {
        std::wcout << L"[DX] Present failed: 0x" << std::hex << presentHr << std::dec << std::endl;
#if ENABLE_D3D12_DIAGNOSTICS
        dxdiag::DumpNewMessages();

        dxdiag::CheckDeviceRemoved(NativeDevice(), 1000);
        dxdiag::CheckDeviceRemoved(device.Get(), 500);
#endif

        // Streamline may report device loss only via Present.
        const bool lostByPresent = presentHr == DXGI_ERROR_DEVICE_REMOVED || presentHr == DXGI_ERROR_DEVICE_HUNG ||
                                   presentHr == DXGI_ERROR_DEVICE_RESET;
        if (lostByPresent || FAILED(NativeDevice()->GetDeviceRemovedReason()) ||
            FAILED(device->GetDeviceRemovedReason()))
            ThrowIfFailed(presentHr);
    }

    const UINT64 fv = nextFenceValue++;
    fenceValues[frameIndex] = fv;
    ThrowIfFailed(cmdQueue->Signal(fence.Get(), fv));

    frameIndex = swapChain->GetCurrentBackBufferIndex();

#if ENABLE_D3D12_DIAGNOSTICS
    dxdiag::CheckDeviceRemoved(NativeDevice());
    dxdiag::DumpNewMessages();
#endif
}

void DeviceContext::WaitForGPU() {
    const UINT64 fv = nextFenceValue++;
    ThrowIfFailed(cmdQueue->Signal(fence.Get(), fv));
    if (fence->GetCompletedValue() < fv) {
        ThrowIfFailed(fence->SetEventOnCompletion(fv, fenceEvent));
        WaitForSingleObject(fenceEvent, INFINITE);
    }
    if (swapChain)
        frameIndex = swapChain->GetCurrentBackBufferIndex();
}

void DeviceContext::FlushAndReset() {
    ThrowIfFailed(cmdList->Close());
    ID3D12CommandList* lists[] = {cmdList.Get()};
    cmdQueue->ExecuteCommandLists(1, lists);
    WaitForGPU();
    ThrowIfFailed(cmdAllocators[frameIndex]->Reset());
    ThrowIfFailed(cmdList->Reset(cmdAllocators[frameIndex].Get(), nullptr));
}

void DeviceContext::Resize(UINT newWidth, UINT newHeight) {
    if (newWidth == 0 || newHeight == 0)
        return;
    if (newWidth == width && newHeight == height)
        return;

    WaitForGPU();

    if (headless) {
        width = newWidth;
        height = newHeight;
        return;
    }

    for (UINT n = 0; n < bufferCount; ++n)
        renderTargets[n].Reset();
    depthStencil.Reset();

    width = newWidth;
    height = newHeight;

    UINT resizeFlags = tearingSupported ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0;
    ThrowIfFailed(swapChain->ResizeBuffers(0, width, height, DXGI_FORMAT_R8G8B8A8_UNORM, resizeFlags));
    frameIndex = swapChain->GetCurrentBackBufferIndex();

    DXGI_SWAP_CHAIN_DESC scDesc{};
    swapChain->GetDesc(&scDesc);
    bufferCount = scDesc.BufferCount;

    CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHandle(rtvHeap->GetCPUDescriptorHandleForHeapStart());
    for (UINT n = 0; n < bufferCount; n++) {
        ThrowIfFailed(swapChain->GetBuffer(n, IID_PPV_ARGS(&renderTargets[n])));
        device->CreateRenderTargetView(renderTargets[n].Get(), nullptr, rtvHandle);
        rtvHandle.Offset(1, rtvDescriptorSize);
    }

    auto hp = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
    auto drd = CD3DX12_RESOURCE_DESC::Tex2D(DXGI_FORMAT_D32_FLOAT, width, height, 1, 1);
    drd.Flags |= D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    CD3DX12_CLEAR_VALUE cv(DXGI_FORMAT_D32_FLOAT, 1.0f, 0);
    ThrowIfFailed(device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &drd, D3D12_RESOURCE_STATE_DEPTH_WRITE,
                                                  &cv, IID_PPV_ARGS(&depthStencil)));

    D3D12_DEPTH_STENCIL_VIEW_DESC dsvDesc = {};
    dsvDesc.Format = DXGI_FORMAT_D32_FLOAT;
    dsvDesc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
    device->CreateDepthStencilView(depthStencil.Get(), &dsvDesc, dsvHeap->GetCPUDescriptorHandleForHeapStart());
}

D3D12_CPU_DESCRIPTOR_HANDLE DeviceContext::CurrentRTV() const {
    return CD3DX12_CPU_DESCRIPTOR_HANDLE(rtvHeap->GetCPUDescriptorHandleForHeapStart(), frameIndex, rtvDescriptorSize);
}

D3D12_CPU_DESCRIPTOR_HANDLE DeviceContext::DSV() const {
    return dsvHeap->GetCPUDescriptorHandleForHeapStart();
}

void DeviceContext::InitStreamline() {
    sl::Preferences pref{};
    pref.renderAPI = sl::RenderAPI::eD3D12;
    pref.engine = sl::EngineType::eCustom;
    pref.applicationId = 231313132;
    pref.showConsole = SL_VERBOSE_LOGGING;
    pref.logLevel = SL_VERBOSE_LOGGING ? sl::LogLevel::eVerbose : sl::LogLevel::eOff;
#if SL_VERBOSE_LOGGING

    static std::wstring s_slLogDir = [] {
        wchar_t exePath[MAX_PATH] = {0};
        GetModuleFileNameW(nullptr, exePath, MAX_PATH);
        std::wstring p(exePath);
        const size_t slash = p.find_last_of(L"\\/");
        if (slash != std::wstring::npos)
            p.resize(slash);
        return p;
    }();
    pref.pathToLogsAndData = s_slLogDir.c_str();
#endif
    pref.flags = sl::PreferenceFlags::eLoadDownloadedPlugins | sl::PreferenceFlags::eUseFrameBasedResourceTagging;

    static sl::Feature featList[] = {sl::kFeatureDLSS, sl::kFeatureDLSS_RR, sl::kFeatureDLSS_G, sl::kFeatureReflex,
                                     sl::kFeaturePCL};
    pref.featuresToLoad = featList;
    pref.numFeaturesToLoad = _countof(featList);

    sl::Result res = slInit(pref, sl::kSDKVersion);
    if (res != sl::Result::eOk)
        std::wcout << L"slInit failed! Error: " << (int)res << std::endl;
}

void DeviceContext::CreateDeviceAndSwapChain(HWND hwnd, bool useWarp) {
    typedef HRESULT(WINAPI * PFunCreateDXGIFactory2)(UINT, REFIID, void**);
    typedef HRESULT(WINAPI * PFunD3D12CreateDevice)(IUnknown*, D3D_FEATURE_LEVEL, REFIID, void**);

    auto slCreateFactory = reinterpret_cast<PFunCreateDXGIFactory2>(GetProcAddress(slModule, "CreateDXGIFactory2"));
    auto slCreateDevice = reinterpret_cast<PFunD3D12CreateDevice>(GetProcAddress(slModule, "D3D12CreateDevice"));

    ComPtr<IDXGIFactory4> factory;
    ThrowIfFailed(slCreateFactory(0, IID_PPV_ARGS(&factory)));

    {
        ComPtr<IDXGIFactory5> factory5;
        if (SUCCEEDED(factory.As(&factory5))) {
            BOOL allowTearing = FALSE;
            factory5->CheckFeatureSupport(DXGI_FEATURE_PRESENT_ALLOW_TEARING, &allowTearing, sizeof(allowTearing));
            tearingSupported = (allowTearing == TRUE);
        }
        DXLOG(L"Tearing support: " << (tearingSupported ? L"yes" : L"no"));
    }

#if ENABLE_D3D12_DIAGNOSTICS
    dxdiag::EnableDebugLayerAndDred();
#endif

#ifndef RT_HOSTED
    // Clustered geometry needs SM 6.10 and the raytracing preview, both experimental: without Windows Developer Mode
    // this fails and the renderer keeps classic BLAS. RDN_CLUSTERS=0 opts out. Must precede device creation.
    bool clustersWanted = true;
    {
        wchar_t v[8] = {};
        if (GetEnvironmentVariableW(L"RDN_CLUSTERS", v, 8) > 0 && v[0] == L'0')
            clustersWanted = false;
    }
    bool experimentalOn = false;
    if (clustersWanted) {
        UUID experimental[] = {D3D12ExperimentalShaderModels, D3D12RaytracingPreview};
        const HRESULT hr = D3D12EnableExperimentalFeatures(_countof(experimental), experimental, nullptr, nullptr);
        experimentalOn = SUCCEEDED(hr);
        if (!experimentalOn)
            std::wcout << L"[DXR] Experimental features unavailable (0x" << std::hex << (uint32_t)hr << std::dec
                       << L"; Developer Mode off?)" << std::endl;
    }
#endif

    if (useWarp) {
        ComPtr<IDXGIAdapter> warpAdapter;
        ThrowIfFailed(factory->EnumWarpAdapter(IID_PPV_ARGS(&warpAdapter)));
        ThrowIfFailed(slCreateDevice(warpAdapter.Get(), D3D_FEATURE_LEVEL_12_1, IID_PPV_ARGS(&device)));
    } else {
        ComPtr<IDXGIAdapter1> hardwareAdapter;

        for (UINT i = 0; factory->EnumAdapters1(i, &hardwareAdapter) != DXGI_ERROR_NOT_FOUND; ++i) {
            DXGI_ADAPTER_DESC1 desc;
            hardwareAdapter->GetDesc1(&desc);
            if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)
                continue;
            if (SUCCEEDED(
                    slCreateDevice(hardwareAdapter.Get(), D3D_FEATURE_LEVEL_12_1, _uuidof(ID3D12Device), nullptr)))
                break;
        }
        ThrowIfFailed(slCreateDevice(hardwareAdapter.Get(), D3D_FEATURE_LEVEL_12_1, IID_PPV_ARGS(&device)));
        {
            void* native = nullptr;
            if (slGetNativeInterface(device.Get(), &native) == sl::Result::eOk && native)
                nativeDevice.Attach(static_cast<ID3D12Device*>(native));
        }
#if ENABLE_D3D12_DIAGNOSTICS
        dxdiag::HookDevice(NativeDevice());
#endif
    }

#ifndef RT_HOSTED
    if (experimentalOn) {
        D3D12_FEATURE_DATA_SHADER_MODEL sm{D3D_SHADER_MODEL_6_10};
        D3D12_FEATURE_DATA_D3D12_OPTIONS_RAYTRACING2 rt2{};
        ComPtr<ID3D12Device16> device16;
        clusteredGeometry =
            SUCCEEDED(NativeDevice()->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &sm, sizeof(sm))) &&
            sm.HighestShaderModel >= D3D_SHADER_MODEL_6_10 &&
            SUCCEEDED(NativeDevice()->QueryInterface(IID_PPV_ARGS(&device16))) &&
            SUCCEEDED(device16->CheckFeatureSupport1(D3D12_FEATURE_D3D12_OPTIONS_RAYTRACING2, &rt2, sizeof(rt2),
                                                     D3D12_PREVIEW_SDK_VERSION)) &&
            rt2.ClustersAndPTLASSupported;
    }
    nv_helpers_dx12::g_clusteredGeometry = clusteredGeometry;
    std::wcout << L"[DXR] Clustered geometry + PTLAS: "
               << (clusteredGeometry ? L"on (SM 6.10)" : clustersWanted ? L"unsupported" : L"off (RDN_CLUSTERS=0)")
               << std::endl;
#endif

    sl::Result res = slSetD3DDevice(device.Get());
    if (res != sl::Result::eOk)
        std::wcout << L"slSetD3DDevice failed: " << (int)res << std::endl;

    LUID luid = device->GetAdapterLuid();
    sl::AdapterInfo ai;
    ai.deviceLUID = (uint8_t*)&luid;
    ai.deviceLUIDSizeInBytes = sizeof(LUID);
    sl::Result sr = slIsFeatureSupported(sl::kFeatureDLSS_RR, ai);
    if (sr != sl::Result::eOk)
        std::wcout << L"[DLSS-RR] not supported: " << (int)sr << std::endl;

    D3D12_COMMAND_QUEUE_DESC qd = {};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    ThrowIfFailed(device->CreateCommandQueue(&qd, IID_PPV_ARGS(&cmdQueue)));

    DXGI_SWAP_CHAIN_DESC1 scd = {};
    scd.BufferCount = FRAME_COUNT;
    scd.Width = width;
    scd.Height = height;
    scd.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    scd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    scd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    scd.SampleDesc.Count = 1;
    if (tearingSupported)
        scd.Flags |= DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING;

    ComPtr<IDXGISwapChain1> sc1;
    ThrowIfFailed(factory->CreateSwapChainForHwnd(cmdQueue.Get(), hwnd, &scd, nullptr, nullptr, &sc1));
    ThrowIfFailed(factory->MakeWindowAssociation(hwnd, DXGI_MWA_NO_ALT_ENTER));
    ThrowIfFailed(sc1.As(&swapChain));
    frameIndex = swapChain->GetCurrentBackBufferIndex();

    slGetNewFrameToken(frameToken, nullptr);
}

void DeviceContext::CreateRTVsAndDepth() {
    D3D12_DESCRIPTOR_HEAP_DESC rtvDesc = {};
    rtvDesc.NumDescriptors = bufferCount;
    rtvDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    ThrowIfFailed(device->CreateDescriptorHeap(&rtvDesc, IID_PPV_ARGS(&rtvHeap)));
    rtvDescriptorSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

    CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHandle(rtvHeap->GetCPUDescriptorHandleForHeapStart());
    for (UINT n = 0; n < bufferCount; n++) {
        ThrowIfFailed(swapChain->GetBuffer(n, IID_PPV_ARGS(&renderTargets[n])));
        device->CreateRenderTargetView(renderTargets[n].Get(), nullptr, rtvHandle);
        rtvHandle.Offset(1, rtvDescriptorSize);
    }

    dsvHeap = nv_helpers_dx12::CreateDescriptorHeap(device.Get(), 1, D3D12_DESCRIPTOR_HEAP_TYPE_DSV, false);

    auto hp = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
    auto drd = CD3DX12_RESOURCE_DESC::Tex2D(DXGI_FORMAT_D32_FLOAT, width, height, 1, 1);
    drd.Flags |= D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    CD3DX12_CLEAR_VALUE cv(DXGI_FORMAT_D32_FLOAT, 1.0f, 0);
    ThrowIfFailed(device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &drd, D3D12_RESOURCE_STATE_DEPTH_WRITE,
                                                  &cv, IID_PPV_ARGS(&depthStencil)));

    D3D12_DEPTH_STENCIL_VIEW_DESC dsvDesc = {};
    dsvDesc.Format = DXGI_FORMAT_D32_FLOAT;
    dsvDesc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
    device->CreateDepthStencilView(depthStencil.Get(), &dsvDesc, dsvHeap->GetCPUDescriptorHandleForHeapStart());
}

void DeviceContext::InitPlanetStreaming() {
    D3D12_COMMAND_QUEUE_DESC cqd = {};
    cqd.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    ThrowIfFailed(device->CreateCommandQueue(&cqd, IID_PPV_ARGS(&planetComputeQueue)));
    D3D12_COMMAND_QUEUE_DESC pqd = {};
    pqd.Type = D3D12_COMMAND_LIST_TYPE_COPY;
    ThrowIfFailed(device->CreateCommandQueue(&pqd, IID_PPV_ARGS(&planetCopyQueue)));

    for (UINT n = 0; n < bufferCount; ++n) {
        ThrowIfFailed(
            device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&planetComputeAllocators[n])));
        ThrowIfFailed(
            device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY, IID_PPV_ARGS(&planetCopyAllocators[n])));
    }
    ThrowIfFailed(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, planetComputeAllocators[0].Get(),
                                            nullptr, IID_PPV_ARGS(&planetComputeList)));
    planetComputeList->SetName(L"PlanetComputeList");
    ThrowIfFailed(planetComputeList->Close());
    ThrowIfFailed(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COPY, planetCopyAllocators[0].Get(), nullptr,
                                            IID_PPV_ARGS(&planetCopyList)));
    planetCopyList->SetName(L"PlanetCopyList");
    ThrowIfFailed(planetCopyList->Close());

    planetComputeFence.init(device.Get());
    planetCopyFence.init(device.Get());
}

void DeviceContext::ResetPlanetLists() {
    planetCopyFence.cpu_wait(planetCopyAtSlot[frameIndex]);
    planetComputeFence.cpu_wait(planetComputeAtSlot[frameIndex]);

    ThrowIfFailed(planetCopyAllocators[frameIndex]->Reset());
    ThrowIfFailed(planetCopyList->Reset(planetCopyAllocators[frameIndex].Get(), nullptr));
    ThrowIfFailed(planetComputeAllocators[frameIndex]->Reset());
    ThrowIfFailed(planetComputeList->Reset(planetComputeAllocators[frameIndex].Get(), nullptr));
}

UINT64 DeviceContext::SubmitPlanetCopy() {
    ThrowIfFailed(planetCopyList->Close());
    ID3D12CommandList* lists[] = {planetCopyList.Get()};
    planetCopyQueue->ExecuteCommandLists(1, lists);
    const UINT64 v = planetCopyFence.signal(planetCopyQueue.Get());
    planetCopyAtSlot[frameIndex] = v;
    return v;
}

UINT64 DeviceContext::SubmitPlanetCompute(UINT64 waitCopyValue) {
    planetCopyFence.queue_wait(planetComputeQueue.Get(), waitCopyValue);
    ThrowIfFailed(planetComputeList->Close());
    ID3D12CommandList* lists[] = {planetComputeList.Get()};
    planetComputeQueue->ExecuteCommandLists(1, lists);
    const UINT64 v = planetComputeFence.signal(planetComputeQueue.Get());
    planetComputeAtSlot[frameIndex] = v;
    return v;
}

UINT64 DeviceContext::PlanetComputeCompleted() const {
    return planetComputeFence.completed();
}
void DeviceContext::PlanetComputeCpuWait(UINT64 value) {
    planetComputeFence.cpu_wait(value);
}
