// RoyalTracerUnity.dll: the renderer hosted in a Unity player or editor.
//
// The renderer runs on its own D3D12 device (Agility 1.619, loaded side by side; Unity's runtime may be older) on
// the host's adapter. Each submitted frame (a ticket) writes a colour/depth pair the host's device opens through
// shared handles. Per ticket the host queue waits for "done" before compositing and signals "release" after; the
// renderer waits for the release of the ticket that used the same pair two frames earlier.
#include "../rdn/stdafx.h"
#include "../rdn/Renderer.h"
#include "../rdn/DXRHelper.h"
#include "RoyalTracerApi.h"
#include "SceneBridge.h"
#include "HostStream.h"
#include "../third_party/unity/IUnityInterface.h"
#include "../third_party/unity/IUnityGraphics.h"
#include "../third_party/unity/IUnityGraphicsD3D12.h"
#include <mutex>
#include <streambuf>
#include <DirectXPackedVector.h>

namespace {

RT_LogFn g_log = nullptr;
std::mutex g_logMutex;
std::string g_lastError;

void Log(int32_t level, const std::string& message) {
    std::lock_guard lock(g_logMutex);
    if (g_log)
        g_log(level, message.c_str());
}

void Fail(const std::string& message) {
    {
        std::lock_guard lock(g_logMutex);
        g_lastError = message;
    }
    Log(RT_LOG_ERROR, message);
}

std::string Utf8(const std::wstring& w) {
    if (w.empty())
        return {};
    const int n = WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), nullptr, 0, nullptr, nullptr);
    std::string s(n, '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), s.data(), n, nullptr, nullptr);
    return s;
}

// The renderer reports through std::wcout/std::cout; a host has no console, so lines go to the host log.
template <class Ch> class LineForwarder : public std::basic_streambuf<Ch> {
    using Traits = std::char_traits<Ch>;
    std::basic_string<Ch> m_line;
    std::mutex m_mutex;

    void Put(Ch c) {
        if (c != Ch('\n')) {
            m_line.push_back(c);
            return;
        }
        if constexpr (std::is_same_v<Ch, wchar_t>)
            Log(RT_LOG_INFO, Utf8(m_line));
        else
            Log(RT_LOG_INFO, m_line);
        m_line.clear();
    }

  protected:
    typename Traits::int_type overflow(typename Traits::int_type c) override {
        if (Traits::eq_int_type(c, Traits::eof()))
            return Traits::not_eof(c);
        std::lock_guard lock(m_mutex);
        Put(Traits::to_char_type(c));
        return c;
    }
    std::streamsize xsputn(const Ch* s, std::streamsize n) override {
        std::lock_guard lock(m_mutex);
        for (std::streamsize i = 0; i < n; ++i)
            Put(s[i]);
        return n;
    }
};

void SlLog(sl::LogType type, const char* message) {
    std::string m(message);
    while (!m.empty() && (m.back() == '\n' || m.back() == '\r'))
        m.pop_back();
    Log(type == sl::LogType::eError ? RT_LOG_ERROR : type == sl::LogType::eWarn ? RT_LOG_WARNING : RT_LOG_INFO,
        "[Streamline] " + m);
}

IUnityInterfaces* g_unity = nullptr;
IUnityGraphics* g_graphics = nullptr;
IUnityGraphicsD3D12v8* g_d3d12 = nullptr;

// Both slots' textures at one size. Tickets hold the set they render into, so a resize never pulls textures out
// from under a frame that is still queued.
struct OutputSet {
    ComPtr<ID3D12Resource> color[2], depth[2];         // renderer device
    ComPtr<ID3D12Resource> hostColor[2], hostDepth[2]; // the same memory, opened on the host device
    UINT width = 0, height = 0, generation = 0;
};

// The water overlays of one frame, with the atlases as they were then (a host may replace them in the next).
struct TicketOverlays {
    Microsoft::WRL::ComPtr<ID3D12Resource> height, foam;
    std::vector<RT_WaterOverlay> list;
};

struct Ticket {
    uint64_t index = 0;
    RT_FrameDesc frame{};
    std::shared_ptr<OutputSet> outputs;
    TicketOverlays overlays;
};

constexpr uint32_t kTicketRing = 8;

struct HostTexture {
    ComPtr<ID3D12Resource> resource, host; // renderer device, and the same memory opened on the host device
};

struct Plugin {
    std::mutex mutex;
    DeviceContext ctx; // lives for the process; renderers come and go on it
    std::unique_ptr<HostStream> host; // the scene; outlives nothing but the renderer that reads it
    std::unique_ptr<Renderer> renderer;
    std::shared_ptr<OutputSet> outputs;
    UINT generation = 0;
    Ticket tickets[kTicketRing];
    uint64_t submitted = 0;
    UINT width = 1280, height = 720;
    sl::DLSSMode dlssMode = sl::DLSSMode::eDLAA;
    bool failed = false;
    ComPtr<ID3D12Fence> done, release;         // renderer device, shared
    ComPtr<ID3D12Fence> hostDone, hostRelease; // opened on the host device
    // Host textures: the host signals upload on its queue at each frame's render event, after its writes to them.
    std::vector<HostTexture> hostTextures; // by id; empty entries are free
    std::vector<std::pair<uint64_t, HostTexture>> retiredHostTextures; // freed after that frame is done
    ComPtr<ID3D12Fence> upload, hostUpload;
    RT_WaterDesc water{}; // the host's ocean, applied at each frame
    int32_t skyTexture = -1; // the host's sky (RT_SetSky)
    float skyScale = 0.0f;
    // Frame generation: the renderer presents the host's frames through a child window of the host's.
    struct Presenter {
        HWND parent = nullptr, window = nullptr;
        UINT width = 0, height = 0;    // the window's client size, followed by the swap chain
        int32_t source = -1;           // host texture with the host's finished frame
        int32_t hudless = -1;          // and with the same frame before its UI (-1: the finished one)
        uint32_t frames = 1;           // frames generated per rendered frame
        bool vsync = false;
        ComPtr<ID3D12Fence> upload, hostUpload; // the host's copy into source, before each present
        uint64_t uploadValue = 0;
        ComPtr<ID3D12CommandAllocator> allocators[3];
        ComPtr<ID3D12GraphicsCommandList> list;
        ComPtr<ID3D12Fence> fence;      // this queue's presents, for the allocators
        uint64_t fenceValue = 0, allocatorValue[3] = {};
        uint64_t presents = 0, displayed = 0; // presents, and frames frame generation put on screen for them
        uint32_t status = 0;
        uint32_t maxGenerated = 0; // as frame generation reports it once it runs
        UINT optionsW = 0, optionsH = 0, optionsMvW = 0, optionsMvH = 0, optionsFrames = 0; // DLSS-G options as last set
        std::string lastError;
    } presenter;
};
Plugin* g_plugin = nullptr;
std::mutex g_initMutex;
RT_Capabilities g_caps{};
bool g_initTried = false;
// Never destroyed: the C++ runtime flushes std::cout/std::wcout at process exit, after this DLL's statics are gone.
LineForwarder<wchar_t>* g_wideForwarder = new LineForwarder<wchar_t>;
LineForwarder<char>* g_narrowForwarder = new LineForwarder<char>;

ID3D12Device* HostDevice() { return g_d3d12 ? g_d3d12->GetDevice() : nullptr; }

template <class T> ComPtr<T> OpenOnHost(ID3D12Device* device, ID3D12DeviceChild* object) {
    ID3D12Device* host = HostDevice();
    if (!host)
        return nullptr;
    HANDLE handle = nullptr;
    ThrowIfFailed(device->CreateSharedHandle(object, nullptr, GENERIC_ALL, nullptr, &handle));
    ComPtr<T> opened;
    const HRESULT hr = host->OpenSharedHandle(handle, IID_PPV_ARGS(&opened));
    CloseHandle(handle);
    ThrowIfFailed(hr);
    return opened;
}

std::shared_ptr<OutputSet> CreateOutputs(Plugin& p, UINT width, UINT height) {
    auto set = std::make_shared<OutputSet>();
    set->width = width;
    set->height = height;
    set->generation = ++p.generation;
    ID3D12Device* device = p.ctx.Device();
    auto create = [&](DXGI_FORMAT format, const wchar_t* name) {
        auto desc = CD3DX12_RESOURCE_DESC::Tex2D(format, width, height, 1, 1, 1, 0,
                                                 D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS |
                                                     D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS);
        auto heap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
        ComPtr<ID3D12Resource> res;
        ThrowIfFailed(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_SHARED, &desc, D3D12_RESOURCE_STATE_COMMON,
                                                      nullptr, IID_PPV_ARGS(&res)));
        res->SetName(name);
        return res;
    };
    for (int s = 0; s < 2; ++s) {
        set->color[s] = create(DXGI_FORMAT_R16G16B16A16_FLOAT, L"HostColor");
        set->depth[s] = create(DXGI_FORMAT_R32_FLOAT, L"HostDepth");
        set->hostColor[s] = OpenOnHost<ID3D12Resource>(device, set->color[s].Get());
        set->hostDepth[s] = OpenOnHost<ID3D12Resource>(device, set->depth[s].Get());
    }
    return set;
}

void ReleaseHostQueue(uint64_t index) {
    if (g_plugin && g_plugin->hostDone)
        g_plugin->hostDone->Signal(index); // CPU-side: the host must never wait on a frame that will not come
}

void RenderTicket(Plugin& p, uint64_t index) {
    std::lock_guard lock(p.mutex);
    // This event flushes everything the host recorded before it, the previous ticket's reads of its outputs included:
    // that ticket is released here, which spares the host's queue a flush of its own for RT_EVENT_RELEASE.
    if (g_d3d12 && p.hostRelease && index > 1)
        g_d3d12->GetCommandQueue()->Signal(p.hostRelease.Get(), index - 1);
    Ticket& t = p.tickets[index % kTicketRing];
    if (p.failed || !p.renderer || t.index != index || !t.outputs) {
        ReleaseHostQueue(index);
        return;
    }
    try {
        Renderer& r = *p.renderer;
        const RT_FrameDesc& f = t.frame;
        if (f.width != r.GetWidth() || f.height != r.GetHeight())
            r.OnResize(f.width, f.height);

        Camera& camera = r.GetCamera();
        const glm::vec3 origin = camera.getSceneOriginWorld();
        const glm::vec3 eye(f.eye[0], f.eye[1], f.eye[2]), target(f.target[0], f.target[1], f.target[2]);
        nv_helpers_dx12::CameraManip.setLookat(eye - origin, target - origin, glm::vec3(f.up[0], f.up[1], f.up[2]));
        camera.fovDegrees = f.fovYDegrees;
        camera.skyGroundY = f.skyGroundY;
        SunSettings& sun = camera.sunSettings;
        sun.sunDirX = f.sunDirection[0];
        sun.sunDirY = f.sunDirection[1];
        sun.sunDirZ = f.sunDirection[2];
        sun.sunDirOverride = 1.0f;
        sun.simSpeed = 0.0f;
        sun.sunIntensity = f.sunIntensity;
        sun.skyIntensity = f.skyIntensity;
        sun.turbidity = f.turbidity;
        sun.skyStarIntensity = 0.0f;

        const uint32_t slot = (uint32_t)(index & 1u);
        if (index > 2 && p.hostRelease) // only a host signals releases
            ThrowIfFailed(p.ctx.CmdQueue()->Wait(p.release.Get(), index - 2));
        // The host's writes to its textures, flushed to its queue before this event, come before this frame's work.
        if (g_d3d12 && p.hostUpload) {
            ThrowIfFailed(g_d3d12->GetCommandQueue()->Signal(p.hostUpload.Get(), index));
            ThrowIfFailed(p.ctx.CmdQueue()->Wait(p.upload.Get(), index));
            ThrowIfFailed(p.ctx.PlanetComputeQueue()->Wait(p.upload.Get(), index));
        }
        const uint64_t completed = p.done->GetCompletedValue();
        std::erase_if(p.retiredHostTextures, [&](const auto& r) { return r.first <= completed; });
        r.SetDlssMode(p.dlssMode);
        if (r.GetOcean().Enabled() && r.GetOcean().GetParams().external) {
            ocean::OceanSystem::External ext;
            const RT_WaterDesc& w = p.water;
            for (uint32_t b = 0; b < std::min<uint32_t>(w.bandCount, OCEAN_CASCADES); ++b) {
                const RT_WaterBand& src = w.bands[b];
                const bool live = src.texture >= 0 && (size_t)src.texture < p.hostTextures.size() &&
                                  p.hostTextures[src.texture].resource;
                if (!live)
                    break;
                ext.band[b] = {p.hostTextures[src.texture].resource.Get(), src.invPatchSize, src.offset[0], src.offset[1],
                               src.amplitude, src.fadeA, src.fadeB};
                ext.bands = b + 1;
            }
            ext.resolution = w.resolution;
            ext.seaLevel = w.seaLevel;
            ext.maxHeight = w.maxWaveHeight;
            r.GetOcean().SetExternal(ext);
            ocean::Params op = r.GetOcean().GetParams(); // the material follows (Renderer::UpdateRenderer)
            const bool optics = std::any_of(std::begin(w.absorption), std::end(w.absorption), [](float v) { return v > 0.0f; }) ||
                                std::any_of(std::begin(w.scattering), std::end(w.scattering), [](float v) { return v > 0.0f; });
            bool changed = false;
            if (op.customOptics != optics || (optics && (std::memcmp(op.absorption, w.absorption, sizeof w.absorption) ||
                                                         std::memcmp(op.scattering, w.scattering, sizeof w.scattering)))) {
                op.customOptics = optics;
                std::memcpy(op.absorption, w.absorption, sizeof op.absorption);
                std::memcpy(op.scattering, w.scattering, sizeof op.scattering);
                changed = true;
            }
            if (w.windSpeed > 0.0f && (op.windSpeed != w.windSpeed || op.foamCoverage != w.foamCoverage)) {
                op.windSpeed = w.windSpeed; // rebakes the renderer's spectrum: its slope statistics follow the wind
                op.foamCoverage = std::max(w.foamCoverage, 0.0f);
                changed = true;
            }
            if (changed)
                r.GetOcean().Configure(op);
            camera.oceanEnabled = r.GetOcean().Visible();
        }
        if (r.GetOcean().Enabled()) {
            ocean::OceanSystem::Overlays overlays;
            overlays.height = t.overlays.height.Get();
            overlays.foam = t.overlays.foam.Get();
            for (const RT_WaterOverlay& src : t.overlays.list) {
                if (overlays.count >= OCEAN_OVERLAYS)
                    break;
                ocean::OceanSystem::Overlay& o = overlays.overlay[overlays.count++];
                o.originX = src.origin[0];
                o.originZ = src.origin[1];
                o.sizeX = src.size[0];
                o.sizeZ = src.size[1];
                std::copy(std::begin(src.heightRegion), std::end(src.heightRegion), o.height);
                std::copy(std::begin(src.foamRegion), std::end(src.foamRegion), o.foam);
                o.heightScale = src.heightScale;
                o.foamScale = src.foamScale;
                o.maxHeight = src.maxHeight;
                o.foamMax = src.foamMax;
            }
            r.GetOcean().SetOverlays(overlays);
        }
        ID3D12Resource* sky = p.skyTexture >= 0 && (size_t)p.skyTexture < p.hostTextures.size()
                                  ? p.hostTextures[p.skyTexture].resource.Get()
                                  : nullptr;
        r.SetHostSky(sky);
        camera.hostSkyScale = sky ? p.skyScale : 0.0f;
        r.UpdateRenderer(f.deltaTime);
        r.SetHostOutputs(t.outputs->color[slot].Get(), t.outputs->depth[slot].Get());
        r.RenderFrame();
        ThrowIfFailed(p.ctx.CmdQueue()->Signal(p.done.Get(), index));
        if (g_d3d12 && p.hostDone)
            g_d3d12->GetCommandQueue()->Wait(p.hostDone.Get(), index);
    } catch (const std::exception& e) {
        p.failed = true;
        Fail(std::string("Frame failed, path tracing stops: ") + e.what());
        ReleaseHostQueue(index);
    }
}

// The host's finished frame of a ticket to the window, with frame generation's inputs from the ticket's rendering.
void PresentTicket(Plugin& p, uint64_t index) {
    std::lock_guard lock(p.mutex);
    Plugin::Presenter& pr = p.presenter;
    IDXGISwapChain3* swapChain = p.ctx.WindowSwapChain();
    if (p.failed || !p.renderer || !swapChain || !g_d3d12 || pr.source < 0 || (size_t)pr.source >= p.hostTextures.size() ||
        !p.hostTextures[pr.source].resource || p.tickets[index % kTicketRing].index != index)
        return;
    try {
        Renderer& r = *p.renderer;
        DeviceContext& ctx = p.ctx;
        ID3D12Resource* source = p.hostTextures[pr.source].resource.Get();
        // The host's copy into source comes first.
        ThrowIfFailed(g_d3d12->GetCommandQueue()->Signal(pr.hostUpload.Get(), ++pr.uploadValue));
        ThrowIfFailed(ctx.CmdQueue()->Wait(pr.upload.Get(), pr.uploadValue));

        // Follow the window; frame generation is off while the buffers change (Streamline's present hook waits).
        DXGI_SWAP_CHAIN_DESC1 scd{};
        swapChain->GetDesc1(&scd);
        const DLSSManager& dlss = r.GetDLSS();
        if (pr.width && pr.height && (scd.Width != pr.width || scd.Height != pr.height)) {
            sl::DLSSGOptions off{};
            off.mode = sl::DLSSGMode::eOff;
            slDLSSGSetOptions(ctx.viewportHandle, off);
            pr.optionsW = 0;
            ctx.WaitForGPU();
            ThrowIfFailed(swapChain->ResizeBuffers(0, pr.width, pr.height, DXGI_FORMAT_UNKNOWN, scd.Flags));
            swapChain->GetDesc1(&scd);
        }
        if (pr.optionsW != scd.Width || pr.optionsH != scd.Height || pr.optionsMvW != dlss.RenderWidth() ||
            pr.optionsMvH != dlss.RenderHeight() || pr.optionsFrames != pr.frames) {
            sl::DLSSGOptions on{};
            on.mode = pr.frames ? sl::DLSSGMode::eOn : sl::DLSSGMode::eOff;
            on.numFramesToGenerate = std::max(pr.frames, 1u);
            on.numBackBuffers = ctx.WindowBufferCount();
            on.mvecDepthWidth = dlss.RenderWidth();
            on.mvecDepthHeight = dlss.RenderHeight();
            on.colorWidth = scd.Width;
            on.colorHeight = scd.Height;
            const sl::Result result = slDLSSGSetOptions(ctx.viewportHandle, on);
            if (result != sl::Result::eOk)
                pr.lastError = std::string("slDLSSGSetOptions: ") + sl::getResultAsStr(result);
            pr.optionsW = scd.Width;
            pr.optionsH = scd.Height;
            pr.optionsMvW = dlss.RenderWidth();
            pr.optionsMvH = dlss.RenderHeight();
            pr.optionsFrames = pr.frames;
        }

        const UINT slot = (UINT)(pr.presents % 3);
        if (pr.fence->GetCompletedValue() < pr.allocatorValue[slot]) {
            ThrowIfFailed(pr.fence->SetEventOnCompletion(pr.allocatorValue[slot], nullptr));
        }
        ThrowIfFailed(pr.allocators[slot]->Reset());
        ThrowIfFailed(pr.list->Reset(pr.allocators[slot].Get(), nullptr));
        ComPtr<ID3D12Resource> backBuffer;
        ThrowIfFailed(swapChain->GetBuffer(swapChain->GetCurrentBackBufferIndex(), IID_PPV_ARGS(&backBuffer)));
        auto toCopy = CD3DX12_RESOURCE_BARRIER::Transition(backBuffer.Get(), D3D12_RESOURCE_STATE_PRESENT,
                                                           D3D12_RESOURCE_STATE_COPY_DEST);
        pr.list->ResourceBarrier(1, &toCopy);
        const D3D12_RESOURCE_DESC sd = source->GetDesc();
        D3D12_BOX box{0, 0, 0, std::min<UINT>((UINT)sd.Width, scd.Width), std::min<UINT>(sd.Height, scd.Height), 1};
        CD3DX12_TEXTURE_COPY_LOCATION dst(backBuffer.Get(), 0), src(source, 0);
        pr.list->CopyTextureRegion(&dst, 0, 0, 0, &src, &box);
        auto toPresent = CD3DX12_RESOURCE_BARRIER::Transition(backBuffer.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                                                              D3D12_RESOURCE_STATE_PRESENT);
        pr.list->ResourceBarrier(1, &toPresent);
        ThrowIfFailed(pr.list->Close());

        // Frame generation's inputs: the ticket's depth and motion from the tracer, and the host's frame before its
        // UI, so the UI (where the two differ) is not interpolated.
        ID3D12Resource* scene = pr.hudless >= 0 && (size_t)pr.hudless < p.hostTextures.size() &&
                                        p.hostTextures[pr.hudless].resource
                                    ? p.hostTextures[pr.hudless].resource.Get()
                                    : source;
        sl::Resource depth(sl::ResourceType::eTex2d, dlss.Depth(), (uint32_t)D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        sl::Resource mvec(sl::ResourceType::eTex2d, dlss.MVec(), (uint32_t)D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        sl::Resource hudless(sl::ResourceType::eTex2d, scene, (uint32_t)D3D12_RESOURCE_STATE_COMMON);
        sl::Resource back(sl::ResourceType::eTex2d, backBuffer.Get(), (uint32_t)D3D12_RESOURCE_STATE_PRESENT);
        sl::Extent renderExtent{0, 0, dlss.RenderWidth(), dlss.RenderHeight()};
        sl::Extent displayExtent{0, 0, scd.Width, scd.Height};
        const auto life = sl::ResourceLifecycle::eValidUntilPresent;
        sl::ResourceTag tags[] = {
            {&depth, sl::kBufferTypeDepth, life, &renderExtent},
            {&mvec, sl::kBufferTypeMotionVectors, life, &renderExtent},
            {&hudless, sl::kBufferTypeHUDLessColor, life, &displayExtent},
            {&back, sl::kBufferTypeBackbuffer, life, &displayExtent},
            {nullptr, sl::kBufferTypeUIColorAndAlpha, life, &displayExtent},
        };
        slSetTagForFrame(*ctx.frameToken, ctx.viewportHandle, tags, _countof(tags), nullptr);

        // Reflex markers of the frame, in order (the host's simulation runs on its own threads).
        slPCLSetMarker(sl::PCLMarker::eSimulationStart, *ctx.frameToken);
        slPCLSetMarker(sl::PCLMarker::eSimulationEnd, *ctx.frameToken);
        slPCLSetMarker(sl::PCLMarker::eRenderSubmitStart, *ctx.frameToken);
        ID3D12CommandList* lists[] = {pr.list.Get()};
        ctx.CmdQueue()->ExecuteCommandLists(1, lists);
        slPCLSetMarker(sl::PCLMarker::eRenderSubmitEnd, *ctx.frameToken);
        slPCLSetMarker(sl::PCLMarker::ePresentStart, *ctx.frameToken);
        const HRESULT hr = swapChain->Present(pr.vsync ? 1 : 0,
                                              !pr.vsync && ctx.WindowTearing() ? DXGI_PRESENT_ALLOW_TEARING : 0);
        slPCLSetMarker(sl::PCLMarker::ePresentEnd, *ctx.frameToken);
        if (FAILED(hr))
            pr.lastError = "Present failed";
        ThrowIfFailed(ctx.CmdQueue()->Signal(pr.fence.Get(), ++pr.fenceValue));
        pr.allocatorValue[slot] = pr.fenceValue;
        ++pr.presents;
        // Frame generation's state belongs to the present thread.
        sl::DLSSGState state{};
        if (slDLSSGGetState(ctx.viewportHandle, state, nullptr) == sl::Result::eOk) {
            pr.displayed += state.numFramesActuallyPresented;
            pr.status = (uint32_t)state.status;
            pr.maxGenerated = state.numFramesToGenerateMax;
        }
    } catch (const std::exception& e) {
        pr.lastError = e.what();
    }
}

LRESULT CALLBACK PresenterWindowProc(HWND window, UINT message, WPARAM w, LPARAM l) {
    if (message == WM_NCHITTEST)
        return HTTRANSPARENT; // the host's window underneath takes the mouse
    if (message == WM_ERASEBKGND)
        return 1;
    return DefWindowProcW(window, message, w, l);
}

void UNITY_INTERFACE_API OnRenderEvent(int eventId, void* data) {
    Plugin* p = g_plugin;
    if (!p)
        return;
    const uint64_t index = (uint64_t)(uintptr_t)data;
    if (eventId == RT_EVENT_RENDER)
        RenderTicket(*p, index);
    else if (eventId == RT_EVENT_RELEASE && g_d3d12 && p->hostRelease)
        g_d3d12->GetCommandQueue()->Signal(p->hostRelease.Get(), index);
    else if (eventId == RT_EVENT_PRESENT)
        PresentTicket(*p, index);
}

void UNITY_INTERFACE_API OnGraphicsDeviceEvent(UnityGfxDeviceEventType type) {
    if (type == kUnityGfxDeviceEventInitialize && g_graphics->GetRenderer() == kUnityGfxRendererD3D12) {
        g_d3d12 = g_unity->Get<IUnityGraphicsD3D12v8>();
        if (!g_d3d12)
            return;
        // Queue access: the events wait and signal on the host's graphics queue.
        UnityD3D12PluginEventConfig config{};
        config.graphicsQueueAccess = kUnityD3D12GraphicsQueueAccess_Allow;
        config.flags = kUnityD3D12EventConfigFlag_FlushCommandBuffers | kUnityD3D12EventConfigFlag_SyncWorkerThreads;
        config.ensureActiveRenderTextureIsBound = false;
        g_d3d12->ConfigureEvent(RT_EVENT_RENDER, &config);
        g_d3d12->ConfigureEvent(RT_EVENT_RELEASE, &config);
        g_d3d12->ConfigureEvent(RT_EVENT_PRESENT, &config);
    } else if (type == kUnityGfxDeviceEventShutdown) {
        g_d3d12 = nullptr;
    }
}

template <class F> int32_t Guarded(const char* what, F&& f) {
    try {
        return f();
    } catch (const std::exception& e) {
        Fail(std::string(what) + ": " + e.what());
        return -1;
    }
}

} // namespace

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API UnityPluginLoad(IUnityInterfaces* unity) {
    g_unity = unity;
    g_graphics = unity->Get<IUnityGraphics>();
    g_graphics->RegisterDeviceEventCallback(OnGraphicsDeviceEvent);
    OnGraphicsDeviceEvent(kUnityGfxDeviceEventInitialize);
}

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API UnityPluginUnload() {
    if (g_graphics)
        g_graphics->UnregisterDeviceEventCallback(OnGraphicsDeviceEvent);
}

RT_API int32_t RT_Init(const RT_InitDesc* desc) {
    std::lock_guard initLock(g_initMutex);
    {
        std::lock_guard lock(g_logMutex);
        g_log = desc->log;
        g_lastError.clear();
    }
    if (g_plugin)
        return 1; // the device outlives play sessions; only the log target changes
    if (g_initTried) {
        Fail(std::string("Path tracing is unavailable: ") + g_caps.reason);
        return -1;
    }
    g_initTried = true;
    auto p = std::make_unique<Plugin>();
    auto report = [&](const char* reason) {
        const AdapterReport& a = p->ctx.adapterReport;
        g_caps.supported = reason ? 0 : 1;
        g_caps.vendorId = a.vendorId;
        g_caps.dedicatedVideoMemory = a.dedicatedVideoMemory;
        g_caps.shaderModel = (uint32_t)a.shaderModel;
        g_caps.raytracingTier = (uint32_t)a.raytracingTier;
        g_caps.serReorders = a.serReorders;
        g_caps.dlssRayReconstruction = a.dlssRayReconstruction;
        strncpy_s(g_caps.adapterName, Utf8(a.name).c_str(), _TRUNCATE);
        strncpy_s(g_caps.reason, reason ? reason : "", _TRUNCATE);
    };
    try {
        const std::wstring dataDir = desc->dataDir;
        // The renderer imports DXC by name; these two then satisfy the import.
        if (!LoadLibraryW((dataDir + L"\\dxil.dll").c_str()) || !LoadLibraryW((dataDir + L"\\dxcompiler.dll").c_str()))
            throw std::runtime_error("The plugin data folder is incomplete (dxil.dll/dxcompiler.dll missing)");
        nv_helpers_dx12::shader_paths::SourceDir() = dataDir + L"\\shaders";
        nv_helpers_dx12::shader_cache::Dir() = desc->cacheDir;
        std::wcout.rdbuf(g_wideForwarder);
        std::cout.rdbuf(g_narrowForwarder);

        HeadlessDeviceDesc dd;
        if (ID3D12Device* host = HostDevice())
            dd.adapterLuid = host->GetAdapterLuid();
        dd.sdkVersion = 619;
        dd.sdkPath = Utf8(dataDir + L"\\D3D12\\");
        dd.pluginDir = dataDir;
        dd.width = p->width;
        dd.height = p->height;
        dd.slLog = SlLog;
        p->ctx.InitHeadless(dd);
        ThrowIfFailed(p->ctx.CmdList()->Close());

        ID3D12Device* device = p->ctx.Device();
        ThrowIfFailed(device->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&p->done)));
        ThrowIfFailed(device->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&p->release)));
        p->hostDone = OpenOnHost<ID3D12Fence>(device, p->done.Get());
        p->hostRelease = OpenOnHost<ID3D12Fence>(device, p->release.Get());
        ThrowIfFailed(device->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&p->upload)));
        p->hostUpload = OpenOnHost<ID3D12Fence>(device, p->upload.Get());
    } catch (const std::exception& e) {
        report(e.what());
        Fail(std::string("Path tracing is unavailable: ") + e.what());
        return -1;
    }
    report(nullptr);
    Log(RT_LOG_INFO, std::string("RoyalTracer device ready on ") + g_caps.adapterName +
                         (HostDevice() ? " (sharing with the host device)" : " (no host device)"));
    g_plugin = p.release();
    return 1;
}

RT_API int32_t RT_GetCapabilities(RT_Capabilities* caps) {
    std::lock_guard initLock(g_initMutex);
    *caps = g_caps;
    return g_initTried ? 1 : 0;
}

RT_API void RT_Shutdown() {
    RT_PresenterEnable(0); // the window goes with the renderer that fed it
    std::lock_guard initLock(g_initMutex);
    if (g_plugin) {
        g_plugin->presenter.source = -1;
        std::lock_guard lock(g_plugin->mutex);
        if (g_plugin->renderer) {
            g_plugin->renderer->DestroyRenderer();
            g_plugin->renderer.reset();
        }
        g_plugin->host.reset();
        g_plugin->failed = false;
        g_plugin->water = RT_WaterDesc{};
        g_plugin->skyTexture = -1;
    }
    std::lock_guard lock(g_logMutex);
    g_log = nullptr;
}

RT_API const char* RT_GetLastError() {
    std::lock_guard lock(g_logMutex);
    return g_lastError.c_str();
}

namespace {

// The renderer starts with the first scene object or frame and lives until RT_Shutdown. Its static scene is a
// placeholder triangle far below the world; everything the host shows streams in through p.host.
void EnsureRenderer(Plugin& p) {
    if (p.renderer)
        return;
    DeviceContext& ctx = p.ctx;
    ctx.WaitForGPU();
    ctx.BeginFrame(); // reopens the shared command list for the uploads below
    auto host = std::make_unique<HostStream>(HostStream::Config{});
    auto renderer = std::make_unique<Renderer>(ctx, RendererConfig{p.width, p.height, true});
    renderer->InitDevice();
    StagedScene placeholder;
    RT_MaterialDesc grey{};
    grey.baseColor[0] = grey.baseColor[1] = grey.baseColor[2] = 0.5f;
    grey.baseColor[3] = 1.0f;
    grey.roughness = 1.0f;
    grey.ior = 1.5f;
    grey.alphaCutoff = 1.0f;
    grey.albedoTexture = grey.normalTexture = grey.rmaTexture = -1;
    grey.uvScale[0] = grey.uvScale[1] = 1.0f;
    placeholder.materials.push_back(grey);
    StagedScene::Mesh triangle;
    const XMFLOAT4 up(0.0f, 1.0f, 0.0f, 0.0f);
    triangle.vertices = {Vertex(XMFLOAT3(0.0f, 0.0f, 0.0f), up, XMFLOAT2(0.0f, 0.0f)),
                         Vertex(XMFLOAT3(0.001f, 0.0f, 0.0f), up, XMFLOAT2(1.0f, 0.0f)),
                         Vertex(XMFLOAT3(0.0f, 0.0f, 0.001f), up, XMFLOAT2(0.0f, 1.0f))};
    triangle.indices = {0, 1, 2};
    triangle.triangleMaterials = {0};
    placeholder.meshes.push_back(std::move(triangle));
    placeholder.instances.push_back({0, XMMatrixTranslation(0.0f, -100000.0f, 0.0f)});
    scene_bridge::Build(placeholder, renderer->GetScene(), ctx.Device(), ctx.CmdList(), [&] { ctx.FlushAndReset(); });
    ctx.FlushAndReset();
    renderer->AttachHostStream(host.get());
    renderer->InitSceneGPU();
    ThrowIfFailed(ctx.CmdList()->Close());
    p.host = std::move(host);
    p.renderer = std::move(renderer);
    p.failed = false;
    Log(RT_LOG_INFO, "Renderer ready");
}

// Scene calls that create: the renderer exists first, and a refusal becomes the last error.
template <class F> int32_t Create(const char* what, F&& f) {
    if (!g_plugin)
        return -1;
    Plugin& p = *g_plugin;
    std::lock_guard lock(p.mutex);
    return Guarded(what, [&] {
        EnsureRenderer(p);
        const int32_t id = f(*p.host);
        if (id < 0)
            throw std::runtime_error(p.host->Error());
        return id;
    });
}

template <class F> void Change(F&& f) {
    if (!g_plugin)
        return;
    std::lock_guard lock(g_plugin->mutex);
    if (g_plugin->host)
        f(*g_plugin->host);
}

} // namespace

RT_API int32_t RT_CreateTexture(const RT_TextureDesc* d) {
    return Create("Create texture", [&](HostStream& h) { return h.CreateTexture(*d); });
}
RT_API void RT_ReleaseTexture(int32_t id) {
    Change([&](HostStream& h) { h.ReleaseTexture(id); });
}
RT_API int32_t RT_CreateMaterial(const RT_MaterialDesc* d) {
    return Create("Create material", [&](HostStream& h) { return h.CreateMaterial(*d); });
}
RT_API int32_t RT_UpdateMaterial(int32_t id, const RT_MaterialDesc* d) {
    return Create("Update material", [&](HostStream& h) { return h.UpdateMaterial(id, *d) ? 1 : -1; });
}
RT_API void RT_ReleaseMaterial(int32_t id) {
    Change([&](HostStream& h) { h.ReleaseMaterial(id); });
}
RT_API int32_t RT_CreateMesh(const RT_MeshDesc* d) {
    return Create("Create mesh", [&](HostStream& h) { return h.CreateMesh(*d); });
}
RT_API void RT_ReleaseMesh(int32_t id) {
    Change([&](HostStream& h) { h.ReleaseMesh(id); });
}
RT_API int32_t RT_UpdateMeshVertices(int32_t id, const float* positions, const float* normals) {
    return Create("Update mesh vertices",
                  [&](HostStream& h) { return h.UpdateVertices(id, positions, normals) ? 1 : -1; });
}
RT_API int32_t RT_CreateInstance(int32_t mesh, const float* rowMajor3x4) {
    return Create("Create instance", [&](HostStream& h) { return h.CreateInstance(mesh, rowMajor3x4); });
}
RT_API void RT_ReleaseInstance(int32_t id) {
    Change([&](HostStream& h) { h.ReleaseInstance(id); });
}
RT_API void RT_SetInstanceFlags(int32_t id, uint32_t flags) {
    Change([&](HostStream& h) { h.SetFlags(id, flags); });
}
RT_API int32_t RT_CreateEmitterSet(uint32_t capacity) {
    return Create("Create emitter set", [&](HostStream& h) { return h.CreateEmitterSet(capacity); });
}
RT_API int32_t RT_UpdateEmitterSet(int32_t id, const RT_Emitter* emitters, uint32_t count) {
    return Create("Update emitter set",
                  [&](HostStream& h) { return h.UpdateEmitterSet(id, emitters, count) ? 1 : -1; });
}
RT_API void RT_ReleaseEmitterSet(int32_t id) {
    Change([&](HostStream& h) { h.ReleaseEmitterSet(id); });
}
RT_API void RT_SetInstanceVisible(int32_t id, int32_t visible) {
    Change([&](HostStream& h) { h.SetVisible(id, visible != 0); });
}
RT_API void RT_SetInstanceTransforms(int32_t count, const int32_t* instances, const float* rowMajor3x4s) {
    Change([&](HostStream& h) {
        for (int32_t i = 0; i < count; ++i)
            h.SetTransform(instances[i], rowMajor3x4s + 12 * i);
    });
}

RT_API const char* RT_GetStats() {
    static std::string stats;
    if (!g_plugin)
        return "";
    std::lock_guard lock(g_plugin->mutex);
    stats = g_plugin->host ? g_plugin->host->Stats() : std::string("no scene");
    return stats.c_str();
}

RT_API void RT_SetQuality(int32_t quality) {
    static const sl::DLSSMode modes[] = {sl::DLSSMode::eDLAA, sl::DLSSMode::eMaxQuality, sl::DLSSMode::eBalanced,
                                         sl::DLSSMode::eMaxPerformance, sl::DLSSMode::eUltraPerformance};
    if (!g_plugin || quality < 0 || quality >= (int32_t)_countof(modes))
        return;
    std::lock_guard lock(g_plugin->mutex);
    g_plugin->dlssMode = modes[quality];
}

RT_API int32_t RT_SubmitFrame(const RT_FrameDesc* desc, RT_FrameTicket* ticket) {
    if (!g_plugin)
        return -1;
    Plugin& p = *g_plugin;
    std::lock_guard lock(p.mutex);
    return Guarded("Submit frame", [&] {
        if (p.failed || !desc->width || !desc->height)
            return 0;
        p.width = desc->width;
        p.height = desc->height;
        EnsureRenderer(p);
        if (!p.outputs || p.outputs->width != desc->width || p.outputs->height != desc->height)
            p.outputs = CreateOutputs(p, desc->width, desc->height);
        const uint64_t index = ++p.submitted;
        Ticket& t = p.tickets[index % kTicketRing];
        t.index = index;
        t.frame = *desc;
        t.outputs = p.outputs;
        t.overlays = TicketOverlays{};
        ticket->index = index;
        ticket->slot = (uint32_t)(index & 1u);
        ticket->width = desc->width;
        ticket->height = desc->height;
        ticket->generation = p.outputs->generation;
        for (int s = 0; s < 2; ++s) {
            ticket->color[s] = p.outputs->hostColor[s].Get();
            ticket->depth[s] = p.outputs->hostDepth[s].Get();
        }
        return 1;
    });
}

RT_API void* RT_GetRenderEventFunc() { return (void*)&OnRenderEvent; }

RT_API int32_t RT_CreateHostTexture(uint32_t width, uint32_t height, int32_t dxgiFormat, void** hostTexture) {
    if (!g_plugin || !hostTexture)
        return -1;
    Plugin& p = *g_plugin;
    std::lock_guard lock(p.mutex);
    return Guarded("Create host texture", [&] {
        if (!HostDevice())
            throw std::runtime_error("no host device to share with");
        HostTexture t;
        auto desc = CD3DX12_RESOURCE_DESC::Tex2D((DXGI_FORMAT)dxgiFormat, width, height, 1, 1, 1, 0,
                                                 D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS);
        auto heap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
        ThrowIfFailed(p.ctx.Device()->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_SHARED, &desc,
                                                              D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                              IID_PPV_ARGS(&t.resource)));
        t.resource->SetName(L"HostTexture");
        t.host = OpenOnHost<ID3D12Resource>(p.ctx.Device(), t.resource.Get());
        *hostTexture = t.host.Get();
        auto free = std::find_if(p.hostTextures.begin(), p.hostTextures.end(), [](const HostTexture& h) { return !h.resource; });
        if (free == p.hostTextures.end())
            free = p.hostTextures.insert(free, HostTexture{});
        *free = std::move(t);
        return (int32_t)(free - p.hostTextures.begin());
    });
}

RT_API int32_t RT_SetWater(const RT_WaterDesc* desc) {
    if (!g_plugin || !desc)
        return -1;
    std::lock_guard lock(g_plugin->mutex);
    g_plugin->water = *desc;
    return 1;
}

RT_API int32_t RT_PresenterEnable(int32_t enable) {
    if (!g_plugin)
        return -1;
    Plugin& p = *g_plugin;
    std::lock_guard lock(p.mutex);
    Plugin::Presenter& pr = p.presenter;
    if (!enable) {
        if (!pr.window)
            return 1;
        sl::DLSSGOptions off{};
        off.mode = sl::DLSSGMode::eOff;
        slDLSSGSetOptions(p.ctx.viewportHandle, off);
        sl::ReflexOptions reflex{};
        reflex.mode = sl::ReflexMode::eOff;
        slReflexSetOptions(reflex);
        p.ctx.WaitForGPU();
        p.ctx.DetachWindow();
        DestroyWindow(pr.window);
        pr.window = pr.parent = nullptr;
        pr.optionsW = pr.optionsH = 0;
        return 1;
    }
    if (pr.window)
        return 1;
    return Guarded("Frame generation", [&] {
        if (!g_d3d12)
            throw std::runtime_error("no host device");
        IDXGISwapChain* hostSwapChain = g_d3d12->GetSwapChain();
        if (!hostSwapChain)
            throw std::runtime_error("it runs in player builds only (the editor shows no swap chain of its own)");
        ComPtr<IDXGISwapChain1> host1;
        ThrowIfFailed(hostSwapChain->QueryInterface(IID_PPV_ARGS(&host1)));
        HWND parent = nullptr;
        ThrowIfFailed(host1->GetHwnd(&parent));
        LUID luid = p.ctx.Device()->GetAdapterLuid();
        sl::AdapterInfo adapter;
        adapter.deviceLUID = (uint8_t*)&luid;
        adapter.deviceLUIDSizeInBytes = sizeof(LUID);
        const sl::Result supported = slIsFeatureSupported(sl::kFeatureDLSS_G, adapter);
        if (supported != sl::Result::eOk)
            throw std::runtime_error(std::string("DLSS frame generation is not supported here (") +
                                     sl::getResultAsStr(supported) + ")");

        static ATOM windowClass = [] {
            WNDCLASSEXW wc{sizeof(wc)};
            wc.lpfnWndProc = PresenterWindowProc;
            wc.hInstance = GetModuleHandleW(nullptr);
            wc.lpszClassName = L"RoyalTracerPresenter";
            return RegisterClassExW(&wc);
        }();
        RECT client{};
        GetClientRect(parent, &client);
        pr.width = (UINT)std::max<LONG>(client.right - client.left, 8);
        pr.height = (UINT)std::max<LONG>(client.bottom - client.top, 8);
        HWND window = CreateWindowExW(WS_EX_NOPARENTNOTIFY, (LPCWSTR)(ULONG_PTR)windowClass, L"RoyalTracer",
                                      WS_CHILD | WS_VISIBLE, 0, 0, (int)pr.width, (int)pr.height, parent, nullptr,
                                      GetModuleHandleW(nullptr), nullptr);
        if (!window)
            throw std::runtime_error("the child window could not be created");
        try {
            p.ctx.AttachWindow(window, pr.width, pr.height);
        } catch (...) {
            DestroyWindow(window);
            throw;
        }
        pr.window = window;
        pr.parent = parent;
        ID3D12Device* device = p.ctx.Device();
        if (!pr.fence) {
            ThrowIfFailed(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&pr.fence)));
            for (auto& allocator : pr.allocators)
                ThrowIfFailed(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)));
            ThrowIfFailed(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, pr.allocators[0].Get(), nullptr,
                                                    IID_PPV_ARGS(&pr.list)));
            ThrowIfFailed(pr.list->Close());
            ThrowIfFailed(device->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&pr.upload)));
            pr.hostUpload = OpenOnHost<ID3D12Fence>(device, pr.upload.Get());
        }
        sl::ReflexOptions reflex{};
        reflex.mode = sl::ReflexMode::eLowLatency; // frame generation needs it
        slReflexSetOptions(reflex);
        pr.lastError.clear();
        return 1;
    });
}

RT_API void RT_PresenterUpdate() {
    if (!g_plugin)
        return;
    Plugin::Presenter& pr = g_plugin->presenter;
    if (!pr.window)
        return;
    RECT client{};
    GetClientRect(pr.parent, &client);
    const UINT w = (UINT)std::max<LONG>(client.right - client.left, 8), h = (UINT)std::max<LONG>(client.bottom - client.top, 8);
    if (w == pr.width && h == pr.height)
        return;
    SetWindowPos(pr.window, HWND_TOP, 0, 0, (int)w, (int)h, SWP_NOACTIVATE);
    std::lock_guard lock(g_plugin->mutex);
    pr.width = w;
    pr.height = h;
}

RT_API int32_t RT_PresenterSource(int32_t finished, int32_t hudless) {
    if (!g_plugin)
        return -1;
    std::lock_guard lock(g_plugin->mutex);
    g_plugin->presenter.source = finished;
    g_plugin->presenter.hudless = hudless;
    return 1;
}

RT_API void RT_PresenterOptions(int32_t framesToGenerate, int32_t vsync) {
    if (!g_plugin)
        return;
    std::lock_guard lock(g_plugin->mutex);
    g_plugin->presenter.frames = (uint32_t)std::clamp(framesToGenerate, 0, 5);
    g_plugin->presenter.vsync = vsync != 0;
}

RT_API int32_t RT_FrameGenerationMax() {
    if (!g_plugin)
        return 0;
    Plugin& p = *g_plugin;
    std::lock_guard lock(p.mutex);
    // Frame generation's state is only asked for once it presents: asked earlier, it takes the hidden swap chain
    // as its own, and letting that go later never returns.
    if (p.presenter.maxGenerated)
        return (int32_t)p.presenter.maxGenerated;
    static int32_t supported = -1;
    if (supported < 0) {
        LUID luid = p.ctx.Device()->GetAdapterLuid();
        sl::AdapterInfo adapter;
        adapter.deviceLUID = (uint8_t*)&luid;
        adapter.deviceLUIDSizeInBytes = sizeof(LUID);
        supported = slIsFeatureSupported(sl::kFeatureDLSS_G, adapter) == sl::Result::eOk ? 1 : 0;
    }
    return supported ? -1 : 0;
}

RT_API const char* RT_PresenterStats() {
    static std::string stats;
    if (!g_plugin)
        return "";
    Plugin& p = *g_plugin;
    std::lock_guard lock(p.mutex);
    Plugin::Presenter& pr = p.presenter;
    if (!pr.window)
        return "off";
    stats = "presents " + std::to_string(pr.presents) + ", displayed " + std::to_string(pr.displayed) + ", window " +
            std::to_string(pr.width) + " by " + std::to_string(pr.height) + ", DLSS-G status " + std::to_string(pr.status) +
            (pr.lastError.empty() ? "" : ", error: " + pr.lastError);
    return stats.c_str();
}

RT_API int32_t RT_SetSky(int32_t texture, float scale) {
    if (!g_plugin)
        return -1;
    std::lock_guard lock(g_plugin->mutex);
    g_plugin->skyTexture = texture;
    g_plugin->skyScale = std::max(scale, 0.0f);
    return 1;
}

RT_API int32_t RT_SetWaterOverlays(uint64_t ticket, int32_t heightAtlas, int32_t foamAtlas, uint32_t count,
                                   const RT_WaterOverlay* overlays) {
    if (!g_plugin || (count && !overlays))
        return -1;
    Plugin& p = *g_plugin;
    std::lock_guard lock(p.mutex);
    Ticket& t = p.tickets[ticket % kTicketRing];
    if (t.index != ticket)
        return 0;
    const auto texture = [&](int32_t id) {
        return id >= 0 && (size_t)id < p.hostTextures.size() ? p.hostTextures[id].resource : nullptr;
    };
    t.overlays.height = texture(heightAtlas);
    t.overlays.foam = texture(foamAtlas);
    t.overlays.list.assign(overlays, overlays + std::min<uint32_t>(count, RT_MAX_WATER_OVERLAYS));
    return 1;
}

RT_API void RT_ReleaseHostTexture(int32_t id) {
    if (!g_plugin)
        return;
    Plugin& p = *g_plugin;
    std::lock_guard lock(p.mutex);
    if (id < 0 || (size_t)id >= p.hostTextures.size() || !p.hostTextures[id].resource)
        return;
    // Both sides wait: the host queue's last writes to it come before the renderer's work for p.submitted.
    p.retiredHostTextures.emplace_back(p.submitted, std::move(p.hostTextures[id]));
    p.hostTextures[id] = HostTexture{};
}

RT_API int32_t RT_RenderTicketForTest(uint64_t index, float* rgbaOut, float* depthOut) {
    if (!g_plugin)
        return -1;
    Plugin& p = *g_plugin;
    RenderTicket(p, index);
    std::lock_guard lock(p.mutex);
    return Guarded("Test readback", [&] {
        const Ticket& t = p.tickets[index % kTicketRing];
        if (p.failed || t.index != index)
            return 0;
        DeviceContext& ctx = p.ctx;
        ctx.WaitForGPU();
        ID3D12Device* device = ctx.Device();
        const uint32_t slot = (uint32_t)(index & 1u);
        ID3D12Resource* sources[2] = {t.outputs->color[slot].Get(), t.outputs->depth[slot].Get()};
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp[2]{};
        UINT64 bytes[2]{};
        ComPtr<ID3D12Resource> readback[2];
        ComPtr<ID3D12CommandAllocator> allocator;
        ComPtr<ID3D12GraphicsCommandList> list;
        ThrowIfFailed(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)));
        ThrowIfFailed(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr,
                                                IID_PPV_ARGS(&list)));
        for (int i = 0; i < 2; ++i) {
            const D3D12_RESOURCE_DESC rd = sources[i]->GetDesc();
            device->GetCopyableFootprints(&rd, 0, 1, 0, &fp[i], nullptr, nullptr, &bytes[i]);
            auto heap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK);
            auto bd = CD3DX12_RESOURCE_DESC::Buffer(bytes[i]);
            ThrowIfFailed(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &bd,
                                                          D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                          IID_PPV_ARGS(&readback[i])));
            CD3DX12_TEXTURE_COPY_LOCATION dst(readback[i].Get(), fp[i]), src(sources[i], 0);
            list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        }
        ThrowIfFailed(list->Close());
        ID3D12CommandList* lists[] = {list.Get()};
        ctx.CmdQueue()->ExecuteCommandLists(1, lists);
        ctx.WaitForGPU();
        const UINT w = t.outputs->width, h = t.outputs->height;
        uint8_t* data = nullptr;
        ThrowIfFailed(readback[0]->Map(0, nullptr, (void**)&data));
        for (UINT y = 0; y < h; ++y) {
            const auto* row = (const DirectX::PackedVector::HALF*)(data + fp[0].Offset + (size_t)y * fp[0].Footprint.RowPitch);
            for (UINT x = 0; x < w * 4; ++x)
                rgbaOut[(size_t)y * w * 4 + x] = DirectX::PackedVector::XMConvertHalfToFloat(row[x]);
        }
        readback[0]->Unmap(0, nullptr);
        ThrowIfFailed(readback[1]->Map(0, nullptr, (void**)&data));
        for (UINT y = 0; y < h; ++y)
            memcpy(depthOut + (size_t)y * w, data + fp[1].Offset + (size_t)y * fp[1].Footprint.RowPitch, w * 4);
        readback[1]->Unmap(0, nullptr);
        return 1;
    });
}
