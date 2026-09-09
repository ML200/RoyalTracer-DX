#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <d3d12sdklayers.h>
#include "../rdn/PostProcess/DLSSNRManager.h"
#include "../rdn/PostProcess/DLSSManager.h"
#include <sl.h>
#include <sl_helpers.h>
#include <cstdio>
#include <numeric>
#include <fstream>
#include <functional>

#if !PATHTRACER_DLSSNR_NATIVE
int main() {
    DLSSNRManager manager; manager.Initialize(1920, 1080); manager.settings.enabled = true;
    manager.PrepareFrameGPUIdle();
    if (manager.GetStatus().backend != DLSSNRManager::BackendState::eStubNoSdk ||
        manager.Evaluate(nullptr, nullptr, nullptr, 0, nullptr, nullptr, 1280, 720)) return 1;
    manager.Shutdown(nullptr);
    puts("PASS: build-time disabled backend links without the bridge and stays inactive.");
    return 0;
}
#else
namespace {
void Require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void Check(HRESULT hr) { if (FAILED(hr)) { char error[40]; sprintf_s(error, "D3D12 0x%08X", (unsigned)hr); throw std::runtime_error(error); } }
struct Gpu {
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> fence;
    ComPtr<ID3D12InfoQueue> messages;
    HANDLE event = nullptr;
    uint64_t serial = 0;
    Gpu() {
        ComPtr<ID3D12Debug> debug;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)))) debug->EnableDebugLayer();
        ComPtr<IDXGIFactory4> factory; Check(CreateDXGIFactory1(IID_PPV_ARGS(&factory)));
        ComPtr<IDXGIAdapter1> adapter;
        for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
            DXGI_ADAPTER_DESC1 desc{}; adapter->GetDesc1(&desc);
            if (desc.VendorId == 0x10de) break;
            adapter.Reset();
        }
        Require(adapter != nullptr, "NVIDIA GPU required");
        Check(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device)));
        device.As(&messages);
        D3D12_COMMAND_QUEUE_DESC qd{}; Check(device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue)));
        Check(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)));
        Check(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr, IID_PPV_ARGS(&list)));
        Check(list->Close());
        Check(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));
        event = CreateEventW(nullptr, FALSE, FALSE, nullptr); Require(event != nullptr, "Fence event failed");
    }
    ~Gpu() { if (event) CloseHandle(event); }
    void Begin() { Check(allocator->Reset()); Check(list->Reset(allocator.Get(), nullptr)); }
    void Submit() {
        Check(list->Close()); ID3D12CommandList* lists[] = { list.Get() }; queue->ExecuteCommandLists(1, lists);
        Check(queue->Signal(fence.Get(), ++serial)); Check(fence->SetEventOnCompletion(serial, event));
        Require(WaitForSingleObject(event, 30000) == WAIT_OBJECT_0, "GPU timeout");
        Check(device->GetDeviceRemovedReason());
        if (messages) {
            for (UINT64 i = 0; i < messages->GetNumStoredMessages(); ++i) {
                SIZE_T size = 0; messages->GetMessage(i, nullptr, &size);
                std::vector<char> bytes(size); auto* msg = reinterpret_cast<D3D12_MESSAGE*>(bytes.data());
                messages->GetMessage(i, msg, &size);
                if (msg->Severity <= D3D12_MESSAGE_SEVERITY_ERROR) {
                    fprintf(stderr, "D3D12: %s\n", msg->pDescription);
                    throw std::runtime_error("D3D12 validation error");
                }
            }
            messages->ClearStoredMessages();
        }
    }
};
struct Inputs {
    UINT width, height, rw, rh;
    ComPtr<ID3D12Resource> color, depth, motion, readback;
    ComPtr<ID3D12DescriptorHeap> cpu, gpu;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT64 totalBytes = 0;
    Inputs(Gpu& g, UINT w, UINT h, UINT renderW, UINT renderH) : width(w), height(h), rw(renderW), rh(renderH) {
        D3D12_DESCRIPTOR_HEAP_DESC hd{}; hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV; hd.NumDescriptors = 3;
        Check(g.device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&cpu)));
        hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE; Check(g.device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&gpu)));
        const UINT stride = g.device->GetDescriptorHandleIncrementSize(hd.Type);
        auto heap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
        ComPtr<ID3D12Resource>* textures[] = { &color, &depth, &motion };
        DXGI_FORMAT formats[] = { DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R32_FLOAT, DXGI_FORMAT_R16G16_FLOAT };
        g.Begin(); ID3D12DescriptorHeap* heaps[] = { gpu.Get() }; g.list->SetDescriptorHeaps(1, heaps);
        for (UINT i = 0; i < 3; ++i) {
            auto desc = CD3DX12_RESOURCE_DESC::Tex2D(formats[i], i ? rw : width, i ? rh : height, 1, 1, 1, 0, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
            Check(g.device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(textures[i]->ReleaseAndGetAddressOf())));
            D3D12_UNORDERED_ACCESS_VIEW_DESC view{}; view.Format = formats[i]; view.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
            auto cpuHandle = CD3DX12_CPU_DESCRIPTOR_HANDLE(cpu->GetCPUDescriptorHandleForHeapStart(), i, stride);
            auto gpuCpu = CD3DX12_CPU_DESCRIPTOR_HANDLE(gpu->GetCPUDescriptorHandleForHeapStart(), i, stride);
            g.device->CreateUnorderedAccessView(textures[i]->Get(), nullptr, &view, cpuHandle);
            g.device->CreateUnorderedAccessView(textures[i]->Get(), nullptr, &view, gpuCpu);
            float clear[3][4] = { {0.2f, 0.4f, 0.6f, 1.0f}, {0.01f, 0, 0, 0}, {0, 0, 0, 0} };
            g.list->ClearUnorderedAccessViewFloat(CD3DX12_GPU_DESCRIPTOR_HANDLE(gpu->GetGPUDescriptorHandleForHeapStart(), i, stride), cpuHandle, textures[i]->Get(), clear[i], 0, nullptr);
        }
        auto barrier = CD3DX12_RESOURCE_BARRIER::Transition(color.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        g.list->ResourceBarrier(1, &barrier); g.Submit();
        auto colorDesc = color->GetDesc();
        g.device->GetCopyableFootprints(&colorDesc, 0, 1, 0, &footprint, nullptr, nullptr, &totalBytes);
        auto readHeap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK); auto buffer = CD3DX12_RESOURCE_DESC::Buffer(totalBytes);
        Check(g.device->CreateCommittedResource(&readHeap, D3D12_HEAP_FLAG_NONE, &buffer, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback)));
    }
    void SetMotion(Gpu& g, float x, float y) {
        g.Begin(); ID3D12DescriptorHeap* heaps[] = { gpu.Get() }; g.list->SetDescriptorHeaps(1, heaps);
        const UINT stride = g.device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        const float value[4] = {x, y, 0, 0};
        g.list->ClearUnorderedAccessViewFloat(
            CD3DX12_GPU_DESCRIPTOR_HANDLE(gpu->GetGPUDescriptorHandleForHeapStart(), 2, stride),
            CD3DX12_CPU_DESCRIPTOR_HANDLE(cpu->GetCPUDescriptorHandleForHeapStart(), 2, stride), motion.Get(), value, 0, nullptr);
        auto barrier = CD3DX12_RESOURCE_BARRIER::UAV(motion.Get());
        g.list->ResourceBarrier(1, &barrier); g.Submit();
    }
    uint64_t Evaluate(Gpu& g, DLSSNRManager& manager, bool expected, const std::function<void()>& beforeNR = {}) {
        manager.PrepareFrameGPUIdle(); g.Begin();
        if (beforeNR) beforeNR();
        bool evaluated = manager.Evaluate(g.list.Get(), g.device.Get(), color.Get(), 0, depth.Get(), motion.Get(), rw, rh);
        // Always submit: feature creation can record work even on fallback.
        auto* source = evaluated ? manager.Output() : color.Get();
        CD3DX12_TEXTURE_COPY_LOCATION src(source, 0), dst(readback.Get(), footprint);
        g.list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr); g.Submit();
        if (evaluated != expected) fprintf(stderr, "NR status: %s / %s\n", manager.GetStatus().backendText.c_str(), manager.GetStatus().lastResult.c_str());
        Require(evaluated == expected, "Unexpected evaluation/fallback result");
        unsigned char* pixels = nullptr; D3D12_RANGE range{0, static_cast<SIZE_T>(totalBytes)};
        Check(readback->Map(0, &range, reinterpret_cast<void**>(&pixels)));
        uint64_t sum = 0;
        for (UINT y = 0; y < height; ++y) for (UINT x = 0; x < width; ++x) for (UINT c = 0; c < 3; ++c)
            sum += pixels[y * footprint.Footprint.RowPitch + x * 4 + c];
        D3D12_RANGE noWrite{0,0}; readback->Unmap(0, &noWrite);
        Require(sum > 0 && sum < uint64_t(width) * height * 3 * 255, "Empty or saturated output");
        if (!expected && !beforeNR) Require(sum == uint64_t(width) * height * (51 + 102 + 153), "Fallback modified source color");
        printf("%ux%u guides %ux%u %s sum=%llu\n", width, height, rw, rh, evaluated ? "NR" : "fallback", (unsigned long long)sum);
        return sum;
    }
};

// Uses the engine's actual RR manager, followed by an opaque SDR conversion and
// NR on the same command list. RR must also survive NR shutdown on this device.
void CheckRRAndNR(Gpu& g, const wchar_t* runtime) {
    SetEnvironmentVariableW(L"DLSSNR_RUNTIME_PATH", runtime);
    DLSSManager rr; rr.mode = sl::DLSSMode::eMaxQuality; rr.CreateResources(g.device.Get(), 1280, 720);
    Inputs input(g, 1280, 720, rr.RenderWidth(), rr.RenderHeight());
    input.depth = rr.Depth(); input.motion = rr.MVec();
    DLSSNRManager nr; nr.Initialize(1280, 720); nr.settings.enabled = true;
    ComPtr<ID3D12DescriptorHeap> cpu, gpu;
    D3D12_DESCRIPTOR_HEAP_DESC hd{}; hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV; hd.NumDescriptors = 14;
    Check(g.device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&cpu)));
    hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE; Check(g.device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&gpu)));
    const UINT stride = g.device->GetDescriptorHandleIncrementSize(hd.Type);
    ID3D12Resource* guides[] = {rr.Input(), rr.Depth(), rr.MVec(), rr.Normals(), rr.DiffuseAlbedo(), rr.SpecularAlbedo(),
        rr.Roughness(), rr.SpecMVec(), rr.SpecHitDist(), rr.Transparency(), rr.ColorBeforeTrans(), rr.BiasHint()};
    g.Begin(); ID3D12DescriptorHeap* heaps[] = {gpu.Get()}; g.list->SetDescriptorHeaps(1, heaps);
    for (UINT i = 0; i < _countof(guides); ++i) {
        D3D12_UNORDERED_ACCESS_VIEW_DESC view{}; view.Format = guides[i]->GetDesc().Format; view.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        auto c = CD3DX12_CPU_DESCRIPTOR_HANDLE(cpu->GetCPUDescriptorHandleForHeapStart(), i, stride);
        g.device->CreateUnorderedAccessView(guides[i], nullptr, &view, c);
        g.device->CreateUnorderedAccessView(guides[i], nullptr, &view, CD3DX12_CPU_DESCRIPTOR_HANDLE(gpu->GetCPUDescriptorHandleForHeapStart(), i, stride));
        float clear[4] = {0,0,0,0};
        if (i == 0) { clear[0] = .2f; clear[1] = .4f; clear[2] = .6f; clear[3] = 1; }
        if (i == 1) clear[0] = .01f;
        if (i == 3) { clear[2] = 1; clear[3] = .6f; }
        if (i == 4) clear[0] = clear[1] = clear[2] = .5f;
        if (i == 5) clear[0] = clear[1] = clear[2] = .04f;
        if (i == 6) clear[0] = .6f;
        g.list->ClearUnorderedAccessViewFloat(CD3DX12_GPU_DESCRIPTOR_HANDLE(gpu->GetGPUDescriptorHandleForHeapStart(), i, stride), c, guides[i], clear, 0, nullptr);
    }
    auto ready = CD3DX12_RESOURCE_BARRIER::UAV(nullptr); g.list->ResourceBarrier(1, &ready); g.Submit();
    D3D12_SHADER_RESOURCE_VIEW_DESC srv{}; srv.Format = rr.Output()->GetDesc().Format;
    srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D; srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING; srv.Texture2D.MipLevels = 1;
    g.device->CreateShaderResourceView(rr.Output(), &srv, CD3DX12_CPU_DESCRIPTOR_HANDLE(gpu->GetCPUDescriptorHandleForHeapStart(), 12, stride));
    D3D12_UNORDERED_ACCESS_VIEW_DESC uav{}; uav.Format = input.color->GetDesc().Format; uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    g.device->CreateUnorderedAccessView(input.color.Get(), nullptr, &uav, CD3DX12_CPU_DESCRIPTOR_HANDLE(gpu->GetCPUDescriptorHandleForHeapStart(), 13, stride));
    CD3DX12_DESCRIPTOR_RANGE ranges[2]; ranges[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0); ranges[1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0);
    CD3DX12_ROOT_PARAMETER params[2]; for (int i = 0; i < 2; ++i) params[i].InitAsDescriptorTable(1, &ranges[i]);
    CD3DX12_ROOT_SIGNATURE_DESC signature(2, params);
    ComPtr<ID3DBlob> serialized, error; Check(D3D12SerializeRootSignature(&signature, D3D_ROOT_SIGNATURE_VERSION_1, &serialized, &error));
    ComPtr<ID3D12RootSignature> root; Check(g.device->CreateRootSignature(0, serialized->GetBufferPointer(), serialized->GetBufferSize(), IID_PPV_ARGS(&root)));
    std::ifstream shaderFile("DLSSNRIntegration.dxil", std::ios::binary);
    std::vector<char> shader((std::istreambuf_iterator<char>(shaderFile)), std::istreambuf_iterator<char>());
    Require(!shader.empty(), "Missing RR-to-NR conversion shader");
    D3D12_COMPUTE_PIPELINE_STATE_DESC pd{}; pd.pRootSignature = root.Get(); pd.CS = {shader.data(), shader.size()};
    ComPtr<ID3D12PipelineState> pso; Check(g.device->CreateComputePipelineState(&pd, IID_PPV_ARGS(&pso)));
    sl::FrameToken* token = nullptr; sl::ViewportHandle viewport(17); uint32_t frame = 0;
    auto reconstruct = [&]() {
        Require(slGetNewFrameToken(token, nullptr) == sl::Result::eOk, "Frame token failed");
        const auto view = XMMatrixIdentity(), projection = XMMatrixPerspectiveFovRH(XMConvertToRadians(60), 1280.f / 720.f, .1f, 10000.f);
        rr.Evaluate(g.list.Get(), g.device.Get(), *token, viewport, 1280.f / 720.f, view, view, projection, 0, 0, ++frame, 60, .1f, 10000);
        if (g.messages) g.messages->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_ERROR, FALSE);
        Require(rr.LastEvaluationSucceeded(), "RR evaluation failed alongside NR");
        if (rr.LastEvaluationReset()) nr.ForceReset();
        D3D12_RESOURCE_BARRIER barriers[] = {
            CD3DX12_RESOURCE_BARRIER::Transition(rr.Output(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
            CD3DX12_RESOURCE_BARRIER::Transition(input.color.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)};
        g.list->ResourceBarrier(2, barriers); g.list->SetDescriptorHeaps(1, heaps);
        g.list->SetComputeRootSignature(root.Get()); g.list->SetPipelineState(pso.Get());
        for (UINT i = 0; i < 2; ++i) g.list->SetComputeRootDescriptorTable(i, CD3DX12_GPU_DESCRIPTOR_HANDLE(gpu->GetGPUDescriptorHandleForHeapStart(), 12 + i, stride));
        g.list->Dispatch(160, 90, 1);
        for (auto& b : barriers) std::swap(b.Transition.StateBefore, b.Transition.StateAfter);
        g.list->ResourceBarrier(2, barriers);
    };
    input.Evaluate(g, nr, false, reconstruct); // RR + NR creation
    input.Evaluate(g, nr, true, reconstruct);
    input.Evaluate(g, nr, true, reconstruct);
    Require(!rr.LastEvaluationReset(), "RR unexpectedly reset steady history");
    rr.ForceReset(); input.Evaluate(g, nr, true, reconstruct);
    Require(rr.LastEvaluationReset(), "RR reset signal was lost");
    nr.Shutdown(g.device.Get()); nr.settings.enabled = false;
    input.Evaluate(g, nr, false, reconstruct); // independent RR still works after NR shutdown
    Require(slFreeResources(sl::kFeatureDLSS_RR, viewport) == sl::Result::eOk, "RR resource release failed");
    puts("PASS: engine RR -> opaque SDR conversion -> NR on the same command list; RR reset propagation and RR after NR shutdown.");
}
}

int wmain(int argc, wchar_t** argv) {
    if (argc != 2) { puts("Usage: DLSSNRGpuTests.exe absolute-path-to-nvngx_dlssnr.dll"); return 2; }
    try {
        SetEnvironmentVariableW(L"DLSSNR_RUNTIME_PATH", argv[1]);
        sl::Preferences preferences{};
        sl::Feature features[] = { sl::kFeatureDLSS, sl::kFeatureDLSS_RR };
        preferences.featuresToLoad = features; preferences.numFeaturesToLoad = 2;
        preferences.engine = sl::EngineType::eCustom; preferences.engineVersion = "1.0.0";
        preferences.projectId = "b3ae1e2f-6cb7-4bb0-99a5-27fa8a920d3e";
        preferences.renderAPI = sl::RenderAPI::eD3D12;
        preferences.flags = sl::PreferenceFlags::eUseManualHooking | sl::PreferenceFlags::eDisableCLStateTracking |
                            sl::PreferenceFlags::eUseFrameBasedResourceTagging;
        Require(slInit(preferences, sl::kSDKVersion) == sl::Result::eOk, "Streamline init failed");
        Gpu g; Require(slSetD3DDevice(g.device.Get()) == sl::Result::eOk, "Streamline device failed");
        LUID luid = g.device->GetAdapterLuid(); sl::AdapterInfo ai{}; ai.deviceLUID = reinterpret_cast<uint8_t*>(&luid); ai.deviceLUIDSizeInBytes = sizeof(luid);
        for (auto feature : features) Require(slIsFeatureSupported(feature, ai) == sl::Result::eOk, "SR/RR support regressed");
        DLSSNRManager manager; manager.Initialize(1920, 1080);
        Inputs inputs(g, 1920, 1080, 1280, 720);
        inputs.Evaluate(g, manager, false); // disabled
        manager.settings.enabled = true;
        inputs.Evaluate(g, manager, false); // create, submit, wait
        auto first = inputs.Evaluate(g, manager, true);
        inputs.Evaluate(g, manager, true); // temporal frame
        manager.ForceReset();
        Require(inputs.Evaluate(g, manager, true) == first, "Reset did not reproduce initial frame");
        inputs.SetMotion(g, 3.0f, -2.0f);
        inputs.Evaluate(g, manager, true); // nonzero guide-pixel vectors with lower-resolution guides
        inputs.SetMotion(g, 0.0f, 0.0f); manager.ForceReset();
        Require(inputs.Evaluate(g, manager, true) == first, "Reset after motion did not restore initial frame");
        for (int style = 0; style < dlssnr::kModelStyleCount; ++style) {
            for (int preset = 0; preset < dlssnr::kRenderPresetCount; ++preset) {
                if (style == 0 && preset == 0) continue; // already exercised above
                manager.settings.modelStyle = style; manager.settings.renderPreset = preset;
                inputs.Evaluate(g, manager, false); // every selection change must recreate before evaluation
                const auto selected = inputs.Evaluate(g, manager, true);
                if (preset == 0) Require(selected != first, "Model-style selection did not affect NR output");
                manager.ForceReset();
                Require(inputs.Evaluate(g, manager, true) == selected, "Selected NR model/preset reset was not deterministic");
                printf("NR selection: style=%d preset=%d sum=%llu\n", style, preset, (unsigned long long)selected);
            }
        }
        manager.settings.modelStyle = 0; manager.settings.renderPreset = 0;
        inputs.Evaluate(g, manager, false);
        Require(inputs.Evaluate(g, manager, true) == first, "Returning to default retained the previous model/preset");
        manager.settings.modelStyle = -1; manager.settings.renderPreset = 99;
        inputs.Evaluate(g, manager, true); // invalid settings normalize to the already-active defaults
        manager.settings.modelStyle = 0; manager.settings.renderPreset = 0;
        manager.settings.intensity = 0.5f;
        inputs.Evaluate(g, manager, false); // recreate for create-time tuning
        inputs.Evaluate(g, manager, true);
        manager.settings.enabled = false; inputs.Evaluate(g, manager, false);
        manager.settings.modelStyle = 2; manager.settings.renderPreset = 3;
        inputs.Evaluate(g, manager, false); // select while disabled
        manager.settings.enabled = true; inputs.Evaluate(g, manager, false); inputs.Evaluate(g, manager, true);
        manager.OnDisplayResolution(1280, 720);
        Inputs resized(g, 1280, 720, 1280, 720);
        resized.Evaluate(g, manager, false); resized.Evaluate(g, manager, true);
        Inputs lowRes(g, 1280, 720, 854, 480);
        lowRes.Evaluate(g, manager, true); // guide resize resets history without recreating output
        g.Begin();
        Require(!manager.Evaluate(g.list.Get(), g.device.Get(), lowRes.color.Get(), 0, lowRes.depth.Get(), lowRes.motion.Get(), 999, 480), "Invalid guide dimensions accepted");
        g.Submit();
        manager.settings.modelStyle = 1;
        lowRes.Evaluate(g, manager, false); // a changed selection retries a failed feature at the idle fence
        lowRes.Evaluate(g, manager, true);
        manager.Shutdown(g.device.Get());
        SetEnvironmentVariableW(L"DLSSNR_RUNTIME_PATH", L"Z:\\missing-test-runtime\\nvngx_dlssnr.dll");
        DLSSNRManager missing; missing.Initialize(1280, 720); missing.settings.enabled = true;
        lowRes.Evaluate(g, missing, false); missing.Shutdown(g.device.Get());
        dlssnr::Context* invalid = nullptr;
        Require(dlssnr::RoyalNRCreateContext(L"Z:\\missing-test-runtime.dll", g.device.Get(), &invalid) != 1 && invalid == nullptr, "Unknown runtime accepted");
        wchar_t wrongRuntime[32768]{}; GetModuleFileNameW(nullptr, wrongRuntime, _countof(wrongRuntime));
        Require(dlssnr::RoyalNRCreateContext(wrongRuntime, g.device.Get(), &invalid) != 1 && invalid == nullptr, "Runtime hash mismatch accepted");
        CheckRRAndNR(g, argv[1]);
        Require(slShutdown() == sl::Result::eOk, "Streamline shutdown failed");
        printf("PASS: SR/RR initialization alongside NR, NR GPU output, nonzero motion, all model styles/presets, default restoration, deterministic reset, tuning, toggles, output/guide resize, invalid inputs, missing runtime; D3D12 debug validation %s.\n", g.messages ? "enabled" : "unavailable");
        return 0;
    } catch (const std::exception& error) { fprintf(stderr, "FAIL: %s\n", error.what()); return 1; }
}
#endif
