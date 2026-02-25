//*********************************************************
//
// Copyright (c) Microsoft. All rights reserved.
// This code is licensed under the MIT License (MIT).
// THIS CODE IS PROVIDED *AS IS* WITHOUT WARRANTY OF
// ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING ANY
// IMPLIED WARRANTIES OF FITNESS FOR A PARTICULAR
// PURPOSE, MERCHANTABILITY, OR NON-INFRINGEMENT.
//
//*********************************************************
#define ENABLE_D3D12_DIAGNOSTICS        1
#include <chrono>
#include "stdafx.h"
#include <unordered_map>
#include "Renderer.h"

#include "DXRHelper.h"
#include "nv_helpers_dx12/BottomLevelASGenerator.h"

#include "nv_helpers_dx12/RaytracingPipelineGenerator.h"
#include "nv_helpers_dx12/RootSignatureGenerator.h"

#include "Windowsx.h"
#include "glm/gtc/type_ptr.hpp"
#include "manipulator.h"
#include "../src/Util/ObjLoader.h"
#include "Diagnostics.h"
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>

static float Halton(uint32_t index, uint32_t base) {
    float f = 1.0f;
    float r = 0.0f;
    while (index > 0) {
        f = f / base;
        r = r + f * (index % base);
        index = index / base;
    }
    return r;
}

#define SL_CHECK(x) do { sl::Result r = (x); if (r != sl::Result::eOk) { \
std::wcout << L"[SL] " << L#x << L" failed: " << (int)r << std::endl; return; } \
} while(0)

void Renderer::EnsureSLViewportAllocated(ID3D12GraphicsCommandList* cmdList)
{
    if (m_viewportHandle) return;
    if (!cmdList) return;

    sl::Result r = slAllocateResources(cmdList, sl::kFeatureDLSS_RR, m_viewportHandle);
    if (r != sl::Result::eOk)
    {
        std::wcout << L"[SL] slAllocateResources failed: " << (int)r << std::endl;
    }
}

void PrintAdapterDetails() {
    ComPtr<IDXGIFactory4> factory;
    CreateDXGIFactory2(0, IID_PPV_ARGS(&factory));

    ComPtr<IDXGIAdapter1> adapter;
    std::wcout << L"\n--- System Adapters ---" << std::endl;

    for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC1 desc;
        adapter->GetDesc1(&desc);

        std::wcout << L"Adapter [" << i << L"]: " << desc.Description << std::endl;
        std::wcout << L"  Vendor ID: 0x" << std::hex << desc.VendorId << std::dec << std::endl;
        std::wcout << L"  LUID: " << desc.AdapterLuid.HighPart << L":" << desc.AdapterLuid.LowPart << std::endl;

        if (desc.VendorId == 0x10DE) { // NVIDIA
            std::wcout << L"  >>> MATCH FOUND (NVIDIA) <<<" << std::endl;
        }
        std::wcout << L"-----------------------" << std::endl;
    }
}


void DumpD3D12Messages(ID3D12Device* device)
{
    ComPtr<ID3D12InfoQueue> info;
    if (FAILED(device->QueryInterface(IID_PPV_ARGS(&info)))) {
        std::wcout << L"No debug layer." << std::endl;
        return;
    }

    const UINT64 n = info->GetNumStoredMessages();
    for (UINT64 i = 0; i < n; ++i)
    {
        SIZE_T size = 0;
        info->GetMessage(i, nullptr, &size);      // get required size
        std::vector<char> bytes(size);
        auto* msg = reinterpret_cast<D3D12_MESSAGE*>(bytes.data());
        info->GetMessage(i, msg, &size);          // get the message

        // --- NEW: Silently ignore Streamline's internal mvec barrier mismatch ---
        if (msg->pDescription != nullptr &&
            strstr(msg->pDescription, "sl.dlss_d.mvec") != nullptr &&
            strstr(msg->pDescription, "RESOURCE_BARRIER_BEFORE_AFTER_MISMATCH") != nullptr)
        {
            continue; // Skip printing this specific spam!
        }

        std::wcout << L"[DX] " << msg->pDescription << std::endl;
    }
    info->ClearStoredMessages();
}

struct ScopedTimer {
    const char* name;
    std::chrono::high_resolution_clock::time_point t0;
    ScopedTimer(const char* n) : name(n), t0(std::chrono::high_resolution_clock::now()) {}
    ~ScopedTimer(){
        using namespace std::chrono;
        auto ms = duration_cast<milliseconds>(high_resolution_clock::now() - t0).count();
        std::wcout << L"[CPU] " << name << L" took " << ms << L" ms" << std::endl;
    }
};
#define SCOPE_TIMER(label) ScopedTimer _scopedTimer_##__LINE__(label)


static std::chrono::steady_clock::time_point g_lastRenderTime
    = std::chrono::steady_clock::now();
static const float FRAME_INTERVAL_SECONDS = 1.00f;

extern "C" {
    __declspec(dllexport) extern const UINT  D3D12SDKVersion = 717;
    __declspec(dllexport) extern const char* D3D12SDKPath    = ".\\";
}

static inline UINT EncodeNormalOct(const XMVECTOR& n)
{
    XMVECTOR p = n / (abs(XMVectorGetX(n)) + abs(XMVectorGetY(n)) + abs(XMVectorGetZ(n)));

    if (XMVectorGetZ(p) < 0.0f)
    {
        float oldX = XMVectorGetX(p);
        float oldY = XMVectorGetY(p);
        p = XMVectorSetX(p, (1.0f - abs(oldY)) * (oldX >= 0.0f ? 1.0f : -1.0f));
        p = XMVectorSetY(p, (1.0f - abs(oldX)) * (oldY >= 0.0f ? 1.0f : -1.0f));
    }
    int ix = static_cast<int>(XMVectorGetX(p) * 32767.0f);
    int iy = static_cast<int>(XMVectorGetY(p) * 32767.0f);

    // Pack into a single UINT
    return (static_cast<uint16_t>(iy) << 16) | static_cast<uint16_t>(ix);
}

void Renderer::BuildGlobalMeshBuffers()
{
    SCOPE_TIMER("BuildGlobalMeshBuffers");
    m_geoOffsets.clear();
    size_t totalVerts = 0, totalIdx = 0;
    m_geoOffsets.resize(m_VB.size());
    for (size_t m = 0; m < m_VB.size(); ++m) {
        m_geoOffsets[m].vertexBase = (UINT)totalVerts;
        m_geoOffsets[m].indexBase = (UINT)totalIdx;
        m_geoOffsets[m].materialBase = m_materialIDOffsets[m];
        totalVerts += m_VertexCount[m];
        totalIdx += m_IndexCount[m];
    }
    m_totalVertexCount = (UINT)totalVerts;
    m_totalIndexCount = (UINT)totalIdx;

    const UINT vbBytes = m_totalVertexCount * sizeof(BTriVertex);
    const UINT ibBytes = m_totalIndexCount * sizeof(uint32_t);

    auto makeDefault = [&](UINT bytes, ComPtr<ID3D12Resource>& dst, const wchar_t* name)
    {
        dst = nv_helpers_dx12::CreateBuffer(
            m_device.Get(), bytes, D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_COPY_DEST, nv_helpers_dx12::kDefaultHeapProps);
        dst->SetName(name);
    };
    makeDefault(vbBytes, m_vertexGlobal, L"GlobalVertexBuffer");
    makeDefault(ibBytes, m_indexGlobal, L"GlobalIndexBuffer");

    ComPtr<ID3D12Resource> upload = nv_helpers_dx12::CreateBuffer(
        m_device.Get(), vbBytes + ibBytes, D3D12_RESOURCE_FLAG_NONE,
        D3D12_RESOURCE_STATE_GENERIC_READ, nv_helpers_dx12::kUploadHeapProps);
    upload->SetName(L"GlobalMeshUploadBuffer");

    uint8_t* pUpload = nullptr; { CD3DX12_RANGE r(0, 0); upload->Map(0, &r, (void**)&pUpload); }
    auto* dstVerts = reinterpret_cast<BTriVertex*>(pUpload + 0);
    auto* dstIdx = reinterpret_cast<uint32_t*>(pUpload + vbBytes);

    for (size_t m = 0; m < m_cpuVertexData.size(); ++m)
    {
        const std::vector<Vertex>& srcV = m_cpuVertexData[m];
        const std::vector<UINT>&   srcI = m_cpuIndexData[m];

        const UINT vCount = m_VertexCount[m];
        const UINT iCount = m_IndexCount[m];
        const UINT vBase = m_geoOffsets[m].vertexBase;
        const UINT iBase = m_geoOffsets[m].indexBase;

        BTriVertex* outV = dstVerts + vBase;
        for (UINT i = 0; i < vCount; ++i) {
            outV[i].vertex = srcV[i].position;
            XMVECTOR normal = XMVectorSet(srcV[i].normal_material.x, srcV[i].normal_material.y, srcV[i].normal_material.z, 0.0f);
            normal = XMVector3Normalize(normal);
            outV[i].packedNormal = EncodeNormalOct(normal);

            XMVECTOR texCoord = XMLoadFloat2(&srcV[i].texCoord);
            PackedVector::XMStoreHalf2(&outV[i].texCoord, texCoord);
        }
        uint32_t* outI = dstIdx + iBase;
        for (UINT i = 0; i < iCount; ++i) outI[i] = srcI[i] + vBase;
    }

    upload->Unmap(0, nullptr);
    m_commandList->CopyBufferRegion(m_vertexGlobal.Get(), 0, upload.Get(), 0, vbBytes);
    m_commandList->CopyBufferRegion(m_indexGlobal.Get(), 0, upload.Get(), vbBytes, ibBytes);

    CD3DX12_RESOURCE_BARRIER br[] = {
        CD3DX12_RESOURCE_BARRIER::Transition(m_vertexGlobal.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_GENERIC_READ),
        CD3DX12_RESOURCE_BARRIER::Transition(m_indexGlobal.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_GENERIC_READ),
    };
    m_commandList->ResourceBarrier(_countof(br), br);
}

static InstanceXformCPU ToInstanceXform(const XMMATRIX& M)
{
    InstanceXformCPU x{};
    XMStoreFloat4x4(&x.objectToWorld, M);
    return x;
}

void Renderer::CreateTriToLightIdBuffer()
{
    if (m_triToLightId.empty()) return;

    const UINT bytes = static_cast<UINT>(m_triToLightId.size() * sizeof(uint32_t));

    // Upload
    ComPtr<ID3D12Resource> upload = nv_helpers_dx12::CreateBuffer(
        m_device.Get(), bytes, D3D12_RESOURCE_FLAG_NONE,
        D3D12_RESOURCE_STATE_GENERIC_READ, nv_helpers_dx12::kUploadHeapProps);

    {   void* p = nullptr; CD3DX12_RANGE r(0,0);
        ThrowIfFailed(upload->Map(0, &r, &p));
        memcpy(p, m_triToLightId.data(), bytes);
        upload->Unmap(0, nullptr);
    }

    // Default
    m_triToLightIdBuffer = nv_helpers_dx12::CreateBuffer(
        m_device.Get(), bytes, D3D12_RESOURCE_FLAG_NONE,
        D3D12_RESOURCE_STATE_COPY_DEST, nv_helpers_dx12::kDefaultHeapProps);

    m_commandList->CopyBufferRegion(m_triToLightIdBuffer.Get(), 0, upload.Get(), 0, bytes);

    CD3DX12_RESOURCE_BARRIER br = CD3DX12_RESOURCE_BARRIER::Transition(
        m_triToLightIdBuffer.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_GENERIC_READ);
    m_commandList->ResourceBarrier(1, &br);
}



Renderer::Renderer(UINT width, UINT height,
                   std::wstring name)
    : DXSample(width, height, name), m_frameIndex(0),
      m_viewport(0.0f, 0.0f, static_cast<float>(width),
                 static_cast<float>(height)),
      m_scissorRect(0, 0, static_cast<LONG>(width), static_cast<LONG>(height)),
      m_rtvDescriptorSize(0) {

    m_passSequence = {
        L"Pass_raygen_v8.hlsl|rg",
        L"barrier",
        L"Pass_temp_di_v8.hlsl|cs:16x8",
        L"barrier",
        L"Pass_temp_gi_v8.hlsl|cs:16x8",
        L"barrier",
        L"Pass_spat_di_v8.hlsl|cs:16x16",
        L"barrier",
        L"Pass_spat_gi_v8_1.hlsl|cs:16x16",
        L"barrier",
        L"Pass_shading_v8.hlsl|cs:16x16",
        L"barrier"
        /*L"ml",
        L"barrier",
        L"Pass_finalize_v8.hlsl|cs:16x16",
        L"barrier",*/,
        /*L"Pass_denoiser_firefly_v8.hlsl|cs:16x16",
        L"barrier",
        L"Pass_denoiser_temp_v8.hlsl|cs:16x16",
        L"barrier",
        L"Pass_denoiser_blur_1_v8.hlsl|cs:16x16",
        L"barrier",
        L"Pass_denoiser_blur_2_v8.hlsl|cs:16x16",
        L"barrier",
        L"Pass_denoiser_blur_3_v8.hlsl|cs:16x16",
        L"barrier",
        L"Pass_denoiser_copy_v8.hlsl|cs:8x4",*/
    };

    try {
        m_passes.clear();
        for (auto& s : m_passSequence)
            m_passes.push_back(ParsePass(s));
        LinkLoops();
    }
    catch (const std::exception& e) {
        MessageBoxA(nullptr, e.what(), "Pass Parsing Error", MB_OK | MB_ICONERROR);
        exit(1);
    }
}

void Renderer::OnInit() {
    //try :
        m_mod = LoadLibrary("sl.interposer.dll");
        LoadPipeline();

        m_simulator.PromptUserConfiguration();
        m_recorder.Initialize();
        nv_helpers_dx12::CameraManip.setWindowSize(GetWidth(), GetHeight());
        nv_helpers_dx12::CameraManip.setLookat(
            glm::vec3(-1.5f, 1.5f, 3.5f), glm::vec3(0, 1.0f, 0), glm::vec3(0, 1, 0));
        nv_helpers_dx12::CameraManip.setMode(
        nv_helpers_dx12::Manipulator::Fly);
        nv_helpers_dx12::CameraManip.setSpeed(0.0f);

        try {
            LoadAssets();
        }
        catch (const std::exception& e) {
            MessageBoxA(nullptr, e.what(), "An exception occurred", MB_OK | MB_ICONERROR);
            exit(1);
        }

        EnsureSLViewportAllocated(m_commandList.Get());

        GenerateLutTextures();
        CheckRaytracingSupport();

        CreateAccelerationStructures();
        BuildGlobalMeshBuffers();
        ThrowIfFailed(m_commandList->Close());

        ID3D12CommandList* initLists[] = { m_commandList.Get() };
        m_commandQueue->ExecuteCommandLists(_countof(initLists), initLists);
        WaitForPreviousFrame();

        m_textureUploadHeaps.clear();
        m_lutUploadHeaps.clear();

        CreateRaytracingPipeline();

        CreateStreamingCompactionBuffers();
        CreateIndirectCommandSignature();
        CompileSetupIndirectShader();

        CreatePerInstanceConstantBuffers();
        CreateGlobalConstantBuffer();
        CreateRaytracingOutputBuffer();
        CreateMLBuffers();
        CreateReadbackBuffer();
        CreateInstancePropertiesBuffer();
        CreateCameraBuffer();

        CreateDLSSResources();

        CreateShaderResourceHeap();
        CreateShaderBindingTable();
        slGetNewFrameToken(m_frameToken, nullptr);
    /*}
    catch (const std::exception& e) {
        // THIS CATCHES THE CRASH AND SHOWS THE ERROR
        wchar_t wMsg[4096];
        MultiByteToWideChar(CP_UTF8, 0, e.what(), -1, wMsg, 4096);
        MessageBoxW(NULL, wMsg, L"Fatal Initialization Error", MB_OK | MB_ICONERROR);
        exit(1);
    }*/

}

// Load the rendering pipeline dependencies.
void Renderer::LoadPipeline() {

    // 1. Initialize Streamline FIRST
    sl::Preferences pref{};
    // [FIX] Explicitly tell SL we are using D3D12. Without this, plugins may fail to identify the adapter.
    pref.renderAPI = sl::RenderAPI::eD3D12;
    pref.engine = sl::EngineType::eCustom;
    pref.applicationId = 231313132;

    pref.showConsole = true;                            // Spawns a console window specifically for SL logs
    pref.logLevel = sl::LogLevel::eVerbose;             // Maximum log detail

    pref.flags = sl::PreferenceFlags::eLoadDownloadedPlugins
           | sl::PreferenceFlags::eUseFrameBasedResourceTagging;
    static sl::Feature featList[] = { sl::kFeatureDLSS, sl::kFeatureDLSS_RR};
    pref.featuresToLoad = featList;
    pref.numFeaturesToLoad = _countof(featList);

    // Check this result!
    sl::Result res = slInit(pref, sl::kSDKVersion);
    if (res != sl::Result::eOk) {
        std::wcout << L"slInit failed! Error: " << (int)res << std::endl;
    }

    UINT dxgiFactoryFlags = 0;
    typedef HRESULT(WINAPI* PFunCreateDXGIFactory)(REFIID, void**);
    typedef HRESULT(WINAPI* PFunCreateDXGIFactory1)(REFIID, void**);
    typedef HRESULT(WINAPI* PFunCreateDXGIFactory2)(UINT, REFIID, void**);
    typedef HRESULT(WINAPI* PFunDXGIGetDebugInterface1)(UINT, REFIID, void**);
    typedef HRESULT(WINAPI* PFunD3D12CreateDevice)(IUnknown* , D3D_FEATURE_LEVEL, REFIID , void**);
    auto slCreateDXGIFactory = reinterpret_cast<PFunCreateDXGIFactory>(GetProcAddress(m_mod, "CreateDXGIFactory"));
    auto slCreateDXGIFactory1 = reinterpret_cast<PFunCreateDXGIFactory1>(GetProcAddress(m_mod, "CreateDXGIFactory1"));
    auto slCreateDXGIFactory2 = reinterpret_cast<PFunCreateDXGIFactory2>(GetProcAddress(m_mod, "CreateDXGIFactory2"));
    auto slDXGIGetDebugInterface1 = reinterpret_cast<PFunDXGIGetDebugInterface1>(GetProcAddress(m_mod, "DXGIGetDebugInterface1"));
    auto slD3D12CreateDevice = reinterpret_cast<PFunD3D12CreateDevice>(GetProcAddress(m_mod, "D3D12CreateDevice"));


    static const UUID D3D12ExperimentalShaderModels =
    { 0x76f5573e, 0xf13a, 0x40f5, { 0xb2, 0x97, 0x81, 0xce, 0x9e, 0x18, 0x93, 0x3f } };


    HRESULT hr = D3D12EnableExperimentalFeatures(1, &D3D12ExperimentalShaderModels, nullptr, nullptr);
    if (FAILED(hr)) {
        // CRITICAL FAILURE
        MessageBoxA(nullptr, "Failed to enable Experimental Shader Models.\nMake sure Windows Developer Mode is ON.", "Error", MB_OK | MB_ICONERROR);
        exit(1);
    }

  ComPtr<IDXGIFactory4> factory;
  ThrowIfFailed(slCreateDXGIFactory2(dxgiFactoryFlags, IID_PPV_ARGS(&factory)));

#if ENABLE_D3D12_DIAGNOSTICS
    dxdiag::EnableDebugLayerAndDred();
#endif

  if (m_useWarpDevice) {
    ComPtr<IDXGIAdapter> warpAdapter;
    ThrowIfFailed(factory->EnumWarpAdapter(IID_PPV_ARGS(&warpAdapter)));

    ThrowIfFailed(slD3D12CreateDevice(warpAdapter.Get(), D3D_FEATURE_LEVEL_12_1,
                                    IID_PPV_ARGS(&m_device)));
  } else {
    ComPtr<IDXGIAdapter1> hardwareAdapter;
    GetHardwareAdapter(factory.Get(), &hardwareAdapter);

    ThrowIfFailed(slD3D12CreateDevice(hardwareAdapter.Get(),
                                    D3D_FEATURE_LEVEL_12_1,
                                    IID_PPV_ARGS(&m_device)));
    #if ENABLE_D3D12_DIAGNOSTICS
          dxdiag::HookDevice(m_device.Get());
    #endif
  }
    res = slSetD3DDevice(m_device.Get());
    if (res != sl::Result::eOk) {
        std::wcout << L"slSetD3DDevice failed! Code: " << (int)res << std::endl;
    }

    // 1. Store the LUID in a local variable (an l-value)
    LUID adapterLuid = m_device->GetAdapterLuid();

    PrintAdapterDetails();

    sl::AdapterInfo adapterInfo;
    adapterInfo.deviceLUID = (uint8_t*)&adapterLuid;
    adapterInfo.deviceLUIDSizeInBytes = sizeof(LUID);

    // Log what we are sending to NVIDIA
    std::wcout << L"[SL] Sending LUID to SL: " << adapterLuid.HighPart << L":" << adapterLuid.LowPart << std::endl;

    sl::Result supportRes = slIsFeatureSupported(sl::kFeatureDLSS_RR, adapterInfo);
    if (supportRes != sl::Result::eOk) {
        std::wcout << L"[DLSS-RR] Check Failed! Error: " << (int)supportRes << std::endl;
    }


    // Using helpers from sl_dlss.h
    sl::DLSSOptimalSettings dlssSettings;
    sl::DLSSOptions dlssOptions;
    dlssOptions.mode = sl::DLSSMode::eDLAA;
    dlssOptions.outputWidth = m_width;
    dlssOptions.outputHeight = m_height;
    slDLSSGetOptimalSettings(dlssOptions, dlssSettings);
    std::wcout << L"DLSS settings: " << dlssSettings.renderHeightMax << std::endl;

  D3D12_COMMAND_QUEUE_DESC queueDesc = {};
  queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
  queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;

  ThrowIfFailed(
      m_device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&m_commandQueue)));

    InitML_ONNX_DML(L"./models/denoiser_model_fp16.onnx");

  // Describe and create the swap chain.
  DXGI_SWAP_CHAIN_DESC1 swapChainDesc = {};
  swapChainDesc.BufferCount = FrameCount;
  swapChainDesc.Width = m_width;
  swapChainDesc.Height = m_height;
  swapChainDesc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
  swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
  swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
  swapChainDesc.SampleDesc.Count = 1;

  ComPtr<IDXGISwapChain1> swapChain;
  ThrowIfFailed(factory->CreateSwapChainForHwnd(
      m_commandQueue.Get(),
      Win32Application::GetHwnd(), &swapChainDesc, nullptr, nullptr,
      &swapChain));
  ThrowIfFailed(factory->MakeWindowAssociation(Win32Application::GetHwnd(),
                                               DXGI_MWA_NO_ALT_ENTER));

  ThrowIfFailed(swapChain.As(&m_swapChain));
  m_frameIndex = m_swapChain->GetCurrentBackBufferIndex();

  // Create descriptor heaps.
  {
    // Describe and create a render target view (RTV) descriptor heap.
    D3D12_DESCRIPTOR_HEAP_DESC rtvHeapDesc = {};
    rtvHeapDesc.NumDescriptors = FrameCount;
    rtvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    rtvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
    ThrowIfFailed(
        m_device->CreateDescriptorHeap(&rtvHeapDesc, IID_PPV_ARGS(&m_rtvHeap)));

    m_rtvDescriptorSize = m_device->GetDescriptorHandleIncrementSize(
        D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
  }

  // Create frame resources.
  {
    CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHandle(
        m_rtvHeap->GetCPUDescriptorHandleForHeapStart());

    // Create a RTV for each frame.
    for (UINT n = 0; n < FrameCount; n++) {
      ThrowIfFailed(
          m_swapChain->GetBuffer(n, IID_PPV_ARGS(&m_renderTargets[n])));
      m_device->CreateRenderTargetView(m_renderTargets[n].Get(), nullptr,
                                       rtvHandle);
      rtvHandle.Offset(1, m_rtvDescriptorSize);
    }
  }

    for (UINT n = 0; n < FrameCount; ++n)
    {
        ThrowIfFailed(m_device->CreateCommandAllocator(
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                IID_PPV_ARGS(&m_commandAllocators[n])));
    }
  CreateDepthBuffer();
}

void Renderer::LoadAssets() {
    ThrowIfFailed(m_device->CreateCommandList(
        0, D3D12_COMMAND_LIST_TYPE_DIRECT, m_commandAllocators[m_frameIndex].Get(),
        nullptr, IID_PPV_ARGS(&m_commandList)));

    std::map<std::string, uint32_t> textureMap;
    std::vector<TextureData> albedoTextures;
    std::vector<TextureData> normalTextures;
    std::vector<TextureData> rmaTextures;

    // Model loading
    // Model loading
    {
        std::vector<std::string> models = {"./testScene_2/testScene_2.obj", /*"./workshop/workshop.obj",*/ /*"./chungmu/chungmu.obj"*/};
        for (const auto& modelName : models) {

            std::string material_search_path = "./";
            const auto last_slash_idx = modelName.find_last_of("/\\");
            if (std::string::npos != last_slash_idx) {
                material_search_path = modelName.substr(0, last_slash_idx + 1);
            }

            std::vector<Vertex> vertices;
            std::vector<UINT> indices;
            std::vector<Material> modelScopedMaterials;
            std::vector<UINT> modelMaterialIDs;

            ObjLoader::loadObjFile(
                modelName, &vertices, &indices, &modelScopedMaterials, &modelMaterialIDs,
                &materialIDOffset, &materialVertexOffset,
                textureMap, albedoTextures, normalTextures, rmaTextures, material_search_path
            );

            m_materialIDOffsets.push_back(static_cast<UINT>(m_materialIDs.size()));
            m_materialIDs.insert(m_materialIDs.end(), modelMaterialIDs.begin(), modelMaterialIDs.end());
            m_materials.insert(m_materials.end(), modelScopedMaterials.begin(), modelScopedMaterials.end());

            // Create Vertex Buffer for this model
            const UINT vbSize = static_cast<UINT>(vertices.size()) * sizeof(Vertex);
            ComPtr<ID3D12Resource> vb;
            CD3DX12_RESOURCE_DESC vbDesc = CD3DX12_RESOURCE_DESC::Buffer(vbSize);
            ThrowIfFailed(m_device->CreateCommittedResource(
                &nv_helpers_dx12::kUploadHeapProps, D3D12_HEAP_FLAG_NONE, &vbDesc,
                D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&vb)));
            UINT8* pVertexDataBegin;
            vb->Map(0, nullptr, reinterpret_cast<void**>(&pVertexDataBegin));
            memcpy(pVertexDataBegin, vertices.data(), vbSize);
            vb->Unmap(0, nullptr);
            m_VB.push_back(vb);
            m_VBView.push_back({ vb->GetGPUVirtualAddress(), vbSize, sizeof(Vertex) });
            m_VertexCount.push_back(static_cast<UINT>(vertices.size()));

            // Create Index Buffer for this model
            const UINT ibSize = static_cast<UINT>(indices.size()) * sizeof(UINT);
            ComPtr<ID3D12Resource> ib;
            CD3DX12_RESOURCE_DESC ibDesc = CD3DX12_RESOURCE_DESC::Buffer(ibSize);
            ThrowIfFailed(m_device->CreateCommittedResource(
                &nv_helpers_dx12::kUploadHeapProps, D3D12_HEAP_FLAG_NONE, &ibDesc,
                D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&ib)));
            UINT8* pIndexDataBegin;
            ib->Map(0, nullptr, reinterpret_cast<void**>(&pIndexDataBegin));
            memcpy(pIndexDataBegin, indices.data(), ibSize);
            ib->Unmap(0, nullptr);
            m_IB.push_back(ib);
            m_IBView.push_back({ ib->GetGPUVirtualAddress(), ibSize, DXGI_FORMAT_R32_UINT });
            m_IndexCount.push_back(static_cast<UINT>(indices.size()));

            m_cpuVertexData.push_back(std::move(vertices));
            m_cpuIndexData.push_back(std::move(indices));
        }
    }

    // Upload material data
    {
        const UINT materialBufferSize = static_cast<UINT>(m_materials.size()) * sizeof(Material);
        CD3DX12_RESOURCE_DESC materialBufferDesc = CD3DX12_RESOURCE_DESC::Buffer(materialBufferSize);
        ThrowIfFailed(m_device->CreateCommittedResource(
            &nv_helpers_dx12::kUploadHeapProps, D3D12_HEAP_FLAG_NONE, &materialBufferDesc,
            D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&m_materialBuffer)));
        UINT8* pMaterialDataBegin;
        m_materialBuffer->Map(0, nullptr, reinterpret_cast<void**>(&pMaterialDataBegin));
        memcpy(pMaterialDataBegin, m_materials.data(), materialBufferSize);
        m_materialBuffer->Unmap(0, nullptr);
    }
    {
        const UINT materialIndexBufferSize = static_cast<UINT>(m_materialIDs.size()) * sizeof(UINT);
        CD3DX12_RESOURCE_DESC materialIndexBufferDesc = CD3DX12_RESOURCE_DESC::Buffer(materialIndexBufferSize);
        ThrowIfFailed(m_device->CreateCommittedResource(
            &nv_helpers_dx12::kUploadHeapProps, D3D12_HEAP_FLAG_NONE, &materialIndexBufferDesc,
            D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&m_materialIndexBuffer)));
        UINT8* pMaterialIndexDataBegin;
        m_materialIndexBuffer->Map(0, nullptr, reinterpret_cast<void**>(&pMaterialIndexDataBegin));
        memcpy(pMaterialIndexDataBegin, m_materialIDs.data(), materialIndexBufferSize);
        m_materialIndexBuffer->Unmap(0, nullptr);
    }

    // Create and upload GPU texture arrays
    CreateTextureArrays(albedoTextures, normalTextures, rmaTextures);

    {
        ThrowIfFailed(m_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_fence)));
        m_fenceValue = 1;
        m_fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
        if (m_fenceEvent == nullptr) {
            ThrowIfFailed(HRESULT_FROM_WIN32(GetLastError()));
        }
        WaitForPreviousFrame();
    }
}

void Renderer::OnInitTransform() {
    /*XMMATRIX scale        = XMMatrixScaling(1.0f, 1.0f, 1.0f);
    XMMATRIX selfRotation = XMMatrixRotationAxis({0.f, 1.f, 0.f}, 0.0f);
    XMMATRIX translate    = XMMatrixTranslation(0.0f, 0.f, 0.0f);

    m_instances[2].second = scale * selfRotation * translate;*/

    /*XMMATRIX scaleMatrix_1 = XMMatrixScaling(0.3f, 0.3f, 0.3f);
    XMMATRIX rotationMatrix_1 = XMMatrixRotationAxis({0.f, 1.f, 0.f}, 1.1f);
    XMMATRIX translationMatrix_1 = XMMatrixTranslation(-8.f, 1.0f, 3.f);

    m_instances[1].second = scaleMatrix_1 * rotationMatrix_1 * translationMatrix_1;*/

    XMMATRIX scaleMatrix_2 = XMMatrixScaling(1.0f, 1.0f, 1.0f);
    XMMATRIX rotationMatrix_2 = XMMatrixRotationAxis({0.f, 1.f, 0.f}, 0.0);
    XMMATRIX translationMatrix_2 = XMMatrixTranslation(0.f, 0.f, 3.f);

    m_instances[0].second = scaleMatrix_2 * rotationMatrix_2 * translationMatrix_2;
}

// Update frame-based values.
void Renderer::OnUpdate() {
    using clock   = std::chrono::high_resolution_clock;
    static auto tPrev = clock::now();

    auto  tCurr = clock::now();
    float dt    =
        std::chrono::duration<float>(tCurr - tPrev).count(); // seconds
    tPrev = tCurr;

    // --- NEW LOGIC START ---
    if (m_simulator.IsActive()) {
        bool shouldCapture = false;

        // Pass 'shouldCapture' by reference.
        // The simulator sets this to true only when the wait time is over.
        bool finished = m_simulator.Update(dt, nv_helpers_dx12::CameraManip, shouldCapture);

        if (shouldCapture) {
            // Simulator just finished convergence. Save frame N-1.
            const size_t idx = m_simulator.GetLastCaptureIndex();
            SaveSimulationData(idx);
        }

        if (finished) {
            std::wcout << L"\n[Sim] Data Generation Complete. Closing Application.\n";
            PostQuitMessage(0);
            return;
        }
    }
    else {
        // --- EXISTING INPUT LOGIC (Wrap your existing code in this else) ---
        glm::vec3 eye, center, up;
        nv_helpers_dx12::CameraManip.getLookat(eye, center, up);

        glm::vec3 fwd   = glm::normalize(center - eye);
        glm::vec3 right = glm::normalize(glm::cross(fwd, up));
        glm::vec3 move(0.0f);
        float speed = 5.0f;

        if (g_keys['W'])          move +=  fwd;
        if (g_keys['S'])          move -=  fwd;
        if (g_keys['D'])          move +=  right;
        if (g_keys['A'])          move -=  right;
        if (g_keys[VK_SPACE])     move +=  up;
        if (g_keys[VK_CONTROL])   move -=  up;

        if (glm::length(move) > 0.0f)
        {
            move = glm::normalize(move) * (speed * dt);
            eye    += move;
            center += move;
            nv_helpers_dx12::CameraManip.setLookat(eye, center, up);
        }
    }
  // #DXR Extra: Perspective Camera
  UpdateCameraBuffer();


  // #DXR Extra - Refitting
  m_time++;
  /*m_instances[1].second =
      XMMatrixRotationAxis({0.f, 1.f, 0.f},*/
                           //0.0f/*static_cast<float>(m_time) / 20000000.0f*/) *
      //XMMatrixTranslation(0.f, 0.f, 0.f);

    /*float angle = static_cast<float>(m_time) * 0.00f;
    float r     = 4.0f;

    float x = 0.0f;//cosf(angle) * r + 1.0f;   // + centre.x
    float z = 0.0f;//sinf(angle) * r + 0.0f;   // + centre.z

    XMMATRIX scale        = XMMatrixScaling(10.0f, 10.0f, 10.0f);
    XMMATRIX selfRotation = XMMatrixRotationY(angle);
    XMMATRIX translate    = XMMatrixTranslation(x, 0.f, z);

    m_instances[1].second = scale * selfRotation * translate;*/

    /*XMMATRIX scaleMatrix_1 = XMMatrixScaling(1.0f, 1.0f, 1.0f);
    XMMATRIX rotationMatrix_1 = selfRotation;//XMMatrixRotationAxis({0.f, 1.f, 0.f}, 0.785f);
    XMMATRIX translationMatrix_1 = XMMatrixTranslation(0.f, 1.f, 1.f);

    m_instances[1].second = scaleMatrix_1 * rotationMatrix_1 * translationMatrix_1;

    XMMATRIX scaleMatrix_2 = XMMatrixScaling(1.0f, 1.0f, 1.0f);
    XMMATRIX rotationMatrix_2 = XMMatrixRotationAxis({0.f, 1.f, 0.f}, 0.0);
    XMMATRIX translationMatrix_2 = XMMatrixTranslation(0.f, 0.f, 0.f);

    m_instances[0].second = scaleMatrix_2 * rotationMatrix_2 * translationMatrix_2;*/
  // #DXR Extra - Refitting
  UpdateInstancePropertiesBuffer();
}

void Renderer::OnRender()
{
    if (m_frameToken) {
        slGetNewFrameToken(m_frameToken, nullptr);
    }

    static auto s_lastTime = std::chrono::high_resolution_clock::now();
    static int  s_frameCount = 0;

    PopulateCommandList();
    ID3D12CommandList* ppCommandLists[] = { m_commandList.Get() };
    m_commandQueue->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);
    ThrowIfFailed(m_swapChain->Present(0, 0));
    WaitForPreviousFrame();

    // FPS calculation
    s_frameCount++;
    auto currentTime = std::chrono::high_resolution_clock::now();
    float elapsedSec =
        std::chrono::duration<float>(currentTime - s_lastTime).count();

    // Update once per second
    if (elapsedSec >= 1.0f)
    {
        float fps = static_cast<float>(s_frameCount) / elapsedSec;
        float dT = 1000.0f/fps;

        std::wstringstream ss;
        ss << std::fixed << std::setprecision(2)
               << L"Frame Time: " << dT << L" ms (" << fps << L" fps)";

        // Update the window title
        SetWindowTextW(Win32Application::GetHwnd(), ss.str().c_str());

        // Reset for next time
        s_frameCount = 0;
        s_lastTime = currentTime;
    }
    #if ENABLE_D3D12_DIAGNOSTICS
        dxdiag::DumpNewMessages();
    #endif
}

void Renderer::OnDestroy() {
  // Ensure that the GPU is no longer referencing resources that are about to be
  // cleaned up by the destructor.
  WaitForPreviousFrame();

  CloseHandle(m_fenceEvent);
    if(SL_FAILED(res, slShutdown()))
    {
        // TODO: handle error
    }
}

void Renderer::PopulateCommandList()
{
    // ------------------------------------------------------------------------
    // 1. STANDARD SETUP
    // ------------------------------------------------------------------------

    // Reset allocator & list
    auto* allocator = m_commandAllocators[m_frameIndex].Get();
    ThrowIfFailed(allocator->Reset());
    ThrowIfFailed(m_commandList->Reset(allocator, m_pipelineState.Get()));

    // Graphics setup: signature, viewports, RTV/DSV
    m_commandList->SetGraphicsRootSignature(m_rootSignature.Get());
    m_commandList->RSSetViewports(1, &m_viewport);
    m_commandList->RSSetScissorRects(1, &m_scissorRect);

    // Transition backbuffer PRESENT->RENDER_TARGET
    {
        auto b = CD3DX12_RESOURCE_BARRIER::Transition(
                m_renderTargets[m_frameIndex].Get(),
                D3D12_RESOURCE_STATE_PRESENT,
                D3D12_RESOURCE_STATE_RENDER_TARGET);
        m_commandList->ResourceBarrier(1, &b);
    }

    CD3DX12_CPU_DESCRIPTOR_HANDLE rtv(
            m_rtvHeap->GetCPUDescriptorHandleForHeapStart(),
            m_frameIndex, m_rtvDescriptorSize);
    CD3DX12_CPU_DESCRIPTOR_HANDLE dsv(m_dsvHeap->GetCPUDescriptorHandleForHeapStart());
    m_commandList->OMSetRenderTargets(1, &rtv, FALSE, &dsv);

    // Refit TLAS
    CreateTopLevelAS(m_instances, true);

    // Bind SRV/UAV heap once
    {
        ID3D12DescriptorHeap* heaps[] = { m_srvUavHeap.Get() };
        m_commandList->SetDescriptorHeaps(_countof(heaps), heaps);
    }

    // Prepare raytracing descriptors (Legacy DXR / Reference)
    D3D12_DISPATCH_RAYS_DESC raysDesc{};
    raysDesc.Width  = GetWidth();
    raysDesc.Height = GetHeight();
    raysDesc.Depth  = 1;

    const uint64_t sbtStart = m_sbtStorage->GetGPUVirtualAddress();
    const uint32_t rgSize   = m_sbtHelper.GetRayGenEntrySize();
    const uint32_t numRG    = (uint32_t)m_passIndex.size();

    raysDesc.MissShaderTable.StartAddress = sbtStart + m_sbtHelper.GetRayGenSectionSize();
    raysDesc.MissShaderTable.SizeInBytes   = m_sbtHelper.GetMissSectionSize();
    raysDesc.MissShaderTable.StrideInBytes = m_sbtHelper.GetMissEntrySize();

    raysDesc.HitGroupTable.StartAddress    =
            raysDesc.MissShaderTable.StartAddress + raysDesc.MissShaderTable.SizeInBytes;
    raysDesc.HitGroupTable.SizeInBytes     = m_sbtHelper.GetHitGroupSectionSize();
    raysDesc.HitGroupTable.StrideInBytes   = m_sbtHelper.GetHitGroupEntrySize();

    if (m_sbtHelper.GetCallableSectionSize() > 0)
    {
        raysDesc.CallableShaderTable.StartAddress  = raysDesc.HitGroupTable.StartAddress + raysDesc.HitGroupTable.SizeInBytes;
        raysDesc.CallableShaderTable.SizeInBytes   = m_sbtHelper.GetCallableSectionSize();
        raysDesc.CallableShaderTable.StrideInBytes = m_sbtHelper.GetCallableEntrySize();
    }
    else
    {
        raysDesc.CallableShaderTable.StartAddress  = 0;
        raysDesc.CallableShaderTable.SizeInBytes   = 0;
        raysDesc.CallableShaderTable.StrideInBytes = 0;
    }

    // Ray-trace output -> UAV
    {
        auto u = CD3DX12_RESOURCE_BARRIER::Transition(
                m_outputResource.Get(),
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        m_commandList->ResourceBarrier(1, &u);
    }

    {
        auto barrierToDest = CD3DX12_RESOURCE_BARRIER::Transition(
                m_globalCounterBuffer.Get(),
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_DEST);
        m_commandList->ResourceBarrier(1, &barrierToDest);

        m_commandList->CopyBufferRegion(
                m_globalCounterBuffer.Get(), 0,
                m_zeroBuffer.Get(), 0,
                MAX_STACKS * sizeof(uint32_t));

        auto barrierToUAV = CD3DX12_RESOURCE_BARRIER::Transition(
                m_globalCounterBuffer.Get(),
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        m_commandList->ResourceBarrier(1, &barrierToUAV);
    }

    // ------------------------------------------------------------------------
    // 2. STREAMING COMPACTION STATE
    // ------------------------------------------------------------------------

    // Stack 0 starts as input (init shader writes here), Stack 1 as output.
    uint32_t currentStackIdx = 0;
    uint32_t nextStackIdx    = 1;

    // Track loop recursion
    std::vector<std::pair<int, uint32_t>> loopStack;

    // ------------------------------------------------------------------------
    // 3. PASS EXECUTION LOOP
    // ------------------------------------------------------------------------
    for (size_t i = 0; i < m_passes.size(); ++i)
    {
        CD3DX12_RESOURCE_BARRIER barriers[] = {
                CD3DX12_RESOURCE_BARRIER::UAV(m_sortBoundsBuffer.Get()),
                CD3DX12_RESOURCE_BARRIER::UAV(m_sortCountBuffer.Get())
        };
        m_commandList->ResourceBarrier(_countof(barriers), barriers);

        auto& p = m_passes[i];

        switch (p.stage)
        {
            // ── Loop Start ──
            case Stage::LoopStart:
            {
                loopStack.push_back({ -1, p.loopCount });
                break;
            }

            case Stage::PingSwap:
            {
                // Swap the input and output stack indices.
                // The next shader dispatched will use the swapped values
                // via its RootConstants.
                std::swap(currentStackIdx, nextStackIdx);
                break;
            }

                // ── Loop End ──
            case Stage::LoopEnd:
            {
                if (!loopStack.empty())
                {
                    loopStack.back().second--;
                    if (loopStack.back().second > 0)
                    {
                        // Jump back to start
                        i = p.targetIdx;
                    }
                    else
                    {
                        // Loop done, remove from stack
                        loopStack.pop_back();
                    }
                }
                break;
            }

                // ── Barrier ──
            case Stage::Barrier:
            {
                auto u = CD3DX12_RESOURCE_BARRIER::UAV(nullptr);
                m_commandList->ResourceBarrier(1, &u);
                break;
            }

            case Stage::ClearSort:
            {
                ClearSortBuffers(m_commandList.Get());

                // Add barrier to ensure clear finishes before KeyGen
                CD3DX12_RESOURCE_BARRIER barriers[] = {
                        CD3DX12_RESOURCE_BARRIER::UAV(m_sortBoundsBuffer.Get()),
                        CD3DX12_RESOURCE_BARRIER::UAV(m_sortCountBuffer.Get())
                };
                m_commandList->ResourceBarrier(_countof(barriers), barriers);
                break;
            }

            case Stage::ML:
            {
                if (m_enableML) {
                    RunMLPass();
                }
                break;
            }

                // ── Standard DXR (Legacy) ──
            case Stage::RayGen:
            {
                m_commandList->SetPipelineState1(m_rtStateObject.Get());
                m_commandList->SetComputeRootSignature(m_rayGenSignature.Get());
                m_commandList->SetComputeRootDescriptorTable(
                        0, m_srvUavHeap->GetGPUDescriptorHandleForHeapStart());

                UINT constants[6] = { GetWidth(), GetHeight(), 0, 0, 0, 0 };
                m_commandList->SetComputeRoot32BitConstants(1, 6, constants, 0);

                uint32_t currentRgSlot = m_passIndex[p.file];
                raysDesc.RayGenerationShaderRecord.StartAddress = sbtStart + currentRgSlot * rgSize;
                raysDesc.RayGenerationShaderRecord.SizeInBytes  = rgSize;

                m_commandList->DispatchRays(&raysDesc);
                break;
            }

                // ── Standard Compute (Dense) ──
                // Used for Initialization (Pass_init) or Post-Processing
            case Stage::Compute:
            {
                if (p.isWorkGraph)
                {
                    // [Work Graph logic]
                    const uint32_t wgIndex = p.wgIdx;
                    const auto& rt = m_wgRuntime[wgIndex];

                    m_commandList->SetComputeRootSignature(m_computeSignature.Get());
                    m_commandList->SetComputeRootDescriptorTable(
                            0, m_srvUavHeap->GetGPUDescriptorHandleForHeapStart());

                    UINT constants[6] = { GetWidth(), GetHeight(), currentStackIdx, nextStackIdx, 0, 0 };
                    m_commandList->SetComputeRoot32BitConstants(1, 6, constants, 0);

                    D3D12_SET_PROGRAM_DESC setProg{};
                    setProg.Type                        = D3D12_PROGRAM_TYPE_WORK_GRAPH;
                    setProg.WorkGraph.ProgramIdentifier = rt.id;
                    setProg.WorkGraph.BackingMemory     = rt.backing;

                    static std::vector<bool> s_inited;
                    if (s_inited.size() <= wgIndex) s_inited.resize(wgIndex + 1, false);
                    setProg.WorkGraph.Flags = s_inited[wgIndex]
                                              ? D3D12_SET_WORK_GRAPH_FLAG_NONE
                                              : D3D12_SET_WORK_GRAPH_FLAG_INITIALIZE;
                    m_commandList->SetProgram(&setProg);
                    s_inited[wgIndex] = true;

                    D3D12_DISPATCH_GRAPH_DESC dg{};
                    dg.Mode = D3D12_DISPATCH_MODE_NODE_CPU_INPUT;
                    dg.NodeCPUInput.EntrypointIndex     = 0;
                    dg.NodeCPUInput.NumRecords          = 1;
                    m_commandList->DispatchGraph(&dg);
                }
                else
                {
                    // Standard Compute: Runs on EVERY pixel (Dense)
                    // Used for seeding the stack in Pass_init
                    m_commandList->SetPipelineState(m_csPSOs[p.psoIdx].Get());
                    m_commandList->SetComputeRootSignature(m_computeSignature.Get());
                    m_commandList->SetComputeRootDescriptorTable(
                            0, m_srvUavHeap->GetGPUDescriptorHandleForHeapStart());

                    // Pass constants. Even if this is dense, we tell it where the stacks are
                    // so Pass_init knows where to write the initial indices.
                    UINT constants[6] = { GetWidth(), GetHeight(), currentStackIdx, nextStackIdx, 0, 0 };
                    m_commandList->SetComputeRoot32BitConstants(1, 6, constants, 0);

                    uint32_t gx = (GetWidth()  + p.groupX - 1) / p.groupX;
                    uint32_t gy = (GetHeight() + p.groupY - 1) / p.groupY;
                    m_commandList->Dispatch(gx, gy, 1);
                }
                break;
            }

            case Stage::FixedCompute:
            {
                m_commandList->SetPipelineState(m_csPSOs[p.psoIdx].Get());
                m_commandList->SetComputeRootSignature(m_computeSignature.Get());
                m_commandList->SetComputeRootDescriptorTable(0, m_srvUavHeap->GetGPUDescriptorHandleForHeapStart());

                // Bind constants (Width, Height, Stacks...)
                UINT constants[6] = { GetWidth(), GetHeight(), currentStackIdx, nextStackIdx, 0, 0 };
                m_commandList->SetComputeRoot32BitConstants(1, 6, constants, 0);

                // Dispatch exact number of groups specified in string (e.g. fx:1 -> 1,1,1)
                m_commandList->Dispatch(p.groupX, p.groupY, 1);
                break;
            }

                // ── Wavefront Compute (Sparse) ──
                // Used for Tracing/Shading. Uses Indirect Dispatch based on Counters.
            case Stage::Wavefront:
            {
                // === Step A: Run Setup Shader ===

                // 1. Determine which PSO to use
                // If we are reading and writing to the same stack (e.g. Sort Key Generation),
                // we MUST NOT clear the counter, otherwise the next pass sees 0 rays.
                // If we are moving data (Trace: 0->1, Reorder: 1->0), we MUST clear the destination counter.
                bool isInPlace = (currentStackIdx == nextStackIdx);

                if (isInPlace) {
                    m_commandList->SetPipelineState(m_psoSetupIndirectNoClear.Get());
                } else {
                    m_commandList->SetPipelineState(m_psoSetupIndirect.Get());
                }

                m_commandList->SetComputeRootSignature(m_rsSetupIndirect.Get());

                // ... [Bind Descriptors as before] ...
                // Calculate handle for Counters/Args (Slot 33 in heap)
                auto gpuHandle = m_srvUavHeap->GetGPUDescriptorHandleForHeapStart();
                UINT inc = m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
                gpuHandle.ptr += 33 * inc;
                m_commandList->SetComputeRootDescriptorTable(0, gpuHandle);


                // Setup Constants: [ReadCounterIdx, WriteArgIdx, ClearCounterIdx, GroupSize]
                // Note: If using NoClear PSO, the 'nextStackIdx' passed here is ignored by the shader logic,
                // but we pass it anyway for consistency.
                UINT setupConsts[4] = { currentStackIdx, 0, nextStackIdx, p.groupX };
                m_commandList->SetComputeRoot32BitConstants(1, 4, setupConsts, 0);

                m_commandList->Dispatch(1, 1, 1);

                // Barriers: Wait for Args and Counter Reset
                CD3DX12_RESOURCE_BARRIER barriers[] = {
                        CD3DX12_RESOURCE_BARRIER::UAV(m_indirectArgsBuffer.Get()),
                        // Transition Args to INDIRECT_ARGUMENT state
                        CD3DX12_RESOURCE_BARRIER::Transition(m_indirectArgsBuffer.Get(),
                                                             D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT)
                };
                m_commandList->ResourceBarrier(_countof(barriers), barriers);

                // === Step B: Run Wavefront Shader ===

                m_commandList->SetPipelineState(m_csPSOs[p.psoIdx].Get());
                m_commandList->SetComputeRootSignature(m_computeSignature.Get());
                m_commandList->SetComputeRootDescriptorTable(
                        0, m_srvUavHeap->GetGPUDescriptorHandleForHeapStart());

                // Constants: [W, H, InputStack, OutputStack]
                UINT traceConsts[6] = { GetWidth(), GetHeight(), currentStackIdx, nextStackIdx, 0, 0 };
                m_commandList->SetComputeRoot32BitConstants(1, 6, traceConsts, 0);

                // Execute Indirect
                m_commandList->ExecuteIndirect(
                        m_commandSignature.Get(),
                        1,
                        m_indirectArgsBuffer.Get(),
                        0,
                        nullptr, 0);

                // Transition Args back to UAV for next pass + UAV barrier for stacks
                CD3DX12_RESOURCE_BARRIER postBarriers[] = {
                        CD3DX12_RESOURCE_BARRIER::Transition(m_indirectArgsBuffer.Get(),
                                                             D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
                        CD3DX12_RESOURCE_BARRIER::UAV(m_stackBuffers[nextStackIdx].Get()),
                        CD3DX12_RESOURCE_BARRIER::UAV(m_globalCounterBuffer.Get())
                };
                m_commandList->ResourceBarrier(_countof(postBarriers), postBarriers);

                break;
            }

        } // switch
    } // for


    // Ensure all UAV writes to DLSS inputs are visible before Streamline reads them.
    {
        ID3D12Resource* uavs[] =
        {
            m_dlssDepth.Get(),
            m_dlssMVec.Get(),
            m_dlssNormals.Get(),
            m_dlssDiffuseAlbedo.Get(),
            m_dlssSpecularAlbedo.Get(),
            m_dlssRoughness.Get(),
            m_dlssSpecMVec.Get(),
            m_dlssSpecHitDist.Get(),
            m_dlssTransparency.Get(),
            m_dlssColorBeforeTrans.Get(),
            // (include others if you write them)
        };

        for (ID3D12Resource* r : uavs)
        {
            if (!r) continue;
            auto b = CD3DX12_RESOURCE_BARRIER::UAV(r);
            m_commandList->ResourceBarrier(1, &b);
        }
    }

    RunDLSS_RR(m_commandList.Get());

    // --- COPY DLSS RESULT TO OUTPUT BUFFER (SLICE 1) ---
    {
        D3D12_RESOURCE_BARRIER preCopyBarriers[] = {
            // DLSS Output: UAV -> COPY_SOURCE   (FIX)
            CD3DX12_RESOURCE_BARRIER::Transition(
                m_dlssOutput.Get(),
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_SOURCE),

            // Main Output Array: UAV -> COPY_DEST
            CD3DX12_RESOURCE_BARRIER::Transition(
                m_outputResource.Get(),
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_DEST)
        };
        m_commandList->ResourceBarrier(_countof(preCopyBarriers), preCopyBarriers);

        CD3DX12_TEXTURE_COPY_LOCATION src(m_dlssOutput.Get(), 0);
        UINT destSubresource = 2;
        CD3DX12_TEXTURE_COPY_LOCATION dst(m_outputResource.Get(), destSubresource);
        m_commandList->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);

        D3D12_RESOURCE_BARRIER postCopyBarriers[] = {
            // DLSS Output: COPY_SOURCE -> UAV
            CD3DX12_RESOURCE_BARRIER::Transition(
                m_dlssOutput.Get(),
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS),

            // Main Output Array: COPY_DEST -> UAV
            CD3DX12_RESOURCE_BARRIER::Transition(
                m_outputResource.Get(),
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
        };
        m_commandList->ResourceBarrier(_countof(postCopyBarriers), postCopyBarriers);
    }

    // ------------------------------------------------------------------------
    // 4. FINAL OUTPUT COPY
    // ------------------------------------------------------------------------
    {
        auto toSrc = CD3DX12_RESOURCE_BARRIER::Transition(
                m_outputResource.Get(),
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_SOURCE);
        m_commandList->ResourceBarrier(1, &toSrc);

        auto toDst = CD3DX12_RESOURCE_BARRIER::Transition(
                m_renderTargets[m_frameIndex].Get(),
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                D3D12_RESOURCE_STATE_COPY_DEST);
        m_commandList->ResourceBarrier(1, &toDst);

        UINT layer  = m_displayLevels[m_currentDisplayLevel];
        UINT sub    = D3D12CalcSubresource(0, layer, 0, 1, 60);
        CD3DX12_TEXTURE_COPY_LOCATION src(m_outputResource.Get(), sub);
        CD3DX12_TEXTURE_COPY_LOCATION dst(m_renderTargets[m_frameIndex].Get(), 0);
        D3D12_BOX box = {0,0,0, GetWidth(), GetHeight(), 1};
        m_commandList->CopyTextureRegion(&dst,0,0,0, &src, &box);

        auto backToRT = CD3DX12_RESOURCE_BARRIER::Transition(
                m_renderTargets[m_frameIndex].Get(),
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_RENDER_TARGET);
        m_commandList->ResourceBarrier(1, &backToRT);

        // Final PRESENT barrier
        auto pres = CD3DX12_RESOURCE_BARRIER::Transition(
                m_renderTargets[m_frameIndex].Get(),
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                D3D12_RESOURCE_STATE_PRESENT);
        m_commandList->ResourceBarrier(1, &pres);
    }
    ThrowIfFailed(m_commandList->Close());
}



void Renderer::WaitForPreviousFrame() {
  // WAITING FOR THE FRAME TO COMPLETE BEFORE CONTINUING IS NOT BEST PRACTICE.
  // This is code implemented as such for simplicity. The
  // D3D12HelloFrameBuffering sample illustrates how to use fences for efficient
  // resource usage and to maximize GPU utilization.

  // Signal and increment the fence value.
  const UINT64 fence = m_fenceValue;
  ThrowIfFailed(m_commandQueue->Signal(m_fence.Get(), fence));
  m_fenceValue++;

  // Wait until the previous frame is finished.
  if (m_fence->GetCompletedValue() < fence) {
    ThrowIfFailed(m_fence->SetEventOnCompletion(fence, m_fenceEvent));
    WaitForSingleObject(m_fenceEvent, INFINITE);
  }

  m_frameIndex = m_swapChain->GetCurrentBackBufferIndex();
#if ENABLE_D3D12_DIAGNOSTICS
    dxdiag::CheckDeviceRemoved(m_device.Get());   // dumps reason + breadcrumbs
#endif
}

void Renderer::CheckRaytracingSupport() {
  D3D12_FEATURE_DATA_D3D12_OPTIONS5 options5 = {};
  ThrowIfFailed(m_device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5,
                                              &options5, sizeof(options5)));
  if (options5.RaytracingTier < D3D12_RAYTRACING_TIER_1_0)
    throw std::runtime_error("Raytracing not supported on device");

}

// Input stuff
void Renderer::OnKeyDown(UINT8 key)
{
    g_keys[key] = true;
}
void Renderer::OnKeyUp(UINT8 key) {
    g_keys[key] = false;
    // Check if a specific key (e.g., 'C' for cycle) is pressed
    if (key == 'C') {
        m_currentDisplayLevel = (m_currentDisplayLevel + 1) % m_displayLevels.size();
        std::wcout << L"C key pressed, switching to level: " << m_currentDisplayLevel << std::endl;
    }

    if (key == VK_SPACE) {
        m_raster = !m_raster;
        std::wcout << L"Space key pressed, toggling rasterization: " << m_raster << std::endl;
    }
    if (key == 'K') {
        m_recorder.CaptureKeyframe(nv_helpers_dx12::CameraManip);
    }
}



//-----------------------------------------------------------------------------
//
// Create a bottom-level acceleration structure based on a list of vertex
// buffers in GPU memory along with their vertex count. The build is then done
// in 3 steps: gathering the geometry, computing the sizes of the required
// buffers, and building the actual AS
//
// #DXR Extra: Indexed Geometry
constexpr bool kUseBlasCompaction = true;

Renderer::AccelerationStructureBuffers
Renderer::CreateBottomLevelAS(
    std::vector<std::pair<ComPtr<ID3D12Resource>, uint32_t>> vVertexBuffers,
    std::vector<std::pair<ComPtr<ID3D12Resource>, uint32_t>> vIndexBuffers)
{
    nv_helpers_dx12::BottomLevelASGenerator bottomLevelAS;

    for (size_t i = 0; i < vVertexBuffers.size(); i++) {
        bottomLevelAS.AddVertexBuffer(
            vVertexBuffers[i].first.Get(), 0, vVertexBuffers[i].second, sizeof(Vertex),
            vIndexBuffers[i].first.Get(), 0, vIndexBuffers[i].second, nullptr, 0, true);
    }

    UINT64 scratchSizeInBytes = 0;
    UINT64 resultSizeInBytes = 0;

    // 2. BUILD FLAGS: Only add ALLOW_COMPACTION if the flag is true
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS buildFlags =
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;

    if constexpr (kUseBlasCompaction) {
        buildFlags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_COMPACTION;
    }

    bottomLevelAS.ComputeASBufferSizes(m_device.Get(), buildFlags, &scratchSizeInBytes, &resultSizeInBytes);

    AccelerationStructureBuffers buffers;

    // Scratch buffer is required regardless of compaction
    buffers.pScratch = nv_helpers_dx12::CreateBuffer(
        m_device.Get(), scratchSizeInBytes,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON,
        nv_helpers_dx12::kDefaultHeapProps);

    // -----------------------------------------------------------------------
    // PATH A: NO COMPACTION (Fast Build, Higher VRAM Usage)
    // -----------------------------------------------------------------------
    if constexpr (!kUseBlasCompaction)
    {
        // Allocate the result buffer directly using the estimated size
        buffers.pResult = nv_helpers_dx12::CreateBuffer(
            m_device.Get(), resultSizeInBytes,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
            nv_helpers_dx12::kDefaultHeapProps);
        buffers.pResult->SetName(L"BLAS (Uncompacted)");

        // Build directly into the final result buffer
        bottomLevelAS.Generate(m_commandList.Get(), buffers.pScratch.Get(),
                               buffers.pResult.Get(), false, nullptr);

        // No flush, no readback, no copy barriers needed here.
    }
    // -----------------------------------------------------------------------
    // PATH B: COMPACTION ENABLED (Slower Build, Lower VRAM Usage)
    // -----------------------------------------------------------------------
    else
    {
        // Temporary large buffer
        buffers.pResultUncompacted = nv_helpers_dx12::CreateBuffer(
            m_device.Get(), resultSizeInBytes,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
            nv_helpers_dx12::kDefaultHeapProps);

        // Query for the compacted size
        ComPtr<ID3D12Resource> compactedSizeBuffer = nv_helpers_dx12::CreateBuffer(
            m_device.Get(), sizeof(UINT64),
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            nv_helpers_dx12::kDefaultHeapProps);
        compactedSizeBuffer->SetName(L"BLAS Compacted Size Buffer");

        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC postBuildInfoDesc = {};
        postBuildInfoDesc.DestBuffer = compactedSizeBuffer->GetGPUVirtualAddress();
        postBuildInfoDesc.InfoType = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_COMPACTED_SIZE;

        // Build the initial, uncompacted BLAS and enqueue the size query
        bottomLevelAS.Generate(m_commandList.Get(), buffers.pScratch.Get(),
                            buffers.pResultUncompacted.Get(), false, nullptr);

        // Enqueue the query command
        D3D12_GPU_VIRTUAL_ADDRESS sourceASAddress = buffers.pResultUncompacted->GetGPUVirtualAddress();
        m_commandList->EmitRaytracingAccelerationStructurePostbuildInfo(
            &postBuildInfoDesc, 1, &sourceASAddress);

        // Execute, read back size, and create final buffer
        ComPtr<ID3D12Resource> readbackBuffer = nv_helpers_dx12::CreateBuffer(
            m_device.Get(), sizeof(UINT64),
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_COPY_DEST,
            nv_helpers_dx12::kReadbackHeapProps);

        CD3DX12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::Transition(
            compactedSizeBuffer.Get(),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_SOURCE);

        m_commandList->ResourceBarrier(1, &barrier);
        m_commandList->CopyResource(readbackBuffer.Get(), compactedSizeBuffer.Get());

        // --- STALL POINT ---
        // We must flush to CPU to read the size. This is expensive.
        ThrowIfFailed(m_commandList->Close());
        ID3D12CommandList* ppCommandLists[] = { m_commandList.Get() };
        m_commandQueue->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);
        WaitForPreviousFrame();
        ThrowIfFailed(m_commandList->Reset(m_commandAllocators[m_frameIndex].Get(), nullptr));
        // -------------------

        // Read the size
        UINT64 compactedResultSizeInBytes;
        void* pMappedData;
        ThrowIfFailed(readbackBuffer->Map(0, nullptr, &pMappedData));
        memcpy(&compactedResultSizeInBytes, pMappedData, sizeof(UINT64));
        readbackBuffer->Unmap(0, nullptr);

        std::wcout << L"BLAS Original Size: " << resultSizeInBytes << L" | Compacted Size: " << compactedResultSizeInBytes << std::endl;

        // Allocate final buffer
        buffers.pResult = nv_helpers_dx12::CreateBuffer(
            m_device.Get(), compactedResultSizeInBytes,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
            nv_helpers_dx12::kDefaultHeapProps);
        buffers.pResult->SetName(L"Compacted BLAS");

        // Perform compaction copy
        m_commandList->CopyRaytracingAccelerationStructure(
            buffers.pResult->GetGPUVirtualAddress(),
            buffers.pResultUncompacted->GetGPUVirtualAddress(),
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_COMPACT);

        // Barrier for safety
        barrier = CD3DX12_RESOURCE_BARRIER::UAV(buffers.pResult.Get());
        m_commandList->ResourceBarrier(1, &barrier);
    }

    return buffers;
}

//-----------------------------------------------------------------------------
// Create the main acceleration structure that holds all instances of the scene.
// Similarly to the bottom-level AS generation, it is done in 3 steps: gathering
// the instances, computing the memory requirements for the AS, and building the
// AS itself
//
void Renderer::CreateTopLevelAS(
    const std::vector<std::pair<ComPtr<ID3D12Resource>, DirectX::XMMATRIX>>
        &instances, // pair of bottom level AS and matrix of the instance
    // #DXR Extra - Refitting
    bool updateOnly // If true the top-level AS will only be refitted and not
                    // rebuilt from scratch
) {

  // #DXR Extra - Refitting
  if (!updateOnly) {
    // Gather all the instances into the builder helper
    for (size_t i = 0; i < instances.size(); i++) {
      m_topLevelASGenerator.AddInstance(
          instances[i].first.Get(), instances[i].second, static_cast<UINT>(i),
          static_cast<UINT>(2 * i));
    }

    // As for the bottom-level AS, the building the AS requires some scratch
    // space to store temporary data in addition to the actual AS. In the case
    // of the top-level AS, the instance descriptors also need to be stored in
    // GPU memory. This call outputs the memory requirements for each (scratch,
    // results, instance descriptors) so that the application can allocate the
    // corresponding memory
    UINT64 scratchSize, resultSize, instanceDescsSize;

    m_topLevelASGenerator.ComputeASBufferSizes(
        m_device.Get(), true, &scratchSize, &resultSize, &instanceDescsSize);

    // Create the scratch and result buffers. Since the build is all done on
    // GPU, those can be allocated on the default heap
    m_topLevelASBuffers.pScratch = nv_helpers_dx12::CreateBuffer(
        m_device.Get(), scratchSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        nv_helpers_dx12::kDefaultHeapProps);
    m_topLevelASBuffers.pResult = nv_helpers_dx12::CreateBuffer(
        m_device.Get(), resultSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
        nv_helpers_dx12::kDefaultHeapProps);

    // The buffer describing the instances: ID, shader binding information,
    // matrices ... Those will be copied into the buffer by the helper through
    // mapping, so the buffer has to be allocated on the upload heap.
    m_topLevelASBuffers.pInstanceDesc = nv_helpers_dx12::CreateBuffer(
        m_device.Get(), instanceDescsSize, D3D12_RESOURCE_FLAG_NONE,
        D3D12_RESOURCE_STATE_GENERIC_READ, nv_helpers_dx12::kUploadHeapProps);
  }
  // After all the buffers are allocated, or if only an update is required, we
  // can build the acceleration structure. Note that in the case of the update
  // we also pass the existing AS as the 'previous' AS, so that it can be
  // refitted in place.
  m_topLevelASGenerator.Generate(m_commandList.Get(),
                                 m_topLevelASBuffers.pScratch.Get(),
                                 m_topLevelASBuffers.pResult.Get(),
                                 m_topLevelASBuffers.pInstanceDesc.Get(),
                                 updateOnly, m_topLevelASBuffers.pResult.Get());
}

//-----------------------------------------------------------------------------
//
// Combine the BLAS and TLAS builds to construct the entire acceleration
// structure required to raytrace the scene
//
void Renderer::CreateAccelerationStructures() {
  // Build the bottom AS from the Triangle vertex buffer
  /*AccelerationStructureBuffers bottomLevelBuffers = CreateBottomLevelAS(
      {{m_vertexBuffer.Get(), 4}}, {{m_indexBuffer.Get(), 12}});*/

  // #DXR Extra: Indexed Geometry
  // Build the bottom AS from the Menger Sponge vertex buffer
  // #DXR Extra: Indexed Geometry
  // Build the bottom AS from the Menger Sponge vertex buffer

    std::vector<AccelerationStructureBuffers> allBlasBuffers;
    m_instances.clear();
    m_instanceModelIndices.clear();

    for (size_t i = 0; i < m_VB.size(); ++i) {
        // CreateBottomLevelAS now performs compaction and returns the final compacted buffer in pResult.
        // It also handles command list execution and reset internally.
        AccelerationStructureBuffers buffers = CreateBottomLevelAS(
            {{m_VB[i].Get(), m_VertexCount[i]}},
            {{m_IB[i].Get(), m_IndexCount[i]}}
        );

        // We only need the final pResult for the TLAS.
        m_instances.emplace_back(buffers.pResult, XMMatrixIdentity());
        m_instanceModelIndices.push_back(static_cast<UINT>(i));

        // Store the buffer objects to manage their lifetime.
        allBlasBuffers.push_back(buffers);
    }
    OnInitTransform();
  CreateTopLevelAS(m_instances);
    // Collect emissive triangles
    CollectEmissiveTriangles();

    // Build per‑instance object→world matrices for the light tree
    std::vector<InstanceXformCPU> ltXforms;
    ltXforms.reserve(m_instances.size());
    for (size_t i = 0; i < m_instances.size(); ++i)
        ltXforms.push_back(ToInstanceXform(m_instances[i].second));

    // Use the overload that takes transforms
    m_lightTree.Build(m_emissiveTriangles, ltXforms);
    m_lightTree.PrintMetrics();
    {SCOPE_TIMER("LightTree.UploadAll");m_lightTree.UploadAll(m_device.Get(), m_commandList.Get());}

    // Create buffer for emissive triangles
    {SCOPE_TIMER("CreateEmissiveTrianglesBuffer");CreateEmissiveTrianglesBuffer();}
    {SCOPE_TIMER("CreateTriToLightIdBuffer");CreateTriToLightIdBuffer();}

    // Build & upload alias table
    {SCOPE_TIMER("BuildAliasTableSoA");BuildAliasTableSoA(m_emissiveTriangles);}
    {SCOPE_TIMER("CreateAliasBuffers");CreateAliasBuffers();}

      // Flush the command list and wait for it to finish
    {
        SCOPE_TIMER("Execute+Fence (post-LightTree/Uploads)");
        m_commandList->Close();
        ID3D12CommandList *ppCommandLists[] = {m_commandList.Get()};
        m_commandQueue->ExecuteCommandLists(1, ppCommandLists);
        m_fenceValue++;
        m_commandQueue->Signal(m_fence.Get(), m_fenceValue);

        m_fence->SetEventOnCompletion(m_fenceValue, m_fenceEvent);
        WaitForSingleObject(m_fenceEvent, INFINITE);
    }

    m_lightTree.ReleaseStaging();


  // Once the command list is finished executing, reset it to be reused for
  // rendering
    ThrowIfFailed(
            m_commandList->Reset(m_commandAllocators[m_frameIndex].Get(),
                                 m_pipelineState.Get()));

    // Store the AS buffers. The rest of the buffers will be released once we exit
  // the function
  //m_bottomLevelAS = bottomLevelBuffers.pResult;
}

// Create the resource signatures for raygen (legacy) and compute hybrid pipeline
ComPtr<ID3D12RootSignature> Renderer::CreateRayGenSignature() {
    CD3DX12_ROOT_PARAMETER1 rootParameters[2];
    std::vector<CD3DX12_DESCRIPTOR_RANGE1> ranges;
    ranges.reserve(40); // Reserve space

    // Flags
    const auto VOLATILE = D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE;
    const auto STATIC   = D3D12_DESCRIPTOR_RANGE_FLAG_DATA_STATIC;


    // Slot 0:  UAV u0 (Output)
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0, 0, VOLATILE, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);

    // Slot 1:  UAV u1 (Permanent Data / Accumulation)
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 1, 0, VOLATILE, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);

    // Slot 2:  SRV t0 (TLAS)
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, STATIC, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);

    // Slot 3:  SRV t1 (Global Index Buffer)
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 1, 0, STATIC, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);

    // Slot 4:  SRV t2 (Global Vertex Buffer)
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 2, 0, STATIC, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);

    // Slot 5:  CBV b0 (Camera)
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_CBV, 1, 0, 0, STATIC, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);

    // Slot 6:  SRV t3 (Instance Properties)
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 3, 0, STATIC, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);

    // Slot 7:  SRV t4 (Material IDs)
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 4, 0, STATIC, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);

    // Slot 8:  SRV t5 (Materials)
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 5, 0, STATIC, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);

    // Slot 9:  SRV t6 (Emissive Triangles)
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 6, 0, STATIC, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);

    // Slot 10-15: UAVs u2-u7 (Reservoirs & Sample Buffers)
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 2, 0, VOLATILE, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 3, 0, VOLATILE, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 4, 0, VOLATILE, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 5, 0, VOLATILE, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 6, 0, VOLATILE, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 7, 0, VOLATILE, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);

    // Slot 16: SRV t7 (Alias Prob)
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 7, 0, STATIC, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);

    // Slot 17: SRV t8 (Alias Idx)
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 8, 0, STATIC, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);

    // Slot 18: UAV u8 (Scratch Ping)
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 8, 0, VOLATILE, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);

    // Slot 19: UAV u9 (Initial BSDF Rays)
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 9, 0, VOLATILE, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);

    // Slot 20-23: LightTree SRVs (t9-t12)
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 9, 0, STATIC, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 10, 0, STATIC, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 11, 0, STATIC, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 12, 0, STATIC, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);

    // Slot 24: SRV t15 (TriToLightId)
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 15, 0, STATIC, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);

    // Slot 25-27: LightTree Lookups (t16-t18)
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 16, 0, STATIC, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 17, 0, STATIC, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 18, 0, STATIC, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);

    // Slot 28-31: Texture Arrays (t30-t33)
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 30, 0, STATIC, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 31, 0, STATIC, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 32, 0, STATIC, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 33, 0, STATIC, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);

    // Slot 32: UAV u10 (PathState Buffer)
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 10, 0, VOLATILE, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);

    // Slot 33: UAV u34 (Global Counters)
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 34, 0, VOLATILE, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);

    // Slot 34: UAV u35 (Indirect Args)
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 35, 0, VOLATILE, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);

    // Slot 35: UAVs u36-u39 (Pixel Stacks)
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 4, 36, 0, VOLATILE, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);

    ranges.emplace_back().Init(
            D3D12_DESCRIPTOR_RANGE_TYPE_UAV,
            11, // NumDescriptors
            11, // BaseShaderRegister (u11)
            0,  // RegisterSpace
            VOLATILE,
            D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND
    );

    // Slot 36: Sort Buffers (u60-u62)
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 3, 60, 0, VOLATILE,  D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);

    // -----------------------------------------------------------------------

    // Parameter 0: The Descriptor Table (matches global heap exactly)
    rootParameters[0].InitAsDescriptorTable(static_cast<UINT>(ranges.size()), ranges.data(), D3D12_SHADER_VISIBILITY_ALL);

    // Parameter 1: Constants (b1) - Same 6 constants as compute shader
    rootParameters[1].InitAsConstants(6, 1, 0, D3D12_SHADER_VISIBILITY_ALL);

    // Static Samplers
    CD3DX12_STATIC_SAMPLER_DESC staticSamplers[2];
    staticSamplers[0].Init(0, D3D12_FILTER_ANISOTROPIC, D3D12_TEXTURE_ADDRESS_MODE_WRAP, D3D12_TEXTURE_ADDRESS_MODE_WRAP, D3D12_TEXTURE_ADDRESS_MODE_WRAP);
    staticSamplers[0].MaxAnisotropy = 16;
    staticSamplers[1].Init(1, D3D12_FILTER_MIN_MAG_MIP_LINEAR, D3D12_TEXTURE_ADDRESS_MODE_CLAMP, D3D12_TEXTURE_ADDRESS_MODE_CLAMP, D3D12_TEXTURE_ADDRESS_MODE_CLAMP);

    // Serialize and Create
    CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC rootSignatureDesc;
    rootSignatureDesc.Init_1_1(_countof(rootParameters), rootParameters, _countof(staticSamplers), staticSamplers, D3D12_ROOT_SIGNATURE_FLAG_NONE);

    ComPtr<ID3DBlob> signature;
    ComPtr<ID3DBlob> error;
    HRESULT hr = D3D12SerializeVersionedRootSignature(&rootSignatureDesc, &signature, &error);
    if (FAILED(hr)) {
        if (error) { OutputDebugStringA(static_cast<char*>(error->GetBufferPointer())); }
        ThrowIfFailed(hr);
    }

    ComPtr<ID3D12RootSignature> pRootSignature;
    ThrowIfFailed(m_device->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(), IID_PPV_ARGS(&pRootSignature)));
    return pRootSignature;
}

ComPtr<ID3D12RootSignature> Renderer::CreateComputeSignature()
{
    // Define the Root Parameters (Descriptor Table + Constants)
    CD3DX12_ROOT_PARAMETER1 rootParameters[2];
    std::vector<CD3DX12_DESCRIPTOR_RANGE1> ranges;
    ranges.reserve(40); // Reserve space to avoid reallocations

    // Flags to optimize driver behavior
    // VOLATILE: The buffer content might change (UAVs)
    // STATIC: The descriptor/buffer won't change during execution
    const auto VOLATILE = D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE;
    const auto STATIC   = D3D12_DESCRIPTOR_RANGE_FLAG_DATA_STATIC;


    // Slot 0:  UAV u0 (Output)
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0, 0, VOLATILE, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);

    // Slot 1:  UAV u1 (Permanent Data / Accumulation)
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 1, 0, VOLATILE, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);

    // Slot 2:  SRV t0 (TLAS)
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, STATIC, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);

    // Slot 3:  SRV t1 (Global Index Buffer)
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 1, 0, STATIC, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);

    // Slot 4:  SRV t2 (Global Vertex Buffer)
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 2, 0, STATIC, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);

    // Slot 5:  CBV b0 (Camera)
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_CBV, 1, 0, 0, STATIC, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);

    // Slot 6:  SRV t3 (Instance Properties)
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 3, 0, STATIC, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);

    // Slot 7:  SRV t4 (Material IDs)
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 4, 0, STATIC, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);

    // Slot 8:  SRV t5 (Materials)
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 5, 0, STATIC, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);

    // Slot 9:  SRV t6 (Emissive Triangles)
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 6, 0, STATIC, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);

    // Slot 10-15: UAVs u2-u7 (Reservoirs & Sample Buffers)
    // u2
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 2, 0, VOLATILE, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    // u3
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 3, 0, VOLATILE, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    // u4
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 4, 0, VOLATILE, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    // u5
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 5, 0, VOLATILE, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    // u6
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 6, 0, VOLATILE, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    // u7
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 7, 0, VOLATILE, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);

    // Slot 16: SRV t7 (Alias Prob)
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 7, 0, STATIC, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);

    // Slot 17: SRV t8 (Alias Idx)
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 8, 0, STATIC, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);

    // Slot 18: UAV u8 (Scratch Ping)
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 8, 0, VOLATILE, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);

    // Slot 19: UAV u9 (Initial BSDF Rays)
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 9, 0, VOLATILE, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);

    // Slot 20-23: LightTree SRVs (t9-t12)
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 9, 0, STATIC, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 10, 0, STATIC, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 11, 0, STATIC, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 12, 0, STATIC, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);

    // Slot 24: SRV t15 (TriToLightId) -- Note the jump to register t15
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 15, 0, STATIC, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);

    // Slot 25-27: LightTree Lookups (t16-t18)
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 16, 0, STATIC, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 17, 0, STATIC, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 18, 0, STATIC, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);

    // Slot 28-31: Texture Arrays (t30-t33) -- Note jump to register t30
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 30, 0, STATIC, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 31, 0, STATIC, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 32, 0, STATIC, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 33, 0, STATIC, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);

    // Slot 32: UAV u10 (PathState Buffer)
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 10, 0, VOLATILE, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);

    // Slot 33: UAV u34 (Global Counters)
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 34, 0, VOLATILE, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);

    // Slot 34: UAV u35 (Indirect Args)
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 35, 0, VOLATILE, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);

    // Slot 35: UAVs u36-u39 (Pixel Stacks) - Range of 4 descriptors
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 4, 36, 0, VOLATILE, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);

    ranges.emplace_back().Init(
            D3D12_DESCRIPTOR_RANGE_TYPE_UAV,
            11, // NumDescriptors
            11, // BaseShaderRegister (u11)
            0,  // RegisterSpace
            VOLATILE,
            D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND
    );

    // Slot 36: Sort Buffers (3 UAVs)
    // u60: Sort Count
    // u61: Sort Offset
    // u62: Sort Bounds
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV,  3, 60, 0, VOLATILE,  D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);

    // -----------------------------------------------------------------------

    // Parameter 0: The Descriptor Table (SRV/UAV/CBV)
    rootParameters[0].InitAsDescriptorTable(static_cast<UINT>(ranges.size()), ranges.data(), D3D12_SHADER_VISIBILITY_ALL);

    // Parameter 1: Constants (b1) - Used for dimensions, etc.
    rootParameters[1].InitAsConstants(6, 1, 0, D3D12_SHADER_VISIBILITY_ALL);

    // Static Samplers (Unchanged)
    CD3DX12_STATIC_SAMPLER_DESC staticSamplers[2];
    staticSamplers[0].Init(0, D3D12_FILTER_ANISOTROPIC, D3D12_TEXTURE_ADDRESS_MODE_WRAP, D3D12_TEXTURE_ADDRESS_MODE_WRAP, D3D12_TEXTURE_ADDRESS_MODE_WRAP);
    staticSamplers[0].MaxAnisotropy = 16;
    staticSamplers[1].Init(1, D3D12_FILTER_MIN_MAG_MIP_LINEAR, D3D12_TEXTURE_ADDRESS_MODE_CLAMP, D3D12_TEXTURE_ADDRESS_MODE_CLAMP, D3D12_TEXTURE_ADDRESS_MODE_CLAMP);

    // Serialize and Create
    CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC rootSignatureDesc;
    rootSignatureDesc.Init_1_1(_countof(rootParameters), rootParameters, _countof(staticSamplers), staticSamplers, D3D12_ROOT_SIGNATURE_FLAG_NONE);

    ComPtr<ID3DBlob> signature;
    ComPtr<ID3DBlob> error;
    HRESULT hr = D3D12SerializeVersionedRootSignature(&rootSignatureDesc, &signature, &error);
    if (FAILED(hr)) {
        if (error) { OutputDebugStringA(static_cast<char*>(error->GetBufferPointer())); }
        ThrowIfFailed(hr);
    }

    ComPtr<ID3D12RootSignature> pRootSignature;
    ThrowIfFailed(m_device->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(), IID_PPV_ARGS(&pRootSignature)));
    return pRootSignature;
}



//-----------------------------------------------------------------------------
// The hit shader communicates only through the ray payload, and therefore does
// not require any resources
//

ComPtr<ID3D12RootSignature> Renderer::CreateHitSignature() {
    nv_helpers_dx12::RootSignatureGenerator rsc;

    // REMOVE ALL PARAMETERS HERE.
    // We rely on the Global Root Signature for resources (t0, t1, t2, etc.).

    /*
    rsc.AddRootParameter(D3D12_ROOT_PARAMETER_TYPE_SRV, 2);
    rsc.AddRootParameter(D3D12_ROOT_PARAMETER_TYPE_SRV, 1);
    rsc.AddRootParameter(D3D12_ROOT_PARAMETER_TYPE_CBV, 0);
    rsc.AddHeapRangesParameter(...);
    */

    return rsc.Generate(m_device.Get(), true);
}

//-----------------------------------------------------------------------------
// The miss shader communicates only through the ray payload, and therefore
// does not require any resources
//
ComPtr<ID3D12RootSignature> Renderer::CreateMissSignature() {
  nv_helpers_dx12::RootSignatureGenerator rsc;
  return rsc.Generate(m_device.Get(), true);
}

//-----------------------------------------------------------------------------
//
// The raytracing pipeline binds the shader code, root signatures and pipeline
// characteristics in a single structure used by DXR to invoke the shaders and
// manage temporary memory during raytracing
//
//
// -----------------------------------------------------------------------------
// Builds the hybrid DXR + CS pipeline.
// -----------------------------------------------------------------------------
void Renderer::CreateRaytracingPipeline()
{
    nv_helpers_dx12::RayTracingPipelineGenerator pipeline(m_device.Get());

    // Root-signatures
    m_rayGenSignature = CreateRayGenSignature();
    m_computeSignature = CreateComputeSignature();
    m_missSignature   = CreateMissSignature();
    m_hitSignature    = CreateHitSignature();

    pipeline.SetGlobalRootSignature(m_rayGenSignature.Get());

    // Iterate over the parsed pass list
    m_rayGenLibs.clear();
    m_csPSOs   .clear();
    m_passIndex.clear();

    uint32_t nextCs = 0;      // running index for compute PSOs
    uint32_t rgSlot = 0;      // running SBT slot for ray-gen shaders

    for (PassDesc& p : m_passes)
    {
        if (p.stage == Stage::Barrier) continue;

        if (p.stage == Stage::LoopStart || p.stage == Stage::LoopEnd) continue;

        if (p.stage == Stage::PingSwap) continue;

        if (p.stage == Stage::ClearSort) continue;

        if (p.stage == Stage::ML) continue;

        if (p.isWorkGraph)
        {
            // compile WG DXIL
            ComPtr<IDxcBlob> lib = nv_helpers_dx12::CompileWG(p.file.c_str());

            // export every public symbol found in the library
            static const D3D12_EXPORT_DESC kExports[] =
            {
                { L"main",  nullptr, D3D12_EXPORT_FLAG_NONE }   // every compute shader / WG uses the same main entry for now
            };

            D3D12_DXIL_LIBRARY_DESC dxilDesc{};
            dxilDesc.DXILLibrary = { lib->GetBufferPointer(), lib->GetBufferSize() };
            dxilDesc.NumExports  = 0;
            dxilDesc.pExports    = nullptr;

            D3D12_STATE_SUBOBJECT subobjects[3]{};
            subobjects[0].Type  = D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY;
            subobjects[0].pDesc = &dxilDesc;

            subobjects[1].Type  = D3D12_STATE_SUBOBJECT_TYPE_GLOBAL_ROOT_SIGNATURE;
            subobjects[1].pDesc = m_computeSignature.GetAddressOf();

            static const LPCWSTR kWorkGraphName = L"main";

            D3D12_WORK_GRAPH_DESC wgDesc = {};
            wgDesc.ProgramName               = kWorkGraphName;
            wgDesc.Flags                     = D3D12_WORK_GRAPH_FLAG_INCLUDE_ALL_AVAILABLE_NODES;
            wgDesc.NumEntrypoints            = 0;            // no explicit entrypoints
            wgDesc.pEntrypoints              = nullptr;
            wgDesc.NumExplicitlyDefinedNodes = 0;            // no overrides
            wgDesc.pExplicitlyDefinedNodes   = nullptr;

            // plug into your subobject array:
            subobjects[2].Type  = D3D12_STATE_SUBOBJECT_TYPE_WORK_GRAPH;
            subobjects[2].pDesc = &wgDesc;

            D3D12_STATE_OBJECT_DESC soDesc{};
            soDesc.Type          = D3D12_STATE_OBJECT_TYPE_EXECUTABLE;
            soDesc.NumSubobjects = _countof(subobjects);
            soDesc.pSubobjects   = subobjects;

            D3D12_FEATURE_DATA_D3D12_OPTIONS21 opts = {};
            HRESULT hr = m_device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS21, &opts, sizeof(opts));
            if (FAILED(hr) || opts.WorkGraphsTier == D3D12_WORK_GRAPHS_TIER_NOT_SUPPORTED) {
                // Print a fatal error and bail out
                MessageBoxA(nullptr, "Work Graphs not supported on this device/driver/OS", "Error", MB_OK);
                abort();
            }

            ComPtr<ID3D12StateObject> so;
            HRESULT hr2 = m_device->CreateStateObject(&soDesc, IID_PPV_ARGS(&so));
            DumpD3D12Messages(m_device.Get());
            ThrowIfFailed(hr2);

            ComPtr<ID3D12StateObjectProperties1> soProps1;
            ThrowIfFailed(so->QueryInterface(IID_PPV_ARGS(&soProps1)));

            ComPtr<ID3D12WorkGraphProperties> wgProps;
            ThrowIfFailed(so->QueryInterface(IID_PPV_ARGS(&wgProps)));

            WgRuntimeData rt{};

            D3D12_PROGRAM_IDENTIFIER pid = soProps1->GetProgramIdentifier(L"main");
            rt.id = pid;

            D3D12_WORK_GRAPH_MEMORY_REQUIREMENTS mem{};
            wgProps->GetWorkGraphMemoryRequirements(0, &mem);            // index 0

            UINT64 backingSize = mem.MaxSizeInBytes;
            if (backingSize)
            {
                CD3DX12_HEAP_PROPERTIES hp(D3D12_HEAP_TYPE_DEFAULT);
                CD3DX12_RESOURCE_DESC   buf =
                    CD3DX12_RESOURCE_DESC::Buffer(backingSize,
                                                  D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

                ThrowIfFailed(m_device->CreateCommittedResource(
                    &hp, D3D12_HEAP_FLAG_NONE, &buf,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                    IID_PPV_ARGS(&rt.backingRes)));

                rt.backing.StartAddress = rt.backingRes->GetGPUVirtualAddress();
                rt.backing.SizeInBytes  = backingSize;
            }

            p.wgIdx = static_cast<uint32_t>(m_wgStateObjects.size());
            m_wgStateObjects.push_back(so);
            m_wgProps      .push_back(wgProps);
            m_wgRuntime    .push_back(std::move(rt));
            continue;
        }


        if ((p.stage == Stage::Compute || p.stage == Stage::Wavefront || p.stage == Stage::FixedCompute) && !p.isWorkGraph)
        {
            // compile CS and make a PSO
            ComPtr<IDxcBlob> cs = nv_helpers_dx12::CompileCS(p.file.c_str(), L"main");

            D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
            desc.pRootSignature = m_computeSignature.Get();
            desc.CS             = { cs->GetBufferPointer(), cs->GetBufferSize() };

            ComPtr<ID3D12PipelineState> pso;
            ThrowIfFailed(m_device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&pso)));

            m_csPSOs.push_back(pso);
            p.psoIdx = nextCs++;
            continue;
        }

        if (p.stage == Stage::Callable)
        {
            // Extract base name: "Folder/Shader.hlsl" -> "Shader"
            std::wstring base = p.file.substr(p.file.find_last_of(L"/\\") + 1);
            base = base.substr(0, base.rfind(L'.'));

            // Compile the library
            // NOTE: Ensure your HLSL file has [shader("callable")] void ShaderName(...)
            ComPtr<IDxcBlob> lib = nv_helpers_dx12::CompileShaderLibrary(p.file.c_str());

            // Add to pipeline with the filename as the export symbol
            pipeline.AddLibrary(lib.Get(), { base.c_str() });

            // Store for SBT generation
            m_callableShaderNames.push_back(base);
            continue;
        }


        // ray-generation library
        std::wstring base = p.file.substr(p.file.find_last_of(L"/\\") + 1);
        base = base.substr(0, base.rfind(L'.'));

        ComPtr<IDxcBlob> lib =
            nv_helpers_dx12::CompileShaderLibrary(p.file.c_str());

        m_rayGenLibs.push_back(lib);
        pipeline.AddLibrary(lib.Get(), { base.c_str() });

        m_passIndex[p.file] = rgSlot++;     // SBT slot for this RG shader
    }


    for (PassDesc& p : m_passes)
    {
        if (p.stage != Stage::Compute || p.isWorkGraph) continue;
        ComPtr<IDxcBlob> cs = nv_helpers_dx12::CompileCS(p.file.c_str(), L"main");

        D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
        desc.pRootSignature = m_computeSignature.Get();   // same RS for CS
        desc.CS             = { cs->GetBufferPointer(), cs->GetBufferSize() };

        ComPtr<ID3D12PipelineState> pso;
        ThrowIfFailed(m_device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&pso)));

        m_csPSOs.push_back(pso);   // keep alive
        p.psoIdx = nextCs++;
    }


    // Libraries:  miss / hit / shadow
    m_missLibrary   = nv_helpers_dx12::CompileShaderLibrary(L"Miss_v8.hlsl");
    m_hitLibrary    = nv_helpers_dx12::CompileShaderLibrary(L"Hit_v8.hlsl");
    m_shadowLibrary = nv_helpers_dx12::CompileShaderLibrary(L"ShadowRay.hlsl");

    pipeline.AddLibrary(m_missLibrary.Get(),   { L"Miss" });
    pipeline.AddLibrary(m_shadowLibrary.Get(), { L"ShadowClosestHit", L"ShadowMiss" });
    pipeline.AddLibrary(m_hitLibrary.Get(),    { L"ClosestHit" });

    // Hit-groups
    pipeline.AddHitGroup(L"HitGroup",        L"ClosestHit");
    pipeline.AddHitGroup(L"ShadowHitGroup",  L"ShadowClosestHit");

    // Root-signature associations
    /*for (const PassDesc& p : m_passes)
    {
        if (p.stage != Stage::RayGen) continue;

        std::wstring base = p.file.substr(0, p.file.rfind(L'.'));
        pipeline.AddRootSignatureAssociation(
            m_rayGenSignature.Get(), { base.c_str() });
    }*/

    pipeline.AddRootSignatureAssociation(m_missSignature.Get(),  { L"Miss", L"ShadowMiss" });
    pipeline.AddRootSignatureAssociation(m_hitSignature.Get(),   { L"HitGroup" });
    pipeline.AddRootSignatureAssociation(m_shadowSignature.Get(),{ L"ShadowHitGroup" });

    // Payload / attribute / recursion depth (we dont use recursion)
    pipeline.SetMaxPayloadSize( 19*sizeof(float) + 4*sizeof(UINT));
    pipeline.SetMaxAttributeSize( 2*sizeof(float) );       // barycentrics
    pipeline.SetMaxRecursionDepth(1);

    // Generate state object
    m_rtStateObject = pipeline.Generate();
    ThrowIfFailed(
        m_rtStateObject->QueryInterface(IID_PPV_ARGS(&m_rtStateObjectProps)));

    // 1. Query individual shader stack requirements
    // Note: Use the export name (usually the filename without extension based on your compilation logic)
    UINT64 rgStackSize = m_rtStateObjectProps->GetShaderStackSize(L"Pass_raygen_v8");
    std::wcout << L"\n[Stack Analysis] RayGen (Pass_raygen_v8): " << rgStackSize << L" bytes" << std::endl;

    UINT64 maxCallableStack = 0;
    for (const auto& name : m_callableShaderNames)
    {
        UINT64 size = m_rtStateObjectProps->GetShaderStackSize(name.c_str());
        std::wcout << L"[Stack Analysis] Callable (" << name << L"): " << size << L" bytes" << std::endl;
        if (size > maxCallableStack) maxCallableStack = size;
    }

    UINT64 missStackSize = m_rtStateObjectProps->GetShaderStackSize(L"Miss");
    std::wcout << L"[Stack Analysis] Miss (Miss): " << missStackSize << L" bytes" << std::endl;

    // 2. Calculate Total Pipeline Stack Size
    // Formula: RayGen + Max(Hit, Miss, Callable) * RecursionDepth
    // Since you use CallShader (which counts as recursion in terms of stack depth 1->2),
    // and assuming MaxRecursionDepth is 1 (RayGen calls Callable, Callable doesn't recurse):
    UINT64 totalStackSize = rgStackSize + std::max(maxCallableStack, missStackSize);

    // Add a little padding for safety (e.g. 64 bytes) or alignment
    totalStackSize = (totalStackSize + 255) & ~255; // Align to 256 bytes

    std::wcout << L"[Stack Analysis] CALCULATED TOTAL REQUIRED: " << totalStackSize << L" bytes" << std::endl;

    // 3. FORCE the stack size
    // If you don't do this, the driver often defaults to 4096 bytes per ray, killing occupancy!
    m_rtStateObjectProps->SetPipelineStackSize(totalStackSize);
    std::wcout << L"[Stack Analysis] Pipeline Stack Size set to: " << totalStackSize << L" bytes\n" << std::endl;
}


//-----------------------------------------------------------------------------
//
// Allocate the buffer holding the raytracing output, with the same size as the
// output image
//
void Renderer::CreateRaytracingOutputBuffer() {
  D3D12_RESOURCE_DESC resDesc = {};
  resDesc.DepthOrArraySize = 30 * 2;
  resDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
  resDesc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;

  resDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
  resDesc.Width = GetWidth();
  resDesc.Height = GetHeight();
  resDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
  resDesc.MipLevels = 1;
  resDesc.SampleDesc.Count = 1;
  ThrowIfFailed(m_device->CreateCommittedResource(
      &nv_helpers_dx12::kDefaultHeapProps, D3D12_HEAP_FLAG_NONE, &resDesc,
      D3D12_RESOURCE_STATE_COPY_SOURCE, nullptr,
      IID_PPV_ARGS(&m_outputResource)));


    // Create a texture description
    D3D12_RESOURCE_DESC textureDesc = {};
    textureDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    textureDesc.Width = GetWidth();
    textureDesc.Height = GetHeight();
    textureDesc.DepthOrArraySize = 1;
    textureDesc.MipLevels = 1;
    textureDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    textureDesc.SampleDesc.Count = 1;
    textureDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    textureDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    // Create the texture resource
    ThrowIfFailed(m_device->CreateCommittedResource(
            &nv_helpers_dx12::kDefaultHeapProps,
            D3D12_HEAP_FLAG_NONE,
            &textureDesc,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            nullptr,
            IID_PPV_ARGS(&m_permanentDataTexture)));

    // m_scratchPing
    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension        = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width            = GetWidth();
    desc.Height           = GetHeight();
    desc.DepthOrArraySize = 16;
    desc.MipLevels        = 1;
    desc.Format           = DXGI_FORMAT_R32G32B32A32_FLOAT;
    desc.SampleDesc.Count = 1;
    desc.Layout           = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    desc.Flags            = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    ThrowIfFailed(m_device->CreateCommittedResource(
        &nv_helpers_dx12::kDefaultHeapProps,
        D3D12_HEAP_FLAG_NONE,
        &desc,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        nullptr,
        IID_PPV_ARGS(&m_scratchPing)));


    auto CreateRawBuffer = [&](ComPtr<ID3D12Resource>& resource, UINT sizeInBytes, const std::wstring& name) {
        D3D12_RESOURCE_DESC bufferDesc = CD3DX12_RESOURCE_DESC::Buffer(
            sizeInBytes,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
        );

        ThrowIfFailed(m_device->CreateCommittedResource(
            &nv_helpers_dx12::kDefaultHeapProps,
            D3D12_HEAP_FLAG_NONE,
            &bufferDesc,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            nullptr,
            IID_PPV_ARGS(&resource)));

        resource->SetName(name.c_str());
    };

    UINT pixelCount = GetWidth() * GetHeight();
    UINT reservoirSizeDI = pixelCount * sizeof(Reservoir_DI);
    UINT reservoirSizeGI = pixelCount * sizeof(Reservoir_GI);
    UINT sampleSize      = pixelCount * sizeof(SampleData);
    UINT initRaySize     = pixelCount * sizeof(InitialBSDFRay);

    CreateRawBuffer(m_reservoirBuffer,      reservoirSizeDI, L"ReservoirBuffer_DI_1");
    CreateRawBuffer(m_reservoirBuffer_2,    reservoirSizeDI, L"ReservoirBuffer_DI_2");
    CreateRawBuffer(m_reservoirBuffer_3,    reservoirSizeGI, L"ReservoirBuffer_GI_1");
    CreateRawBuffer(m_reservoirBuffer_4,    reservoirSizeGI, L"ReservoirBuffer_GI_2");
    CreateRawBuffer(m_sampleBuffer_current, sampleSize,      L"SampleBuffer_Current");
    CreateRawBuffer(m_sampleBuffer_last,    sampleSize,      L"SampleBuffer_Last");
    CreateRawBuffer(m_initialBSDFRayBuffer, initRaySize,     L"InitialBSDFRayBuffer");

    CreatePathStateBuffer();
}

//-----------------------------------------------------------------------------
//
// Create the main heap used by the shaders, which will give access to the
// raytracing output and the top-level acceleration structure
//
// REPLACE THE ENTIRE CONTENTS OF THIS FUNCTION

void Renderer::CreateShaderResourceHeap() {
    // ---------------------------------------------------------------------------
    // 1. Create Main Heap (Shader Visible - GPU Access)
    // ---------------------------------------------------------------------------
    // We need space for ~40 descriptors + buffers + sort buffers
    m_srvUavHeap = nv_helpers_dx12::CreateDescriptorHeap(
        m_device.Get(), 100, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, true);
    m_srvUavHeap->SetName(L"Main SRV/UAV Heap");

    // ---------------------------------------------------------------------------
    // 2. Create Staging Heap (Non-Shader Visible - CPU Access)
    // ---------------------------------------------------------------------------
    // CRITICAL: This heap must NOT be shader visible (Flag = NONE).
    // It must persist (be a member variable), or handles become invalid.
    D3D12_DESCRIPTOR_HEAP_DESC stagingDesc = {};
    stagingDesc.NumDescriptors = 4; // SortCount, SortOffset, SortBounds
    stagingDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    stagingDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE; // CPU Only
    stagingDesc.NodeMask = 0;
    ThrowIfFailed(m_device->CreateDescriptorHeap(&stagingDesc, IID_PPV_ARGS(&m_stagingUavHeap)));
    m_stagingUavHeap->SetName(L"Staging UAV Heap");

    // ---------------------------------------------------------------------------
    // 3. Setup Handles
    // ---------------------------------------------------------------------------
    // Handles for Main Heap
    CD3DX12_CPU_DESCRIPTOR_HANDLE handle(m_srvUavHeap->GetCPUDescriptorHandleForHeapStart());
    CD3DX12_GPU_DESCRIPTOR_HANDLE gpuHandle(m_srvUavHeap->GetGPUDescriptorHandleForHeapStart());
    const UINT inc = m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    // Handles for Staging Heap
    CD3DX12_CPU_DESCRIPTOR_HANDLE stagingHandle(m_stagingUavHeap->GetCPUDescriptorHandleForHeapStart());

    // Helper to advance Main Heap slots
    auto nextSlot = [&]() {
        handle.Offset(1, inc);
        gpuHandle.Offset(1, inc);
    };

    // Helper to advance Staging Heap slots
    auto nextStaging = [&]() {
        stagingHandle.Offset(1, inc);
    };

    // Helper for null descriptors
    auto createNullSRV = [&](D3D12_SRV_DIMENSION dim = D3D12_SRV_DIMENSION_BUFFER) {
        D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
        srvDesc.Format = DXGI_FORMAT_R32_UINT;
        srvDesc.ViewDimension = dim;
        srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        if(dim == D3D12_SRV_DIMENSION_BUFFER) srvDesc.Buffer.NumElements = 1;
        m_device->CreateShaderResourceView(nullptr, &srvDesc, handle);
        nextSlot();
    };

    // ---------------------------------------------------------------------------
    // 4. Fill Descriptors
    // ---------------------------------------------------------------------------

    // --- RANGE 0: UAV u0 (Output) ---
    {
        D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
        uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
        uavDesc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
        uavDesc.Texture2DArray.ArraySize = m_outputResource->GetDesc().DepthOrArraySize;
        m_device->CreateUnorderedAccessView(m_outputResource.Get(), nullptr, &uavDesc, handle);
        nextSlot();
    }

    // --- RANGE 1: UAV u1 (Permanent Data / Accumulation) ---
    {
        D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
        uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        uavDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
        m_device->CreateUnorderedAccessView(m_permanentDataTexture.Get(), nullptr, &uavDesc, handle);
        nextSlot();
    }

    // --- RANGE 2: SRV t0 (TLAS) ---
    {
        D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
        srvDesc.Format = DXGI_FORMAT_UNKNOWN;
        srvDesc.ViewDimension = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE;
        srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srvDesc.RaytracingAccelerationStructure.Location = m_topLevelASBuffers.pResult->GetGPUVirtualAddress();
        m_device->CreateShaderResourceView(nullptr, &srvDesc, handle);
        nextSlot();
    }

    // --- RANGE 3: SRV t1 (Global Index Buffer) ---
    {
        D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
        srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srvDesc.Format = DXGI_FORMAT_R32_UINT;
        srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        srvDesc.Buffer.NumElements = m_totalIndexCount;
        m_device->CreateShaderResourceView(m_indexGlobal.Get(), &srvDesc, handle);
        nextSlot();
    }

    // --- RANGE 4: SRV t2 (Global Vertex Buffer) ---
    {
        D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
        srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srvDesc.Format = DXGI_FORMAT_UNKNOWN;
        srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        srvDesc.Buffer.NumElements = m_totalVertexCount;
        srvDesc.Buffer.StructureByteStride = sizeof(BTriVertex);
        m_device->CreateShaderResourceView(m_vertexGlobal.Get(), &srvDesc, handle);
        nextSlot();
    }

    // --- RANGE 5: CBV b0 (Camera) ---
    {
        D3D12_CONSTANT_BUFFER_VIEW_DESC cbvDesc = {};
        cbvDesc.BufferLocation = m_cameraBuffer->GetGPUVirtualAddress();
        cbvDesc.SizeInBytes = m_cameraBufferSize;
        m_device->CreateConstantBufferView(&cbvDesc, handle);
        nextSlot();
    }

    // --- RANGE 6: SRV t3 (Instance Properties) ---
    {
        D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
        srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srvDesc.Format = DXGI_FORMAT_UNKNOWN;
        srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        srvDesc.Buffer.NumElements = static_cast<UINT>(m_instances.size());
        srvDesc.Buffer.StructureByteStride = sizeof(InstanceProperties);
        m_device->CreateShaderResourceView(m_instanceProperties.Get(), &srvDesc, handle);
        nextSlot();
    }

    // --- RANGE 7: SRV t4 (Material IDs) ---
    {
        D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
        srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srvDesc.Format = DXGI_FORMAT_R32_UINT;
        srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        srvDesc.Buffer.NumElements = static_cast<UINT>(m_materialIDs.size());
        m_device->CreateShaderResourceView(m_materialIndexBuffer.Get(), &srvDesc, handle);
        nextSlot();
    }

    // --- RANGE 8: SRV t5 (Materials) ---
    {
        D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
        srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srvDesc.Format = DXGI_FORMAT_UNKNOWN;
        srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        srvDesc.Buffer.NumElements = static_cast<UINT>(m_materials.size());
        srvDesc.Buffer.StructureByteStride = sizeof(Material);
        m_device->CreateShaderResourceView(m_materialBuffer.Get(), &srvDesc, handle);
        nextSlot();
    }

    // --- RANGE 9: SRV t6 (Emissive Triangles) ---
    {
        if (m_emissiveTrianglesBuffer) {
            D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
            srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            srvDesc.Format = DXGI_FORMAT_UNKNOWN;
            srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
            srvDesc.Buffer.NumElements = static_cast<UINT>(m_emissiveTriangles.size());
            srvDesc.Buffer.StructureByteStride = sizeof(LightTriangle);
            m_device->CreateShaderResourceView(m_emissiveTrianglesBuffer.Get(), &srvDesc, handle);
        } else { createNullSRV(); }
        nextSlot();
    }

    // Helper for Raw Buffer UAVs
    auto createRawUAV = [&](ComPtr<ID3D12Resource>& res, UINT sizeInBytes) {
        D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
        uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        uavDesc.Format = DXGI_FORMAT_R32_TYPELESS;
        uavDesc.Buffer.NumElements = sizeInBytes / 4;
        uavDesc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
        m_device->CreateUnorderedAccessView(res.Get(), nullptr, &uavDesc, handle);
        nextSlot();
    };

    UINT reservoirBytesDI = GetWidth() * GetHeight() * sizeof(Reservoir_DI);
    UINT reservoirBytesGI = GetWidth() * GetHeight() * sizeof(Reservoir_GI);
    UINT sampleBytes      = GetWidth() * GetHeight() * sizeof(SampleData);

    // --- RANGE 10-15: Reservoirs & Sample Buffers (u2-u7) ---
    createRawUAV(m_reservoirBuffer, reservoirBytesDI);       // u2
    createRawUAV(m_reservoirBuffer_2, reservoirBytesDI);     // u3
    createRawUAV(m_reservoirBuffer_3, reservoirBytesGI);     // u4
    createRawUAV(m_reservoirBuffer_4, reservoirBytesGI);     // u5
    createRawUAV(m_sampleBuffer_current, sampleBytes);       // u6
    createRawUAV(m_sampleBuffer_last, sampleBytes);          // u7

    // --- RANGE 16: SRV t7 (Alias Prob) ---
    {
        if (m_aliasProbBuffer) {
            D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
            srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            srvDesc.Format = DXGI_FORMAT_R32_FLOAT;
            srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
            srvDesc.Buffer.NumElements = static_cast<UINT>(m_aliasProb.size());
            m_device->CreateShaderResourceView(m_aliasProbBuffer.Get(), &srvDesc, handle);
        } else { createNullSRV(); }
        nextSlot();
    }

    // --- RANGE 17: SRV t8 (Alias Idx) ---
    {
        if (m_aliasIdxBuffer) {
            D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
            srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            srvDesc.Format = DXGI_FORMAT_R32_UINT;
            srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
            srvDesc.Buffer.NumElements = static_cast<UINT>(m_aliasIdx.size());
            m_device->CreateShaderResourceView(m_aliasIdxBuffer.Get(), &srvDesc, handle);
        } else { createNullSRV(); }
        nextSlot();
    }

    // --- RANGE 18: UAV u8 (Scratch Ping) ---
    {
        D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
        uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
        uavDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
        uavDesc.Texture2DArray.ArraySize = 16;
        m_device->CreateUnorderedAccessView(m_scratchPing.Get(), nullptr, &uavDesc, handle);
        nextSlot();
    }

    // --- RANGE 19: UAV u9 (Initial Rays) ---
    createRawUAV(m_initialBSDFRayBuffer, GetWidth() * GetHeight() * sizeof(InitialBSDFRay));

    // --- RANGES 20-23: LightTree SRVs (t9-t12) ---
    m_lightTree.WriteSrvs(m_device.Get(), handle);
    handle.Offset(4, inc); // Skip 4 slots
    gpuHandle.Offset(4, inc);

    // --- RANGES 24-27: More LightTree SRVs (t15-t18) ---
    // Range 24: SRV t15 (TriToLightId)
    {
        if (m_triToLightIdBuffer) {
            D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
            srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            srvDesc.Format = DXGI_FORMAT_R32_UINT;
            srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
            srvDesc.Buffer.NumElements = static_cast<UINT>(m_triToLightId.size());
            m_device->CreateShaderResourceView(m_triToLightIdBuffer.Get(), &srvDesc, handle);
        } else { createNullSRV(); }
        nextSlot();
    }

    // Range 25, 26, 27: LightTree Lookups (t16, t17, t18)
    m_lightTree.WriteLookupSrvs(m_device.Get(), handle);
    handle.Offset(3, inc);
    gpuHandle.Offset(3, inc);

    // --- RANGES 28-31: Texture Arrays (t30-t33) ---
    auto createTexSrv = [&](ComPtr<ID3D12Resource>& res) {
        if (!res) { createNullSRV(D3D12_SRV_DIMENSION_TEXTURE2DARRAY); return; }
        D3D12_SHADER_RESOURCE_VIEW_DESC desc = {};
        desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        desc.Format = res->GetDesc().Format;
        desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
        desc.Texture2DArray.MipLevels = res->GetDesc().MipLevels;
        desc.Texture2DArray.ArraySize = res->GetDesc().DepthOrArraySize;
        m_device->CreateShaderResourceView(res.Get(), &desc, handle);
        nextSlot();
    };

    createTexSrv(m_albedoTextureArray);
    createTexSrv(m_normalTextureArray);
    createTexSrv(m_rmaTextureArray);
    createTexSrv(m_lutTextureArray);

    // --- RANGE 32: UAV u10 (PathState) ---
    createRawUAV(m_pathStateBuffer, GetWidth() * GetHeight() * 88);

    // --- RANGE 33: UAV u34 (Global Counters) ---
    {
        D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
        uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        uavDesc.Format = DXGI_FORMAT_R32_TYPELESS;
        uavDesc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
        uavDesc.Buffer.NumElements = MAX_STACKS;
        m_device->CreateUnorderedAccessView(m_globalCounterBuffer.Get(), nullptr, &uavDesc, handle);
        nextSlot();
    }

    // --- RANGE 34: UAV u35 (Indirect Args) ---
    {
        D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
        uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        uavDesc.Format = DXGI_FORMAT_UNKNOWN; // Structured
        uavDesc.Buffer.NumElements = MAX_INDIRECT_COMMANDS;
        uavDesc.Buffer.StructureByteStride = sizeof(D3D12_DISPATCH_ARGUMENTS);
        m_device->CreateUnorderedAccessView(m_indirectArgsBuffer.Get(), nullptr, &uavDesc, handle);
        nextSlot();
    }

    // --- RANGE 35: UAVs u36-u39 (Pixel Stacks) ---
    for(int i=0; i<4; ++i)
    {
        D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
        uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        uavDesc.Format = DXGI_FORMAT_UNKNOWN;
        uavDesc.Buffer.NumElements = GetWidth() * GetHeight();
        uavDesc.Buffer.StructureByteStride = 8; // uint2
        m_device->CreateUnorderedAccessView(m_stackBuffers[i].Get(), nullptr, &uavDesc, handle);
        nextSlot();
    }

    // --- RANGE 36: Sort Buffers (u60, u61, u62) ---
    // [CRITICAL FIX] We create in Staging, Store Staging Handle, Copy to Main, Store Main Handle.

    // [Inside CreateShaderResourceHeap, appending to the end]

    // --- RANGE 37: DLSS Full Inputs (u11 - u21) ---
    {
        auto createUAV = [&](ID3D12Resource* res, DXGI_FORMAT fmt) {
            D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
            uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
            uavDesc.Format = fmt;
            m_device->CreateUnorderedAccessView(res, nullptr, &uavDesc, handle);
            nextSlot();
        };

        // Core
        createUAV(m_dlssDepth.Get(),          DXGI_FORMAT_R32_FLOAT);           // u11
        createUAV(m_dlssMVec.Get(),           DXGI_FORMAT_R16G16_FLOAT);        // u12
        createUAV(m_dlssNormals.Get(),        DXGI_FORMAT_R16G16B16A16_FLOAT);  // u13
        createUAV(m_dlssDiffuseAlbedo.Get(),  DXGI_FORMAT_R16G16B16A16_FLOAT);  // u14
        createUAV(m_dlssOutput.Get(),         DXGI_FORMAT_R16G16B16A16_FLOAT);  // u15

        // Extra RR
        createUAV(m_dlssSpecularAlbedo.Get(), DXGI_FORMAT_R16G16B16A16_FLOAT);  // u16
        createUAV(m_dlssRoughness.Get(),      DXGI_FORMAT_R16_FLOAT);           // u17
        createUAV(m_dlssSpecMVec.Get(),       DXGI_FORMAT_R16G16_FLOAT);        // u18
        createUAV(m_dlssSpecHitDist.Get(),    DXGI_FORMAT_R16_FLOAT);           // u19

        // Optional
        createUAV(m_dlssTransparency.Get(),     DXGI_FORMAT_R16G16B16A16_FLOAT); // u20
        createUAV(m_dlssColorBeforeTrans.Get(), DXGI_FORMAT_R16G16B16A16_FLOAT); // u21
    }

    // 1. Sort Count (u60)
    {
        D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
        uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        uavDesc.Format = DXGI_FORMAT_R32_TYPELESS;
        uavDesc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
        uavDesc.Buffer.NumElements = SORT_BUCKETS;

        m_device->CreateUnorderedAccessView(m_sortCountBuffer.Get(), nullptr, &uavDesc, stagingHandle);
        m_sortCountCpuHandle = stagingHandle;

        m_device->CopyDescriptorsSimple(1, handle, stagingHandle, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        m_sortCountGpuHandle = gpuHandle;

        nextSlot();
        nextStaging();
    }

    // 2. Sort Offset (u61)
    {
        D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
        uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        uavDesc.Format = DXGI_FORMAT_R32_TYPELESS;
        uavDesc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
        uavDesc.Buffer.NumElements = SORT_BUCKETS;

        m_device->CreateUnorderedAccessView(m_sortOffsetBuffer.Get(), nullptr, &uavDesc, stagingHandle);
        m_sortOffsetCpuHandle = stagingHandle;

        m_device->CopyDescriptorsSimple(1, handle, stagingHandle, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        m_sortOffsetGpuHandle = gpuHandle;

        nextSlot();
        nextStaging();
    }

    // 3. Sort Bounds (u62)
    {
        D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
        uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        uavDesc.Format = DXGI_FORMAT_R32_TYPELESS;
        uavDesc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
        uavDesc.Buffer.NumElements = 8;

        m_device->CreateUnorderedAccessView(m_sortBoundsBuffer.Get(), nullptr, &uavDesc, stagingHandle);
        m_sortBoundsCpuHandle = stagingHandle;

        m_device->CopyDescriptorsSimple(1, handle, stagingHandle, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        m_sortBoundsGpuHandle = gpuHandle;

        nextSlot();
        nextStaging();
    }

    std::wcout << L"Heap recreated strictly matching Root Signature." << std::endl;
}

//-----------------------------------------------------------------------------
//
// The Shader Binding Table (SBT) is the cornerstone of the raytracing setup:
// this is where the shader resources are bound to the shaders, in a way that
// can be interpreted by the raytracer on GPU. In terms of layout, the SBT
// contains a series of shader IDs with their resource pointers. The SBT
// contains the ray generation shader, the miss shaders, then the hit groups.
// Using the helper class, those can be specified in arbitrary order.
//
void Renderer::CreateShaderBindingTable() {
    m_sbtHelper.Reset();
    D3D12_GPU_DESCRIPTOR_HANDLE heapHandle =
        m_srvUavHeap->GetGPUDescriptorHandleForHeapStart();
    auto heapPointer = reinterpret_cast<UINT64*>(heapHandle.ptr);

    //  RAY‑GEN SECTION
    for (const auto& entry : m_passSequence)
    {
        if (entry == L"barrier") continue;
        if (entry.rfind(L"loop:", 0) == 0) continue;
        if (entry == L"endloop") continue;
        if (entry == L"pingswap") continue;
        if (entry == L"clearsort") continue;
        if (entry == L"ml") continue;
        if (entry.find(L"|cs:") != std::wstring::npos) continue;
        if (entry.find(L"|wf:") != std::wstring::npos) continue;
        if (entry.find(L"|wg:") != std::wstring::npos) continue;
        if (entry.find(L"|fx:") != std::wstring::npos) continue;
        if (entry.find(L"|call") != std::wstring::npos) continue;

        std::wstring base = entry.substr(entry.find_last_of(L"/\\") + 1);
        base = base.substr(0, base.rfind(L'.'));
        m_sbtHelper.AddRayGenerationProgram(base.c_str(), { heapPointer });
    }

    // The miss and hit shaders do not access any external resources
    std::wcout << L"Adding miss programs..." << std::endl;
    m_sbtHelper.AddMissProgram(L"Miss", {});
    m_sbtHelper.AddMissProgram(L"ShadowMiss", {});

    // Adding hit groups for each instance
    std::wcout << L"Adding hit groups for instances..." << std::endl;
    for (int i = 0; i < m_instances.size(); ++i) {
        std::wcout << L"Adding hit group for instance " << i << std::endl;
        m_sbtHelper.AddHitGroup(L"HitGroup", {});
        m_sbtHelper.AddHitGroup(L"ShadowHitGroup", {});
    }

    // Adding final ShadowHitGroup
    std::wcout << L"Adding ShadowHitGroup..." << std::endl;
    m_sbtHelper.AddHitGroup(L"ShadowHitGroup", {});

    // --- NEW BLOCK: Add Callable Programs ---
    std::wcout << L"Adding callable programs..." << std::endl;
    for (const auto& name : m_callableShaderNames)
    {
        // Bind the global heap pointer just like RayGen
        m_sbtHelper.AddCallableProgram(name, { heapPointer });
    }

    // Compute the size of the SBT
    std::wcout << L"Computing SBT size..." << std::endl;
    uint32_t sbtSize = m_sbtHelper.ComputeSBTSize();
    std::wcout << L"SBT size: " << sbtSize << L" bytes." << std::endl;

    // Create the SBT on the upload heap
    std::wcout << L"Creating shader binding table buffer..." << std::endl;
    m_sbtStorage = nv_helpers_dx12::CreateBuffer(
            m_device.Get(), sbtSize, D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_GENERIC_READ, nv_helpers_dx12::kUploadHeapProps);

    // Check if SBT buffer creation was successful
    if (!m_sbtStorage) {
        std::wcout << L"Could not allocate the shader binding table!" << std::endl;
        throw std::logic_error("Could not allocate the shader binding table");
    }

    // Compile the SBT
    std::wcout << L"Generating the Shader Binding Table..." << std::endl;
    m_sbtHelper.Generate(m_sbtStorage.Get(), m_rtStateObjectProps.Get());

    // SBT creation completed
    std::wcout << L"Shader Binding Table created successfully." << std::endl;
}


//----------------------------------------------------------------------------------
//
// The camera buffer is a constant buffer that stores the transform matrices of
// the camera, for use by both the rasterization and raytracing. This method
// allocates the buffer where the matrices will be copied. For the sake of code
// clarity, it also creates a heap containing only this buffer, to use in the
// rasterization path.
//
// #DXR Extra: Perspective Camera
void Renderer::CreateCameraBuffer() {
    // Calculate the buffer size
    // 6 matrices + 1 float time + 8 planes of type XMFLOAT4
    uint32_t nbMatrix   = 6;                 // view, proj, viewInv, projInv, prevView, prevProj
    m_cameraBufferSize  = nbMatrix * sizeof(XMMATRIX)
                        + sizeof(float) * 4;     // for time
    // Round up to 256 for constant‐buffer alignment
    m_cameraBufferSize = (m_cameraBufferSize + 255) & ~255;


    // Debug output: Display the calculated buffer size
    std::wcout << L"Camera buffer size (in bytes): " << m_cameraBufferSize << std::endl;

    // Create the constant buffer for all matrices and additional parameters
    m_cameraBuffer = nv_helpers_dx12::CreateBuffer(
            m_device.Get(), m_cameraBufferSize, D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_GENERIC_READ, nv_helpers_dx12::kUploadHeapProps);

    // Debug output: Check if the buffer was created successfully
    if (m_cameraBuffer)
        std::wcout << L"Camera buffer created successfully." << std::endl;
    else
        std::wcout << L"Failed to create camera buffer!" << std::endl;

    // Create the descriptor heap
    m_constHeap = nv_helpers_dx12::CreateDescriptorHeap(
            m_device.Get(), 2, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, true);

    // Debug output: Check if the descriptor heap was created successfully
    if (m_constHeap)
        std::wcout << L"Descriptor heap created successfully." << std::endl;
    else
        std::wcout << L"Failed to create descriptor heap!" << std::endl;

    // Describe and create the constant buffer view
    D3D12_CONSTANT_BUFFER_VIEW_DESC cbvDesc = {};
    cbvDesc.BufferLocation = m_cameraBuffer->GetGPUVirtualAddress();
    cbvDesc.SizeInBytes = m_cameraBufferSize;

    // Debug output: Display buffer location and size
    std::wcout << L"Buffer location (GPU virtual address): " << cbvDesc.BufferLocation << std::endl;
    std::wcout << L"Constant buffer view size (in bytes): " << cbvDesc.SizeInBytes << std::endl;
    std::wcout << L"________________________________________________" << std::endl;


}


// #DXR Extra: Perspective Camera
//--------------------------------------------------------------------------------
// Create and copies the viewmodel and perspective matrices of the camera
//
void Renderer::UpdateCameraBuffer() {
    std::vector<XMMATRIX> matrices(6); // view, projection, viewInv, projectionInv, prevView, prevProjection

    // Initialize the current view matrix
    const glm::mat4 &viewMat = nv_helpers_dx12::CameraManip.getMatrix();
    memcpy(&matrices[0].r->m128_f32[0], glm::value_ptr(viewMat), 16 * sizeof(float));

    // Current projection matrix (UNJITTERED - exactly as DLSS wants it)
    float fovAngleY = 60.0f * XM_PI / 180.0f;
    matrices[1] = XMMatrixPerspectiveFovRH(fovAngleY, m_aspectRatio, 0.00001f, 10000.0f);

    // Inverse matrices
    XMVECTOR det;
    matrices[2] = XMMatrixInverse(&det, matrices[0]);  // viewInv
    matrices[3] = XMMatrixInverse(&det, matrices[1]);  // projectionInv

    // Previous frame matrices (For shaders)
    matrices[4] = m_prevViewMatrix;
    matrices[5] = m_prevProjMatrix;

    // --- COMPUTE PIXEL-SCALE JITTER ---
    m_jitterFrameIndex++;
    m_jitterX = Halton(m_jitterFrameIndex % 16 + 1, 2) - 0.5f;
    m_jitterY = Halton(m_jitterFrameIndex % 16 + 1, 3) - 0.5f;

    // Copy matrix contents to the buffer
    uint8_t *pData;
    HRESULT hr = m_cameraBuffer->Map(0, nullptr, (void **)&pData);
    if (FAILED(hr)) {
        std::wcerr << L"Failed to map camera buffer!" << std::endl;
        return;
    }

    // Copy the 6 matrices
    memcpy(pData, matrices.data(), 6 * sizeof(XMMATRIX));

    // Add the current system time and jitter as a float4
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    auto nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
    float currentTime = static_cast<uint32_t>(nanos);

    float extraData[4] = { currentTime, m_jitterX, m_jitterY, 0.0f };
    memcpy(pData + (6 * sizeof(XMMATRIX)), extraData, sizeof(extraData));

    m_cameraBuffer->Unmap(0, nullptr);

    // Save the current matrices for use in the next frame (for shaders)
    m_prevViewMatrix = matrices[0];
    m_prevProjMatrix = matrices[1];
}


void Renderer::ExtractFrustumPlanes(const XMMATRIX& viewProjMatrix, XMFLOAT4* planes) {
    // Extract the rows of the view-projection matrix
    XMVECTOR row1 = viewProjMatrix.r[0]; // First row
    XMVECTOR row2 = viewProjMatrix.r[1]; // Second row
    XMVECTOR row3 = viewProjMatrix.r[2]; // Third row
    XMVECTOR row4 = viewProjMatrix.r[3]; // Fourth row

    // Left plane (row4 + row1)
    planes[0] = XMFLOAT4(
            XMVectorGetX(row4) + XMVectorGetX(row1),
            XMVectorGetY(row4) + XMVectorGetY(row1),
            XMVectorGetZ(row4) + XMVectorGetZ(row1),
            XMVectorGetW(row4) + XMVectorGetW(row1) // Correct calculation of D
    );

    // Right plane (row4 - row1)
    planes[1] = XMFLOAT4(
            XMVectorGetX(row4) - XMVectorGetX(row1),
            XMVectorGetY(row4) - XMVectorGetY(row1),
            XMVectorGetZ(row4) - XMVectorGetZ(row1),
            XMVectorGetW(row4) - XMVectorGetW(row1)
    );

    // Top plane (row4 + row2)
    planes[2] = XMFLOAT4(
            XMVectorGetX(row4) + XMVectorGetX(row2),
            XMVectorGetY(row4) + XMVectorGetY(row2),
            XMVectorGetZ(row4) + XMVectorGetZ(row2),
            XMVectorGetW(row4) + XMVectorGetW(row2)
    );

    // Bottom plane (row4 - row2)
    planes[3] = XMFLOAT4(
            XMVectorGetX(row4) - XMVectorGetX(row2),
            XMVectorGetY(row4) - XMVectorGetY(row2),
            XMVectorGetZ(row4) - XMVectorGetZ(row2),
            XMVectorGetW(row4) - XMVectorGetW(row2)
    );

    // Normalize the planes
    for (int i = 0; i < 4; i++) {
        XMVECTOR plane = XMLoadFloat4(&planes[i]);
        XMVECTOR planeNormal = XMVectorSet(XMVectorGetX(plane), XMVectorGetY(plane), XMVectorGetZ(plane), 0.0f);
        float length = XMVectorGetX(XMVector3Length(planeNormal));
        if (length != 0.0f) {
            planes[i].x /= length;
            planes[i].y /= length;
            planes[i].z /= length;
            planes[i].w /= length;
        }
    }
}






//--------------------------------------------------------------------------------------------------
//
//
void Renderer::OnButtonDown(UINT32 lParam) {
  nv_helpers_dx12::CameraManip.setMousePosition(-GET_X_LPARAM(lParam),
                                                -GET_Y_LPARAM(lParam));
}

//--------------------------------------------------------------------------------------------------
//
//
void Renderer::OnMouseMove(UINT8 wParam, UINT32 lParam) {
  using nv_helpers_dx12::Manipulator;
  Manipulator::Inputs inputs;
  inputs.lmb = wParam & MK_LBUTTON;
  inputs.mmb = wParam & MK_MBUTTON;
  inputs.rmb = wParam & MK_RBUTTON;
  if (!inputs.lmb && !inputs.rmb && !inputs.mmb)
    return; // no mouse button pressed

  inputs.ctrl = GetAsyncKeyState(VK_CONTROL);
  inputs.shift = GetAsyncKeyState(VK_SHIFT);
  inputs.alt = GetAsyncKeyState(VK_MENU);

  CameraManip.mouseMove(-GET_X_LPARAM(lParam), -GET_Y_LPARAM(lParam), inputs);
}

//-----------------------------------------------------------------------------
//
// #DXR Extra: Per-Instance Data
void Renderer::CreateGlobalConstantBuffer() {
  // Due to HLSL packing rules, we create the CB with 9 float4 (each needs to
  // start on a 16-byte boundary)
  XMVECTOR bufferData[] = {
      // A
      XMVECTOR{1.0f, 0.0f, 0.0f, 1.0f},
      XMVECTOR{0.7f, 0.4f, 0.0f, 1.0f},
      XMVECTOR{0.4f, 0.7f, 0.0f, 1.0f},

      // B
      XMVECTOR{0.0f, 1.0f, 0.0f, 1.0f},
      XMVECTOR{0.0f, 0.7f, 0.4f, 1.0f},
      XMVECTOR{0.0f, 0.4f, 0.7f, 1.0f},

      // C
      XMVECTOR{0.0f, 0.0f, 1.0f, 1.0f},
      XMVECTOR{0.4f, 0.0f, 0.7f, 1.0f},
      XMVECTOR{0.7f, 0.0f, 0.4f, 1.0f},
  };

  // Create our buffer
  m_globalConstantBuffer = nv_helpers_dx12::CreateBuffer(
      m_device.Get(), sizeof(bufferData), D3D12_RESOURCE_FLAG_NONE,
      D3D12_RESOURCE_STATE_GENERIC_READ, nv_helpers_dx12::kUploadHeapProps);

  // Copy CPU memory to GPU
  uint8_t *pData;
  ThrowIfFailed(m_globalConstantBuffer->Map(0, nullptr, (void **)&pData));
  memcpy(pData, bufferData, sizeof(bufferData));
  m_globalConstantBuffer->Unmap(0, nullptr);
}

//-----------------------------------------------------------------------------
//
// #DXR Extra: Per-Instance Data
void Renderer::CreatePerInstanceConstantBuffers() {
  // Due to HLSL packing rules, we create the CB with 9 float4 (each needs to
  // start on a 16-byte boundary)
  XMVECTOR bufferData[] = {
      // A
      XMVECTOR{1.0f, 0.0f, 0.0f, 1.0f},
      XMVECTOR{1.0f, 0.4f, 0.0f, 1.0f},
      XMVECTOR{1.f, 0.7f, 0.0f, 1.0f},

      // B
      XMVECTOR{0.0f, 1.0f, 0.0f, 1.0f},
      XMVECTOR{0.0f, 1.0f, 0.4f, 1.0f},
      XMVECTOR{0.0f, 1.0f, 0.7f, 1.0f},

      // C
      XMVECTOR{0.0f, 0.0f, 1.0f, 1.0f},
      XMVECTOR{0.4f, 0.0f, 1.0f, 1.0f},
      XMVECTOR{0.7f, 0.0f, 1.0f, 1.0f},
  };

  m_perInstanceConstantBuffers.resize(3);
  int i(0);
  for (auto &cb : m_perInstanceConstantBuffers) {
    const uint32_t bufferSize = sizeof(XMVECTOR) * 3;
    cb = nv_helpers_dx12::CreateBuffer(
        m_device.Get(), bufferSize, D3D12_RESOURCE_FLAG_NONE,
        D3D12_RESOURCE_STATE_GENERIC_READ, nv_helpers_dx12::kUploadHeapProps);
    uint8_t *pData;
    ThrowIfFailed(cb->Map(0, nullptr, (void **)&pData));
    memcpy(pData, &bufferData[i * 3], bufferSize);
    cb->Unmap(0, nullptr);
    ++i;
  }
}

//-----------------------------------------------------------------------------
//
// Create the depth buffer for rasterization. This buffer needs to be kept in a
// separate heap
//
// #DXR Extra: Depth Buffering
void Renderer::CreateDepthBuffer() {

  // The depth buffer heap type is specific for that usage, and the heap
  // contents are not visible from the shaders
  m_dsvHeap = nv_helpers_dx12::CreateDescriptorHeap(
      m_device.Get(), 1, D3D12_DESCRIPTOR_HEAP_TYPE_DSV, false);

  // The depth and stencil can be packed into a single 32-bit texture buffer.
  // Since we do not need stencil, we use the 32 bits to store depth information
  // (DXGI_FORMAT_D32_FLOAT).
  D3D12_HEAP_PROPERTIES depthHeapProperties =
      CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);

  D3D12_RESOURCE_DESC depthResourceDesc = CD3DX12_RESOURCE_DESC::Tex2D(
      DXGI_FORMAT_D32_FLOAT, m_width, m_height, 1, 1);
  depthResourceDesc.Flags |= D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;

  // The depth values will be initialized to 1
  CD3DX12_CLEAR_VALUE depthOptimizedClearValue(DXGI_FORMAT_D32_FLOAT, 1.0f, 0);

  // Allocate the buffer itself, with a state allowing depth writes
  ThrowIfFailed(m_device->CreateCommittedResource(
      &depthHeapProperties, D3D12_HEAP_FLAG_NONE, &depthResourceDesc,
      D3D12_RESOURCE_STATE_DEPTH_WRITE, &depthOptimizedClearValue,
      IID_PPV_ARGS(&m_depthStencil)));

  // Write the depth buffer view into the depth buffer heap
  D3D12_DEPTH_STENCIL_VIEW_DESC dsvDesc = {};
  dsvDesc.Format = DXGI_FORMAT_D32_FLOAT;
  dsvDesc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
  dsvDesc.Flags = D3D12_DSV_FLAG_NONE;

  m_device->CreateDepthStencilView(
      m_depthStencil.Get(), &dsvDesc,
      m_dsvHeap->GetCPUDescriptorHandleForHeapStart());
}

void Renderer::CreateTextureArrays(
    const std::vector<TextureData>& albedoTextures,
    const std::vector<TextureData>& normalTextures,
    const std::vector<TextureData>& rmaTextures)
{
    // Hilfs-Lambda zum Erstellen eines Textur-Arrays (jetzt für Mipmaps und Komprimierung aktualisiert)
    auto createArray = [&](
        const std::vector<TextureData>& textures,
        ComPtr<ID3D12Resource>& textureArrayResource,
        const std::wstring& resourceName)
    {
        if (textures.empty()) return;

        // Schritt 1: Metadaten aus dem ersten ScratchImage auslesen (Breite, Höhe, Format, Mip-Levels)
        const DirectX::TexMetadata& metadata = textures[0].image.GetMetadata();
        UINT width = static_cast<UINT>(metadata.width);
        UINT height = static_cast<UINT>(metadata.height);
        UINT arraySize = static_cast<UINT>(textures.size());
        UINT mipLevels = static_cast<UINT>(metadata.mipLevels);
        DXGI_FORMAT format = metadata.format;

        // (Optional) Konsistenzprüfung: Sicherstellen, dass alle Texturen die gleichen Dimensionen/Formate haben
        for (const auto& tex : textures) {
            const auto& currentMeta = tex.image.GetMetadata();
            if (currentMeta.width != width || currentMeta.height != height || currentMeta.format != format || currentMeta.mipLevels != mipLevels) {
                throw std::runtime_error("All textures in an array must have the same dimensions, format, and mip level count. Failed for array: " + std::string(resourceName.begin(), resourceName.end()));
            }
        }

        // Schritt 2: Die Texture2D-Array-Ressource auf der GPU erstellen
        D3D12_RESOURCE_DESC texDesc = {};
        texDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        texDesc.Alignment = 0;
        texDesc.Width = width;
        texDesc.Height = height;
        texDesc.DepthOrArraySize = arraySize;
        texDesc.MipLevels = mipLevels; // WICHTIG: Verwende Mip-Levels aus den Metadaten
        texDesc.Format = format;       // WICHTIG: Verwende das komprimierte Format aus den Metadaten
        texDesc.SampleDesc.Count = 1;
        texDesc.SampleDesc.Quality = 0;
        texDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
        texDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

        ThrowIfFailed(m_device->CreateCommittedResource(
            &nv_helpers_dx12::kDefaultHeapProps,
            D3D12_HEAP_FLAG_NONE,
            &texDesc,
            D3D12_RESOURCE_STATE_COPY_DEST,
            nullptr,
            IID_PPV_ARGS(&textureArrayResource)));
        textureArrayResource->SetName(resourceName.c_str());

        // Schritt 3: Subressourcen-Daten für JEDEN Mip-Level JEDER Textur vorbereiten
        std::vector<D3D12_SUBRESOURCE_DATA> subresources;
        subresources.reserve(arraySize * mipLevels);

        for (const auto& texData : textures)
        {
            // Durchlaufe jedes Bild (Mip-Level) im ScratchImage
            for (size_t i = 0; i < texData.image.GetImageCount(); ++i)
            {
                const DirectX::Image* img = texData.image.GetImage(i, 0, 0);
                if (!img)
                {
                    throw std::runtime_error("Failed to get image from ScratchImage.");
                }

                D3D12_SUBRESOURCE_DATA subresourceData = {};
                subresourceData.pData = img->pixels;
                subresourceData.RowPitch = static_cast<LONG_PTR>(img->rowPitch);
                subresourceData.SlicePitch = static_cast<LONG_PTR>(img->slicePitch);

                subresources.push_back(subresourceData);
            }
        }

        // Schritt 4: Upload-Heap erstellen und Daten hochladen
        const UINT64 uploadBufferSize = GetRequiredIntermediateSize(textureArrayResource.Get(), 0, static_cast<UINT>(subresources.size()));

        m_textureUploadHeaps.emplace_back();
        ComPtr<ID3D12Resource>& textureUploadHeap = m_textureUploadHeaps.back();

        CD3DX12_RESOURCE_DESC uploadBufferDesc = CD3DX12_RESOURCE_DESC::Buffer(uploadBufferSize);
        ThrowIfFailed(m_device->CreateCommittedResource(
            &nv_helpers_dx12::kUploadHeapProps,
            D3D12_HEAP_FLAG_NONE,
            &uploadBufferDesc,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr,
            IID_PPV_ARGS(&textureUploadHeap)));
        textureUploadHeap->SetName((resourceName + L" Upload Heap").c_str());

        UpdateSubresources(m_commandList.Get(), textureArrayResource.Get(), textureUploadHeap.Get(), 0, 0, static_cast<UINT>(subresources.size()), subresources.data());

        // Schritt 5: Ressource für die Shader-Nutzung bereit machen
        CD3DX12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::Transition(
            textureArrayResource.Get(),
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        m_commandList->ResourceBarrier(1, &barrier);
    };

    // Rufe das Lambda für jeden Texturtyp auf. Die Format-Argumente werden nicht mehr benötigt.
    createArray(albedoTextures, m_albedoTextureArray, L"AlbedoTextureArray");
    createArray(normalTextures, m_normalTextureArray, L"NormalTextureArray");
    createArray(rmaTextures, m_rmaTextureArray, L"RmaTextureArray");
}

//--------------------------------------------------------------------------------------------------
// Allocate memory to hold per-instance information
// #DXR Extra - Refitting
void Renderer::CreateInstancePropertiesBuffer() {
  uint32_t bufferSize = ROUND_UP(
      static_cast<uint32_t>(m_instances.size()) * sizeof(InstanceProperties),
      D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);

  // Create the constant buffer for all matrices
  m_instanceProperties = nv_helpers_dx12::CreateBuffer(
      m_device.Get(), bufferSize, D3D12_RESOURCE_FLAG_NONE,
      D3D12_RESOURCE_STATE_GENERIC_READ, nv_helpers_dx12::kUploadHeapProps);
}

// Copy the per‑instance data into the buffer
void Renderer::UpdateInstancePropertiesBuffer()
{
    InstanceProperties* dst = nullptr;
    CD3DX12_RANGE  r(0,0);                         // we never read from CPU
    ThrowIfFailed(
        m_instanceProperties->Map(0,&r,
                                  reinterpret_cast<void**>(&dst)));

    for (size_t inst = 0; inst < m_instances.size(); ++inst, ++dst)
    {
        // existing matrix bookkeeping
        const XMMATRIX& M = m_instances[inst].second;
        XMVECTOR det;

        dst->prevObjectToWorld        = dst->objectToWorld;
        dst->prevObjectToWorldInverse = XMMatrixInverse(&det, dst->objectToWorld);

        dst->objectToWorld            = M;
        dst->objectToWorldInverse     = XMMatrixInverse(&det, M);

        XMMATRIX upper3x3             = M;
        upper3x3.r[0].m128_f32[3] = upper3x3.r[1].m128_f32[3] =
        upper3x3.r[2].m128_f32[3] = upper3x3.r[3].m128_f32[0] =
        upper3x3.r[3].m128_f32[1] = upper3x3.r[3].m128_f32[2] = 0.f;
        upper3x3.r[3].m128_f32[3] = 1.f;

        dst->prevObjectToWorldNormal  = dst->objectToWorldNormal;
        dst->objectToWorldNormal      = XMMatrixTranspose(
                                            XMMatrixInverse(&det, upper3x3));

        // per‑geometry offsets
        UINT   modelIdx = m_instanceModelIndices[inst];   // which mesh?
        const GeometryOffsets& go = m_geoOffsets[modelIdx];

        dst->indexBase    = go.indexBase;     // first index of this mesh
        dst->vertexBase   = go.vertexBase;    // first vertex of this mesh
        dst->materialBase = go.materialBase;  // first mat‑ID of this mesh
        dst->triToLightBase = m_instTriOffset[inst];
    }
    m_instanceProperties->Unmap(0,nullptr);
}

void Renderer::CollectEmissiveTriangles() {
    m_emissiveTriangles.clear();

    // allocate tri->light map for all instances
    m_instTriOffset.resize(m_instances.size());
    size_t totalTris = 0;
    for (size_t inst = 0; inst < m_instances.size(); ++inst) {
        UINT modelIndex = m_instanceModelIndices[inst];
        totalTris += m_IndexCount[modelIndex] / 3;
    }
    m_triToLightId.assign(totalTris, 0xFFFFFFFFu);  // not emissive

    // compute per-instance base offsets
    uint32_t runningBase = 0;
    for (size_t inst = 0; inst < m_instances.size(); ++inst) {
        m_instTriOffset[inst] = runningBase;
        UINT modelIndex = m_instanceModelIndices[inst];
        runningBase += m_IndexCount[modelIndex] / 3;
    }

    for (size_t instanceIndex = 0; instanceIndex < m_instances.size(); ++instanceIndex) {
        UINT modelIndex = m_instanceModelIndices[instanceIndex];
        UINT e_materialIDOffset = m_materialIDOffsets[modelIndex];
        UINT triangleCount = m_IndexCount[modelIndex] / 3;

        Vertex* vertices = nullptr;
        UINT* indices = nullptr;
        CD3DX12_RANGE readRange(0, 0);

        ThrowIfFailed(m_VB[modelIndex]->Map(0, &readRange, reinterpret_cast<void**>(&vertices)));
        ThrowIfFailed(m_IB[modelIndex]->Map(0, &readRange, reinterpret_cast<void**>(&indices)));

        const uint32_t triBase = m_instTriOffset[instanceIndex];

        for (UINT t = 0; t < triangleCount; ++t) {
            UINT idx0 = indices[3 * t + 0];
            UINT idx1 = indices[3 * t + 1];
            UINT idx2 = indices[3 * t + 2];

            // material id per tri
            UINT mid = m_materialIDs[e_materialIDOffset + t];
            const Material& mat = m_materials[mid];
            const float emissive = mat.Ke.x + mat.Ke.y + mat.Ke.z;

            if (emissive > 0.0f) {
                // Push light triangle
                const Vertex& v0 = vertices[idx0];
                const Vertex& v1 = vertices[idx1];
                const Vertex& v2 = vertices[idx2];

                LightTriangle lt{};
                lt.x = v0.position; lt.y = v1.position; lt.z = v2.position;
                lt.instanceID = static_cast<UINT>(instanceIndex);
                const XMMATRIX& M = m_instances[instanceIndex].second;
                lt.weight     = ComputeTriangleWeight(v0.position, v1.position, v2.position, mat.Ke, M);
                lt.emission   = mat.Ke;
                // lt.primID     = t;

                const uint32_t lightIdx = static_cast<uint32_t>(m_emissiveTriangles.size());
                m_emissiveTriangles.push_back(lt);
                m_triToLightId[triBase + t] = lightIdx;
            }
        }

        m_VB[modelIndex]->Unmap(0, nullptr);
        m_IB[modelIndex]->Unmap(0, nullptr);
    }

    // Sort the emissive triangles based on weight in descending order
    /*std::sort(m_emissiveTriangles.begin(), m_emissiveTriangles.end(),
              [](const LightTriangle& a, const LightTriangle& b) {
                  return a.weight > b.weight;
              });*/

    // Calculate the total weight
    /*float totalWeight = 0.0f;
    for (const auto& triangle : m_emissiveTriangles) {
        totalWeight += triangle.weight;
    }*/

    // Calculate relative weights and cumulative distribution function (CDF)
    /*float cumulativeWeight = 0.0f;
    for (auto& triangle : m_emissiveTriangles) {
        triangle.weight /= totalWeight; // Normalize weight
        cumulativeWeight += triangle.weight;
        triangle.cdf = cumulativeWeight;
        triangle.totalWeight = totalWeight;
    }

    // Ensure the last CDF value is exactly 1.0f
    if (!m_emissiveTriangles.empty()) {
        m_emissiveTriangles.back().cdf = 1.0f;
    }*/

    std::wcout << L"Emissive Triangles: " << m_emissiveTriangles.size() << std::endl;
}



/*float Renderer::ComputeTriangleWeight(const XMFLOAT3& v0, const XMFLOAT3& v1, const XMFLOAT3& v2, const XMFLOAT3& emissiveColor) {
    // Compute the area of the triangle
    XMVECTOR p0 = XMLoadFloat3(&v0);
    XMVECTOR p1 = XMLoadFloat3(&v1);
    XMVECTOR p2 = XMLoadFloat3(&v2);

    XMVECTOR edge1 = XMVectorSubtract(p1, p0);
    XMVECTOR edge2 = XMVectorSubtract(p2, p0);
    XMVECTOR crossProduct = XMVector3Cross(edge1, edge2);
    float area = (std::fmax)(0.5f * XMVectorGetX(XMVector3Length(crossProduct)),0.001f);

    // Compute the average emissive intensity
    float emissiveIntensity = (emissiveColor.x + emissiveColor.y + emissiveColor.z) / 3.0f;

    // The weight is proportional to area and emissive intensity
    return area * emissiveIntensity;
}*/

static inline float Luminance(const XMFLOAT3& c) {
    return 0.2126f*c.x + 0.7152f*c.y + 0.0722f*c.z;
}

float Renderer::ComputeTriangleWeight(const XMFLOAT3& v0,
                                           const XMFLOAT3& v1,
                                           const XMFLOAT3& v2,
                                           const XMFLOAT3& emissiveColor,
                                           const XMMATRIX& M)
{
    // transform verts to WORLD
    XMVECTOR p0 = XMVector3TransformCoord(XMLoadFloat3(&v0), M);
    XMVECTOR p1 = XMVector3TransformCoord(XMLoadFloat3(&v1), M);
    XMVECTOR p2 = XMVector3TransformCoord(XMLoadFloat3(&v2), M);

    // world-space geometric area
    XMVECTOR e1 = p1 - p0, e2 = p2 - p0;
    float area = 0.5f * XMVectorGetX( XMVector3Length( XMVector3Cross(e1,e2) ) );

    float lum = Luminance(emissiveColor);
    return std::max(area, 1e-10f) * lum;
}




void Renderer::CreateEmissiveTrianglesBuffer() {
    size_t bufferSize = m_emissiveTriangles.size() * sizeof(LightTriangle);

    // Ensure triCount is set
    for (auto& m_emissiveTriangle : m_emissiveTriangles) {
        m_emissiveTriangle.triCount = static_cast<UINT>(m_emissiveTriangles.size());
    }

    // Create an upload buffer
    ComPtr<ID3D12Resource> emissiveTrianglesUploadBuffer = nv_helpers_dx12::CreateBuffer(
            m_device.Get(), static_cast<UINT>(bufferSize), D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_GENERIC_READ, nv_helpers_dx12::kUploadHeapProps);

    // Copy data to the upload buffer
    {
        LightTriangle* pData = nullptr;
        CD3DX12_RANGE readRange(0, 0);
        ThrowIfFailed(emissiveTrianglesUploadBuffer->Map(0, &readRange, reinterpret_cast<void**>(&pData)));
        memcpy(pData, m_emissiveTriangles.data(), bufferSize);
        emissiveTrianglesUploadBuffer->Unmap(0, nullptr);
    }

    // Create the default heap buffer
    m_emissiveTrianglesBuffer = nv_helpers_dx12::CreateBuffer(
            m_device.Get(), static_cast<UINT>(bufferSize), D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_COPY_DEST, nv_helpers_dx12::kDefaultHeapProps);

    // Copy data from upload buffer to default heap buffer
    m_commandList->CopyBufferRegion(m_emissiveTrianglesBuffer.Get(), 0, emissiveTrianglesUploadBuffer.Get(), 0, bufferSize);

    // Transition the buffer to GENERIC_READ for shader access
    CD3DX12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::Transition(
            m_emissiveTrianglesBuffer.Get(),
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_GENERIC_READ);
    m_commandList->ResourceBarrier(1, &barrier);

    ThrowIfFailed(m_commandList->Close());

    ID3D12CommandList* ppCommandLists[] = { m_commandList.Get() };
    m_commandQueue->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);

    // Fence-sync
    WaitForPreviousFrame();

    auto* allocator = m_commandAllocators[m_frameIndex].Get();   // pick the right one
    ThrowIfFailed(allocator->Reset());
    ThrowIfFailed(m_commandList->Reset(allocator, nullptr));
}

// Small helper that builds two SoA arrays (prob + alias index)
void Renderer::BuildAliasTableSoA(const std::vector<LightTriangle>& tris)
{
    const uint32_t N = static_cast<uint32_t>(tris.size());
    std::vector<float>      scaled(N);
    std::vector<uint32_t>   _small, _large;

    m_aliasProb.resize(N);
    m_aliasIdx .resize(N);

    // scale weights
    for (uint32_t i=0;i<N;++i) scaled[i] = tris[i].weight * N;
    for (uint32_t i=0;i<N;++i) (scaled[i] < 1.f ? _small:_large).push_back(i);

    // alias algorithm
    while (!_small.empty() && !_large.empty()) {
        uint32_t s = _small.back(); _small.pop_back();
        uint32_t l = _large.back(); _large.pop_back();
        m_aliasProb[s] = scaled[s];
        m_aliasIdx [s] = l;
        scaled[l] = (scaled[l]+scaled[s]) - 1.f;
        (scaled[l] < 1.f ? _small:_large).push_back(l);
    }
    for (uint32_t i : _large) { m_aliasProb[i] = 1.f; m_aliasIdx[i] = i; }
    for (uint32_t i : _small) { m_aliasProb[i] = 1.f; m_aliasIdx[i] = i; }
}

void Renderer::CreateAliasBuffers()
{
    if (m_aliasProb.empty()) return;

    UINT N          = static_cast<UINT>(m_aliasProb.size());
    UINT probBytes  = N*sizeof(float);
    UINT idxBytes   = N*sizeof(uint32_t);

    auto makeDefault = [&](const void* src, UINT bytes,
                           ComPtr<ID3D12Resource>& dst)
    {
        ComPtr<ID3D12Resource> upload =
            nv_helpers_dx12::CreateBuffer(m_device.Get(), bytes, D3D12_RESOURCE_FLAG_NONE,
                  D3D12_RESOURCE_STATE_GENERIC_READ,
                  nv_helpers_dx12::kUploadHeapProps);
        void* p; CD3DX12_RANGE r(0,0); upload->Map(0,&r,&p);
        memcpy(p,src,bytes); upload->Unmap(0,nullptr);

        dst = nv_helpers_dx12::CreateBuffer(m_device.Get(), bytes, D3D12_RESOURCE_FLAG_NONE,
                  D3D12_RESOURCE_STATE_COPY_DEST,
                  nv_helpers_dx12::kDefaultHeapProps);

        m_commandList->CopyBufferRegion(dst.Get(),0, upload.Get(),0, bytes);
        CD3DX12_RESOURCE_BARRIER br = CD3DX12_RESOURCE_BARRIER::Transition(
            dst.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_GENERIC_READ);
        m_commandList->ResourceBarrier(1,&br);
    };

    makeDefault(m_aliasProb.data(), probBytes, m_aliasProbBuffer);
    makeDefault(m_aliasIdx .data(), idxBytes , m_aliasIdxBuffer );
}

void Renderer::GenerateLutTextures()
{
    SCOPE_TIMER("GenerateLutTextures");
    std::wcout << L"Starting generation of LUT Texture Array (" << NUM_LUTS << " slices)..." << std::endl;

    // Create a vector of vectors to hold the data for each LUT slice
    std::vector<std::vector<float>> allLutData(NUM_LUTS, std::vector<float>(LUT_RESOLUTION * LUT_RESOLUTION));

    const XMFLOAT3 N = {0.0f, 0.0f, 1.0f};
    Material tempMat;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (int y = 0; y < LUT_RESOLUTION; ++y) {
        float cosTheta = static_cast<float>(y) / (LUT_RESOLUTION - 1);
        cosTheta = std::max(0.01f, cosTheta);
        float sinTheta = sqrt(1.0f - cosTheta * cosTheta);
        const XMFLOAT3 V = {sinTheta, 0.0f, cosTheta};

        for (int x = 0; x < LUT_RESOLUTION; ++x) {
            float roughness = static_cast<float>(x) / (LUT_RESOLUTION - 1);
            roughness = std::max(0.01f, roughness);
            tempMat.Pr_Pm_Ps_Pc.x = roughness;

            size_t pixelIndex = (size_t)y * LUT_RESOLUTION + x;

            // --- Calculate Sheen Albedo and store in slice 0 ---
            float sheenAlbedo = ComputeSheenDirectionalAlbedo(N, V, roughness, NUM_SAMPLES_LUT);
            allLutData[0][pixelIndex] = sheenAlbedo;

            // --- Calculate E(ss) for GGX and store in slice 1 ---
            float Ess = 0.0f;
            for (int i = 0; i < NUM_SAMPLES_LUT; ++i) {
                XMFLOAT3 L;
                SampleGGX(tempMat, V, N, L, dist(gen), dist(gen));
                if (dot(N, L) <= 0.0f) continue;
                XMFLOAT3 brdf3 = EvaluateBRDF_GGX(V, L, N, {}, roughness);
                float pdf = BRDF_PDF_GGX(roughness, N, L * -1.0f, V);
                if (pdf > 1e-6f) Ess += (brdf3.x * dot(N, L)) / pdf;
            }
            allLutData[1][pixelIndex] = (NUM_SAMPLES_LUT > 0) ? (Ess / NUM_SAMPLES_LUT) : 0.0f;
        }
        std::wcout << L"  - LUT Generation Progress: " << ((y + 1) * 100 / LUT_RESOLUTION) << L"%\r";
    }

    // ====================================================================
    PrintFullLutMatrix(allLutData[0], L"Sheen Directional Albedo LUT (Slice 0)");
    PrintFullLutMatrix(allLutData[1], L"GGX E(ss) LUT (Slice 1)");
    // ====================================================================

    std::wcout << std::endl << L"LUT data generation complete. Uploading to GPU..." << std::endl;

    CreateAndUploadLutArray(allLutData, m_lutTextureArray, L"LutTextureArray");
}

void Renderer::CreateAndUploadLutArray(const std::vector<std::vector<float>>& allLutData, ComPtr<ID3D12Resource>& textureArrayResource, const std::wstring& resourceName)
{
    if (allLutData.empty()) return;

    UINT arraySize = static_cast<UINT>(allLutData.size());

    D3D12_RESOURCE_DESC texDesc = {};
    texDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    texDesc.Alignment = 0;
    texDesc.Width = LUT_RESOLUTION;
    texDesc.Height = LUT_RESOLUTION;
    texDesc.DepthOrArraySize = arraySize; // The key change is here
    texDesc.MipLevels = 1;
    texDesc.Format = DXGI_FORMAT_R32_FLOAT;
    texDesc.SampleDesc.Count = 1;
    texDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;

    ThrowIfFailed(m_device->CreateCommittedResource(
        &nv_helpers_dx12::kDefaultHeapProps, D3D12_HEAP_FLAG_NONE, &texDesc,
        D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&textureArrayResource)));
    textureArrayResource->SetName(resourceName.c_str());

    const UINT64 uploadBufferSize = GetRequiredIntermediateSize(textureArrayResource.Get(), 0, arraySize);

    m_lutUploadHeaps.emplace_back();
    ComPtr<ID3D12Resource>& uploadHeap = m_lutUploadHeaps.back();
    CD3DX12_RESOURCE_DESC uploadBufferDesc = CD3DX12_RESOURCE_DESC::Buffer(uploadBufferSize);

    ThrowIfFailed(m_device->CreateCommittedResource(
        &nv_helpers_dx12::kUploadHeapProps, D3D12_HEAP_FLAG_NONE, &uploadBufferDesc,
        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&uploadHeap)));

    // Prepare subresource data for each slice
    std::vector<D3D12_SUBRESOURCE_DATA> subresources(arraySize);
    for (UINT i = 0; i < arraySize; ++i) {
        subresources[i].pData = allLutData[i].data();
        subresources[i].RowPitch = LUT_RESOLUTION * sizeof(float);
        subresources[i].SlicePitch = subresources[i].RowPitch * LUT_RESOLUTION;
    }

    UpdateSubresources(m_commandList.Get(), textureArrayResource.Get(), uploadHeap.Get(), 0, 0, arraySize, subresources.data());

    CD3DX12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::Transition(
        textureArrayResource.Get(),
        D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    m_commandList->ResourceBarrier(1, &barrier);
}


void Renderer::CreatePathStateBuffer() {
    // 88 bytes per pixel as defined in the shader (BYTES_PS)
    UINT elementSize = 88;
    UINT bufferSize = GetWidth() * GetHeight() * elementSize;

    // Create the description struct first (as a local l-value)
    auto bufferDesc = CD3DX12_RESOURCE_DESC::Buffer(bufferSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

    // Create committed resource (Default Heap)
    ThrowIfFailed(m_device->CreateCommittedResource(
        &nv_helpers_dx12::kDefaultHeapProps,
        D3D12_HEAP_FLAG_NONE,
        &bufferDesc,    // Now passing the address of a named variable
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        nullptr,
        IID_PPV_ARGS(&m_pathStateBuffer)));

    m_pathStateBuffer->SetName(L"PathStateBuffer");
}

void Renderer::CreateStreamingCompactionBuffers()
{
    UINT totalPixels = GetWidth() * GetHeight();

    // 1. Create Stack Buffers
    for (int i = 0; i < MAX_STACKS; ++i) {
        auto desc = CD3DX12_RESOURCE_DESC::Buffer(
            totalPixels * sizeof(uint32_t) * 2,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
        );
        ThrowIfFailed(m_device->CreateCommittedResource(
            &nv_helpers_dx12::kDefaultHeapProps, D3D12_HEAP_FLAG_NONE, &desc,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
            IID_PPV_ARGS(&m_stackBuffers[i])));

        m_stackBuffers[i]->SetName((L"PixelStack_" + std::to_wstring(i)).c_str());
    }

    // 2. Create Global Counter Buffer
    {
        auto desc = CD3DX12_RESOURCE_DESC::Buffer(
            MAX_STACKS * sizeof(uint32_t),
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
        );
        ThrowIfFailed(m_device->CreateCommittedResource(
            &nv_helpers_dx12::kDefaultHeapProps, D3D12_HEAP_FLAG_NONE, &desc,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
            IID_PPV_ARGS(&m_globalCounterBuffer)));
        m_globalCounterBuffer->SetName(L"GlobalStackCounters");
    }

    // 3. Create Indirect Arguments Buffer
    {
        auto desc = CD3DX12_RESOURCE_DESC::Buffer(
            MAX_INDIRECT_COMMANDS * sizeof(D3D12_DISPATCH_ARGUMENTS),
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
        );
        // FIX: Start in UNORDERED_ACCESS state to match the loop's first barrier expectation
        ThrowIfFailed(m_device->CreateCommittedResource(
            &nv_helpers_dx12::kDefaultHeapProps, D3D12_HEAP_FLAG_NONE, &desc,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
            IID_PPV_ARGS(&m_indirectArgsBuffer)));
        m_indirectArgsBuffer->SetName(L"IndirectArgsBuffer");
    }

    // 4. FIX: Create and Fill Zero Buffer (Upload Heap)
    {
        auto uploadHeapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
        auto bufferDesc = CD3DX12_RESOURCE_DESC::Buffer(MAX_STACKS * sizeof(uint32_t));
        ThrowIfFailed(m_device->CreateCommittedResource(
            &uploadHeapProps,
            D3D12_HEAP_FLAG_NONE,
            &bufferDesc,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr,
            IID_PPV_ARGS(&m_zeroBuffer)));

        // Map and zero it out
        void* pData;
        m_zeroBuffer->Map(0, nullptr, &pData);
        memset(pData, 0, MAX_STACKS * sizeof(uint32_t));
        m_zeroBuffer->Unmap(0, nullptr);
    }

    // A. Sort Count Histogram (65536 * 4 bytes)
    // Needs UAV access.
    {
        auto desc = CD3DX12_RESOURCE_DESC::Buffer(
            SORT_BUCKETS * sizeof(uint32_t),
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
        );
        ThrowIfFailed(m_device->CreateCommittedResource(
            &nv_helpers_dx12::kDefaultHeapProps, D3D12_HEAP_FLAG_NONE, &desc,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
            IID_PPV_ARGS(&m_sortCountBuffer)));
        m_sortCountBuffer->SetName(L"SortCountBuffer");
    }

    // B. Sort Offset Buffer (Prefix Sum) (65536 * 4 bytes)
    {
        auto desc = CD3DX12_RESOURCE_DESC::Buffer(
            SORT_BUCKETS * sizeof(uint32_t),
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
        );
        ThrowIfFailed(m_device->CreateCommittedResource(
            &nv_helpers_dx12::kDefaultHeapProps, D3D12_HEAP_FLAG_NONE, &desc,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
            IID_PPV_ARGS(&m_sortOffsetBuffer)));
        m_sortOffsetBuffer->SetName(L"SortOffsetBuffer");
    }

    // C. Sort Bounds Buffer (Min/Max) (8 bytes)
    {
        auto desc = CD3DX12_RESOURCE_DESC::Buffer(
            8 * sizeof(uint32_t), // [0]=Min, [1]=Max
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
        );
        ThrowIfFailed(m_device->CreateCommittedResource(
            &nv_helpers_dx12::kDefaultHeapProps, D3D12_HEAP_FLAG_NONE, &desc,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
            IID_PPV_ARGS(&m_sortBoundsBuffer)));
        m_sortBoundsBuffer->SetName(L"SortBoundsBuffer");
    }

    // --- 1. Create Sort Count Buffer (Keep existing) ---
    {
        auto desc = CD3DX12_RESOURCE_DESC::Buffer(
            SORT_BUCKETS * sizeof(uint32_t),
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
        );
        ThrowIfFailed(m_device->CreateCommittedResource(
            &nv_helpers_dx12::kDefaultHeapProps, D3D12_HEAP_FLAG_NONE, &desc,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
            IID_PPV_ARGS(&m_sortCountBuffer)));
        m_sortCountBuffer->SetName(L"SortCountBuffer");
    }

    // --- 2. Create Sort Offset Buffer (Keep existing) ---
    {
        auto desc = CD3DX12_RESOURCE_DESC::Buffer(
            SORT_BUCKETS * sizeof(uint32_t),
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
        );
        ThrowIfFailed(m_device->CreateCommittedResource(
            &nv_helpers_dx12::kDefaultHeapProps, D3D12_HEAP_FLAG_NONE, &desc,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
            IID_PPV_ARGS(&m_sortOffsetBuffer)));
        m_sortOffsetBuffer->SetName(L"SortOffsetBuffer");
    }

    // --- 3. Create Sort Bounds Buffer (Keep existing) ---
    {
        auto desc = CD3DX12_RESOURCE_DESC::Buffer(
            8 * sizeof(uint32_t), // [0]=Min, [1]=Max
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
        );
        ThrowIfFailed(m_device->CreateCommittedResource(
            &nv_helpers_dx12::kDefaultHeapProps, D3D12_HEAP_FLAG_NONE, &desc,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
            IID_PPV_ARGS(&m_sortBoundsBuffer)));
        m_sortBoundsBuffer->SetName(L"SortBoundsBuffer");
    }

    // -------------------------------------------------------------
    // --- 4. NEW: Create Bounds Reset Buffer (Init Values) ---
    // -------------------------------------------------------------
    {
        // Define the "Ordered Float" initialization pattern
        const UINT MAX_U = 0xFFFFFFFF; // Maps to +FloatMax in ordered encoding
        const UINT MIN_U = 0;          // Maps to -FloatMax in ordered encoding

        // Layout:
        // 0-2: MinOrigin (X,Y,Z) -> Set to MAX so InterlockedMin works
        // 3-5: MaxOrigin (X,Y,Z) -> Set to MIN so InterlockedMax works
        // 6:   MinKey            -> Set to MAX
        // 7:   MaxKey            -> Set to MIN
        const UINT initData[8] = {
            MAX_U, MAX_U, MAX_U,
            MIN_U, MIN_U, MIN_U,
            MAX_U, MIN_U
        };

        const UINT bufferSize = sizeof(initData);

        // Create Upload Buffer
        auto heapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
        auto bufferDesc = CD3DX12_RESOURCE_DESC::Buffer(bufferSize);

        ThrowIfFailed(m_device->CreateCommittedResource(
            &heapProps,
            D3D12_HEAP_FLAG_NONE,
            &bufferDesc,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr,
            IID_PPV_ARGS(&m_sortBoundsResetBuffer)));
        m_sortBoundsResetBuffer->SetName(L"SortBoundsResetBuffer");

        // Map and Copy Data
        void* pData;
        CD3DX12_RANGE readRange(0, 0); // We do not intend to read from this resource on the CPU
        ThrowIfFailed(m_sortBoundsResetBuffer->Map(0, &readRange, &pData));
        memcpy(pData, initData, bufferSize);
        m_sortBoundsResetBuffer->Unmap(0, nullptr);
    }
}

void Renderer::CreateIndirectCommandSignature()
{
    D3D12_INDIRECT_ARGUMENT_DESC argDesc = {};
    argDesc.Type = D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH;

    D3D12_COMMAND_SIGNATURE_DESC cmdSigDesc = {};
    cmdSigDesc.ByteStride = sizeof(D3D12_DISPATCH_ARGUMENTS);
    cmdSigDesc.NumArgumentDescs = 1;
    cmdSigDesc.pArgumentDescs = &argDesc;
    cmdSigDesc.NodeMask = 0;

    // Use the compute root signature if you are changing root constants per dispatch (optional),
    // otherwise nullptr is fine for just Dispatch args.
    ThrowIfFailed(m_device->CreateCommandSignature(
        &cmdSigDesc, nullptr, IID_PPV_ARGS(&m_commandSignature)));
}

void Renderer::CompileSetupIndirectShader()
{
    // HLSL Code with preprocessor directive
    const char* shaderCode = R"(
        RWByteAddressBuffer Counters : register(u0);
        RWStructuredBuffer<uint3> IndirectArgs : register(u1);

        cbuffer Params : register(b0) {
            uint counterReadIdx;   // Which counter to read (Active paths)
            uint argWriteIdx;      // Where to write args
            uint counterClearIdx;  // Which counter to reset (Next pass)
            uint groupSize;        // e.g. 64 (threads per group in Trace shader)
        };

        [numthreads(1, 1, 1)]
        void main() {
            // 1. Read the number of active rays
            uint activeCount = Counters.Load(counterReadIdx * 4);

            // 2. Calculate Dispatch Args (Round up)
            uint groups = (activeCount + groupSize - 1) / groupSize;

            // 3. Write Dispatch Arguments (X, Y, Z)
            IndirectArgs[argWriteIdx] = uint3(groups, 1, 1);

            // 4. Reset the 'Next' counter ONLY if defined
            #if CLEAR_COUNTER
                Counters.Store(counterClearIdx * 4, 0);
            #endif
        }
    )";

    // --- Helper to compile with macros ---
    auto CompileVariant = [&](const char* name, bool doClear, ComPtr<ID3D12PipelineState>& pso) {
        D3D_SHADER_MACRO macros[] = {
            { "CLEAR_COUNTER", doClear ? "1" : "0" },
            { NULL, NULL }
        };

        ComPtr<ID3DBlob> shaderBlob;
        ComPtr<ID3DBlob> errorBlob;
        HRESULT hr = D3DCompile(shaderCode, strlen(shaderCode), name,
                                macros, nullptr, "main", "cs_5_0", 0, 0, &shaderBlob, &errorBlob);

        if (FAILED(hr)) {
            if (errorBlob) OutputDebugStringA((char*)errorBlob->GetBufferPointer());
            throw std::runtime_error(std::string("Failed to compile Setup shader: ") + name);
        }

        D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
        psoDesc.pRootSignature = m_rsSetupIndirect.Get();
        psoDesc.CS = { shaderBlob->GetBufferPointer(), shaderBlob->GetBufferSize() };
        ThrowIfFailed(m_device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&pso)));
    };

    // 1. Create Root Signature (Shared by both)
    {
        CD3DX12_ROOT_PARAMETER1 rootParams[2];
        CD3DX12_DESCRIPTOR_RANGE1 range[2];

        // u0 (Counters), u1 (Args)
        range[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 2, 0, 0, D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE);
        rootParams[0].InitAsDescriptorTable(1, &range[0]);

        // b0 (Constants)
        rootParams[1].InitAsConstants(4, 0);

        CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC rsDesc;
        rsDesc.Init_1_1(2, rootParams, 0, nullptr);

        ComPtr<ID3DBlob> sigBlob, errBlob;
        D3D12SerializeVersionedRootSignature(&rsDesc, &sigBlob, &errBlob);
        ThrowIfFailed(m_device->CreateRootSignature(0, sigBlob->GetBufferPointer(), sigBlob->GetBufferSize(), IID_PPV_ARGS(&m_rsSetupIndirect)));
    }

    // 2. Compile Both Variants
    CompileVariant("SetupIndirect_Clear",   true,  m_psoSetupIndirect);
    CompileVariant("SetupIndirect_NoClear", false, m_psoSetupIndirectNoClear);
}

void Renderer::ClearSortBuffers(ID3D12GraphicsCommandList* cmdList)
{
    // -----------------------------------------------------------------------
    // 1. Clear Histogram (Count Buffer) to 0
    // -----------------------------------------------------------------------
    // Since this buffer needs to be all Zeros, ClearUnorderedAccessViewUint is fastest.
    UINT clearValuesCount[4] = { 0, 0, 0, 0 };
    cmdList->ClearUnorderedAccessViewUint(
        m_sortCountGpuHandle,
        m_sortCountCpuHandle,
        m_sortCountBuffer.Get(),
        clearValuesCount,
        0, nullptr
    );

    // -----------------------------------------------------------------------
    // 2. Reset Bounds Buffer to {MAX, MAX, MAX, 0, 0, 0...}
    // -----------------------------------------------------------------------
    // We cannot use ClearUAV here because we need mixed values.
    // We perform a Copy from the pre-calculated reset buffer.

    // A. Transition Bounds Buffer to COPY_DEST
    CD3DX12_RESOURCE_BARRIER barrierToDest = CD3DX12_RESOURCE_BARRIER::Transition(
        m_sortBoundsBuffer.Get(),
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_COPY_DEST
    );
    cmdList->ResourceBarrier(1, &barrierToDest);

    // B. Perform the Copy
    cmdList->CopyResource(m_sortBoundsBuffer.Get(), m_sortBoundsResetBuffer.Get());

    // C. Transition Bounds Buffer back to UAV
    CD3DX12_RESOURCE_BARRIER barrierToUAV = CD3DX12_RESOURCE_BARRIER::Transition(
        m_sortBoundsBuffer.Get(),
        D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS
    );
    cmdList->ResourceBarrier(1, &barrierToUAV);
}

void Renderer::CreateReadbackBuffer() {
    D3D12_RESOURCE_DESC texDesc = m_scratchPing->GetDesc();
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint;
    UINT64 totalBytes = 0;

    m_device->GetCopyableFootprints(&texDesc, 0, 1, 0, &footprint, nullptr, nullptr, &totalBytes);

    auto heapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK);
    auto bufDesc   = CD3DX12_RESOURCE_DESC::Buffer(totalBytes);

    ThrowIfFailed(m_device->CreateCommittedResource(
        &heapProps, D3D12_HEAP_FLAG_NONE, &bufDesc,
        D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
        IID_PPV_ARGS(&m_readbackBuffer)));

    m_readbackBuffer->SetName(L"SimulationReadbackBuffer");
    std::wcout << L"[Init] Readback buffer created. Size: " << totalBytes << L" bytes." << std::endl;
}

void Renderer::SaveSimulationData(uint32_t stepIndex) {
    namespace fs = std::filesystem;
    if (!fs::exists("output")) {
        fs::create_directory("output");
    }

    D3D12_RESOURCE_DESC texDesc = m_scratchPing->GetDesc();
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint;
    m_device->GetCopyableFootprints(&texDesc, 0, 1, 0, &footprint, nullptr, nullptr, nullptr);

    UINT width = static_cast<UINT>(texDesc.Width);
    UINT height = texDesc.Height;
    UINT rowPitch = footprint.Footprint.RowPitch;

    // --- SETUP START ---
    // Ensure we start with a fresh allocator and an OPEN command list for the first slice
    ThrowIfFailed(m_commandAllocators[m_frameIndex]->Reset());
    ThrowIfFailed(m_commandList->Reset(m_commandAllocators[m_frameIndex].Get(), m_pipelineState.Get()));
    // -------------------

    // Helper to process one slice
    auto ProcessSlice = [&](UINT sliceIndex, auto dataExtractionFunc) {
        // 1. Barrier: UAV -> COPY_SOURCE
        auto b1 = CD3DX12_RESOURCE_BARRIER::Transition(m_scratchPing.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        m_commandList->ResourceBarrier(1, &b1);

        // 2. Copy Texture -> Buffer
        UINT subresourceIndex = D3D12CalcSubresource(0, sliceIndex, 0, 1, texDesc.DepthOrArraySize);
        CD3DX12_TEXTURE_COPY_LOCATION dst(m_readbackBuffer.Get(), footprint);
        CD3DX12_TEXTURE_COPY_LOCATION src(m_scratchPing.Get(), subresourceIndex);
        m_commandList->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);

        // 3. Barrier: COPY_SOURCE -> UAV (Restore state for next usage/slice)
        auto b2 = CD3DX12_RESOURCE_BARRIER::Transition(m_scratchPing.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        m_commandList->ResourceBarrier(1, &b2);

        // 4. Flush and Wait
        ThrowIfFailed(m_commandList->Close());
        ID3D12CommandList* ppCommandLists[] = { m_commandList.Get() };
        m_commandQueue->ExecuteCommandLists(1, ppCommandLists);

        m_fenceValue++;
        m_commandQueue->Signal(m_fence.Get(), m_fenceValue);

        // Wait for GPU to finish copy before mapping CPU buffer
        if (m_fence->GetCompletedValue() < m_fenceValue) {
            m_fence->SetEventOnCompletion(m_fenceValue, m_fenceEvent);
            WaitForSingleObject(m_fenceEvent, INFINITE);
        }

        // 5. Read Data
        uint8_t* pMappedData = nullptr;
        // Range 0,0 indicates we do not intend to write, but we are reading
        CD3DX12_RANGE readRange(0, footprint.Footprint.RowPitch * height);
        ThrowIfFailed(m_readbackBuffer->Map(0, &readRange, reinterpret_cast<void**>(&pMappedData)));

        dataExtractionFunc(pMappedData);

        // Range 0,0 indicates we didn't write anything to the CPU pointer
        CD3DX12_RANGE writeRange(0, 0);
        m_readbackBuffer->Unmap(0, &writeRange);

        // 6. Reset for the NEXT iteration (or for final cleanup)
        // Since we waited on the fence, the GPU is done with this allocator, so we can reset it.
        ThrowIfFailed(m_commandAllocators[m_frameIndex]->Reset());
        ThrowIfFailed(m_commandList->Reset(m_commandAllocators[m_frameIndex].Get(), m_pipelineState.Get()));
    };

    // Helper to write binary
    auto WriteBinary = [&](const std::string& suffix, const std::vector<float>& data) {
        std::string filename = "output/" + std::to_string(stepIndex) + "_" + suffix + ".bin";
        std::ofstream outFile(filename, std::ios::binary);
        if (outFile.is_open()) {
            outFile.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
            outFile.close();
        }
    };

    // Slice 7: Restir(x), GT(y), Init(z), Albedo(w)
    ProcessSlice(7, [&](uint8_t* rawData) {
        std::vector<float> a(width * height), b(width * height), c(width * height), d(width * height);
        for (UINT y = 0; y < height; ++y) {
            float* row = reinterpret_cast<float*>(rawData + (y * rowPitch));
            for (UINT x = 0; x < width; ++x) {
                a[y * width + x] = row[x * 4 + 0];
                b[y * width + x] = row[x * 4 + 1];
                c[y * width + x] = row[x * 4 + 2];
                d[y * width + x] = row[x * 4 + 3];
            }
        }
        WriteBinary("restir", a); WriteBinary("gt", b); WriteBinary("init", c); WriteBinary("albedo", d);
    });

    // Slice 8: Roughness(x), Depth(y)
    ProcessSlice(8, [&](uint8_t* rawData) {
        std::vector<float> e(width * height), f(width * height);
        for (UINT y = 0; y < height; ++y) {
            float* row = reinterpret_cast<float*>(rawData + (y * rowPitch));
            for (UINT x = 0; x < width; ++x) {
                e[y * width + x] = row[x * 4 + 0];
                f[y * width + x] = row[x * 4 + 1];
            }
        }
        WriteBinary("roughness", e); WriteBinary("depth", f);
    });

    // Slice 9: Normal(xyz)
    ProcessSlice(9, [&](uint8_t* rawData) {
        std::vector<float> g(width * height * 3);
        for (UINT y = 0; y < height; ++y) {
            float* row = reinterpret_cast<float*>(rawData + (y * rowPitch));
            for (UINT x = 0; x < width; ++x) {
                size_t idx = (y * width + x) * 3;
                g[idx + 0] = row[x * 4 + 0];
                g[idx + 1] = row[x * 4 + 1];
                g[idx + 2] = row[x * 4 + 2];
            }
        }
        WriteBinary("normal", g);
    });

    // --- CRITICAL FIX ---
    // ProcessSlice leaves the command list in an OPEN (Reset) state.
    // We must CLOSE it before returning, otherwise OnRender() will crash
    // when it tries to reset the allocator.
    ThrowIfFailed(m_commandList->Close());
    // --------------------

    std::wcout << L"[IO] Captured Step " << stepIndex << std::endl;
}


void Renderer::FlushExecuteAndWait()
{
    ThrowIfFailed(m_commandList->Close());
    ID3D12CommandList* lists[] = { m_commandList.Get() };
    m_commandQueue->ExecuteCommandLists(1, lists);

    const UINT64 fenceToWait = m_fenceValue;
    ThrowIfFailed(m_commandQueue->Signal(m_fence.Get(), fenceToWait));
    m_fenceValue++;

    if (m_fence->GetCompletedValue() < fenceToWait) {
        ThrowIfFailed(m_fence->SetEventOnCompletion(fenceToWait, m_fenceEvent));
        WaitForSingleObject(m_fenceEvent, INFINITE);
    }

    // Reset allocator + command list for continued recording
    auto* allocator = m_commandAllocators[m_frameIndex].Get();
    ThrowIfFailed(allocator->Reset());
    ThrowIfFailed(m_commandList->Reset(allocator, nullptr));

    RebindAfterReset();
}

void Renderer::RebindAfterReset()
{
    // Rebind state that does NOT persist across Reset()
    m_commandList->SetGraphicsRootSignature(m_rootSignature.Get());
    m_commandList->RSSetViewports(1, &m_viewport);
    m_commandList->RSSetScissorRects(1, &m_scissorRect);

    CD3DX12_CPU_DESCRIPTOR_HANDLE rtv(
        m_rtvHeap->GetCPUDescriptorHandleForHeapStart(),
        m_frameIndex, m_rtvDescriptorSize);
    CD3DX12_CPU_DESCRIPTOR_HANDLE dsv(m_dsvHeap->GetCPUDescriptorHandleForHeapStart());
    m_commandList->OMSetRenderTargets(1, &rtv, FALSE, &dsv);

    ID3D12DescriptorHeap* heaps[] = { m_srvUavHeap.Get() };
    m_commandList->SetDescriptorHeaps(_countof(heaps), heaps);
}


static std::string WideToUtf8(const std::wstring& ws)
{
    if (ws.empty()) return {};
    int len = WideCharToMultiByte(CP_UTF8, 0, ws.data(), (int)ws.size(), nullptr, 0, nullptr, nullptr);
    std::string s(len, 0);
    WideCharToMultiByte(CP_UTF8, 0, ws.data(), (int)ws.size(), s.data(), len, nullptr, nullptr);
    return s;
}

void Renderer::InitML_ONNX_DML(const wchar_t* onnxPath)
{
    // 1) Create a DirectML device (requested). Not strictly required for ORT-DML EP,
    // but good sanity check that DirectML is available on this D3D12 device.
    ThrowIfFailed(DMLCreateDevice(m_device.Get(), DML_CREATE_DEVICE_FLAG_NONE, IID_PPV_ARGS(&m_dmlDevice)));

    // 2) ORT environment/session
    m_ortEnv = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "RendererML");

    m_ortSessionOptions = Ort::SessionOptions();
    m_ortSessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    m_ortSessionOptions.SetIntraOpNumThreads(1);

    // Append DML execution provider (device_id=0 is typical; choose adapter index if needed)
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_DML(m_ortSessionOptions, /*device_id*/ 0));

    // 3) Create session
    // FIX: Pass 'onnxPath' (wchar_t*) directly. Do not convert to UTF-8 on Windows.
    m_ortSession = std::make_unique<Ort::Session>(*m_ortEnv, onnxPath, m_ortSessionOptions);

    // 4) Cache input/output names (using Allocator to fix memory safety error)
    Ort::AllocatorWithDefaultOptions alloc;

    {
        // FIX: Use GetInputNameAllocated (returns smart pointer)
        auto inputNamePtr = m_ortSession->GetInputNameAllocated(0, alloc);
        m_mlInputNameStr = inputNamePtr.get();
        m_mlInputName = m_mlInputNameStr.c_str();
    }
    {
        // FIX: Use GetOutputNameAllocated (returns smart pointer)
        auto outputNamePtr = m_ortSession->GetOutputNameAllocated(0, alloc);
        m_mlOutputNameStr = outputNamePtr.get();
        m_mlOutputName = m_mlOutputNameStr.c_str();
    }

    std::wcout << L"[ML] ORT session created. Input=" << std::wstring(m_mlInputNameStr.begin(), m_mlInputNameStr.end())
               << L" Output=" << std::wstring(m_mlOutputNameStr.begin(), m_mlOutputNameStr.end()) << std::endl;
}

static UINT64 AlignUp(UINT64 v, UINT64 a) { return (v + (a - 1)) & ~(a - 1); }
void Renderer::CreateMLBuffers()
{
    if (!m_scratchPing || !m_ortSession) return;

    // --- CONFIG ---
    m_mlInputSlices = { 7, 8, 9 };
    m_mlOutputSlice = 10;
    // --------------

    const auto desc = m_scratchPing->GetDesc();
    const UINT W = (UINT)desc.Width;
    const UINT H = (UINT)desc.Height;

    // 1. Calculate Readback Size (Keep this as is, we read Float32 from GPU)
    UINT numRows = 0;
    UINT64 rowSizeInBytes = 0;
    m_device->GetCopyableFootprints(&desc, 0, 1, 0, &m_mlFootprint, &numRows, &rowSizeInBytes, &m_mlSliceBytes);
    m_mlRowPitch = m_mlFootprint.Footprint.RowPitch;
    m_mlSliceBytesAligned = AlignUp(m_mlSliceBytes, D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT);

    // 2. Create Resources (D3D12 Resources remain FP32 as the Texture is FP32)
    auto readbackHeapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK);
    auto readbackBufDesc   = CD3DX12_RESOURCE_DESC::Buffer(m_mlSliceBytesAligned * m_mlInputSlices.size());
    ThrowIfFailed(m_device->CreateCommittedResource(
        &readbackHeapProps, D3D12_HEAP_FLAG_NONE, &readbackBufDesc,
        D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&m_mlReadbackBuffer)));

    auto uploadHeapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
    auto uploadBufDesc   = CD3DX12_RESOURCE_DESC::Buffer(m_mlSliceBytesAligned);
    ThrowIfFailed(m_device->CreateCommittedResource(
        &uploadHeapProps, D3D12_HEAP_FLAG_NONE, &uploadBufDesc,
        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&m_mlUploadBuffer)));

    // 3. Define Shapes
    m_mlInputShape  = { 1, 8, (int64_t)H, (int64_t)W };
    m_mlOutputShape = { 1, 1, (int64_t)H, (int64_t)W };

    m_mlInputElems  = (size_t)(1 * 8 * H * W);
    m_mlOutputElems = (size_t)(1 * 1 * H * W);

    // [CHANGE] Resize to uint16_t size
    m_mlInputCPU.resize(m_mlInputElems);
    m_mlOutputCPU.resize(m_mlOutputElems);

    std::wcout << L"[ML] Buffers Ready (FP16 Mode). Input [1,8], Output [1,1]" << std::endl;
}

void Renderer::RunMLPass()
{
    if (!m_enableML || !m_scratchPing || !m_mlReadbackBuffer || !m_mlUploadBuffer || !m_ortSession) return;

    const auto desc = m_scratchPing->GetDesc();
    const UINT W = (UINT)desc.Width;
    const UINT H = (UINT)desc.Height;

    // --- STEP A: GPU -> CPU Copy (Standard FP32 Readback) ---
    FlushExecuteAndWait();

    {
        auto toSrc = CD3DX12_RESOURCE_BARRIER::Transition(m_scratchPing.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        m_commandList->ResourceBarrier(1, &toSrc);
    }

    for (UINT i = 0; i < m_mlInputSlices.size(); ++i) {
        UINT slice = m_mlInputSlices[i];
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp = m_mlFootprint;
        fp.Offset = i * m_mlSliceBytesAligned;

        CD3DX12_TEXTURE_COPY_LOCATION dst(m_mlReadbackBuffer.Get(), fp);
        CD3DX12_TEXTURE_COPY_LOCATION src(m_scratchPing.Get(), D3D12CalcSubresource(0, slice, 0, 1, desc.DepthOrArraySize));
        m_commandList->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    }

    {
        auto toUav = CD3DX12_RESOURCE_BARRIER::Transition(m_scratchPing.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        m_commandList->ResourceBarrier(1, &toUav);
    }

    FlushExecuteAndWait();

    // --- STEP B: Inference (FP16 Conversion) ---
    try {
        uint8_t* mapped = nullptr;
        ThrowIfFailed(m_mlReadbackBuffer->Map(0, nullptr, (void**)&mapped));

        const uint8_t* pSlice7 = mapped + (0 * m_mlSliceBytesAligned);
        const uint8_t* pSlice8 = mapped + (1 * m_mlSliceBytesAligned);
        const uint8_t* pSlice9 = mapped + (2 * m_mlSliceBytesAligned);

        const size_t stride = W * H;

        // Pointers for 8 Input Channels (pointing to uint16_t now)
        uint16_t* pCh0 = m_mlInputCPU.data() + (0 * stride);
        uint16_t* pCh1 = m_mlInputCPU.data() + (1 * stride);
        uint16_t* pCh2 = m_mlInputCPU.data() + (2 * stride);
        uint16_t* pCh3 = m_mlInputCPU.data() + (3 * stride);
        uint16_t* pCh4 = m_mlInputCPU.data() + (4 * stride);
        uint16_t* pCh5 = m_mlInputCPU.data() + (5 * stride);
        uint16_t* pCh6 = m_mlInputCPU.data() + (6 * stride);
        uint16_t* pCh7 = m_mlInputCPU.data() + (7 * stride);

        // 1. Pack Input (Float -> Half)
        #pragma omp parallel for schedule(static)
        for (int y = 0; y < (int)H; ++y)
        {
            const float* row7 = (const float*)(pSlice7 + y * m_mlRowPitch);
            const float* row8 = (const float*)(pSlice8 + y * m_mlRowPitch);
            const float* row9 = (const float*)(pSlice9 + y * m_mlRowPitch);

            for (int x = 0; x < (int)W; ++x)
            {
                size_t pixelIdx = y * W + x;

                // Helper to convert float to half
                auto F2H = [](float f) { return DirectX::PackedVector::XMConvertFloatToHalf(f); };

                pCh0[pixelIdx] = F2H(row7[x * 4 + 0]); // Restir
                pCh1[pixelIdx] = F2H(row7[x * 4 + 2]); // Init
                pCh2[pixelIdx] = F2H(row7[x * 4 + 3]); // Albedo

                pCh3[pixelIdx] = F2H(row9[x * 4 + 0]); // Normal X
                pCh4[pixelIdx] = F2H(row9[x * 4 + 1]); // Normal Y
                pCh5[pixelIdx] = F2H(row9[x * 4 + 2]); // Normal Z

                pCh6[pixelIdx] = F2H(1.0f - row8[x * 4 + 0]); // Roughness
                pCh7[pixelIdx] = F2H(row8[x * 4 + 1]); // Depth
            }
        }
        m_mlReadbackBuffer->Unmap(0, nullptr);

        // 2. Run ORT with FP16 Type
        Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        // [CHANGE] Create Tensor with FLOAT16 type using uint16_t data container
        Ort::Value inputTensor = Ort::Value::CreateTensor(
            memInfo,
            m_mlInputCPU.data(),
            m_mlInputCPU.size() * sizeof(uint16_t), // Size in bytes
            m_mlInputShape.data(),
            m_mlInputShape.size(),
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16   // Explicitly define as FP16
        );

        Ort::RunOptions runOpts;
        auto outputs = m_ortSession->Run(runOpts, &m_mlInputName, &inputTensor, 1, &m_mlOutputName, 1);

        // 3. Get Output (FP16)
        // Note: GetTensorMutableData returns a void* or typed pointer. For FP16, it effectively returns uint16_t*
        uint16_t* outPtr = outputs[0].GetTensorMutableData<uint16_t>();
        size_t outCount = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();

        if (outCount != m_mlOutputCPU.size()) {
             if (outCount > m_mlOutputCPU.size()) outCount = m_mlOutputCPU.size();
        }
        memcpy(m_mlOutputCPU.data(), outPtr, outCount * sizeof(uint16_t));

        // 4. Unpack Output (Half -> Float -> Texture)
        uint8_t* up = nullptr;
        ThrowIfFailed(m_mlUploadBuffer->Map(0, nullptr, (void**)&up));

        const uint16_t* pOut0 = m_mlOutputCPU.data();

        for (int y = 0; y < (int)H; ++y)
        {
            float* rowOut = (float*)(up + y * m_mlRowPitch);
            for (int x = 0; x < (int)W; ++x)
            {
                size_t pixelIdx = y * W + x;

                // Write Model Output to R (Half to Float)
                rowOut[x * 4 + 0] = DirectX::PackedVector::XMConvertHalfToFloat(pOut0[pixelIdx]);

                // Zero out G, B; Alpha = 1
                rowOut[x * 4 + 1] = 0.0f;
                rowOut[x * 4 + 2] = 0.0f;
                rowOut[x * 4 + 3] = 1.0f;
            }
        }
        m_mlUploadBuffer->Unmap(0, nullptr);

    }
    catch (const Ort::Exception& e) {
        std::wcerr << L"[ML] ORT ERROR: " << e.what() << std::endl;
        m_enableML = false;
        return;
    }

    // --- STEP C: CPU -> GPU Copy (Standard) ---
    {
        auto toDest = CD3DX12_RESOURCE_BARRIER::Transition(m_scratchPing.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST);
        m_commandList->ResourceBarrier(1, &toDest);
    }

    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp = m_mlFootprint;
    fp.Offset = 0;
    CD3DX12_TEXTURE_COPY_LOCATION src(m_mlUploadBuffer.Get(), fp);
    CD3DX12_TEXTURE_COPY_LOCATION dst(m_scratchPing.Get(), D3D12CalcSubresource(0, m_mlOutputSlice, 0, 1, desc.DepthOrArraySize));
    m_commandList->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);

    {
        auto toUav = CD3DX12_RESOURCE_BARRIER::Transition(m_scratchPing.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        m_commandList->ResourceBarrier(1, &toUav);
    }
}

void Renderer::CreateDLSSResources() {
    auto createTex = [&](ComPtr<ID3D12Resource>& res, DXGI_FORMAT fmt, const wchar_t* name) {
        D3D12_RESOURCE_DESC desc = {};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        desc.Width = GetWidth();
        desc.Height = GetHeight();
        desc.DepthOrArraySize = 1;
        desc.MipLevels = 1;
        desc.Format = fmt;
        desc.SampleDesc.Count = 1;
        desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

        // Change: Start in UNORDERED_ACCESS so RayGen can write to them immediately
        ThrowIfFailed(m_device->CreateCommittedResource(
                &nv_helpers_dx12::kDefaultHeapProps, D3D12_HEAP_FLAG_NONE, &desc,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&res)));
        res->SetName(name);
    };

    // 1. Core Inputs
    createTex(m_dlssDepth,          DXGI_FORMAT_R32_FLOAT,          L"DLSS_Depth");
    createTex(m_dlssMVec,           DXGI_FORMAT_R16G16_FLOAT,       L"DLSS_MVec");
    createTex(m_dlssNormals,        DXGI_FORMAT_R16G16B16A16_FLOAT, L"DLSS_Normals");
    createTex(m_dlssDiffuseAlbedo,  DXGI_FORMAT_R16G16B16A16_FLOAT, L"DLSS_DiffuseAlbedo");
    createTex(m_dlssOutput,         DXGI_FORMAT_R16G16B16A16_FLOAT, L"DLSS_Output");

    // 2. RR Specific Inputs
    createTex(m_dlssSpecularAlbedo, DXGI_FORMAT_R16G16B16A16_FLOAT, L"DLSS_SpecAlbedo");
    createTex(m_dlssRoughness,      DXGI_FORMAT_R16_FLOAT,          L"DLSS_Roughness");
    createTex(m_dlssSpecMVec,       DXGI_FORMAT_R16G16_FLOAT,       L"DLSS_SpecMVec");
    createTex(m_dlssSpecHitDist,    DXGI_FORMAT_R16_FLOAT,          L"DLSS_HitDist");

    // 3. Optional Guides
    createTex(m_dlssTransparency,     DXGI_FORMAT_R16G16B16A16_FLOAT, L"DLSS_Trans");
    createTex(m_dlssColorBeforeTrans, DXGI_FORMAT_R16G16B16A16_FLOAT, L"DLSS_ColorPreTrans");

    // After createTex(...) sets UAV initial state:
    m_state.SetInitialState(m_dlssDepth.Get(),          D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    m_state.SetInitialState(m_dlssMVec.Get(),           D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    m_state.SetInitialState(m_dlssNormals.Get(),        D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    m_state.SetInitialState(m_dlssDiffuseAlbedo.Get(),  D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    m_state.SetInitialState(m_dlssOutput.Get(),         D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    m_state.SetInitialState(m_dlssSpecularAlbedo.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    m_state.SetInitialState(m_dlssRoughness.Get(),      D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    m_state.SetInitialState(m_dlssSpecMVec.Get(),       D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    m_state.SetInitialState(m_dlssSpecHitDist.Get(),    D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    m_state.SetInitialState(m_dlssTransparency.Get(),   D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    m_state.SetInitialState(m_dlssColorBeforeTrans.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    // Initialize Streamline Viewport
    if (!m_viewportHandle) {
        slAllocateResources(m_commandList.Get(), sl::kFeatureDLSS_RR, m_viewportHandle);
    }
}


static inline void TransitionToUAV_AssumingSRV(
    ID3D12GraphicsCommandList* cmd,
    ID3D12Resource* res)
{
    if (!cmd || !res) return;

    constexpr D3D12_RESOURCE_STATES kSRV =
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE |
        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;

    auto b = CD3DX12_RESOURCE_BARRIER::Transition(
        res, kSRV, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    cmd->ResourceBarrier(1, &b);
}


void Renderer::RunDLSS_RR(ID3D12GraphicsCommandList* cmdList)
{
    if (!cmdList || !m_frameToken) return;

    if (!m_outputResource || !m_dlssOutput || !m_dlssDepth || !m_dlssMVec || !m_dlssNormals ||
        !m_dlssDiffuseAlbedo || !m_dlssSpecularAlbedo || !m_dlssRoughness || !m_dlssSpecHitDist)
        return;

    // 1) DLSS-RR Guide: Inputs should be SRV, Output should be UAV.
    // We explicitly transition them so Streamline doesn't inject incorrect batch barriers.
    constexpr D3D12_RESOURCE_STATES stateUAV = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    constexpr D3D12_RESOURCE_STATES stateSRV = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;

    ID3D12Resource* dlssInputs[] = {
        m_dlssDepth.Get(),
        m_dlssMVec.Get(),
        m_dlssNormals.Get(),
        m_dlssDiffuseAlbedo.Get(),
        m_dlssSpecularAlbedo.Get(),
        m_dlssRoughness.Get(),
        m_dlssSpecHitDist.Get() // Crucial for Ray Reconstruction
    };

    std::vector<D3D12_RESOURCE_BARRIER> preBarriers;
    for (auto* res : dlssInputs) {
        preBarriers.push_back(CD3DX12_RESOURCE_BARRIER::Transition(res, stateUAV, stateSRV));
    }
    // The main color input also goes to SRV
    preBarriers.push_back(CD3DX12_RESOURCE_BARRIER::Transition(m_outputResource.Get(), stateUAV, stateSRV));

    cmdList->ResourceBarrier(static_cast<UINT>(preBarriers.size()), preBarriers.data());

    // 2) Set up Streamline Constants
    sl::Constants constants{};

    glm::mat4 viewGlm = nv_helpers_dx12::CameraManip.getMatrix();
    glm::mat4 viewT   = glm::transpose(viewGlm);

    DirectX::XMFLOAT4X4 viewF{};
    std::memcpy(&viewF, glm::value_ptr(viewT), sizeof(viewF));
    DirectX::XMMATRIX xmView = DirectX::XMLoadFloat4x4(&viewF);

    DirectX::XMMATRIX xmProj = DirectX::XMMatrixPerspectiveFovRH(
        DirectX::XMConvertToRadians(60.0f), m_aspectRatio, 0.00001f, 10000.0f);

    // Use DLSS-specific previous matrices to bypass the timing bug
    DirectX::XMMATRIX xmViewProj     = XMMatrixMultiply(xmView, xmProj);
    DirectX::XMMATRIX xmPrevViewProj = XMMatrixMultiply(m_dlssPrevViewMatrix, m_dlssPrevProjMatrix);

    auto XmToSl = [](const DirectX::XMMATRIX& m) -> sl::float4x4 {
        DirectX::XMFLOAT4X4 tmp;
        DirectX::XMStoreFloat4x4(&tmp, m);
        sl::float4x4 out{};
        std::memcpy(&out, &tmp, sizeof(out));
        return out;
    };

    constants.cameraViewToClip = XmToSl(xmProj);
    constants.clipToCameraView = XmToSl(XMMatrixInverse(nullptr, xmProj));
    constants.clipToPrevClip   = XmToSl(XMMatrixMultiply(XMMatrixInverse(nullptr, xmViewProj), xmPrevViewProj));
    constants.prevClipToClip   = XmToSl(XMMatrixMultiply(XMMatrixInverse(nullptr, xmPrevViewProj), xmViewProj));

    constants.cameraFOV         = XMConvertToRadians(60.0f);
    constants.cameraAspectRatio = m_aspectRatio;
    constants.cameraNear        = 0.00001f;
    constants.cameraFar         = 10000.0f;
    constants.cameraPinholeOffset = { 0.0f, 0.0f };

    constants.jitterOffset = { -m_jitterX, -m_jitterY };

    float w = (float)GetWidth();
    float h = (float)GetHeight();
    constants.mvecScale = { 1.0f / w, 1.0f / h };

    constants.motionVectorsInvalidValue = -1.0f;
    constants.cameraMotionIncluded      = sl::Boolean::eTrue;
    constants.depthInverted             = sl::Boolean::eFalse;
    constants.motionVectors3D           = sl::Boolean::eFalse;
    constants.motionVectorsJittered     = sl::Boolean::eTrue;

    constants.reset = (m_jitterFrameIndex <= 1) ? sl::Boolean::eTrue : sl::Boolean::eFalse;

    {
        auto invView = XMMatrixInverse(nullptr, xmView);
        DirectX::XMFLOAT4X4 invF{};
        XMStoreFloat4x4(&invF, invView);
        constants.cameraPos   = { invF._41, invF._42, invF._43 };
        constants.cameraRight = { invF._11, invF._12, invF._13 };
        constants.cameraUp    = { invF._21, invF._22, invF._23 };
        constants.cameraFwd   = { -invF._31, -invF._32, -invF._33 };
    }

    SL_CHECK(slSetConstants(constants, *m_frameToken, m_viewportHandle));

    // DLSS-RR requires World/Camera matrices in the options struct
    sl::DLSSDOptions options{};
    options.mode                = sl::DLSSMode::eDLAA;
    options.outputWidth         = GetWidth();
    options.outputHeight        = GetHeight();
    options.colorBuffersHDR     = sl::Boolean::eTrue;
    options.normalRoughnessMode = sl::DLSSDNormalRoughnessMode::eUnpacked;
    options.worldToCameraView   = XmToSl(xmView);
    options.cameraViewToWorld   = XmToSl(XMMatrixInverse(nullptr, xmView));

    SL_CHECK(slDLSSDSetOptions(m_viewportHandle, options));

    // 3) Create Tag Structures
    // Explicitly pass (uint32_t)stateSRV so Streamline knows they are ALREADY in SRV.
    sl::Resource slDepth   (sl::ResourceType::eTex2d, m_dlssDepth.Get(),          (uint32_t)stateSRV);
    sl::Resource slMVec    (sl::ResourceType::eTex2d, m_dlssMVec.Get(),           (uint32_t)stateSRV);
    sl::Resource slNormals (sl::ResourceType::eTex2d, m_dlssNormals.Get(),        (uint32_t)stateSRV);
    sl::Resource slAlbedo  (sl::ResourceType::eTex2d, m_dlssDiffuseAlbedo.Get(),  (uint32_t)stateSRV);
    sl::Resource slSpecAlb (sl::ResourceType::eTex2d, m_dlssSpecularAlbedo.Get(), (uint32_t)stateSRV);
    sl::Resource slRough   (sl::ResourceType::eTex2d, m_dlssRoughness.Get(),      (uint32_t)stateSRV);
    sl::Resource slSpecHit (sl::ResourceType::eTex2d, m_dlssSpecHitDist.Get(),    (uint32_t)stateSRV);
    sl::Resource slInput   (sl::ResourceType::eTex2d, m_outputResource.Get(),     (uint32_t)stateSRV);
    sl::Resource slOutput  (sl::ResourceType::eTex2d, m_dlssOutput.Get(),         (uint32_t)stateUAV);

    sl::Extent extent { 0, 0, GetWidth(), GetHeight() };
    auto life = sl::ResourceLifecycle::eValidUntilEvaluate;

    std::vector<sl::ResourceTag> tags = {
        {&slDepth,   sl::kBufferTypeLinearDepth,               life, &extent},
        {&slMVec,    sl::kBufferTypeMotionVectors,       life, &extent},
        {&slNormals, sl::kBufferTypeNormals,             life, &extent},
        {&slRough,   sl::kBufferTypeRoughness,           life, &extent},
        {&slAlbedo,  sl::kBufferTypeAlbedo,              life, &extent},
        {&slSpecAlb, sl::kBufferTypeSpecularAlbedo,      life, &extent},
        {&slSpecHit, sl::kBufferTypeSpecularHitDistance, life, &extent},
        {&slInput,   sl::kBufferTypeScalingInputColor,   life, &extent},
        {&slOutput,  sl::kBufferTypeScalingOutputColor,  life, &extent}
    };

    SL_CHECK(slSetTagForFrame(*m_frameToken, m_viewportHandle, tags.data(), (uint32_t)tags.size(), cmdList));

    // 4) Evaluate DLSS
    // Note: Due to your mid-frame queue execution (RunMLPass), D3D12 decays Streamline's internal buffers to COMMON.
    // Streamline's tracker gets out of sync and fires an invalid barrier on `sl.dlss_d.mvec`.
    // We MUST temporarily suppress D3D12 breaking on error here, otherwise Streamline catches the break,
    // thinks the app crashed, and aborts evaluation (returning error 24).
    ComPtr<ID3D12InfoQueue> infoQueue;
    if (SUCCEEDED(m_device.As(&infoQueue))) {
        infoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_ERROR, FALSE);
    }

    const sl::BaseStructure* evalInputs[] = { &m_viewportHandle, &options };
    sl::Result evalResult = slEvaluateFeature(sl::kFeatureDLSS_RR, *m_frameToken, evalInputs, _countof(evalInputs), cmdList);

    // Re-enable strict debugging immediately after Streamline finishes.
    if (infoQueue) {
        infoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_ERROR, TRUE);
    }

    if (evalResult != sl::Result::eOk) {
        std::wcout << L"[DLSS-RR] slEvaluateFeature failed: " << (int)evalResult << std::endl;
        return;
    }

    // 5) POST-DLSS BARRIERS
    // Streamline evaluated and read the inputs as SRVs.
    // We MUST manually transition them back to UAVs so your Raytracing/Compute shaders
    // can safely write to them at the beginning of the next frame.
    std::vector<D3D12_RESOURCE_BARRIER> postBarriers;
    for (auto* res : dlssInputs) {
        postBarriers.push_back(CD3DX12_RESOURCE_BARRIER::Transition(res, stateSRV, stateUAV));
    }
    postBarriers.push_back(CD3DX12_RESOURCE_BARRIER::Transition(m_outputResource.Get(), stateSRV, stateUAV));

    cmdList->ResourceBarrier(static_cast<UINT>(postBarriers.size()), postBarriers.data());

    m_dlssPrevViewMatrix = xmView;
    m_dlssPrevProjMatrix = xmProj;
}