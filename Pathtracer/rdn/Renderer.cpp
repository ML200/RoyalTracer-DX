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
    float f = 1.0f, r = 0.0f;
    while (index > 0) { f = f / base; r = r + f * (index % base); index = index / base; }
    return r;
}

#define SL_CHECK(x) do { sl::Result r = (x); if (r != sl::Result::eOk) { \
std::wcout << L"[SL] " << L#x << L" failed: " << (int)r << std::endl; return; } \
} while(0)

void Renderer::EnsureSLViewportAllocated(ID3D12GraphicsCommandList* cmdList) {
    if (m_viewportHandle) return;
    if (!cmdList) return;
    sl::Result r = slAllocateResources(cmdList, sl::kFeatureDLSS_RR, m_viewportHandle);
    if (r != sl::Result::eOk) std::wcout << L"[SL] slAllocateResources failed: " << (int)r << std::endl;
}

void PrintAdapterDetails() {
    ComPtr<IDXGIFactory4> factory; CreateDXGIFactory2(0, IID_PPV_ARGS(&factory));
    ComPtr<IDXGIAdapter1> adapter;
    std::wcout << L"\n--- System Adapters ---" << std::endl;
    for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC1 desc; adapter->GetDesc1(&desc);
        std::wcout << L"Adapter [" << i << L"]: " << desc.Description << std::endl;
        std::wcout << L"  Vendor ID: 0x" << std::hex << desc.VendorId << std::dec << std::endl;
        std::wcout << L"  LUID: " << desc.AdapterLuid.HighPart << L":" << desc.AdapterLuid.LowPart << std::endl;
        if (desc.VendorId == 0x10DE) std::wcout << L"  >>> MATCH FOUND (NVIDIA) <<<" << std::endl;
        std::wcout << L"-----------------------" << std::endl;
    }
}

void DumpD3D12Messages(ID3D12Device* device) {
    ComPtr<ID3D12InfoQueue> info;
    if (FAILED(device->QueryInterface(IID_PPV_ARGS(&info)))) return;
    const UINT64 n = info->GetNumStoredMessages();
    for (UINT64 i = 0; i < n; ++i) {
        SIZE_T size = 0; info->GetMessage(i, nullptr, &size);
        std::vector<char> bytes(size);
        auto* msg = reinterpret_cast<D3D12_MESSAGE*>(bytes.data());
        info->GetMessage(i, msg, &size);
        if (msg->pDescription && strstr(msg->pDescription, "sl.dlss_d.mvec") && strstr(msg->pDescription, "RESOURCE_BARRIER_BEFORE_AFTER_MISMATCH")) continue;
        std::wcout << L"[DX] " << msg->pDescription << std::endl;
    }
    info->ClearStoredMessages();
}

struct ScopedTimer {
    const char* name; std::chrono::high_resolution_clock::time_point t0;
    ScopedTimer(const char* n) : name(n), t0(std::chrono::high_resolution_clock::now()) {}
    ~ScopedTimer(){ auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - t0).count(); std::wcout << L"[CPU] " << name << L" took " << ms << L" ms" << std::endl; }
};
#define SCOPE_TIMER(label) ScopedTimer _scopedTimer_##__LINE__(label)

extern "C" {
    __declspec(dllexport) extern const UINT  D3D12SDKVersion = 719;
    __declspec(dllexport) extern const char* D3D12SDKPath    = ".\\";
}

static inline UINT EncodeNormalOct(const XMVECTOR& n) {
    XMVECTOR p = n / (abs(XMVectorGetX(n)) + abs(XMVectorGetY(n)) + abs(XMVectorGetZ(n)));
    if (XMVectorGetZ(p) < 0.0f) {
        float oldX = XMVectorGetX(p), oldY = XMVectorGetY(p);
        p = XMVectorSetX(p, (1.0f - abs(oldY)) * (oldX >= 0.0f ? 1.0f : -1.0f));
        p = XMVectorSetY(p, (1.0f - abs(oldX)) * (oldY >= 0.0f ? 1.0f : -1.0f));
    }
    return (static_cast<uint16_t>(static_cast<int>(XMVectorGetY(p) * 32767.0f)) << 16) | static_cast<uint16_t>(static_cast<int>(XMVectorGetX(p) * 32767.0f));
}

struct MeshSplitResult {
    std::vector<UINT> reorderedIndices, reorderedMaterialIDs;
    UINT opaqueTriCount, alphaTriCount;
};

static MeshSplitResult SplitOpaqueAlpha(const std::vector<UINT>& indices, const std::vector<UINT>& perTriMatIDs, const std::vector<Material>& allMaterials) {
    MeshSplitResult r;
    const UINT triCount = static_cast<UINT>(indices.size()) / 3;
    std::vector<UINT> opaqueIdx, alphaIdx, opaqueMatIDs, alphaMatIDs;
    for (UINT t = 0; t < triCount; ++t) {
        bool isAlpha = (allMaterials[perTriMatIDs[t]].alphaThreshold < 1.0f);
        auto& dstIdx = isAlpha ? alphaIdx : opaqueIdx;
        auto& dstMat = isAlpha ? alphaMatIDs : opaqueMatIDs;
        dstIdx.push_back(indices[t*3+0]); dstIdx.push_back(indices[t*3+1]); dstIdx.push_back(indices[t*3+2]);
        dstMat.push_back(perTriMatIDs[t]);
    }
    r.opaqueTriCount = static_cast<UINT>(opaqueIdx.size()) / 3;
    r.alphaTriCount  = static_cast<UINT>(alphaIdx.size()) / 3;
    r.reorderedIndices = std::move(opaqueIdx);
    r.reorderedIndices.insert(r.reorderedIndices.end(), alphaIdx.begin(), alphaIdx.end());
    r.reorderedMaterialIDs = std::move(opaqueMatIDs);
    r.reorderedMaterialIDs.insert(r.reorderedMaterialIDs.end(), alphaMatIDs.begin(), alphaMatIDs.end());
    return r;
}

// ═══════════════════════════════════════════════════════════════════
// BuildGlobalMeshBuffers — NEW: iterates m_meshes
// ═══════════════════════════════════════════════════════════════════
void Renderer::BuildGlobalMeshBuffers()
{
    SCOPE_TIMER("BuildGlobalMeshBuffers");
    m_geoOffsets.clear();
    m_geoOffsets.resize(m_meshes.size());
    size_t totalVerts = 0, totalIdx = 0;
    for (size_t m = 0; m < m_meshes.size(); ++m) {
        m_meshes[m].globalVertexBase = (UINT)totalVerts;
        m_meshes[m].globalIndexBase  = (UINT)totalIdx;
        m_geoOffsets[m].vertexBase   = (UINT)totalVerts;
        m_geoOffsets[m].indexBase    = (UINT)totalIdx;
        m_geoOffsets[m].materialBase = m_meshes[m].materialIDBase;
        totalVerts += m_meshes[m].vertexCount;
        totalIdx   += m_meshes[m].indexCount;
    }
    m_totalVertexCount = (UINT)totalVerts;
    m_totalIndexCount  = (UINT)totalIdx;

    const UINT vbBytes = m_totalVertexCount * sizeof(BTriVertex);
    const UINT ibBytes = m_totalIndexCount * sizeof(uint32_t);
    auto makeDefault = [&](UINT bytes, ComPtr<ID3D12Resource>& dst, const wchar_t* name) {
        dst = nv_helpers_dx12::CreateBuffer(m_device.Get(), bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST, nv_helpers_dx12::kDefaultHeapProps);
        dst->SetName(name);
    };
    makeDefault(vbBytes, m_vertexGlobal, L"GlobalVertexBuffer");
    makeDefault(ibBytes, m_indexGlobal,  L"GlobalIndexBuffer");

    ComPtr<ID3D12Resource> upload = nv_helpers_dx12::CreateBuffer(m_device.Get(), vbBytes + ibBytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ, nv_helpers_dx12::kUploadHeapProps);
    uint8_t* pUpload = nullptr; { CD3DX12_RANGE r(0, 0); upload->Map(0, &r, (void**)&pUpload); }
    auto* dstVerts = reinterpret_cast<BTriVertex*>(pUpload);
    auto* dstIdx   = reinterpret_cast<uint32_t*>(pUpload + vbBytes);

    for (size_t m = 0; m < m_meshes.size(); ++m) {
        const auto& mesh = m_meshes[m];
        const UINT vBase = mesh.globalVertexBase, iBase = mesh.globalIndexBase;
        BTriVertex* outV = dstVerts + vBase;
        for (UINT i = 0; i < mesh.vertexCount; ++i) {
            const Vertex& sv = mesh.cpuVertices[i];
            outV[i].vertex = sv.position;
            XMVECTOR normal = XMVector3Normalize(XMVectorSet(sv.normal_material.x, sv.normal_material.y, sv.normal_material.z, 0.0f));
            outV[i].packedNormal = EncodeNormalOct(normal);
            PackedVector::XMStoreHalf2(&outV[i].texCoord, XMLoadFloat2(&sv.texCoord));
        }
        uint32_t* outI = dstIdx + iBase;
        for (UINT i = 0; i < mesh.indexCount; ++i) outI[i] = mesh.cpuIndices[i] + vBase;
    }
    upload->Unmap(0, nullptr);
    m_commandList->CopyBufferRegion(m_vertexGlobal.Get(), 0, upload.Get(), 0, vbBytes);
    m_commandList->CopyBufferRegion(m_indexGlobal.Get(),  0, upload.Get(), vbBytes, ibBytes);
    CD3DX12_RESOURCE_BARRIER br[] = {
        CD3DX12_RESOURCE_BARRIER::Transition(m_vertexGlobal.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_GENERIC_READ),
        CD3DX12_RESOURCE_BARRIER::Transition(m_indexGlobal.Get(),  D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_GENERIC_READ),
    };
    m_commandList->ResourceBarrier(_countof(br), br);
}

static InstanceXformCPU ToInstanceXform(const XMMATRIX& M) {
    InstanceXformCPU x{}; XMStoreFloat4x4(&x.objectToWorld, M); return x;
}

void Renderer::CreateTriToLightIdBuffer() {
    if (m_triToLightId.empty()) return;
    const UINT bytes = static_cast<UINT>(m_triToLightId.size() * sizeof(uint32_t));
    ComPtr<ID3D12Resource> upload = nv_helpers_dx12::CreateBuffer(m_device.Get(), bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ, nv_helpers_dx12::kUploadHeapProps);
    { void* p = nullptr; CD3DX12_RANGE r(0,0); ThrowIfFailed(upload->Map(0, &r, &p)); memcpy(p, m_triToLightId.data(), bytes); upload->Unmap(0, nullptr); }
    m_triToLightIdBuffer = nv_helpers_dx12::CreateBuffer(m_device.Get(), bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST, nv_helpers_dx12::kDefaultHeapProps);
    m_commandList->CopyBufferRegion(m_triToLightIdBuffer.Get(), 0, upload.Get(), 0, bytes);
    auto br = CD3DX12_RESOURCE_BARRIER::Transition(m_triToLightIdBuffer.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_GENERIC_READ);
    m_commandList->ResourceBarrier(1, &br);
}

Renderer::Renderer(UINT width, UINT height, std::wstring name)
    : DXSample(width, height, name), m_frameIndex(0),
      m_viewport(0.0f, 0.0f, static_cast<float>(width), static_cast<float>(height)),
      m_scissorRect(0, 0, static_cast<LONG>(width), static_cast<LONG>(height)),
      m_rtvDescriptorSize(0) {
    m_passSequence = {
        L"Pass_raygen_v8.hlsl|rg", L"barrier",
        L"Pass_temp_di_v8.hlsl|cs:16x8", L"barrier",
        L"Pass_temp_gi_v8.hlsl|cs:8x8", L"barrier",
        L"Pass_spat_di_v8.hlsl|cs:16x16", L"barrier",
        L"Pass_spat_gi_v8_1.hlsl|cs:16x16", L"barrier",
        L"Pass_shading_v8.hlsl|cs:16x16", L"barrier",
        L"dlss", L"barrier",
        L"Pass_postprocess_v8.hlsl|cs:8x4", L"barrier",
    };
    try { m_passes.clear(); for (auto& s : m_passSequence) m_passes.push_back(ParsePass(s)); LinkLoops(); }
    catch (const std::exception& e) { MessageBoxA(nullptr, e.what(), "Pass Parsing Error", MB_OK | MB_ICONERROR); exit(1); }
}

void Renderer::OnInit() {
    try {
        m_mod = LoadLibrary("sl.interposer.dll");
        LoadPipeline();
        m_simulator.PromptUserConfiguration();
        m_recorder.Initialize();
        nv_helpers_dx12::CameraManip.setWindowSize(GetWidth(), GetHeight());
        nv_helpers_dx12::CameraManip.setLookat(glm::vec3(-1.5f, 1.5f, 3.5f), glm::vec3(0, 1.0f, 0), glm::vec3(0, 1, 0));
        nv_helpers_dx12::CameraManip.setMode(nv_helpers_dx12::Manipulator::Fly);
        nv_helpers_dx12::CameraManip.setSpeed(0.0f);
        try { LoadAssets(); } catch (const std::exception& e) { MessageBoxA(nullptr, e.what(), "An exception occurred", MB_OK | MB_ICONERROR); exit(1); }
        EnsureSLViewportAllocated(m_commandList.Get());
        GenerateLutTextures();
        CheckRaytracingSupport();
        CreateAccelerationStructures();
        BuildGlobalMeshBuffers();
        ThrowIfFailed(m_commandList->Close());
        ID3D12CommandList* initLists[] = { m_commandList.Get() };
        m_commandQueue->ExecuteCommandLists(_countof(initLists), initLists);
        WaitForPreviousFrame();
        m_lutUploadHeaps.clear();
        m_bindlessUploadHeaps.clear();
        CreateRaytracingPipeline();
        CreateStreamingCompactionBuffers();
        CreateIndirectCommandSignature();
        CompileSetupIndirectShader();
        CreatePerInstanceConstantBuffers();
        CreateGlobalConstantBuffer();
        CreateRaytracingOutputBuffer();
        CreateReadbackBuffer();
        CreateInstancePropertiesBuffer();
        CreateCameraBuffer();
        CreateDLSSResources();
        CreateShaderResourceHeap();
        CreateShaderBindingTable();
        slGetNewFrameToken(m_frameToken, nullptr);
    } catch (const std::exception& e) {
        wchar_t wMsg[4096]; MultiByteToWideChar(CP_UTF8, 0, e.what(), -1, wMsg, 4096);
        MessageBoxW(NULL, wMsg, L"Fatal Initialization Error", MB_OK | MB_ICONERROR); exit(1);
    }
}

// LoadPipeline — unchanged from original
void Renderer::LoadPipeline() {
    sl::Preferences pref{};
    pref.renderAPI = sl::RenderAPI::eD3D12;
    pref.engine = sl::EngineType::eCustom;
    pref.applicationId = 231313132;
    pref.showConsole = true;
    pref.logLevel = sl::LogLevel::eVerbose;
    pref.flags = sl::PreferenceFlags::eLoadDownloadedPlugins | sl::PreferenceFlags::eUseFrameBasedResourceTagging;
    static sl::Feature featList[] = { sl::kFeatureDLSS, sl::kFeatureDLSS_RR};
    pref.featuresToLoad = featList;
    pref.numFeaturesToLoad = _countof(featList);
    sl::Result res = slInit(pref, sl::kSDKVersion);
    if (res != sl::Result::eOk) std::wcout << L"slInit failed! Error: " << (int)res << std::endl;

    UINT dxgiFactoryFlags = 0;
    typedef HRESULT(WINAPI* PFunCreateDXGIFactory2)(UINT, REFIID, void**);
    typedef HRESULT(WINAPI* PFunD3D12CreateDevice)(IUnknown*, D3D_FEATURE_LEVEL, REFIID, void**);
    auto slCreateDXGIFactory2 = reinterpret_cast<PFunCreateDXGIFactory2>(GetProcAddress(m_mod, "CreateDXGIFactory2"));
    auto slD3D12CreateDevice = reinterpret_cast<PFunD3D12CreateDevice>(GetProcAddress(m_mod, "D3D12CreateDevice"));

    ComPtr<IDXGIFactory4> factory;
    ThrowIfFailed(slCreateDXGIFactory2(dxgiFactoryFlags, IID_PPV_ARGS(&factory)));
#if ENABLE_D3D12_DIAGNOSTICS
    dxdiag::EnableDebugLayerAndDred();
#endif
    if (m_useWarpDevice) {
        ComPtr<IDXGIAdapter> warpAdapter;
        ThrowIfFailed(factory->EnumWarpAdapter(IID_PPV_ARGS(&warpAdapter)));
        ThrowIfFailed(slD3D12CreateDevice(warpAdapter.Get(), D3D_FEATURE_LEVEL_12_1, IID_PPV_ARGS(&m_device)));
    } else {
        ComPtr<IDXGIAdapter1> hardwareAdapter;
        GetHardwareAdapter(factory.Get(), &hardwareAdapter);
        ThrowIfFailed(slD3D12CreateDevice(hardwareAdapter.Get(), D3D_FEATURE_LEVEL_12_1, IID_PPV_ARGS(&m_device)));
#if ENABLE_D3D12_DIAGNOSTICS
        dxdiag::HookDevice(m_device.Get());
#endif
    }
    res = slSetD3DDevice(m_device.Get());
    if (res != sl::Result::eOk) std::wcout << L"slSetD3DDevice failed! Code: " << (int)res << std::endl;

    LUID adapterLuid = m_device->GetAdapterLuid();
    PrintAdapterDetails();
    sl::AdapterInfo adapterInfo;
    adapterInfo.deviceLUID = (uint8_t*)&adapterLuid;
    adapterInfo.deviceLUIDSizeInBytes = sizeof(LUID);
    std::wcout << L"[SL] Sending LUID to SL: " << adapterLuid.HighPart << L":" << adapterLuid.LowPart << std::endl;
    sl::Result supportRes = slIsFeatureSupported(sl::kFeatureDLSS_RR, adapterInfo);
    if (supportRes != sl::Result::eOk) std::wcout << L"[DLSS-RR] Check Failed! Error: " << (int)supportRes << std::endl;

    sl::DLSSOptimalSettings dlssSettings;
    sl::DLSSOptions dlssOptions;
    dlssOptions.mode = sl::DLSSMode::eDLAA;
    dlssOptions.outputWidth = m_width; dlssOptions.outputHeight = m_height;
    slDLSSGetOptimalSettings(dlssOptions, dlssSettings);

    D3D12_COMMAND_QUEUE_DESC queueDesc = {};
    queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    ThrowIfFailed(m_device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&m_commandQueue)));

    DXGI_SWAP_CHAIN_DESC1 swapChainDesc = {};
    swapChainDesc.BufferCount = FrameCount; swapChainDesc.Width = m_width; swapChainDesc.Height = m_height;
    swapChainDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM; swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD; swapChainDesc.SampleDesc.Count = 1;
    ComPtr<IDXGISwapChain1> swapChain;
    ThrowIfFailed(factory->CreateSwapChainForHwnd(m_commandQueue.Get(), Win32Application::GetHwnd(), &swapChainDesc, nullptr, nullptr, &swapChain));
    ThrowIfFailed(factory->MakeWindowAssociation(Win32Application::GetHwnd(), DXGI_MWA_NO_ALT_ENTER));
    ThrowIfFailed(swapChain.As(&m_swapChain));
    m_frameIndex = m_swapChain->GetCurrentBackBufferIndex();

    { D3D12_DESCRIPTOR_HEAP_DESC rtvHeapDesc = {}; rtvHeapDesc.NumDescriptors = FrameCount;
      rtvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV; rtvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
      ThrowIfFailed(m_device->CreateDescriptorHeap(&rtvHeapDesc, IID_PPV_ARGS(&m_rtvHeap)));
      m_rtvDescriptorSize = m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV); }
    { CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHandle(m_rtvHeap->GetCPUDescriptorHandleForHeapStart());
      for (UINT n = 0; n < FrameCount; n++) {
          ThrowIfFailed(m_swapChain->GetBuffer(n, IID_PPV_ARGS(&m_renderTargets[n])));
          m_device->CreateRenderTargetView(m_renderTargets[n].Get(), nullptr, rtvHandle);
          rtvHandle.Offset(1, m_rtvDescriptorSize); } }
    for (UINT n = 0; n < FrameCount; ++n)
        ThrowIfFailed(m_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&m_commandAllocators[n])));
    CreateDepthBuffer();
}

// ═══════════════════════════════════════════════════════════════════
// LoadAssets — NEW: uses LoadedScene API + per-model transforms
// ═══════════════════════════════════════════════════════════════════
void Renderer::LoadAssets() {
    ThrowIfFailed(m_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
        m_commandAllocators[m_frameIndex].Get(), nullptr, IID_PPV_ARGS(&m_commandList)));

    std::map<std::string, uint32_t> textureMap;
    std::vector<TextureData> albedoTextures, normalTextures, rmaTextures;

    {
        struct ModelEntry { std::string path; XMMATRIX transform; };
        std::vector<ModelEntry> models = {
            { "./twr.glb", XMMatrixIdentity() },
            // { "./car.glb", XMMatrixScaling(0.4f,0.4f,0.4f) * XMMatrixTranslation(2,0,0) },
        };

        for (const auto& entry : models) {
            const auto& modelName = entry.path;
            const XMMATRIX& modelTransform = entry.transform;

            std::string material_search_path = "./";
            const auto last_slash_idx = modelName.find_last_of("/\\");
            if (std::string::npos != last_slash_idx)
                material_search_path = modelName.substr(0, last_slash_idx + 1);

            std::string ext = modelName.substr(modelName.find_last_of('.') + 1);
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

            LoadedScene scene;
            if (ext == "glb" || ext == "gltf")
                scene = ObjLoader::loadGlbFile(modelName, textureMap, albedoTextures, normalTextures, rmaTextures, material_search_path);
            else
                scene = ObjLoader::loadObjFile(modelName, textureMap, albedoTextures, normalTextures, rmaTextures, material_search_path);

            const UINT globalMatBase = static_cast<UINT>(m_materials.size());
            m_materials.insert(m_materials.end(), scene.materials.begin(), scene.materials.end());

            const UINT meshBaseIdx = static_cast<UINT>(m_meshes.size());

            for (size_t mi = 0; mi < scene.meshes.size(); ++mi) {
                auto& srcMesh = scene.meshes[mi];
                MeshGPU gpu{};
                std::vector<UINT> globalMatIDs = srcMesh.perTriMaterialIDs;
                for (auto& mid : globalMatIDs) mid += globalMatBase;
                MeshSplitResult split = SplitOpaqueAlpha(srcMesh.indices, globalMatIDs, m_materials);
                gpu.cpuVertices    = srcMesh.vertices;
                gpu.cpuIndices     = std::move(split.reorderedIndices);
                gpu.cpuMaterialIDs = std::move(split.reorderedMaterialIDs);
                gpu.vertexCount    = static_cast<UINT>(gpu.cpuVertices.size());
                gpu.indexCount     = static_cast<UINT>(gpu.cpuIndices.size());
                gpu.opaqueTriCount = split.opaqueTriCount;
                gpu.alphaTriCount  = split.alphaTriCount;
                gpu.materialIDBase = static_cast<UINT>(m_materialIDs.size());
                m_materialIDs.insert(m_materialIDs.end(), gpu.cpuMaterialIDs.begin(), gpu.cpuMaterialIDs.end());

                const UINT vbSize = gpu.vertexCount * sizeof(Vertex);
                CD3DX12_RESOURCE_DESC vbDesc = CD3DX12_RESOURCE_DESC::Buffer(vbSize);
                ThrowIfFailed(m_device->CreateCommittedResource(&nv_helpers_dx12::kUploadHeapProps, D3D12_HEAP_FLAG_NONE, &vbDesc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&gpu.vertexBuffer)));
                { UINT8* p; gpu.vertexBuffer->Map(0, nullptr, (void**)&p); memcpy(p, gpu.cpuVertices.data(), vbSize); gpu.vertexBuffer->Unmap(0, nullptr); }

                const UINT ibSize = gpu.indexCount * sizeof(UINT);
                CD3DX12_RESOURCE_DESC ibDesc = CD3DX12_RESOURCE_DESC::Buffer(ibSize);
                ThrowIfFailed(m_device->CreateCommittedResource(&nv_helpers_dx12::kUploadHeapProps, D3D12_HEAP_FLAG_NONE, &ibDesc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&gpu.indexBuffer)));
                { UINT8* p; gpu.indexBuffer->Map(0, nullptr, (void**)&p); memcpy(p, gpu.cpuIndices.data(), ibSize); gpu.indexBuffer->Unmap(0, nullptr); }

                m_meshes.push_back(std::move(gpu));
            }

            for (const auto& [meshIdx, xform] : scene.instances) {
                SceneInstance si{};
                si.meshIndex     = meshBaseIdx + meshIdx;
                si.transform     = xform * modelTransform;
                si.prevTransform = si.transform;
                m_sceneInstances.push_back(si);
            }
        }
    }

    m_bindlessAlbedoBase = BINDLESS_HEAP_START;
    m_bindlessNormalBase = m_bindlessAlbedoBase + static_cast<UINT>(albedoTextures.size());
    m_bindlessRmaBase    = m_bindlessNormalBase + static_cast<UINT>(normalTextures.size());
    m_totalBindlessTextures = static_cast<UINT>(albedoTextures.size() + normalTextures.size() + rmaTextures.size());

    for (auto& mat : m_materials) {
        if (mat.albedoTexID >= 0) mat.albedoTexID += (int)m_bindlessAlbedoBase;
        if (mat.normalTexID >= 0) mat.normalTexID += (int)m_bindlessNormalBase;
        if (mat.rmaTexID >= 0)    mat.rmaTexID    += (int)m_bindlessRmaBase;
    }

    { const UINT sz = static_cast<UINT>(m_materials.size()) * sizeof(Material);
      CD3DX12_RESOURCE_DESC d = CD3DX12_RESOURCE_DESC::Buffer(sz);
      ThrowIfFailed(m_device->CreateCommittedResource(&nv_helpers_dx12::kUploadHeapProps, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&m_materialBuffer)));
      UINT8* p; m_materialBuffer->Map(0, nullptr, (void**)&p); memcpy(p, m_materials.data(), sz); m_materialBuffer->Unmap(0, nullptr); }
    { const UINT sz = static_cast<UINT>(m_materialIDs.size()) * sizeof(UINT);
      CD3DX12_RESOURCE_DESC d = CD3DX12_RESOURCE_DESC::Buffer(sz);
      ThrowIfFailed(m_device->CreateCommittedResource(&nv_helpers_dx12::kUploadHeapProps, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&m_materialIndexBuffer)));
      UINT8* p; m_materialIndexBuffer->Map(0, nullptr, (void**)&p); memcpy(p, m_materialIDs.data(), sz); m_materialIndexBuffer->Unmap(0, nullptr); }

    CreateBindlessTextures(albedoTextures, m_bindlessAlbedoBase, L"Albedo");
    CreateBindlessTextures(normalTextures, m_bindlessNormalBase, L"Normal");
    CreateBindlessTextures(rmaTextures,    m_bindlessRmaBase,    L"RMA");

    { ThrowIfFailed(m_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_fence)));
      m_fenceValue = 1; m_fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
      if (!m_fenceEvent) ThrowIfFailed(HRESULT_FROM_WIN32(GetLastError()));
      WaitForPreviousFrame(); }
}

void Renderer::OnInitTransform() {
    // Transforms come from the scene graph now — nothing to hardcode.
}

void Renderer::OnUpdate() {
    using clock = std::chrono::high_resolution_clock;
    static auto tPrev = clock::now();
    auto tCurr = clock::now();
    float dt = std::chrono::duration<float>(tCurr - tPrev).count();
    tPrev = tCurr;

    if (m_simulator.IsActive()) {
        bool shouldCapture = false;
        bool finished = m_simulator.Update(dt, nv_helpers_dx12::CameraManip, shouldCapture);
        if (shouldCapture) { SaveSimulationData(m_simulator.GetLastCaptureIndex()); }
        if (finished) { std::wcout << L"\n[Sim] Data Generation Complete.\n"; PostQuitMessage(0); return; }
    } else {
        glm::vec3 eye, center, up;
        nv_helpers_dx12::CameraManip.getLookat(eye, center, up);
        glm::vec3 fwd = glm::normalize(center - eye), right = glm::normalize(glm::cross(fwd, up));
        glm::vec3 move(0.0f); float speed = 5.0f;
        if (g_keys['W']) move += fwd; if (g_keys['S']) move -= fwd;
        if (g_keys['D']) move += right; if (g_keys['A']) move -= right;
        if (g_keys[VK_SPACE]) move += up; if (g_keys[VK_CONTROL]) move -= up;
        if (glm::length(move) > 0.0f) { move = glm::normalize(move)*(speed*dt); eye += move; center += move; nv_helpers_dx12::CameraManip.setLookat(eye, center, up); }
    }
    UpdateCameraBuffer();
    m_time++;
    UpdateInstancePropertiesBuffer();
}

void Renderer::OnRender() {
    if (m_frameToken) slGetNewFrameToken(m_frameToken, nullptr);
    static auto s_lastTime = std::chrono::high_resolution_clock::now();
    static int s_frameCount = 0;
    PopulateCommandList();
    ID3D12CommandList* ppCL[] = { m_commandList.Get() };
    m_commandQueue->ExecuteCommandLists(1, ppCL);
    ThrowIfFailed(m_swapChain->Present(0, 0));
    WaitForPreviousFrame();
    s_frameCount++;
    auto currentTime = std::chrono::high_resolution_clock::now();
    float elapsedSec = std::chrono::duration<float>(currentTime - s_lastTime).count();
    if (elapsedSec >= 1.0f) {
        float fps = s_frameCount / elapsedSec;
        std::wstringstream ss; ss << std::fixed << std::setprecision(2) << L"Frame Time: " << 1000.0f/fps << L" ms (" << fps << L" fps)";
        SetWindowTextW(Win32Application::GetHwnd(), ss.str().c_str());
        s_frameCount = 0; s_lastTime = currentTime;
    }
#if ENABLE_D3D12_DIAGNOSTICS
    dxdiag::DumpNewMessages();
#endif
}

void Renderer::OnDestroy() {
    WaitForPreviousFrame();
    CloseHandle(m_fenceEvent);
    if(SL_FAILED(res, slShutdown())) {}
}

void Renderer::PopulateCommandList()
{
    auto* allocator = m_commandAllocators[m_frameIndex].Get();
    ThrowIfFailed(allocator->Reset());
    ThrowIfFailed(m_commandList->Reset(allocator, m_pipelineState.Get()));
    m_commandList->SetGraphicsRootSignature(m_rootSignature.Get());
    m_commandList->RSSetViewports(1, &m_viewport);
    m_commandList->RSSetScissorRects(1, &m_scissorRect);
    { auto b = CD3DX12_RESOURCE_BARRIER::Transition(m_renderTargets[m_frameIndex].Get(), D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET); m_commandList->ResourceBarrier(1, &b); }
    CD3DX12_CPU_DESCRIPTOR_HANDLE rtv(m_rtvHeap->GetCPUDescriptorHandleForHeapStart(), m_frameIndex, m_rtvDescriptorSize);
    CD3DX12_CPU_DESCRIPTOR_HANDLE dsv(m_dsvHeap->GetCPUDescriptorHandleForHeapStart());
    m_commandList->OMSetRenderTargets(1, &rtv, FALSE, &dsv);

    // *** KEY CHANGE: use m_tlasInstances ***
    CreateTopLevelAS(m_tlasInstances, true);
    { auto b = CD3DX12_RESOURCE_BARRIER::UAV(m_topLevelASBuffers.pResult.Get()); m_commandList->ResourceBarrier(1, &b); }
    { ID3D12DescriptorHeap* heaps[] = { m_srvUavHeap.Get() }; m_commandList->SetDescriptorHeaps(1, heaps); }

    D3D12_DISPATCH_RAYS_DESC raysDesc{};
    raysDesc.Width = GetWidth(); raysDesc.Height = GetHeight(); raysDesc.Depth = 1;
    const uint64_t sbtStart = m_sbtStorage->GetGPUVirtualAddress();
    const uint32_t rgSize = m_sbtHelper.GetRayGenEntrySize();
    raysDesc.MissShaderTable.StartAddress = sbtStart + m_sbtHelper.GetRayGenSectionSize();
    raysDesc.MissShaderTable.SizeInBytes = m_sbtHelper.GetMissSectionSize();
    raysDesc.MissShaderTable.StrideInBytes = m_sbtHelper.GetMissEntrySize();
    raysDesc.HitGroupTable.StartAddress = raysDesc.MissShaderTable.StartAddress + raysDesc.MissShaderTable.SizeInBytes;
    raysDesc.HitGroupTable.SizeInBytes = m_sbtHelper.GetHitGroupSectionSize();
    raysDesc.HitGroupTable.StrideInBytes = m_sbtHelper.GetHitGroupEntrySize();
    if (m_sbtHelper.GetCallableSectionSize() > 0) {
        raysDesc.CallableShaderTable.StartAddress = raysDesc.HitGroupTable.StartAddress + raysDesc.HitGroupTable.SizeInBytes;
        raysDesc.CallableShaderTable.SizeInBytes = m_sbtHelper.GetCallableSectionSize();
        raysDesc.CallableShaderTable.StrideInBytes = m_sbtHelper.GetCallableEntrySize();
    }
    { auto u = CD3DX12_RESOURCE_BARRIER::Transition(m_outputResource.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS); m_commandList->ResourceBarrier(1, &u); }
    { auto b = CD3DX12_RESOURCE_BARRIER::Transition(m_globalCounterBuffer.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST); m_commandList->ResourceBarrier(1, &b);
      m_commandList->CopyBufferRegion(m_globalCounterBuffer.Get(), 0, m_zeroBuffer.Get(), 0, MAX_STACKS*sizeof(uint32_t));
      auto b2 = CD3DX12_RESOURCE_BARRIER::Transition(m_globalCounterBuffer.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS); m_commandList->ResourceBarrier(1, &b2); }

    uint32_t currentStackIdx = 0, nextStackIdx = 1;
    std::vector<std::pair<int, uint32_t>> loopStack;

    for (size_t i = 0; i < m_passes.size(); ++i) {
        { CD3DX12_RESOURCE_BARRIER barriers[] = { CD3DX12_RESOURCE_BARRIER::UAV(m_sortBoundsBuffer.Get()), CD3DX12_RESOURCE_BARRIER::UAV(m_sortCountBuffer.Get()) }; m_commandList->ResourceBarrier(2, barriers); }
        auto& p = m_passes[i];
        switch (p.stage) {
        case Stage::LoopStart: loopStack.push_back({-1, p.loopCount}); break;
        case Stage::PingSwap: std::swap(currentStackIdx, nextStackIdx); break;
        case Stage::LoopEnd:
            if (!loopStack.empty()) { loopStack.back().second--; if (loopStack.back().second > 0) i = p.targetIdx; else loopStack.pop_back(); } break;
        case Stage::Barrier: { auto u = CD3DX12_RESOURCE_BARRIER::UAV(nullptr); m_commandList->ResourceBarrier(1, &u); } break;
        case Stage::ClearSort: {
            ClearSortBuffers(m_commandList.Get());
            CD3DX12_RESOURCE_BARRIER barriers[] = { CD3DX12_RESOURCE_BARRIER::UAV(m_sortBoundsBuffer.Get()), CD3DX12_RESOURCE_BARRIER::UAV(m_sortCountBuffer.Get()) };
            m_commandList->ResourceBarrier(2, barriers); } break;
        case Stage::RayGen: {
            m_commandList->SetPipelineState1(m_rtStateObject.Get());
            m_commandList->SetComputeRootSignature(m_rayGenSignature.Get());
            m_commandList->SetComputeRootDescriptorTable(0, m_srvUavHeap->GetGPUDescriptorHandleForHeapStart());
            UINT constants[6] = { GetWidth(), GetHeight(), 0, 0, 0, 0 };
            m_commandList->SetComputeRoot32BitConstants(1, 6, constants, 0);
            uint32_t currentRgSlot = m_passIndex[p.file];
            raysDesc.RayGenerationShaderRecord.StartAddress = sbtStart + currentRgSlot * rgSize;
            raysDesc.RayGenerationShaderRecord.SizeInBytes = rgSize;
            m_commandList->DispatchRays(&raysDesc); } break;
        case Stage::Compute: {
            if (p.isWorkGraph) {
                const auto& rt = m_wgRuntime[p.wgIdx];
                m_commandList->SetComputeRootSignature(m_computeSignature.Get());
                m_commandList->SetComputeRootDescriptorTable(0, m_srvUavHeap->GetGPUDescriptorHandleForHeapStart());
                UINT constants[6] = { GetWidth(), GetHeight(), currentStackIdx, nextStackIdx, 0, 0 };
                m_commandList->SetComputeRoot32BitConstants(1, 6, constants, 0);
                D3D12_SET_PROGRAM_DESC setProg{}; setProg.Type = D3D12_PROGRAM_TYPE_WORK_GRAPH;
                setProg.WorkGraph.ProgramIdentifier = rt.id; setProg.WorkGraph.BackingMemory = rt.backing;
                static std::vector<bool> s_inited;
                if (s_inited.size() <= p.wgIdx) s_inited.resize(p.wgIdx + 1, false);
                setProg.WorkGraph.Flags = s_inited[p.wgIdx] ? D3D12_SET_WORK_GRAPH_FLAG_NONE : D3D12_SET_WORK_GRAPH_FLAG_INITIALIZE;
                m_commandList->SetProgram(&setProg); s_inited[p.wgIdx] = true;
                D3D12_DISPATCH_GRAPH_DESC dg{}; dg.Mode = D3D12_DISPATCH_MODE_NODE_CPU_INPUT;
                dg.NodeCPUInput.EntrypointIndex = 0; dg.NodeCPUInput.NumRecords = 1;
                m_commandList->DispatchGraph(&dg);
            } else {
                m_commandList->SetPipelineState(m_csPSOs[p.psoIdx].Get());
                m_commandList->SetComputeRootSignature(m_computeSignature.Get());
                m_commandList->SetComputeRootDescriptorTable(0, m_srvUavHeap->GetGPUDescriptorHandleForHeapStart());
                UINT constants[6] = { GetWidth(), GetHeight(), currentStackIdx, nextStackIdx, 0, 0 };
                m_commandList->SetComputeRoot32BitConstants(1, 6, constants, 0);
                m_commandList->Dispatch((GetWidth()+p.groupX-1)/p.groupX, (GetHeight()+p.groupY-1)/p.groupY, 1);
            } } break;
        case Stage::FixedCompute: {
            m_commandList->SetPipelineState(m_csPSOs[p.psoIdx].Get());
            m_commandList->SetComputeRootSignature(m_computeSignature.Get());
            m_commandList->SetComputeRootDescriptorTable(0, m_srvUavHeap->GetGPUDescriptorHandleForHeapStart());
            UINT constants[6] = { GetWidth(), GetHeight(), currentStackIdx, nextStackIdx, 0, 0 };
            m_commandList->SetComputeRoot32BitConstants(1, 6, constants, 0);
            m_commandList->Dispatch(p.groupX, p.groupY, 1); } break;
        case Stage::Wavefront: {
            bool isInPlace = (currentStackIdx == nextStackIdx);
            m_commandList->SetPipelineState(isInPlace ? m_psoSetupIndirectNoClear.Get() : m_psoSetupIndirect.Get());
            m_commandList->SetComputeRootSignature(m_rsSetupIndirect.Get());
            auto gpuHandle = m_srvUavHeap->GetGPUDescriptorHandleForHeapStart();
            UINT inc = m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            gpuHandle.ptr += 33 * inc;
            m_commandList->SetComputeRootDescriptorTable(0, gpuHandle);
            UINT setupConsts[4] = { currentStackIdx, 0, nextStackIdx, p.groupX };
            m_commandList->SetComputeRoot32BitConstants(1, 4, setupConsts, 0);
            m_commandList->Dispatch(1, 1, 1);
            CD3DX12_RESOURCE_BARRIER barriers[] = { CD3DX12_RESOURCE_BARRIER::UAV(m_indirectArgsBuffer.Get()), CD3DX12_RESOURCE_BARRIER::Transition(m_indirectArgsBuffer.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT) };
            m_commandList->ResourceBarrier(2, barriers);
            m_commandList->SetPipelineState(m_csPSOs[p.psoIdx].Get());
            m_commandList->SetComputeRootSignature(m_computeSignature.Get());
            m_commandList->SetComputeRootDescriptorTable(0, m_srvUavHeap->GetGPUDescriptorHandleForHeapStart());
            UINT traceConsts[6] = { GetWidth(), GetHeight(), currentStackIdx, nextStackIdx, 0, 0 };
            m_commandList->SetComputeRoot32BitConstants(1, 6, traceConsts, 0);
            m_commandList->ExecuteIndirect(m_commandSignature.Get(), 1, m_indirectArgsBuffer.Get(), 0, nullptr, 0);
            CD3DX12_RESOURCE_BARRIER postBarriers[] = { CD3DX12_RESOURCE_BARRIER::Transition(m_indirectArgsBuffer.Get(), D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS), CD3DX12_RESOURCE_BARRIER::UAV(m_stackBuffers[nextStackIdx].Get()), CD3DX12_RESOURCE_BARRIER::UAV(m_globalCounterBuffer.Get()) };
            m_commandList->ResourceBarrier(3, postBarriers); } break;
        case Stage::DLSS: {
            ID3D12Resource* uavs[] = { m_dlssDepth.Get(), m_dlssMVec.Get(), m_dlssNormals.Get(), m_dlssDiffuseAlbedo.Get(), m_dlssSpecularAlbedo.Get(), m_dlssRoughness.Get(), m_dlssSpecMVec.Get(), m_dlssSpecHitDist.Get(), m_dlssTransparency.Get(), m_dlssColorBeforeTrans.Get(), m_dlssInput.Get() };
            for (auto* r : uavs) { if (r) { auto b = CD3DX12_RESOURCE_BARRIER::UAV(r); m_commandList->ResourceBarrier(1, &b); } }
            RunDLSS_RR(m_commandList.Get());
            ID3D12DescriptorHeap* heaps[] = { m_srvUavHeap.Get() }; m_commandList->SetDescriptorHeaps(1, heaps); } break;
        }
    }

    { auto toSrc = CD3DX12_RESOURCE_BARRIER::Transition(m_outputResource.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE); m_commandList->ResourceBarrier(1, &toSrc);
      auto toDst = CD3DX12_RESOURCE_BARRIER::Transition(m_renderTargets[m_frameIndex].Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_DEST); m_commandList->ResourceBarrier(1, &toDst);
      UINT layer = m_displayLevels[m_currentDisplayLevel];
      UINT sub = D3D12CalcSubresource(0, layer, 0, 1, 60);
      CD3DX12_TEXTURE_COPY_LOCATION src(m_outputResource.Get(), sub), dst(m_renderTargets[m_frameIndex].Get(), 0);
      D3D12_BOX box = {0,0,0, GetWidth(), GetHeight(), 1};
      m_commandList->CopyTextureRegion(&dst,0,0,0, &src, &box);
      auto backToRT = CD3DX12_RESOURCE_BARRIER::Transition(m_renderTargets[m_frameIndex].Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_RENDER_TARGET); m_commandList->ResourceBarrier(1, &backToRT);
      auto pres = CD3DX12_RESOURCE_BARRIER::Transition(m_renderTargets[m_frameIndex].Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT); m_commandList->ResourceBarrier(1, &pres); }
    ThrowIfFailed(m_commandList->Close());
}

void Renderer::WaitForPreviousFrame() {
    const UINT64 fence = m_fenceValue;
    ThrowIfFailed(m_commandQueue->Signal(m_fence.Get(), fence));
    m_fenceValue++;
    if (m_fence->GetCompletedValue() < fence) { ThrowIfFailed(m_fence->SetEventOnCompletion(fence, m_fenceEvent)); WaitForSingleObject(m_fenceEvent, INFINITE); }
    m_frameIndex = m_swapChain->GetCurrentBackBufferIndex();
#if ENABLE_D3D12_DIAGNOSTICS
    dxdiag::CheckDeviceRemoved(m_device.Get());
#endif
}

void Renderer::CheckRaytracingSupport() {
    D3D12_FEATURE_DATA_D3D12_OPTIONS5 options5 = {};
    ThrowIfFailed(m_device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &options5, sizeof(options5)));
    if (options5.RaytracingTier < D3D12_RAYTRACING_TIER_1_0) throw std::runtime_error("Raytracing not supported on device");
}

void Renderer::OnKeyDown(UINT8 key) { g_keys[key] = true; }
void Renderer::OnKeyUp(UINT8 key) {
    g_keys[key] = false;
    if (key == 'C') { m_currentDisplayLevel = (m_currentDisplayLevel + 1) % m_displayLevels.size(); }
    if (key == VK_SPACE) { m_raster = !m_raster; }
    if (key == 'K') { m_recorder.CaptureKeyframe(nv_helpers_dx12::CameraManip); }
}

// ═══════════════════════════════════════════════════════════════════
// BLAS / TLAS
// ═══════════════════════════════════════════════════════════════════
constexpr bool kUseBlasCompaction = true;

Renderer::AccelerationStructureBuffers
Renderer::CreateBottomLevelAS(
    std::vector<std::pair<ComPtr<ID3D12Resource>, uint32_t>> vVertexBuffers,
    std::vector<std::pair<ComPtr<ID3D12Resource>, uint32_t>> vIndexBuffers,
    UINT opaqueTriCount, UINT alphaTriCount)
{
    nv_helpers_dx12::BottomLevelASGenerator bottomLevelAS;
    for (size_t i = 0; i < vVertexBuffers.size(); i++) {
        UINT opaqueIdxCount = opaqueTriCount * 3, alphaIdxCount = alphaTriCount * 3;
        if (opaqueIdxCount > 0)
            bottomLevelAS.AddVertexBuffer(vVertexBuffers[i].first.Get(), 0, vVertexBuffers[i].second, sizeof(Vertex), vIndexBuffers[i].first.Get(), 0, opaqueIdxCount, nullptr, 0, true);
        if (alphaIdxCount > 0)
            bottomLevelAS.AddVertexBuffer(vVertexBuffers[i].first.Get(), 0, vVertexBuffers[i].second, sizeof(Vertex), vIndexBuffers[i].first.Get(), opaqueIdxCount * sizeof(UINT), alphaIdxCount, nullptr, 0, false);
    }
    UINT64 scratchSizeInBytes = 0, resultSizeInBytes = 0;
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS buildFlags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
    if constexpr (kUseBlasCompaction) buildFlags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_COMPACTION;
    bottomLevelAS.ComputeASBufferSizes(m_device.Get(), buildFlags, &scratchSizeInBytes, &resultSizeInBytes);
    AccelerationStructureBuffers buffers;
    buffers.pScratch = nv_helpers_dx12::CreateBuffer(m_device.Get(), scratchSizeInBytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON, nv_helpers_dx12::kDefaultHeapProps);

    if constexpr (!kUseBlasCompaction) {
        buffers.pResult = nv_helpers_dx12::CreateBuffer(m_device.Get(), resultSizeInBytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, nv_helpers_dx12::kDefaultHeapProps);
        bottomLevelAS.Generate(m_commandList.Get(), buffers.pScratch.Get(), buffers.pResult.Get(), false, nullptr);
    } else {
        buffers.pResultUncompacted = nv_helpers_dx12::CreateBuffer(m_device.Get(), resultSizeInBytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, nv_helpers_dx12::kDefaultHeapProps);
        ComPtr<ID3D12Resource> compactedSizeBuffer = nv_helpers_dx12::CreateBuffer(m_device.Get(), sizeof(UINT64), D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nv_helpers_dx12::kDefaultHeapProps);
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC postBuildInfoDesc = {};
        postBuildInfoDesc.DestBuffer = compactedSizeBuffer->GetGPUVirtualAddress();
        postBuildInfoDesc.InfoType = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_COMPACTED_SIZE;
        bottomLevelAS.Generate(m_commandList.Get(), buffers.pScratch.Get(), buffers.pResultUncompacted.Get(), false, nullptr);
        D3D12_GPU_VIRTUAL_ADDRESS src = buffers.pResultUncompacted->GetGPUVirtualAddress();
        m_commandList->EmitRaytracingAccelerationStructurePostbuildInfo(&postBuildInfoDesc, 1, &src);
        ComPtr<ID3D12Resource> readbackBuffer = nv_helpers_dx12::CreateBuffer(m_device.Get(), sizeof(UINT64), D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST, nv_helpers_dx12::kReadbackHeapProps);
        auto barrier = CD3DX12_RESOURCE_BARRIER::Transition(compactedSizeBuffer.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        m_commandList->ResourceBarrier(1, &barrier);
        m_commandList->CopyResource(readbackBuffer.Get(), compactedSizeBuffer.Get());
        ThrowIfFailed(m_commandList->Close());
        ID3D12CommandList* ppCL[] = { m_commandList.Get() };
        m_commandQueue->ExecuteCommandLists(1, ppCL); WaitForPreviousFrame();
        ThrowIfFailed(m_commandList->Reset(m_commandAllocators[m_frameIndex].Get(), nullptr));
        UINT64 compactedSize; void* pMap;
        ThrowIfFailed(readbackBuffer->Map(0, nullptr, &pMap)); memcpy(&compactedSize, pMap, sizeof(UINT64)); readbackBuffer->Unmap(0, nullptr);
        buffers.pResult = nv_helpers_dx12::CreateBuffer(m_device.Get(), compactedSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, nv_helpers_dx12::kDefaultHeapProps);
        buffers.pResult->SetName(L"Compacted BLAS");
        m_commandList->CopyRaytracingAccelerationStructure(buffers.pResult->GetGPUVirtualAddress(), buffers.pResultUncompacted->GetGPUVirtualAddress(), D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_COMPACT);
        barrier = CD3DX12_RESOURCE_BARRIER::UAV(buffers.pResult.Get()); m_commandList->ResourceBarrier(1, &barrier);
    }
    return buffers;
}

void Renderer::CreateTopLevelAS(const std::vector<std::pair<ComPtr<ID3D12Resource>, DirectX::XMMATRIX>>& instances, bool updateOnly) {
    if (!updateOnly) {
        for (size_t i = 0; i < instances.size(); i++)
            m_topLevelASGenerator.AddInstance(instances[i].first.Get(), instances[i].second, static_cast<UINT>(i), static_cast<UINT>(2 * i));
        UINT64 scratchSize, resultSize, instanceDescsSize;
        m_topLevelASGenerator.ComputeASBufferSizes(m_device.Get(), true, &scratchSize, &resultSize, &instanceDescsSize);
        m_topLevelASBuffers.pScratch = nv_helpers_dx12::CreateBuffer(m_device.Get(), scratchSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nv_helpers_dx12::kDefaultHeapProps);
        m_topLevelASBuffers.pResult = nv_helpers_dx12::CreateBuffer(m_device.Get(), resultSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, nv_helpers_dx12::kDefaultHeapProps);
        m_topLevelASBuffers.pInstanceDesc = nv_helpers_dx12::CreateBuffer(m_device.Get(), instanceDescsSize, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ, nv_helpers_dx12::kUploadHeapProps);
    }
    m_topLevelASGenerator.Generate(m_commandList.Get(), m_topLevelASBuffers.pScratch.Get(), m_topLevelASBuffers.pResult.Get(), m_topLevelASBuffers.pInstanceDesc.Get(), updateOnly, m_topLevelASBuffers.pResult.Get());
}

// ═══════════════════════════════════════════════════════════════════
// CreateAccelerationStructures — NEW: uses m_meshes / m_sceneInstances
// ═══════════════════════════════════════════════════════════════════
void Renderer::CreateAccelerationStructures() {
    // Build one BLAS per unique mesh
    for (size_t m = 0; m < m_meshes.size(); ++m) {
        auto& mesh = m_meshes[m];
        AccelerationStructureBuffers buffers = CreateBottomLevelAS(
            {{ mesh.vertexBuffer.Get(), mesh.vertexCount }},
            {{ mesh.indexBuffer.Get(),  mesh.indexCount  }},
            mesh.opaqueTriCount, mesh.alphaTriCount);
        mesh.blas = buffers.pResult;
    }

    // Build TLAS instances (multiple can share one BLAS)
    m_tlasInstances.clear();
    m_tlasInstances.reserve(m_sceneInstances.size());
    for (const auto& si : m_sceneInstances)
        m_tlasInstances.emplace_back(m_meshes[si.meshIndex].blas, si.transform);

    OnInitTransform();
    CreateTopLevelAS(m_tlasInstances);

    CollectEmissiveTriangles();
    std::vector<InstanceXformCPU> ltXforms;
    ltXforms.reserve(m_sceneInstances.size());
    for (const auto& si : m_sceneInstances) ltXforms.push_back(ToInstanceXform(si.transform));
    m_lightTree.Build(m_emissiveTriangles, ltXforms);
    m_lightTree.PrintMetrics();
    { SCOPE_TIMER("LightTree.UploadAll"); m_lightTree.UploadAll(m_device.Get(), m_commandList.Get()); }
    { SCOPE_TIMER("CreateEmissiveTrianglesBuffer"); CreateEmissiveTrianglesBuffer(); }
    { SCOPE_TIMER("CreateTriToLightIdBuffer"); CreateTriToLightIdBuffer(); }

    { SCOPE_TIMER("Execute+Fence (post-LightTree/Uploads)");
      m_commandList->Close();
      ID3D12CommandList* ppCL[] = { m_commandList.Get() };
      m_commandQueue->ExecuteCommandLists(1, ppCL);
      m_fenceValue++; m_commandQueue->Signal(m_fence.Get(), m_fenceValue);
      m_fence->SetEventOnCompletion(m_fenceValue, m_fenceEvent); WaitForSingleObject(m_fenceEvent, INFINITE); }
    m_lightTree.ReleaseStaging();
    ThrowIfFailed(m_commandList->Reset(m_commandAllocators[m_frameIndex].Get(), m_pipelineState.Get()));
}

// Root signatures and pipeline — unchanged from working code
// (These reference no old member variables - they only define descriptor ranges)

ComPtr<ID3D12RootSignature> Renderer::CreateRayGenSignature() {
    CD3DX12_ROOT_PARAMETER1 rootParameters[2];
    std::vector<CD3DX12_DESCRIPTOR_RANGE1> ranges;
    ranges.reserve(40);
    const auto VOLATILE = D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE;
    const auto STATIC = D3D12_DESCRIPTOR_RANGE_FLAG_DATA_STATIC_WHILE_SET_AT_EXECUTE;

    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0, 0, VOLATILE, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 1, 0, VOLATILE, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, STATIC, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 1, 0, STATIC, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 2, 0, STATIC, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_CBV, 1, 0, 0, STATIC, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 3, 0, STATIC, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 4, 0, STATIC, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 5, 0, STATIC, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 6, 0, STATIC, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    for (UINT u = 2; u <= 7; ++u) ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, u, 0, VOLATILE, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 7, 0, STATIC, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 8, 0, STATIC, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 8, 0, VOLATILE, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 9, 0, VOLATILE, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    for (UINT t = 9; t <= 12; ++t) ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, t, 0, STATIC, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 15, 0, STATIC, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    for (UINT t = 16; t <= 18; ++t) ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, t, 0, STATIC, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    for (UINT t = 30; t <= 33; ++t) ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, t, 0, STATIC, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 10, 0, VOLATILE, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 34, 0, VOLATILE, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 35, 0, VOLATILE, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 4, 36, 0, VOLATILE, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 12, 11, 0, VOLATILE, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
    ranges.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 3, 60, 0, VOLATILE, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);

    rootParameters[0].InitAsDescriptorTable(static_cast<UINT>(ranges.size()), ranges.data(), D3D12_SHADER_VISIBILITY_ALL);
    rootParameters[1].InitAsConstants(6, 1, 0, D3D12_SHADER_VISIBILITY_ALL);
    CD3DX12_STATIC_SAMPLER_DESC staticSamplers[2];
    staticSamplers[0].Init(0, D3D12_FILTER_ANISOTROPIC, D3D12_TEXTURE_ADDRESS_MODE_WRAP, D3D12_TEXTURE_ADDRESS_MODE_WRAP, D3D12_TEXTURE_ADDRESS_MODE_WRAP); staticSamplers[0].MaxAnisotropy = 16;
    staticSamplers[1].Init(1, D3D12_FILTER_MIN_MAG_MIP_LINEAR, D3D12_TEXTURE_ADDRESS_MODE_CLAMP, D3D12_TEXTURE_ADDRESS_MODE_CLAMP, D3D12_TEXTURE_ADDRESS_MODE_CLAMP);
    CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC rootSignatureDesc;
    rootSignatureDesc.Init_1_1(_countof(rootParameters), rootParameters, _countof(staticSamplers), staticSamplers, D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED);
    ComPtr<ID3DBlob> signature, error;
    HRESULT hr = D3D12SerializeVersionedRootSignature(&rootSignatureDesc, &signature, &error);
    if (FAILED(hr)) { if (error) OutputDebugStringA(static_cast<char*>(error->GetBufferPointer())); ThrowIfFailed(hr); }
    ComPtr<ID3D12RootSignature> pRS;
    ThrowIfFailed(m_device->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(), IID_PPV_ARGS(&pRS)));
    return pRS;
}

ComPtr<ID3D12RootSignature> Renderer::CreateComputeSignature() {
    // Identical layout to RayGen signature
    return CreateRayGenSignature();
}

ComPtr<ID3D12RootSignature> Renderer::CreateHitSignature() {
    nv_helpers_dx12::RootSignatureGenerator rsc;
    return rsc.Generate(m_device.Get(), true);
}

ComPtr<ID3D12RootSignature> Renderer::CreateMissSignature() {
    nv_helpers_dx12::RootSignatureGenerator rsc;
    return rsc.Generate(m_device.Get(), true);
}

// CreateRaytracingPipeline — unchanged (uses m_passes, no old members)
void Renderer::CreateRaytracingPipeline() {
    nv_helpers_dx12::RayTracingPipelineGenerator pipeline(m_device.Get());
    m_rayGenSignature = CreateRayGenSignature();
    m_computeSignature = CreateComputeSignature();
    m_missSignature = CreateMissSignature();
    m_hitSignature = CreateHitSignature();
    pipeline.SetGlobalRootSignature(m_rayGenSignature.Get());
    m_rayGenLibs.clear(); m_csPSOs.clear(); m_passIndex.clear();
    uint32_t nextCs = 0, rgSlot = 0;

    for (PassDesc& p : m_passes) {
        if (p.stage == Stage::Barrier || p.stage == Stage::LoopStart || p.stage == Stage::LoopEnd || p.stage == Stage::PingSwap || p.stage == Stage::ClearSort || p.stage == Stage::DLSS) continue;
        if (p.isWorkGraph) {
            ComPtr<IDxcBlob> lib = nv_helpers_dx12::CompileWG(p.file.c_str());
            D3D12_DXIL_LIBRARY_DESC dxilDesc{}; dxilDesc.DXILLibrary = { lib->GetBufferPointer(), lib->GetBufferSize() }; dxilDesc.NumExports = 0;
            D3D12_STATE_SUBOBJECT subobjects[3]{};
            subobjects[0].Type = D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY; subobjects[0].pDesc = &dxilDesc;
            subobjects[1].Type = D3D12_STATE_SUBOBJECT_TYPE_GLOBAL_ROOT_SIGNATURE; subobjects[1].pDesc = m_computeSignature.GetAddressOf();
            static const LPCWSTR kWGName = L"main";
            D3D12_WORK_GRAPH_DESC wgDesc = {}; wgDesc.ProgramName = kWGName; wgDesc.Flags = D3D12_WORK_GRAPH_FLAG_INCLUDE_ALL_AVAILABLE_NODES;
            subobjects[2].Type = D3D12_STATE_SUBOBJECT_TYPE_WORK_GRAPH; subobjects[2].pDesc = &wgDesc;
            D3D12_STATE_OBJECT_DESC soDesc{}; soDesc.Type = D3D12_STATE_OBJECT_TYPE_EXECUTABLE; soDesc.NumSubobjects = 3; soDesc.pSubobjects = subobjects;
            ComPtr<ID3D12StateObject> so; ThrowIfFailed(m_device->CreateStateObject(&soDesc, IID_PPV_ARGS(&so)));
            ComPtr<ID3D12StateObjectProperties1> soProps1; ThrowIfFailed(so->QueryInterface(IID_PPV_ARGS(&soProps1)));
            ComPtr<ID3D12WorkGraphProperties> wgProps; ThrowIfFailed(so->QueryInterface(IID_PPV_ARGS(&wgProps)));
            WgRuntimeData rt{}; rt.id = soProps1->GetProgramIdentifier(L"main");
            D3D12_WORK_GRAPH_MEMORY_REQUIREMENTS mem{}; wgProps->GetWorkGraphMemoryRequirements(0, &mem);
            if (mem.MaxSizeInBytes) {
                CD3DX12_HEAP_PROPERTIES hp(D3D12_HEAP_TYPE_DEFAULT);
                auto buf = CD3DX12_RESOURCE_DESC::Buffer(mem.MaxSizeInBytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
                ThrowIfFailed(m_device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &buf, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&rt.backingRes)));
                rt.backing.StartAddress = rt.backingRes->GetGPUVirtualAddress(); rt.backing.SizeInBytes = mem.MaxSizeInBytes;
            }
            p.wgIdx = static_cast<uint32_t>(m_wgStateObjects.size());
            m_wgStateObjects.push_back(so); m_wgProps.push_back(wgProps); m_wgRuntime.push_back(std::move(rt));
            continue;
        }
        if ((p.stage == Stage::Compute || p.stage == Stage::Wavefront || p.stage == Stage::FixedCompute) && !p.isWorkGraph) {
            ComPtr<IDxcBlob> cs = nv_helpers_dx12::CompileCS(p.file.c_str(), L"main");
            D3D12_COMPUTE_PIPELINE_STATE_DESC desc{}; desc.pRootSignature = m_computeSignature.Get(); desc.CS = { cs->GetBufferPointer(), cs->GetBufferSize() };
            ComPtr<ID3D12PipelineState> pso; ThrowIfFailed(m_device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&pso)));
            m_csPSOs.push_back(pso); p.psoIdx = nextCs++; continue;
        }
        if (p.stage == Stage::Callable) {
            std::wstring base = p.file.substr(p.file.find_last_of(L"/\\") + 1); base = base.substr(0, base.rfind(L'.'));
            ComPtr<IDxcBlob> lib = nv_helpers_dx12::CompileShaderLibrary(p.file.c_str());
            pipeline.AddLibrary(lib.Get(), { base.c_str() }); m_callableShaderNames.push_back(base); continue;
        }
        std::wstring base = p.file.substr(p.file.find_last_of(L"/\\") + 1); base = base.substr(0, base.rfind(L'.'));
        ComPtr<IDxcBlob> lib = nv_helpers_dx12::CompileShaderLibrary(p.file.c_str());
        m_rayGenLibs.push_back(lib); pipeline.AddLibrary(lib.Get(), { base.c_str() }); m_passIndex[p.file] = rgSlot++;
    }

    m_missLibrary = nv_helpers_dx12::CompileShaderLibrary(L"Miss_v8.hlsl");
    m_hitLibrary = nv_helpers_dx12::CompileShaderLibrary(L"Hit_v8.hlsl");
    m_shadowLibrary = nv_helpers_dx12::CompileShaderLibrary(L"ShadowRay.hlsl");
    m_anyHitLibrary = nv_helpers_dx12::CompileShaderLibrary(L"AnyHit.hlsl");
    pipeline.AddLibrary(m_missLibrary.Get(), { L"Miss" });
    pipeline.AddLibrary(m_shadowLibrary.Get(), { L"ShadowClosestHit", L"ShadowMiss" });
    pipeline.AddLibrary(m_hitLibrary.Get(), { L"ClosestHit" });
    pipeline.AddLibrary(m_anyHitLibrary.Get(), { L"AlphaTestAnyHit" });
    pipeline.AddHitGroup(L"HitGroup", L"ClosestHit", L"AlphaTestAnyHit");
    pipeline.AddHitGroup(L"ShadowHitGroup", L"ShadowClosestHit", L"AlphaTestAnyHit");
    pipeline.AddRootSignatureAssociation(m_missSignature.Get(), { L"Miss", L"ShadowMiss" });
    pipeline.AddRootSignatureAssociation(m_hitSignature.Get(), { L"HitGroup", L"ShadowHitGroup" });
    pipeline.SetMaxPayloadSize(128); pipeline.SetMaxAttributeSize(2*sizeof(float)); pipeline.SetMaxRecursionDepth(1);
    m_rtStateObject = pipeline.Generate();
    ThrowIfFailed(m_rtStateObject->QueryInterface(IID_PPV_ARGS(&m_rtStateObjectProps)));

    UINT64 rgStackSize = m_rtStateObjectProps->GetShaderStackSize(L"Pass_raygen_v8");
    UINT64 maxCallableStack = 0;
    for (const auto& name : m_callableShaderNames) { UINT64 sz = m_rtStateObjectProps->GetShaderStackSize(name.c_str()); if (sz > maxCallableStack) maxCallableStack = sz; }
    UINT64 missStackSize = m_rtStateObjectProps->GetShaderStackSize(L"Miss");
    UINT64 totalStackSize = (rgStackSize + std::max(maxCallableStack, missStackSize) + 255) & ~255;
    m_rtStateObjectProps->SetPipelineStackSize(totalStackSize);
}

// CreateRaytracingOutputBuffer — unchanged
void Renderer::CreateRaytracingOutputBuffer() {
    D3D12_RESOURCE_DESC resDesc = {}; resDesc.DepthOrArraySize = 60; resDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    resDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM; resDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    resDesc.Width = GetWidth(); resDesc.Height = GetHeight(); resDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN; resDesc.MipLevels = 1; resDesc.SampleDesc.Count = 1;
    ThrowIfFailed(m_device->CreateCommittedResource(&nv_helpers_dx12::kDefaultHeapProps, D3D12_HEAP_FLAG_NONE, &resDesc, D3D12_RESOURCE_STATE_COPY_SOURCE, nullptr, IID_PPV_ARGS(&m_outputResource)));

    D3D12_RESOURCE_DESC td = {}; td.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D; td.Width = GetWidth(); td.Height = GetHeight();
    td.DepthOrArraySize = 1; td.MipLevels = 1; td.Format = DXGI_FORMAT_R32G32B32A32_FLOAT; td.SampleDesc.Count = 1; td.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    ThrowIfFailed(m_device->CreateCommittedResource(&nv_helpers_dx12::kDefaultHeapProps, D3D12_HEAP_FLAG_NONE, &td, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&m_permanentDataTexture)));

    D3D12_RESOURCE_DESC desc = {}; desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D; desc.Width = GetWidth(); desc.Height = GetHeight();
    desc.DepthOrArraySize = 16; desc.MipLevels = 1; desc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT; desc.SampleDesc.Count = 1; desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    ThrowIfFailed(m_device->CreateCommittedResource(&nv_helpers_dx12::kDefaultHeapProps, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&m_scratchPing)));

    auto CreateRawBuffer = [&](ComPtr<ID3D12Resource>& resource, UINT sizeInBytes, const std::wstring& name) {
        auto bd = CD3DX12_RESOURCE_DESC::Buffer(sizeInBytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        ThrowIfFailed(m_device->CreateCommittedResource(&nv_helpers_dx12::kDefaultHeapProps, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&resource)));
        resource->SetName(name.c_str());
    };
    UINT px = GetWidth() * GetHeight();
    CreateRawBuffer(m_reservoirBuffer,      px*sizeof(Reservoir_DI), L"ReservoirBuffer_DI_1");
    CreateRawBuffer(m_reservoirBuffer_2,    px*sizeof(Reservoir_DI), L"ReservoirBuffer_DI_2");
    CreateRawBuffer(m_reservoirBuffer_3,    px*sizeof(Reservoir_GI), L"ReservoirBuffer_GI_1");
    CreateRawBuffer(m_reservoirBuffer_4,    px*sizeof(Reservoir_GI), L"ReservoirBuffer_GI_2");
    CreateRawBuffer(m_sampleBuffer_current, px*sizeof(SampleData),   L"SampleBuffer_Current");
    CreateRawBuffer(m_sampleBuffer_last,    px*sizeof(SampleData),   L"SampleBuffer_Last");
    CreateRawBuffer(m_initialBSDFRayBuffer, px*sizeof(InitialBSDFRay), L"InitialBSDFRayBuffer");
    CreatePathStateBuffer();
}

// CreateShaderResourceHeap — uses m_sceneInstances.size() instead of m_instances.size()
void Renderer::CreateShaderResourceHeap() {
    m_srvUavHeap = nv_helpers_dx12::CreateDescriptorHeap(m_device.Get(), 1000000, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, true);
    D3D12_DESCRIPTOR_HEAP_DESC stagingDesc = {}; stagingDesc.NumDescriptors = 4; stagingDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV; stagingDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
    ThrowIfFailed(m_device->CreateDescriptorHeap(&stagingDesc, IID_PPV_ARGS(&m_stagingUavHeap)));
    CD3DX12_CPU_DESCRIPTOR_HANDLE handle(m_srvUavHeap->GetCPUDescriptorHandleForHeapStart());
    CD3DX12_GPU_DESCRIPTOR_HANDLE gpuHandle(m_srvUavHeap->GetGPUDescriptorHandleForHeapStart());
    const UINT inc = m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    CD3DX12_CPU_DESCRIPTOR_HANDLE stagingHandle(m_stagingUavHeap->GetCPUDescriptorHandleForHeapStart());
    auto nextSlot = [&]() { handle.Offset(1, inc); gpuHandle.Offset(1, inc); };
    auto nextStaging = [&]() { stagingHandle.Offset(1, inc); };
    auto createNullSRV = [&](D3D12_SRV_DIMENSION dim = D3D12_SRV_DIMENSION_BUFFER) {
        D3D12_SHADER_RESOURCE_VIEW_DESC sd = {}; sd.Format = DXGI_FORMAT_R32_UINT; sd.ViewDimension = dim; sd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        if(dim == D3D12_SRV_DIMENSION_BUFFER) sd.Buffer.NumElements = 1;
        m_device->CreateShaderResourceView(nullptr, &sd, handle); nextSlot();
    };

    // Slot 0: UAV u0
    { D3D12_UNORDERED_ACCESS_VIEW_DESC ud = {}; ud.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY; ud.Format = DXGI_FORMAT_R8G8B8A8_UNORM; ud.Texture2DArray.ArraySize = m_outputResource->GetDesc().DepthOrArraySize; m_device->CreateUnorderedAccessView(m_outputResource.Get(), nullptr, &ud, handle); nextSlot(); }
    // Slot 1: UAV u1
    { D3D12_UNORDERED_ACCESS_VIEW_DESC ud = {}; ud.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D; ud.Format = DXGI_FORMAT_R32G32B32A32_FLOAT; m_device->CreateUnorderedAccessView(m_permanentDataTexture.Get(), nullptr, &ud, handle); nextSlot(); }
    // Slot 2: SRV t0 (TLAS)
    { D3D12_SHADER_RESOURCE_VIEW_DESC sd = {}; sd.Format = DXGI_FORMAT_UNKNOWN; sd.ViewDimension = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE; sd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING; sd.RaytracingAccelerationStructure.Location = m_topLevelASBuffers.pResult->GetGPUVirtualAddress(); m_device->CreateShaderResourceView(nullptr, &sd, handle); nextSlot(); }
    // Slot 3: SRV t1 (Global IB)
    { D3D12_SHADER_RESOURCE_VIEW_DESC sd = {}; sd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING; sd.Format = DXGI_FORMAT_R32_UINT; sd.ViewDimension = D3D12_SRV_DIMENSION_BUFFER; sd.Buffer.NumElements = m_totalIndexCount; m_device->CreateShaderResourceView(m_indexGlobal.Get(), &sd, handle); nextSlot(); }
    // Slot 4: SRV t2 (Global VB)
    { D3D12_SHADER_RESOURCE_VIEW_DESC sd = {}; sd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING; sd.Format = DXGI_FORMAT_UNKNOWN; sd.ViewDimension = D3D12_SRV_DIMENSION_BUFFER; sd.Buffer.NumElements = m_totalVertexCount; sd.Buffer.StructureByteStride = sizeof(BTriVertex); m_device->CreateShaderResourceView(m_vertexGlobal.Get(), &sd, handle); nextSlot(); }
    // Slot 5: CBV b0
    { D3D12_CONSTANT_BUFFER_VIEW_DESC cd = {}; cd.BufferLocation = m_cameraBuffer->GetGPUVirtualAddress(); cd.SizeInBytes = m_cameraBufferSize; m_device->CreateConstantBufferView(&cd, handle); nextSlot(); }
    // Slot 6: SRV t3 (Instance Props) *** KEY CHANGE ***
    { D3D12_SHADER_RESOURCE_VIEW_DESC sd = {}; sd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING; sd.Format = DXGI_FORMAT_UNKNOWN; sd.ViewDimension = D3D12_SRV_DIMENSION_BUFFER; sd.Buffer.NumElements = static_cast<UINT>(m_sceneInstances.size()); sd.Buffer.StructureByteStride = sizeof(InstanceProperties); m_device->CreateShaderResourceView(m_instanceProperties.Get(), &sd, handle); nextSlot(); }
    // Slot 7: SRV t4 (Material IDs)
    { D3D12_SHADER_RESOURCE_VIEW_DESC sd = {}; sd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING; sd.Format = DXGI_FORMAT_R32_UINT; sd.ViewDimension = D3D12_SRV_DIMENSION_BUFFER; sd.Buffer.NumElements = static_cast<UINT>(m_materialIDs.size()); m_device->CreateShaderResourceView(m_materialIndexBuffer.Get(), &sd, handle); nextSlot(); }
    // Slot 8: SRV t5 (Materials)
    { D3D12_SHADER_RESOURCE_VIEW_DESC sd = {}; sd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING; sd.Format = DXGI_FORMAT_UNKNOWN; sd.ViewDimension = D3D12_SRV_DIMENSION_BUFFER; sd.Buffer.NumElements = static_cast<UINT>(m_materials.size()); sd.Buffer.StructureByteStride = sizeof(Material); m_device->CreateShaderResourceView(m_materialBuffer.Get(), &sd, handle); nextSlot(); }
    // Slot 9: SRV t6 (Emissive Triangles)
    { if (m_emissiveTrianglesBuffer) { D3D12_SHADER_RESOURCE_VIEW_DESC sd = {}; sd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING; sd.Format = DXGI_FORMAT_UNKNOWN; sd.ViewDimension = D3D12_SRV_DIMENSION_BUFFER; sd.Buffer.NumElements = static_cast<UINT>(m_emissiveTriangles.size()); sd.Buffer.StructureByteStride = sizeof(LightTriangle); m_device->CreateShaderResourceView(m_emissiveTrianglesBuffer.Get(), &sd, handle); } else createNullSRV(); nextSlot(); }

    auto createRawUAV = [&](ComPtr<ID3D12Resource>& res, UINT sizeInBytes) { D3D12_UNORDERED_ACCESS_VIEW_DESC ud = {}; ud.ViewDimension = D3D12_UAV_DIMENSION_BUFFER; ud.Format = DXGI_FORMAT_R32_TYPELESS; ud.Buffer.NumElements = sizeInBytes/4; ud.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW; m_device->CreateUnorderedAccessView(res.Get(), nullptr, &ud, handle); nextSlot(); };
    UINT rDI = GetWidth()*GetHeight()*sizeof(Reservoir_DI), rGI = GetWidth()*GetHeight()*sizeof(Reservoir_GI), sB = GetWidth()*GetHeight()*sizeof(SampleData);
    createRawUAV(m_reservoirBuffer, rDI); createRawUAV(m_reservoirBuffer_2, rDI);
    createRawUAV(m_reservoirBuffer_3, rGI); createRawUAV(m_reservoirBuffer_4, rGI);
    createRawUAV(m_sampleBuffer_current, sB); createRawUAV(m_sampleBuffer_last, sB);
    createNullSRV(); createNullSRV(); // slots 16-17 alias placeholders
    { D3D12_UNORDERED_ACCESS_VIEW_DESC ud = {}; ud.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY; ud.Format = DXGI_FORMAT_R32G32B32A32_FLOAT; ud.Texture2DArray.ArraySize = 16; m_device->CreateUnorderedAccessView(m_scratchPing.Get(), nullptr, &ud, handle); nextSlot(); }
    createRawUAV(m_initialBSDFRayBuffer, GetWidth()*GetHeight()*sizeof(InitialBSDFRay));
    m_lightTree.WriteSrvs(m_device.Get(), handle); handle.Offset(4, inc); gpuHandle.Offset(4, inc);
    { if (m_triToLightIdBuffer) { D3D12_SHADER_RESOURCE_VIEW_DESC sd = {}; sd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING; sd.Format = DXGI_FORMAT_R32_UINT; sd.ViewDimension = D3D12_SRV_DIMENSION_BUFFER; sd.Buffer.NumElements = static_cast<UINT>(m_triToLightId.size()); m_device->CreateShaderResourceView(m_triToLightIdBuffer.Get(), &sd, handle); } else createNullSRV(); nextSlot(); }
    m_lightTree.WriteLookupSrvs(m_device.Get(), handle); handle.Offset(3, inc); gpuHandle.Offset(3, inc);

    { auto createNullTex2D = [&]() { D3D12_SHADER_RESOURCE_VIEW_DESC d = {}; d.Format = DXGI_FORMAT_R8G8B8A8_UNORM; d.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D; d.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING; d.Texture2D.MipLevels = 1; m_device->CreateShaderResourceView(nullptr, &d, handle); nextSlot(); };
      createNullTex2D(); createNullTex2D(); createNullTex2D(); }
    { auto& res = m_lutTextureArray; if (res) { D3D12_SHADER_RESOURCE_VIEW_DESC d = {}; d.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING; d.Format = res->GetDesc().Format; d.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY; d.Texture2DArray.MipLevels = res->GetDesc().MipLevels; d.Texture2DArray.ArraySize = res->GetDesc().DepthOrArraySize; m_device->CreateShaderResourceView(res.Get(), &d, handle); nextSlot(); } else { createNullSRV(D3D12_SRV_DIMENSION_TEXTURE2DARRAY); } }

    createRawUAV(m_pathStateBuffer, GetWidth()*GetHeight()*88);
    { D3D12_UNORDERED_ACCESS_VIEW_DESC ud = {}; ud.ViewDimension = D3D12_UAV_DIMENSION_BUFFER; ud.Format = DXGI_FORMAT_R32_TYPELESS; ud.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW; ud.Buffer.NumElements = MAX_STACKS; m_device->CreateUnorderedAccessView(m_globalCounterBuffer.Get(), nullptr, &ud, handle); nextSlot(); }
    { D3D12_UNORDERED_ACCESS_VIEW_DESC ud = {}; ud.ViewDimension = D3D12_UAV_DIMENSION_BUFFER; ud.Format = DXGI_FORMAT_UNKNOWN; ud.Buffer.NumElements = MAX_INDIRECT_COMMANDS; ud.Buffer.StructureByteStride = sizeof(D3D12_DISPATCH_ARGUMENTS); m_device->CreateUnorderedAccessView(m_indirectArgsBuffer.Get(), nullptr, &ud, handle); nextSlot(); }
    for(int i=0; i<4; ++i) { D3D12_UNORDERED_ACCESS_VIEW_DESC ud = {}; ud.ViewDimension = D3D12_UAV_DIMENSION_BUFFER; ud.Format = DXGI_FORMAT_UNKNOWN; ud.Buffer.NumElements = GetWidth()*GetHeight(); ud.Buffer.StructureByteStride = 8; m_device->CreateUnorderedAccessView(m_stackBuffers[i].Get(), nullptr, &ud, handle); nextSlot(); }

    { auto createUAV = [&](ID3D12Resource* res, DXGI_FORMAT fmt) { D3D12_UNORDERED_ACCESS_VIEW_DESC ud = {}; ud.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D; ud.Format = fmt; m_device->CreateUnorderedAccessView(res, nullptr, &ud, handle); nextSlot(); };
      createUAV(m_dlssDepth.Get(), DXGI_FORMAT_R32_FLOAT); createUAV(m_dlssMVec.Get(), DXGI_FORMAT_R16G16_FLOAT);
      createUAV(m_dlssNormals.Get(), DXGI_FORMAT_R16G16B16A16_FLOAT); createUAV(m_dlssDiffuseAlbedo.Get(), DXGI_FORMAT_R16G16B16A16_FLOAT);
      createUAV(m_dlssOutput.Get(), DXGI_FORMAT_R16G16B16A16_FLOAT); createUAV(m_dlssSpecularAlbedo.Get(), DXGI_FORMAT_R16G16B16A16_FLOAT);
      createUAV(m_dlssRoughness.Get(), DXGI_FORMAT_R16_FLOAT); createUAV(m_dlssSpecMVec.Get(), DXGI_FORMAT_R16G16_FLOAT);
      createUAV(m_dlssSpecHitDist.Get(), DXGI_FORMAT_R16_FLOAT); createUAV(m_dlssTransparency.Get(), DXGI_FORMAT_R16G16B16A16_FLOAT);
      createUAV(m_dlssColorBeforeTrans.Get(), DXGI_FORMAT_R16G16B16A16_FLOAT); createUAV(m_dlssInput.Get(), DXGI_FORMAT_R16G16B16A16_FLOAT); }

    auto makeSortUAV = [&](ComPtr<ID3D12Resource>& buf, UINT numElements) {
        D3D12_UNORDERED_ACCESS_VIEW_DESC ud = {}; ud.ViewDimension = D3D12_UAV_DIMENSION_BUFFER; ud.Format = DXGI_FORMAT_R32_TYPELESS; ud.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW; ud.Buffer.NumElements = numElements;
        m_device->CreateUnorderedAccessView(buf.Get(), nullptr, &ud, stagingHandle);
        auto cpuH = stagingHandle; m_device->CopyDescriptorsSimple(1, handle, stagingHandle, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        auto gpuH = gpuHandle; nextSlot(); nextStaging(); return std::make_pair(cpuH, gpuH);
    };
    { auto [c,g] = makeSortUAV(m_sortCountBuffer, SORT_BUCKETS); m_sortCountCpuHandle = c; m_sortCountGpuHandle = g; }
    { auto [c,g] = makeSortUAV(m_sortOffsetBuffer, SORT_BUCKETS); m_sortOffsetCpuHandle = c; m_sortOffsetGpuHandle = g; }
    { auto [c,g] = makeSortUAV(m_sortBoundsBuffer, 8); m_sortBoundsCpuHandle = c; m_sortBoundsGpuHandle = g; }

    { UINT globalTexIdx = 0;
      auto writeBindlessBatch = [&](UINT heapBase, UINT count) { for (UINT i = 0; i < count; ++i) { CD3DX12_CPU_DESCRIPTOR_HANDLE dst(m_srvUavHeap->GetCPUDescriptorHandleForHeapStart(), heapBase+i, inc); auto* res = m_bindlessGpuTextures[globalTexIdx].Get(); D3D12_SHADER_RESOURCE_VIEW_DESC srv = {}; srv.Format = res->GetDesc().Format; srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D; srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING; srv.Texture2D.MipLevels = res->GetDesc().MipLevels; m_device->CreateShaderResourceView(res, &srv, dst); globalTexIdx++; } };
      UINT albedoCount = m_bindlessNormalBase - m_bindlessAlbedoBase, normalCount = m_bindlessRmaBase - m_bindlessNormalBase, rmaCount = m_totalBindlessTextures - albedoCount - normalCount;
      writeBindlessBatch(m_bindlessAlbedoBase, albedoCount); writeBindlessBatch(m_bindlessNormalBase, normalCount); writeBindlessBatch(m_bindlessRmaBase, rmaCount); }
}

// ═══════════════════════════════════════════════════════════════════
// CreateShaderBindingTable — uses m_sceneInstances.size()
// ═══════════════════════════════════════════════════════════════════
void Renderer::CreateShaderBindingTable() {
    m_sbtHelper.Reset();
    D3D12_GPU_DESCRIPTOR_HANDLE heapHandle = m_srvUavHeap->GetGPUDescriptorHandleForHeapStart();
    auto heapPointer = reinterpret_cast<UINT64*>(heapHandle.ptr);
    for (const auto& entry : m_passSequence) {
        if (entry == L"barrier" || entry.rfind(L"loop:", 0) == 0 || entry == L"endloop" || entry == L"pingswap" || entry == L"clearsort" || entry == L"dlss") continue;
        if (entry.find(L"|cs:") != std::wstring::npos || entry.find(L"|wf:") != std::wstring::npos || entry.find(L"|wg:") != std::wstring::npos || entry.find(L"|fx:") != std::wstring::npos || entry.find(L"|call") != std::wstring::npos) continue;
        std::wstring base = entry.substr(entry.find_last_of(L"/\\") + 1); base = base.substr(0, base.rfind(L'.'));
        m_sbtHelper.AddRayGenerationProgram(base.c_str(), { heapPointer });
    }
    m_sbtHelper.AddMissProgram(L"Miss", {}); m_sbtHelper.AddMissProgram(L"ShadowMiss", {});
    for (size_t i = 0; i < m_sceneInstances.size(); ++i) { m_sbtHelper.AddHitGroup(L"HitGroup", {}); m_sbtHelper.AddHitGroup(L"ShadowHitGroup", {}); }
    m_sbtHelper.AddHitGroup(L"ShadowHitGroup", {});
    for (const auto& name : m_callableShaderNames) m_sbtHelper.AddCallableProgram(name, { heapPointer });
    uint32_t sbtSize = m_sbtHelper.ComputeSBTSize();
    m_sbtStorage = nv_helpers_dx12::CreateBuffer(m_device.Get(), sbtSize, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ, nv_helpers_dx12::kUploadHeapProps);
    if (!m_sbtStorage) throw std::logic_error("Could not allocate the shader binding table");
    m_sbtHelper.Generate(m_sbtStorage.Get(), m_rtStateObjectProps.Get());
}

// Camera
void Renderer::CreateCameraBuffer() {
    uint32_t nbMatrix = 6; m_cameraBufferSize = nbMatrix * sizeof(XMMATRIX) + sizeof(float) * 4;
    m_cameraBufferSize = (m_cameraBufferSize + 255) & ~255;
    m_cameraBuffer = nv_helpers_dx12::CreateBuffer(m_device.Get(), m_cameraBufferSize, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ, nv_helpers_dx12::kUploadHeapProps);
    m_constHeap = nv_helpers_dx12::CreateDescriptorHeap(m_device.Get(), 2, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, true);
}

void Renderer::UpdateCameraBuffer() {
    std::vector<XMMATRIX> matrices(6);
    const glm::mat4& viewMat = nv_helpers_dx12::CameraManip.getMatrix();
    memcpy(&matrices[0].r->m128_f32[0], glm::value_ptr(viewMat), 16 * sizeof(float));
    matrices[1] = XMMatrixPerspectiveFovRH(60.0f * XM_PI / 180.0f, m_aspectRatio, 0.00001f, 10000.0f);
    XMVECTOR det;
    matrices[2] = XMMatrixInverse(&det, matrices[0]); matrices[3] = XMMatrixInverse(&det, matrices[1]);
    matrices[4] = m_prevViewMatrix; matrices[5] = m_prevProjMatrix;
    m_jitterFrameIndex++; m_jitterX = Halton(m_jitterFrameIndex%16+1, 2) - 0.5f; m_jitterY = Halton(m_jitterFrameIndex%16+1, 3) - 0.5f;
    uint8_t* pData; if (FAILED(m_cameraBuffer->Map(0, nullptr, (void**)&pData))) return;
    memcpy(pData, matrices.data(), 6 * sizeof(XMMATRIX));
    float extraData[4] = { static_cast<float>(m_jitterFrameIndex), m_jitterX, m_jitterY, 0.0f };
    memcpy(pData + 6*sizeof(XMMATRIX), extraData, sizeof(extraData));
    m_cameraBuffer->Unmap(0, nullptr);
    m_prevViewMatrix = matrices[0]; m_prevProjMatrix = matrices[1];
}

// ═══════════════════════════════════════════════════════════════════
// Instance Properties — NEW: uses m_sceneInstances / m_meshes
// ═══════════════════════════════════════════════════════════════════
void Renderer::CreateInstancePropertiesBuffer() {
    uint32_t bufferSize = ROUND_UP(static_cast<uint32_t>(m_sceneInstances.size()) * sizeof(InstanceProperties), D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);
    m_instanceProperties = nv_helpers_dx12::CreateBuffer(m_device.Get(), bufferSize, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ, nv_helpers_dx12::kUploadHeapProps);
}

void Renderer::UpdateInstancePropertiesBuffer() {
    InstanceProperties* dst = nullptr; CD3DX12_RANGE r(0,0);
    ThrowIfFailed(m_instanceProperties->Map(0, &r, reinterpret_cast<void**>(&dst)));
    for (size_t i = 0; i < m_sceneInstances.size(); ++i, ++dst) {
        auto& si = m_sceneInstances[i];
        const XMMATRIX& M = si.transform; XMVECTOR det;
        dst->prevObjectToWorld = dst->objectToWorld;
        dst->prevObjectToWorldInverse = XMMatrixInverse(&det, dst->objectToWorld);
        dst->objectToWorld = M; dst->objectToWorldInverse = XMMatrixInverse(&det, M);
        XMMATRIX upper3x3 = M;
        upper3x3.r[0].m128_f32[3] = upper3x3.r[1].m128_f32[3] = upper3x3.r[2].m128_f32[3] = upper3x3.r[3].m128_f32[0] = upper3x3.r[3].m128_f32[1] = upper3x3.r[3].m128_f32[2] = 0.f; upper3x3.r[3].m128_f32[3] = 1.f;
        dst->prevObjectToWorldNormal = dst->objectToWorldNormal;
        dst->objectToWorldNormal = XMMatrixTranspose(XMMatrixInverse(&det, upper3x3));
        const MeshGPU& mesh = m_meshes[si.meshIndex];
        dst->opaqueTriCount = mesh.opaqueTriCount;
        dst->indexBase = mesh.globalIndexBase; dst->vertexBase = mesh.globalVertexBase;
        dst->materialBase = mesh.materialIDBase; dst->triToLightBase = m_instTriOffset[i];
        m_tlasInstances[i].second = M;
    }
    m_instanceProperties->Unmap(0, nullptr);
}

// ═══════════════════════════════════════════════════════════════════
// CollectEmissiveTriangles — NEW: uses m_sceneInstances / m_meshes CPU data
// ═══════════════════════════════════════════════════════════════════
void Renderer::CollectEmissiveTriangles() {
    m_emissiveTriangles.clear();
    m_instTriOffset.resize(m_sceneInstances.size());
    size_t totalTris = 0;
    for (size_t inst = 0; inst < m_sceneInstances.size(); ++inst)
        totalTris += m_meshes[m_sceneInstances[inst].meshIndex].indexCount / 3;
    m_triToLightId.assign(totalTris, 0xFFFFFFFFu);
    uint32_t runningBase = 0;
    for (size_t inst = 0; inst < m_sceneInstances.size(); ++inst) {
        m_instTriOffset[inst] = runningBase;
        runningBase += m_meshes[m_sceneInstances[inst].meshIndex].indexCount / 3;
    }
    for (size_t inst = 0; inst < m_sceneInstances.size(); ++inst) {
        const auto& si = m_sceneInstances[inst];
        const MeshGPU& mesh = m_meshes[si.meshIndex];
        const UINT triCount = mesh.indexCount / 3;
        const uint32_t triBase = m_instTriOffset[inst];
        for (UINT t = 0; t < triCount; ++t) {
            UINT mid = mesh.cpuMaterialIDs[t];
            const Material& mat = m_materials[mid];
            if (mat.Ke.x + mat.Ke.y + mat.Ke.z > 0.0f) {
                LightTriangle lt{};
                lt.x = mesh.cpuVertices[mesh.cpuIndices[3*t+0]].position;
                lt.y = mesh.cpuVertices[mesh.cpuIndices[3*t+1]].position;
                lt.z = mesh.cpuVertices[mesh.cpuIndices[3*t+2]].position;
                lt.instanceID = static_cast<UINT>(inst);
                lt.weight = ComputeTriangleWeight(lt.x, lt.y, lt.z, mat.Ke, si.transform);
                lt.emission = mat.Ke;
                m_triToLightId[triBase + t] = static_cast<uint32_t>(m_emissiveTriangles.size());
                m_emissiveTriangles.push_back(lt);
            }
        }
    }
    if (m_emissiveTriangles.empty()) { LightTriangle d = {}; m_emissiveTriangles.push_back(d); }
    std::wcout << L"Emissive Triangles: " << m_emissiveTriangles.size() << std::endl;
}

static inline float Luminance(const XMFLOAT3& c) { return 0.2126f*c.x + 0.7152f*c.y + 0.0722f*c.z; }

float Renderer::ComputeTriangleWeight(const XMFLOAT3& v0, const XMFLOAT3& v1, const XMFLOAT3& v2, const XMFLOAT3& emissiveColor, const XMMATRIX& M) {
    XMVECTOR p0 = XMVector3TransformCoord(XMLoadFloat3(&v0), M), p1 = XMVector3TransformCoord(XMLoadFloat3(&v1), M), p2 = XMVector3TransformCoord(XMLoadFloat3(&v2), M);
    float area = 0.5f * XMVectorGetX(XMVector3Length(XMVector3Cross(p1-p0, p2-p0)));
    return std::max(area, 1e-10f) * Luminance(emissiveColor);
}

void Renderer::CreateEmissiveTrianglesBuffer() {
    size_t bufferSize = m_emissiveTriangles.size() * sizeof(LightTriangle);
    for (auto& t : m_emissiveTriangles) t.triCount = static_cast<UINT>(m_emissiveTriangles.size());
    ComPtr<ID3D12Resource> upload = nv_helpers_dx12::CreateBuffer(m_device.Get(), static_cast<UINT>(bufferSize), D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ, nv_helpers_dx12::kUploadHeapProps);
    { LightTriangle* p = nullptr; CD3DX12_RANGE rr(0,0); ThrowIfFailed(upload->Map(0, &rr, (void**)&p)); memcpy(p, m_emissiveTriangles.data(), bufferSize); upload->Unmap(0, nullptr); }
    m_emissiveTrianglesBuffer = nv_helpers_dx12::CreateBuffer(m_device.Get(), static_cast<UINT>(bufferSize), D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST, nv_helpers_dx12::kDefaultHeapProps);
    m_commandList->CopyBufferRegion(m_emissiveTrianglesBuffer.Get(), 0, upload.Get(), 0, bufferSize);
    auto barrier = CD3DX12_RESOURCE_BARRIER::Transition(m_emissiveTrianglesBuffer.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_GENERIC_READ);
    m_commandList->ResourceBarrier(1, &barrier);
    ThrowIfFailed(m_commandList->Close());
    ID3D12CommandList* ppCL[] = { m_commandList.Get() }; m_commandQueue->ExecuteCommandLists(1, ppCL);
    WaitForPreviousFrame();
    ThrowIfFailed(m_commandAllocators[m_frameIndex]->Reset());
    ThrowIfFailed(m_commandList->Reset(m_commandAllocators[m_frameIndex].Get(), nullptr));
}

void Renderer::ExtractFrustumPlanes(const XMMATRIX& vp, XMFLOAT4* planes) {
    XMVECTOR r1=vp.r[0], r2=vp.r[1], r3=vp.r[2], r4=vp.r[3];
    planes[0] = XMFLOAT4(XMVectorGetX(r4)+XMVectorGetX(r1), XMVectorGetY(r4)+XMVectorGetY(r1), XMVectorGetZ(r4)+XMVectorGetZ(r1), XMVectorGetW(r4)+XMVectorGetW(r1));
    planes[1] = XMFLOAT4(XMVectorGetX(r4)-XMVectorGetX(r1), XMVectorGetY(r4)-XMVectorGetY(r1), XMVectorGetZ(r4)-XMVectorGetZ(r1), XMVectorGetW(r4)-XMVectorGetW(r1));
    planes[2] = XMFLOAT4(XMVectorGetX(r4)+XMVectorGetX(r2), XMVectorGetY(r4)+XMVectorGetY(r2), XMVectorGetZ(r4)+XMVectorGetZ(r2), XMVectorGetW(r4)+XMVectorGetW(r2));
    planes[3] = XMFLOAT4(XMVectorGetX(r4)-XMVectorGetX(r2), XMVectorGetY(r4)-XMVectorGetY(r2), XMVectorGetZ(r4)-XMVectorGetZ(r2), XMVectorGetW(r4)-XMVectorGetW(r2));
    for (int i = 0; i < 4; i++) { float len = XMVectorGetX(XMVector3Length(XMVectorSet(planes[i].x, planes[i].y, planes[i].z, 0))); if (len != 0) { planes[i].x/=len; planes[i].y/=len; planes[i].z/=len; planes[i].w/=len; } }
}

void Renderer::OnButtonDown(UINT32 lParam) { nv_helpers_dx12::CameraManip.setMousePosition(-GET_X_LPARAM(lParam), -GET_Y_LPARAM(lParam)); }
void Renderer::OnMouseMove(UINT8 wParam, UINT32 lParam) {
    using nv_helpers_dx12::Manipulator; Manipulator::Inputs inputs;
    inputs.lmb = wParam & MK_LBUTTON; inputs.mmb = wParam & MK_MBUTTON; inputs.rmb = wParam & MK_RBUTTON;
    if (!inputs.lmb && !inputs.rmb && !inputs.mmb) return;
    inputs.ctrl = GetAsyncKeyState(VK_CONTROL); inputs.shift = GetAsyncKeyState(VK_SHIFT); inputs.alt = GetAsyncKeyState(VK_MENU);
    CameraManip.mouseMove(-GET_X_LPARAM(lParam), -GET_Y_LPARAM(lParam), inputs);
}

void Renderer::CreateGlobalConstantBuffer() {
    XMVECTOR bd[] = { {1,0,0,1},{.7f,.4f,0,1},{.4f,.7f,0,1},{0,1,0,1},{0,.7f,.4f,1},{0,.4f,.7f,1},{0,0,1,1},{.4f,0,.7f,1},{.7f,0,.4f,1} };
    m_globalConstantBuffer = nv_helpers_dx12::CreateBuffer(m_device.Get(), sizeof(bd), D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ, nv_helpers_dx12::kUploadHeapProps);
    uint8_t* p; ThrowIfFailed(m_globalConstantBuffer->Map(0, nullptr, (void**)&p)); memcpy(p, bd, sizeof(bd)); m_globalConstantBuffer->Unmap(0, nullptr);
}

void Renderer::CreatePerInstanceConstantBuffers() {
    XMVECTOR bd[] = { {1,0,0,1},{1,.4f,0,1},{1,.7f,0,1},{0,1,0,1},{0,1,.4f,1},{0,1,.7f,1},{0,0,1,1},{.4f,0,1,1},{.7f,0,1,1} };
    m_perInstanceConstantBuffers.resize(3); int i = 0;
    for (auto& cb : m_perInstanceConstantBuffers) {
        cb = nv_helpers_dx12::CreateBuffer(m_device.Get(), sizeof(XMVECTOR)*3, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ, nv_helpers_dx12::kUploadHeapProps);
        uint8_t* p; ThrowIfFailed(cb->Map(0, nullptr, (void**)&p)); memcpy(p, &bd[i*3], sizeof(XMVECTOR)*3); cb->Unmap(0, nullptr); ++i;
    }
}

void Renderer::CreateDepthBuffer() {
    m_dsvHeap = nv_helpers_dx12::CreateDescriptorHeap(m_device.Get(), 1, D3D12_DESCRIPTOR_HEAP_TYPE_DSV, false);
    auto dhp = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
    auto drd = CD3DX12_RESOURCE_DESC::Tex2D(DXGI_FORMAT_D32_FLOAT, m_width, m_height, 1, 1); drd.Flags |= D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    CD3DX12_CLEAR_VALUE cv(DXGI_FORMAT_D32_FLOAT, 1.0f, 0);
    ThrowIfFailed(m_device->CreateCommittedResource(&dhp, D3D12_HEAP_FLAG_NONE, &drd, D3D12_RESOURCE_STATE_DEPTH_WRITE, &cv, IID_PPV_ARGS(&m_depthStencil)));
    D3D12_DEPTH_STENCIL_VIEW_DESC dsvDesc = {}; dsvDesc.Format = DXGI_FORMAT_D32_FLOAT; dsvDesc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
    m_device->CreateDepthStencilView(m_depthStencil.Get(), &dsvDesc, m_dsvHeap->GetCPUDescriptorHandleForHeapStart());
}

void Renderer::GenerateLutTextures() {
    SCOPE_TIMER("GenerateLutTextures");
    std::vector<std::vector<float>> allLutData(NUM_LUTS, std::vector<float>(LUT_RESOLUTION * LUT_RESOLUTION));
    const XMFLOAT3 N = {0,0,1}; Material tempMat;
    std::random_device rd; std::mt19937 gen(rd()); std::uniform_real_distribution<float> dist(0,1);
    for (int y = 0; y < LUT_RESOLUTION; ++y) {
        float cosTheta = std::max(0.01f, (float)y / (LUT_RESOLUTION-1)); float sinTheta = sqrt(1-cosTheta*cosTheta);
        const XMFLOAT3 V = {sinTheta, 0, cosTheta};
        for (int x = 0; x < LUT_RESOLUTION; ++x) {
            float roughness = std::max(0.01f, (float)x / (LUT_RESOLUTION-1)); tempMat.Pr_Pm_Ps_Pc.x = roughness;
            size_t pi = (size_t)y * LUT_RESOLUTION + x;
            allLutData[0][pi] = ComputeSheenDirectionalAlbedo(N, V, roughness, NUM_SAMPLES_LUT);
            float Ess = 0; for (int i = 0; i < NUM_SAMPLES_LUT; ++i) { XMFLOAT3 L; SampleGGX(tempMat, V, N, L, dist(gen), dist(gen)); if (dot(N,L)<=0) continue; XMFLOAT3 b = EvaluateBRDF_GGX(V,L,N,{},roughness); float pdf = BRDF_PDF_GGX(roughness,N,L*-1.0f,V); if (pdf>1e-6f) Ess += (b.x*dot(N,L))/pdf; }
            allLutData[1][pi] = NUM_SAMPLES_LUT > 0 ? Ess/NUM_SAMPLES_LUT : 0;
        }
    }
    PrintFullLutMatrix(allLutData[0], L"Sheen Directional Albedo LUT");
    PrintFullLutMatrix(allLutData[1], L"GGX E(ss) LUT");
    CreateAndUploadLutArray(allLutData, m_lutTextureArray, L"LutTextureArray");
}

void Renderer::CreateAndUploadLutArray(const std::vector<std::vector<float>>& allLutData, ComPtr<ID3D12Resource>& tar, const std::wstring& rn) {
    if (allLutData.empty()) return;
    UINT arraySize = (UINT)allLutData.size();
    D3D12_RESOURCE_DESC td = {}; td.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D; td.Width = LUT_RESOLUTION; td.Height = LUT_RESOLUTION;
    td.DepthOrArraySize = arraySize; td.MipLevels = 1; td.Format = DXGI_FORMAT_R32_FLOAT; td.SampleDesc.Count = 1;
    ThrowIfFailed(m_device->CreateCommittedResource(&nv_helpers_dx12::kDefaultHeapProps, D3D12_HEAP_FLAG_NONE, &td, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&tar))); tar->SetName(rn.c_str());
    UINT64 ubs = GetRequiredIntermediateSize(tar.Get(), 0, arraySize);
    m_lutUploadHeaps.emplace_back(); auto& uh = m_lutUploadHeaps.back();
    auto ubd = CD3DX12_RESOURCE_DESC::Buffer(ubs);
    ThrowIfFailed(m_device->CreateCommittedResource(&nv_helpers_dx12::kUploadHeapProps, D3D12_HEAP_FLAG_NONE, &ubd, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&uh)));
    std::vector<D3D12_SUBRESOURCE_DATA> sr(arraySize);
    for (UINT i = 0; i < arraySize; ++i) { sr[i].pData = allLutData[i].data(); sr[i].RowPitch = LUT_RESOLUTION*sizeof(float); sr[i].SlicePitch = sr[i].RowPitch*LUT_RESOLUTION; }
    UpdateSubresources(m_commandList.Get(), tar.Get(), uh.Get(), 0, 0, arraySize, sr.data());
    auto b = CD3DX12_RESOURCE_BARRIER::Transition(tar.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE|D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    m_commandList->ResourceBarrier(1, &b);
}

void Renderer::CreatePathStateBuffer() {
    UINT bs = GetWidth()*GetHeight()*88;
    auto bd = CD3DX12_RESOURCE_DESC::Buffer(bs, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    ThrowIfFailed(m_device->CreateCommittedResource(&nv_helpers_dx12::kDefaultHeapProps, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&m_pathStateBuffer)));
    m_pathStateBuffer->SetName(L"PathStateBuffer");
}

void Renderer::CreateStreamingCompactionBuffers() {
    UINT totalPixels = GetWidth() * GetHeight();
    for (int i = 0; i < MAX_STACKS; ++i) {
        auto d = CD3DX12_RESOURCE_DESC::Buffer(totalPixels*sizeof(uint32_t)*2, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        ThrowIfFailed(m_device->CreateCommittedResource(&nv_helpers_dx12::kDefaultHeapProps, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&m_stackBuffers[i])));
        m_stackBuffers[i]->SetName((L"PixelStack_" + std::to_wstring(i)).c_str());
    }
    { auto d = CD3DX12_RESOURCE_DESC::Buffer(MAX_STACKS*sizeof(uint32_t), D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS); ThrowIfFailed(m_device->CreateCommittedResource(&nv_helpers_dx12::kDefaultHeapProps, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&m_globalCounterBuffer))); }
    { auto d = CD3DX12_RESOURCE_DESC::Buffer(MAX_INDIRECT_COMMANDS*sizeof(D3D12_DISPATCH_ARGUMENTS), D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS); ThrowIfFailed(m_device->CreateCommittedResource(&nv_helpers_dx12::kDefaultHeapProps, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&m_indirectArgsBuffer))); }
    { auto hp = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD); auto bd = CD3DX12_RESOURCE_DESC::Buffer(MAX_STACKS*sizeof(uint32_t));
      ThrowIfFailed(m_device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&m_zeroBuffer)));
      void* p; m_zeroBuffer->Map(0, nullptr, &p); memset(p, 0, MAX_STACKS*sizeof(uint32_t)); m_zeroBuffer->Unmap(0, nullptr); }
    { auto d = CD3DX12_RESOURCE_DESC::Buffer(SORT_BUCKETS*sizeof(uint32_t), D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS); ThrowIfFailed(m_device->CreateCommittedResource(&nv_helpers_dx12::kDefaultHeapProps, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&m_sortCountBuffer))); }
    { auto d = CD3DX12_RESOURCE_DESC::Buffer(SORT_BUCKETS*sizeof(uint32_t), D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS); ThrowIfFailed(m_device->CreateCommittedResource(&nv_helpers_dx12::kDefaultHeapProps, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&m_sortOffsetBuffer))); }
    { auto d = CD3DX12_RESOURCE_DESC::Buffer(8*sizeof(uint32_t), D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS); ThrowIfFailed(m_device->CreateCommittedResource(&nv_helpers_dx12::kDefaultHeapProps, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&m_sortBoundsBuffer))); }
    { const UINT MAX_U = 0xFFFFFFFF, MIN_U = 0; const UINT initData[8] = { MAX_U, MAX_U, MAX_U, MIN_U, MIN_U, MIN_U, MAX_U, MIN_U };
      auto hp = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD); auto bd = CD3DX12_RESOURCE_DESC::Buffer(sizeof(initData));
      ThrowIfFailed(m_device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&m_sortBoundsResetBuffer)));
      void* p; CD3DX12_RANGE rr(0,0); ThrowIfFailed(m_sortBoundsResetBuffer->Map(0, &rr, &p)); memcpy(p, initData, sizeof(initData)); m_sortBoundsResetBuffer->Unmap(0, nullptr); }
}

void Renderer::CreateIndirectCommandSignature() {
    D3D12_INDIRECT_ARGUMENT_DESC ad = {}; ad.Type = D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH;
    D3D12_COMMAND_SIGNATURE_DESC cd = {}; cd.ByteStride = sizeof(D3D12_DISPATCH_ARGUMENTS); cd.NumArgumentDescs = 1; cd.pArgumentDescs = &ad;
    ThrowIfFailed(m_device->CreateCommandSignature(&cd, nullptr, IID_PPV_ARGS(&m_commandSignature)));
}

void Renderer::CompileSetupIndirectShader() {
    const char* shaderCode = R"(
        RWByteAddressBuffer Counters : register(u0); RWStructuredBuffer<uint3> IndirectArgs : register(u1);
        cbuffer Params : register(b0) { uint counterReadIdx; uint argWriteIdx; uint counterClearIdx; uint groupSize; };
        [numthreads(1,1,1)] void main() { uint ac = Counters.Load(counterReadIdx*4); IndirectArgs[argWriteIdx] = uint3((ac+groupSize-1)/groupSize, 1, 1);
        #if CLEAR_COUNTER
            Counters.Store(counterClearIdx*4, 0);
        #endif
        })";
    auto CompileVariant = [&](const char* name, bool doClear, ComPtr<ID3D12PipelineState>& pso) {
        D3D_SHADER_MACRO macros[] = { { "CLEAR_COUNTER", doClear ? "1" : "0" }, { NULL, NULL } };
        ComPtr<ID3DBlob> sb, eb;
        ThrowIfFailed(D3DCompile(shaderCode, strlen(shaderCode), name, macros, nullptr, "main", "cs_5_0", 0, 0, &sb, &eb));
        D3D12_COMPUTE_PIPELINE_STATE_DESC pd = {}; pd.pRootSignature = m_rsSetupIndirect.Get(); pd.CS = { sb->GetBufferPointer(), sb->GetBufferSize() };
        ThrowIfFailed(m_device->CreateComputePipelineState(&pd, IID_PPV_ARGS(&pso)));
    };
    { CD3DX12_ROOT_PARAMETER1 rp[2]; CD3DX12_DESCRIPTOR_RANGE1 range[1];
      range[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 2, 0, 0, D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE);
      rp[0].InitAsDescriptorTable(1, range); rp[1].InitAsConstants(4, 0);
      CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC rd; rd.Init_1_1(2, rp, 0, nullptr);
      ComPtr<ID3DBlob> sb, eb; D3D12SerializeVersionedRootSignature(&rd, &sb, &eb);
      ThrowIfFailed(m_device->CreateRootSignature(0, sb->GetBufferPointer(), sb->GetBufferSize(), IID_PPV_ARGS(&m_rsSetupIndirect))); }
    CompileVariant("SetupIndirect_Clear", true, m_psoSetupIndirect);
    CompileVariant("SetupIndirect_NoClear", false, m_psoSetupIndirectNoClear);
}

void Renderer::CreateBindlessTextures(std::vector<TextureData>& textures, UINT heapBaseSlot, const std::wstring& debugPrefix) {
    for (size_t i = 0; i < textures.size(); ++i) {
        auto& tex = textures[i]; const auto& meta = tex.image.GetMetadata();
        D3D12_RESOURCE_DESC d = {}; d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D; d.Width = meta.width; d.Height = (UINT)meta.height;
        d.DepthOrArraySize = 1; d.MipLevels = (UINT16)meta.mipLevels; d.Format = meta.format; d.SampleDesc.Count = 1;
        ComPtr<ID3D12Resource> gpuTex;
        ThrowIfFailed(m_device->CreateCommittedResource(&nv_helpers_dx12::kDefaultHeapProps, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&gpuTex)));
        const UINT sc = (UINT)meta.mipLevels; std::vector<D3D12_SUBRESOURCE_DATA> sr(sc);
        for (UINT m = 0; m < sc; ++m) { const auto* img = tex.image.GetImage(m,0,0); sr[m].pData = img->pixels; sr[m].RowPitch = (LONG_PTR)img->rowPitch; sr[m].SlicePitch = (LONG_PTR)img->slicePitch; }
        UINT64 us = GetRequiredIntermediateSize(gpuTex.Get(), 0, sc);
        ComPtr<ID3D12Resource> upload; auto ud = CD3DX12_RESOURCE_DESC::Buffer(us);
        ThrowIfFailed(m_device->CreateCommittedResource(&nv_helpers_dx12::kUploadHeapProps, D3D12_HEAP_FLAG_NONE, &ud, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&upload)));
        UpdateSubresources(m_commandList.Get(), gpuTex.Get(), upload.Get(), 0, 0, sc, sr.data());
        auto b = CD3DX12_RESOURCE_BARRIER::Transition(gpuTex.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE|D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        m_commandList->ResourceBarrier(1, &b);
        m_bindlessGpuTextures.push_back(gpuTex); m_bindlessUploadHeaps.push_back(upload);
    }
}

void Renderer::ClearSortBuffers(ID3D12GraphicsCommandList* cmdList) {
    UINT cv[4] = {0,0,0,0}; cmdList->ClearUnorderedAccessViewUint(m_sortCountGpuHandle, m_sortCountCpuHandle, m_sortCountBuffer.Get(), cv, 0, nullptr);
    auto b1 = CD3DX12_RESOURCE_BARRIER::Transition(m_sortBoundsBuffer.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST); cmdList->ResourceBarrier(1, &b1);
    cmdList->CopyResource(m_sortBoundsBuffer.Get(), m_sortBoundsResetBuffer.Get());
    auto b2 = CD3DX12_RESOURCE_BARRIER::Transition(m_sortBoundsBuffer.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS); cmdList->ResourceBarrier(1, &b2);
}

void Renderer::CreateReadbackBuffer() {
    D3D12_RESOURCE_DESC td = m_scratchPing->GetDesc(); D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp; UINT64 tb = 0;
    m_device->GetCopyableFootprints(&td, 0, 1, 0, &fp, nullptr, nullptr, &tb);
    auto hp = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK); auto bd = CD3DX12_RESOURCE_DESC::Buffer(tb);
    ThrowIfFailed(m_device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&m_readbackBuffer)));
}

void Renderer::SaveSimulationData(uint32_t stepIndex) {
    namespace fs = std::filesystem; if (!fs::exists("output")) fs::create_directory("output");
    D3D12_RESOURCE_DESC td = m_scratchPing->GetDesc(); D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp;
    m_device->GetCopyableFootprints(&td, 0, 1, 0, &fp, nullptr, nullptr, nullptr);
    UINT width = (UINT)td.Width, height = td.Height, rowPitch = fp.Footprint.RowPitch;
    ThrowIfFailed(m_commandAllocators[m_frameIndex]->Reset());
    ThrowIfFailed(m_commandList->Reset(m_commandAllocators[m_frameIndex].Get(), m_pipelineState.Get()));
    auto ProcessSlice = [&](UINT slice, auto func) {
        auto b1 = CD3DX12_RESOURCE_BARRIER::Transition(m_scratchPing.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE); m_commandList->ResourceBarrier(1, &b1);
        UINT sub = D3D12CalcSubresource(0, slice, 0, 1, td.DepthOrArraySize);
        CD3DX12_TEXTURE_COPY_LOCATION dst(m_readbackBuffer.Get(), fp), src(m_scratchPing.Get(), sub);
        m_commandList->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        auto b2 = CD3DX12_RESOURCE_BARRIER::Transition(m_scratchPing.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS); m_commandList->ResourceBarrier(1, &b2);
        ThrowIfFailed(m_commandList->Close()); ID3D12CommandList* cl[] = {m_commandList.Get()}; m_commandQueue->ExecuteCommandLists(1, cl);
        m_fenceValue++; m_commandQueue->Signal(m_fence.Get(), m_fenceValue);
        if (m_fence->GetCompletedValue() < m_fenceValue) { m_fence->SetEventOnCompletion(m_fenceValue, m_fenceEvent); WaitForSingleObject(m_fenceEvent, INFINITE); }
        uint8_t* p = nullptr; CD3DX12_RANGE rr(0, fp.Footprint.RowPitch*height); ThrowIfFailed(m_readbackBuffer->Map(0, &rr, (void**)&p)); func(p);
        CD3DX12_RANGE wr(0,0); m_readbackBuffer->Unmap(0, &wr);
        ThrowIfFailed(m_commandAllocators[m_frameIndex]->Reset()); ThrowIfFailed(m_commandList->Reset(m_commandAllocators[m_frameIndex].Get(), m_pipelineState.Get()));
    };
    auto WriteBin = [&](const std::string& s, const std::vector<float>& d) { std::ofstream f("output/" + std::to_string(stepIndex) + "_" + s + ".bin", std::ios::binary); if (f) f.write((const char*)d.data(), d.size()*sizeof(float)); };
    ProcessSlice(7, [&](uint8_t* rd) { std::vector<float> a(width*height),b(width*height),c(width*height),d(width*height); for (UINT y=0;y<height;++y) { float* row=(float*)(rd+y*rowPitch); for (UINT x=0;x<width;++x) { a[y*width+x]=row[x*4]; b[y*width+x]=row[x*4+1]; c[y*width+x]=row[x*4+2]; d[y*width+x]=row[x*4+3]; } } WriteBin("restir",a); WriteBin("gt",b); WriteBin("init",c); WriteBin("albedo",d); });
    ProcessSlice(8, [&](uint8_t* rd) { std::vector<float> e(width*height),f(width*height); for (UINT y=0;y<height;++y) { float* row=(float*)(rd+y*rowPitch); for (UINT x=0;x<width;++x) { e[y*width+x]=row[x*4]; f[y*width+x]=row[x*4+1]; } } WriteBin("roughness",e); WriteBin("depth",f); });
    ProcessSlice(9, [&](uint8_t* rd) { std::vector<float> g(width*height*3); for (UINT y=0;y<height;++y) { float* row=(float*)(rd+y*rowPitch); for (UINT x=0;x<width;++x) { size_t idx=(y*width+x)*3; g[idx]=row[x*4]; g[idx+1]=row[x*4+1]; g[idx+2]=row[x*4+2]; } } WriteBin("normal",g); });
    ThrowIfFailed(m_commandList->Close());
}

void Renderer::CreateDLSSResources() {
    auto createTex = [&](ComPtr<ID3D12Resource>& res, DXGI_FORMAT fmt, const wchar_t* name) {
        D3D12_RESOURCE_DESC d = {}; d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D; d.Width = GetWidth(); d.Height = GetHeight();
        d.DepthOrArraySize = 1; d.MipLevels = 1; d.Format = fmt; d.SampleDesc.Count = 1; d.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        ThrowIfFailed(m_device->CreateCommittedResource(&nv_helpers_dx12::kDefaultHeapProps, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&res))); res->SetName(name);
    };
    createTex(m_dlssInput, DXGI_FORMAT_R16G16B16A16_FLOAT, L"DLSS_Input");
    createTex(m_dlssDepth, DXGI_FORMAT_R32_FLOAT, L"DLSS_Depth");
    createTex(m_dlssMVec, DXGI_FORMAT_R16G16_FLOAT, L"DLSS_MVec");
    createTex(m_dlssNormals, DXGI_FORMAT_R16G16B16A16_FLOAT, L"DLSS_Normals");
    createTex(m_dlssDiffuseAlbedo, DXGI_FORMAT_R16G16B16A16_FLOAT, L"DLSS_DiffuseAlbedo");
    createTex(m_dlssOutput, DXGI_FORMAT_R16G16B16A16_FLOAT, L"DLSS_Output");
    createTex(m_dlssSpecularAlbedo, DXGI_FORMAT_R16G16B16A16_FLOAT, L"DLSS_SpecAlbedo");
    createTex(m_dlssRoughness, DXGI_FORMAT_R16_FLOAT, L"DLSS_Roughness");
    createTex(m_dlssSpecMVec, DXGI_FORMAT_R16G16_FLOAT, L"DLSS_SpecMVec");
    createTex(m_dlssSpecHitDist, DXGI_FORMAT_R16_FLOAT, L"DLSS_HitDist");
    createTex(m_dlssTransparency, DXGI_FORMAT_R16G16B16A16_FLOAT, L"DLSS_Trans");
    createTex(m_dlssColorBeforeTrans, DXGI_FORMAT_R16G16B16A16_FLOAT, L"DLSS_ColorPreTrans");
    m_state.SetInitialState(m_dlssDepth.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    m_state.SetInitialState(m_dlssMVec.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    m_state.SetInitialState(m_dlssNormals.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    m_state.SetInitialState(m_dlssDiffuseAlbedo.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    m_state.SetInitialState(m_dlssOutput.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    m_state.SetInitialState(m_dlssSpecularAlbedo.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    m_state.SetInitialState(m_dlssRoughness.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    m_state.SetInitialState(m_dlssSpecMVec.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    m_state.SetInitialState(m_dlssSpecHitDist.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    m_state.SetInitialState(m_dlssTransparency.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    m_state.SetInitialState(m_dlssColorBeforeTrans.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    if (!m_viewportHandle) slAllocateResources(m_commandList.Get(), sl::kFeatureDLSS_RR, m_viewportHandle);
}

void Renderer::RunDLSS_RR(ID3D12GraphicsCommandList* cmdList) {
    if (!cmdList || !m_frameToken || !m_outputResource || !m_dlssOutput || !m_dlssDepth || !m_dlssMVec || !m_dlssNormals || !m_dlssDiffuseAlbedo || !m_dlssSpecularAlbedo || !m_dlssRoughness || !m_dlssSpecHitDist) return;
    constexpr D3D12_RESOURCE_STATES stateUAV = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    constexpr D3D12_RESOURCE_STATES stateSRV = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    ID3D12Resource* dlssInputs[] = { m_dlssDepth.Get(), m_dlssMVec.Get(), m_dlssNormals.Get(), m_dlssDiffuseAlbedo.Get(), m_dlssSpecularAlbedo.Get(), m_dlssRoughness.Get(), m_dlssSpecHitDist.Get(), m_dlssInput.Get() };
    std::vector<D3D12_RESOURCE_BARRIER> preB; for (auto* r : dlssInputs) if (r) preB.push_back(CD3DX12_RESOURCE_BARRIER::Transition(r, stateUAV, stateSRV));
    cmdList->ResourceBarrier((UINT)preB.size(), preB.data());

    sl::Constants constants{};
    glm::mat4 viewGlm = nv_helpers_dx12::CameraManip.getMatrix(), viewT = glm::transpose(viewGlm);
    DirectX::XMFLOAT4X4 viewF{}; std::memcpy(&viewF, glm::value_ptr(viewT), sizeof(viewF));
    DirectX::XMMATRIX xmView = DirectX::XMLoadFloat4x4(&viewF);
    DirectX::XMMATRIX xmProj = DirectX::XMMatrixPerspectiveFovRH(DirectX::XMConvertToRadians(60.0f), m_aspectRatio, 0.00001f, 10000.0f);
    DirectX::XMMATRIX xmViewProj = XMMatrixMultiply(xmView, xmProj), xmPrevViewProj = XMMatrixMultiply(m_dlssPrevViewMatrix, m_dlssPrevProjMatrix);
    auto XmToSl = [](const DirectX::XMMATRIX& m) -> sl::float4x4 { DirectX::XMFLOAT4X4 t; DirectX::XMStoreFloat4x4(&t, m); sl::float4x4 o{}; std::memcpy(&o, &t, sizeof(o)); return o; };
    constants.cameraViewToClip = XmToSl(xmProj); constants.clipToCameraView = XmToSl(XMMatrixInverse(nullptr, xmProj));
    constants.clipToPrevClip = XmToSl(XMMatrixMultiply(XMMatrixInverse(nullptr, xmViewProj), xmPrevViewProj));
    constants.prevClipToClip = XmToSl(XMMatrixMultiply(XMMatrixInverse(nullptr, xmPrevViewProj), xmViewProj));
    constants.cameraFOV = XMConvertToRadians(60.0f); constants.cameraAspectRatio = m_aspectRatio; constants.cameraNear = 0.00001f; constants.cameraFar = 10000.0f;
    constants.jitterOffset = { -m_jitterX, -m_jitterY }; constants.mvecScale = { 1.0f/(float)GetWidth(), 1.0f/(float)GetHeight() };
    constants.motionVectorsInvalidValue = -1.0f; constants.cameraMotionIncluded = sl::Boolean::eTrue; constants.depthInverted = sl::Boolean::eFalse;
    constants.motionVectors3D = sl::Boolean::eFalse; constants.motionVectorsJittered = sl::Boolean::eFalse;
    constants.reset = (m_jitterFrameIndex <= 1) ? sl::Boolean::eTrue : sl::Boolean::eFalse;
    { auto iv = XMMatrixInverse(nullptr, xmView); DirectX::XMFLOAT4X4 f; XMStoreFloat4x4(&f, iv);
      constants.cameraPos = {f._41,f._42,f._43}; constants.cameraRight = {f._11,f._12,f._13}; constants.cameraUp = {f._21,f._22,f._23}; constants.cameraFwd = {-f._31,-f._32,-f._33}; }
    SL_CHECK(slSetConstants(constants, *m_frameToken, m_viewportHandle));

    sl::DLSSDOptions options{}; options.mode = sl::DLSSMode::eDLAA; options.outputWidth = GetWidth(); options.outputHeight = GetHeight();
    options.colorBuffersHDR = sl::Boolean::eTrue; options.normalRoughnessMode = sl::DLSSDNormalRoughnessMode::eUnpacked;
    options.worldToCameraView = XmToSl(xmView); options.cameraViewToWorld = XmToSl(XMMatrixInverse(nullptr, xmView));
    sl::DLSSDPreset p = sl::DLSSDPreset::ePresetE;
    options.dlaaPreset = options.qualityPreset = options.balancedPreset = options.performancePreset = options.ultraPerformancePreset = options.ultraQualityPreset = p;
    SL_CHECK(slDLSSDSetOptions(m_viewportHandle, options));

    sl::Resource slDepth(sl::ResourceType::eTex2d, m_dlssDepth.Get(), (uint32_t)stateSRV);
    sl::Resource slMVec(sl::ResourceType::eTex2d, m_dlssMVec.Get(), (uint32_t)stateSRV);
    sl::Resource slNormals(sl::ResourceType::eTex2d, m_dlssNormals.Get(), (uint32_t)stateSRV);
    sl::Resource slAlbedo(sl::ResourceType::eTex2d, m_dlssDiffuseAlbedo.Get(), (uint32_t)stateSRV);
    sl::Resource slSpecAlb(sl::ResourceType::eTex2d, m_dlssSpecularAlbedo.Get(), (uint32_t)stateSRV);
    sl::Resource slRough(sl::ResourceType::eTex2d, m_dlssRoughness.Get(), (uint32_t)stateSRV);
    sl::Resource slSpecHit(sl::ResourceType::eTex2d, m_dlssSpecHitDist.Get(), (uint32_t)stateSRV);
    sl::Resource slInput(sl::ResourceType::eTex2d, m_dlssInput.Get(), (uint32_t)stateSRV);
    sl::Resource slOutput(sl::ResourceType::eTex2d, m_dlssOutput.Get(), (uint32_t)stateUAV);
    sl::Extent extent{0, 0, GetWidth(), GetHeight()}; auto life = sl::ResourceLifecycle::eValidUntilEvaluate;
    std::vector<sl::ResourceTag> tags = {
        {&slDepth, sl::kBufferTypeLinearDepth, life, &extent}, {&slMVec, sl::kBufferTypeMotionVectors, life, &extent},
        {&slNormals, sl::kBufferTypeNormals, life, &extent}, {&slRough, sl::kBufferTypeRoughness, life, &extent},
        {&slAlbedo, sl::kBufferTypeAlbedo, life, &extent}, {&slSpecAlb, sl::kBufferTypeSpecularAlbedo, life, &extent},
        {&slSpecHit, sl::kBufferTypeSpecularHitDistance, life, &extent}, {&slInput, sl::kBufferTypeScalingInputColor, life, &extent},
        {&slOutput, sl::kBufferTypeScalingOutputColor, life, &extent} };
    SL_CHECK(slSetTagForFrame(*m_frameToken, m_viewportHandle, tags.data(), (uint32_t)tags.size(), cmdList));

    ComPtr<ID3D12InfoQueue> infoQueue;
    if (SUCCEEDED(m_device.As(&infoQueue))) infoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_ERROR, FALSE);
    const sl::BaseStructure* evalInputs[] = { &m_viewportHandle, &options };
    sl::Result evalResult = slEvaluateFeature(sl::kFeatureDLSS_RR, *m_frameToken, evalInputs, _countof(evalInputs), cmdList);
    if (infoQueue) infoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_ERROR, TRUE);
    if (evalResult != sl::Result::eOk) { std::wcout << L"[DLSS-RR] slEvaluateFeature failed: " << (int)evalResult << std::endl; return; }

    std::vector<D3D12_RESOURCE_BARRIER> postB; for (auto* r : dlssInputs) postB.push_back(CD3DX12_RESOURCE_BARRIER::Transition(r, stateSRV, stateUAV));
    cmdList->ResourceBarrier((UINT)postB.size(), postB.data());
    m_dlssPrevViewMatrix = xmView; m_dlssPrevProjMatrix = xmProj;
}
