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

#pragma once

#include "DXSample.h"

#include <dxcapi.h>
#include <vector>
#include <d3d12video.h>
#include <DirectXPackedVector.h>

#include "nv_helpers_dx12/ShaderBindingTableGenerator.h"
#include "nv_helpers_dx12/TopLevelASGenerator.h"
#include "../src/Components/Vertex.h"
#include <DirectML.h>
#include <onnxruntime_cxx_api.h>
#include <dml_provider_factory.h>
#include "glm/gtc/matrix_transform.hpp"
#include "ResourceStateTracker.h"

#include <sl.h>
#include <sl_consts.h>
#include <sl_helpers.h>
#include <sl_dlss.h>

#include "LightTree.h"
#include "sl_dlss_d.h"

#include <unordered_map>
#include <iostream>

#ifndef LT_ENABLE_LOGS
#define LT_ENABLE_LOGS 1
#endif
#ifndef LT_LOG_BUILD_SPAM
#define LT_LOG_BUILD_SPAM 0
#endif
#if LT_ENABLE_LOGS
  #define LT_LOG(expr)  do { std::wcout << L"[LightTree] "      << expr << std::endl; } while(0)
  #define LT_WARN(expr) do { std::wcout << L"[LightTree][WARN] " << expr << std::endl; } while(0)
#else
  #define LT_LOG(expr)  do {} while(0)
  #define LT_WARN(expr) do {} while(0)
#endif

#include "../lib/imgui/imgui.h"
#include "../lib/imgui/imgui_impl_dx12.h"
#include "../lib/imgui/imgui_impl_win32.h"
#include "CameraRecorder.h"
#include "CameraPathSimulator.h"

using namespace DirectX;
using Microsoft::WRL::ComPtr;

constexpr int NUM_LUTS = 2;
constexpr int LUT_RESOLUTION = 16;
constexpr int NUM_SAMPLES_LUT = 32000;

static const UINT MAX_STACKS = 4;
static const UINT MAX_INDIRECT_COMMANDS = MAX_STACKS;

// Represents a unique piece of geometry (one BLAS)
struct MeshGPU {
    ComPtr<ID3D12Resource>  blas;
    ComPtr<ID3D12Resource>  vertexBuffer;
    ComPtr<ID3D12Resource>  indexBuffer;

    UINT                    vertexCount = 0;
    UINT                    indexCount  = 0;
    UINT                    opaqueTriCount = 0;
    UINT                    alphaTriCount  = 0;

    UINT                    globalVertexBase = 0;
    UINT                    globalIndexBase  = 0;
    UINT                    materialIDBase   = 0;

    std::vector<Vertex>     cpuVertices;
    std::vector<UINT>       cpuIndices;
    std::vector<UINT>       cpuMaterialIDs;
};

// Represents one visible object in the scene
struct SceneInstance {
    UINT        meshIndex;
    XMMATRIX    transform;
    XMMATRIX    prevTransform;
};

class Renderer : public DXSample {
public:
    void EnsureSLViewportAllocated(ID3D12GraphicsCommandList *cmdList);
    void BuildGlobalMeshBuffers();
    void CreateTriToLightIdBuffer();
    Renderer(UINT width, UINT height, std::wstring name);
    void CreateDLSSResources();
    void RunDLSS_RR(ID3D12GraphicsCommandList* cmdList);
    virtual void OnInit();
    virtual void OnUpdate();
    virtual void OnRender();
    virtual void OnDestroy();

private:
    enum class Stage { RayGen, Compute, FixedCompute, Wavefront, Barrier, LoopStart, LoopEnd, PingSwap, ClearSort, Callable, DLSS };

    struct PassDesc {
        std::wstring  file;
        Stage         stage   = Stage::RayGen;
        uint32_t      groupX  = 0, groupY = 0;
        uint32_t      psoIdx  = UINT32_MAX;
        bool          isWorkGraph = false;
        uint32_t      wgIdx  = UINT32_MAX;
        uint32_t      loopCount = 0;
        int32_t       targetIdx = -1;
    };

    static PassDesc ParsePass(const std::wstring& token) {
        PassDesc p{};
        if (token == L"barrier")   { p.stage = Stage::Barrier;   return p; }
        if (token == L"pingswap")  { p.stage = Stage::PingSwap;  return p; }
        if (token == L"endloop")   { p.stage = Stage::LoopEnd;   return p; }
        if (token == L"clearsort") { p.stage = Stage::ClearSort; return p; }
        if (token == L"dlss")      { p.stage = Stage::DLSS;      return p; }
        if (token.rfind(L"loop:", 0) == 0) {
            p.stage = Stage::LoopStart;
            if (swscanf_s(token.c_str() + 5, L"%u", &p.loopCount) != 1)
                throw std::runtime_error("Invalid loop count format");
            return p;
        }
        const size_t bar = token.find(L'|');
        p.file = token.substr(0, bar);
        if (bar == std::wstring::npos) return p;
        const std::wstring tail = token.substr(bar + 1);
        if (tail == L"rg" || tail == L"raygen") return p;
        if (tail == L"call") { p.stage = Stage::Callable; return p; }
        if (tail.rfind(L"wg:", 0) == 0) { p.stage = Stage::Compute; p.isWorkGraph = true; swscanf_s(tail.c_str() + 3, L"%ux%u", &p.groupX, &p.groupY); return p; }
        if (tail.rfind(L"wf:", 0) == 0) { p.stage = Stage::Wavefront; if (swscanf_s(tail.c_str() + 3, L"%u", &p.groupX) != 1) throw std::runtime_error("Invalid wf size"); p.groupY = 1; return p; }
        if (tail.rfind(L"cs:", 0) == 0) { p.stage = Stage::Compute; if (swscanf_s(tail.c_str() + 3, L"%ux%u", &p.groupX, &p.groupY) != 2) throw std::runtime_error("Invalid cs size"); return p; }
        if (tail.rfind(L"fx:", 0) == 0) { p.stage = Stage::FixedCompute; if (swscanf_s(tail.c_str() + 3, L"%u", &p.groupX) != 1) throw std::runtime_error("Invalid fx size"); p.groupY = 1; return p; }
        throw std::runtime_error("Unknown stage spec in pass string");
    }

    void LinkLoops() {
        std::vector<size_t> stack;
        for (size_t i = 0; i < m_passes.size(); ++i) {
            if (m_passes[i].stage == Stage::LoopStart) stack.push_back(i);
            else if (m_passes[i].stage == Stage::LoopEnd) {
                if (stack.empty()) throw std::runtime_error("Found 'endloop' without matching 'loop:'");
                m_passes[i].targetIdx = static_cast<int32_t>(stack.back());
                stack.pop_back();
            }
        }
        if (!stack.empty()) throw std::runtime_error("Found 'loop:' without matching 'endloop'");
    }

    std::vector<std::wstring>                    m_passSequence;
    std::unordered_map<std::wstring, uint32_t>   m_passIndex;
    std::vector<Microsoft::WRL::ComPtr<IDxcBlob>> m_rayGenLibs;

    struct BTriVertex {
        XMFLOAT3 vertex;
        UINT     packedNormal;
        PackedVector::XMHALF2  texCoord;
    };

    ComPtr<ID3D12Resource> m_vertexGlobal;
    ComPtr<ID3D12Resource> m_indexGlobal;
    UINT                   m_totalVertexCount = 0;
    UINT                   m_totalIndexCount  = 0;

    std::vector<MeshGPU>                m_meshes;
    std::vector<SceneInstance>          m_sceneInstances;
    std::vector<std::pair<ComPtr<ID3D12Resource>, XMMATRIX>> m_tlasInstances;

    static const UINT FrameCount = 2;

    sl::FrameToken*     m_frameToken     = nullptr;
    sl::ViewportHandle  m_viewportHandle = sl::ViewportHandle(0);
    sl::DLSSDOptions     m_dlssdOptions   {};
    sl::Constants        m_slConstants    {};

    std::vector<PassDesc> m_passes;
    std::vector<ComPtr<ID3D12PipelineState>> m_csPSOs;
    std::vector<ComPtr<ID3D12PipelineState>> m_wgPSOs;

    struct WgRuntimeData {
        D3D12_PROGRAM_IDENTIFIER        id;
        D3D12_GPU_VIRTUAL_ADDRESS_RANGE backing;
        ComPtr<ID3D12Resource>          backingRes;
    };
    std::vector<WgRuntimeData> m_wgRuntime;

    CD3DX12_VIEWPORT m_viewport;
    CD3DX12_RECT m_scissorRect;
    ComPtr<IDXGISwapChain3> m_swapChain;
    ComPtr<ID3D12Device10> m_device;
    ComPtr<ID3D12Resource> m_renderTargets[FrameCount];
    ComPtr<ID3D12CommandAllocator> m_commandAllocators[FrameCount];
    ComPtr<ID3D12CommandQueue> m_commandQueue;
    ComPtr<ID3D12RootSignature> m_rootSignature;
    ComPtr<ID3D12RootSignature>   m_computeSignature;
    ComPtr<ID3D12DescriptorHeap> m_rtvHeap;
    ComPtr<ID3D12PipelineState> m_pipelineState;
    ComPtr<ID3D12GraphicsCommandList10> m_commandList;
    UINT m_rtvDescriptorSize;

    ComPtr<ID3D12Resource> m_vertexBuffer;
    D3D12_VERTEX_BUFFER_VIEW m_vertexBufferView;
    bool g_keys[256] = {};
    UINT m_frameIndex;
    HANDLE m_fenceEvent;
    ComPtr<ID3D12Fence> m_fence;
    UINT64 m_fenceValue;

    void LoadPipeline();
    void LoadAssets();
    void OnInitTransform();
    void PopulateCommandList();
    void WaitForPreviousFrame();
    void CheckRaytracingSupport();
    void OnKeyDown(UINT8 key);
    virtual void OnKeyUp(UINT8 key);
    bool m_raster = true;

    struct AccelerationStructureBuffers {
        ComPtr<ID3D12Resource> pScratch;
        ComPtr<ID3D12Resource> pResult;
        ComPtr<ID3D12Resource> pResultUncompacted;
        ComPtr<ID3D12Resource> pInstanceDesc;
    };

    ComPtr<ID3D12Resource> m_bottomLevelAS;
    nv_helpers_dx12::TopLevelASGenerator m_topLevelASGenerator;
    AccelerationStructureBuffers m_topLevelASBuffers;
    ComPtr<ID3D12Resource> m_pathStateBuffer;
    void CreatePathStateBuffer();
    lt::LightTreeBuilder m_lightTree;

    struct Reservoir_DI  { uint8_t pad[100]; };
    struct Reservoir_GI  { uint8_t pad[100]; };
    struct SampleData    { uint8_t pad[100]; };
    struct InitialBSDFRay{ uint8_t pad[100]; };

    std::vector<LightTriangle> m_emissiveTriangles;
    ComPtr<ID3D12Resource> m_emissiveTrianglesBuffer;
    std::vector<uint32_t>           m_instTriOffset;
    std::vector<uint32_t>           m_triToLightId;
    ComPtr<ID3D12Resource>          m_triToLightIdBuffer;

    ComPtr<ID3D12Resource> m_lutTextureArray;
    std::vector<ComPtr<ID3D12Resource>> m_lutUploadHeaps;
    void GenerateLutTextures();
    void CreateAndUploadLutArray(const std::vector<std::vector<float>>& allLutData, ComPtr<ID3D12Resource>& textureArrayResource, const std::wstring& resourceName);

    AccelerationStructureBuffers CreateBottomLevelAS(
        std::vector<std::pair<ComPtr<ID3D12Resource>, uint32_t>> vVertexBuffers,
        std::vector<std::pair<ComPtr<ID3D12Resource>, uint32_t>> vIndexBuffers,
        UINT opaqueTriCount, UINT alphaTriCount);

    void CreateTopLevelAS(
        const std::vector<std::pair<ComPtr<ID3D12Resource>, DirectX::XMMATRIX>> &instances,
        bool updateOnly = false);

    void CreateAccelerationStructures();

    ComPtr<ID3D12RootSignature> CreateRayGenSignature();
    ComPtr<ID3D12RootSignature> CreateComputeSignature();
    ComPtr<ID3D12RootSignature> CreateMissSignature();
    ComPtr<ID3D12RootSignature> CreateHitSignature();
    void CreateRaytracingPipeline();

    ComPtr<IDxcBlob> m_rayGenLibrary, m_rayGenLibrary2, m_rayGenLibrary3;
    ComPtr<IDxcBlob> m_hitLibrary, m_missLibrary, m_anyHitLibrary;
    ComPtr<ID3D12RootSignature> m_rayGenSignature, m_hitSignature, m_missSignature;
    std::vector<std::wstring> m_callableShaderNames;
    ComPtr<ID3D12StateObject> m_rtStateObject;
    ComPtr<ID3D12StateObjectProperties> m_rtStateObjectProps;
    std::vector<ComPtr<ID3D12StateObject>> m_wgStateObjects;
    std::vector<ComPtr<ID3D12WorkGraphProperties>> m_wgProps;

    void CreateRaytracingOutputBuffer();
    void CreateShaderResourceHeap();
    ComPtr<ID3D12Resource> m_outputResource;
    ComPtr<ID3D12Resource> m_dlssOutputBuffer;
    ComPtr<ID3D12Resource> m_permanentDataTexture;
    ComPtr<ID3D12Resource> m_scratchPing;
    ComPtr<ID3D12Resource> m_svgfConstBuffer;
    ComPtr<ID3D12DescriptorHeap> m_srvUavHeap;
    ComPtr<ID3D12DescriptorHeap> m_stagingUavHeap;

    D3D12_GPU_DESCRIPTOR_HANDLE m_sortCountGpuHandle;
    D3D12_CPU_DESCRIPTOR_HANDLE m_sortCountCpuHandle;
    D3D12_GPU_DESCRIPTOR_HANDLE m_sortOffsetGpuHandle;
    D3D12_CPU_DESCRIPTOR_HANDLE m_sortOffsetCpuHandle;
    D3D12_GPU_DESCRIPTOR_HANDLE m_sortBoundsGpuHandle;
    D3D12_CPU_DESCRIPTOR_HANDLE m_sortBoundsCpuHandle;

    void CreateShaderBindingTable();
    nv_helpers_dx12::ShaderBindingTableGenerator m_sbtHelper;
    ComPtr<ID3D12Resource> m_sbtStorage;

    void CreateCameraBuffer();
    void UpdateCameraBuffer();
    ComPtr<ID3D12Resource> m_cameraBuffer;
    ComPtr<ID3D12Resource> m_sampleBuffer_current, m_sampleBuffer_last;
    ComPtr<ID3D12Resource> m_reservoirBuffer, m_reservoirBuffer_2, m_reservoirBuffer_3, m_reservoirBuffer_4;
    ComPtr<ID3D12Resource> m_initialBSDFRayBuffer;
    ComPtr<ID3D12DescriptorHeap> m_constHeap;
    uint32_t m_cameraBufferSize = 0;

    void OnButtonDown(UINT32 lParam);
    void OnMouseMove(UINT8 wParam, UINT32 lParam);
    XMMATRIX m_prevViewMatrix, m_prevProjMatrix;
    XMMATRIX m_dlssPrevViewMatrix, m_dlssPrevProjMatrix;
    float m_jitterX = 0.0f, m_jitterY = 0.0f;
    uint32_t m_jitterFrameIndex = 0;

    ComPtr<ID3D12Resource> m_planeBuffer;
    D3D12_VERTEX_BUFFER_VIEW m_planeBufferView;
    void CreatePlaneVB();

    void CreateGlobalConstantBuffer();
    ComPtr<ID3D12Resource> m_globalConstantBuffer;
    void CreatePerInstanceConstantBuffers();
    std::vector<ComPtr<ID3D12Resource>> m_perInstanceConstantBuffers;

    void CreateDepthBuffer();
    ComPtr<ID3D12DescriptorHeap> m_dsvHeap;
    ComPtr<ID3D12Resource> m_depthStencil;

    ComPtr<ID3D12Resource> m_indexBuffer;
    D3D12_INDEX_BUFFER_VIEW m_indexBufferView;

    void CreateVB(std::string name);
    ComPtr<ID3D12Resource> m_materialBuffer;
    ComPtr<ID3D12Resource> m_materialIndexBuffer;
    std::vector<UINT> m_materialIDs;
    std::vector<Material> m_materials;

    std::vector<ComPtr<ID3D12Resource>> m_material;
    std::vector<ComPtr<ID3D12Resource>> m_materialID;

    ComPtr<IDxcBlob> m_shadowLibrary;
    ComPtr<ID3D12RootSignature> m_shadowSignature;
    uint32_t m_time = 0;

    struct InstanceProperties {
        XMMATRIX objectToWorld;
        XMMATRIX objectToWorldInverse;
        XMMATRIX prevObjectToWorld;
        XMMATRIX prevObjectToWorldInverse;
        XMMATRIX objectToWorldNormal;
        XMMATRIX prevObjectToWorldNormal;
        UINT  indexBase;
        UINT  vertexBase;
        UINT  materialBase;
        UINT  triToLightBase;
        UINT opaqueTriCount;
        UINT _pad[3];
    };

    struct FrameData { float Time; };

    ComPtr<ID3D12Resource> m_instanceProperties;
    ComPtr<ID3D12Resource> m_instancePropertiesPrevious;
    struct GeometryOffsets { UINT vertexBase; UINT indexBase; UINT materialBase; };
    std::vector<GeometryOffsets> m_geoOffsets;
    void CreateInstancePropertiesBuffer();
    void UpdateInstancePropertiesBuffer();

    HINSTANCE__ *m_mod;
    UINT m_currentDisplayLevel = 0;
    std::vector<UINT> m_displayLevels = {0, 1, 2, 3};
    void ExtractFrustumPlanes(const XMMATRIX &viewProjMatrix, XMFLOAT4 *planes);

    void CollectEmissiveTriangles();
    void CreateEmissiveTrianglesBuffer();
    float ComputeTriangleWeight(const XMFLOAT3 &v0, const XMFLOAT3 &v1, const XMFLOAT3 &v2, const XMFLOAT3 &emissiveColor, const DirectX::XMMATRIX &M);

    ComPtr<ID3D12Resource> m_stackBuffers[MAX_STACKS];
    ComPtr<ID3D12Resource> m_globalCounterBuffer;
    ComPtr<ID3D12Resource> m_indirectArgsBuffer;
    ComPtr<ID3D12CommandSignature> m_commandSignature;
    ComPtr<ID3D12PipelineState> m_psoSetupIndirect;
    ComPtr<ID3D12PipelineState> m_psoSetupIndirectNoClear;
    ComPtr<ID3D12Resource> m_sortBoundsResetBuffer;
    ComPtr<ID3D12RootSignature> m_rsSetupIndirect;
    void CreateStreamingCompactionBuffers();
    void CreateIndirectCommandSignature();
    void CompileSetupIndirectShader();
    ComPtr<ID3D12Resource> m_zeroBuffer;

    ComPtr<ID3D12Resource> m_sortCountBuffer, m_sortOffsetBuffer, m_sortBoundsBuffer;
    static const UINT SORT_BUCKETS = 65536;
    void ClearSortBuffers(ID3D12GraphicsCommandList* cmdList);

    ComPtr<ID3D12Resource> m_readbackBuffer;
    void CreateReadbackBuffer();
    void SaveSimulationData(uint32_t stepIndex);
    CameraRecorder m_recorder;
    CameraPathSimulator m_simulator;

    ComPtr<ID3D12Resource> m_dlssInput, m_dlssDepth, m_dlssMVec, m_dlssNormals;
    ComPtr<ID3D12Resource> m_dlssDiffuseAlbedo, m_dlssOutput;
    ComPtr<ID3D12Resource> m_dlssSpecularAlbedo, m_dlssRoughness, m_dlssSpecMVec, m_dlssSpecHitDist;
    ComPtr<ID3D12Resource> m_dlssTransparency, m_dlssColorBeforeTrans;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_dlssInputExtracted2D;
    D3D12_RESOURCE_STATES m_dlssInputExtracted2DState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    ResourceStateTracker m_state;
    static constexpr D3D12_RESOURCE_STATES kSRV = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;

    std::vector<ComPtr<ID3D12Resource>> m_pendingUploads;
    void KeepAliveUpload(Microsoft::WRL::ComPtr<ID3D12Resource> r) { if (r) m_pendingUploads.push_back(std::move(r)); }
    void ClearPendingUploadsAfterGPUIdle() { m_pendingUploads.clear(); }

    void BindMainHeap(ID3D12GraphicsCommandList* cl) {
        if (!cl || !m_srvUavHeap) return;
        ID3D12DescriptorHeap* heaps[] = { m_srvUavHeap.Get() };
        cl->SetDescriptorHeaps(1, heaps);
    }

    D3D12_GPU_DESCRIPTOR_HANDLE m_globalCountersGpuHandle{};
    D3D12_GPU_DESCRIPTOR_HANDLE m_indirectArgsGpuHandle{};

    static constexpr UINT BINDLESS_HEAP_START = 60;
    UINT m_bindlessAlbedoBase = 0, m_bindlessNormalBase = 0, m_bindlessRmaBase = 0;
    UINT m_totalBindlessTextures = 0;
    std::vector<ComPtr<ID3D12Resource>> m_bindlessGpuTextures;
    std::vector<ComPtr<ID3D12Resource>> m_bindlessUploadHeaps;
    void CreateBindlessTextures(std::vector<TextureData>& textures, UINT heapBaseSlot, const std::wstring& debugPrefix);
};
