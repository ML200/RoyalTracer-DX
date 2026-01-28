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

#include <sl.h>            // core SL types: sl::Result, sl::FeatureHandle, etc.
#include <sl_consts.h>     // the sl::kFeature… enum values
#include <sl_helpers.h>
#include <sl_dlss.h>       // DLSS Super Resolution API

#include "LightTree.h"
#include "sl_dlss_d.h"

#include <unordered_map>
#include <iostream>

// Toggle logs at compile time (define LT_ENABLE_LOGS=0 to silence)
#ifndef LT_ENABLE_LOGS
#define LT_ENABLE_LOGS 1
#endif

// Extra-verbose per-leaf/per-node logs (off by default)
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

// Note that while ComPtr is used to manage the lifetime of resources on the
// CPU, it has no understanding of the lifetime of resources on the GPU. Apps
// must account for the GPU lifetime of resources to avoid destroying objects
// that may still be referenced by the GPU. An example of this can be found in
// the class method: OnDestroy().
using Microsoft::WRL::ComPtr;

// Lut settings
constexpr int NUM_LUTS = 2;
constexpr int LUT_RESOLUTION = 16;
constexpr int NUM_SAMPLES_LUT = 32000;

static const UINT MAX_STACKS = 4;
static const UINT MAX_INDIRECT_COMMANDS = MAX_STACKS;

class Renderer : public DXSample {
public:
    void BuildGlobalMeshBuffers();

    void CreateTriToLightIdBuffer();

  Renderer(UINT width, UINT height, std::wstring name);

  void DLSSRR_Init();

  virtual void OnInit();
  virtual void OnUpdate();
  virtual void OnRender();
  virtual void OnDestroy();

private:
    // ── utilities ───────────────────────────────────────────────────────────────
    // 1. Update the Stage Enum to include control flow
    enum class Stage {
        RayGen,
        Compute,
        FixedCompute,
        Wavefront,
        Barrier,
        LoopStart,
        LoopEnd,
        PingSwap,
        ClearSort,
        Callable
    };

    // 2. Update PassDesc to hold loop information
    struct PassDesc
    {
        std::wstring  file;               // *.hlsl
        Stage         stage   = Stage::RayGen;
        uint32_t      groupX  = 0, groupY = 0;   // CS
        uint32_t      psoIdx  = UINT32_MAX;      // CS
        bool          isWorkGraph = false;
        uint32_t      wgIdx  = UINT32_MAX;       // index into state-object array

        // Loop specific data
        uint32_t      loopCount = 0;     // How many times to loop (for LoopStart)
        int32_t       targetIdx = -1;    // Index to jump to (LoopEnd -> LoopStart)
    };

    // 3. Update ParsePass to detect new keywords
    static PassDesc ParsePass(const std::wstring& token)
    {
        PassDesc p{};

        // --- Control Flow & Barriers ---
        if (token == L"barrier")   { p.stage = Stage::Barrier;   return p; }
        if (token == L"pingswap")  { p.stage = Stage::PingSwap;  return p; }
        if (token == L"endloop")   { p.stage = Stage::LoopEnd;   return p; }
        if (token == L"clearsort") { p.stage = Stage::ClearSort; return p; }

        if (token.rfind(L"loop:", 0) == 0) {
            p.stage = Stage::LoopStart;
            if (swscanf_s(token.c_str() + 5, L"%u", &p.loopCount) != 1)
                throw std::runtime_error("Invalid loop count format");
            return p;
        }

        // --- Shader Files ---
        const size_t bar = token.find(L'|');
        p.file = token.substr(0, bar);
        if (bar == std::wstring::npos) return p; // Assume RayGen if no pipe

        const std::wstring tail = token.substr(bar + 1);

        if (tail == L"rg" || tail == L"raygen") return p; // Default is Stage::RayGen
        if (tail == L"call") {
            p.stage = Stage::Callable;
            return p;
        }

        // Work Graph
        if (tail.rfind(L"wg:", 0) == 0) {
            p.stage = Stage::Compute;
            p.isWorkGraph = true;
            swscanf_s(tail.c_str() + 3, L"%ux%u", &p.groupX, &p.groupY);
            return p;
        }

        // Wavefront (Indirect)
        if (tail.rfind(L"wf:", 0) == 0) {
            p.stage = Stage::Wavefront;
            if (swscanf_s(tail.c_str() + 3, L"%u", &p.groupX) != 1)
                throw std::runtime_error("Invalid wf size");
            p.groupY = 1;
            return p;
        }

        // Standard Compute (Dense)
        if (tail.rfind(L"cs:", 0) == 0) {
            p.stage = Stage::Compute;
            if (swscanf_s(tail.c_str() + 3, L"%ux%u", &p.groupX, &p.groupY) != 2)
                throw std::runtime_error("Invalid cs size");
            return p;
        }

        // Fixed Compute (Direct, Explicit Group Count) <--- NEW
        if (tail.rfind(L"fx:", 0) == 0) {
            p.stage = Stage::FixedCompute;
            if (swscanf_s(tail.c_str() + 3, L"%u", &p.groupX) != 1) // e.g., fx:1
                throw std::runtime_error("Invalid fx size");
            p.groupY = 1;
            return p;
        }

        throw std::runtime_error("Unknown stage spec in pass string");
    }

    // 4. NEW: Helper to link "endloop" back to "loop" and validate nesting.
    // Call this in your Constructor/OnInit after parsing all strings into m_passes.
    void LinkLoops()
    {
        std::vector<size_t> stack;
        for (size_t i = 0; i < m_passes.size(); ++i)
        {
            if (m_passes[i].stage == Stage::LoopStart)
            {
                stack.push_back(i);
            }
            else if (m_passes[i].stage == Stage::LoopEnd)
            {
                if (stack.empty())
                    throw std::runtime_error("Found 'endloop' without matching 'loop:'");

                size_t startIdx = stack.back();
                stack.pop_back();

                // Link the end node to the start node
                m_passes[i].targetIdx = static_cast<int32_t>(startIdx);
            }
        }
        if (!stack.empty())
            throw std::runtime_error("Found 'loop:' without matching 'endloop'");
    }
    std::vector<std::wstring>                    m_passSequence;
    std::unordered_map<std::wstring, uint32_t>   m_passIndex;
    std::vector<Microsoft::WRL::ComPtr<IDxcBlob>> m_rayGenLibs;

    //  Global arrays used by EvalSurface() in the inline‑ray‑query path
    struct BTriVertex          // same layout as in HLSL
    {
        XMFLOAT3 vertex;
        UINT     packedNormal;
        PackedVector::XMHALF2  texCoord;
    };

    ComPtr<ID3D12Resource> m_vertexGlobal;
    ComPtr<ID3D12Resource> m_indexGlobal;
    UINT                   m_totalVertexCount = 0;
    UINT                   m_totalIndexCount  = 0;


  static const UINT FrameCount = 2;

    // Streamline frame and viewport tracking
    sl::FrameToken*     m_frameToken     = nullptr;
    sl::ViewportHandle  m_viewportHandle = sl::ViewportHandle(0);
    sl::DLSSDOptions     m_dlssdOptions   {};
    sl::Constants        m_slConstants    {};

    std::vector<PassDesc> m_passes;
    std::vector<ComPtr<ID3D12PipelineState>> m_csPSOs;
    std::vector<ComPtr<ID3D12PipelineState>> m_wgPSOs;

    std::vector<ComPtr<ID3D12Resource>> m_textureUploadHeaps;

    std::vector<std::vector<Vertex>> m_cpuVertexData;
    std::vector<std::vector<UINT>> m_cpuIndexData;

    struct WgRuntimeData
    {
        D3D12_PROGRAM_IDENTIFIER        id;
        D3D12_GPU_VIRTUAL_ADDRESS_RANGE backing;
        ComPtr<ID3D12Resource>          backingRes;
    };
    std::vector<WgRuntimeData> m_wgRuntime;


  // Pipeline objects.
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

  // App resources.
  ComPtr<ID3D12Resource> m_vertexBuffer;
  D3D12_VERTEX_BUFFER_VIEW m_vertexBufferView;

    bool g_keys[256] = {};
  // Synchronization objects.
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

  // #DXR
  struct AccelerationStructureBuffers {
    ComPtr<ID3D12Resource> pScratch;      // Scratch memory for AS builder
    ComPtr<ID3D12Resource> pResult;       // Where the AS is
    ComPtr<ID3D12Resource> pResultUncompacted;
    ComPtr<ID3D12Resource> pInstanceDesc; // Hold the matrices of the instances
  };

  ComPtr<ID3D12Resource> m_bottomLevelAS; // Storage for the bottom Level AS

  nv_helpers_dx12::TopLevelASGenerator m_topLevelASGenerator;
  AccelerationStructureBuffers m_topLevelASBuffers;
  std::vector<std::pair<ComPtr<ID3D12Resource>, DirectX::XMMATRIX>> m_instances;

    // Map from instance index to model index
    std::vector<UINT> m_instanceModelIndices;
    std::vector<UINT> m_materialIDOffsets;

    // alias table
    std::vector<float> m_aliasProb;
    // probability array
    std::vector<uint32_t> m_aliasIdx;
    // alias‑index array
    ComPtr<ID3D12Resource> m_aliasProbBuffer;
    // default‑heap GPU copies
    ComPtr<ID3D12Resource> m_aliasIdxBuffer;
    ComPtr<ID3D12Resource> m_initialBSDFRayBuffer;

    ComPtr<ID3D12Resource> m_pathStateBuffer;
    void CreatePathStateBuffer();

    lt::LightTreeBuilder m_lightTree;

    struct Reservoir_DI
    {
        uint8_t  pad[100];
    };

    struct Reservoir_GI
    {
        uint8_t  pad[100];
    };

    struct SampleData
    {
        uint8_t  pad[100];
    };

    struct InitialBSDFRay
    {
        uint8_t  pad[100];
    };


// Buffer to store emissive triangles
    std::vector<LightTriangle> m_emissiveTriangles;
    ComPtr<ID3D12Resource> m_emissiveTrianglesBuffer;

    // Per-instance base into the tri->light map and the map itself
    std::vector<uint32_t>           m_instTriOffset;    // size = #instances
    std::vector<uint32_t>           m_triToLightId;     // size = sum over instances of tri-count
    ComPtr<ID3D12Resource>          m_triToLightIdBuffer;

    // LUTs
    ComPtr<ID3D12Resource> m_lutTextureArray;
    std::vector<ComPtr<ID3D12Resource>> m_lutUploadHeaps;

    void GenerateLutTextures();
    void CreateAndUploadLutArray(const std::vector<std::vector<float>>& allLutData,
                                 ComPtr<ID3D12Resource>& textureArrayResource,
                                 const std::wstring& resourceName);


    /// Create the acceleration structure of an instance
  ///
  /// \param     vVertexBuffers : pair of buffer and vertex count
  /// \return    AccelerationStructureBuffers for TLAS
  AccelerationStructureBuffers CreateBottomLevelAS(
      std::vector<std::pair<ComPtr<ID3D12Resource>, uint32_t>> vVertexBuffers,
      std::vector<std::pair<ComPtr<ID3D12Resource>, uint32_t>> vIndexBuffers =
          {});

  /// Create the main acceleration structure that holds
  /// all instances of the scene
  /// \param     instances : pair of BLAS and transform
  // #DXR Extra - Refitting
  /// \param     updateOnly: if true, perform a refit instead of a full build
  void CreateTopLevelAS(
      const std::vector<std::pair<ComPtr<ID3D12Resource>, DirectX::XMMATRIX>>
          &instances,
      bool updateOnly = false);

  /// Create all acceleration structures, bottom and top
  void CreateAccelerationStructures();

  // #DXR
  ComPtr<ID3D12RootSignature> CreateRayGenSignature();

  ComPtr<ID3D12RootSignature> CreateComputeSignature();

  ComPtr<ID3D12RootSignature> CreateMissSignature();
  ComPtr<ID3D12RootSignature> CreateHitSignature();

  void CreateRaytracingPipeline();

  ComPtr<IDxcBlob> m_rayGenLibrary;
  ComPtr<IDxcBlob> m_rayGenLibrary2;
  ComPtr<IDxcBlob> m_rayGenLibrary3;
  ComPtr<IDxcBlob> m_hitLibrary;
  ComPtr<IDxcBlob> m_missLibrary;

  ComPtr<ID3D12RootSignature> m_rayGenSignature;
  ComPtr<ID3D12RootSignature> m_hitSignature;
  ComPtr<ID3D12RootSignature> m_missSignature;

    std::vector<std::wstring> m_callableShaderNames;

  // Ray tracing pipeline state
  ComPtr<ID3D12StateObject> m_rtStateObject;
  // Ray tracing pipeline state properties, retaining the shader identifiers
  // to use in the Shader Binding Table
  ComPtr<ID3D12StateObjectProperties> m_rtStateObjectProps;

    std::vector< ComPtr<ID3D12StateObject>          > m_wgStateObjects;
    std::vector< ComPtr<ID3D12WorkGraphProperties>  > m_wgProps;

  // #DXR
  void CreateRaytracingOutputBuffer();
  void CreateShaderResourceHeap();
  ComPtr<ID3D12Resource> m_outputResource;
    ComPtr<ID3D12Resource> m_dlssOutputBuffer;
    ComPtr<ID3D12Resource> m_permanentDataTexture;
    ComPtr<ID3D12Resource> m_scratchPing;
    ComPtr<ID3D12Resource> m_svgfConstBuffer;
    // Heaps
    ComPtr<ID3D12DescriptorHeap> m_srvUavHeap;     // Main Shader-Visible Heap
    ComPtr<ID3D12DescriptorHeap> m_stagingUavHeap; // Non-Shader-Visible Heap (Required for ClearUAV)

    // Handles for Sort Buffers
    D3D12_GPU_DESCRIPTOR_HANDLE m_sortCountGpuHandle;
    D3D12_CPU_DESCRIPTOR_HANDLE m_sortCountCpuHandle;

    D3D12_GPU_DESCRIPTOR_HANDLE m_sortOffsetGpuHandle;
    D3D12_CPU_DESCRIPTOR_HANDLE m_sortOffsetCpuHandle;

    D3D12_GPU_DESCRIPTOR_HANDLE m_sortBoundsGpuHandle;
    D3D12_CPU_DESCRIPTOR_HANDLE m_sortBoundsCpuHandle;

  // #DXR
  void CreateShaderBindingTable();
  nv_helpers_dx12::ShaderBindingTableGenerator m_sbtHelper;
  ComPtr<ID3D12Resource> m_sbtStorage;

  // #DXR Extra: Perspective Camera
  void CreateCameraBuffer();
  void UpdateCameraBuffer();
  ComPtr<ID3D12Resource> m_cameraBuffer;
  ComPtr<ID3D12Resource> m_sampleBuffer_current;
  ComPtr<ID3D12Resource> m_sampleBuffer_last;
  ComPtr<ID3D12Resource> m_reservoirBuffer;
  ComPtr<ID3D12Resource> m_reservoirBuffer_2;
  ComPtr<ID3D12Resource> m_reservoirBuffer_3;
  ComPtr<ID3D12Resource> m_reservoirBuffer_4;
  ComPtr<ID3D12DescriptorHeap> m_constHeap;
  uint32_t m_cameraBufferSize = 0;

  // #DXR Extra: Perspective Camera++
  void OnButtonDown(UINT32 lParam);
  void OnMouseMove(UINT8 wParam, UINT32 lParam);
    XMMATRIX m_prevViewMatrix;
    XMMATRIX m_prevProjMatrix;

  // #DXR Extra: Per-Instance Data
  ComPtr<ID3D12Resource> m_planeBuffer;
  D3D12_VERTEX_BUFFER_VIEW m_planeBufferView;
  void CreatePlaneVB();

  // #DXR Extra: Per-Instance Data
  void CreateGlobalConstantBuffer();
  ComPtr<ID3D12Resource> m_globalConstantBuffer;

  // #DXR Extra: Per-Instance Data
  void CreatePerInstanceConstantBuffers();
  std::vector<ComPtr<ID3D12Resource>> m_perInstanceConstantBuffers;

  // #DXR Extra: Depth Buffering
  void CreateDepthBuffer();
  ComPtr<ID3D12DescriptorHeap> m_dsvHeap;
  ComPtr<ID3D12Resource> m_depthStencil;

  ComPtr<ID3D12Resource> m_indexBuffer;
  D3D12_INDEX_BUFFER_VIEW m_indexBufferView;

  // #DXR Extra: Indexed Geometry
  void CreateVB(std::string name);
  ComPtr<ID3D12Resource> m_materialBuffer;
  ComPtr<ID3D12Resource> m_materialIndexBuffer;
  std::vector<UINT> m_materialIDs;
  std::vector<Material> m_materials;
  UINT materialIDOffset = 0;
  UINT materialVertexOffset = 0;

    ComPtr<ID3D12Resource> m_albedoTextureArray;
    ComPtr<ID3D12Resource> m_normalTextureArray;
    ComPtr<ID3D12Resource> m_rmaTextureArray;

  //Support for several objects (instanced optionally)
  std::vector<ComPtr<ID3D12Resource>> m_VB;
  std::vector<ComPtr<ID3D12Resource>> m_IB;
  std::vector<D3D12_VERTEX_BUFFER_VIEW> m_VBView;
  std::vector<D3D12_INDEX_BUFFER_VIEW> m_IBView;
  std::vector<ComPtr<ID3D12Resource>> m_material;
  std::vector<ComPtr<ID3D12Resource>> m_materialID;
  std::vector<UINT> m_IndexCount;
  std::vector<UINT> m_VertexCount;


  // #DXR Extra - Another ray type
  ComPtr<IDxcBlob> m_shadowLibrary;
  ComPtr<ID3D12RootSignature> m_shadowSignature;

  // #DXR Extra - Refitting
  uint32_t m_time = 0;

  // #DXR Extra - Refitting
  /// Per-instance properties
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
  };

    //Frametime
    struct FrameData
    {
        float Time;
    };

  ComPtr<ID3D12Resource> m_instanceProperties;
  ComPtr<ID3D12Resource> m_instancePropertiesPrevious;
    struct GeometryOffsets
    {
        UINT vertexBase;
        UINT indexBase;
        UINT materialBase;
    };
    std::vector<GeometryOffsets> m_geoOffsets;
  void CreateInstancePropertiesBuffer();
  void UpdateInstancePropertiesBuffer();

  //SL specific
  HINSTANCE__ *m_mod;

  UINT m_currentDisplayLevel = 0; // Start with the main image at level 0
  std::vector<UINT> m_displayLevels = {0, 10, 11, /*12, 13, 14, 15, 16, 17, 20,21,22,23,24,25,26,27,28*/}; // Levels to cycle through
  void ExtractFrustumPlanes(const XMMATRIX &viewProjMatrix, XMFLOAT4 *planes);


    void CollectEmissiveTriangles();

    void CreateEmissiveTrianglesBuffer();

  void BuildAliasTableSoA(const std::vector<LightTriangle> &tris);

  void CreateAliasBuffers();
    void CreateTextureArrays(
    const std::vector<TextureData>& albedoTextures,
    const std::vector<TextureData>& normalTextures,
    const std::vector<TextureData>& rmaTextures);

  float ComputeTriangleWeight(const XMFLOAT3 &v0, const XMFLOAT3 &v1, const XMFLOAT3 &v2, const XMFLOAT3 &emissiveColor, const DirectX::XMMATRIX &M);

    // --- Streaming Compaction Resources ---
    ComPtr<ID3D12Resource> m_stackBuffers[MAX_STACKS];    // The ping-pong stacks (hold uint pixel indices)
    ComPtr<ID3D12Resource> m_globalCounterBuffer;         // Holds one UINT counter per stack
    ComPtr<ID3D12Resource> m_indirectArgsBuffer;          // Holds D3D12_DISPATCH_ARGUMENTS
    ComPtr<ID3D12CommandSignature> m_commandSignature;    // Defines the indirect execution layout

    ComPtr<ID3D12PipelineState> m_psoSetupIndirect;       // Pipeline for the helper shader
    ComPtr<ID3D12PipelineState> m_psoSetupIndirectNoClear;
    ComPtr<ID3D12Resource> m_sortBoundsResetBuffer;
    ComPtr<ID3D12RootSignature> m_rsSetupIndirect;        // Root sig for the helper shader

    // Helpers
    void CreateStreamingCompactionBuffers();
    void CreateIndirectCommandSignature();
    void CompileSetupIndirectShader();
    ComPtr<ID3D12Resource> m_zeroBuffer;

    // Sorting Resources
    ComPtr<ID3D12Resource> m_sortCountBuffer;
    ComPtr<ID3D12Resource> m_sortOffsetBuffer;
    ComPtr<ID3D12Resource> m_sortBoundsBuffer;

    // Constants
    static const UINT SORT_BUCKETS = 65536;
    void ClearSortBuffers(ID3D12GraphicsCommandList* cmdList);

    ComPtr<ID3D12Resource> m_readbackBuffer;

    void CreateReadbackBuffer();

    void SaveSimulationData(uint32_t stepIndex);

    CameraRecorder m_recorder;
    CameraPathSimulator m_simulator;
};
