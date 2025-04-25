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
#include <chrono>
#include "stdafx.h"
#include <unordered_map>          // NEW
#include "Renderer.h"

#include "DXRHelper.h"
#include "nv_helpers_dx12/BottomLevelASGenerator.h"

#include "nv_helpers_dx12/RaytracingPipelineGenerator.h"
#include "nv_helpers_dx12/RootSignatureGenerator.h"

#include "Windowsx.h"
#include "glm/gtc/type_ptr.hpp"
#include "manipulator.h"
#include "../src/Util/ObjLoader.h"

// This is a static/global to store the last time we actually rendered a frame.
static std::chrono::steady_clock::time_point g_lastRenderTime
    = std::chrono::steady_clock::now();

// Our desired interval: 1 frame every 5 seconds => 0.2 FPS
static const float FRAME_INTERVAL_SECONDS = 10.00f;

Renderer::Renderer(UINT width, UINT height,
                   std::wstring name)
    : DXSample(width, height, name), m_frameIndex(0),
      m_viewport(0.0f, 0.0f, static_cast<float>(width),
                 static_cast<float>(height)),
      m_scissorRect(0, 0, static_cast<LONG>(width), static_cast<LONG>(height)),
      m_rtvDescriptorSize(0) {
    m_mod = LoadLibrary("sl.interposer.dll");

    m_passSequence = {
        L"Pass_init_di_v7.hlsl",
        /*L"barrier",
        L"Pass_init_gi_v7.hlsl",
        L"barrier",
        L"Pass_temp_di_v7.hlsl",
        L"Pass_temp_gi_v7.hlsl",
        L"barrier",
        L"Pass_spat_di_v7.hlsl",
        L"Pass_spat_gi_v7.hlsl",*/
        L"barrier",
        L"Pass_shading_v7.hlsl",
    };
}


void Renderer::OnInit() {

  nv_helpers_dx12::CameraManip.setWindowSize(GetWidth(), GetHeight());
  nv_helpers_dx12::CameraManip.setLookat(
      glm::vec3(-1.5f, 1.5f, 3.5f), glm::vec3(0, 1.0f, 0), glm::vec3(0, 1, 0));

  LoadPipeline();
  LoadAssets();

  // Check the raytracing capabilities of the device
  CheckRaytracingSupport();

  // Setup the acceleration structures (AS) for raytracing. When setting up
  // geometry, each bottom-level AS has its own transform matrix.
  CreateAccelerationStructures();

  // Command lists are created in the recording state, but there is
  // nothing to record yet. The main loop expects it to be closed, so
  // close it now.
  ThrowIfFailed(m_commandList->Close());

  // Create the raytracing pipeline, associating the shader code to symbol names
  // and to their root signatures, and defining the amount of memory carried by
  // rays (ray payload)
  CreateRaytracingPipeline(); // #DXR

  // #DXR Extra: Per-Instance Data
  CreatePerInstanceConstantBuffers();

  // #DXR Extra: Per-Instance Data
  // Create a constant buffers, with a color for each vertex of the triangle,
  // for each triangle instance
  CreateGlobalConstantBuffer();

  // Allocate the buffer storing the raytracing output, with the same dimensions
  // as the target image
  CreateRaytracingOutputBuffer(); // #DXR

  // #DXR Extra - Refitting
  CreateInstancePropertiesBuffer();

  // #DXR Extra: Perspective Camera
  // Create a buffer to store the modelview and perspective camera matrices
  CreateCameraBuffer();

  // Create the buffer containing the raytracing result (always output in a
  // UAV), and create the heap referencing the resources used by the raytracing,
  // such as the acceleration structure
  CreateShaderResourceHeap(); // #DXR

  // Create the shader binding table and indicating which shaders
  // are invoked for each instance in the  AS
  CreateShaderBindingTable();

    slGetNewFrameToken(m_frameToken, nullptr);   // token is valid forever, SL recycles it internally




}

// Load the rendering pipeline dependencies.
void Renderer::LoadPipeline() {
    // 3.1 Build the preferences
    sl::Preferences pref{};
    pref.flags  = sl::PreferenceFlags::eDisableCLStateTracking |
              sl::PreferenceFlags::eLoadDownloadedPlugins;
    static sl::Feature featList[] = { sl::kFeatureDLSS, sl::kFeatureDLSS_RR };
    pref.featuresToLoad    = featList;
    pref.numFeaturesToLoad = _countof(featList);

    // 3.2 Initialize Streamline and give it our D3D12 device
    slInit(pref, sl::kSDKVersion);                             // :contentReference[oaicite:0]{index=0}

  UINT dxgiFactoryFlags = 0;
    // These are the exports from SL library
    typedef HRESULT(WINAPI* PFunCreateDXGIFactory)(REFIID, void**);
    typedef HRESULT(WINAPI* PFunCreateDXGIFactory1)(REFIID, void**);
    typedef HRESULT(WINAPI* PFunCreateDXGIFactory2)(UINT, REFIID, void**);
    typedef HRESULT(WINAPI* PFunDXGIGetDebugInterface1)(UINT, REFIID, void**);
    typedef HRESULT(WINAPI* PFunD3D12CreateDevice)(IUnknown* , D3D_FEATURE_LEVEL, REFIID , void**);

    #if defined(_DEBUG)
        // Enable the debug layer (requires the Graphics Tools "optional feature").
        // NOTE: Enabling the debug layer after device creation will invalidate the
        // active device.
        {
          ComPtr<ID3D12Debug> debugController;
          if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debugController)))) {
              debugController->EnableDebugLayer();

              // Enable additional debug layers.
              dxgiFactoryFlags |= DXGI_CREATE_FACTORY_DEBUG;
          }
        }
    #endif

    // Map functions from SL and use them instead of standard DXGI/D3D12 API
    auto slCreateDXGIFactory = reinterpret_cast<PFunCreateDXGIFactory>(GetProcAddress(m_mod, "CreateDXGIFactory"));
    auto slCreateDXGIFactory1 = reinterpret_cast<PFunCreateDXGIFactory1>(GetProcAddress(m_mod, "CreateDXGIFactory1"));
    auto slCreateDXGIFactory2 = reinterpret_cast<PFunCreateDXGIFactory2>(GetProcAddress(m_mod, "CreateDXGIFactory2"));
    auto slDXGIGetDebugInterface1 = reinterpret_cast<PFunDXGIGetDebugInterface1>(GetProcAddress(m_mod, "DXGIGetDebugInterface1"));
    auto slD3D12CreateDevice = reinterpret_cast<PFunD3D12CreateDevice>(GetProcAddress(m_mod, "D3D12CreateDevice"));


  ComPtr<IDXGIFactory4> factory;
  ThrowIfFailed(slCreateDXGIFactory2(dxgiFactoryFlags, IID_PPV_ARGS(&factory)));

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
  }
  if(SL_FAILED(res, slSetD3DDevice(m_device.Get())))
{
    // Handle error, check the logs
}


    // Using helpers from sl_dlss.h

    sl::DLSSOptimalSettings dlssSettings;
    sl::DLSSOptions dlssOptions;
    // These are populated based on user selection in the UI
    dlssOptions.mode = sl::DLSSMode::eDLAA; // e.g. sl::eDLSSModeBalanced;
    dlssOptions.outputWidth = 1920;    // e.g 1920;
    dlssOptions.outputHeight = 1080; // e.g. 1080;
    // Now let's check what should our rendering resolution be
    slDLSSGetOptimalSettings(dlssOptions, dlssSettings);
    // print the optimal settings:
    std::wcout << L"DLSS settings: " << dlssSettings.renderHeightMax << std::endl;



  // Describe and create the command queue.
  D3D12_COMMAND_QUEUE_DESC queueDesc = {};
  queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
  queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;

  ThrowIfFailed(
      m_device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&m_commandQueue)));

  // Describe and create the swap chain.
  DXGI_SWAP_CHAIN_DESC1 swapChainDesc = {};
  swapChainDesc.BufferCount = FrameCount;
  swapChainDesc.Width = m_width;
  swapChainDesc.Height = m_height;
  swapChainDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
  swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
  swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
  swapChainDesc.SampleDesc.Count = 1;

  ComPtr<IDXGISwapChain1> swapChain;
  ThrowIfFailed(factory->CreateSwapChainForHwnd(
      m_commandQueue.Get(), // Swap chain needs the queue so that it can force a
                            // flush on it.
      Win32Application::GetHwnd(), &swapChainDesc, nullptr, nullptr,
      &swapChain));

  // This sample does not support fullscreen transitions.
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

  ThrowIfFailed(m_device->CreateCommandAllocator(
      D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&m_commandAllocator)));

  // #DXR Extra: Depth Buffering
  // The original sample does not support depth buffering, so we need to
  // allocate a depth buffer, and later bind it before rasterization
  CreateDepthBuffer();
}

// Load the sample assets.
void Renderer::LoadAssets() {
  // Create an empty root signature.
  {
    // #DXR Extra: Perspective Camera
    // The root signature describes which data is accessed by the shader. The
    // camera matrices are held in a constant buffer, itself referenced the
    // heap. To do this we reference a range in the heap, and use that range as
    // the sole parameter of the shader. The camera buffer is associated in the
    // index 0, making it accessible in the shader in the b0 register.
    CD3DX12_ROOT_PARAMETER constantParameter;
    CD3DX12_DESCRIPTOR_RANGE range;
    range.Init(D3D12_DESCRIPTOR_RANGE_TYPE_CBV, 1, 0);
    constantParameter.InitAsDescriptorTable(1, &range,
                                            D3D12_SHADER_VISIBILITY_ALL);

    // #DXR Extra - Refitting
    // Per-instance properties buffer
    CD3DX12_ROOT_PARAMETER matricesParameter;
    CD3DX12_DESCRIPTOR_RANGE matricesRange;
    matricesRange.Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1 /* desc count*/,
                       0 /*register*/, 0 /*space*/, 1 /*heap slot*/);
    matricesParameter.InitAsDescriptorTable(1, &matricesRange,
                                            D3D12_SHADER_VISIBILITY_ALL);

    // #DXR Extra - Refitting
    // Per-instance properties index for the current geometry
    CD3DX12_ROOT_PARAMETER indexParameter;
    indexParameter.InitAsConstants(1 /*value count*/, 1 /*register*/);

    // #DXR Extra - Refitting
    std::vector<CD3DX12_ROOT_PARAMETER> params = {
        constantParameter, matricesParameter, indexParameter};

    CD3DX12_ROOT_SIGNATURE_DESC rootSignatureDesc;
    rootSignatureDesc.Init(
        static_cast<UINT>(params.size()), params.data(), 0, nullptr,
        D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);

    ComPtr<ID3DBlob> signature;
    ComPtr<ID3DBlob> error;
    ThrowIfFailed(D3D12SerializeRootSignature(
        &rootSignatureDesc, D3D_ROOT_SIGNATURE_VERSION_1, &signature, &error));
    ThrowIfFailed(m_device->CreateRootSignature(
        0, signature->GetBufferPointer(), signature->GetBufferSize(),
        IID_PPV_ARGS(&m_rootSignature)));
  }

  // Create the pipeline state, which includes compiling and loading shaders.
  {
    ComPtr<ID3DBlob> vertexShader;
    ComPtr<ID3DBlob> pixelShader;

#if defined(_DEBUG)
    // Enable better shader debugging with the graphics debugging tools.
    UINT compileFlags = D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#else
    UINT compileFlags = 0;
#endif

    ThrowIfFailed(D3DCompileFromFile(L"shaders.hlsl",
                                     nullptr, nullptr, "VSMain", "vs_5_0",
                                     compileFlags, 0, &vertexShader, nullptr));
    ThrowIfFailed(D3DCompileFromFile(L"shaders.hlsl",
                                     nullptr, nullptr, "PSMain", "ps_5_0",
                                     compileFlags, 0, &pixelShader, nullptr));

    // Define the vertex input layout.
    D3D12_INPUT_ELEMENT_DESC inputElementDescs[] = {
        {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0,
         D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"COLOR", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 12,
         D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0}};

    // Describe and create the graphics pipeline state object (PSO).
    D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {};
    psoDesc.InputLayout = {inputElementDescs, _countof(inputElementDescs)};
    psoDesc.pRootSignature = m_rootSignature.Get();
    psoDesc.VS = CD3DX12_SHADER_BYTECODE(vertexShader.Get());
    psoDesc.PS = CD3DX12_SHADER_BYTECODE(pixelShader.Get());
    psoDesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
    psoDesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
    psoDesc.DepthStencilState.DepthEnable = FALSE;
    psoDesc.DepthStencilState.StencilEnable = FALSE;
    psoDesc.SampleMask = UINT_MAX;
    psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    psoDesc.NumRenderTargets = 1;
    psoDesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    psoDesc.SampleDesc.Count = 1;
    // #DXR Extra: Depth Buffering
    // Add support for depth testing, using a 32-bit floating-point depth buffer
    psoDesc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
    psoDesc.DSVFormat = DXGI_FORMAT_D32_FLOAT;

    // #DXR Extra - Refitting
    psoDesc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;

    ThrowIfFailed(m_device->CreateGraphicsPipelineState(
        &psoDesc, IID_PPV_ARGS(&m_pipelineState)));
  }

  // Create the command list.
  ThrowIfFailed(m_device->CreateCommandList(
      0, D3D12_COMMAND_LIST_TYPE_DIRECT, m_commandAllocator.Get(),
      m_pipelineState.Get(), IID_PPV_ARGS(&m_commandList)));

  {
    std::vector<std::string> models = {"garage.obj", "monke.obj"};



    //Iterate through the models in the scene (currently one hardcoded, later provided by list)
    for(int i=0; i<models.size(); i++){
        CreateVB(models[i]);
    }

    //Upload the models
      //Material:
      {
          const UINT materialBufferSize = static_cast<UINT>(m_materials.size()) * sizeof(Material);

          CD3DX12_HEAP_PROPERTIES heapProp = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
          CD3DX12_RESOURCE_DESC bufferRes = CD3DX12_RESOURCE_DESC::Buffer(materialBufferSize);
          ThrowIfFailed(m_device->CreateCommittedResource(
                  &heapProp, D3D12_HEAP_FLAG_NONE, &bufferRes, //
                  D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&m_materialBuffer)));

          // Copy material data to the buffer.
          UINT8* pMaterialDataBegin;
          CD3DX12_RANGE readRange(0, 0);
          ThrowIfFailed(m_materialBuffer->Map(0, &readRange, reinterpret_cast<void**>(&pMaterialDataBegin)));
          memcpy(pMaterialDataBegin, m_materials.data(), materialBufferSize);
          m_materialBuffer->Unmap(0, nullptr);
      }

      //Material Indices
      {
          const UINT materialIndexBufferSize = static_cast<UINT>(m_materialIDs.size()) * sizeof(UINT);

          CD3DX12_HEAP_PROPERTIES heapProp = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
          CD3DX12_RESOURCE_DESC bufferRes = CD3DX12_RESOURCE_DESC::Buffer(materialIndexBufferSize);
          ThrowIfFailed(m_device->CreateCommittedResource(
                  &heapProp, D3D12_HEAP_FLAG_NONE, &bufferRes, //
                  D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&m_materialIndexBuffer)));

          // Copy material index data to the buffer.
          UINT8* pMaterialIndexDataBegin;
          CD3DX12_RANGE readRange(0, 0);
          ThrowIfFailed(m_materialIndexBuffer->Map(0, &readRange, reinterpret_cast<void**>(&pMaterialIndexDataBegin)));
          memcpy(pMaterialIndexDataBegin, m_materialIDs.data(), materialIndexBufferSize);
          m_materialIndexBuffer->Unmap(0, nullptr);
      }
  }

  // Create synchronization objects and wait until assets have been uploaded to
  // the GPU.
  {
    ThrowIfFailed(m_device->CreateFence(0, D3D12_FENCE_FLAG_NONE,
                                        IID_PPV_ARGS(&m_fence)));
    m_fenceValue = 1;

    // Create an event handle to use for frame synchronization.
    m_fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    if (m_fenceEvent == nullptr) {
      ThrowIfFailed(HRESULT_FROM_WIN32(GetLastError()));
    }

    // Wait for the command list to execute; we are reusing the same command
    // list in our main loop but for now, we just want to wait for setup to
    // complete before continuing.
    WaitForPreviousFrame();
  }
}

// Update frame-based values.
void Renderer::OnUpdate() {
  // #DXR Extra: Perspective Camera
  UpdateCameraBuffer();


  // #DXR Extra - Refitting
  // Increment the time counter at each frame, and update the corresponding
  // instance matrix of the first triangle to animate its position
  m_time++;
  /*m_instances[1].second =
      XMMatrixRotationAxis({0.f, 1.f, 0.f},*/
                           //0.0f/*static_cast<float>(m_time) / 20000000.0f*/) *
      //XMMatrixTranslation(0.f, 0.f, 0.f);
    XMMATRIX scaleMatrix = XMMatrixScaling(1.0f, 1.0f, 1.0f);
    XMMATRIX rotationMatrix = XMMatrixRotationAxis({0.f, 1.f, 0.f}, 1.57f);
    XMMATRIX translationMatrix = XMMatrixTranslation(0.f, 1.f, 0.f);

    // Multiply them in the order Scale -> Rotate -> Translate
    m_instances[1].second = scaleMatrix * rotationMatrix * translationMatrix;
  // #DXR Extra - Refitting
  UpdateInstancePropertiesBuffer();
}

/*void Renderer::OnRender() {
    // Record all the commands we need to render the scene into the command list.
    PopulateCommandList();

    // Execute the command list.
    ID3D12CommandList* ppCommandLists[] = { m_commandList.Get() };
    m_commandQueue->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);
    // Present the frame
    ThrowIfFailed(m_swapChain->Present(1, 0));

    // Wait for the frame to finish
    WaitForPreviousFrame();
}*/

void Renderer::OnRender()
{
    static auto s_lastTime = std::chrono::high_resolution_clock::now();
    static int  s_frameCount = 0;

    // Normal rendering steps
    // ----------------------------------------
    PopulateCommandList();
    ID3D12CommandList* ppCommandLists[] = { m_commandList.Get() };
    m_commandQueue->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);
    ThrowIfFailed(m_swapChain->Present(0, 0));
    WaitForPreviousFrame();
    // ----------------------------------------

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

        // Build the string
        std::wstringstream ss;
        ss << std::fixed << std::setprecision(2)
               << L"Frame Time: " << dT << L" ms (" << fps << L" fps)";

        // Update the window title
        SetWindowTextW(Win32Application::GetHwnd(), ss.str().c_str());

        // Reset for next time
        s_frameCount = 0;
        s_lastTime = currentTime;
    }
}

/*void Renderer::OnRender()
{
    using namespace std::chrono;

    // 1) Check how long it's been since we last rendered.
    auto now = steady_clock::now();
    float elapsedSec = duration<float>(now - g_lastRenderTime).count();

    // 2) If < 5 seconds have passed, skip GPU work entirely -> GPU stays idle.
    if (elapsedSec < FRAME_INTERVAL_SECONDS)
    {
        // Optional: You can still process input messages or do CPU tasks,
        // but skip issuing any GPU commands or calls to Present().
        return;
    }

    // 3) Otherwise, it's time for a new frame -> do the normal render steps.

    // Record the time we last rendered
    g_lastRenderTime = now;

    // [A] Record GPU commands
    PopulateCommandList();

    // [B] Execute them
    ID3D12CommandList* ppCommandLists[] = { m_commandList.Get() };
    m_commandQueue->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);

    // [C] Present the frame (1, 0)
    ThrowIfFailed(m_swapChain->Present(1, 0));

    // [D] Wait for GPU to finish (or use your existing fence logic)
    WaitForPreviousFrame();
}*/


void Renderer::OnDestroy() {
  // Ensure that the GPU is no longer referencing resources that are about to be
  // cleaned up by the destructor.
  WaitForPreviousFrame();

  CloseHandle(m_fenceEvent);
    if(SL_FAILED(res, slShutdown()))
    {
        // Handle error, check the logs
    }
}

void Renderer::PopulateCommandList() {
  // Command list allocators can only be reset when the associated
  // command lists have finished execution on the GPU; apps should use
  // fences to determine GPU execution progress.
  ThrowIfFailed(m_commandAllocator->Reset());

  // However, when ExecuteCommandList() is called on a particular command
  // list, that command list can then be reset at any time and must be before
  // re-recording.
  ThrowIfFailed(
      m_commandList->Reset(m_commandAllocator.Get(), m_pipelineState.Get()));

  // Set necessary state.
  m_commandList->SetGraphicsRootSignature(m_rootSignature.Get());
  m_commandList->RSSetViewports(1, &m_viewport);
  m_commandList->RSSetScissorRects(1, &m_scissorRect);

    CD3DX12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::Transition(
            m_renderTargets[m_frameIndex].Get(),
            D3D12_RESOURCE_STATE_PRESENT,
            D3D12_RESOURCE_STATE_RENDER_TARGET);

    m_commandList->ResourceBarrier(1, &barrier);

  CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHandle(
      m_rtvHeap->GetCPUDescriptorHandleForHeapStart(), m_frameIndex,
      m_rtvDescriptorSize);
  // #DXR Extra: Depth Buffering
  // Bind the depth buffer as a render target
  CD3DX12_CPU_DESCRIPTOR_HANDLE dsvHandle(
      m_dsvHeap->GetCPUDescriptorHandleForHeapStart());

  m_commandList->OMSetRenderTargets(1, &rtvHandle, FALSE, &dsvHandle);

    // #DXR Extra - Refitting
    // Refit the top-level acceleration structure to account for the new
    // transform matrix of the triangle. Note that the build contains a barrier,
    // hence we can do the rendering in the same command list
    CreateTopLevelAS(m_instances, true);
    // #DXR
    // Bind the descriptor heap giving access to the top-level acceleration
    // structure, as well as the raytracing output
    std::vector<ID3D12DescriptorHeap *> heaps = {m_srvUavHeap.Get()};
    m_commandList->SetDescriptorHeaps(static_cast<UINT>(heaps.size()),
                                      heaps.data());

    // On the last frame, the raytracing output was used as a copy source, to
    // copy its contents into the render target. Now we need to transition it to
    // a UAV so that the shaders can write in it.
    CD3DX12_RESOURCE_BARRIER transition = CD3DX12_RESOURCE_BARRIER::Transition(
        m_outputResource.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    m_commandList->ResourceBarrier(1, &transition);

    D3D12_DISPATCH_RAYS_DESC desc = {};
    desc.Width  = GetWidth();
    desc.Height = GetHeight();
    desc.Depth  = 1;

    const uint64_t sbtStart = m_sbtStorage->GetGPUVirtualAddress();
    const uint32_t rgSize   = m_sbtHelper.GetRayGenEntrySize();
    const uint32_t numRG    = static_cast<uint32_t>(m_passIndex.size());

    desc.MissShaderTable.StartAddress  = sbtStart + numRG * rgSize;
    desc.MissShaderTable.SizeInBytes   = m_sbtHelper.GetMissSectionSize();
    desc.MissShaderTable.StrideInBytes = m_sbtHelper.GetMissEntrySize();

    desc.HitGroupTable.StartAddress    =
        desc.MissShaderTable.StartAddress + desc.MissShaderTable.SizeInBytes;
    desc.HitGroupTable.SizeInBytes     = m_sbtHelper.GetHitGroupSectionSize();
    desc.HitGroupTable.StrideInBytes   = m_sbtHelper.GetHitGroupEntrySize();

    m_commandList->SetPipelineState1(m_rtStateObject.Get());

    uint32_t slot = 0;
    for (const auto& entry : m_passSequence) {

        if (entry == L"barrier") {                 // UAV barrier request
            CD3DX12_RESOURCE_BARRIER uavBarrier =
                CD3DX12_RESOURCE_BARRIER::UAV(nullptr);
            m_commandList->ResourceBarrier(1, &uavBarrier);
            continue;
        }

        // Dispatch the ray‑generation shader that lives at `slot`
        desc.RayGenerationShaderRecord.StartAddress = sbtStart + slot * rgSize;
        desc.RayGenerationShaderRecord.SizeInBytes  = rgSize;
        m_commandList->DispatchRays(&desc);
        ++slot;
    }

    // The raytracing output needs to be copied to the actual render target used
    // for display. For this, we need to transition the raytracing output from a
    // UAV to a copy source, and the render target buffer to a copy destination.
    // We can then do the actual copy, before transitioning the render target
    // buffer into a render target, that will be then used to display the image
    // Transition the raytracing output texture array from UAV to COPY_SOURCE
    transition = CD3DX12_RESOURCE_BARRIER::Transition(
            m_outputResource.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    m_commandList->ResourceBarrier(1, &transition);

    // Transition the current render target from RENDER_TARGET to COPY_DEST
    transition = CD3DX12_RESOURCE_BARRIER::Transition(
            m_renderTargets[m_frameIndex].Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_DEST);
    m_commandList->ResourceBarrier(1, &transition);

    UINT selectedLayer = m_displayLevels[m_currentDisplayLevel];
    // Calculate the subresource index for the specific layer of the texture array
    UINT subresourceIndex = D3D12CalcSubresource(0, selectedLayer, 0, 1, 30);
    CD3DX12_TEXTURE_COPY_LOCATION src(m_outputResource.Get(), subresourceIndex);
    CD3DX12_TEXTURE_COPY_LOCATION dest(m_renderTargets[m_frameIndex].Get(), 0);

    // Define the region to copy - in this case, the whole layer
    D3D12_BOX srcBox = {0, 0, 0, static_cast<UINT>(m_width), static_cast<UINT>(m_height), 1};
    m_commandList->CopyTextureRegion(&dest, 0, 0, 0, &src, &srcBox);

    // Transition the render target back to RENDER_TARGET to be used for presentation
    transition = CD3DX12_RESOURCE_BARRIER::Transition(
            m_renderTargets[m_frameIndex].Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_RENDER_TARGET);
    m_commandList->ResourceBarrier(1, &transition);

  //}

    barrier = CD3DX12_RESOURCE_BARRIER::Transition(
        m_renderTargets[m_frameIndex].Get(),
        D3D12_RESOURCE_STATE_RENDER_TARGET,
        D3D12_RESOURCE_STATE_PRESENT
    );
    m_commandList->ResourceBarrier(1, &barrier);

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
}

void Renderer::CheckRaytracingSupport() {
  D3D12_FEATURE_DATA_D3D12_OPTIONS5 options5 = {};
  ThrowIfFailed(m_device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5,
                                              &options5, sizeof(options5)));
  if (options5.RaytracingTier < D3D12_RAYTRACING_TIER_1_0)
    throw std::runtime_error("Raytracing not supported on device");
}

//-----------------------------------------------------------------------------
//
//
void Renderer::OnKeyUp(UINT8 key) {
    // Check if a specific key (e.g., 'C' for cycle) is pressed
    if (key == 'C') {
        m_currentDisplayLevel = (m_currentDisplayLevel + 1) % m_displayLevels.size();
        std::wcout << L"C key pressed, switching to level: " << m_currentDisplayLevel << std::endl;
    }

    if (key == VK_SPACE) {
        m_raster = !m_raster;
        std::wcout << L"Space key pressed, toggling rasterization: " << m_raster << std::endl;
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
Renderer::AccelerationStructureBuffers
Renderer::CreateBottomLevelAS(
    std::vector<std::pair<ComPtr<ID3D12Resource>, uint32_t>> vVertexBuffers,
    std::vector<std::pair<ComPtr<ID3D12Resource>, uint32_t>> vIndexBuffers) {
  nv_helpers_dx12::BottomLevelASGenerator bottomLevelAS;

  // Adding all vertex buffers and not transforming their position.
  for (size_t i = 0; i < vVertexBuffers.size(); i++) {
    // for (const auto &buffer : vVertexBuffers) {
    if (i < vIndexBuffers.size() && vIndexBuffers[i].second > 0)
      bottomLevelAS.AddVertexBuffer(vVertexBuffers[i].first.Get(), 0,
                                    vVertexBuffers[i].second, sizeof(Vertex),
                                    vIndexBuffers[i].first.Get(), 0,
                                    vIndexBuffers[i].second, nullptr, 0, true);

    else
      bottomLevelAS.AddVertexBuffer(vVertexBuffers[i].first.Get(), 0,
                                    vVertexBuffers[i].second, sizeof(Vertex), 0,
                                    0);
  }

  // The AS build requires some scratch space to store temporary information.
  // The amount of scratch memory is dependent on the scene complexity.
  UINT64 scratchSizeInBytes = 0;
  // The final AS also needs to be stored in addition to the existing vertex
  // buffers. It size is also dependent on the scene complexity.
  UINT64 resultSizeInBytes = 0;

  bottomLevelAS.ComputeASBufferSizes(m_device.Get(), false, &scratchSizeInBytes,
                                     &resultSizeInBytes);

  // Once the sizes are obtained, the application is responsible for allocating
  // the necessary buffers. Since the entire generation will be done on the GPU,
  // we can directly allocate those on the default heap
  AccelerationStructureBuffers buffers;
  buffers.pScratch = nv_helpers_dx12::CreateBuffer(
      m_device.Get(), scratchSizeInBytes,
      D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON,
      nv_helpers_dx12::kDefaultHeapProps);
  buffers.pResult = nv_helpers_dx12::CreateBuffer(
      m_device.Get(), resultSizeInBytes,
      D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
      D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
      nv_helpers_dx12::kDefaultHeapProps);

  // Build the acceleration structure. Note that this call integrates a barrier
  // on the generated AS, so that it can be used to compute a top-level AS right
  // after this method.
  bottomLevelAS.Generate(m_commandList.Get(), buffers.pScratch.Get(),
                         buffers.pResult.Get(), false, nullptr);

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

    std::vector<AccelerationStructureBuffers> blasBuffers;
    m_instances.clear();
    m_instanceModelIndices.clear();

    // Assume m_VB, m_IB, m_VertexCount, and m_IndexCount are all std::vector and have the same size.
    for (size_t i = 0; i < m_VB.size(); ++i) {
        AccelerationStructureBuffers buffers = CreateBottomLevelAS(
                {{m_VB[i].Get(), m_VertexCount[i]}},
                {{m_IB[i].Get(), m_IndexCount[i]}}
        );

        blasBuffers.push_back(buffers);

        // Assuming each instance will use an identity matrix for simplicity,
        // but you can replace XMMatrixIdentity() with any transformation matrix.
        m_instances.emplace_back(buffers.pResult, XMMatrixIdentity());
        m_instanceModelIndices.push_back(static_cast<UINT>(i));
    }
  CreateTopLevelAS(m_instances);
    // Collect emissive triangles
    CollectEmissiveTriangles();

    // Create buffer for emissive triangles
    CreateEmissiveTrianglesBuffer();

    // Build & upload alias table ---------------------------------------------
    BuildAliasTableSoA(m_emissiveTriangles);
    CreateAliasBuffers();

  // Flush the command list and wait for it to finish
  m_commandList->Close();
  ID3D12CommandList *ppCommandLists[] = {m_commandList.Get()};
  m_commandQueue->ExecuteCommandLists(1, ppCommandLists);
  m_fenceValue++;
  m_commandQueue->Signal(m_fence.Get(), m_fenceValue);

  m_fence->SetEventOnCompletion(m_fenceValue, m_fenceEvent);
  WaitForSingleObject(m_fenceEvent, INFINITE);

  // Once the command list is finished executing, reset it to be reused for
  // rendering
  ThrowIfFailed(
      m_commandList->Reset(m_commandAllocator.Get(), m_pipelineState.Get()));

  // Store the AS buffers. The rest of the buffers will be released once we exit
  // the function
  //m_bottomLevelAS = bottomLevelBuffers.pResult;
}

//-----------------------------------------------------------------------------
// The ray generation shader needs to access 2 resources: the raytracing output
// and the top-level acceleration structure
//

ComPtr<ID3D12RootSignature> Renderer::CreateRayGenSignature() {
    nv_helpers_dx12::RootSignatureGenerator rsc;

    rsc.AddHeapRangesParameter(
            {
                    {0 /*u0*/, 1 /*1 descriptor */, 0 /*use the implicit register space 0*/, D3D12_DESCRIPTOR_RANGE_TYPE_UAV /* UAV representing the output buffer*/,0 /*heap slot where the UAV is defined*/},
                    {1 /*u1*/, 1, 0, D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 7},
                    {0 /*t0*/, 1, 0, D3D12_DESCRIPTOR_RANGE_TYPE_SRV /*Top-level acceleration structure*/,1},
                    {0 /*b0*/, 1, 0, D3D12_DESCRIPTOR_RANGE_TYPE_CBV /*Camera parameters*/,2},
                    {3 /*t3*/, 1, 0, D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 3}, // **Added SRV for t3**
                    {4 /*t4*/, 1, 0, D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 4 /*5th slot - Material IDs*/},
                    {5 /*t5*/, 1, 0, D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 5 /*6th slot - Materials*/},
                    {6 /*t6*/, 1, 0, D3D12_DESCRIPTOR_RANGE_TYPE_SRV,6},
                    {2 /*u2*/, 1, 0, D3D12_DESCRIPTOR_RANGE_TYPE_UAV,8},
                    {3 /*u3*/, 1, 0, D3D12_DESCRIPTOR_RANGE_TYPE_UAV,9},
                    {4 /*u4*/, 1, 0, D3D12_DESCRIPTOR_RANGE_TYPE_UAV,10},
                    {5 /*u5*/, 1, 0, D3D12_DESCRIPTOR_RANGE_TYPE_UAV,11},
                    {6 /*u6*/, 1, 0, D3D12_DESCRIPTOR_RANGE_TYPE_UAV,12},
                    {7 /*u7*/, 1, 0, D3D12_DESCRIPTOR_RANGE_TYPE_UAV,13},
                    {7 /*t7*/, 1, 0, D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 14}, // aliasProb
                    {8 /*t8*/, 1, 0, D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 15} // aliasIdx
            }
    );

    return rsc.Generate(m_device.Get(), true);
}


//-----------------------------------------------------------------------------
// The hit shader communicates only through the ray payload, and therefore does
// not require any resources
//
ComPtr<ID3D12RootSignature> Renderer::CreateHitSignature() {
  nv_helpers_dx12::RootSignatureGenerator rsc;
  rsc.AddRootParameter(D3D12_ROOT_PARAMETER_TYPE_SRV,
                       2 /*t0*/); // vertices and colors
  rsc.AddRootParameter(D3D12_ROOT_PARAMETER_TYPE_SRV, 1 /*t1*/); // indices
  // #DXR Extra: Per-Instance Data
  // The vertex colors may differ for each instance, so it is not possible to
  // point to a single buffer in the heap. Instead we use the concept of root
  // parameters, which are defined directly by a pointer in memory. In the
  // shader binding table we will associate each hit shader instance with its
  // constant buffer. Here we bind the buffer to the first slot, accessible in
  // HLSL as register(b0)
  rsc.AddRootParameter(D3D12_ROOT_PARAMETER_TYPE_CBV, 0);
  // #DXR Extra - Another ray type
  // Add a single range pointing to the TLAS in the heap
    rsc.AddHeapRangesParameter(
            {{0 /*t2*/, 1, 0, D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1 /*2nd slot of the heap*/},
             //{0 /*b0*/, 1, 0, D3D12_DESCRIPTOR_RANGE_TYPE_CBV /*Scene data*/, 2},
                    // # DXR Extra - Simple Lighting
             {3 /*t3*/, 1, 0, D3D12_DESCRIPTOR_RANGE_TYPE_SRV /*Per-instance data*/, 3},
             {4 /*t4*/, 1, 0, D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 4 /*5th slot - Material IDs*/},
             {5 /*t5*/, 1, 0, D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 5 /*6th slot - Materials*/},
                    {6 /*t6*/, 1, 0, D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 6 /*7th slot - Light triangles*/}
            });
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
void Renderer::CreateRaytracingPipeline() {

    nv_helpers_dx12::RayTracingPipelineGenerator pipeline(m_device.Get());

    // 1)  RAY‑GENERATION LIBRARIES  (from .hlsl file names)
    // ------------------------------------------------------------------------
    m_rayGenLibs.clear();
    m_passIndex.clear();
    uint32_t shaderSlot = 0;

    for (const auto& entry : m_passSequence) {

        if (entry == L"barrier")           // skip barriers
            continue;

        const std::wstring& fileName = entry;

        // Derive the exported symbol:  strip folder & extension
        std::wstring base = fileName.substr(fileName.find_last_of(L"/\\") + 1);
        base = base.substr(0, base.rfind(L'.'));

        // NEW – always a WRL smart pointer, so .Get() is available
        Microsoft::WRL::ComPtr<IDxcBlob> lib =
                nv_helpers_dx12::CompileShaderLibrary(fileName.c_str());

        m_rayGenLibs.push_back(lib);                         // keep it alive
        pipeline.AddLibrary(lib.Get(), { base.c_str() });    // ← now compiles

        m_passIndex[fileName] = shaderSlot++;              // map file ➜ slot
    }

    // 2)  ALL *FIXED* LIBRARIES AND ROOT‑SIGS (unchanged from original)
    // ----------------------------------------------------------------
    m_missLibrary   = nv_helpers_dx12::CompileShaderLibrary(L"Miss_v6.hlsl");
    m_hitLibrary    = nv_helpers_dx12::CompileShaderLibrary(L"Hit_v7.hlsl");
    m_shadowLibrary = nv_helpers_dx12::CompileShaderLibrary(L"ShadowRay.hlsl");

    pipeline.AddLibrary(m_missLibrary.Get(),    { L"Miss" });
    pipeline.AddLibrary(m_shadowLibrary.Get(),  { L"ShadowClosestHit", L"ShadowMiss" });
    pipeline.AddLibrary(m_hitLibrary.Get(),     { L"ClosestHit" });

  // To be used, each DX12 shader needs a root signature defining which
  // parameters and buffers will be accessed.
  m_rayGenSignature = CreateRayGenSignature();
  m_missSignature = CreateMissSignature();
  m_hitSignature = CreateHitSignature();

  // 3 different shaders can be invoked to obtain an intersection: an
  // intersection shader is called
  // when hitting the bounding box of non-triangular geometry. This is beyond
  // the scope of this tutorial. An any-hit shader is called on potential
  // intersections. This shader can, for example, perform alpha-testing and
  // discard some intersections. Finally, the closest-hit program is invoked on
  // the intersection point closest to the ray origin. Those 3 shaders are bound
  // together into a hit group.

  // Note that for triangular geometry the intersection shader is built-in. An
  // empty any-hit shader is also defined by default, so in our simple case each
  // hit group contains only the closest hit shader. Note that since the
  // exported symbols are defined above the shaders can be simply referred to by
  // name.

  // Hit group for the triangles, with a shader simply interpolating vertex
  // colors
  pipeline.AddHitGroup(L"HitGroup", L"ClosestHit");
  // #DXR Extra - Another ray type
  // Hit group for all geometry when hit by a shadow ray
  pipeline.AddHitGroup(L"ShadowHitGroup", L"ShadowClosestHit");

  // The following section associates the root signature to each shader. Note
  // that we can explicitly show that some shaders share the same root signature
  // (eg. Miss and ShadowMiss). Note that the hit shaders are now only referred
  // to as hit groups, meaning that the underlying intersection, any-hit and
  // closest-hit shaders share the same root signature.
    // After you’ve done:
    //   for each entry in m_passSequence (skipping "barrier")
    //     CompileShaderLibrary(fileName) and pipeline.AddLibrary(lib, {baseName})
    // Now wire up the root‑signature for each of those bases:

    for (const auto& entry : m_passSequence)
    {
        if (entry == L"barrier") continue;

        // strip off the “.hlsl” to get exactly the name you exported
        std::wstring base = entry.substr(0, entry.rfind(L'.'));

        pipeline.AddRootSignatureAssociation(
            m_rayGenSignature.Get(),
            { base.c_str() }
        );
    }


    pipeline.AddRootSignatureAssociation(m_missSignature.Get(), {L"Miss"});
  pipeline.AddRootSignatureAssociation(m_hitSignature.Get(), {L"HitGroup"});

  // #DXR Extra - Another ray type
  pipeline.AddRootSignatureAssociation(m_shadowSignature.Get(),
                                       {L"ShadowHitGroup"});
  // #DXR Extra - Another ray type
  pipeline.AddRootSignatureAssociation(m_missSignature.Get(),
                                       {L"Miss", L"ShadowMiss"});

  // #DXR Extra: Per-Instance Data
  pipeline.AddRootSignatureAssociation(m_hitSignature.Get(),
                                       {L"HitGroup"});
  // The payload size defines the maximum size of the data carried by the rays,
  // ie. the the data
  // exchanged between shaders, such as the HitInfo structure in the HLSL code.
  // It is important to keep this value as low as possible as a too high value
  // would result in unnecessary memory consumption and cache trashing.
    pipeline.SetMaxPayloadSize(7 * sizeof(float) + 2 * sizeof(UINT) + sizeof(BOOL));

  // Upon hitting a surface, DXR can provide several attributes to the hit. In
  // our sample we just use the barycentric coordinates defined by the weights
  // u,v of the last two vertices of the triangle. The actual barycentrics can
  // be obtained using float3 barycentrics = float3(1.f-u-v, u, v);
  pipeline.SetMaxAttributeSize(2 * sizeof(float)); // barycentric coordinates

  // The raytracing process can shoot rays from existing hit points, resulting
  // in nested TraceRay calls. Our sample code traces only primary rays, which
  // then requires a trace depth of 1. Note that this recursion depth should be
  // kept to a minimum for best performance. Path tracing algorithms can be
  // easily flattened into a simple loop in the ray generation.
  pipeline.SetMaxRecursionDepth(1);

  // Compile the pipeline for execution on the GPU
  m_rtStateObject = pipeline.Generate();

  // Cast the state object into a properties object, allowing to later access
  // the shader pointers by name
  ThrowIfFailed(
      m_rtStateObject->QueryInterface(IID_PPV_ARGS(&m_rtStateObjectProps)));
}

//-----------------------------------------------------------------------------
//
// Allocate the buffer holding the raytracing output, with the same size as the
// output image
//
void Renderer::CreateRaytracingOutputBuffer() {
  D3D12_RESOURCE_DESC resDesc = {};
  //10 backtracking size, times the buffer number
  resDesc.DepthOrArraySize = 30 * 2;
  resDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
  // The backbuffer is actually DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, but sRGB
  // formats cannot be used with UAVs. For accuracy we should convert to sRGB
  // ourselves in the shader
  resDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;

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
    textureDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT; // Choose an appropriate format
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

}

//-----------------------------------------------------------------------------
//
// Create the main heap used by the shaders, which will give access to the
// raytracing output and the top-level acceleration structure
//
void Renderer::CreateShaderResourceHeap() {
  // #DXR Extra: Perspective Camera
  // Create a SRV/UAV/CBV descriptor heap. We need 3 entries - 1 SRV for the
  // TLAS, 1 UAV for the raytracing output and 1 CBV for the camera matrices
// Create a SRV/UAV/CBV descriptor heap. We need 4 entries - 1 SRV for the TLAS, 1 UAV for the
// raytracing output, 1 CBV for the camera matrices, 1 SRV for the
// per-instance data (# DXR Extra - Simple Lighting)
    m_srvUavHeap = nv_helpers_dx12::CreateDescriptorHeap(
            m_device.Get(), 22, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, true);

  // Get a handle to the heap memory on the CPU side, to be able to write the
  // descriptors directly
  D3D12_CPU_DESCRIPTOR_HANDLE srvHandle =
      m_srvUavHeap->GetCPUDescriptorHandleForHeapStart();

  // Create the UAV. Based on the root signature we created it is the first
  // entry. The Create*View methods write the view information directly into
  // srvHandle
  D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
    uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
    uavDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM; // Ensure this matches your resource format
    uavDesc.Texture2DArray.MipSlice = 0; // Assuming you're using the first MIP level
    uavDesc.Texture2DArray.FirstArraySlice = 0; // Starting at the first layer of the array
    uavDesc.Texture2DArray.ArraySize = 30; // The number of layers in the array
  m_device->CreateUnorderedAccessView(m_outputResource.Get(), nullptr, &uavDesc,
                                      srvHandle);

  // Add the Top Level AS SRV right after the raytracing output buffer
  srvHandle.ptr += m_device->GetDescriptorHandleIncrementSize(
      D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

  D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc;
  srvDesc.Format = DXGI_FORMAT_UNKNOWN;
  srvDesc.ViewDimension = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE;
  srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
  srvDesc.RaytracingAccelerationStructure.Location =
      m_topLevelASBuffers.pResult->GetGPUVirtualAddress();
  // Write the acceleration structure view in the heap
  m_device->CreateShaderResourceView(nullptr, &srvDesc, srvHandle);

  // #DXR Extra: Perspective Camera
  // Add the constant buffer for the camera after the TLAS
  srvHandle.ptr += m_device->GetDescriptorHandleIncrementSize(
      D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

// Describe and create a constant buffer view for the camera
    D3D12_CONSTANT_BUFFER_VIEW_DESC cbvDesc = {};
    cbvDesc.BufferLocation = m_cameraBuffer->GetGPUVirtualAddress();
    cbvDesc.SizeInBytes = m_cameraBufferSize;

// Debug output: Check the buffer location and size
    std::wcout << L"Camera buffer GPU virtual address: " << cbvDesc.BufferLocation << std::endl;
    std::wcout << L"Camera buffer size (in bytes): " << cbvDesc.SizeInBytes << std::endl;

    m_device->CreateConstantBufferView(&cbvDesc, srvHandle);

// Debug output: Confirm that the CreateConstantBufferView call was made
    std::wcout << L"Constant buffer view created for the camera." << std::endl;


    //# DXR Extra - Simple Lighting
    srvHandle.ptr +=
            m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc1;
    srvDesc1.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srvDesc1.Format = DXGI_FORMAT_UNKNOWN;
    srvDesc1.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    srvDesc1.Buffer.FirstElement = 0;
    srvDesc1.Buffer.NumElements = static_cast<UINT>(m_instances.size());
    srvDesc1.Buffer.StructureByteStride = sizeof(InstanceProperties);
    srvDesc1.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
// Write the per-instance properties buffer view in the heap
    m_device->CreateShaderResourceView(m_instanceProperties.Get(), &srvDesc1, srvHandle);

    // Move to the next descriptor slot
    srvHandle.ptr += m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    // Create SRV for the Material IDs buffer
    D3D12_SHADER_RESOURCE_VIEW_DESC materialIdSrvDesc = {};
    materialIdSrvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    materialIdSrvDesc.Format = DXGI_FORMAT_R32_UINT; // Assuming material IDs are 32-bit unsigned integers
    materialIdSrvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    materialIdSrvDesc.Buffer.FirstElement = 0;
    materialIdSrvDesc.Buffer.NumElements = static_cast<UINT>(m_materialIDs.size()); // Assuming materialIDs is a std::vector<UINT>
    materialIdSrvDesc.Buffer.StructureByteStride = 0; // Not a structured buffer
    materialIdSrvDesc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
    m_device->CreateShaderResourceView(m_materialIndexBuffer.Get(), &materialIdSrvDesc, srvHandle);

    // Move to the next descriptor slot
    srvHandle.ptr += m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    // Create SRV for the Materials buffer
    D3D12_SHADER_RESOURCE_VIEW_DESC materialsSrvDesc = {};
    materialsSrvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    materialsSrvDesc.Format = DXGI_FORMAT_UNKNOWN; // Use DXGI_FORMAT_UNKNOWN for structured buffers
    materialsSrvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    materialsSrvDesc.Buffer.FirstElement = 0;
    materialsSrvDesc.Buffer.NumElements = static_cast<UINT>(m_materials.size()); // Assuming materials is a std::vector<Material>
    materialsSrvDesc.Buffer.StructureByteStride = sizeof(Material); // Assuming Material is your material struct
    materialsSrvDesc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
    m_device->CreateShaderResourceView(m_materialBuffer.Get(), &materialsSrvDesc, srvHandle);

// After existing descriptors
    srvHandle.ptr += m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

// Create SRV for the Emissive Triangles buffer
    D3D12_SHADER_RESOURCE_VIEW_DESC emissiveTrianglesSrvDesc = {};
    emissiveTrianglesSrvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    emissiveTrianglesSrvDesc.Format = DXGI_FORMAT_UNKNOWN; // Structured buffer
    emissiveTrianglesSrvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    emissiveTrianglesSrvDesc.Buffer.FirstElement = 0;
    emissiveTrianglesSrvDesc.Buffer.NumElements = static_cast<UINT>(m_emissiveTriangles.size());
    emissiveTrianglesSrvDesc.Buffer.StructureByteStride = sizeof(LightTriangle);
    emissiveTrianglesSrvDesc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
    m_device->CreateShaderResourceView(m_emissiveTrianglesBuffer.Get(), &emissiveTrianglesSrvDesc, srvHandle);

    // Move to the next descriptor slot after the last SRV
    srvHandle.ptr += m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    // Create UAV for the permanent data texture
    D3D12_UNORDERED_ACCESS_VIEW_DESC permanentDataUavDesc = {};
    permanentDataUavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    permanentDataUavDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT; // Ensure this matches your resource format
    permanentDataUavDesc.Texture2D.MipSlice = 0;
    permanentDataUavDesc.Texture2D.PlaneSlice = 0;

    // Create the UAV in the heap at the current descriptor slot (heap slot 7)
    m_device->CreateUnorderedAccessView(m_permanentDataTexture.Get(), nullptr, &permanentDataUavDesc, srvHandle);


    //_________________________________
    srvHandle.ptr += m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    // Assuming you know the number of elements and structure size
    UINT width = GetWidth();
    UINT height = GetHeight();
    UINT reservoirCount = width * height;
    UINT reservoirElementSize_di = sizeof(Reservoir_DI);
    UINT reservoirElementSize_gi = sizeof(Reservoir_GI);
    UINT reservoirElementSize_sample = sizeof(SampleData);
    UINT reservoirBufferSize_di = reservoirCount * reservoirElementSize_di;
    UINT reservoirBufferSize_gi = reservoirCount * reservoirElementSize_gi;
    UINT reservoirBufferSize_sample = reservoirCount * reservoirElementSize_sample;

// Create default-heap buffer with UAV for random read/write
    D3D12_RESOURCE_DESC reservoirDesc = {};
    reservoirDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    reservoirDesc.Alignment = 0;
    reservoirDesc.Width = reservoirBufferSize_di;
    reservoirDesc.Height = 1;
    reservoirDesc.DepthOrArraySize = 1;
    reservoirDesc.MipLevels = 1;
    reservoirDesc.Format = DXGI_FORMAT_UNKNOWN;
    reservoirDesc.SampleDesc.Count = 1;
    reservoirDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    reservoirDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    ThrowIfFailed(m_device->CreateCommittedResource(
            &nv_helpers_dx12::kDefaultHeapProps,
            D3D12_HEAP_FLAG_NONE,
            &reservoirDesc,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            nullptr,
            IID_PPV_ARGS(&m_reservoirBuffer)
    ));

    D3D12_UNORDERED_ACCESS_VIEW_DESC reservoirUavDesc = {};
    reservoirUavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    reservoirUavDesc.Format = DXGI_FORMAT_UNKNOWN; // For structured buffers
    reservoirUavDesc.Buffer.FirstElement = 0;
    reservoirUavDesc.Buffer.NumElements = reservoirCount;
    reservoirUavDesc.Buffer.StructureByteStride = reservoirElementSize_di;
    reservoirUavDesc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;

    m_device->CreateUnorderedAccessView(
            m_reservoirBuffer.Get(),
            nullptr,
            &reservoirUavDesc,
            srvHandle
    );
    //_________________________________

    //_________________________________
    srvHandle.ptr += m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

// Create default-heap buffer with UAV for random read/write
    D3D12_RESOURCE_DESC reservoirDesc_2 = {};
    reservoirDesc_2.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    reservoirDesc_2.Alignment = 0;
    reservoirDesc_2.Width = reservoirBufferSize_di;
    reservoirDesc_2.Height = 1;
    reservoirDesc_2.DepthOrArraySize = 1;
    reservoirDesc_2.MipLevels = 1;
    reservoirDesc_2.Format = DXGI_FORMAT_UNKNOWN;
    reservoirDesc_2.SampleDesc.Count = 1;
    reservoirDesc_2.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    reservoirDesc_2.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    ThrowIfFailed(m_device->CreateCommittedResource(
            &nv_helpers_dx12::kDefaultHeapProps,
            D3D12_HEAP_FLAG_NONE,
            &reservoirDesc_2,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            nullptr,
            IID_PPV_ARGS(&m_reservoirBuffer_2)
    ));

    D3D12_UNORDERED_ACCESS_VIEW_DESC reservoirUavDesc_2 = {};
    reservoirUavDesc_2.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    reservoirUavDesc_2.Format = DXGI_FORMAT_UNKNOWN; // For structured buffers
    reservoirUavDesc_2.Buffer.FirstElement = 0;
    reservoirUavDesc_2.Buffer.NumElements = reservoirCount;
    reservoirUavDesc_2.Buffer.StructureByteStride = reservoirElementSize_di;
    reservoirUavDesc_2.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;

    m_device->CreateUnorderedAccessView(
            m_reservoirBuffer_2.Get(),
            nullptr,
            &reservoirUavDesc_2,
            srvHandle
    );
    //_________________________________

    //_________________________________
    srvHandle.ptr += m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    // Create default-heap buffer with UAV for random read/write
    D3D12_RESOURCE_DESC reservoirDesc_3 = {};
    reservoirDesc_3.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    reservoirDesc_3.Alignment = 0;
    reservoirDesc_3.Width = reservoirBufferSize_gi;
    reservoirDesc_3.Height = 1;
    reservoirDesc_3.DepthOrArraySize = 1;
    reservoirDesc_3.MipLevels = 1;
    reservoirDesc_3.Format = DXGI_FORMAT_UNKNOWN;
    reservoirDesc_3.SampleDesc.Count = 1;
    reservoirDesc_3.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    reservoirDesc_3.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    ThrowIfFailed(m_device->CreateCommittedResource(
            &nv_helpers_dx12::kDefaultHeapProps,
            D3D12_HEAP_FLAG_NONE,
            &reservoirDesc_3,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            nullptr,
            IID_PPV_ARGS(&m_reservoirBuffer_3)
    ));

    D3D12_UNORDERED_ACCESS_VIEW_DESC reservoirUavDesc_3 = {};
    reservoirUavDesc_2.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    reservoirUavDesc_2.Format = DXGI_FORMAT_UNKNOWN; // For structured buffers
    reservoirUavDesc_2.Buffer.FirstElement = 0;
    reservoirUavDesc_2.Buffer.NumElements = reservoirCount;
    reservoirUavDesc_2.Buffer.StructureByteStride = reservoirElementSize_gi;
    reservoirUavDesc_2.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;

    m_device->CreateUnorderedAccessView(
            m_reservoirBuffer_3.Get(),
            nullptr,
            &reservoirUavDesc_2,
            srvHandle
    );
    //_________________________________

    //_________________________________
    srvHandle.ptr += m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    // Create default-heap buffer with UAV for random read/write
    D3D12_RESOURCE_DESC reservoirDesc_4 = {};
    reservoirDesc_4.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    reservoirDesc_4.Alignment = 0;
    reservoirDesc_4.Width = reservoirBufferSize_gi;
    reservoirDesc_4.Height = 1;
    reservoirDesc_4.DepthOrArraySize = 1;
    reservoirDesc_4.MipLevels = 1;
    reservoirDesc_4.Format = DXGI_FORMAT_UNKNOWN;
    reservoirDesc_4.SampleDesc.Count = 1;
    reservoirDesc_4.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    reservoirDesc_4.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    ThrowIfFailed(m_device->CreateCommittedResource(
            &nv_helpers_dx12::kDefaultHeapProps,
            D3D12_HEAP_FLAG_NONE,
            &reservoirDesc_4,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            nullptr,
            IID_PPV_ARGS(&m_reservoirBuffer_4)
    ));

    D3D12_UNORDERED_ACCESS_VIEW_DESC reservoirUavDesc_4 = {};
    reservoirUavDesc_4.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    reservoirUavDesc_4.Format = DXGI_FORMAT_UNKNOWN; // For structured buffers
    reservoirUavDesc_4.Buffer.FirstElement = 0;
    reservoirUavDesc_4.Buffer.NumElements = reservoirCount;
    reservoirUavDesc_4.Buffer.StructureByteStride = reservoirElementSize_gi;
    reservoirUavDesc_4.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;

    m_device->CreateUnorderedAccessView(
            m_reservoirBuffer_4.Get(),
            nullptr,
            &reservoirUavDesc_4,
            srvHandle
    );
    //_________________________________
    //_________________________________
    srvHandle.ptr += m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    // Create default-heap buffer with UAV for random read/write
    D3D12_RESOURCE_DESC reservoirDesc_5 = {};
    reservoirDesc_5.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    reservoirDesc_5.Alignment = 0;
    reservoirDesc_5.Width = reservoirBufferSize_sample;
    reservoirDesc_5.Height = 1;
    reservoirDesc_5.DepthOrArraySize = 1;
    reservoirDesc_5.MipLevels = 1;
    reservoirDesc_5.Format = DXGI_FORMAT_UNKNOWN;
    reservoirDesc_5.SampleDesc.Count = 1;
    reservoirDesc_5.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    reservoirDesc_5.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    ThrowIfFailed(m_device->CreateCommittedResource(
            &nv_helpers_dx12::kDefaultHeapProps,
            D3D12_HEAP_FLAG_NONE,
            &reservoirDesc_5,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            nullptr,
            IID_PPV_ARGS(&m_sampleBuffer_current)
    ));

    D3D12_UNORDERED_ACCESS_VIEW_DESC reservoirUavDesc_5 = {};
    reservoirUavDesc_5.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    reservoirUavDesc_5.Format = DXGI_FORMAT_UNKNOWN; // For structured buffers
    reservoirUavDesc_5.Buffer.FirstElement = 0;
    reservoirUavDesc_5.Buffer.NumElements = reservoirCount;
    reservoirUavDesc_5.Buffer.StructureByteStride = reservoirElementSize_sample;
    reservoirUavDesc_5.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;

    m_device->CreateUnorderedAccessView(
            m_sampleBuffer_current.Get(),
            nullptr,
            &reservoirUavDesc_5,
            srvHandle
    );
    //_________________________________

    //_________________________________
    srvHandle.ptr += m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    // Create default-heap buffer with UAV for random read/write
    D3D12_RESOURCE_DESC reservoirDesc_6 = {};
    reservoirDesc_6.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    reservoirDesc_6.Alignment = 0;
    reservoirDesc_6.Width = reservoirBufferSize_sample;
    reservoirDesc_6.Height = 1;
    reservoirDesc_6.DepthOrArraySize = 1;
    reservoirDesc_6.MipLevels = 1;
    reservoirDesc_6.Format = DXGI_FORMAT_UNKNOWN;
    reservoirDesc_6.SampleDesc.Count = 1;
    reservoirDesc_6.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    reservoirDesc_6.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    ThrowIfFailed(m_device->CreateCommittedResource(
            &nv_helpers_dx12::kDefaultHeapProps,
            D3D12_HEAP_FLAG_NONE,
            &reservoirDesc_6,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            nullptr,
            IID_PPV_ARGS(&m_sampleBuffer_last)
    ));

    D3D12_UNORDERED_ACCESS_VIEW_DESC reservoirUavDesc_6 = {};
    reservoirUavDesc_6.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    reservoirUavDesc_6.Format = DXGI_FORMAT_UNKNOWN; // For structured buffers
    reservoirUavDesc_6.Buffer.FirstElement = 0;
    reservoirUavDesc_6.Buffer.NumElements = reservoirCount;
    reservoirUavDesc_6.Buffer.StructureByteStride = reservoirElementSize_sample;
    reservoirUavDesc_6.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;

    m_device->CreateUnorderedAccessView(
            m_sampleBuffer_last.Get(),
            nullptr,
            &reservoirUavDesc_6,
            srvHandle
    );
    //_________________________________

    // ── alias PROB array  (R32_FLOAT) ────────────────────────────────────────────
    srvHandle.ptr += m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);   // ← reuse the increment variable you already use
    D3D12_SHADER_RESOURCE_VIEW_DESC probDesc = {};
    probDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    probDesc.Format            = DXGI_FORMAT_R32_FLOAT;
    probDesc.ViewDimension     = D3D12_SRV_DIMENSION_BUFFER;
    probDesc.Buffer.FirstElement = 0;
    probDesc.Buffer.NumElements  =
            static_cast<UINT>(m_aliasProb.size());
    probDesc.Buffer.StructureByteStride = 0;
    m_device->CreateShaderResourceView(
            m_aliasProbBuffer.Get(), &probDesc, srvHandle);

    // ── alias IDX array (R32_UINT) ───────────────────────────────────────────────
    srvHandle.ptr += m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    D3D12_SHADER_RESOURCE_VIEW_DESC idxDesc = {};
    idxDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    idxDesc.Format            = DXGI_FORMAT_R32_UINT;
    idxDesc.ViewDimension     = DXGI_FORMAT_UNKNOWN ?
                                 D3D12_SRV_DIMENSION_BUFFER :
                                 D3D12_SRV_DIMENSION_BUFFER;
    idxDesc.Buffer.FirstElement = 0;
    idxDesc.Buffer.NumElements  =
            static_cast<UINT>(m_aliasIdx.size());
    idxDesc.Buffer.StructureByteStride = 0;
    m_device->CreateShaderResourceView(
            m_aliasIdxBuffer.Get(), &idxDesc, srvHandle);

    std::wcout << L"SRVs created!" << std::endl;
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

    //  RAY‑GEN SECTION  -------------------------------------------------------
    for (const auto& entry : m_passSequence) {

        if (entry == L"barrier")
            continue;

        std::wstring base = entry.substr(entry.find_last_of(L"/\\") + 1);
        base = base.substr(0, base.rfind(L'.'));          // strip extension
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
        m_sbtHelper.AddHitGroup(
                L"HitGroup",
                {(void *) (m_VB[i]->GetGPUVirtualAddress()),
                 (void *) (m_IB[i]->GetGPUVirtualAddress()),
                 (void *) (m_perInstanceConstantBuffers[0]->GetGPUVirtualAddress()), heapPointer});
        m_sbtHelper.AddHitGroup(L"ShadowHitGroup", {});
    }

    // Adding final ShadowHitGroup
    std::wcout << L"Adding ShadowHitGroup..." << std::endl;
    m_sbtHelper.AddHitGroup(L"ShadowHitGroup", {});

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
                        + sizeof(float);     // for time
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

    // Current projection matrix
    float fovAngleY = 60.0f * XM_PI / 180.0f;
    matrices[1] = XMMatrixPerspectiveFovRH(fovAngleY, m_aspectRatio, 0.1f, 1000.0f);

    // Inverse matrices
    XMVECTOR det;
    matrices[2] = XMMatrixInverse(&det, matrices[0]);  // viewInv
    matrices[3] = XMMatrixInverse(&det, matrices[1]);  // projectionInv

    // Previous frame matrices
    matrices[4] = m_prevViewMatrix;
    matrices[5] = m_prevProjMatrix;

    // Copy matrix contents to the buffer
    uint8_t *pData;
    HRESULT hr = m_cameraBuffer->Map(0, nullptr, (void **)&pData);
    if (FAILED(hr)) {
        std::wcerr << L"Failed to map camera buffer!" << std::endl;
        return;
    }

    // Copy the 6 matrices
    memcpy(pData, matrices.data(), 6 * sizeof(XMMATRIX));

    // Add the current system time as float
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    auto nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
    //float currentTime = static_cast<float>(nanos % 1000);  // Convert milliseconds to seconds as float
    float currentTime = static_cast<uint32_t>(nanos);

    memcpy(pData + (6 * sizeof(XMMATRIX)), &currentTime, sizeof(float));


    m_cameraBuffer->Unmap(0, nullptr);

    // Save the current matrices for use in the next frame
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

// #DXR Extra: Indexed Geometry
void Renderer::CreateVB(std::string name) {
  std::vector<Vertex> vertices;
  std::vector<UINT> indices;
  std::vector<Material> materials;
  std::vector<UINT> materialIDs;

  ComPtr<ID3D12Resource> l_VB;
  ComPtr<ID3D12Resource> l_IB;
  D3D12_VERTEX_BUFFER_VIEW l_VBView;
  D3D12_INDEX_BUFFER_VIEW l_IBView;
  ComPtr<ID3D12Resource> l_material;
  ComPtr<ID3D12Resource> l_materialID;
  UINT l_IndexCount;
  UINT l_VertexCount;

  //nv_helpers_dx12::GenerateMengerSponge(3, 0.75, vertices, indices);
  ObjLoader::loadObjFile(name,&vertices, &indices, &materials, &materialIDs, &materialIDOffset, &materialVertexOffset);
    // Before inserting new material IDs, store the current offset
    m_materialIDOffsets.push_back(static_cast<UINT>(m_materialIDs.size()));

    // Insert the material IDs and materials
    m_materialIDs.insert(m_materialIDs.end(), materialIDs.begin(), materialIDs.end());
    m_materials.insert(m_materials.end(), materials.begin(), materials.end());
  materialVertexOffset=m_materialIDs.size();
  std::wcout << L"Triangle Offset: " << materialVertexOffset << std::endl;
  {
    const UINT mengerVBSize =
        static_cast<UINT>(vertices.size()) * sizeof(Vertex);

    // Note: using upload heaps to transfer static data like vert buffers is not
    // recommended. Every time the GPU needs it, the upload heap will be
    // marshalled over. Please read up on Default Heap usage. An upload heap is
    // used here for code simplicity and because there are very few verts to
    // actually transfer.
    CD3DX12_HEAP_PROPERTIES heapProperty =
        CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
    CD3DX12_RESOURCE_DESC bufferResource =
        CD3DX12_RESOURCE_DESC::Buffer(mengerVBSize);
    ThrowIfFailed(m_device->CreateCommittedResource(
        &heapProperty, D3D12_HEAP_FLAG_NONE, &bufferResource, //
        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&l_VB)));

    // Copy the triangle data to the vertex buffer.
    UINT8 *pVertexDataBegin;
    CD3DX12_RANGE readRange(
        0, 0); // We do not intend to read from this resource on the CPU.
    ThrowIfFailed(l_VB->Map(
        0, &readRange, reinterpret_cast<void **>(&pVertexDataBegin)));
    memcpy(pVertexDataBegin, vertices.data(), mengerVBSize);
      l_VB->Unmap(0, nullptr);

    // Initialize the vertex buffer view.
      l_VBView.BufferLocation = l_VB->GetGPUVirtualAddress();
      l_VBView.StrideInBytes = sizeof(Vertex);
      l_VBView.SizeInBytes = mengerVBSize;
  }
  {
    const UINT IBSize = static_cast<UINT>(indices.size()) * sizeof(UINT);

    // Note: using upload heaps to transfer static data like vert buffers is not
    // recommended. Every time the GPU needs it, the upload heap will be
    // marshalled over. Please read up on Default Heap usage. An upload heap is
    // used here for code simplicity and because there are very few verts to
    // actually transfer.
    CD3DX12_HEAP_PROPERTIES heapProperty =
        CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
    CD3DX12_RESOURCE_DESC bufferResource =
        CD3DX12_RESOURCE_DESC::Buffer(IBSize);
    ThrowIfFailed(m_device->CreateCommittedResource(
        &heapProperty, D3D12_HEAP_FLAG_NONE, &bufferResource, //
        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&l_IB)));

    // Copy the triangle data to the index buffer.
    UINT8 *pIndexDataBegin;
    CD3DX12_RANGE readRange(
        0, 0); // We do not intend to read from this resource on the CPU.
    ThrowIfFailed(l_IB->Map(0, &readRange,
                                  reinterpret_cast<void **>(&pIndexDataBegin)));
    memcpy(pIndexDataBegin, indices.data(), IBSize);
      l_IB->Unmap(0, nullptr);

    // Initialize the index buffer view.
      l_IBView.BufferLocation = l_IB->GetGPUVirtualAddress();
      l_IBView.Format = DXGI_FORMAT_R32_UINT;
      l_IBView.SizeInBytes = IBSize;

      l_IndexCount = static_cast<UINT>(indices.size());
      l_VertexCount = static_cast<UINT>(vertices.size());
  }

    //Fill the vectors with data
    m_VB.push_back(l_VB);
    m_VBView.push_back(l_VBView);
    m_IB.push_back(l_IB);
    m_IBView.push_back(l_IBView);
    m_VertexCount.push_back(l_VertexCount);
    m_IndexCount.push_back(l_IndexCount);
    m_material.push_back(l_material);
    m_materialID.push_back(l_materialID);
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

//--------------------------------------------------------------------------------------------------
// Copy the per-instance data into the buffer
// #DXR Extra - Refitting
void Renderer::UpdateInstancePropertiesBuffer() {
  InstanceProperties *current = nullptr;
  CD3DX12_RANGE readRange(
      0, 0); // We do not intend to read from this resource on the CPU.
  ThrowIfFailed(m_instanceProperties->Map(0, &readRange,
                                          reinterpret_cast<void **>(&current)));
    for (const auto &inst : m_instances)
    {
        XMVECTOR det_filler;
        current->prevObjectToWorld = current->objectToWorld;
        current->prevObjectToWorldInverse = XMMatrixInverse(&det_filler,current->objectToWorld);
        current->objectToWorld = inst.second;
        current->objectToWorldInverse = XMMatrixInverse(&det_filler,inst.second);

        //# DXR Extra - Simple Lighting
        XMMATRIX upper3x3 = inst.second;
        // Remove the translation and lower vector of the matrix
        upper3x3.r[0].m128_f32[3] = 0.f;
        upper3x3.r[1].m128_f32[3] = 0.f;
        upper3x3.r[2].m128_f32[3] = 0.f;
        upper3x3.r[3].m128_f32[0] = 0.f;
        upper3x3.r[3].m128_f32[1] = 0.f;
        upper3x3.r[3].m128_f32[2] = 0.f;
        upper3x3.r[3].m128_f32[3] = 1.f;
        XMVECTOR det;
        current->prevObjectToWorldNormal = current->objectToWorldNormal;
        current->objectToWorldNormal = XMMatrixTranspose(XMMatrixInverse(&det, upper3x3));
        current++;
    }
  m_instanceProperties->Unmap(0, nullptr);
}

void Renderer::CollectEmissiveTriangles() {
    m_emissiveTriangles.clear();

    for (size_t instanceIndex = 0; instanceIndex < m_instances.size(); ++instanceIndex) {
        UINT modelIndex = m_instanceModelIndices[instanceIndex];

        UINT e_materialIDOffset = m_materialIDOffsets[modelIndex];
        UINT triangleCount = m_IndexCount[modelIndex] / 3;

        // Map the vertex and index buffers for the model
        Vertex* vertices = nullptr;
        UINT* indices = nullptr;
        CD3DX12_RANGE readRange(0, 0);

        // Map the vertex buffer
        ThrowIfFailed(m_VB[modelIndex]->Map(0, &readRange, reinterpret_cast<void**>(&vertices)));

        // Map the index buffer
        ThrowIfFailed(m_IB[modelIndex]->Map(0, &readRange, reinterpret_cast<void**>(&indices)));

        for (UINT t = 0; t < triangleCount; ++t) {
            UINT idx0 = indices[t * 3 + 0];
            UINT idx1 = indices[t * 3 + 1];
            UINT idx2 = indices[t * 3 + 2];

            // Get the material IDs for the triangle's vertices
            UINT materialID0 = m_materialIDs[e_materialIDOffset + t * 3 + 0];
            UINT materialID1 = m_materialIDs[e_materialIDOffset + t * 3 + 1];
            UINT materialID2 = m_materialIDs[e_materialIDOffset + t * 3 + 2];

            // Ensure all vertices of the triangle have the same material ID
            if (materialID0 != materialID1 || materialID0 != materialID2) {
                std::wcout << "Warning: Triangle vertices have different material IDs!" << std::endl;
                continue; // Skip this triangle or handle as needed
            }

            UINT materialID = materialID0; // Use the consistent material ID
            const Material& material = m_materials[materialID];

            // Check if the material is emissive
            if (material.Ke.x + material.Ke.y + material.Ke.z > 0.0f) {
                // Get the positions of the vertices
                const Vertex& v0 = vertices[idx0];
                const Vertex& v1 = vertices[idx1];
                const Vertex& v2 = vertices[idx2];

                LightTriangle lt{};
                lt.x = v0.position;
                lt.y = v1.position;
                lt.z = v2.position;
                lt.instanceID = static_cast<UINT>(instanceIndex);
                lt.weight = ComputeTriangleWeight(v0.position, v1.position, v2.position, material.Ke);
                lt.emission = material.Ke;

                m_emissiveTriangles.push_back(lt);
            }
        }

        // Unmap the vertex and index buffers
        m_VB[modelIndex]->Unmap(0, nullptr);
        m_IB[modelIndex]->Unmap(0, nullptr);
    }

    // Sort the emissive triangles based on weight in descending order
    std::sort(m_emissiveTriangles.begin(), m_emissiveTriangles.end(),
              [](const LightTriangle& a, const LightTriangle& b) {
                  return a.weight > b.weight;
              });

    // Calculate the total weight
    float totalWeight = 0.0f;
    for (const auto& triangle : m_emissiveTriangles) {
        totalWeight += triangle.weight;
    }

    // Calculate relative weights and cumulative distribution function (CDF)
    float cumulativeWeight = 0.0f;
    for (auto& triangle : m_emissiveTriangles) {
        triangle.weight /= totalWeight; // Normalize weight
        cumulativeWeight += triangle.weight;
        triangle.cdf = cumulativeWeight;
        triangle.totalWeight = totalWeight;
    }

    // Ensure the last CDF value is exactly 1.0f
    if (!m_emissiveTriangles.empty()) {
        m_emissiveTriangles.back().cdf = 1.0f;
    }

    std::wcout << L"Emissive Triangles: " << m_emissiveTriangles.size() << std::endl;
}



float Renderer::ComputeTriangleWeight(const XMFLOAT3& v0, const XMFLOAT3& v1, const XMFLOAT3& v2, const XMFLOAT3& emissiveColor) {
    // Compute the area of the triangle
    XMVECTOR p0 = XMLoadFloat3(&v0);
    XMVECTOR p1 = XMLoadFloat3(&v1);
    XMVECTOR p2 = XMLoadFloat3(&v2);

    XMVECTOR edge1 = XMVectorSubtract(p1, p0);
    XMVECTOR edge2 = XMVectorSubtract(p2, p0);
    XMVECTOR crossProduct = XMVector3Cross(edge1, edge2);
    float area = 0.5f * XMVectorGetX(XMVector3Length(crossProduct));

    // Compute the average emissive intensity
    float emissiveIntensity = (emissiveColor.x + emissiveColor.y + emissiveColor.z) / 3.0f;

    // The weight is proportional to area and emissive intensity
    return area * emissiveIntensity;
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

    // Execute and flush the command list
    ThrowIfFailed(m_commandList->Close());
    ID3D12CommandList* ppCommandLists[] = { m_commandList.Get() };
    m_commandQueue->ExecuteCommandLists(1, ppCommandLists);
    WaitForPreviousFrame();
    ThrowIfFailed(m_commandList->Reset(m_commandAllocator.Get(), nullptr));
}

//──────────────────────────────────────────────────────────────────────────────
// Small helper that builds two SoA arrays (prob + alias index)
void Renderer::BuildAliasTableSoA(const std::vector<LightTriangle>& tris)
{
    const uint32_t N = static_cast<uint32_t>(tris.size());
    std::vector<float>      scaled(N);
    std::vector<uint32_t>   _small, _large;

    m_aliasProb.resize(N);
    m_aliasIdx .resize(N);

    // 1) scale weights so that Σ w = N
    for (uint32_t i=0;i<N;++i) scaled[i] = tris[i].weight * N;
    for (uint32_t i=0;i<N;++i) (scaled[i] < 1.f ? _small:_large).push_back(i);

    // 2) alias algorithm
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

//──────────────────────────────────────────────────────────────────────────────
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