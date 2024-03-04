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

#include "Renderer.h"

#include "DXRHelper.h"
#include "nv_helpers_dx12/BottomLevelASGenerator.h"

#include "nv_helpers_dx12/RaytracingPipelineGenerator.h"
#include "nv_helpers_dx12/RootSignatureGenerator.h"

#include "Windowsx.h"
#include "glm/gtc/type_ptr.hpp"
#include "manipulator.h"
#include "../src/Util/ObjLoader.h"


Renderer::Renderer(UINT width, UINT height,
                   std::wstring name)
    : DXSample(width, height, name), m_frameIndex(0),
      m_viewport(0.0f, 0.0f, static_cast<float>(width),
                 static_cast<float>(height)),
      m_scissorRect(0, 0, static_cast<LONG>(width), static_cast<LONG>(height)),
      m_rtvDescriptorSize(0) {}

void Renderer::OnInit() {

  nv_helpers_dx12::CameraManip.setWindowSize(GetWidth(), GetHeight());
  nv_helpers_dx12::CameraManip.setLookat(
      glm::vec3(1.5f, 1.5f, 1.5f), glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));

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
}

// Load the rendering pipeline dependencies.
void Renderer::LoadPipeline() {
  UINT dxgiFactoryFlags = 0;

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

  ComPtr<IDXGIFactory4> factory;
  ThrowIfFailed(CreateDXGIFactory2(dxgiFactoryFlags, IID_PPV_ARGS(&factory)));

  if (m_useWarpDevice) {
    ComPtr<IDXGIAdapter> warpAdapter;
    ThrowIfFailed(factory->EnumWarpAdapter(IID_PPV_ARGS(&warpAdapter)));

    ThrowIfFailed(D3D12CreateDevice(warpAdapter.Get(), D3D_FEATURE_LEVEL_11_0,
                                    IID_PPV_ARGS(&m_device)));
  } else {
    ComPtr<IDXGIAdapter1> hardwareAdapter;
    GetHardwareAdapter(factory.Get(), &hardwareAdapter);

    ThrowIfFailed(D3D12CreateDevice(hardwareAdapter.Get(),
                                    D3D_FEATURE_LEVEL_11_0,
                                    IID_PPV_ARGS(&m_device)));
  }

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
    std::vector<std::string> models = {"garage.obj","A8.obj"};

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
  /*m_instances[0].second =
      XMMatrixRotationAxis({0.f, 1.f, 0.f},
                           static_cast<float>(m_time) / 1000.0f) *
      XMMatrixTranslation(0.f, 0.1f * cosf(m_time / 2000000.f), 0.f);*/
    m_instances[1].second =
            XMMatrixRotationAxis({0.f, 1.f, 0.f},
                                 static_cast<float>(m_time) / 1000.0f) *
            XMMatrixTranslation(0.f, 0.0f * cosf(m_time / 2000000.f), 0.f);
  // #DXR Extra - Refitting
  UpdateInstancePropertiesBuffer();
}

// Render the scene.
void Renderer::OnRender() {
    // Record all the commands we need to render the scene into the command list.
    PopulateCommandList();

    // Execute the command list.
    ID3D12CommandList* ppCommandLists[] = { m_commandList.Get() };
    m_commandQueue->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);

    // Present the frame.
    ThrowIfFailed(m_swapChain->Present(1, 0));

    WaitForPreviousFrame();
}

void Renderer::OnDestroy() {
  // Ensure that the GPU is no longer referencing resources that are about to be
  // cleaned up by the destructor.
  WaitForPreviousFrame();

  CloseHandle(m_fenceEvent);
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

    // Setup the raytracing task
    D3D12_DISPATCH_RAYS_DESC desc = {};
    // The layout of the SBT is as follows: ray generation shader, miss
    // shaders, hit groups. As described in the CreateShaderBindingTable method,
    // all SBT entries of a given type have the same size to allow a fixed
    // stride.

    // The ray generation shaders are always at the beginning of the SBT.
    uint32_t rayGenerationSectionSizeInBytes =
        m_sbtHelper.GetRayGenSectionSize();
    desc.RayGenerationShaderRecord.StartAddress =
        m_sbtStorage->GetGPUVirtualAddress();
    desc.RayGenerationShaderRecord.SizeInBytes =
        rayGenerationSectionSizeInBytes;

    // The miss shaders are in the second SBT section, right after the ray
    // generation shader. We have one miss shader for the camera rays and one
    // for the shadow rays, so this section has a size of 2*m_sbtEntrySize. We
    // also indicate the stride between the two miss shaders, which is the size
    // of a SBT entry
    uint32_t missSectionSizeInBytes = m_sbtHelper.GetMissSectionSize();
    desc.MissShaderTable.StartAddress =
        m_sbtStorage->GetGPUVirtualAddress() + rayGenerationSectionSizeInBytes;
    desc.MissShaderTable.SizeInBytes = missSectionSizeInBytes;
    desc.MissShaderTable.StrideInBytes = m_sbtHelper.GetMissEntrySize();

    // The hit groups section start after the miss shaders. In this sample we
    // have one 1 hit group for the triangle
    uint32_t hitGroupsSectionSize = m_sbtHelper.GetHitGroupSectionSize();
    desc.HitGroupTable.StartAddress = m_sbtStorage->GetGPUVirtualAddress() +
                                      rayGenerationSectionSizeInBytes +
                                      missSectionSizeInBytes;
    desc.HitGroupTable.SizeInBytes = hitGroupsSectionSize;
    desc.HitGroupTable.StrideInBytes = m_sbtHelper.GetHitGroupEntrySize();

    // Dimensions of the image to render, identical to a kernel launch dimension
    desc.Width = GetWidth();
    desc.Height = GetHeight();
    desc.Depth = 1;

    // Bind the raytracing pipeline
    m_commandList->SetPipelineState1(m_rtStateObject.Get());
    // Dispatch the rays and write to the raytracing output
    m_commandList->DispatchRays(&desc);

    // The raytracing output needs to be copied to the actual render target used
    // for display. For this, we need to transition the raytracing output from a
    // UAV to a copy source, and the render target buffer to a copy destination.
    // We can then do the actual copy, before transitioning the render target
    // buffer into a render target, that will be then used to display the image
    transition = CD3DX12_RESOURCE_BARRIER::Transition(
        m_outputResource.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_COPY_SOURCE);
    m_commandList->ResourceBarrier(1, &transition);
    transition = CD3DX12_RESOURCE_BARRIER::Transition(
        m_renderTargets[m_frameIndex].Get(), D3D12_RESOURCE_STATE_RENDER_TARGET,
        D3D12_RESOURCE_STATE_COPY_DEST);
    m_commandList->ResourceBarrier(1, &transition);

    m_commandList->CopyResource(m_renderTargets[m_frameIndex].Get(),
                                m_outputResource.Get());

    transition = CD3DX12_RESOURCE_BARRIER::Transition(
        m_renderTargets[m_frameIndex].Get(), D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_STATE_RENDER_TARGET);
    m_commandList->ResourceBarrier(1, &transition);
  //}

    barrier = CD3DX12_RESOURCE_BARRIER::Transition(
            m_renderTargets[m_frameIndex].Get(),
            D3D12_RESOURCE_STATE_PRESENT,
            D3D12_RESOURCE_STATE_RENDER_TARGET);

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
  // Alternate between rasterization and raytracing using the spacebar
  if (key == VK_SPACE) {
    m_raster = !m_raster;
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

    // Assume m_VB, m_IB, m_VertexCount, and m_IndexCount are all std::vector and have the same size.
    for (size_t i = 0; i < m_VB.size(); ++i) {
        AccelerationStructureBuffers buffers = CreateBottomLevelAS(
                {{m_VB[i].Get(), m_VertexCount[i]}},
                {{m_IB[i].Get(), m_IndexCount[i]}}
        );

        blasBuffers.push_back(buffers);

        // Assuming each instance will use an identity matrix for simplicity,
        // but you can replace XMMatrixIdentity() with any transformation matrix.
        m_instances.push_back({buffers.pResult, XMMatrixIdentity()});
    }
  CreateTopLevelAS(m_instances);

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
      {{0 /*u0*/, 1 /*1 descriptor */, 0 /*use the implicit register space 0*/,
        D3D12_DESCRIPTOR_RANGE_TYPE_UAV /* UAV representing the output buffer*/,
        0 /*heap slot where the UAV is defined*/},
       {0 /*t0*/, 1, 0,
        D3D12_DESCRIPTOR_RANGE_TYPE_SRV /*Top-level acceleration structure*/,
        1},
       {0 /*b0*/, 1, 0, D3D12_DESCRIPTOR_RANGE_TYPE_CBV /*Camera parameters*/,
        2}});

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
             {5 /*t5*/, 1, 0, D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 5 /*6th slot - Materials*/}
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

  // The pipeline contains the DXIL code of all the shaders potentially executed
  // during the raytracing process. This section compiles the HLSL code into a
  // set of DXIL libraries. We chose to separate the code in several libraries
  // by semantic (ray generation, hit, miss) for clarity. Any code layout can be
  // used.
  m_rayGenLibrary = nv_helpers_dx12::CompileShaderLibrary(L"RayGen.hlsl");
  m_missLibrary = nv_helpers_dx12::CompileShaderLibrary(L"Miss.hlsl");
  m_hitLibrary = nv_helpers_dx12::CompileShaderLibrary(L"Hit.hlsl");

  // #DXR Extra - Another ray type
  m_shadowLibrary = nv_helpers_dx12::CompileShaderLibrary(L"ShadowRay.hlsl");
  pipeline.AddLibrary(m_shadowLibrary.Get(),
                      {L"ShadowClosestHit", L"ShadowMiss"});
  m_shadowSignature = CreateHitSignature();

  // In a way similar to DLLs, each library is associated with a number of
  // exported symbols. This
  // has to be done explicitly in the lines below. Note that a single library
  // can contain an arbitrary number of symbols, whose semantic is given in HLSL
  // using the [shader("xxx")] syntax
  pipeline.AddLibrary(m_rayGenLibrary.Get(), {L"RayGen"});
  pipeline.AddLibrary(m_missLibrary.Get(), {L"Miss"});
  // #DXR Extra: Per-Instance Data
  pipeline.AddLibrary(m_hitLibrary.Get(), {L"ClosestHit", L"PlaneClosestHit"});

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
    pipeline.AddHitGroup(L"HitGroup1", L"PlaneClosestHit");
  // #DXR Extra: Per-Instance Data
  pipeline.AddHitGroup(L"PlaneHitGroup", L"PlaneClosestHit");
  // #DXR Extra - Another ray type
  // Hit group for all geometry when hit by a shadow ray
  pipeline.AddHitGroup(L"ShadowHitGroup", L"ShadowClosestHit");

  // The following section associates the root signature to each shader. Note
  // that we can explicitly show that some shaders share the same root signature
  // (eg. Miss and ShadowMiss). Note that the hit shaders are now only referred
  // to as hit groups, meaning that the underlying intersection, any-hit and
  // closest-hit shaders share the same root signature.
  pipeline.AddRootSignatureAssociation(m_rayGenSignature.Get(), {L"RayGen"});
  pipeline.AddRootSignatureAssociation(m_missSignature.Get(), {L"Miss"});
  pipeline.AddRootSignatureAssociation(m_hitSignature.Get(), {L"HitGroup"});
    pipeline.AddRootSignatureAssociation(m_hitSignature.Get(), {L"HitGroup1"});

  // #DXR Extra - Another ray type
  pipeline.AddRootSignatureAssociation(m_shadowSignature.Get(),
                                       {L"ShadowHitGroup"});
  // #DXR Extra - Another ray type
  pipeline.AddRootSignatureAssociation(m_missSignature.Get(),
                                       {L"Miss", L"ShadowMiss"});

  // #DXR Extra: Per-Instance Data
  pipeline.AddRootSignatureAssociation(m_hitSignature.Get(),
                                       {L"HitGroup", L"PlaneHitGroup", L"HitGroup1"});
  // The payload size defines the maximum size of the data carried by the rays,
  // ie. the the data
  // exchanged between shaders, such as the HitInfo structure in the HLSL code.
  // It is important to keep this value as low as possible as a too high value
  // would result in unnecessary memory consumption and cache trashing.
  pipeline.SetMaxPayloadSize(15 * sizeof(float)+ 2*sizeof(UINT)); // RGB + distance

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
  pipeline.SetMaxRecursionDepth(2);

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
  resDesc.DepthOrArraySize = 1;
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
            m_device.Get(), 6, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, true);

  // Get a handle to the heap memory on the CPU side, to be able to write the
  // descriptors directly
  D3D12_CPU_DESCRIPTOR_HANDLE srvHandle =
      m_srvUavHeap->GetCPUDescriptorHandleForHeapStart();

  // Create the UAV. Based on the root signature we created it is the first
  // entry. The Create*View methods write the view information directly into
  // srvHandle
  D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
  uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
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
  m_device->CreateConstantBufferView(&cbvDesc, srvHandle);

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
    // The SBT helper class collects calls to Add*Program.  If called several
    // times, the helper must be emptied before re-adding shaders.
    m_sbtHelper.Reset();

    // The pointer to the beginning of the heap is the only parameter required by
    // shaders without root parameters
    D3D12_GPU_DESCRIPTOR_HANDLE srvUavHeapHandle =
            m_srvUavHeap->GetGPUDescriptorHandleForHeapStart();

    // The helper treats both root parameter pointers and heap pointers as void*,
    // while DX12 uses the
    // D3D12_GPU_DESCRIPTOR_HANDLE to define heap pointers. The pointer in this
    // struct is a UINT64, which then has to be reinterpreted as a pointer.
    auto heapPointer = reinterpret_cast<UINT64 *>(srvUavHeapHandle.ptr);

    // The ray generation only uses heap data
    m_sbtHelper.AddRayGenerationProgram(L"RayGen", {heapPointer});

    // The miss and hit shaders do not access any external resources: instead they
    // communicate their results through the ray payload
    m_sbtHelper.AddMissProgram(L"Miss", {});
    // #DXR Extra - Another ray type
    m_sbtHelper.AddMissProgram(L"ShadowMiss", {});

    // #DXR Extra: Per-Instance Data
    // We have 3 triangles, each of which needs to access its own constant buffer
    // as a root parameter in its primary hit shader. The shadow hit only sets a
    // boolean visibility in the payload, and does not require external data
    for (int i = 0; i < m_instances.size(); ++i){
        m_sbtHelper.AddHitGroup(
                L"HitGroup",
                {(void *) (m_VB[i]->GetGPUVirtualAddress()),
                 (void *) (m_IB[i]->GetGPUVirtualAddress()),(void *)(m_perInstanceConstantBuffers[0]->GetGPUVirtualAddress()), heapPointer});
        m_sbtHelper.AddHitGroup(L"ShadowHitGroup", {});
    }

  // #DXR Extra - Another ray type
  m_sbtHelper.AddHitGroup(L"ShadowHitGroup", {});

  // Compute the size of the SBT given the number of shaders and their
  // parameters
  uint32_t sbtSize = m_sbtHelper.ComputeSBTSize();

  // Create the SBT on the upload heap. This is required as the helper will use
  // mapping to write the SBT contents. After the SBT compilation it could be
  // copied to the default heap for performance.
  m_sbtStorage = nv_helpers_dx12::CreateBuffer(
      m_device.Get(), sbtSize, D3D12_RESOURCE_FLAG_NONE,
      D3D12_RESOURCE_STATE_GENERIC_READ, nv_helpers_dx12::kUploadHeapProps);
  if (!m_sbtStorage) {
    throw std::logic_error("Could not allocate the shader binding table");
  }
  // Compile the SBT from the shader and parameters info
  m_sbtHelper.Generate(m_sbtStorage.Get(), m_rtStateObjectProps.Get());
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
  uint32_t nbMatrix = 4; // view, perspective, viewInv, perspectiveInv
  m_cameraBufferSize = nbMatrix * sizeof(XMMATRIX) + sizeof(float)*64;

  // Create the constant buffer for all matrices
  m_cameraBuffer = nv_helpers_dx12::CreateBuffer(
      m_device.Get(), m_cameraBufferSize, D3D12_RESOURCE_FLAG_NONE,
      D3D12_RESOURCE_STATE_GENERIC_READ, nv_helpers_dx12::kUploadHeapProps);

  // #DXR Extra - Refitting
  // Create a descriptor heap that will be used by the rasterization shaders:
  // Camera matrices and per-instance matrices
  m_constHeap = nv_helpers_dx12::CreateDescriptorHeap(
      m_device.Get(), 2, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, true);

  // Describe and create the constant buffer view.
  D3D12_CONSTANT_BUFFER_VIEW_DESC cbvDesc = {};
  cbvDesc.BufferLocation = m_cameraBuffer->GetGPUVirtualAddress();
  cbvDesc.SizeInBytes = m_cameraBufferSize;
}

// #DXR Extra: Perspective Camera
//--------------------------------------------------------------------------------
// Create and copies the viewmodel and perspective matrices of the camera
//
void Renderer::UpdateCameraBuffer() {
    std::vector<XMMATRIX> matrices(4);

    // Initialize the view matrix
    const glm::mat4 &mat = nv_helpers_dx12::CameraManip.getMatrix();
    memcpy(&matrices[0].r->m128_f32[0], glm::value_ptr(mat), 16 * sizeof(float));

    float fovAngleY = 45.0f * XM_PI / 180.0f;
    matrices[1] =
            XMMatrixPerspectiveFovRH(fovAngleY, m_aspectRatio, 0.1f, 1000.0f);

    XMVECTOR det;
    matrices[2] = XMMatrixInverse(&det, matrices[0]);
    matrices[3] = XMMatrixInverse(&det, matrices[1]);

    // Copy the matrix contents
    uint8_t *pData;
    ThrowIfFailed(m_cameraBuffer->Map(0, nullptr, (void **)&pData));
    memcpy(pData, matrices.data(), m_cameraBufferSize - sizeof(float)); // Adjust for the size of the float

    // Add the current system time as float
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
    float currentTime = static_cast<float>(millis % 100); // Convert milliseconds to seconds as float

    // Copy the current time into the buffer, right after the matrices
    memcpy(pData + m_cameraBufferSize - sizeof(float)*64, &currentTime, sizeof(float));

    m_cameraBuffer->Unmap(0, nullptr);
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
  m_materials.insert(m_materials.end(), materials.begin(), materials.end());
  m_materialIDs.insert(m_materialIDs.end(),materialIDs.begin(), materialIDs.end());
  materialVertexOffset=materialIDs.size();
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
        current->objectToWorld = inst.second;

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
        current->objectToWorldNormal = XMMatrixTranspose(XMMatrixInverse(&det, upper3x3));
        current++;
    }
  m_instanceProperties->Unmap(0, nullptr);
}
