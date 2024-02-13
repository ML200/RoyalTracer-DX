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

#include "nv_helpers_dx12/ShaderBindingTableGenerator.h"
#include "nv_helpers_dx12/TopLevelASGenerator.h"
#include "../src/Components/Vertex.h"

using namespace DirectX;

// Note that while ComPtr is used to manage the lifetime of resources on the
// CPU, it has no understanding of the lifetime of resources on the GPU. Apps
// must account for the GPU lifetime of resources to avoid destroying objects
// that may still be referenced by the GPU. An example of this can be found in
// the class method: OnDestroy().
using Microsoft::WRL::ComPtr;

class Renderer : public DXSample {
public:
  Renderer(UINT width, UINT height, std::wstring name);

  virtual void OnInit();
  virtual void OnUpdate();
  virtual void OnRender();
  virtual void OnDestroy();

private:
  static const UINT FrameCount = 2;

  // Pipeline objects.
  CD3DX12_VIEWPORT m_viewport;
  CD3DX12_RECT m_scissorRect;
  ComPtr<IDXGISwapChain3> m_swapChain;
  ComPtr<ID3D12Device5> m_device;
  ComPtr<ID3D12Resource> m_renderTargets[FrameCount];
  ComPtr<ID3D12CommandAllocator> m_commandAllocator;
  ComPtr<ID3D12CommandQueue> m_commandQueue;
  ComPtr<ID3D12RootSignature> m_rootSignature;
  ComPtr<ID3D12DescriptorHeap> m_rtvHeap;
  ComPtr<ID3D12PipelineState> m_pipelineState;
  ComPtr<ID3D12GraphicsCommandList4> m_commandList;
  UINT m_rtvDescriptorSize;

  // App resources.
  ComPtr<ID3D12Resource> m_vertexBuffer;
  D3D12_VERTEX_BUFFER_VIEW m_vertexBufferView;

  // Synchronization objects.
  UINT m_frameIndex;
  HANDLE m_fenceEvent;
  ComPtr<ID3D12Fence> m_fence;
  UINT64 m_fenceValue;

  void LoadPipeline();
  void LoadAssets();
  void PopulateCommandList();
  void WaitForPreviousFrame();

  void CheckRaytracingSupport();

  virtual void OnKeyUp(UINT8 key);
  bool m_raster = true;

  // #DXR
  struct AccelerationStructureBuffers {
    ComPtr<ID3D12Resource> pScratch;      // Scratch memory for AS builder
    ComPtr<ID3D12Resource> pResult;       // Where the AS is
    ComPtr<ID3D12Resource> pInstanceDesc; // Hold the matrices of the instances
  };

  ComPtr<ID3D12Resource> m_bottomLevelAS; // Storage for the bottom Level AS

  nv_helpers_dx12::TopLevelASGenerator m_topLevelASGenerator;
  AccelerationStructureBuffers m_topLevelASBuffers;
  std::vector<std::pair<ComPtr<ID3D12Resource>, DirectX::XMMATRIX>> m_instances;

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
  ComPtr<ID3D12RootSignature> CreateMissSignature();
  ComPtr<ID3D12RootSignature> CreateHitSignature();

  void CreateRaytracingPipeline();

  ComPtr<IDxcBlob> m_rayGenLibrary;
  ComPtr<IDxcBlob> m_hitLibrary;
  ComPtr<IDxcBlob> m_missLibrary;

  ComPtr<ID3D12RootSignature> m_rayGenSignature;
  ComPtr<ID3D12RootSignature> m_hitSignature;
  ComPtr<ID3D12RootSignature> m_missSignature;

  // Ray tracing pipeline state
  ComPtr<ID3D12StateObject> m_rtStateObject;
  // Ray tracing pipeline state properties, retaining the shader identifiers
  // to use in the Shader Binding Table
  ComPtr<ID3D12StateObjectProperties> m_rtStateObjectProps;

  // #DXR
  void CreateRaytracingOutputBuffer();
  void CreateShaderResourceHeap();
  ComPtr<ID3D12Resource> m_outputResource;
  ComPtr<ID3D12DescriptorHeap> m_srvUavHeap;

  // #DXR
  void CreateShaderBindingTable();
  nv_helpers_dx12::ShaderBindingTableGenerator m_sbtHelper;
  ComPtr<ID3D12Resource> m_sbtStorage;

  // #DXR Extra: Perspective Camera
  void CreateCameraBuffer();
  void UpdateCameraBuffer();
  ComPtr<ID3D12Resource> m_cameraBuffer;
  ComPtr<ID3D12DescriptorHeap> m_constHeap;
  uint32_t m_cameraBufferSize = 0;

  // #DXR Extra: Perspective Camera++
  void OnButtonDown(UINT32 lParam);
  void OnMouseMove(UINT8 wParam, UINT32 lParam);

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
  void CreateMengerSpongeVB();
  ComPtr<ID3D12Resource> m_mengerVB;
  ComPtr<ID3D12Resource> m_mengerIB;
  D3D12_VERTEX_BUFFER_VIEW m_mengerVBView;
  D3D12_INDEX_BUFFER_VIEW m_mengerIBView;

  UINT m_mengerIndexCount;
  UINT m_mengerVertexCount;

  // #DXR Extra - Another ray type
  ComPtr<IDxcBlob> m_shadowLibrary;
  ComPtr<ID3D12RootSignature> m_shadowSignature;

  // #DXR Extra - Refitting
  uint32_t m_time = 0;

  // #DXR Extra - Refitting
  /// Per-instance properties
  struct InstanceProperties {
    XMMATRIX objectToWorld;
    XMMATRIX objectToWorldNormal;
  };

  ComPtr<ID3D12Resource> m_instanceProperties;
  void CreateInstancePropertiesBuffer();
  void UpdateInstancePropertiesBuffer();
};
