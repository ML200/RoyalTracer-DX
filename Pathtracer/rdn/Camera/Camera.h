#pragma once
// ═══════════════════════════════════════════════════════════════════
// Camera/Camera.h — Camera state, GPU buffer, jitter, and input.
//                   Self-contained; knows nothing about rendering.
// ═══════════════════════════════════════════════════════════════════

#include "../Common.h"
#include "../glm/gtc/type_ptr.hpp"
#include "../manipulator.h"
#include "../DXRHelper.h"

class Camera {
public:
    void Init(ID3D12Device* device, UINT width, UINT height);

    // ── Per-frame ────────────────────────────────────────────────
    void Update(float dt, bool keysDown[256], float aspectRatio);
    void UploadGPUBuffer(float aspectRatio);

    // ── Input ────────────────────────────────────────────────────
    void OnMouseButton(int x, int y);
    void OnMouseMove(int x, int y, bool lmb, bool rmb, bool mmb);

    // ── Accessors ────────────────────────────────────────────────
    ID3D12Resource*           GPUBuffer()    const { return m_buffer.Get(); }
    UINT                      BufferSize()   const { return m_bufferSize; }
    XMMATRIX                  ViewMatrix()   const { return m_viewMatrix; }
    XMMATRIX                  ProjMatrix()   const { return m_projMatrix; }
    XMMATRIX                  PrevView()     const { return m_prevView; }
    XMMATRIX                  PrevProj()     const { return m_prevProj; }
    float                     JitterX()      const { return m_jitterX; }
    float                     JitterY()      const { return m_jitterY; }
    uint32_t                  JitterFrame()  const { return m_jitterFrameIndex; }

    nv_helpers_dx12::Manipulator& Manipulator() { return nv_helpers_dx12::CameraManip; }

    // ── Editor-friendly state ────────────────────────────────────
    float   fovDegrees  = 60.0f;
    float   nearPlane   = 0.00001f;
    float   farPlane    = 10000.0f;
    float   moveSpeed   = 5.0f;

private:
    ComPtr<ID3D12Resource>         m_buffer;
    ComPtr<ID3D12DescriptorHeap>   m_constHeap;
    UINT                           m_bufferSize = 0;

    XMMATRIX m_viewMatrix = XMMatrixIdentity();
    XMMATRIX m_projMatrix = XMMatrixIdentity();
    XMMATRIX m_prevView   = XMMatrixIdentity();
    XMMATRIX m_prevProj   = XMMatrixIdentity();

    float    m_jitterX = 0.0f, m_jitterY = 0.0f;
    uint32_t m_jitterFrameIndex = 0;
};
