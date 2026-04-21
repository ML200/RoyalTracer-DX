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
    void AdvanceFrame();

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
    XMMATRIX                  PrevProjUnjittered() const { return m_prevProjUnjittered; }
    float                     JitterX()      const { return m_jitterX; }
    float                     JitterY()      const { return m_jitterY; }
    uint32_t                  JitterFrame()  const { return m_jitterFrameIndex; }
    void                      ResetJitter()        { m_jitterFrameIndex = 0; }

    nv_helpers_dx12::Manipulator& Manipulator() { return nv_helpers_dx12::CameraManip; }

    // ── Editor-friendly state ────────────────────────────────────
    float   fovDegrees  = 60.0f;
    //Purely projection-matrix parameters. Primary rays are path-traced
    //so there's no actual near-plane culling; near/far exist only to
    //feed DLSS-RR's projection and any host-side clip-space math. Keep
    //near above ~0.01 so DLSS's derived view/clip math stays well-
    //conditioned (very small values collapse the z-to-NDC mapping and
    //produce orientation-dependent stability issues in the denoiser).
    float   nearPlane   = 0.01f;
    float   farPlane    = 10000.0f;
    float   moveSpeed   = 5.0f;

    SunSettings sunSettings;

private:
    ComPtr<ID3D12Resource>         m_buffer;
    ComPtr<ID3D12DescriptorHeap>   m_constHeap;
    UINT                           m_bufferSize = 0;

    XMMATRIX m_viewMatrix          = XMMatrixIdentity();
    XMMATRIX m_projMatrix          = XMMatrixIdentity(); // jittered
    XMMATRIX m_projMatrixUnjittered= XMMatrixIdentity();
    XMMATRIX m_prevView            = XMMatrixIdentity();
    XMMATRIX m_prevProj            = XMMatrixIdentity(); // jittered
    XMMATRIX m_prevProjUnjittered  = XMMatrixIdentity();

    float    m_jitterX = 0.0f, m_jitterY = 0.0f;
    uint32_t m_jitterFrameIndex = 0;
};
