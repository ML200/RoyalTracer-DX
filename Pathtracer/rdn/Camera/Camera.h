#pragma once
//====================================
//CAMERA STATE GPU BUFFER JITTER INPUT
//====================================
//self-contained, no rendering knowledge

#include "../Common.h"
#include "../glm/gtc/type_ptr.hpp"
#include "../manipulator.h"
#include "../DXRHelper.h"

class Camera {
public:
    void Init(ID3D12Device* device, UINT width, UINT height);

    //====================================
    //PER-FRAME
    //====================================
    void Update(float dt, bool keysDown[256], float aspectRatio);
    void UploadGPUBuffer(float aspectRatio);
    void AdvanceFrame();

    //====================================
    //INPUT
    //====================================
    void OnMouseButton(int x, int y);
    void OnMouseMove(int x, int y, bool lmb, bool rmb, bool mmb);

    //====================================
    //ACCESSORS
    //====================================
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

    //====================================
    //EDITOR-FRIENDLY STATE
    //====================================
    float   fovDegrees  = 60.0f;
    //projection-matrix params only, paths traced so no near-plane culling
    //keep near above ~0.01 so DLSS-RR projection stays well-conditioned
    float   nearPlane   = 0.01f;
    float   farPlane    = 10000.0f;
    float   moveSpeed   = 5.0f;

    //thin-lens DoF, aperture in world units, focus distance along view forward axis
    //synced into sunSettings (cbuffer tail) inside UploadGPUBuffer
    //aperture defaults to 0 = pinhole, raise in the camera panel to enable DoF
    float   apertureRadius = 0.0f;
    float   focusDistance  = 10.0f;

    SunSettings sunSettings;

private:
    ComPtr<ID3D12Resource>         m_buffer;
    ComPtr<ID3D12DescriptorHeap>   m_constHeap;
    UINT                           m_bufferSize = 0;

    XMMATRIX m_viewMatrix          = XMMatrixIdentity();
    XMMATRIX m_projMatrix          = XMMatrixIdentity();
    XMMATRIX m_projMatrixUnjittered= XMMatrixIdentity();
    XMMATRIX m_prevView            = XMMatrixIdentity();
    XMMATRIX m_prevProj            = XMMatrixIdentity();
    XMMATRIX m_prevProjUnjittered  = XMMatrixIdentity();

    float    m_jitterX = 0.0f, m_jitterY = 0.0f;
    uint32_t m_jitterFrameIndex = 0;
};
