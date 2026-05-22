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
    //Snap the manipulator pose back to the initial spawn (matches Camera::Init).
    //Editor "Reset Camera" button calls this; pair with FlyCamController::Reset()
    //so the controller's cached forward doesn't immediately overwrite the pose.
    void ResetView();

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

    //True for one frame after Camera::ResetView() has been called. The
    //renderer reads it (and clears it) to invalidate temporal history that
    //references the pre-reset camera pose — DLSS RR history in particular,
    //since the upload step will have snapped prevView = view so DLSS sees
    //zero motion and would otherwise reuse the old frame's pixels.
    bool ConsumeResetPending() {
        const bool r = m_resetPending;
        m_resetPending = false;
        return r;
    }

    //====================================
    //FLOATING ORIGIN
    //====================================
    //Scene origin that follows the camera in 1 km steps. Subtracted from
    //world coords before they reach the GPU so ray origins / BLAS hits
    //stay in float32 range even at orbital altitudes. Use absolute camera
    //position = camera_in_shifted_frame + sceneOriginWorld for any math
    //that needs planet centered coords (atmosphere observer altitude).
    glm::vec3 getSceneOriginWorld() const { return m_sceneOriginWorld; }
    //Call BEFORE Scene::PrepareInstanceProperties each frame: snaps the
    //origin to a new grid point if the camera has drifted too far, sets
    //m_originShifted. Separate from UploadGPUBuffer so the renderer can
    //react (force TLAS rebuild, mark all instances dirty) in the same
    //frame instead of one frame late.
    void PollSceneOrigin();
    //True for one frame after the origin snapped to a new grid point; the
    //renderer consumes this to force a TLAS rebuild + re-shift of all
    //instance transforms. Cleared on read.
    bool consumeOriginShifted() {
        const bool s = m_originShifted;
        m_originShifted = false;
        return s;
    }

    //====================================
    //EDITOR-FRIENDLY STATE
    //====================================
    float   fovDegrees  = 60.0f;
    //projection-matrix params only, paths traced so no near-plane culling
    //keep near above ~0.01 so DLSS-RR projection stays well-conditioned
    float   nearPlane   = 0.01f;
    //farPlane drives both the projection matrix (DLSS RR) and the sky
    //depth fallback in Pass_shading_v8.hlsl. Must be >= the raygen ray
    //TMax (RAY_TMAX_PLANET = 1e9 in Constants_v8.hlsli) or distant surface
    //hits show up closer than the sky in the DLSS depth buffer. 1e9 m
    //covers planet diameter (~1.27e7) and orbital-scale grazing rays.
    float   farPlane    = 1.0e9f;
    float   moveSpeed   = 5.0f;

    //thin-lens DoF, aperture in world units, focus distance along view forward axis
    //synced into sunSettings (cbuffer tail) inside UploadGPUBuffer
    //aperture defaults to 0 = pinhole, raise in the camera panel to enable DoF
    float   apertureRadius = 0.0f;
    float   focusDistance  = 10.0f;

    SunSettings   sunSettings;
    //Volumetric cloud knobs, appended to the camera cbuffer tail after
    //SunSettings. Driven from the editor's Clouds panel.
    CloudSettings cloudSettings;

    //====================================
    //PLANET TERRAIN (Phase 5)
    //====================================
    //Procedural cube-sphere terrain params, appended to the camera cbuffer tail
    //after CloudSettings (6 scalar floats). planetCenter is ABSOLUTE world
    //coords. Set by the renderer from the planet StreamConfig at init so the
    //HLSL terrain shader samples the same surface the CPU tessellator built.
    glm::vec3 planetCenter           = glm::vec3(0.0f);
    float     planetRadius           = 6371000.0f;
    float     terrainHeightAmplitude = 0.001f;
    float     terrainHeightFrequency = 8.0f;

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
    //accumulated wall-clock seconds since Init, written into the cbuffer so
    //auto-exposure (and any future framerate-independent system) can derive dt
    float    m_wallTimeSec = 0.0f;

    //floating origin: scene origin snaps to a quantised grid as the camera
    //flies, so absolute world coords never get large enough to chew through
    //float32 precision. 1 km grid gives sub millimetre precision anywhere
    //inside one cell.
    glm::vec3 m_sceneOriginWorld = glm::vec3(0.0f);
    bool      m_originShifted    = false;
    //Set by ResetView, consumed by Renderer after UploadGPUBuffer. Drives
    //the prevView = view snap inside UploadGPUBuffer and the DLSS history
    //flush in the renderer.
    bool      m_resetPending     = false;
    static constexpr float kSceneOriginQuantumMeters = 1000.0f;
};
