#pragma once
#include <array>

#include "../Common.h"
#include "../glm/gtc/type_ptr.hpp"
#include "../manipulator.h"
#include "../DXRHelper.h"

class Camera {
  public:
    void Init(ID3D12Device* device, UINT width, UINT height);

    void ResetView();

    void AdvanceTime(float dt) { m_wallTimeSec += dt; }
    void UploadGPUBuffer(float aspectRatio);
    void AdvanceFrame();

    void OnMouseMove(int x, int y, bool lmb, bool rmb, bool mmb);

    ID3D12Resource* GPUBuffer() const { return m_buffer.Get(); }
    UINT BufferSize() const { return m_bufferSize; }
    XMMATRIX ViewMatrix() const { return m_viewMatrix; }
    XMMATRIX PrevView() const { return m_prevView; }
    XMMATRIX PrevProj() const { return m_prevProj; }
    float JitterX() const { return m_jitterX; }
    float JitterY() const { return m_jitterY; }
    uint32_t JitterFrame() const { return m_jitterFrameIndex; }
    void ResetJitter() { m_jitterFrameIndex = 0; }

    nv_helpers_dx12::Manipulator& Manipulator() { return nv_helpers_dx12::CameraManip; }

    bool ConsumeResetPending() {
        const bool r = m_resetPending;
        m_resetPending = false;
        return r;
    }

    glm::vec3 getSceneOriginWorld() const { return m_sceneOriginWorld; }

    void PollSceneOrigin(); // Call before preparing scene instance transforms.

    bool consumeOriginShifted() {
        const bool s = m_originShifted;
        m_originShifted = false;
        return s;
    }

    float fovDegrees = 60.0f;

    float nearPlane = 0.01f;

    float farPlane = 1.0e9f;
    float moveSpeed = 5.0f;

    float apertureRadius = 0.0f;
    float focusDistance = 10.0f;

    float jitterScale = 1.0f;

    SunSettings sunSettings;

    glm::vec3 planetCenter = glm::vec3(0.0f);
    float planetRadius = 6371000.0f;

    float skyGroundY = 0.0f;
    // Below the lowest trough, so troughs never read as underground.
    float oceanGroundY = 3.0e38f;
    float terrainHeightFrequency = 0.0f;

    // Instances at or above this are ocean tiles.
    uint32_t oceanInstanceBase = 0xFFFFFFFFu;
    bool oceanEnabled = false;

    // DLSS responsivity mask values, mirrored from DLSSManager.
    float dlssResponsivityRough = -1.0f;
    float dlssResponsivityMirror = -0.5f;
    float dlssWaterResponsivity = 1.0f;

  private:
    ComPtr<ID3D12Resource> m_buffer;
    UINT m_bufferSize = 0;

    XMMATRIX m_viewMatrix = XMMatrixIdentity();
    XMMATRIX m_projMatrix = XMMatrixIdentity();
    XMMATRIX m_prevView = XMMatrixIdentity();
    XMMATRIX m_prevProj = XMMatrixIdentity();

    float m_jitterX = 0.0f, m_jitterY = 0.0f;
    uint32_t m_jitterFrameIndex = 0;

    float m_wallTimeSec = 0.0f;

    glm::vec3 m_sceneOriginWorld = glm::vec3(0.0f);
    bool m_originShifted = false;

    bool m_resetPending = false;
    static constexpr float kSceneOriginQuantumMeters = 1000.0f;
};
