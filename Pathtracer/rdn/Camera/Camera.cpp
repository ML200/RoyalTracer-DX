//====================================
//CAMERA
//====================================

#include "Camera.h"
#include "../nv_helpers_dx12/BottomLevelASGenerator.h"
#include "Windowsx.h"
#include "../DXRHelper.h"

void Camera::Init(ID3D12Device* device, UINT width, UINT height) {
    nv_helpers_dx12::CameraManip.setWindowSize(width, height);
    nv_helpers_dx12::CameraManip.setLookat(
        glm::vec3(-1.5f, 1.5f, 3.5f), glm::vec3(0, 1.0f, 0), glm::vec3(0, 1, 0));
    nv_helpers_dx12::CameraManip.setMode(nv_helpers_dx12::Manipulator::Fly);
    nv_helpers_dx12::CameraManip.setSpeed(moveSpeed);

    //GPU CB, 6 matrices + 8 extras (frameIdx, jitter.xy, cameraFar, walltime, pad*3) + SunSettings
    uint32_t matCount = 6;
    m_bufferSize = matCount * sizeof(XMMATRIX) + sizeof(float) * 8 + sizeof(SunSettings);
    m_bufferSize = (m_bufferSize + 255) & ~255;

    m_buffer = nv_helpers_dx12::CreateBuffer(
        device, m_bufferSize, D3D12_RESOURCE_FLAG_NONE,
        D3D12_RESOURCE_STATE_GENERIC_READ, nv_helpers_dx12::kUploadHeapProps);

    m_constHeap = nv_helpers_dx12::CreateDescriptorHeap(
        device, 2, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, true);
}

//====================================
//UPDATE
//====================================
void Camera::Update(float dt, bool keysDown[256], float aspectRatio) {
    m_wallTimeSec += dt;

    glm::vec3 eye, center, up;
    nv_helpers_dx12::CameraManip.getLookat(eye, center, up);
    glm::vec3 fwd   = glm::normalize(center - eye);
    glm::vec3 right = glm::normalize(glm::cross(fwd, up));

    glm::vec3 move(0.0f);
    if (keysDown['W'])        move += fwd;
    if (keysDown['S'])        move -= fwd;
    if (keysDown['D'])        move += right;
    if (keysDown['A'])        move -= right;
    if (keysDown[VK_SPACE])   move += up;
    if (keysDown[VK_CONTROL]) move -= up;

    if (glm::length(move) > 0.0f) {
        move = glm::normalize(move) * (moveSpeed * dt);
        eye += move; center += move;
        nv_helpers_dx12::CameraManip.setLookat(eye, center, up);
    }
}

//====================================
//UPLOAD GPU BUFFER
//====================================
void Camera::UploadGPUBuffer(float aspectRatio) {
    std::vector<XMMATRIX> matrices(6);
    const glm::mat4& viewMat = nv_helpers_dx12::CameraManip.getMatrix();
    memcpy(&matrices[0].r->m128_f32[0], glm::value_ptr(viewMat), 16 * sizeof(float));

    matrices[1] = XMMatrixPerspectiveFovRH(
        XMConvertToRadians(fovDegrees), aspectRatio, nearPlane, farPlane);

    m_jitterFrameIndex++;
    m_jitterX = Halton(m_jitterFrameIndex % 16 + 1, 2) - 0.5f;
    m_jitterY = Halton(m_jitterFrameIndex % 16 + 1, 3) - 0.5f;

    XMVECTOR det;
    matrices[2] = XMMatrixInverse(&det, matrices[0]);
    matrices[3] = XMMatrixInverse(&det, matrices[1]);
    matrices[4] = m_prevView;
    matrices[5] = m_prevProj;

    uint8_t* pData;
    if (FAILED(m_buffer->Map(0, nullptr, (void**)&pData))) return;
    memcpy(pData, matrices.data(), 6 * sizeof(XMMATRIX));
    //extra[3] = cameraFar (sky depth + spec hit distance)
    //extra[4] = walltime in seconds (drives framerate-independent auto-exposure)
    //extras 5..7 are pad to keep sunSettings 16B aligned
    float extra[8] = {
        (float)m_jitterFrameIndex, m_jitterX, m_jitterY, farPlane,
        m_wallTimeSec, 0.0f, 0.0f, 0.0f
    };
    memcpy(pData + 6 * sizeof(XMMATRIX), extra, sizeof(extra));
    //mirror camera DoF settings into the cbuffer tail before upload
    sunSettings.dofApertureRadius = apertureRadius;
    sunSettings.dofFocusDistance  = focusDistance;
    memcpy(pData + 6 * sizeof(XMMATRIX) + sizeof(extra), &sunSettings, sizeof(SunSettings));
    m_buffer->Unmap(0, nullptr);

    m_viewMatrix           = matrices[0];
    m_projMatrix           = matrices[1];
    m_projMatrixUnjittered = matrices[1];
    //prev matrices advanced in AdvanceFrame, after DLSS evaluates
}

//====================================
//ADVANCE FRAME
//====================================
void Camera::AdvanceFrame() {
    m_prevView            = m_viewMatrix;
    m_prevProj            = m_projMatrix;
    m_prevProjUnjittered  = m_projMatrixUnjittered;
}

//====================================
//MOUSE INPUT
//====================================
void Camera::OnMouseButton(int x, int y) {
    nv_helpers_dx12::CameraManip.setMousePosition(-x, -y);
}

void Camera::OnMouseMove(int x, int y, bool lmb, bool rmb, bool mmb) {
    if (!lmb && !rmb && !mmb) return;
    nv_helpers_dx12::Manipulator::Inputs inputs;
    inputs.lmb   = lmb;
    inputs.rmb   = rmb;
    inputs.mmb   = mmb;
    inputs.ctrl  = GetAsyncKeyState(VK_CONTROL);
    inputs.shift = GetAsyncKeyState(VK_SHIFT);
    inputs.alt   = GetAsyncKeyState(VK_MENU);
    nv_helpers_dx12::CameraManip.mouseMove(-x, -y, inputs);
}
