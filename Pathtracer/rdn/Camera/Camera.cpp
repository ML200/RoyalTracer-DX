//====================================
//CAMERA
//====================================

#include "Camera.h"
#include "../nv_helpers_dx12/BottomLevelASGenerator.h"
#include "Windowsx.h"
#include "../DXRHelper.h"
#include "../glm/gtc/matrix_transform.hpp"  // glm::lookAt for floating origin

void Camera::ResetView() {
    nv_helpers_dx12::CameraManip.setLookat(
        glm::vec3(-1.5f, 1.5f, 3.5f), glm::vec3(0, 1.0f, 0), glm::vec3(0, 1, 0));

    //Snap the floating origin back to absolute zero. Without this, the
    //manipulator eye lives at (-1.5, 1.5, 3.5) in the OLD shifted frame
    //while sceneOriginWorld is still parked far away from a previous
    //flight — the absolute spawn comes terrain at (-1.5 + sceneOrigin,
    //1.5 + sceneOrigin.y, 3.5 + sceneOrigin.z), nowhere near the
    //expected world zero. Mirrors PollSceneOrigin's prev-view fix-up so
    //the snap stays MV-continuous even before UploadGPUBuffer overrides
    //prev = view below.
    if (m_sceneOriginWorld != glm::vec3(0.0f)) {
        const glm::vec3 shiftDelta = -m_sceneOriginWorld;  // newOrigin - oldOrigin = 0 - old
        m_sceneOriginWorld = glm::vec3(0.0f);
        m_originShifted    = true;
        const XMMATRIX T = XMMatrixTranslation(shiftDelta.x,
                                               shiftDelta.y,
                                               shiftDelta.z);
        m_prevView = XMMatrixMultiply(T, m_prevView);
    }

    //Flag the reset for UploadGPUBuffer (prevView=view snap) and for the
    //renderer (DLSS history flush). Cleared by Camera::ConsumeResetPending.
    m_resetPending = true;
}

void Camera::Init(ID3D12Device* device, UINT width, UINT height) {
    nv_helpers_dx12::CameraManip.setWindowSize(width, height);
    ResetView();
    nv_helpers_dx12::CameraManip.setMode(nv_helpers_dx12::Manipulator::Fly);
    nv_helpers_dx12::CameraManip.setSpeed(moveSpeed);

    //GPU CB, 6 matrices + 8 extras (frameIdx, jitter.xy, cameraFar, walltime, pad*3)
    //+ SunSettings + CloudSettings + 16 planet-terrain floats appended at the
    //tail (6 base: centre xyz, radius, amp, freq; +10 camera-local noise
    //frame: noiseOrigin xyz, noiseFrac xyz, camDir xyz, radialScale).
    uint32_t matCount = 6;
    m_bufferSize = matCount * sizeof(XMMATRIX) + sizeof(float) * 8
                 + sizeof(SunSettings) + sizeof(CloudSettings)
                 + sizeof(float) * 16;
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
    //Floating origin snap lives in UploadGPUBuffer instead of here so the
    //engine layer's FlyCamController (which bypasses Camera::Update) still
    //gets the shift treatment. See UploadGPUBuffer for the actual logic.
}

//====================================
//UPLOAD GPU BUFFER
//====================================
//Snap m_sceneOriginWorld to the nearest 1 km grid point if the camera has
//drifted more than half a quantum away.
//
//Convention: the *manipulator stores eye/center in shifted coords*, so the
//absolute world position is always manipulator_eye + m_sceneOriginWorld.
//When we snap, we have to also shift the manipulator by the same delta so
//its stored values stay small (within ±500 m of zero). Without this, the
//manipulator accumulates the full absolute eye position as float32 and
//`center = eye + fwd` loses direction precision at orbital distances —
//that's what causes the rotation wobble at large absolute coordinates.
//
//Side effect: prevView is pre-multiplied by T(shiftDelta) so motion
//vectors stay continuous across the snap (no per-pixel MV spike).
void Camera::PollSceneOrigin() {
    glm::vec3 eye, center, up;
    nv_helpers_dx12::CameraManip.getLookat(eye, center, up);
    //eye is already in shifted coords (the convention above). Recover
    //absolute by adding sceneOriginWorld back; that's the value we snap.
    const glm::vec3 eyeAbs = eye + m_sceneOriginWorld;
    const glm::vec3 driftAbs = eyeAbs - m_sceneOriginWorld;
    if (std::fabs(driftAbs.x) > kSceneOriginQuantumMeters * 0.5f ||
        std::fabs(driftAbs.y) > kSceneOriginQuantumMeters * 0.5f ||
        std::fabs(driftAbs.z) > kSceneOriginQuantumMeters * 0.5f) {
        const glm::vec3 snapped = glm::round(eyeAbs / kSceneOriginQuantumMeters)
                                * kSceneOriginQuantumMeters;
        if (snapped != m_sceneOriginWorld) {
            const glm::vec3 shiftDelta = snapped - m_sceneOriginWorld;
            m_sceneOriginWorld = snapped;
            m_originShifted    = true;

            //Push the shift into the manipulator so its stored eye/center
            //stay small in float32. Equivalent statement of the invariant:
            //  eye_manip_new + sceneOrigin_new == eye_manip_old + sceneOrigin_old
            //  => eye_manip_new = eye_manip_old - shiftDelta
            const glm::vec3 eyeShifted    = eye    - shiftDelta;
            const glm::vec3 centerShifted = center - shiftDelta;
            nv_helpers_dx12::CameraManip.setLookat(eyeShifted, centerShifted, up);

            //Re-express prevView in the new shifted frame. For row vector
            //pos: pos_new_shifted = pos_old_shifted - shiftDelta, so
            //pos_new_shifted * (T(shiftDelta) * prevView_old) reproduces
            //the same view-space result as the original pos_old_shifted *
            //prevView_old. Pre-multiplying by T(shiftDelta) just adjusts
            //the translation row of prevView; projection is untouched.
            const XMMATRIX T = XMMatrixTranslation(shiftDelta.x,
                                                   shiftDelta.y,
                                                   shiftDelta.z);
            m_prevView = XMMatrixMultiply(T, m_prevView);
        }
    }
}

void Camera::UploadGPUBuffer(float aspectRatio) {
    std::vector<XMMATRIX> matrices(6);

    //Floating origin: the manipulator stores eye/center already in shifted
    //coords (PollSceneOrigin maintains that invariant). So we use them
    //as-is for the view matrix — no further subtraction needed.
    //getMatrix() returns the lookAt of those shifted values, which is
    //exactly what the GPU wants.
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

    //On a camera reset, force prev = current so motion vectors land at zero.
    //The previous pose is from a totally different (pre-reset) camera, so
    //any non-zero MV is garbage — DLSS would lock onto the wrong history
    //and ReSTIR's temporal reproject would land in arbitrary pixels. The
    //flag stays set so the renderer can also flush DLSS RR history via
    //ForceReset right after this call.
    if (m_resetPending) {
        m_prevView           = matrices[0];
        m_prevProj           = matrices[1];
        m_prevProjUnjittered = matrices[1];
    }

    matrices[4] = m_prevView;
    matrices[5] = m_prevProj;

    uint8_t* pData;
    if (FAILED(m_buffer->Map(0, nullptr, (void**)&pData))) return;
    memcpy(pData, matrices.data(), 6 * sizeof(XMMATRIX));
    //extra[3] = cameraFar (sky depth + spec hit distance)
    //extra[4] = walltime in seconds (drives framerate-independent auto-exposure)
    //extras 5..7 = sceneOriginWorld.xyz (floating origin shift, used by shaders
    //              to reconstruct absolute camera position for atmosphere math)
    float extra[8] = {
        (float)m_jitterFrameIndex, m_jitterX, m_jitterY, farPlane,
        m_wallTimeSec,
        m_sceneOriginWorld.x, m_sceneOriginWorld.y, m_sceneOriginWorld.z
    };
    memcpy(pData + 6 * sizeof(XMMATRIX), extra, sizeof(extra));
    //mirror camera DoF settings into the cbuffer tail before upload
    sunSettings.dofApertureRadius = apertureRadius;
    sunSettings.dofFocusDistance  = focusDistance;
    memcpy(pData + 6 * sizeof(XMMATRIX) + sizeof(extra), &sunSettings, sizeof(SunSettings));
    //CloudSettings follows SunSettings; both packed as scalar floats so the
    //HLSL CB packing rules concatenate them cleanly across register slots.
    memcpy(pData + 6 * sizeof(XMMATRIX) + sizeof(extra) + sizeof(SunSettings),
           &cloudSettings, sizeof(CloudSettings));
    //Procedural-terrain tail: 6 scalar floats following CloudSettings - planet
    //centre xyz, radius, and the (vestigial) heightmap amplitude/frequency.
    //planetCentre/radius feed the cloud + atmosphere code (TerrainHeight, planet
    //sphere). Must match the cbuffer tail in Includes_v8.hlsli exactly.
    const float planetTail[6] = {
        planetCenter.x, planetCenter.y, planetCenter.z,
        planetRadius, terrainHeightAmplitude, terrainHeightFrequency
    };
    memcpy(pData + 6 * sizeof(XMMATRIX) + sizeof(extra)
                 + sizeof(SunSettings) + sizeof(CloudSettings),
           planetTail, sizeof(planetTail));
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
