#pragma once

#include "../Common.h"
#include "../Core/ResourceFactory.h"

#include <sl.h>
#include <sl_consts.h>
#include <sl_dlss.h>
#include "sl_dlss_d.h"

class DLSSManager {
  public:
    void CreateResources(ID3D12Device* device, UINT displayWidth, UINT displayHeight);

    bool UpdateMode(ID3D12Device* device);

    void ForceReset() { m_forceReset = true; }

    bool LastEvaluationSucceeded() const { return m_lastEvaluationSucceeded; }
    bool LastEvaluationReset() const { return m_lastEvaluationReset; }

    void Evaluate(ID3D12GraphicsCommandList* cmdList, ID3D12Device* device, sl::FrameToken& frameToken,
                  sl::ViewportHandle viewport, float aspectRatio, const XMMATRIX& viewMatrix,
                  const XMMATRIX& prevViewMatrix, const XMMATRIX& prevProjMatrix, float jitterX, float jitterY,
                  uint32_t jitterFrameIndex, float fov, float nearZ, float farZ);

    UINT RenderWidth() const { return m_renderWidth; }
    UINT RenderHeight() const { return m_renderHeight; }
    UINT DisplayWidth() const { return m_displayWidth; }
    UINT DisplayHeight() const { return m_displayHeight; }
    sl::DLSSMode ActiveMode() const { return m_activeMode; }

    ID3D12Resource* Input() const { return m_input.Get(); }
    ID3D12Resource* Depth() const { return m_depth.Get(); }
    ID3D12Resource* MVec() const { return m_mvec.Get(); }
    ID3D12Resource* Normals() const { return m_normals.Get(); }
    ID3D12Resource* DiffuseAlbedo() const { return m_diffuseAlbedo.Get(); }
    ID3D12Resource* Output() const { return m_output.Get(); }
    ID3D12Resource* SpecularAlbedo() const { return m_specAlbedo.Get(); }
    ID3D12Resource* Roughness() const { return m_roughness.Get(); }
    ID3D12Resource* SpecMVec() const { return m_specMvec.Get(); }
    ID3D12Resource* SpecHitDist() const { return m_specHitDist.Get(); }
    ID3D12Resource* Transparency() const { return m_transparency.Get(); }
    ID3D12Resource* ColorBeforeTrans() const { return m_colorBeforeTrans.Get(); }
    ID3D12Resource* BiasHint() const { return m_biasHint.Get(); }
    ID3D12Resource* ResponsivityMask() const { return m_responsivityMask.Get(); }

    sl::DLSSMode mode = sl::DLSSMode::eDLAA;

    enum PresetSlot {
        kPresetDLAA = 0,
        kPresetQuality,
        kPresetBalanced,
        kPresetPerformance,
        kPresetUltraPerformance,
        kPresetUltraQuality,
        kPresetSlotCount
    };
    sl::DLSSDPreset rrPresets[kPresetSlotCount] = {sl::DLSSDPreset::ePresetF, sl::DLSSDPreset::ePresetF,
                                                   sl::DLSSDPreset::ePresetF, sl::DLSSDPreset::ePresetF,
                                                   sl::DLSSDPreset::ePresetF, sl::DLSSDPreset::ePresetF};

    bool rrLinkPresets = true;

    // How readily reconstruction drops accumulated history: -1 accumulates longest, +1 is most
    // responsive. Written per pixel by the shading pass rather than applied uniformly. A rough
    // surface's shading barely moves between frames and is happy accumulating; a smooth one
    // carries a sharp reflection that slides across it, so it is ramped part of the way back.
    // Moving water sits at the far end of its own: its sun glitter is a different set of crests
    // every frame, and the history length that resolves a static surface smears it.
    float rrResponsivityRough = -1.0f;  // at roughness 1
    float rrResponsivityMirror = -0.5f; // at roughness 0
    float rrWaterResponsivity = 1.0f;

    float sharpness = 0.5f;

    float jitterScale[2] = {1.0f, 1.0f};

    bool guideOffDepth = false;
    bool guideOffMV = false;
    bool guideOffNormals = false;
    bool guideOffRough = false;
    bool guideOffAlbedo = false;
    bool guideOffSpecAlb = false;
    bool guideOffSpecMV = false;
    bool guideOffPsr = false;
    bool guideOffMvBlend = false;

    bool untagSpecMV = false;

    uint32_t GuideOffFlags() const {
        // These bits match RS_FLAG_GUIDE_OFF_* in the shader interface.
        return (guideOffDepth ? 0x02000000u : 0u) | (guideOffMV ? 0x04000000u : 0u) |
               (guideOffNormals ? 0x08000000u : 0u) | (guideOffRough ? 0x10000000u : 0u) |
               (guideOffAlbedo ? 0x20000000u : 0u) | (guideOffSpecAlb ? 0x40000000u : 0u) |
               (guideOffSpecMV ? 0x80000000u : 0u);
    }

    bool clampEmitterSpikes = true;

    struct GuideSentinel {
        uint32_t mask = 0;
        uint32_t badCount = 0;
        uint32_t capCount = 0;
        uint32_t firstBad = 0;
        float maxLuma = 0.f;
        float maxMV = 0.f;
        float maxSpecMV = 0.f;
        uint64_t frame = 0;
        uint32_t lastMask = 0;
        uint32_t lastBad = 0;
        uint64_t lastFrame = 0;
    } sentinel;

    std::vector<std::string> evalDxMessages;
    uint64_t evalDxMessageTotal = 0;

  private:
    void CreateInputTextures(ID3D12Device* device);
    void ComputeRenderResolution();
    void RevertPresetIfPending(const wchar_t* stage);

    ComPtr<ID3D12Resource> m_input, m_depth, m_mvec, m_normals;
    ComPtr<ID3D12Resource> m_diffuseAlbedo, m_output;
    ComPtr<ID3D12Resource> m_specAlbedo, m_roughness, m_specMvec, m_specHitDist;
    ComPtr<ID3D12Resource> m_transparency, m_colorBeforeTrans;
    ComPtr<ID3D12Resource> m_biasHint;
    ComPtr<ID3D12Resource> m_responsivityMask;

    UINT m_displayWidth = 0, m_displayHeight = 0;
    UINT m_renderWidth = 0, m_renderHeight = 0;
    sl::DLSSMode m_activeMode = sl::DLSSMode::eOff;
    bool m_forceReset = false;
    bool m_lastEvaluationSucceeded = false;
    bool m_lastEvaluationReset = true;

    sl::DLSSDPreset m_activePresets[kPresetSlotCount] = {sl::DLSSDPreset::ePresetE, sl::DLSSDPreset::ePresetE,
                                                         sl::DLSSDPreset::ePresetE, sl::DLSSDPreset::ePresetE,
                                                         sl::DLSSDPreset::ePresetE, sl::DLSSDPreset::ePresetE};

    sl::DLSSDPreset m_lastGoodPresets[kPresetSlotCount] = {sl::DLSSDPreset::ePresetE, sl::DLSSDPreset::ePresetE,
                                                           sl::DLSSDPreset::ePresetE, sl::DLSSDPreset::ePresetE,
                                                           sl::DLSSDPreset::ePresetE, sl::DLSSDPreset::ePresetE};
    bool m_presetChangePending = false;

    XMMATRIX m_dlssPrevView = XMMatrixIdentity();
    XMMATRIX m_dlssPrevProj = XMMatrixIdentity();
};
