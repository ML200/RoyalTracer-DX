#pragma once
//====================================
//DLSS MANAGER
//====================================

#include "../Common.h"
#include "../Core/ResourceFactory.h"
#include "../ResourceStateTracker.h"

#include <sl.h>
#include <sl_consts.h>
#include <sl_dlss.h>
#include "sl_dlss_d.h"

class DLSSManager {
public:
    void CreateResources(ID3D12Device* device, UINT displayWidth, UINT displayHeight);

    //returns true if resources were recreated, caller should rebuild SRV heap
    bool UpdateMode(ID3D12Device* device);

    //Flag the next Evaluate to pass reset=eTrue to DLSS RR, dropping all
    //temporal history. Called by the renderer when the camera teleports
    //(Camera::ResetView) — without this, prevView=view from the camera
    //reset path makes DLSS think the camera was static, so it would reuse
    //pixels rendered from the pre-reset pose.
    void ForceReset() { m_forceReset = true; }

    void Evaluate(
        ID3D12GraphicsCommandList* cmdList,
        ID3D12Device* device,
        sl::FrameToken& frameToken,
        sl::ViewportHandle viewport,
        float aspectRatio,
        const XMMATRIX& viewMatrix,
        const XMMATRIX& prevViewMatrix,
        const XMMATRIX& prevProjMatrix,
        float jitterX, float jitterY,
        uint32_t jitterFrameIndex,
        float fov, float nearZ, float farZ);

    //====================================
    //RESOLUTION ACCESSORS
    //====================================
    UINT RenderWidth()  const { return m_renderWidth; }
    UINT RenderHeight() const { return m_renderHeight; }
    UINT DisplayWidth() const { return m_displayWidth; }
    UINT DisplayHeight()const { return m_displayHeight; }
    sl::DLSSMode ActiveMode() const { return m_activeMode; }

    //====================================
    //RESOURCE ACCESSORS
    //====================================
    ID3D12Resource* Input()            const { return m_input.Get(); }
    ID3D12Resource* Depth()            const { return m_depth.Get(); }
    ID3D12Resource* MVec()             const { return m_mvec.Get(); }
    ID3D12Resource* Normals()          const { return m_normals.Get(); }
    ID3D12Resource* DiffuseAlbedo()    const { return m_diffuseAlbedo.Get(); }
    ID3D12Resource* Output()           const { return m_output.Get(); }
    ID3D12Resource* SpecularAlbedo()   const { return m_specAlbedo.Get(); }
    ID3D12Resource* Roughness()        const { return m_roughness.Get(); }
    ID3D12Resource* SpecMVec()         const { return m_specMvec.Get(); }
    ID3D12Resource* SpecHitDist()      const { return m_specHitDist.Get(); }
    ID3D12Resource* Transparency()     const { return m_transparency.Get(); }
    ID3D12Resource* ColorBeforeTrans() const { return m_colorBeforeTrans.Get(); }
    ID3D12Resource* BiasHint()         const { return m_biasHint.Get(); }

    //editor-exposed
    sl::DLSSMode mode = sl::DLSSMode::eDLAA;

    //====================================
    //RR MODEL PRESETS (editor-exposed)
    //====================================
    //DLSSDOptions carries one preset slot per quality mode and the plugin reads
    //the slot matching the active DLSSMode, so all six travel together.
    //
    //The letter -> model mapping lives in the runtime DLLs (sl.dlss_d.dll /
    //nvngx_dlssd.dll), not in these headers: we compile against the SL 2.12.0
    //headers but ship SL 2.12.129 / NGX 310.7.129 binaries, so the "reverts to
    //default" comments in sl_dlss_d.h understate what the runtime actually
    //supports. The enum is a plain uint32 handed straight to the plugin, so the
    //whole A..O range is selectable and a letter the runtime doesn't know falls
    //back to the default model rather than failing.
    //
    //Specifically: sl_dlss_d.h still documents ePresetF as "reverts to default",
    //but on NGX 310.7.128 and later F IS the DLSS 4.5 / transformer-2 RR model,
    //and it is what eDefault resolves to there (310.7.0 defaulted to D). F is
    //pinned explicitly below rather than left at eDefault so the model stays put
    //if the shipped DLLs are ever rolled back; against an older runtime F falls
    //back to that runtime's default rather than failing, per the paragraph above.
    enum PresetSlot {
        kPresetDLAA = 0,
        kPresetQuality,
        kPresetBalanced,
        kPresetPerformance,
        kPresetUltraPerformance,
        kPresetUltraQuality,
        kPresetSlotCount
    };
    sl::DLSSDPreset rrPresets[kPresetSlotCount] = {
        sl::DLSSDPreset::ePresetF, sl::DLSSDPreset::ePresetF, sl::DLSSDPreset::ePresetF,
        sl::DLSSDPreset::ePresetF, sl::DLSSDPreset::ePresetF, sl::DLSSDPreset::ePresetF
    };
    //UI convenience: when set, picking a letter writes it into every slot.
    bool rrLinkPresets = true;

    //RCAS sharpening strength applied to the DLSS output, [0,1]; 0 = off.
    //NOT DLSSDOptions::sharpness — DLSS-RR ignores that field outright (NVIDIA's
    //DLSS-RR Programming Guide: "DLSS-RR will ignore DLSS options sharpness and
    //useAutoExposure"), so a slider bound to it would be a dead control. This
    //rides root constant 43 into Pass_postprocess_v8, which sharpens after AgX.
    float sharpness = 0.5f;

    //Scales ONLY the value reported to sl::Constants::jitterOffset. The raygen
    //keeps sampling at the full Halton [-0.5,+0.5] offset regardless, so at
    //anything other than 1.0 we are deliberately telling DLSS a different offset
    //than we actually sampled with.
    //
    //That is normally a correctness bug — the upscaler places each sample using
    //this value, so a mismatch misplaces the whole reconstruction. It is exposed
    //on purpose as a diagnostic: if RR2 (preset F) changed how it interprets
    //jitterOffset relative to RR, sweeping this reveals it as a stability minimum
    //somewhere off 1.0. 1.0 is the truthful value and the only one that is
    //correct by construction.
    float jitterScale = 1.0f;

    //When true, Pass_shading luminance-clamps emitter radiance (DLSS_EMITTER_CAP)
    //before the reversible DlssReinhard pre-tonemap. This pulls a large bright
    //emitter (e.g. a lamp filling much of a dark scene) off the [0,1] rail so
    //DLSS RR — and the postprocess inverse — stop amplifying denoiser residual
    //error into artefacts. Only the emitter's DLSS colour input changes; the rest
    //of the pipeline (guides, decode, post-process) is untouched, so it is immune
    //to DLSS sub-pixel jitter. Off = unclamped (classic) DLSS input.
    bool clampEmitterSpikes = false;

private:
    void CreateInputTextures(ID3D12Device* device);
    void ComputeRenderResolution();
    void RevertPresetIfPending(const wchar_t* stage);

    ComPtr<ID3D12Resource> m_input, m_depth, m_mvec, m_normals;
    ComPtr<ID3D12Resource> m_diffuseAlbedo, m_output;
    ComPtr<ID3D12Resource> m_specAlbedo, m_roughness, m_specMvec, m_specHitDist;
    ComPtr<ID3D12Resource> m_transparency, m_colorBeforeTrans;
    ComPtr<ID3D12Resource> m_biasHint;

    ResourceStateTracker m_state;

    UINT m_displayWidth  = 0, m_displayHeight = 0;
    UINT m_renderWidth   = 0, m_renderHeight  = 0;
    sl::DLSSMode m_activeMode = sl::DLSSMode::eOff;
    bool m_forceReset = false;

    //Mirror of rrPresets as last handed to slDLSSDSetOptions. Swapping preset makes
    //the plugin rebuild the feature around a different model, so the history that
    //accumulated under the old one has to be dropped along with it.
    sl::DLSSDPreset m_activePresets[kPresetSlotCount] = {
        sl::DLSSDPreset::ePresetF, sl::DLSSDPreset::ePresetF, sl::DLSSDPreset::ePresetF,
        sl::DLSSDPreset::ePresetF, sl::DLSSDPreset::ePresetF, sl::DLSSDPreset::ePresetF
    };
    //Last letter that completed a frame, and whether the one in flight still has to
    //prove itself. See RevertPresetIfPending — NGX has no "is this preset supported"
    //query, so a bad letter can only be detected by trying it.
    sl::DLSSDPreset m_lastGoodPresets[kPresetSlotCount] = {
        sl::DLSSDPreset::ePresetF, sl::DLSSDPreset::ePresetF, sl::DLSSDPreset::ePresetF,
        sl::DLSSDPreset::ePresetF, sl::DLSSDPreset::ePresetF, sl::DLSSDPreset::ePresetF
    };
    bool m_presetChangePending = false;

    XMMATRIX m_dlssPrevView = XMMatrixIdentity();
    XMMATRIX m_dlssPrevProj = XMMatrixIdentity();
};
