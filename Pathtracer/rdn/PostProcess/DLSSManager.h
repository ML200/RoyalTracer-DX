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

    // Downstream temporal effects must follow RR resets and skip failed frames.
    bool LastEvaluationSucceeded() const { return m_lastEvaluationSucceeded; }
    bool LastEvaluationReset() const { return m_lastEvaluationReset; }

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
    //SL 2.14.1 documents F as the latest/default RR transformer. The editor
    //retains the numeric A..O range for diagnostics; removed/unsupported letters
    //are not distinct models and may revert to the runtime default or fail.
    //
    //Pin preset F explicitly for all quality modes so runtime updates do not
    //change the selected model. Other presets remain available in the editor.
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

    //Uniform signed responsivity mask; zero leaves the optional guide untagged.
    float rrResponsivity = -1.0f;

    //RCAS sharpening strength applied to the DLSS output, [0,1]; 0 = off.
    //NOT DLSSDOptions::sharpness — DLSS-RR ignores that field outright (NVIDIA's
    //DLSS-RR Programming Guide: "DLSS-RR will ignore DLSS options sharpness and
    //useAutoExposure"), so a slider bound to it would be a dead control. This
    //rides root constant 43 into Pass_postprocess_v8, which sharpens after AgX.
    float sharpness = 0.7f;

    //Scales ONLY the value reported to sl::Constants::jitterOffset. The raygen
    //keeps sampling at the camera's actual offset regardless (full Halton
    //[-0.5,+0.5] times Camera::jitterScale — the knob that scales the REAL
    //amplitude for both sides consistently), so at anything other than 1.0 we
    //are deliberately telling DLSS a different offset than we actually sampled
    //with.
    //
    //That is normally a correctness bug — the upscaler places each sample using
    //this value, so a mismatch misplaces the whole reconstruction. It is exposed
    //on purpose as a diagnostic: if RR2 (preset F) changed how it interprets
    //jitterOffset relative to RR, sweeping this reveals it as a stability minimum
    //somewhere off 1.0. {1,1} is the truthful value and the only one that is
    //correct by construction.
    //
    //PER-AXIS so sign-convention errors are separable: {-1,-1} tests a full
    //sign inversion of the report, {1,-1} tests a Y-only inversion (the
    //pixel-y-down vs NDC-y-up trap — the one that would preferentially
    //stair-step HORIZONTAL edges under static-camera history accumulation
    //while motion, which drops history, straightens them).
    float jitterScale[2] = { 1.0f, 1.0f };

    //====================================
    //GUIDE KILL-SWITCHES (diagnostics)
    //====================================
    //Per-guide disable toggles, editor "Guide inputs" tickboxes. true = that
    //guide is replaced with a NEUTRAL constant field at the end of
    //Pass_shading (depth 0 = far, MV 0, normals 0, roughness 1, diffuse
    //albedo white, spec albedo black, spec MV 0) — the tag stays in place
    //because RR treats these buffers as required. The Renderer packs them
    //into rs_flags high bits (RS_FLAG_GUIDE_OFF_* in Includes_v8.hlsli).
    //For isolating which guide the creeping preset-F instability follows;
    //flip one off, reset history, watch whether the creep still develops.
    bool guideOffDepth    = false;
    bool guideOffMV       = false;
    bool guideOffNormals  = false;
    bool guideOffRough    = false;
    bool guideOffAlbedo   = false;
    bool guideOffSpecAlb  = false;
    bool guideOffSpecMV   = false;
    //Spec MV is the one OPTIONAL guide: this drops its TAG entirely (null
    //resource, stale tag cleared) so RR falls back to internal specular
    //tracking — a different experiment than feeding it zeros.
    bool untagSpecMV      = false;

    //Renderer helper: the RS_FLAG_GUIDE_OFF_* bits for rs_flags. Values must
    //match Includes_v8.hlsli.
    uint32_t GuideOffFlags() const {
        return (guideOffDepth   ? 0x02000000u : 0u)
             | (guideOffMV      ? 0x04000000u : 0u)
             | (guideOffNormals ? 0x08000000u : 0u)
             | (guideOffRough   ? 0x10000000u : 0u)
             | (guideOffAlbedo  ? 0x20000000u : 0u)
             | (guideOffSpecAlb ? 0x40000000u : 0u)
             | (guideOffSpecMV  ? 0x80000000u : 0u);
    }

    //When true, Pass_shading luminance-clamps emitter radiance (DLSS_EMITTER_CAP)
    //before the reversible DlssReinhard pre-tonemap. This pulls a large bright
    //emitter (e.g. a lamp filling much of a dark scene) off the [0,1] rail so
    //DLSS RR — and the postprocess inverse — stop amplifying denoiser residual
    //error into artefacts. Only the emitter's DLSS colour input changes; the rest
    //of the pipeline (guides, decode, post-process) is untouched, so it is immune
    //to DLSS sub-pixel jitter. Off = unclamped (classic) DLSS input.
    bool clampEmitterSpikes = true;

    //====================================
    //GUIDE SENTINEL + EVALUATE-WINDOW DX MESSAGES (diagnostics)
    //====================================
    //Per-frame anomaly stats over the guide values handed to RR, computed by
    //the Pass_shading sentinel block into gAutoExpose bytes 32..63 and copied
    //back by the Renderer (2-frame latency). `last*` latch the most recent
    //anomalous frame so a one-frame glitch stays visible in the editor.
    struct GuideSentinel {
        uint32_t mask      = 0;    //anomaly bits, see the Pass_shading sentinel block
        uint32_t badCount  = 0;    //pixels with any anomaly bit set
        uint32_t capCount  = 0;    //pixels at/above the PT input luma cap
        uint32_t firstBad  = 0;    //((y+1)<<16)|(x+1) of first bad pixel, 0 = none
        float    maxLuma   = 0.f;  //max input luma (post-encode)
        float    maxMV     = 0.f;  //max |mv| component, pixels
        float    maxSpecMV = 0.f;  //max |spec mv| component, pixels
        uint64_t frame     = 0;    //renderer frame the stats belong to
        uint32_t lastMask  = 0;    //latched: mask of the most recent anomalous frame
        uint32_t lastBad   = 0;    //latched: firstBad of that frame
        uint64_t lastFrame = 0;    //latched: which frame that was (0 = never)
    } sentinel;

    //D3D12 debug-layer messages generated DURING slEvaluateFeature. Break-on-
    //error is deliberately suppressed for the evaluate window (SL plugins trip
    //benign validation), which also meant any REAL violation inside evaluate
    //was console-only and easy to miss — captured here for the editor instead.
    //Tail of the last few (severity <= warning); total counts every one seen.
    std::vector<std::string> evalDxMessages;
    uint64_t                 evalDxMessageTotal = 0;

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
    ComPtr<ID3D12DescriptorHeap> m_responsivityGpuHeap, m_responsivityCpuHeap;

    ResourceStateTracker m_state;

    UINT m_displayWidth  = 0, m_displayHeight = 0;
    UINT m_renderWidth   = 0, m_renderHeight  = 0;
    sl::DLSSMode m_activeMode = sl::DLSSMode::eOff;
    bool m_forceReset = false;
    bool m_lastEvaluationSucceeded = false;
    bool m_lastEvaluationReset = true;

    //Mirror of rrPresets as last handed to slDLSSDSetOptions. Swapping preset makes
    //the plugin rebuild the feature around a different model, so the history that
    //accumulated under the old one has to be dropped along with it.
    sl::DLSSDPreset m_activePresets[kPresetSlotCount] = {
        sl::DLSSDPreset::ePresetE, sl::DLSSDPreset::ePresetE, sl::DLSSDPreset::ePresetE,
        sl::DLSSDPreset::ePresetE, sl::DLSSDPreset::ePresetE, sl::DLSSDPreset::ePresetE
    };
    //Last letter that completed a frame, and whether the one in flight still has to
    //prove itself. See RevertPresetIfPending — NGX has no "is this preset supported"
    //query, so a bad letter can only be detected by trying it.
    sl::DLSSDPreset m_lastGoodPresets[kPresetSlotCount] = {
        sl::DLSSDPreset::ePresetE, sl::DLSSDPreset::ePresetE, sl::DLSSDPreset::ePresetE,
        sl::DLSSDPreset::ePresetE, sl::DLSSDPreset::ePresetE, sl::DLSSDPreset::ePresetE
    };
    bool m_presetChangePending = false;

    XMMATRIX m_dlssPrevView = XMMatrixIdentity();
    XMMATRIX m_dlssPrevProj = XMMatrixIdentity();
};
