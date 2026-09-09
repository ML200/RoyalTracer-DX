//====================================
//DLSS MANAGER
//====================================

#include "../stdafx.h"
#include "DLSSManager.h"
#include "../../shaders/DlssGuideLayout.h"
#include "../DXRHelper.h"
#include "../glm/gtc/type_ptr.hpp"
#include "../manipulator.h"

#include <sl.h>
#include <sl_consts.h>
#include <sl_helpers.h>
#include <sl_dlss.h>
#include "sl_dlss_d.h"
#include <limits>   //motionVectorsInvalidValue sentinel

#undef SL_CHECK
#define SL_CHECK(x) do { sl::Result r = (x); if (r != sl::Result::eOk) { \
    std::wcout << L"[SL] " << L#x << L" failed: " << (int)r << std::endl; return; } \
} while(0)

// ─────────────────────────────────────────────────────────────────
void DLSSManager::ComputeRenderResolution() {
    // DLSS-RR supported modes and their scale factors.
    // sl::DLSSMode enum: eOff=0, eDLAA=5(?), eMaxQuality=3, eBalanced=4, etc.
    // Use the actual enum constants rather than integer assumptions.
    float scale = 1.0f;

    if (mode == sl::DLSSMode::eOff || mode == sl::DLSSMode::eDLAA) {
        scale = 1.0f;
    } else if (mode == sl::DLSSMode::eMaxQuality) {
        scale = 1.0f / 1.5f;   // "Quality" in DLSS-RR
    } else if (mode == sl::DLSSMode::eBalanced) {
        scale = 1.0f / 1.7f;
    } else {
        // Fallback for any unsupported mode — run at full res
        scale = 1.0f;
    }

    m_renderWidth  = (std::max)(2u, (UINT)(m_displayWidth  * scale) & ~1u);
    m_renderHeight = (std::max)(2u, (UINT)(m_displayHeight * scale) & ~1u);
}

// ─────────────────────────────────────────────────────────────────
void DLSSManager::CreateInputTextures(ID3D12Device* device) {
    // Input textures at render resolution
    auto createRenderTex = [&](ComPtr<ID3D12Resource>& res, DXGI_FORMAT fmt, const wchar_t* name) {
        D3D12_RESOURCE_DESC d = {};
        d.Dimension        = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        d.Width            = m_renderWidth;
        d.Height           = m_renderHeight;
        d.DepthOrArraySize = 1;
        d.MipLevels        = 1;
        d.Format           = fmt;
        d.SampleDesc.Count = 1;
        d.Flags            = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        ThrowIfFailed(device->CreateCommittedResource(
            &nv_helpers_dx12::kDefaultHeapProps, D3D12_HEAP_FLAG_NONE,
            &d, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
            IID_PPV_ARGS(&res)));
        res->SetName(name);
    };

    // Output texture at display resolution
    auto createDisplayTex = [&](ComPtr<ID3D12Resource>& res, DXGI_FORMAT fmt, const wchar_t* name) {
        D3D12_RESOURCE_DESC d = {};
        d.Dimension        = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        d.Width            = m_displayWidth;
        d.Height           = m_displayHeight;
        d.DepthOrArraySize = 1;
        d.MipLevels        = 1;
        d.Format           = fmt;
        d.SampleDesc.Count = 1;
        d.Flags            = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        ThrowIfFailed(device->CreateCommittedResource(
            &nv_helpers_dx12::kDefaultHeapProps, D3D12_HEAP_FLAG_NONE,
            &d, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
            IID_PPV_ARGS(&res)));
        res->SetName(name);
    };

    // Inputs (at render resolution)
    createRenderTex(m_input,            DXGI_FORMAT_R16G16B16A16_FLOAT, L"DLSS_Input");
    createRenderTex(m_depth,            DXGI_FORMAT_R32_FLOAT,          L"DLSS_Depth");
    createRenderTex(m_mvec,             DXGI_FORMAT_R16G16_FLOAT,       L"DLSS_MVec");
    createRenderTex(m_normals,          DXGI_FORMAT_R16G16B16A16_FLOAT, L"DLSS_Normals");
    createRenderTex(m_diffuseAlbedo,    DXGI_FORMAT_R16G16B16A16_FLOAT, L"DLSS_DiffuseAlbedo");
    createRenderTex(m_specAlbedo,       DXGI_FORMAT_R16G16B16A16_FLOAT, L"DLSS_SpecAlbedo");
    createRenderTex(m_roughness,        DXGI_FORMAT_R16_FLOAT,          L"DLSS_Roughness");
    createRenderTex(m_specMvec,         DXGI_FORMAT_R16G16_FLOAT,       L"DLSS_SpecMVec");
    createRenderTex(m_specHitDist,      DXGI_FORMAT_R16_FLOAT,          L"DLSS_HitDist");
    createRenderTex(m_transparency,     DXGI_FORMAT_R16G16B16A16_FLOAT, L"DLSS_Trans");
    createRenderTex(m_colorBeforeTrans, DXGI_FORMAT_R16G16B16A16_FLOAT, L"DLSS_ColorPreTrans");
    createRenderTex(m_biasHint,         DXGI_FORMAT_R8_UNORM,           L"DLSS_BiasHint");

    //A signed, render-resolution mask filled without a shader dispatch.
    createRenderTex(m_responsivityMask, DXGI_FORMAT_R16_FLOAT, L"DLSS_ResponsivityMask");
    if (!m_responsivityGpuHeap) {
        D3D12_DESCRIPTOR_HEAP_DESC heapDesc = {};
        heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        heapDesc.NumDescriptors = 1;
        heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        ThrowIfFailed(device->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&m_responsivityGpuHeap)));
        heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
        ThrowIfFailed(device->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&m_responsivityCpuHeap)));
    }
    D3D12_UNORDERED_ACCESS_VIEW_DESC maskUav = {};
    maskUav.Format = DXGI_FORMAT_R16_FLOAT;
    maskUav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    device->CreateUnorderedAccessView(m_responsivityMask.Get(), nullptr, &maskUav,
        m_responsivityGpuHeap->GetCPUDescriptorHandleForHeapStart());
    device->CreateUnorderedAccessView(m_responsivityMask.Get(), nullptr, &maskUav,
        m_responsivityCpuHeap->GetCPUDescriptorHandleForHeapStart());

    // Output (at display resolution — DLSS upscales to this)
    createDisplayTex(m_output,          DXGI_FORMAT_R16G16B16A16_FLOAT, L"DLSS_Output");

    // Register states
    ID3D12Resource* allRes[] = {
        m_depth.Get(), m_mvec.Get(), m_normals.Get(), m_diffuseAlbedo.Get(),
        m_output.Get(), m_specAlbedo.Get(), m_roughness.Get(), m_specMvec.Get(),
        m_specHitDist.Get(), m_transparency.Get(), m_colorBeforeTrans.Get(),
        m_biasHint.Get(), m_responsivityMask.Get()
    };
    for (auto* r : allRes)
        m_state.SetInitialState(r, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    m_activeMode = mode;
}

// ─────────────────────────────────────────────────────────────────
void DLSSManager::CreateResources(ID3D12Device* device, UINT displayWidth, UINT displayHeight) {
    m_displayWidth  = displayWidth;
    m_displayHeight = displayHeight;
    ComputeRenderResolution();

    std::wcout << L"[DLSS] CreateResources: display=" << m_displayWidth << L"x" << m_displayHeight
               << L" render=" << m_renderWidth << L"x" << m_renderHeight
               << L" mode=" << (int)mode << std::endl;

    CreateInputTextures(device);
}

// ─────────────────────────────────────────────────────────────────
bool DLSSManager::UpdateMode(ID3D12Device* device) {
    if (mode == m_activeMode) return false;

    ComputeRenderResolution();

    std::wcout << L"[DLSS] Mode change: " << (int)m_activeMode << L" -> " << (int)mode
               << L" render=" << m_renderWidth << L"x" << m_renderHeight << std::endl;

    CreateInputTextures(device);
    m_forceReset = true;  // tell DLSS to discard temporal history
    return true;
}

// ─────────────────────────────────────────────────────────────────
// A preset letter is handed to NGX unvalidated — there is no API to ask which
// letters the installed nvngx_dlssd.dll actually implements, so an unsupported
// one only surfaces as a failed SetOptions/Evaluate. Without this, the failure
// repeats every frame and DLSS stays dead for the rest of the session, so fall
// back to the last letter that completed a frame and let the UI show the snap-back.
void DLSSManager::RevertPresetIfPending(const wchar_t* stage) {
    if (!m_presetChangePending) return;

    std::wcout << L"[DLSS-RR] preset " << (int)m_activePresets[kPresetDLAA]
               << L" rejected by " << stage << L" — reverting to preset "
               << (int)m_lastGoodPresets[kPresetDLAA] << std::endl;

    std::memcpy(rrPresets,       m_lastGoodPresets, sizeof(rrPresets));
    std::memcpy(m_activePresets, m_lastGoodPresets, sizeof(m_activePresets));
    m_presetChangePending = false;
    m_forceReset = true;
}

// ─────────────────────────────────────────────────────────────────
void DLSSManager::Evaluate(
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
    float fovDegrees, float nearPlane, float farPlane)
{
    m_lastEvaluationSucceeded = false;
    m_lastEvaluationReset = true;
    if (!cmdList || !m_output || !m_depth || !m_mvec || !m_normals ||
        !m_diffuseAlbedo || !m_specAlbedo || !m_roughness || !m_specHitDist || !m_biasHint ||
        !m_responsivityMask || !m_responsivityGpuHeap || !m_responsivityCpuHeap)
        return;

    // A preset swap rebuilds DLSS-RR around a different model. Checked before the
    // constants block below so the reset flag it raises lands on THIS frame — the
    // frames the old model accumulated are not valid history for the new one.
    if (std::memcmp(m_activePresets, rrPresets, sizeof(rrPresets)) != 0) {
        std::memcpy(m_activePresets, rrPresets, sizeof(rrPresets));
        m_forceReset = true;
        m_presetChangePending = true;
    }

    constexpr D3D12_RESOURCE_STATES stateUAV = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    constexpr D3D12_RESOURCE_STATES stateSRV =
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE |
        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;

    const float responsivity = std::clamp(rrResponsivity, -1.0f, 1.0f);
    const bool useResponsivityMask = responsivity != 0.0f;
    if (useResponsivityMask) {
        const float clearColor[4] = { responsivity, 0.0f, 0.0f, 0.0f };
        //The caller already rebinds its heaps after Evaluate for Streamline.
        ID3D12DescriptorHeap* heaps[] = { m_responsivityGpuHeap.Get() };
        cmdList->SetDescriptorHeaps(1, heaps);
        cmdList->ClearUnorderedAccessViewFloat(
            m_responsivityGpuHeap->GetGPUDescriptorHandleForHeapStart(),
            m_responsivityCpuHeap->GetCPUDescriptorHandleForHeapStart(),
            m_responsivityMask.Get(), clearColor, 0, nullptr);
        auto ready = CD3DX12_RESOURCE_BARRIER::UAV(m_responsivityMask.Get());
        cmdList->ResourceBarrier(1, &ready);
    }

    // Transition inputs UAV → SRV
    ID3D12Resource* dlssInputs[] = {
        m_depth.Get(), m_mvec.Get(), m_normals.Get(), m_diffuseAlbedo.Get(),
        m_specAlbedo.Get(), m_roughness.Get(), m_specHitDist.Get(), m_input.Get(),
        m_biasHint.Get(), m_specMvec.Get(), useResponsivityMask ? m_responsivityMask.Get() : nullptr
    };
    std::vector<D3D12_RESOURCE_BARRIER> preB;
    for (auto* r : dlssInputs)
        if (r) preB.push_back(CD3DX12_RESOURCE_BARRIER::Transition(r, stateUAV, stateSRV));
    cmdList->ResourceBarrier((UINT)preB.size(), preB.data());

    // Everything below here can bail early — a preset the runtime rejects, a failed
    // tag, a failed evaluate — so the SRV -> UAV restore has to be unconditional.
    // Returning with the guides still in SRV makes the NEXT frame's UAV -> SRV
    // barrier a before-state mismatch, which the debug layer raises as an ERROR;
    // break-on-error then takes the process down a frame away from the real cause.
    // Built from the same null filter as preB so the two lists always pair up.
    struct GuideStateRestore {
        ID3D12GraphicsCommandList* cmdList;
        std::vector<D3D12_RESOURCE_BARRIER> barriers;
        ~GuideStateRestore() {
            if (!barriers.empty())
                cmdList->ResourceBarrier((UINT)barriers.size(), barriers.data());
        }
    } guideRestore{ cmdList, {} };
    for (auto* r : dlssInputs)
        if (r) guideRestore.barriers.push_back(
            CD3DX12_RESOURCE_BARRIER::Transition(r, stateSRV, stateUAV));

    // ── Build Streamline constants ────────────────────────────────
    // R32F reverse-Z, with a shared planet-scale range: a 10 km guide far plane
    // aliases finite horizon clouds with infinite sky. This projection avoids
    // forward-Z's far/(near-far) cancellation. Specular hit-distance storage is
    // separately capped to fp16; it does not define the camera's far plane.
    constexpr float kGuideDepthNear = DLSS_GUIDE_DEPTH_NEAR;
    constexpr float kGuideDepthFar  = DLSS_GUIDE_DEPTH_FAR;
    (void)nearPlane; (void)farPlane;

    // Use render resolution for aspect ratio in the projection. Swapped
    // near/far args = the standard reverse-Z construction (matches the
    // shader-side d = n(f-z)/((f-n)z) exactly).
    float renderAspect = (float)m_renderWidth / (float)m_renderHeight;
    XMMATRIX xmProj = XMMatrixPerspectiveFovRH(
        XMConvertToRadians(fovDegrees), renderAspect, kGuideDepthFar, kGuideDepthNear);
    XMMATRIX xmViewProj     = XMMatrixMultiply(viewMatrix, xmProj);
    // Use unjittered prev projection for clip-to-prev-clip (DLSS handles jitter
    // separately). Rebuilt with the SAME guide planes instead of the caller's
    // prevProjMatrix (which carries the 1e9 forward-Z camera projection) so
    // this frame's and last frame's clip spaces share one Z mapping; XY rows
    // don't depend on near/far, so reprojection is unaffected. (Assumes fov is
    // frame-coherent — a live fov edit mismatches for one frame, harmless.)
    (void)prevProjMatrix;
    XMMATRIX xmPrevProj = XMMatrixPerspectiveFovRH(
        XMConvertToRadians(fovDegrees), renderAspect, kGuideDepthFar, kGuideDepthNear);
    XMMATRIX xmPrevViewProj = XMMatrixMultiply(prevViewMatrix, xmPrevProj);

    auto XmToSl = [](const XMMATRIX& m) -> sl::float4x4 {
        XMFLOAT4X4 t; XMStoreFloat4x4(&t, m);
        sl::float4x4 o{}; std::memcpy(&o, &t, sizeof(o)); return o;
    };

    sl::Constants constants{};
    constants.cameraViewToClip  = XmToSl(xmProj);
    constants.clipToCameraView  = XmToSl(XMMatrixInverse(nullptr, xmProj));
    constants.clipToPrevClip    = XmToSl(XMMatrixMultiply(XMMatrixInverse(nullptr, xmViewProj), xmPrevViewProj));
    constants.prevClipToClip    = XmToSl(XMMatrixMultiply(XMMatrixInverse(nullptr, xmPrevViewProj), xmViewProj));
    constants.cameraFOV         = XMConvertToRadians(fovDegrees);
    constants.cameraAspectRatio = renderAspect;
    constants.cameraNear        = kGuideDepthNear;   //the guide range, see above
    constants.cameraFar         = kGuideDepthFar;
    //Scaled report only — the raygen already sampled at the unscaled offset. See
    //DLSSManager::jitterScale; {1,1} is the truthful value. Per-axis so a
    //Y-only sign flip is testable independently of X.
    constants.jitterOffset      = { -jitterX * jitterScale[0], -jitterY * jitterScale[1] };
    constants.mvecScale         = { 1.0f / (float)m_renderWidth, 1.0f / (float)m_renderHeight };
    //FLT_MIN sentinel, matching NVIDIA's RTXPT reference — the old -1.0f is a
    //legitimately occurring MV value (per docs the field is only consumed when
    //cameraMotionIncluded is false, but exact-match collisions cost nothing to
    //rule out).
    constants.motionVectorsInvalidValue = std::numeric_limits<float>::min();
    constants.cameraMotionIncluded      = sl::Boolean::eTrue;
    constants.depthInverted             = sl::Boolean::eTrue;   //reverse-Z guide depth
    constants.motionVectors3D           = sl::Boolean::eFalse;
    constants.motionVectorsJittered     = sl::Boolean::eFalse;
    //"Optional - specifies camera pinhole offset IF USED" (sl_consts.h) — this
    //camera is a centered pinhole (thin-lens DoF is not a pinhole shift), so
    //the correct value is ZERO. The previous {0.5, 0.5} declared a phantom
    //half-pixel pinhole displacement (pixel-space units, same as jitterOffset)
    //= a permanent sub-pixel misregistration of the whole reconstruction —
    //visible as stair-stepping / crawling on shallow edges that builds as
    //history accumulates against the misplaced reference.
    constants.cameraPinholeOffset       = { 0.0f, 0.0f };
    constants.reset = (jitterFrameIndex <= 1 || m_forceReset) ? sl::Boolean::eTrue : sl::Boolean::eFalse;
    m_lastEvaluationReset = constants.reset == sl::Boolean::eTrue;
    m_forceReset = false;

    {
        auto iv = XMMatrixInverse(nullptr, viewMatrix);
        XMFLOAT4X4 f; XMStoreFloat4x4(&f, iv);
        constants.cameraPos   = { f._41, f._42, f._43 };
        constants.cameraRight = { f._11, f._12, f._13 };
        constants.cameraUp    = { f._21, f._22, f._23 };
        constants.cameraFwd   = { -f._31, -f._32, -f._33 };
    }
    SL_CHECK(slSetConstants(constants, frameToken, viewport));

    // ── DLSS-RR options ──────────────────────────────────────────
    sl::DLSSDOptions options{};
    options.mode             = mode;
    options.outputWidth      = m_displayWidth;
    options.outputHeight     = m_displayHeight;
    options.colorBuffersHDR  = sl::Boolean::eTrue;
    //PACKED normal-roughness: roughness rides in the normals buffer's .w and
    //is tagged as kBufferTypeNormalRoughness. This is the convention every
    //shipping RR integration uses (Cyberpunk, RTXPT reference — which passes
    //a null standalone-roughness texture and tags only the packed buffer).
    //The eUnpacked path with a separate kBufferTypeRoughness tag is rarely
    //exercised in the wild and correlated with preset-F temporal instability
    //here — do not switch back without retesting F on grazing geometry.
    options.normalRoughnessMode = sl::DLSSDNormalRoughnessMode::ePacked;
    options.worldToCameraView   = XmToSl(viewMatrix);
    options.cameraViewToWorld   = XmToSl(XMMatrixInverse(nullptr, viewMatrix));

    // Model preset per quality mode — editor-driven, see DLSSManager.h. The plugin
    // only consults the slot matching options.mode, but the struct takes all six.
    options.dlaaPreset             = rrPresets[kPresetDLAA];
    options.qualityPreset          = rrPresets[kPresetQuality];
    options.balancedPreset         = rrPresets[kPresetBalanced];
    options.performancePreset      = rrPresets[kPresetPerformance];
    options.ultraPerformancePreset = rrPresets[kPresetUltraPerformance];
    options.ultraQualityPreset     = rrPresets[kPresetUltraQuality];
    if (sl::Result r = slDLSSDSetOptions(viewport, options); r != sl::Result::eOk) {
        std::wcout << L"[DLSS-RR] slDLSSDSetOptions failed: " << (int)r << std::endl;
        RevertPresetIfPending(L"slDLSSDSetOptions");
        return;  // guides restored by guideRestore
    }

    // ── Tag resources ────────────────────────────────────────────
    sl::Resource slDepth   (sl::ResourceType::eTex2d, m_depth.Get(),         (uint32_t)stateSRV);
    sl::Resource slMVec    (sl::ResourceType::eTex2d, m_mvec.Get(),          (uint32_t)stateSRV);
    sl::Resource slNormals (sl::ResourceType::eTex2d, m_normals.Get(),       (uint32_t)stateSRV);
    sl::Resource slAlbedo  (sl::ResourceType::eTex2d, m_diffuseAlbedo.Get(), (uint32_t)stateSRV);
    sl::Resource slSpecAlb (sl::ResourceType::eTex2d, m_specAlbedo.Get(),    (uint32_t)stateSRV);
    sl::Resource slRough   (sl::ResourceType::eTex2d, m_roughness.Get(),     (uint32_t)stateSRV);
    sl::Resource slSpecHit (sl::ResourceType::eTex2d, m_specHitDist.Get(),   (uint32_t)stateSRV);
    sl::Resource slInput   (sl::ResourceType::eTex2d, m_input.Get(),         (uint32_t)stateSRV);
    sl::Resource slSpecMV  (sl::ResourceType::eTex2d, m_specMvec.Get(),     (uint32_t)stateSRV);
    sl::Resource slOutput  (sl::ResourceType::eTex2d, m_output.Get(),        (uint32_t)stateUAV);

    sl::Resource slBiasHint(sl::ResourceType::eTex2d, m_biasHint.Get(), (uint32_t)stateSRV);
    sl::Resource slResponsivity(sl::ResourceType::eTex2d, m_responsivityMask.Get(), (uint32_t)stateSRV);

    // Inputs use render extent, output uses display extent
    sl::Extent renderExtent { 0, 0, m_renderWidth,  m_renderHeight  };
    sl::Extent displayExtent{ 0, 0, m_displayWidth, m_displayHeight };
    auto life = sl::ResourceLifecycle::eValidUntilEvaluate;

    std::vector<sl::ResourceTag> tags = {
        //reverse-Z device depth (see the constants block) — kBufferTypeDepth,
        //the primary convention ("must be suitable for clipToPrevClip"), not
        //the linear tag: linear metres proved preset-sensitive (striping).
        { &slDepth,   sl::kBufferTypeDepth,                life, &renderExtent  },
        { &slMVec,    sl::kBufferTypeMotionVectors,        life, &renderExtent  },
        //packed mode: normals .w carries roughness, tagged as the combined
        //buffer; the standalone roughness tag is retired (null resource
        //clears any stale tag — the texture itself is still written for the
        //layer inspector).
        { &slNormals, sl::kBufferTypeNormalRoughness,      life, &renderExtent  },
        { nullptr,    sl::kBufferTypeRoughness,            life, &renderExtent  },
        { &slAlbedo,  sl::kBufferTypeAlbedo,               life, &renderExtent  },
        { &slSpecAlb, sl::kBufferTypeSpecularAlbedo,       life, &renderExtent  },
        //SPECULAR HIT DISTANCE IS DELIBERATELY UNTAGGED (null resource keeps
        //any stale tag cleared). Per NVIDIA's RTXPT reference the spec-MV and
        //spec-hit-dist guides are MUTUALLY EXCLUSIVE — their wrapper hard-
        //errors when both are provided — and their call site ships spec MVs
        //with the hitT path disabled ("it's buggy"). We were tagging BOTH,
        //an unsupported combination no shipping title runs. We keep the
        //(deterministic, probe-based) spec MVs; m_specHitDist is still
        //written for the guide-inspector view, just never handed to RR.
        { nullptr,    sl::kBufferTypeSpecularHitDistance,   life, &renderExtent  },
        { &slInput,   sl::kBufferTypeScalingInputColor,    life, &renderExtent  },
        //untagSpecMV: spec MV is the one optional guide — null resource drops
        //the tag (and clears a stale one) so RR falls back to internal
        //specular tracking. See DLSSManager.h.
        { untagSpecMV ? nullptr : &slSpecMV,
                      sl::kBufferTypeSpecularMotionVectors, life, &renderExtent },
        //Binary visible-emitter mask: 1 favors current color on emitters, 0 elsewhere.
        { &slBiasHint, sl::kBufferTypeBiasCurrentColorHint, life, &renderExtent },
        //Zero disables the override and clears any tag from the previous frame.
        { useResponsivityMask ? &slResponsivity : nullptr,
                      sl::kBufferTypeResponsivityMask, life, &renderExtent },
        { &slOutput,  sl::kBufferTypeScalingOutputColor,   life, &displayExtent },
    };
    SL_CHECK(slSetTagForFrame(frameToken, viewport, tags.data(), (uint32_t)tags.size(), cmdList));

    // ── Evaluate ─────────────────────────────────────────────────
    ComPtr<ID3D12InfoQueue> infoQueue;
    if (SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&infoQueue))))
        infoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_ERROR, FALSE);

    //Suppressing break-on-error above also silences any REAL validation
    //violation inside the evaluate window, so capture the messages the call
    //generates and surface them (editor DLSS Inputs panel + console). The
    //per-frame DumpNewMessages clears the queue at present, so indices here
    //are frame-relative and stable.
    const UINT64 dxMsgsBefore = infoQueue ? infoQueue->GetNumStoredMessages() : 0;

    const sl::BaseStructure* evalInputs[] = { &viewport, &options };
    sl::Result evalResult = slEvaluateFeature(
        sl::kFeatureDLSS_RR, frameToken, evalInputs, _countof(evalInputs), cmdList);

    if (infoQueue) {
        infoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_ERROR, TRUE);

        const UINT64 dxMsgsAfter = infoQueue->GetNumStoredMessages();
        for (UINT64 i = dxMsgsBefore; i < dxMsgsAfter; ++i) {
            SIZE_T sz = 0;
            infoQueue->GetMessage(i, nullptr, &sz);
            if (!sz) continue;
            std::vector<uint8_t> blob(sz);
            auto* msg = reinterpret_cast<D3D12_MESSAGE*>(blob.data());
            if (FAILED(infoQueue->GetMessage(i, msg, &sz))) continue;
            //INFO spam (state-decay notices etc.) is noise; keep warnings up
            if (msg->Severity > D3D12_MESSAGE_SEVERITY_WARNING) continue;
            ++evalDxMessageTotal;
            evalDxMessages.emplace_back(msg->pDescription ? msg->pDescription : "<no description>");
            if (evalDxMessages.size() > 8)
                evalDxMessages.erase(evalDxMessages.begin());
            std::wcout << L"[DLSS-EVAL DX] " << (msg->pDescription ? msg->pDescription : "") << std::endl;
        }
    }

    if (evalResult != sl::Result::eOk) {
        std::wcout << L"[DLSS-RR] slEvaluateFeature failed: " << (int)evalResult << std::endl;
        RevertPresetIfPending(L"slEvaluateFeature");
        return;  // guides restored by guideRestore
    }

    // Survived a full frame, so this preset is the one to fall back to next time.
    if (m_presetChangePending) {
        std::memcpy(m_lastGoodPresets, m_activePresets, sizeof(m_lastGoodPresets));
        m_presetChangePending = false;
    }

    m_dlssPrevView = viewMatrix;
    m_dlssPrevProj = xmProj;
    m_lastEvaluationSucceeded = true;
}
