//====================================
//DLSS MANAGER
//====================================

#include "../stdafx.h"
#include "DLSSManager.h"
#include "../DXRHelper.h"
#include "../glm/gtc/type_ptr.hpp"
#include "../manipulator.h"

#include <sl.h>
#include <sl_consts.h>
#include <sl_helpers.h>
#include <sl_dlss.h>
#include "sl_dlss_d.h"

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

    // Output (at display resolution — DLSS upscales to this)
    createDisplayTex(m_output,          DXGI_FORMAT_R16G16B16A16_FLOAT, L"DLSS_Output");

    // Register states
    ID3D12Resource* allRes[] = {
        m_depth.Get(), m_mvec.Get(), m_normals.Get(), m_diffuseAlbedo.Get(),
        m_output.Get(), m_specAlbedo.Get(), m_roughness.Get(), m_specMvec.Get(),
        m_specHitDist.Get(), m_transparency.Get(), m_colorBeforeTrans.Get(),
        m_biasHint.Get()
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
    if (!cmdList || !m_output || !m_depth || !m_mvec || !m_normals ||
        !m_diffuseAlbedo || !m_specAlbedo || !m_roughness || !m_specHitDist)
        return;

    constexpr D3D12_RESOURCE_STATES stateUAV = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    constexpr D3D12_RESOURCE_STATES stateSRV =
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE |
        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;

    // Transition inputs UAV → SRV
    ID3D12Resource* dlssInputs[] = {
        m_depth.Get(), m_mvec.Get(), m_normals.Get(), m_diffuseAlbedo.Get(),
        m_specAlbedo.Get(), m_roughness.Get(), m_specHitDist.Get(), m_input.Get(),
        m_biasHint.Get(), m_specMvec.Get()
    };
    std::vector<D3D12_RESOURCE_BARRIER> preB;
    for (auto* r : dlssInputs)
        if (r) preB.push_back(CD3DX12_RESOURCE_BARRIER::Transition(r, stateUAV, stateSRV));
    cmdList->ResourceBarrier((UINT)preB.size(), preB.data());

    // ── Build Streamline constants ────────────────────────────────
    // Use render resolution for aspect ratio in the projection
    float renderAspect = (float)m_renderWidth / (float)m_renderHeight;
    XMMATRIX xmProj = XMMatrixPerspectiveFovRH(
        XMConvertToRadians(fovDegrees), renderAspect, nearPlane, farPlane);
    XMMATRIX xmViewProj     = XMMatrixMultiply(viewMatrix, xmProj);
    // Use unjittered prev projection for clip-to-prev-clip (DLSS handles jitter separately)
    XMMATRIX xmPrevViewProj = XMMatrixMultiply(prevViewMatrix, prevProjMatrix);

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
    constants.cameraNear        = nearPlane;
    constants.cameraFar         = farPlane;
    constants.jitterOffset      = { -jitterX, -jitterY };
    constants.mvecScale         = { 1.0f / (float)m_renderWidth, 1.0f / (float)m_renderHeight };
    constants.motionVectorsInvalidValue = -1.0f;
    constants.cameraMotionIncluded      = sl::Boolean::eTrue;
    constants.depthInverted             = sl::Boolean::eFalse;
    constants.motionVectors3D           = sl::Boolean::eFalse;
    constants.motionVectorsJittered     = sl::Boolean::eFalse;
    constants.cameraPinholeOffset       = { 0.5f, 0.5f };
    constants.reset = (jitterFrameIndex <= 1 || m_forceReset) ? sl::Boolean::eTrue : sl::Boolean::eFalse;
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
    options.normalRoughnessMode = sl::DLSSDNormalRoughnessMode::eUnpacked;
    options.worldToCameraView   = XmToSl(viewMatrix);
    options.cameraViewToWorld   = XmToSl(XMMatrixInverse(nullptr, viewMatrix));

    sl::DLSSDPreset preset = sl::DLSSDPreset::ePresetE;
    options.dlaaPreset = options.qualityPreset = options.balancedPreset =
        options.performancePreset = options.ultraPerformancePreset =
        options.ultraQualityPreset = preset;
    SL_CHECK(slDLSSDSetOptions(viewport, options));

    // ── Tag resources ────────────────────────────────────────────
    sl::Resource slDepth   (sl::ResourceType::eTex2d, m_depth.Get(),         (uint32_t)stateSRV);
    sl::Resource slMVec    (sl::ResourceType::eTex2d, m_mvec.Get(),          (uint32_t)stateSRV);
    sl::Resource slNormals (sl::ResourceType::eTex2d, m_normals.Get(),       (uint32_t)stateSRV);
    sl::Resource slAlbedo  (sl::ResourceType::eTex2d, m_diffuseAlbedo.Get(), (uint32_t)stateSRV);
    sl::Resource slSpecAlb (sl::ResourceType::eTex2d, m_specAlbedo.Get(),    (uint32_t)stateSRV);
    sl::Resource slRough   (sl::ResourceType::eTex2d, m_roughness.Get(),     (uint32_t)stateSRV);
    sl::Resource slSpecHit (sl::ResourceType::eTex2d, m_specHitDist.Get(),   (uint32_t)stateSRV);
    sl::Resource slInput   (sl::ResourceType::eTex2d, m_input.Get(),         (uint32_t)stateSRV);
    sl::Resource slBias    (sl::ResourceType::eTex2d, m_biasHint.Get(),      (uint32_t)stateSRV);
    sl::Resource slSpecMV  (sl::ResourceType::eTex2d, m_specMvec.Get(),     (uint32_t)stateSRV);
    sl::Resource slOutput  (sl::ResourceType::eTex2d, m_output.Get(),        (uint32_t)stateUAV);

    // Inputs use render extent, output uses display extent
    sl::Extent renderExtent { 0, 0, m_renderWidth,  m_renderHeight  };
    sl::Extent displayExtent{ 0, 0, m_displayWidth, m_displayHeight };
    auto life = sl::ResourceLifecycle::eValidUntilEvaluate;

    std::vector<sl::ResourceTag> tags = {
        { &slDepth,   sl::kBufferTypeLinearDepth,          life, &renderExtent  },
        { &slMVec,    sl::kBufferTypeMotionVectors,        life, &renderExtent  },
        { &slNormals, sl::kBufferTypeNormals,              life, &renderExtent  },
        { &slRough,   sl::kBufferTypeRoughness,            life, &renderExtent  },
        { &slAlbedo,  sl::kBufferTypeAlbedo,               life, &renderExtent  },
        { &slSpecAlb, sl::kBufferTypeSpecularAlbedo,       life, &renderExtent  },
        { &slSpecHit, sl::kBufferTypeSpecularHitDistance,   life, &renderExtent  },
        { &slInput,   sl::kBufferTypeScalingInputColor,    life, &renderExtent  },
        { &slBias,    sl::kBufferTypeBiasCurrentColorHint, life, &renderExtent  },
        { &slSpecMV,  sl::kBufferTypeSpecularMotionVectors, life, &renderExtent },
        { &slOutput,  sl::kBufferTypeScalingOutputColor,   life, &displayExtent },
    };
    SL_CHECK(slSetTagForFrame(frameToken, viewport, tags.data(), (uint32_t)tags.size(), cmdList));

    // ── Evaluate ─────────────────────────────────────────────────
    ComPtr<ID3D12InfoQueue> infoQueue;
    if (SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&infoQueue))))
        infoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_ERROR, FALSE);

    const sl::BaseStructure* evalInputs[] = { &viewport, &options };
    sl::Result evalResult = slEvaluateFeature(
        sl::kFeatureDLSS_RR, frameToken, evalInputs, _countof(evalInputs), cmdList);

    if (infoQueue)
        infoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_ERROR, TRUE);

    if (evalResult != sl::Result::eOk) {
        std::wcout << L"[DLSS-RR] slEvaluateFeature failed: " << (int)evalResult << std::endl;
        return;
    }

    // Transition inputs back SRV → UAV
    std::vector<D3D12_RESOURCE_BARRIER> postB;
    for (auto* r : dlssInputs)
        postB.push_back(CD3DX12_RESOURCE_BARRIER::Transition(r, stateSRV, stateUAV));
    cmdList->ResourceBarrier((UINT)postB.size(), postB.data());

    m_dlssPrevView = viewMatrix;
    m_dlssPrevProj = xmProj;
}
