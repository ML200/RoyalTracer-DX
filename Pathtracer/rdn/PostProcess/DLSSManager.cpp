// ═══════════════════════════════════════════════════════════════════
// PostProcess/DLSSManager.cpp — DLSS-RR resource creation + eval
// ═══════════════════════════════════════════════════════════════════

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
void DLSSManager::CreateResources(ID3D12Device* device, UINT width, UINT height) {
    auto createTex = [&](ComPtr<ID3D12Resource>& res, DXGI_FORMAT fmt, const wchar_t* name) {
        D3D12_RESOURCE_DESC d = {};
        d.Dimension        = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        d.Width            = width;
        d.Height           = height;
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

    createTex(m_input,            DXGI_FORMAT_R16G16B16A16_FLOAT, L"DLSS_Input");
    createTex(m_depth,            DXGI_FORMAT_R32_FLOAT,          L"DLSS_Depth");
    createTex(m_mvec,             DXGI_FORMAT_R16G16_FLOAT,       L"DLSS_MVec");
    createTex(m_normals,          DXGI_FORMAT_R16G16B16A16_FLOAT, L"DLSS_Normals");
    createTex(m_diffuseAlbedo,    DXGI_FORMAT_R16G16B16A16_FLOAT, L"DLSS_DiffuseAlbedo");
    createTex(m_output,           DXGI_FORMAT_R16G16B16A16_FLOAT, L"DLSS_Output");
    createTex(m_specAlbedo,       DXGI_FORMAT_R16G16B16A16_FLOAT, L"DLSS_SpecAlbedo");
    createTex(m_roughness,        DXGI_FORMAT_R16_FLOAT,          L"DLSS_Roughness");
    createTex(m_specMvec,         DXGI_FORMAT_R16G16_FLOAT,       L"DLSS_SpecMVec");
    createTex(m_specHitDist,      DXGI_FORMAT_R16_FLOAT,          L"DLSS_HitDist");
    createTex(m_transparency,     DXGI_FORMAT_R16G16B16A16_FLOAT, L"DLSS_Trans");
    createTex(m_colorBeforeTrans, DXGI_FORMAT_R16G16B16A16_FLOAT, L"DLSS_ColorPreTrans");

    // Register initial states for the tracker
    ID3D12Resource* allRes[] = {
        m_depth.Get(), m_mvec.Get(), m_normals.Get(), m_diffuseAlbedo.Get(),
        m_output.Get(), m_specAlbedo.Get(), m_roughness.Get(), m_specMvec.Get(),
        m_specHitDist.Get(), m_transparency.Get(), m_colorBeforeTrans.Get()
    };
    for (auto* r : allRes)
        m_state.SetInitialState(r, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
}

// ─────────────────────────────────────────────────────────────────
void DLSSManager::Evaluate(
    ID3D12GraphicsCommandList* cmdList,
    ID3D12Device* device,
    sl::FrameToken& frameToken,
    sl::ViewportHandle viewport,
    UINT width, UINT height,
    float aspectRatio,
    const XMMATRIX& viewMatrix,
    const XMMATRIX& prevViewMatrix,
    const XMMATRIX& prevProjMatrix,
    float jitterX, float jitterY,
    uint32_t jitterFrameIndex)
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
        m_specAlbedo.Get(), m_roughness.Get(), m_specHitDist.Get(), m_input.Get()
    };
    std::vector<D3D12_RESOURCE_BARRIER> preB;
    for (auto* r : dlssInputs)
        if (r) preB.push_back(CD3DX12_RESOURCE_BARRIER::Transition(r, stateUAV, stateSRV));
    cmdList->ResourceBarrier((UINT)preB.size(), preB.data());

    // ── Build Streamline constants ───────────────────────────────
    XMMATRIX xmProj = XMMatrixPerspectiveFovRH(
        XMConvertToRadians(fovDegrees), aspectRatio, 0.00001f, 10000.0f);
    XMMATRIX xmViewProj     = XMMatrixMultiply(viewMatrix, xmProj);
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
    constants.cameraAspectRatio = aspectRatio;
    constants.cameraNear        = 0.00001f;
    constants.cameraFar         = 10000.0f;
    constants.jitterOffset      = { -jitterX, -jitterY };
    constants.mvecScale         = { 1.0f / (float)width, 1.0f / (float)height };
    constants.motionVectorsInvalidValue = -1.0f;
    constants.cameraMotionIncluded      = sl::Boolean::eTrue;
    constants.depthInverted             = sl::Boolean::eFalse;
    constants.motionVectors3D           = sl::Boolean::eFalse;
    constants.motionVectorsJittered     = sl::Boolean::eFalse;
    constants.reset = (jitterFrameIndex <= 1) ? sl::Boolean::eTrue : sl::Boolean::eFalse;

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
    options.outputWidth      = width;
    options.outputHeight     = height;
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
    sl::Resource slOutput  (sl::ResourceType::eTex2d, m_output.Get(),        (uint32_t)stateUAV);

    sl::Extent extent{ 0, 0, width, height };
    auto life = sl::ResourceLifecycle::eValidUntilEvaluate;

    std::vector<sl::ResourceTag> tags = {
        { &slDepth,   sl::kBufferTypeLinearDepth,          life, &extent },
        { &slMVec,    sl::kBufferTypeMotionVectors,        life, &extent },
        { &slNormals, sl::kBufferTypeNormals,              life, &extent },
        { &slRough,   sl::kBufferTypeRoughness,            life, &extent },
        { &slAlbedo,  sl::kBufferTypeAlbedo,               life, &extent },
        { &slSpecAlb, sl::kBufferTypeSpecularAlbedo,       life, &extent },
        { &slSpecHit, sl::kBufferTypeSpecularHitDistance,   life, &extent },
        { &slInput,   sl::kBufferTypeScalingInputColor,    life, &extent },
        { &slOutput,  sl::kBufferTypeScalingOutputColor,   life, &extent },
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
