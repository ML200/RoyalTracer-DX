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
#include <limits>

#undef SL_CHECK
#define SL_CHECK(x)                                                                                                    \
    do {                                                                                                               \
        sl::Result r = (x);                                                                                            \
        if (r != sl::Result::eOk) {                                                                                    \
            std::wcout << L"[SL] " << L#x << L" failed: " << (int)r << std::endl;                                      \
            return;                                                                                                    \
        }                                                                                                              \
    } while (0)

void DLSSManager::ComputeRenderResolution() {
    float scale = 1.0f;

    if (mode == sl::DLSSMode::eOff || mode == sl::DLSSMode::eDLAA) {
        scale = 1.0f;
    } else if (mode == sl::DLSSMode::eMaxQuality) {
        scale = 1.0f / 1.5f;
    } else if (mode == sl::DLSSMode::eBalanced) {
        scale = 1.0f / 1.7f;
    } else {
        scale = 1.0f;
    }

    m_renderWidth = (std::max)(2u, (UINT)(m_displayWidth * scale) & ~1u);
    m_renderHeight = (std::max)(2u, (UINT)(m_displayHeight * scale) & ~1u);
}

void DLSSManager::CreateInputTextures(ID3D12Device* device) {
    auto createRenderTex = [&](ComPtr<ID3D12Resource>& res, DXGI_FORMAT fmt, const wchar_t* name) {
        D3D12_RESOURCE_DESC d = {};
        d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        d.Width = m_renderWidth;
        d.Height = m_renderHeight;
        d.DepthOrArraySize = 1;
        d.MipLevels = 1;
        d.Format = fmt;
        d.SampleDesc.Count = 1;
        d.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        ThrowIfFailed(device->CreateCommittedResource(&nv_helpers_dx12::kDefaultHeapProps, D3D12_HEAP_FLAG_NONE, &d,
                                                      D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                                                      IID_PPV_ARGS(&res)));
        res->SetName(name);
    };

    auto createDisplayTex = [&](ComPtr<ID3D12Resource>& res, DXGI_FORMAT fmt, const wchar_t* name) {
        D3D12_RESOURCE_DESC d = {};
        d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        d.Width = m_displayWidth;
        d.Height = m_displayHeight;
        d.DepthOrArraySize = 1;
        d.MipLevels = 1;
        d.Format = fmt;
        d.SampleDesc.Count = 1;
        d.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        ThrowIfFailed(device->CreateCommittedResource(&nv_helpers_dx12::kDefaultHeapProps, D3D12_HEAP_FLAG_NONE, &d,
                                                      D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                                                      IID_PPV_ARGS(&res)));
        res->SetName(name);
    };

    createRenderTex(m_input, DXGI_FORMAT_R16G16B16A16_FLOAT, L"DLSS_Input");
    createRenderTex(m_depth, DXGI_FORMAT_R32_FLOAT, L"DLSS_Depth");
    createRenderTex(m_mvec, DXGI_FORMAT_R16G16_FLOAT, L"DLSS_MVec");
    createRenderTex(m_normals, DXGI_FORMAT_R16G16B16A16_FLOAT, L"DLSS_Normals");
    createRenderTex(m_diffuseAlbedo, DXGI_FORMAT_R16G16B16A16_FLOAT, L"DLSS_DiffuseAlbedo");
    createRenderTex(m_specAlbedo, DXGI_FORMAT_R16G16B16A16_FLOAT, L"DLSS_SpecAlbedo");
    createRenderTex(m_roughness, DXGI_FORMAT_R16_FLOAT, L"DLSS_Roughness");
    createRenderTex(m_specMvec, DXGI_FORMAT_R16G16_FLOAT, L"DLSS_SpecMVec");
    createRenderTex(m_specHitDist, DXGI_FORMAT_R16_FLOAT, L"DLSS_HitDist");
    createRenderTex(m_transparency, DXGI_FORMAT_R16G16B16A16_FLOAT, L"DLSS_Trans");
    createRenderTex(m_colorBeforeTrans, DXGI_FORMAT_R16G16B16A16_FLOAT, L"DLSS_ColorPreTrans");
    createRenderTex(m_biasHint, DXGI_FORMAT_R8_UNORM, L"DLSS_BiasHint");

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

    createDisplayTex(m_output, DXGI_FORMAT_R16G16B16A16_FLOAT, L"DLSS_Output");

    m_activeMode = mode;
}

void DLSSManager::CreateResources(ID3D12Device* device, UINT displayWidth, UINT displayHeight) {
    m_displayWidth = displayWidth;
    m_displayHeight = displayHeight;
    ComputeRenderResolution();

    std::wcout << L"[DLSS] CreateResources: display=" << m_displayWidth << L"x" << m_displayHeight << L" render="
               << m_renderWidth << L"x" << m_renderHeight << L" mode=" << (int)mode << std::endl;

    CreateInputTextures(device);
}

bool DLSSManager::UpdateMode(ID3D12Device* device) {
    if (mode == m_activeMode)
        return false;

    ComputeRenderResolution();

    std::wcout << L"[DLSS] Mode change: " << (int)m_activeMode << L" -> " << (int)mode << L" render=" << m_renderWidth
               << L"x" << m_renderHeight << std::endl;

    CreateInputTextures(device);
    m_forceReset = true;
    return true;
}

void DLSSManager::RevertPresetIfPending(const wchar_t* stage) {
    if (!m_presetChangePending)
        return;

    std::wcout << L"[DLSS-RR] preset " << (int)m_activePresets[kPresetDLAA] << L" rejected by " << stage
               << L" — reverting to preset " << (int)m_lastGoodPresets[kPresetDLAA] << std::endl;

    std::memcpy(rrPresets, m_lastGoodPresets, sizeof(rrPresets));
    std::memcpy(m_activePresets, m_lastGoodPresets, sizeof(m_activePresets));
    m_presetChangePending = false;
    m_forceReset = true;
}

void DLSSManager::Evaluate(ID3D12GraphicsCommandList* cmdList, ID3D12Device* device, sl::FrameToken& frameToken,
                           sl::ViewportHandle viewport, float aspectRatio, const XMMATRIX& viewMatrix,
                           const XMMATRIX& prevViewMatrix, const XMMATRIX& prevProjMatrix, float jitterX, float jitterY,
                           uint32_t jitterFrameIndex, float fovDegrees, float nearPlane, float farPlane) {
    m_lastEvaluationSucceeded = false;
    m_lastEvaluationReset = true;
    if (!cmdList || !m_output || !m_depth || !m_mvec || !m_normals || !m_diffuseAlbedo || !m_specAlbedo ||
        !m_roughness || !m_specHitDist || !m_biasHint || !m_responsivityMask || !m_responsivityGpuHeap ||
        !m_responsivityCpuHeap)
        return;

    if (std::memcmp(m_activePresets, rrPresets, sizeof(rrPresets)) != 0) {
        std::memcpy(m_activePresets, rrPresets, sizeof(rrPresets));
        m_forceReset = true;
        m_presetChangePending = true;
    }

    constexpr D3D12_RESOURCE_STATES stateUAV = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    constexpr D3D12_RESOURCE_STATES stateSRV =
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;

    // DLSS consumes guides as SRVs and returns them to UAV state afterward.
    const float responsivity = std::clamp(rrResponsivity, -1.0f, 1.0f);
    const bool useResponsivityMask = responsivity != 0.0f;
    if (useResponsivityMask) {
        const float clearColor[4] = {responsivity, 0.0f, 0.0f, 0.0f};

        ID3D12DescriptorHeap* heaps[] = {m_responsivityGpuHeap.Get()};
        cmdList->SetDescriptorHeaps(1, heaps);
        cmdList->ClearUnorderedAccessViewFloat(m_responsivityGpuHeap->GetGPUDescriptorHandleForHeapStart(),
                                               m_responsivityCpuHeap->GetCPUDescriptorHandleForHeapStart(),
                                               m_responsivityMask.Get(), clearColor, 0, nullptr);
        auto ready = CD3DX12_RESOURCE_BARRIER::UAV(m_responsivityMask.Get());
        cmdList->ResourceBarrier(1, &ready);
    }

    ID3D12Resource* dlssInputs[] = {m_depth.Get(),
                                    m_mvec.Get(),
                                    m_normals.Get(),
                                    m_diffuseAlbedo.Get(),
                                    m_specAlbedo.Get(),
                                    m_roughness.Get(),
                                    m_specHitDist.Get(),
                                    m_input.Get(),
                                    m_biasHint.Get(),
                                    m_specMvec.Get(),
                                    useResponsivityMask ? m_responsivityMask.Get() : nullptr};
    std::vector<D3D12_RESOURCE_BARRIER> preB;
    for (auto* r : dlssInputs)
        if (r)
            preB.push_back(CD3DX12_RESOURCE_BARRIER::Transition(r, stateUAV, stateSRV));
    cmdList->ResourceBarrier((UINT)preB.size(), preB.data());

    struct GuideStateRestore {
        ID3D12GraphicsCommandList* cmdList;
        std::vector<D3D12_RESOURCE_BARRIER> barriers;
        ~GuideStateRestore() {
            if (!barriers.empty())
                cmdList->ResourceBarrier((UINT)barriers.size(), barriers.data());
        }
    } guideRestore{cmdList, {}};
    for (auto* r : dlssInputs)
        if (r)
            guideRestore.barriers.push_back(CD3DX12_RESOURCE_BARRIER::Transition(r, stateSRV, stateUAV));

    constexpr float kGuideDepthNear = DLSS_GUIDE_DEPTH_NEAR;
    constexpr float kGuideDepthFar = DLSS_GUIDE_DEPTH_FAR;
    (void)nearPlane;
    (void)farPlane;

    float renderAspect = (float)m_renderWidth / (float)m_renderHeight;
    XMMATRIX xmProj =
        XMMatrixPerspectiveFovRH(XMConvertToRadians(fovDegrees), renderAspect, kGuideDepthFar, kGuideDepthNear);
    XMMATRIX xmViewProj = XMMatrixMultiply(viewMatrix, xmProj);

    (void)prevProjMatrix;
    XMMATRIX xmPrevProj =
        XMMatrixPerspectiveFovRH(XMConvertToRadians(fovDegrees), renderAspect, kGuideDepthFar, kGuideDepthNear);
    XMMATRIX xmPrevViewProj = XMMatrixMultiply(prevViewMatrix, xmPrevProj);

    auto XmToSl = [](const XMMATRIX& m) -> sl::float4x4 {
        XMFLOAT4X4 t;
        XMStoreFloat4x4(&t, m);
        sl::float4x4 o{};
        std::memcpy(&o, &t, sizeof(o));
        return o;
    };

    sl::Constants constants{};
    constants.cameraViewToClip = XmToSl(xmProj);
    constants.clipToCameraView = XmToSl(XMMatrixInverse(nullptr, xmProj));
    constants.clipToPrevClip = XmToSl(XMMatrixMultiply(XMMatrixInverse(nullptr, xmViewProj), xmPrevViewProj));
    constants.prevClipToClip = XmToSl(XMMatrixMultiply(XMMatrixInverse(nullptr, xmPrevViewProj), xmViewProj));
    constants.cameraFOV = XMConvertToRadians(fovDegrees);
    constants.cameraAspectRatio = renderAspect;
    constants.cameraNear = kGuideDepthNear;
    constants.cameraFar = kGuideDepthFar;

    constants.jitterOffset = {-jitterX * jitterScale[0], -jitterY * jitterScale[1]};
    constants.mvecScale = {1.0f / (float)m_renderWidth, 1.0f / (float)m_renderHeight};

    constants.motionVectorsInvalidValue = std::numeric_limits<float>::min();
    constants.cameraMotionIncluded = sl::Boolean::eTrue;
    constants.depthInverted = sl::Boolean::eTrue;
    constants.motionVectors3D = sl::Boolean::eFalse;
    constants.motionVectorsJittered = sl::Boolean::eFalse;

    constants.cameraPinholeOffset = {0.0f, 0.0f};
    constants.reset = (jitterFrameIndex <= 1 || m_forceReset) ? sl::Boolean::eTrue : sl::Boolean::eFalse;
    m_lastEvaluationReset = constants.reset == sl::Boolean::eTrue;
    m_forceReset = false;

    {
        auto iv = XMMatrixInverse(nullptr, viewMatrix);
        XMFLOAT4X4 f;
        XMStoreFloat4x4(&f, iv);
        constants.cameraPos = {f._41, f._42, f._43};
        constants.cameraRight = {f._11, f._12, f._13};
        constants.cameraUp = {f._21, f._22, f._23};
        constants.cameraFwd = {-f._31, -f._32, -f._33};
    }
    SL_CHECK(slSetConstants(constants, frameToken, viewport));

    sl::DLSSDOptions options{};
    options.mode = mode;
    options.outputWidth = m_displayWidth;
    options.outputHeight = m_displayHeight;
    options.colorBuffersHDR = sl::Boolean::eTrue;

    options.normalRoughnessMode = sl::DLSSDNormalRoughnessMode::ePacked;
    options.worldToCameraView = XmToSl(viewMatrix);
    options.cameraViewToWorld = XmToSl(XMMatrixInverse(nullptr, viewMatrix));

    options.dlaaPreset = rrPresets[kPresetDLAA];
    options.qualityPreset = rrPresets[kPresetQuality];
    options.balancedPreset = rrPresets[kPresetBalanced];
    options.performancePreset = rrPresets[kPresetPerformance];
    options.ultraPerformancePreset = rrPresets[kPresetUltraPerformance];
    options.ultraQualityPreset = rrPresets[kPresetUltraQuality];
    if (sl::Result r = slDLSSDSetOptions(viewport, options); r != sl::Result::eOk) {
        std::wcout << L"[DLSS-RR] slDLSSDSetOptions failed: " << (int)r << std::endl;
        RevertPresetIfPending(L"slDLSSDSetOptions");
        return;
    }

    sl::Resource slDepth(sl::ResourceType::eTex2d, m_depth.Get(), (uint32_t)stateSRV);
    sl::Resource slMVec(sl::ResourceType::eTex2d, m_mvec.Get(), (uint32_t)stateSRV);
    sl::Resource slNormals(sl::ResourceType::eTex2d, m_normals.Get(), (uint32_t)stateSRV);
    sl::Resource slAlbedo(sl::ResourceType::eTex2d, m_diffuseAlbedo.Get(), (uint32_t)stateSRV);
    sl::Resource slSpecAlb(sl::ResourceType::eTex2d, m_specAlbedo.Get(), (uint32_t)stateSRV);
    sl::Resource slRough(sl::ResourceType::eTex2d, m_roughness.Get(), (uint32_t)stateSRV);
    sl::Resource slSpecHit(sl::ResourceType::eTex2d, m_specHitDist.Get(), (uint32_t)stateSRV);
    sl::Resource slInput(sl::ResourceType::eTex2d, m_input.Get(), (uint32_t)stateSRV);
    sl::Resource slSpecMV(sl::ResourceType::eTex2d, m_specMvec.Get(), (uint32_t)stateSRV);
    sl::Resource slOutput(sl::ResourceType::eTex2d, m_output.Get(), (uint32_t)stateUAV);

    sl::Resource slBiasHint(sl::ResourceType::eTex2d, m_biasHint.Get(), (uint32_t)stateSRV);
    sl::Resource slResponsivity(sl::ResourceType::eTex2d, m_responsivityMask.Get(), (uint32_t)stateSRV);

    sl::Extent renderExtent{0, 0, m_renderWidth, m_renderHeight};
    sl::Extent displayExtent{0, 0, m_displayWidth, m_displayHeight};
    auto life = sl::ResourceLifecycle::eValidUntilEvaluate;

    std::vector<sl::ResourceTag> tags = {
        {&slDepth, sl::kBufferTypeDepth, life, &renderExtent},
        {&slMVec, sl::kBufferTypeMotionVectors, life, &renderExtent},

        {&slNormals, sl::kBufferTypeNormalRoughness, life, &renderExtent},
        {nullptr, sl::kBufferTypeRoughness, life, &renderExtent},
        {&slAlbedo, sl::kBufferTypeAlbedo, life, &renderExtent},
        {&slSpecAlb, sl::kBufferTypeSpecularAlbedo, life, &renderExtent},

        {nullptr, sl::kBufferTypeSpecularHitDistance, life, &renderExtent},
        {&slInput, sl::kBufferTypeScalingInputColor, life, &renderExtent},

        {untagSpecMV ? nullptr : &slSpecMV, sl::kBufferTypeSpecularMotionVectors, life, &renderExtent},

        {&slBiasHint, sl::kBufferTypeBiasCurrentColorHint, life, &renderExtent},

        {useResponsivityMask ? &slResponsivity : nullptr, sl::kBufferTypeResponsivityMask, life, &renderExtent},
        {&slOutput, sl::kBufferTypeScalingOutputColor, life, &displayExtent},
    };
    SL_CHECK(slSetTagForFrame(frameToken, viewport, tags.data(), (uint32_t)tags.size(), cmdList));

    ComPtr<ID3D12InfoQueue> infoQueue;
    if (SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&infoQueue))))
        infoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_ERROR, FALSE);

    const UINT64 dxMsgsBefore = infoQueue ? infoQueue->GetNumStoredMessages() : 0;

    const sl::BaseStructure* evalInputs[] = {&viewport, &options};
    sl::Result evalResult =
        slEvaluateFeature(sl::kFeatureDLSS_RR, frameToken, evalInputs, _countof(evalInputs), cmdList);

    if (infoQueue) {
        if (IsDebuggerPresent())
            infoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_ERROR, TRUE);

        const UINT64 dxMsgsAfter = infoQueue->GetNumStoredMessages();
        for (UINT64 i = dxMsgsBefore; i < dxMsgsAfter; ++i) {
            SIZE_T sz = 0;
            infoQueue->GetMessage(i, nullptr, &sz);
            if (!sz)
                continue;
            std::vector<uint8_t> blob(sz);
            auto* msg = reinterpret_cast<D3D12_MESSAGE*>(blob.data());
            if (FAILED(infoQueue->GetMessage(i, msg, &sz)))
                continue;

            if (msg->Severity > D3D12_MESSAGE_SEVERITY_WARNING)
                continue;
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
        return;
    }

    if (m_presetChangePending) {
        std::memcpy(m_lastGoodPresets, m_activePresets, sizeof(m_lastGoodPresets));
        m_presetChangePending = false;
    }

    m_dlssPrevView = viewMatrix;
    m_dlssPrevProj = xmProj;
    m_lastEvaluationSucceeded = true;
}
