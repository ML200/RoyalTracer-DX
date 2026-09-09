#include "../stdafx.h"
#include "DLSSNRManager.h"
#include <sl.h>
#include <wintrust.h>
#include <softpub.h>
#include <cmath>

namespace {
constexpr auto kRead = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
std::string Utf8(const std::wstring& w) {
    if (w.empty()) return {};
    int n = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int)w.size(), nullptr, 0, nullptr, nullptr);
    std::string s(n, '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int)w.size(), s.data(), n, nullptr, nullptr);
    return s;
}
std::wstring ExeDirectory() {
    wchar_t path[32768]{};
    GetModuleFileNameW(nullptr, path, _countof(path));
    std::wstring s(path);
    return s.substr(0, s.find_last_of(L"\\/"));
}
LONG VerifySignature(const std::wstring& path) {
    WINTRUST_FILE_INFO file{}; file.cbStruct = sizeof(file); file.pcwszFilePath = path.c_str();
    WINTRUST_DATA data{}; data.cbStruct = sizeof(data); data.dwUIChoice = WTD_UI_NONE;
    data.fdwRevocationChecks = WTD_REVOKE_NONE; data.dwUnionChoice = WTD_CHOICE_FILE;
    data.pFile = &file; data.dwStateAction = WTD_STATEACTION_VERIFY; data.dwProvFlags = WTD_CACHE_ONLY_URL_RETRIEVAL;
    GUID policy = WINTRUST_ACTION_GENERIC_VERIFY_V2;
    LONG result = WinVerifyTrust(nullptr, &policy, &data);
    data.dwStateAction = WTD_STATEACTION_CLOSE; WinVerifyTrust(nullptr, &policy, &data);
    return result;
}
std::string Version(const std::wstring& path) {
    DWORD unused = 0, size = GetFileVersionInfoSizeW(path.c_str(), &unused);
    std::vector<uint8_t> data(size);
    if (!size || !GetFileVersionInfoW(path.c_str(), 0, size, data.data())) return "unknown";
    VS_FIXEDFILEINFO* info = nullptr; UINT length = 0;
    if (!VerQueryValueW(data.data(), L"\\", (void**)&info, &length) || length < sizeof(*info)) return "unknown";
    char version[64]; snprintf(version, sizeof(version), "%u.%u.%u.%u",
        HIWORD(info->dwFileVersionMS), LOWORD(info->dwFileVersionMS), HIWORD(info->dwFileVersionLS), LOWORD(info->dwFileVersionLS));
    return version;
}
template<class T> ComPtr<T> Native(T* object) {
    ComPtr<T> native;
    void* pointer = nullptr;
    // SL adds a reference even when the input is already native.
    if (slGetNativeInterface(object, &pointer) == sl::Result::eOk && pointer)
        native.Attach(static_cast<T*>(pointer));
    else native = object;
    return native;
}
void Transition(ID3D12GraphicsCommandList* cmd, ID3D12Resource* resource,
                D3D12_RESOURCE_STATES& state, D3D12_RESOURCE_STATES next) {
    if (state == next) return;
    auto barrier = CD3DX12_RESOURCE_BARRIER::Transition(resource, state, next);
    cmd->ResourceBarrier(1, &barrier); state = next;
}
float FiniteClamp(float value, float fallback, float low, float high) {
    return std::isfinite(value) ? std::clamp(value, low, high) : fallback;
}
}

void DLSSNRManager::DiscoverRuntime() {
    std::vector<std::wstring> candidates;
    wchar_t overridePath[32768]{};
    DWORD length = GetEnvironmentVariableW(L"DLSSNR_RUNTIME_PATH", overridePath, _countof(overridePath));
    if (length && length < _countof(overridePath)) {
        std::wstring path(overridePath);
        DWORD attr = GetFileAttributesW(path.c_str());
        if (attr != INVALID_FILE_ATTRIBUTES && (attr & FILE_ATTRIBUTE_DIRECTORY)) path += L"\\nvngx_dlssnr.dll";
        wchar_t full[32768]{};
        DWORD count = GetFullPathNameW(path.c_str(), _countof(full), full, nullptr);
        if (count && count < _countof(full)) path = full;
        candidates.push_back(path); // Explicit override is authoritative, including errors.
    } else {
        candidates.push_back(ExeDirectory() + L"\\dlssnr\\nvngx_dlssnr.dll");
        candidates.push_back(ExeDirectory() + L"\\nvngx_dlssnr.dll");
    }
    m_runtimeFile.clear(); m_status.runtimeSearched.clear();
    for (const auto& path : candidates) {
        m_status.runtimeSearched += Utf8(path) + "\n";
        DWORD attr = GetFileAttributesW(path.c_str());
        if (m_runtimeFile.empty() && attr != INVALID_FILE_ATTRIBUTES && !(attr & FILE_ATTRIBUTE_DIRECTORY)) m_runtimeFile = path;
    }
    if (m_runtimeFile.empty()) {
        m_status.runtimePath = "not found"; m_status.runtimeVersion = "-";
        m_status.signature = SignatureState::eNotFound; m_status.signatureText = "Runtime missing";
        return;
    }
    m_status.runtimePath = Utf8(m_runtimeFile); m_status.runtimeVersion = Version(m_runtimeFile);
    LONG signature = VerifySignature(m_runtimeFile);
    m_status.signature = signature == ERROR_SUCCESS ? SignatureState::eSignedValid :
        signature == TRUST_E_BAD_DIGEST ? SignatureState::eHashMismatch :
        signature == TRUST_E_NOSIGNATURE ? SignatureState::eUnsigned : SignatureState::eUntrusted;
    m_status.signatureText = signature == ERROR_SUCCESS ? "Valid digital signature" : "Signature could not be verified";
}
void DLSSNRManager::Initialize(UINT width, UINT height) {
    m_width = width; m_height = height; DiscoverRuntime();
#if PATHTRACER_DLSSNR_NATIVE
    m_status.backend = m_runtimeFile.empty() ? BackendState::eRuntimeMissing : BackendState::eIdle;
    m_status.backendText = m_runtimeFile.empty() ? "Runtime missing; configure DLSSNR_RUNTIME_FILE or DLSSNR_RUNTIME_PATH" : "Native experimental bridge ready";
#else
    m_status.backend = BackendState::eStubNoSdk; m_status.backendText = "Disabled at build time (PATHTRACER_ENABLE_DLSSNR=OFF)";
#endif
}
dlssnr::Tuning DLSSNRManager::Tuning() const {
    dlssnr::Tuning result;
    result.intensity = FiniteClamp(settings.intensity, 1.0f, 0.0f, 1.0f);
    result.localTone = FiniteClamp(settings.localToneStrength, 1.0f, 0.0f, 1.0f);
    result.localStructure = FiniteClamp(settings.localStructureStrength, 1.0f, 0.0f, 1.0f);
    result.skinStructure = FiniteClamp(settings.skinStructureStrength, -1.0f, -1.0f, 1.0f);
    result.autoMask = settings.useAutoMask ? 1u : 0u;
    result.modelStyle = settings.modelStyle >= 0 && settings.modelStyle < dlssnr::kModelStyleCount ? (uint32_t)settings.modelStyle : 0u;
    result.renderPreset = settings.renderPreset >= 0 && settings.renderPreset < dlssnr::kRenderPresetCount ? (uint32_t)settings.renderPreset : 0u;
    return result;
}
void DLSSNRManager::Fail(const char* stage, uint32_t result) {
    char code[24]; snprintf(code, sizeof(code), "0x%08X", result);
    m_status.lastResult = std::string(stage) + ": " + code;
    m_status.backend = BackendState::eFailed;
    m_status.backendText = "Failed; original scene remains visible. Change settings or toggle Enable to retry.";
    m_forceReset = true;
    LOG(L"[DLSS-NR] " << stage << L" failed: " << result);
}
void DLSSNRManager::ReleaseFeature() {
#if PATHTRACER_DLSSNR_NATIVE
    if (m_featureCreated) {
        uint32_t r = dlssnr::RoyalNRReleaseFeature(m_context);
        if (r != 1u) { Fail("ReleaseFeature", r); return; }
    }
#endif
    m_featureCreated = false; m_forceReset = true;
}
void DLSSNRManager::PrepareFrameGPUIdle() {
    const auto tuning = Tuning();
    const bool changed = tuning.intensity != m_tuning.intensity || tuning.localTone != m_tuning.localTone ||
        tuning.localStructure != m_tuning.localStructure || tuning.skinStructure != m_tuning.skinStructure || tuning.autoMask != m_tuning.autoMask ||
        tuning.modelStyle != m_tuning.modelStyle || tuning.renderPreset != m_tuning.renderPreset;
    if (m_featureCreated && (!settings.enabled || changed)) ReleaseFeature();
    if (settings.enabled && (!m_prevEnabled || changed) && m_status.backend == BackendState::eFailed) {
#if PATHTRACER_DLSSNR_NATIVE
        dlssnr::RoyalNRDestroyContext(m_context); m_context = nullptr;
#endif
        m_featureCreated = false;
        m_status.backend = BackendState::eIdle; m_status.backendText = "Retrying native bridge";
    }
    if (!settings.enabled || changed || !m_prevEnabled) m_forceReset = true;
    if (!settings.enabled && m_status.backend == BackendState::eFeatureActive) {
        m_status.backend = BackendState::eIdle; m_status.backendText = "Native bridge ready (disabled)";
    }
    m_tuning = tuning; m_prevEnabled = settings.enabled;
}
void DLSSNRManager::OnDisplayResolution(UINT width, UINT height) {
    ReleaseFeature(); m_input.Reset(); m_output.Reset();
    m_width = width; m_height = height; m_guideWidth = m_guideHeight = 0;
    m_inputState = D3D12_RESOURCE_STATE_COPY_DEST; m_outputState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
}
void DLSSNRManager::Shutdown(ID3D12Device*) {
#if PATHTRACER_DLSSNR_NATIVE
    dlssnr::RoyalNRDestroyContext(m_context);
#endif
    m_context = nullptr; m_featureCreated = false; m_input.Reset(); m_output.Reset();
}
bool DLSSNRManager::EnsureTextures(ID3D12Device* device) {
    if (m_input && m_output) return true;
    auto desc = CD3DX12_RESOURCE_DESC::Tex2D(DXGI_FORMAT_R8G8B8A8_UNORM, m_width, m_height, 1, 1, 1, 0, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    auto heap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
    HRESULT hr = device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
        D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&m_input));
    if (SUCCEEDED(hr)) hr = device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&m_output));
    if (FAILED(hr)) { m_input.Reset(); m_output.Reset(); Fail("CreateTextures", hr); return false; }
    m_input->SetName(L"DLSSNR_OpaqueSDRInput"); m_output->SetName(L"DLSSNR_SDRResult");
    m_inputState = D3D12_RESOURCE_STATE_COPY_DEST; m_outputState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    return true;
}
bool DLSSNRManager::Evaluate(ID3D12GraphicsCommandList* cmd, ID3D12Device* device,
    ID3D12Resource* color, UINT subresource, ID3D12Resource* depth, ID3D12Resource* motion, UINT rw, UINT rh) {
#if PATHTRACER_DLSSNR_NATIVE
    if (!settings.enabled || m_status.backend == BackendState::eFailed || m_status.backend == BackendState::eRuntimeMissing) { m_forceReset = true; return false; }
    if (!cmd || !device || !color || !depth || !motion || !rw || !rh) { m_forceReset = true; return false; }
    const auto cd = color->GetDesc(), dd = depth->GetDesc(), md = motion->GetDesc();
    if (cd.Format != DXGI_FORMAT_R8G8B8A8_UNORM || cd.Width != m_width || cd.Height != m_height || cd.MipLevels != 1 ||
        subresource >= cd.DepthOrArraySize || dd.Format != DXGI_FORMAT_R32_FLOAT || md.Format != DXGI_FORMAT_R16G16_FLOAT ||
        dd.Width != rw || dd.Height != rh || md.Width != rw || md.Height != rh) { Fail("Input contract", 0xBAD00005u); return false; }
    if (rw != m_guideWidth || rh != m_guideHeight) { m_guideWidth = rw; m_guideHeight = rh; m_forceReset = true; }
    if (!EnsureTextures(device)) return false;
    if (!m_context) {
        auto native = Native(device);
        uint32_t r = dlssnr::RoyalNRCreateContext(m_runtimeFile.c_str(), native.Get(), &m_context);
        if (r != 1u) { Fail("Initialize pinned 310.8.0 runtime", r); return false; }
    }
    auto nativeCmd = Native(cmd);
    if (!m_featureCreated) {
        uint32_t r = dlssnr::RoyalNRCreateFeature(m_context, nativeCmd.Get(), m_width, m_height, &m_tuning);
        if (r != 1u) { Fail("CreateFeature", r); return false; }
        m_featureCreated = true; m_forceReset = true;
        m_status.backendText = "Preparing model; processing starts next frame";
        // Creation records GPU work. Submit normally; next frame's fence must
        // complete it before the first evaluate. No same-list create/evaluate.
        return false;
    }
    Transition(cmd, m_input.Get(), m_inputState, D3D12_RESOURCE_STATE_COPY_DEST);
    CD3DX12_TEXTURE_COPY_LOCATION src(color, subresource), dst(m_input.Get(), 0);
    cmd->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    Transition(cmd, m_input.Get(), m_inputState, kRead);
    Transition(cmd, m_output.Get(), m_outputState, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    D3D12_RESOURCE_BARRIER barriers[] = {
        CD3DX12_RESOURCE_BARRIER::Transition(depth, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, kRead),
        CD3DX12_RESOURCE_BARRIER::Transition(motion, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, kRead)
    };
    cmd->ResourceBarrier(2, barriers);
    dlssnr::Frame frame;
    frame.color = m_input.Get(); frame.output = m_output.Get(); frame.depth = depth; frame.motion = motion;
    frame.width = m_width; frame.height = m_height; frame.guideWidth = rw; frame.guideHeight = rh;
    // NR takes guide-pixel units: the supplied NVIDIA plugin multiplies SL's
    // normalized mvecScale by the motion subrect size. Our vectors are already
    // render pixels, so use 1, not display/render (NR handles the subrect ratio).
    frame.motionScaleX = 1.0f; frame.motionScaleY = 1.0f;
    frame.reset = m_forceReset ? 1u : 0u;
    uint32_t result = dlssnr::RoyalNREvaluate(m_context, nativeCmd.Get(), &frame);
    for (auto& b : barriers) std::swap(b.Transition.StateBefore, b.Transition.StateAfter);
    cmd->ResourceBarrier(2, barriers); // restore guides on success AND failure
    if (result != 1u) { Fail("EvaluateFeature", result); return false; }
    Transition(cmd, m_output.Get(), m_outputState, D3D12_RESOURCE_STATE_COPY_SOURCE);
    m_forceReset = false; ++m_status.evalCount;
    m_status.backend = BackendState::eFeatureActive; m_status.backendText = "DLSS 5 Neural Rendering active (experimental native bridge)";
    m_status.lastResult = "Success";
    return true;
#else
    return false;
#endif
}
