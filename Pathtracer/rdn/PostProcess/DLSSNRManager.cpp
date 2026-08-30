//====================================
//DLSS-NR MANAGER
//====================================
//See DLSSNRManager.h for the architecture / trust model. Everything NGX lives
//behind PATHTRACER_DLSSNR_WITH_NGX; the stub build keeps runtime discovery
//(path / Authenticode / version) so the editor panel stays informative.

#include "../stdafx.h"
#include "DLSSNRManager.h"
#include "../DXRHelper.h"

#include <sl.h>            //slGetNativeInterface — NGX must not see SL proxies

#include <wintrust.h>
#include <softpub.h>
#include <deque>
#include <mutex>

#if PATHTRACER_DLSSNR_WITH_NGX
#include <nvsdk_ngx.h>
#include <nvsdk_ngx_defs.h>
#endif

namespace {

//====================================
//NGX LOG RING (callback fires on NGX-owned threads)
//====================================
std::mutex              g_ngxLogMutex;
std::deque<std::string> g_ngxLog;
constexpr size_t        kNgxLogCap = 12;

void PushNgxLog(const char* message) {
    if (!message || !*message) return;
    std::string line(message);
    while (!line.empty() && (line.back() == '\n' || line.back() == '\r'))
        line.pop_back();
    if (line.empty()) return;
    std::lock_guard<std::mutex> lk(g_ngxLogMutex);
    g_ngxLog.push_back(std::move(line));
    while (g_ngxLog.size() > kNgxLogCap) g_ngxLog.pop_front();
}

//====================================
//STRING / PATH HELPERS
//====================================
std::string Utf8FromWide(const std::wstring& w) {
    if (w.empty()) return {};
    int n = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int)w.size(), nullptr, 0, nullptr, nullptr);
    std::string s(n, '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int)w.size(), s.data(), n, nullptr, nullptr);
    return s;
}

bool FileExists(const std::wstring& p) {
    DWORD a = GetFileAttributesW(p.c_str());
    return a != INVALID_FILE_ATTRIBUTES && !(a & FILE_ATTRIBUTE_DIRECTORY);
}

std::wstring DirOf(const std::wstring& file) {
    size_t pos = file.find_last_of(L"\\/");
    return pos == std::wstring::npos ? L"." : file.substr(0, pos);
}

//====================================
//AUTHENTICODE (WinVerifyTrust, offline — no revocation fetch)
//====================================
LONG VerifyEmbeddedSignature(const std::wstring& path) {
    WINTRUST_FILE_INFO fileInfo{};
    fileInfo.cbStruct      = sizeof(fileInfo);
    fileInfo.pcwszFilePath = path.c_str();

    WINTRUST_DATA wtd{};
    wtd.cbStruct            = sizeof(wtd);
    wtd.dwUIChoice          = WTD_UI_NONE;
    wtd.fdwRevocationChecks = WTD_REVOKE_NONE;
    wtd.dwUnionChoice       = WTD_CHOICE_FILE;
    wtd.pFile               = &fileInfo;
    wtd.dwStateAction       = WTD_STATEACTION_VERIFY;
    wtd.dwProvFlags         = WTD_CACHE_ONLY_URL_RETRIEVAL;

    GUID policy = WINTRUST_ACTION_GENERIC_VERIFY_V2;
    LONG result = WinVerifyTrust(nullptr, &policy, &wtd);

    wtd.dwStateAction = WTD_STATEACTION_CLOSE;
    WinVerifyTrust(nullptr, &policy, &wtd);
    return result;
}

std::string FileVersionString(const std::wstring& path) {
    DWORD dummy = 0;
    DWORD size = GetFileVersionInfoSizeW(path.c_str(), &dummy);
    if (!size) return "(unknown)";
    std::vector<uint8_t> data(size);
    if (!GetFileVersionInfoW(path.c_str(), 0, size, data.data())) return "(unknown)";
    VS_FIXEDFILEINFO* ffi = nullptr;
    UINT len = 0;
    if (!VerQueryValueW(data.data(), L"\\", (void**)&ffi, &len) || !ffi || len < sizeof(*ffi))
        return "(unknown)";
    char buf[64];
    snprintf(buf, sizeof(buf), "%u.%u.%u.%u",
             HIWORD(ffi->dwFileVersionMS), LOWORD(ffi->dwFileVersionMS),
             HIWORD(ffi->dwFileVersionLS), LOWORD(ffi->dwFileVersionLS));
    return buf;
}

#if PATHTRACER_DLSSNR_WITH_NGX

void NVSDK_CONV NgxLogCallback(const char* message, NVSDK_NGX_Logging_Level, NVSDK_NGX_Feature) {
    PushNgxLog(message);
}

//Stable project identity for NVSDK_NGX_D3D12_Init_with_ProjectID (custom engine).
constexpr const char* kNgxProjectId     = "b3ae1e2f-6cb7-4bb0-99a5-27fa8a920d3e";
constexpr const char* kNgxEngineVersion = "1.0.0";

//DLSS-NR = NGX feature id 18 — a first-class value in the official enum
//(NVSDK_NGX_Feature_Reserved18, nvsdk_ngx_defs.h), so it is hosted through the
//exact CreateFeature/EvaluateFeature path every documented feature uses.
constexpr NVSDK_NGX_Feature kFeatureDLSSNR = NVSDK_NGX_Feature_Reserved18;

std::string NgxResultToString(NVSDK_NGX_Result r) {
    switch (r) {
        case NVSDK_NGX_Result_Success:                          return "Success";
        case NVSDK_NGX_Result_Fail:                             return "Fail (generic)";
        case NVSDK_NGX_Result_FAIL_FeatureNotSupported:         return "FeatureNotSupported";
        case NVSDK_NGX_Result_FAIL_PlatformError:               return "PlatformError";
        case NVSDK_NGX_Result_FAIL_FeatureAlreadyExists:        return "FeatureAlreadyExists";
        case NVSDK_NGX_Result_FAIL_FeatureNotFound:             return "FeatureNotFound";
        case NVSDK_NGX_Result_FAIL_InvalidParameter:            return "InvalidParameter";
        case NVSDK_NGX_Result_FAIL_ScratchBufferTooSmall:       return "ScratchBufferTooSmall";
        case NVSDK_NGX_Result_FAIL_NotInitialized:              return "NotInitialized";
        case NVSDK_NGX_Result_FAIL_UnsupportedInputFormat:      return "UnsupportedInputFormat";
        case NVSDK_NGX_Result_FAIL_RWFlagMissing:               return "RWFlagMissing";
        case NVSDK_NGX_Result_FAIL_MissingInput:                return "MissingInput";
        case NVSDK_NGX_Result_FAIL_UnableToInitializeFeature:   return "UnableToInitializeFeature";
        //Verbatim NGX result. NOTE: for feature 18 this does NOT necessarily
        //mean "driver too old" — observed cause is the installed NGX runtime
        //not exposing/provisioning feature 18 to an unregistered application
        //(no [dlssnr] config section; deny-list allows it; snippet located but
        //load returns FeatureNotFound). See the panel diagnostics note.
        case NVSDK_NGX_Result_FAIL_OutOfDate:                   return "OutOfDate";
        case NVSDK_NGX_Result_FAIL_OutOfGPUMemory:              return "OutOfGPUMemory";
        case NVSDK_NGX_Result_FAIL_UnsupportedFormat:           return "UnsupportedFormat";
        case NVSDK_NGX_Result_FAIL_UnableToWriteToAppDataPath:  return "UnableToWriteToAppDataPath";
        case NVSDK_NGX_Result_FAIL_UnsupportedParameter:        return "UnsupportedParameter";
        case NVSDK_NGX_Result_FAIL_Denied:                      return "Denied (feature disabled by driver/other)";
        case NVSDK_NGX_Result_FAIL_NotImplemented:              return "NotImplemented";
        default: {
            char buf[32];
            snprintf(buf, sizeof(buf), "0x%08X", (unsigned)r);
            return buf;
        }
    }
}

#endif // PATHTRACER_DLSSNR_WITH_NGX

} // namespace

//====================================
//DISCOVERY (both build modes)
//====================================
void DLSSNRManager::DiscoverRuntime() {
    const std::wstring dllName = L"nvngx_dlssnr.dll";
    std::vector<std::wstring> candidates;

    //1) explicit override: DLSSNR_RUNTIME_PATH may name the DLL or a directory
    wchar_t envBuf[1024]{};
    DWORD n = GetEnvironmentVariableW(L"DLSSNR_RUNTIME_PATH", envBuf, _countof(envBuf));
    if (n && n < _countof(envBuf)) {
        std::wstring p = envBuf;
        DWORD attrs = GetFileAttributesW(p.c_str());
        if (attrs != INVALID_FILE_ATTRIBUTES && (attrs & FILE_ATTRIBUTE_DIRECTORY))
            p += L"\\" + dllName;
        candidates.push_back(std::move(p));
    }

    //2) next to the executable (user-placed; the build never copies it there)
    wchar_t exeBuf[MAX_PATH]{};
    if (GetModuleFileNameW(nullptr, exeBuf, _countof(exeBuf)))
        candidates.push_back(DirOf(exeBuf) + L"\\" + dllName);

    //3) the user's Downloads folder (where standalone runtime drops land)
    wchar_t profBuf[MAX_PATH]{};
    n = GetEnvironmentVariableW(L"USERPROFILE", profBuf, _countof(profBuf));
    if (n && n < _countof(profBuf))
        candidates.push_back(std::wstring(profBuf) + L"\\Downloads\\" + dllName);

    m_runtimeFileW.clear();
    m_runtimeDirW.clear();
    std::string searched;
    for (const auto& c : candidates) {
        if (!searched.empty()) searched += "\n";
        searched += Utf8FromWide(c);
        if (m_runtimeFileW.empty() && FileExists(c)) {
            m_runtimeFileW = c;
            m_runtimeDirW  = DirOf(c);
        }
    }
    m_status.runtimeSearched = searched;

    if (m_runtimeFileW.empty()) {
        m_status.signature      = SignatureState::eNotFound;
        m_status.signatureText  = "no runtime to verify";
        m_status.runtimePath    = "(not found)";
        m_status.runtimeVersion = "-";
        return;
    }

    m_status.runtimePath    = Utf8FromWide(m_runtimeFileW);
    m_status.runtimeVersion = FileVersionString(m_runtimeFileW);

    const LONG trust = VerifyEmbeddedSignature(m_runtimeFileW);
    switch (trust) {
        case ERROR_SUCCESS:
            m_status.signature     = SignatureState::eSignedValid;
            m_status.signatureText = "Authenticode: valid signature";
            break;
        case TRUST_E_BAD_DIGEST:
            m_status.signature     = SignatureState::eHashMismatch;
            m_status.signatureText = "Authenticode: HASH MISMATCH — file content differs from its signature (modified binary)";
            break;
        case TRUST_E_NOSIGNATURE:
            m_status.signature     = SignatureState::eUnsigned;
            m_status.signatureText = "Authenticode: not signed";
            break;
        default: {
            char buf[96];
            snprintf(buf, sizeof(buf), "Authenticode: untrusted (WinVerifyTrust 0x%08X)", (unsigned)trust);
            m_status.signature     = SignatureState::eUntrusted;
            m_status.signatureText = buf;
            break;
        }
    }

    LOG(L"[DLSS-NR] runtime: " << m_runtimeFileW
        << L" (trust=" << (int)m_status.signature << L")");
}

// ─────────────────────────────────────────────────────────────────
void DLSSNRManager::Initialize(UINT displayWidth, UINT displayHeight) {
    m_displayWidth  = displayWidth;
    m_displayHeight = displayHeight;

    DiscoverRuntime();

#if PATHTRACER_DLSSNR_WITH_NGX
    if (m_runtimeFileW.empty()) {
        m_status.backend     = BackendState::eRuntimeMissing;
        m_status.backendText = "compiled in, runtime not found (nvngx_dlssnr.dll)";
    } else {
        m_status.backend     = BackendState::eIdle;
        m_status.backendText = "ready — enable to initialize NGX";
    }
#else
    m_status.backend     = BackendState::eStubNoSdk;
    m_status.backendText = "not compiled — configure with DLSS_NGX_SDK_ROOT pointing at the official DLSS SDK";
#endif
    m_status.lastResult = "-";
}

// ─────────────────────────────────────────────────────────────────
bool DLSSNRManager::TrustGatePassed() const {
    if (m_status.signature == SignatureState::eSignedValid) return true;
    return settings.allowModifiedRuntime;
}

void DLSSNRManager::SetFailure(const std::string& what) {
    m_status.backend     = BackendState::eFailed;
    m_status.backendText = "failed (latched — toggle Enable to retry)";
    m_status.lastResult  = what;
    std::wcout << L"[DLSS-NR] " << std::wstring(what.begin(), what.end()) << std::endl;
}

std::vector<std::string> DLSSNRManager::NgxLogTail() const {
    std::lock_guard<std::mutex> lk(g_ngxLogMutex);
    return { g_ngxLog.begin(), g_ngxLog.end() };
}

// ─────────────────────────────────────────────────────────────────
void DLSSNRManager::ReleaseFeatureOnly() {
#if PATHTRACER_DLSSNR_WITH_NGX
    if (m_feature) {
        NVSDK_NGX_D3D12_ReleaseFeature(m_feature);
        m_feature = nullptr;
    }
#endif
}

// ─────────────────────────────────────────────────────────────────
void DLSSNRManager::OnDisplayResolution(UINT displayWidth, UINT displayHeight) {
    if (displayWidth == m_displayWidth && displayHeight == m_displayHeight) return;
    m_displayWidth  = displayWidth;
    m_displayHeight = displayHeight;

    //Feature is created against fixed output dims — rebuild lazily. Caller
    //guarantees GPU idle (Renderer::OnResize WaitForGPU).
    ReleaseFeatureOnly();
    m_input.Reset();
    m_output.Reset();
    m_inputState  = D3D12_RESOURCE_STATE_COPY_DEST;
    m_outputState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    m_forceReset  = true;

    if (m_status.backend == BackendState::eFeatureActive) {
        m_status.backend     = BackendState::eNgxInitialized;
        m_status.backendText = "NGX initialized — feature recreates on next frame";
    }
}

// ─────────────────────────────────────────────────────────────────
void DLSSNRManager::Shutdown(ID3D12Device* device) {
    (void)device;
#if PATHTRACER_DLSSNR_WITH_NGX
    ReleaseFeatureOnly();
    if (m_params) {
        NVSDK_NGX_D3D12_DestroyParameters(m_params);
        m_params = nullptr;
    }
    if (m_ngxInitialized && m_ngxDevice) {
        NVSDK_NGX_D3D12_Shutdown1(m_ngxDevice);
        m_ngxInitialized = false;
    }
    m_ngxDevice = nullptr;
#endif
    m_input.Reset();
    m_output.Reset();
}

//====================================
//REAL BACKEND
//====================================
#if PATHTRACER_DLSSNR_WITH_NGX

bool DLSSNRManager::EnsureIoTextures(ID3D12Device* device) {
    if (m_input && m_output) return true;
    if (!m_displayWidth || !m_displayHeight) return false;

    //R8G8B8A8_UNORM to match m_outputResource: the input is a plain copy of
    //the displayed slice, the output is copy-compatible with the back buffer.
    //Both get UAV capability (the output requires it; the input costs nothing
    //and sidesteps NGX RWFlag validation surprises on an undocumented feature).
    auto makeTex = [&](ComPtr<ID3D12Resource>& res, D3D12_RESOURCE_STATES initial,
                       const wchar_t* name) -> bool {
        D3D12_RESOURCE_DESC d{};
        d.Dimension        = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        d.Width            = m_displayWidth;
        d.Height           = m_displayHeight;
        d.DepthOrArraySize = 1;
        d.MipLevels        = 1;
        d.Format           = DXGI_FORMAT_R8G8B8A8_UNORM;
        d.SampleDesc.Count = 1;
        d.Flags            = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        if (FAILED(device->CreateCommittedResource(
                &nv_helpers_dx12::kDefaultHeapProps, D3D12_HEAP_FLAG_NONE,
                &d, initial, nullptr, IID_PPV_ARGS(&res))))
            return false;
        res->SetName(name);
        return true;
    };

    if (!makeTex(m_input, D3D12_RESOURCE_STATE_COPY_DEST, L"DLSSNR_Input") ||
        !makeTex(m_output, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, L"DLSSNR_Output")) {
        SetFailure("failed to allocate DLSS-NR IO textures");
        return false;
    }
    m_inputState  = D3D12_RESOURCE_STATE_COPY_DEST;
    m_outputState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    return true;
}

// ─────────────────────────────────────────────────────────────────
//Informational only: ask the driver (via the official requirements API)
//whether it knows feature id 18 on this adapter. Whatever it answers, the
//authoritative gate stays CreateFeature — this just makes the panel clearer.
void DLSSNRManager::QueryDriverSupport() {
    ComPtr<IDXGIFactory4> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))) || !m_ngxDevice) {
        m_status.driverProbe = "probe unavailable";
        return;
    }
    LUID luid = m_ngxDevice->GetAdapterLuid();
    ComPtr<IDXGIAdapter> adapter;
    if (FAILED(factory->EnumAdapterByLuid(luid, IID_PPV_ARGS(&adapter)))) {
        m_status.driverProbe = "probe unavailable (adapter)";
        return;
    }

    wchar_t tmpPath[MAX_PATH]{};
    GetTempPathW(MAX_PATH, tmpPath);

    NVSDK_NGX_FeatureDiscoveryInfo disc{};
    disc.SDKVersion                             = NVSDK_NGX_Version_API;
    disc.FeatureID                              = kFeatureDLSSNR;
    disc.Identifier.IdentifierType              = NVSDK_NGX_Application_Identifier_Type_Project_Id;
    disc.Identifier.v.ProjectDesc.ProjectId     = kNgxProjectId;
    disc.Identifier.v.ProjectDesc.EngineType    = NVSDK_NGX_ENGINE_TYPE_CUSTOM;
    disc.Identifier.v.ProjectDesc.EngineVersion = kNgxEngineVersion;
    disc.ApplicationDataPath                    = tmpPath;

    NVSDK_NGX_FeatureRequirement req{};
    NVSDK_NGX_Result r = NVSDK_NGX_D3D12_GetFeatureRequirements(adapter.Get(), &disc, &req);
    if (NVSDK_NGX_FAILED(r)) {
        m_status.driverProbe = "GetFeatureRequirements(18): " + NgxResultToString(r);
        return;
    }
    if (req.FeatureSupported == 0) {
        m_status.driverProbe = "driver reports feature 18 supported";
        return;
    }
    std::string bits;
    auto add = [&](const char* s) { if (!bits.empty()) bits += ", "; bits += s; };
    if (req.FeatureSupported & NVSDK_NGX_FeatureSupportResult_CheckNotPresent)                add("no support-check in driver");
    if (req.FeatureSupported & NVSDK_NGX_FeatureSupportResult_DriverVersionUnsupported)       add("driver too old");
    if (req.FeatureSupported & NVSDK_NGX_FeatureSupportResult_AdapterUnsupported)             add("adapter unsupported");
    if (req.FeatureSupported & NVSDK_NGX_FeatureSupportResult_OSVersionBelowMinimumSupported) add("OS too old");
    if (req.FeatureSupported & NVSDK_NGX_FeatureSupportResult_NotImplemented)                 add("check not implemented");
    m_status.driverProbe = "driver probe: " + bits;
}

// ─────────────────────────────────────────────────────────────────
bool DLSSNRManager::EnsureNgx(ID3D12Device* appDevice) {
    if (m_ngxInitialized) return true;

    //NGX gets the NATIVE device — the renderer's device came from the SL
    //interposer and may be a proxy. Falls back to the given pointer when SL
    //says it is not proxied.
    void* native = nullptr;
    if (slGetNativeInterface(appDevice, &native) != sl::Result::eOk || !native)
        native = appDevice;
    m_ngxDevice = static_cast<ID3D12Device*>(native);

    QueryDriverSupport();

    //Writable scratch for NGX (file sinks are disabled below, but the API
    //wants a path).
    wchar_t tmpPath[MAX_PATH]{};
    GetTempPathW(MAX_PATH, tmpPath);

    //Official search-path mechanism: NGX looks for nvngx_dlssnr.dll in these
    //directories (plus the driver store). This is the ONLY way the discovered
    //runtime enters the process — we never LoadLibrary it, and NGX applies its
    //own snippet validation to whatever it finds.
    const wchar_t* searchPaths[] = { m_runtimeDirW.c_str() };
    NVSDK_NGX_FeatureCommonInfo common{};
    common.PathListInfo.Path                  = searchPaths;
    common.PathListInfo.Length                = m_runtimeDirW.empty() ? 0u : 1u;
    common.LoggingInfo.LoggingCallback        = &NgxLogCallback;
    common.LoggingInfo.MinimumLoggingLevel    = NVSDK_NGX_LOGGING_LEVEL_ON;
    common.LoggingInfo.DisableOtherLoggingSinks = true;

    NVSDK_NGX_Result r = NVSDK_NGX_D3D12_Init_with_ProjectID(
        kNgxProjectId, NVSDK_NGX_ENGINE_TYPE_CUSTOM, kNgxEngineVersion,
        tmpPath, m_ngxDevice, &common, NVSDK_NGX_Version_API);
    if (NVSDK_NGX_FAILED(r)) {
        SetFailure("NVSDK_NGX_D3D12_Init_with_ProjectID: " + NgxResultToString(r));
        m_ngxDevice = nullptr;
        return false;
    }

    r = NVSDK_NGX_D3D12_AllocateParameters(&m_params);
    if (NVSDK_NGX_FAILED(r) || !m_params) {
        m_params = nullptr;
        NVSDK_NGX_D3D12_Shutdown1(m_ngxDevice);   //unwind so a retry starts clean
        m_ngxDevice = nullptr;
        SetFailure("NVSDK_NGX_D3D12_AllocateParameters: " + NgxResultToString(r));
        return false;
    }

    m_ngxInitialized     = true;
    m_status.backend     = BackendState::eNgxInitialized;
    m_status.backendText = "NGX initialized";
    LOG(L"[DLSS-NR] NGX initialized (native device unwrap "
        << (native != appDevice ? L"applied" : L"not needed") << L")");
    return true;
}

// ─────────────────────────────────────────────────────────────────
bool DLSSNRManager::EnsureFeature(ID3D12GraphicsCommandList* nativeCmdList) {
    if (m_feature) return true;

    //Creation parameters — names discovered from the runtime; non-scaling
    //display-res operation (ScalingRatio 1, Upscaling off), preset 0.
    NVSDK_NGX_Parameter_SetUI(m_params, "DLSSNR.Width",              m_displayWidth);
    NVSDK_NGX_Parameter_SetUI(m_params, "DLSSNR.Height",             m_displayHeight);
    NVSDK_NGX_Parameter_SetUI(m_params, "DLSSNR.InputWidth",         m_displayWidth);
    NVSDK_NGX_Parameter_SetUI(m_params, "DLSSNR.InputHeight",        m_displayHeight);
    NVSDK_NGX_Parameter_SetUI(m_params, "DLSSNR.OutputWidth",        m_displayWidth);
    NVSDK_NGX_Parameter_SetUI(m_params, "DLSSNR.OutputHeight",       m_displayHeight);
    NVSDK_NGX_Parameter_SetF (m_params, "DLSSNR.ScalingRatio",       1.0f);
    NVSDK_NGX_Parameter_SetUI(m_params, "DLSSNR.Upscaling",          0);
    NVSDK_NGX_Parameter_SetUI(m_params, "DLSSNR.Hint.Render.Preset", 0);
    NVSDK_NGX_Parameter_SetUI(m_params, NVSDK_NGX_Parameter_CreationNodeMask,   1);
    NVSDK_NGX_Parameter_SetUI(m_params, NVSDK_NGX_Parameter_VisibilityNodeMask, 1);

    NVSDK_NGX_Result r = NVSDK_NGX_D3D12_CreateFeature(
        nativeCmdList, kFeatureDLSSNR, m_params, &m_feature);
    if (NVSDK_NGX_FAILED(r) || !m_feature) {
        m_feature = nullptr;
        SetFailure("CreateFeature(18 / DLSS-NR): " + NgxResultToString(r));
        return false;
    }

    m_forceReset         = true;   //fresh feature has no history
    m_status.backend     = BackendState::eFeatureActive;
    m_status.backendText = "feature active";
    LOG(L"[DLSS-NR] feature created (" << m_displayWidth << L"x" << m_displayHeight << L")");
    return true;
}

// ─────────────────────────────────────────────────────────────────
bool DLSSNRManager::Evaluate(ID3D12GraphicsCommandList* cmdList, ID3D12Device* device,
                             ID3D12Resource* finalColor, UINT finalColorSub,
                             ID3D12Resource* depth, ID3D12Resource* mvec,
                             UINT renderWidth, UINT renderHeight) {
    //off->on edge: re-arm a latched failure and drop history — an explicit
    //user retry is the only thing that clears eFailed.
    if (settings.enabled && !m_prevEnabled) {
        if (m_status.backend == BackendState::eFailed) {
            m_status.backend     = m_ngxInitialized ? BackendState::eNgxInitialized
                                                    : BackendState::eIdle;
            m_status.backendText = "retrying after failure";
        }
        m_consecutiveEvalFailures = 0;
        m_forceReset = true;
    }
    m_prevEnabled = settings.enabled;

    if (!settings.enabled) return false;
    if (!cmdList || !device || !finalColor) return false;
    if (m_status.backend == BackendState::eRuntimeMissing ||
        m_status.backend == BackendState::eFailed)
        return false;

    //Trust gate: a runtime that fails Authenticode needs the session-only
    //opt-in before its directory is ever handed to NGX.
    if (!TrustGatePassed()) {
        m_status.backend     = BackendState::eBlockedUntrusted;
        m_status.backendText = "blocked: runtime failed signature validation — tick 'Allow modified runtime' to proceed";
        return false;
    }
    if (m_status.backend == BackendState::eBlockedUntrusted) {
        m_status.backend     = BackendState::eIdle;
        m_status.backendText = "ready";
    }

    //History-poisoning setting flips pulse a reset.
    if (settings.style != m_prevStyle ||
        settings.useAutoMask != m_prevAutoMask ||
        settings.depthInverted != m_prevDepthInverted) {
        m_prevStyle         = settings.style;
        m_prevAutoMask      = settings.useAutoMask;
        m_prevDepthInverted = settings.depthInverted;
        m_forceReset = true;
    }

    if (!EnsureIoTextures(device)) return false;
    if (!EnsureNgx(device))        return false;

    //NGX records through the NATIVE command list (cached unwrap); our own
    //barriers/copies stay on the app's (possibly proxied) list like every
    //other pass, which lands on the same underlying list in call order.
    if (static_cast<void*>(cmdList) != m_proxyCmdList) {
        void* native = nullptr;
        if (slGetNativeInterface(cmdList, &native) != sl::Result::eOk || !native)
            native = cmdList;
        m_proxyCmdList  = cmdList;
        m_nativeCmdList = native;
    }
    auto* ngxCmdList = static_cast<ID3D12GraphicsCommandList*>(m_nativeCmdList);

    //── input copy: the displayed slice of the composited frame ──────────
    //finalColor arrives in COPY_SOURCE (the renderer's present transition
    //already happened, which also synchronized the post-process UAV writes).
    if (m_inputState != D3D12_RESOURCE_STATE_COPY_DEST) {
        auto b = CD3DX12_RESOURCE_BARRIER::Transition(m_input.Get(), m_inputState,
                                                      D3D12_RESOURCE_STATE_COPY_DEST);
        cmdList->ResourceBarrier(1, &b);
        m_inputState = D3D12_RESOURCE_STATE_COPY_DEST;
    }
    {
        CD3DX12_TEXTURE_COPY_LOCATION src(finalColor, finalColorSub);
        CD3DX12_TEXTURE_COPY_LOCATION dst(m_input.Get(), 0);
        cmdList->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    }
    {
        auto b = CD3DX12_RESOURCE_BARRIER::Transition(m_input.Get(),
            D3D12_RESOURCE_STATE_COPY_DEST, kSRV);
        cmdList->ResourceBarrier(1, &b);
        m_inputState = kSRV;
    }

    //── guides: UAV -> SRV, restored on EVERY path out ────────────────────
    //(the transition out of UAV also synchronizes the writes that produced
    //them; same discipline as DLSSManager::Evaluate)
    ID3D12Resource* guides[] = { depth, mvec };
    std::vector<D3D12_RESOURCE_BARRIER> preB;
    for (auto* g : guides)
        if (g) preB.push_back(CD3DX12_RESOURCE_BARRIER::Transition(g,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, kSRV));
    if (!preB.empty()) cmdList->ResourceBarrier((UINT)preB.size(), preB.data());

    struct GuideRestore {
        ID3D12GraphicsCommandList* cl;
        std::vector<D3D12_RESOURCE_BARRIER> barriers;
        ~GuideRestore() {
            if (!barriers.empty()) cl->ResourceBarrier((UINT)barriers.size(), barriers.data());
        }
    } restore{ cmdList, {} };
    for (auto* g : guides)
        if (g) restore.barriers.push_back(CD3DX12_RESOURCE_BARRIER::Transition(g,
            kSRV, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));

    //── output: into UAV state + UAV barrier ordering vs its last consumer ──
    if (m_outputState != D3D12_RESOURCE_STATE_UNORDERED_ACCESS) {
        auto b = CD3DX12_RESOURCE_BARRIER::Transition(m_output.Get(), m_outputState,
                                                      D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        cmdList->ResourceBarrier(1, &b);
        m_outputState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    }
    {
        auto b = CD3DX12_RESOURCE_BARRIER::UAV(m_output.Get());
        cmdList->ResourceBarrier(1, &b);
    }

    //── feature (recorded into this command list on first enabled frame) ──
    if (!EnsureFeature(ngxCmdList)) return false;   //guides restored by `restore`

    //── evaluation parameters (all names runtime-discovered) ──────────────
    //Resources are re-set every frame — the parameter object persists and
    //stale pointers from a previous frame must never survive.
    NVSDK_NGX_Parameter_SetD3d12Resource(m_params, "DLSSNR.Color",  m_input.Get());
    NVSDK_NGX_Parameter_SetD3d12Resource(m_params, "DLSSNR.Depth",  depth);
    NVSDK_NGX_Parameter_SetD3d12Resource(m_params, "DLSSNR.MVec",   mvec);
    NVSDK_NGX_Parameter_SetD3d12Resource(m_params, "DLSSNR.Output", m_output.Get());
    NVSDK_NGX_Parameter_SetUI(m_params, "DLSSNR.Enabled",       1);
    NVSDK_NGX_Parameter_SetUI(m_params, "DLSSNR.Reset",         m_forceReset ? 1u : 0u);
    NVSDK_NGX_Parameter_SetF (m_params, "DLSSNR.Intensity",              settings.intensity);
    NVSDK_NGX_Parameter_SetF (m_params, "DLSSNR.LocalToneStrength",      settings.localToneStrength);
    NVSDK_NGX_Parameter_SetF (m_params, "DLSSNR.LocalStructureStrength", settings.localStructureStrength);
    NVSDK_NGX_Parameter_SetF (m_params, "DLSSNR.SkinStructureStrength",  settings.skinStructureStrength);
    NVSDK_NGX_Parameter_SetUI(m_params, "DLSSNR.UseAutoMask",   settings.useAutoMask ? 1u : 0u);
    NVSDK_NGX_Parameter_SetI (m_params, "DLSSNR.Style",         settings.style);
    NVSDK_NGX_Parameter_SetUI(m_params, "DLSSNR.DepthInverted", settings.depthInverted ? 1u : 0u);
    NVSDK_NGX_Parameter_SetF (m_params, "DLSSNR.MVecScaleX",    settings.mvecScaleX);
    NVSDK_NGX_Parameter_SetF (m_params, "DLSSNR.MVecScaleY",    settings.mvecScaleY);
    //DLSSNR.UICorrection deliberately left unset: ImGui renders after this
    //stage, so there is no UI in the processed image to correct.
    (void)renderWidth; (void)renderHeight;   //guides keep their own (render-res) dims

    NVSDK_NGX_Result r = NVSDK_NGX_D3D12_EvaluateFeature(ngxCmdList, m_feature, m_params, nullptr);
    if (NVSDK_NGX_FAILED(r)) {
        m_status.lastResult = "EvaluateFeature: " + NgxResultToString(r);
        if (++m_consecutiveEvalFailures >= 3)
            SetFailure("EvaluateFeature failed 3x in a row: " + NgxResultToString(r));
        return false;   //guides restored; output stays UAV; input state tracked
    }

    //keep m_forceReset armed across failed attempts; a delivered reset clears it
    m_forceReset = false;
    m_consecutiveEvalFailures = 0;
    ++m_status.evalCount;
    m_status.lastResult  = "OK";
    m_status.backend     = BackendState::eFeatureActive;
    m_status.backendText = "feature active";

    {
        auto b = CD3DX12_RESOURCE_BARRIER::Transition(m_output.Get(),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        cmdList->ResourceBarrier(1, &b);
        m_outputState = D3D12_RESOURCE_STATE_COPY_SOURCE;
    }
    return true;
}

#else // !PATHTRACER_DLSSNR_WITH_NGX

//====================================
//STATUS-ONLY STUB
//====================================
bool DLSSNRManager::Evaluate(ID3D12GraphicsCommandList*, ID3D12Device*,
                             ID3D12Resource*, UINT,
                             ID3D12Resource*, ID3D12Resource*, UINT, UINT) {
    m_prevEnabled = settings.enabled;
    return false;   //backend state set in Initialize explains why
}

#endif // PATHTRACER_DLSSNR_WITH_NGX
