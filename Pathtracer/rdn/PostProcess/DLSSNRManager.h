#pragma once
//====================================
//DLSS-NR MANAGER (DLSS Neural Rendering)
//====================================
//Optional NGX feature id 18 (NVSDK_NGX_Feature_Reserved18 in the official
//enum), hosted through NVIDIA's public DLSS/NGX SDK (github.com/NVIDIA/DLSS)
//and driven purely by runtime-discovered "DLSSNR.*" parameter names — there is
//no public header or documentation for the feature itself.
//
//Deliberately isolated from DLSSManager (Streamline DLSS-RR): DLSS-NR is a
//DISPLAY-RESOLUTION post-process on the final composited LDR frame, not a
//replacement for the reconstruction stage. It runs after every post-process
//pass and BEFORE ImGui, reading a copy of the displayed m_outputResource slice
//and writing its own output texture (never in place).
//
//Build modes (see PATHTRACER_ENABLE_DLSSNR / DLSS_NGX_SDK_ROOT in
//CMakeLists.txt):
//  - Real backend when the official NGX SDK headers + import lib are present.
//  - Status-only stub otherwise: everything still compiles, Evaluate() is a
//    no-op returning false, and the editor panel reports why.
//This header must stay free of NGX includes so both modes share it; the NGX
//handle/parameter types below are opaque C structs, referenced by pointer only.
//
//Trust model: the runtime DLL is DISCOVERED (path, Authenticode status, file
//version) at startup but never loaded by us — no LoadLibrary here. When the
//user enables the feature, the DLL's directory is handed to NGX via the
//official PathListInfo search list and NGX performs its own snippet
//validation. A runtime that fails Authenticode (e.g. hash mismatch = modified
//binary) is refused until the user ticks the session-only
//"Allow modified runtime" opt-in; there is no bypass of any NGX-side check —
//NGX failures are reported verbatim in the status block.

#include "../Common.h"

struct NVSDK_NGX_Handle;
struct NVSDK_NGX_Parameter;

//====================================
//SETTINGS (editor-owned values; UI only writes here, never calls NGX)
//====================================
struct DLSSNRSettings {
    //Off on every launch by design; enabling is an explicit per-session act.
    bool  enabled = false;

    //Session-only consent to hand an Authenticode-failing runtime to NGX.
    //Never persisted anywhere. NGX still runs its own signature validation
    //on the snippet — this flag only stops US from refusing beforehand.
    bool  allowModifiedRuntime = false;

    //Evaluation parameters. Defaults mirror the internal defaults observed in
    //the 310.8 runtime (runtime analysis; no public documentation exists).
    float intensity              = 1.0f;
    float localToneStrength      = 1.0f;
    float localStructureStrength = 1.0f;
    float skinStructureStrength  = -1.0f;
    bool  useAutoMask            = false;
    //Valid range unverified — plumbed through but NOT exposed in the editor
    //until the range is known (a bad value's effect on the feature is unknown).
    int   style                  = 0;
    //DLL-side default is "inverted". Our depth guide is DLSSManager's linear
    //depth (not inverted), so flipping this is a legitimate experiment; kept
    //at the runtime's own default until the semantics can be verified.
    bool  depthInverted          = true;
    float mvecScaleX             = 1.0f;
    float mvecScaleY             = 1.0f;
};

class DLSSNRManager {
public:
    enum class SignatureState {
        eNotFound,      //no candidate file exists
        eSignedValid,   //Authenticode chain + digest OK
        eHashMismatch,  //signature present but file content differs (modified)
        eUnsigned,      //no signature at all
        eUntrusted      //any other WinVerifyTrust failure
    };
    enum class BackendState {
        eStubNoSdk,         //compiled without the NGX SDK
        eRuntimeMissing,    //backend compiled, no nvngx_dlssnr.dll found
        eBlockedUntrusted,  //runtime found but modified and not opted in
        eIdle,              //ready; waiting for Enable
        eNgxInitialized,    //NGX up, feature not created yet
        eFeatureActive,     //feature created, evaluating
        eFailed             //latched failure — see lastResult
    };

    //UTF-8 strings for direct ImGui display.
    struct Status {
        BackendState   backend   = BackendState::eStubNoSdk;
        SignatureState signature = SignatureState::eNotFound;
        std::string    backendText;      //human-readable backend state
        std::string    runtimePath;      //discovered DLL, or "(not found)"
        std::string    runtimeSearched;  //candidate list, for the tooltip
        std::string    runtimeVersion;   //file version, e.g. "310.8.0.0"
        std::string    signatureText;    //human-readable signature verdict
        std::string    driverProbe;      //NGX GetFeatureRequirements result (informational)
        std::string    lastResult;       //last NGX result / error, verbatim
        uint64_t       evalCount = 0;    //successful evaluations this session
    };

    //Runtime discovery only (path/signature/version) — touches no GPU or NGX
    //state, so it is safe during InitDevice.
    void Initialize(UINT displayWidth, UINT displayHeight);

    //Display resolution changed: drop IO textures, release the feature (it is
    //created against fixed output dims) and pulse a temporal reset. The caller
    //must have drained the GPU (Renderer::OnResize does WaitForGPU first).
    void OnDisplayResolution(UINT displayWidth, UINT displayHeight);

    //Record DLSS-NR into cmdList. finalColor is the composited display-res
    //frame in COPY_SOURCE state (its subresource `finalColorSub` is what the
    //back-buffer copy would present); depth/mvec are DLSSManager's render-res
    //guides in UAV state, restored to UAV on every path out. Returns true iff
    //the NGX evaluation was recorded successfully — Output() then holds the
    //processed frame in COPY_SOURCE state. On false the caller presents the
    //untouched finalColor. NGX swaps descriptor heaps: the caller rebinds its
    //own heaps after this call.
    bool Evaluate(ID3D12GraphicsCommandList* cmdList, ID3D12Device* device,
                  ID3D12Resource* finalColor, UINT finalColorSub,
                  ID3D12Resource* depth, ID3D12Resource* mvec,
                  UINT renderWidth, UINT renderHeight);

    ID3D12Resource* Output() const { return m_output.Get(); }

    //Pulse reset=1 into the next evaluation (camera cuts, history-poisoning
    //setting flips, editor button). Also raised internally on feature
    //(re)creation and resolution changes.
    void ForceReset() { m_forceReset = true; }

    //Full NGX teardown (release feature, destroy parameters, NGX shutdown).
    //Call with the GPU idle and the device still alive.
    void Shutdown(ID3D12Device* device);

    //Thread-safe copy of the NGX log tail (the log callback fires on NGX's
    //own threads) — shown collapsed in the editor panel.
    std::vector<std::string> NgxLogTail() const;

    DLSSNRSettings settings;
    const Status& GetStatus() const { return m_status; }

private:
    void DiscoverRuntime();
    void SetFailure(const std::string& what);      //latch eFailed + lastResult
    void ReleaseFeatureOnly();                     //feature handle only, keeps NGX up
    bool TrustGatePassed() const;

#if PATHTRACER_DLSSNR_WITH_NGX
    bool EnsureIoTextures(ID3D12Device* device);
    bool EnsureNgx(ID3D12Device* appDevice);       //init NGX (native device unwrap inside)
    bool EnsureFeature(ID3D12GraphicsCommandList* nativeCmdList);
    void QueryDriverSupport();                     //informational GetFeatureRequirements probe
#endif

    Status m_status;

    //discovery results (wide for the APIs, narrow copies live in m_status)
    std::wstring m_runtimeFileW;   //full path of the discovered DLL
    std::wstring m_runtimeDirW;    //its directory, handed to NGX PathListInfo

    UINT m_displayWidth = 0, m_displayHeight = 0;

    //IO textures, display resolution, R8G8B8A8_UNORM to match m_outputResource
    //(copy-compatible with the back buffer). States tracked explicitly.
    ComPtr<ID3D12Resource> m_input, m_output;
    D3D12_RESOURCE_STATES  m_inputState  = D3D12_RESOURCE_STATE_COPY_DEST;
    D3D12_RESOURCE_STATES  m_outputState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;

    //NGX objects (opaque; only meaningful when built with the SDK)
    NVSDK_NGX_Handle*    m_feature   = nullptr;
    NVSDK_NGX_Parameter* m_params    = nullptr;
    ID3D12Device*        m_ngxDevice = nullptr;   //native (SL-unwrapped) device NGX was initialised with
    bool                 m_ngxInitialized = false;

    //slGetNativeInterface once per proxy pointer — the unwrap result is cached
    //so any AddRef behaviour inside SL stays bounded to a single call.
    void* m_proxyCmdList  = nullptr;
    void* m_nativeCmdList = nullptr;

    bool     m_forceReset = false;
    bool     m_prevEnabled = false;               //off->on edge re-arms a latched failure
    uint32_t m_consecutiveEvalFailures = 0;

    //settings whose change invalidates temporal history -> pulse reset
    int  m_prevStyle = 0;
    bool m_prevAutoMask = false;
    bool m_prevDepthInverted = true;
};
