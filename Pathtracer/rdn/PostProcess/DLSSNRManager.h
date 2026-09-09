#pragma once
#include "../Common.h"
#include "DLSSNRBridge.h"

struct DLSSNRSettings {
    bool enabled = false;
    int modelStyle = 0;
    int renderPreset = 0;
    float intensity = 1.0f;
    float localToneStrength = 1.0f;
    float localStructureStrength = 1.0f;
    float skinStructureStrength = -1.0f;
    bool useAutoMask = false;
};

// Display-resolution SDR post-process, after tone mapping, before UI.
// Pinned experimental runtime and bridge contract: docs/DLSS5.md.
class DLSSNRManager {
public:
    enum class SignatureState { eNotFound, eSignedValid, eHashMismatch, eUnsigned, eUntrusted };
    enum class BackendState { eStubNoSdk, eRuntimeMissing, eIdle, eFeatureActive, eFailed };
    struct Status {
        BackendState backend = BackendState::eStubNoSdk;
        SignatureState signature = SignatureState::eNotFound;
        std::string backendText, runtimePath, runtimeSearched, runtimeVersion, signatureText, lastResult;
        uint64_t evalCount = 0;
    };
    void Initialize(UINT displayWidth, UINT displayHeight);
    // After the previous frame's COMPLETE fence, including Present, before
    // new command recording. 310.8.0 reads tuning at feature creation.
    void PrepareFrameGPUIdle();
    void OnDisplayResolution(UINT displayWidth, UINT displayHeight); // GPU idle
    void Shutdown(ID3D12Device* device); // GPU idle, before Streamline shutdown
    // Color: opaque, display-encoded R8G8B8A8_UNORM in COPY_SOURCE.
    // Guides: reverse-Z R32_FLOAT depth; jitter-free current-to-previous
    // R16G16_FLOAT motion in render pixels. Both render-resolution, in UAV.
    // Returns true only when Output() contains this frame in COPY_SOURCE.
    // Caller rebinds its descriptor heaps afterwards, even on failure.
    bool Evaluate(ID3D12GraphicsCommandList*, ID3D12Device*, ID3D12Resource* finalColor,
                  UINT finalColorSub, ID3D12Resource* depth, ID3D12Resource* motion,
                  UINT renderWidth, UINT renderHeight);
    void ForceReset() { m_forceReset = true; }
    ID3D12Resource* Output() const { return m_output.Get(); }
    DLSSNRSettings settings;
    const Status& GetStatus() const { return m_status; }
private:
    void DiscoverRuntime();
    void Fail(const char* stage, uint32_t result);
    void ReleaseFeature();
    bool EnsureTextures(ID3D12Device*);
    dlssnr::Tuning Tuning() const;
    Status m_status;
    std::wstring m_runtimeFile;
    UINT m_width = 0, m_height = 0;
    UINT m_guideWidth = 0, m_guideHeight = 0;
    ComPtr<ID3D12Resource> m_input, m_output;
    D3D12_RESOURCE_STATES m_inputState = D3D12_RESOURCE_STATE_COPY_DEST;
    D3D12_RESOURCE_STATES m_outputState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    dlssnr::Context* m_context = nullptr;
    dlssnr::Tuning m_tuning;
    bool m_featureCreated = false;
    bool m_forceReset = true;
    bool m_prevEnabled = false;
};
