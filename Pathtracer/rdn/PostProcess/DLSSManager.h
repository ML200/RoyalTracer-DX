#pragma once
// ═══════════════════════════════════════════════════════════════════
// PostProcess/DLSSManager.h
// ═══════════════════════════════════════════════════════════════════

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

    // Returns true if resources were recreated (caller should rebuild SRV heap entries)
    bool UpdateMode(ID3D12Device* device);

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
        uint32_t jitterFrameIndex);

    // ── Resolution accessors ──────────────────────────────────────
    UINT RenderWidth()  const { return m_renderWidth; }
    UINT RenderHeight() const { return m_renderHeight; }
    UINT DisplayWidth() const { return m_displayWidth; }
    UINT DisplayHeight()const { return m_displayHeight; }

    // ── Resource accessors for SRV/UAV heap setup ─────────────────
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

    // Editor-exposed settings
    sl::DLSSMode mode = sl::DLSSMode::eDLAA;
    float fovDegrees  = 60.0f;

private:
    void CreateInputTextures(ID3D12Device* device);
    void ComputeRenderResolution();

    ComPtr<ID3D12Resource> m_input, m_depth, m_mvec, m_normals;
    ComPtr<ID3D12Resource> m_diffuseAlbedo, m_output;
    ComPtr<ID3D12Resource> m_specAlbedo, m_roughness, m_specMvec, m_specHitDist;
    ComPtr<ID3D12Resource> m_transparency, m_colorBeforeTrans;
    ComPtr<ID3D12Resource> m_biasHint;

    ResourceStateTracker m_state;

    UINT m_displayWidth  = 0, m_displayHeight = 0;
    UINT m_renderWidth   = 0, m_renderHeight  = 0;
    sl::DLSSMode m_activeMode = sl::DLSSMode::eOff;  // mode resources were built for

    XMMATRIX m_dlssPrevView = XMMatrixIdentity();
    XMMATRIX m_dlssPrevProj = XMMatrixIdentity();
};
