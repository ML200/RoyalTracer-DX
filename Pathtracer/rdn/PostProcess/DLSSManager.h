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
    sl::DLSSMode m_activeMode = sl::DLSSMode::eOff;
    bool m_forceReset = false;

    XMMATRIX m_dlssPrevView = XMMatrixIdentity();
    XMMATRIX m_dlssPrevProj = XMMatrixIdentity();
};
