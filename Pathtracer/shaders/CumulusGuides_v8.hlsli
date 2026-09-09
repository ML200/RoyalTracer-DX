#ifndef CUMULUS_GUIDES_V8
#define CUMULUS_GUIDES_V8
#include "CumulusGuideMath_v8.hlsli"
void ApplyCumulusGuides(uint2 pixel,uint pixelIdx,float3 camera)
{
    if(cloudEnabled<.5f) return;
    float4 normalOpacity=gScratchPing[uint3(pixel,CUMULUS_NORMAL_SLOT)];
    float4 depth=gScratchPing[uint3(pixel,CUMULUS_DEPTH_SLOT)];
    float opacity=saturate(normalOpacity.w);
    uint seed=initRandomData(pixel,uint2(0,0),(uint)time,71u);
    float3 origin,dir; InitCameraRayDoF(pixel,uint2(IMG_W,IMG_H),seed,origin,dir);
    float3 P=origin+dir*(depth.x*WORLD_UNITS_PER_KM);
    float3 wind=float3(cloudWindX,0,cloudWindZ)*cloudDeltaSeconds;
    float2 cloudMV=CumulusMotion(P,P-wind);
    bool mesh=load_instID(g_sample_current,pixelIdx)!=0xFFFFFFFFu;

    // RR exposes one primary surface record, not multiple volume layers. Choose a coherent
    // record when the volume dominates; never blend two positions/normals into a fake surface.
    bool cloudGuide=opacity>=(mesh ? cloudGuideThreshold : .015f) && depth.x>0.0f;
    if(cloudGuide) {
        float3 n=normalOpacity.xyz;
        n=dot(n,n)>1e-10f ? normalize(n) : -dir;
        g_dlssDepth[pixel]=DLSS_GuideDepthFromWorldPos(P);
        g_dlssMVec[pixel]=cloudMV;
        g_dlssNormals[pixel]=float4(n,1.0f);
        g_dlssDiffuseAlbedo[pixel]=float4(.999f,.999f,.999f,1.0f);
        g_dlssSpecularAlbedo[pixel]=0.0f;
        g_dlssRoughness[pixel]=1.0f;
        g_dlssSpecMVec[pixel]=cloudMV;
        g_dlssSpecHitDist[pixel]=0.0f;
    }

    if(mesh && !cloudGuide) {
        float4 probe=gScratchPing[uint3(pixel,4)];
        if(asuint(probe.w)==0xFFFFFFFFu) {
            float3 surface=load_x1(g_sample_current,pixelIdx);
            SurfaceVertex sv=BuildVertex(g_sample_current,pixelIdx,surface,camera);
            if(sv.Pr<DLSS_SPEC_ROUGHNESS_THRESHOLD) {
                float3 ray=reflect(normalize(surface-camera),sv.n_s);
                uint address=CumulusQueryAddress(pixel);
                uint flags=g_cumulusQueries.Load(address+4);
                float4 resolved=asfloat(g_cumulusQueries.Load4(address+16));
                float confidence=asfloat(g_cumulusQueries.Load(address+44));
                // Only an actual first-reflection cloud sample can supply a finite guide.
                // A camera-centred environment lookup cannot establish reflection parallax.
                if((flags&3u)==3u && confidence>.1f && resolved.w>0) {
                    float3 hit=resolved.xyz;
                    float distance=length(hit-surface);
                    float3 virtualHit=hit-2.0f*dot(hit-surface,sv.n_s)*sv.n_s;
                    float3 oldHit=hit-wind;
                    uint instance=load_instID(g_sample_current,pixelIdx);
                    float3 localN=WorldToObjectNrm(instance,sv.n_s);
                    float3 localT=normalize(cross(localN,abs(localN.y)<.9f ? float3(0,1,0) : float3(1,0,0)));
                    float3 localB=cross(localN,localT);
                    float3 oldT=mul((float3x3)instanceProps[instance].prevObjectToWorld,localT);
                    float3 oldB=mul((float3x3)instanceProps[instance].prevObjectToWorld,localB);
                    float3 oldCross=cross(oldT,oldB);
                    float3 oldN=dot(oldCross,oldCross)>1e-10f ? normalize(oldCross) : sv.n_s;
                    float3 oldSurface=mul(instanceProps[instance].prevObjectToWorld,float4(WorldToObjectPos(instance,surface),1));
                    float3 oldVirtual=oldHit-2.0f*dot(oldHit-oldSurface,oldN)*oldN;
                    g_dlssSpecMVec[pixel]=CumulusMotion(virtualHit,oldVirtual);
                    g_dlssSpecHitDist[pixel]=min(distance,DLSS_SPEC_HIT_MAX);
                }
            }
        }
    }
    // Same inspector channels used by normal RR frames; no separate history pass.
    float3 debugColor=0.0f;
    if(cloudDebugView>0.5f) {
        uint mode=(uint)cloudDebugView;
        if(mode==1u) debugColor=opacity;
        if(mode==2u) debugColor=opacity>.01f ? normalOpacity.xyz*.5f+.5f : 0.0f;
        if(mode==3u) debugColor=opacity>.01f ? 1.0f-exp(-depth.x*.1f) : 0.0f;
        if(mode==4u) debugColor=float3(cloudMV*.05f+.5f,opacity);
        if(mode==5u) debugColor=float3(saturate(depth.y/max(depth.x,.001f)*4.0f),opacity,cloudGuide ? 1.0f : 0.0f);
        g_dlssInput[pixel]=float4(DlssEncode(debugColor),1.0f);
    }
}
#endif
