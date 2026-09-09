#define COMPUTE_PASS
#include "Includes_v8.hlsli"
#include "CumulusRender_v8.hlsli"
// Independent stable density quadrature; colour uses temporally changing nodes.
// Only guide outputs are consumed, so DXC removes the radiance integration.
[numthreads(8,8,1)]
void main(uint3 id:SV_DispatchThreadID)
{
    if(any(id.xy>=uint2(IMG_W,IMG_H)))return;
    gDispatchIdx=id;
    uint pixelIdx=MapPixelID(float2(IMG_W,IMG_H),id.xy);
    uint seed=initRandomData(id.xy,uint2(0,0),(uint)time,71u);
    float3 O,V;InitCameraRayDoF(id.xy,uint2(IMG_W,IMG_H),seed,O,V);
    SetSkyObserver(O+sceneOriginWorld);
    if(SkyObserverIsUnderground())return; // Primary pass already cleared both slots.
    bool mesh=load_instID(g_sample_current,pixelIdx)!=0xFFFFFFFFu;
    float limit=mesh ? length(load_x1(g_sample_current,pixelIdx)-O)/WORLD_UNITS_PER_KM : -1.0f;
    CumulusResult r=IntegrateCumulus(V,ComputeSunStateInline().dirWS,limit,(uint)cloudViewSteps,.5f,true,
        2/(abs(projection._m11)*IMG_H));
    gScratchPing[uint3(id.xy,CUMULUS_NORMAL_SLOT)]=float4(r.normal,r.guideOpacity);
    gScratchPing[uint3(id.xy,CUMULUS_DEPTH_SLOT)]=float4(r.depthKm,r.deviationKm,V.x,V.z);
}
