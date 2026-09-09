#define COMPUTE_PASS
#include "Includes_v8.hlsli"
#include "CumulusRender_v8.hlsli"
[numthreads(8,8,1)]
void main(uint3 id:SV_DispatchThreadID)
{
    if(any(id.xy>=gImageSize))return;
    gDispatchIdx=id;uint address=CumulusQueryAddress(id.xy);
    uint count=g_cumulusQueries.Load(address);if(count==0)return;
    float3 origin=asfloat(g_cumulusQueries.Load3(address+16));
    float3 direction=asfloat(g_cumulusQueries.Load3(address+32));
    float3 weight=asfloat(g_cumulusQueries.Load3(address+48))*float(count)/max(float(pt_initialSamples),1.0f);
    uint flags=g_cumulusQueries.Load(address+4),seed=initRandomData(id.xy,uint2(0,0),(uint)time,211u);
    SetSkyObserver(origin+sceneOriginWorld);SunState sun=ComputeSunStateInline();
    float roughness=saturate(asfloat(g_cumulusQueries.Load(address+8)));
    float angle=max(2.0f/(abs(projection._m11)*float(IMG_H)),roughness*roughness*.025f);
    uint steps=(uint)lerp(clamp(cloudReflectionSteps,4.0f,32.0f),4.0f,roughness);
    CumulusResult r=IntegrateCumulus(direction,sun.dirWS,-1.0f,steps,
        RandomFloatSingle(seed),(flags&1u)!=0u,angle,true);
    float3 sky=r.radiance+EvaluateSkyBackgroundBehind(direction,sun,r.hitPlanet,r.radiance)*r.transmittance;
    float3 contribution=weight*sky;
    if(all(isfinite(contribution))) gScratchPing[uint3(id.xy,2)]+=float4(contribution,0);
    // Resolve the queue in place. The shading pass consumes this actual cloud intersection.
    g_cumulusQueries.Store(address+4,flags|2u);
    g_cumulusQueries.Store4(address+16,asuint(float4(origin+direction*(r.depthKm*WORLD_UNITS_PER_KM),r.depthKm)));
    g_cumulusQueries.Store4(address+32,asuint(float4(r.normal,r.guideOpacity)));
}
