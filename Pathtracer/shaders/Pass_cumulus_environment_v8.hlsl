#define COMPUTE_PASS
#include "Includes_v8.hlsli"
#include "CumulusRender_v8.hlsli"
RWTexture2DArray<float4> g_cumulusEnvironmentOut : register(u30);
[numthreads(64,1,1)]
void main(uint3 id:SV_DispatchThreadID)
{
    uint i=id.x; if(i>=CUMULUS_ENV_W*CUMULUS_ENV_H) return;
    uint2 p=uint2(i%CUMULUS_ENV_W,i/CUMULUS_ENV_W); gDispatchIdx=uint3(p,0);
    SetSkyObserver(InitOrigin()+sceneOriginWorld);
    SunState sun=ComputeSunStateInline();
    float3 V=CumulusEnvDirection((float2(p)+.5f)/float2(CUMULUS_ENV_W,CUMULUS_ENV_H));
    // Fresh integration samples; the broad cache is reserved for reused diffuse/GI directions.
    uint hash=initRandomData(p,uint2(0,0),(uint)time,97u);
    float sampleJitter=(float(hash&65535u)+.5f)/65536.0f;
    float y=(float(p.y)+.5f)/float(CUMULUS_ENV_H)*2.0f-1.0f;
    float pixelAngle=max(2.0f*PI*cos(y*abs(y)*.5f*PI)/float(CUMULUS_ENV_W),
        2.0f*PI*abs(y)/float(CUMULUS_ENV_H));
    CumulusResult r=IntegrateCumulus(V,sun.dirWS,-1.0f,24u,sampleJitter,false,pixelAngle,true);
    float3 sky=r.radiance+EvaluateSkyBackgroundBehind(V,sun,r.hitPlanet,r.radiance)*r.transmittance;
    g_cumulusEnvironmentOut[uint3(p,0)]=float4(min(max(sky,0.0f),65000.0f),r.cloudT);
}
