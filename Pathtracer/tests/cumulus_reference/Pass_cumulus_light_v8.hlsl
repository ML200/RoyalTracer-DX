#define COMPUTE_PASS
#include "Includes_v8.hlsli"
#include "CumulusDensity_v8.hlsli"
RWTexture3D<float2> g_cumulusLightOut : register(u29);
[numthreads(64,1,1)]
void main(uint3 id:SV_DispatchThreadID)
{
    uint i=id.x;
    if(i>=CUMULUS_LIGHT_XZ*CUMULUS_LIGHT_Y*CUMULUS_LIGHT_XZ) return;
    uint3 p=uint3(i%CUMULUS_LIGHT_XZ,(i/CUMULUS_LIGHT_XZ)%CUMULUS_LIGHT_Y,i/(CUMULUS_LIGHT_XZ*CUMULUS_LIGHT_Y));
    SetSkyObserver(InitOrigin()+sceneOriginWorld);
    g_cumulusLightOut[p]=CumulusTraceOpticalDepth(CumulusLightPosition(p),ComputeSunStateInline().dirWS);
}
