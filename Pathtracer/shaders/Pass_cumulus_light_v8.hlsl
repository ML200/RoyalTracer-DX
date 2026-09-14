#define COMPUTE_PASS
#include "Includes_v8.hlsli"
#include "CumulusDensity_v8.hlsli"
RWTexture3D<float2> g_cumulusLightOut : register(u29);
groupshared float columnExtinction[64];
[numthreads(64,1,1)]
void main(uint3 id:SV_DispatchThreadID)
{
    uint i=id.x;
    const uint nearCount=CUMULUS_LIGHT_XZ*CUMULUS_LIGHT_Y*CUMULUS_LIGHT_XZ;
    if(i>=nearCount+CUMULUS_FAR_LIGHT_XZ*CUMULUS_FAR_LIGHT_Y*CUMULUS_FAR_LIGHT_XZ) return;
    uint cascade=i>=nearCount ? 1u : 0u;
    i-=cascade*nearCount;
    uint xz=cascade==0u ? CUMULUS_LIGHT_XZ : CUMULUS_FAR_LIGHT_XZ;
    uint y=cascade==0u ? CUMULUS_LIGHT_Y : CUMULUS_FAR_LIGHT_Y;
    uint3 cell=uint3((i/y)%xz,i%y,i/(xz*y));
    uint3 p=cell+uint3(0,0,cascade*CUMULUS_LIGHT_XZ);
    SetSkyObserver(InitOrigin()+sceneOriginWorld);
    float3 P=CumulusLightPosition(cell,cascade),L=ComputeSunStateInline().dirWS;

    uint lane=id.x%64u;
    columnExtinction[lane]=CumulusSampleMaterial(P,.1f,false,true).density*cloudExtinction*cloudThicknessKm/float(y);
    GroupMemoryBarrierWithGroupSync();
    float skyTau=.5f*columnExtinction[lane];
    [loop] for(uint h=cell.y+1u;h<y;h++)skyTau+=columnExtinction[lane-cell.y+h];

    float r=length(P),highest=r+cloudThicknessKm/float(y)+.24f;
    float angularPad=(1.415f*CumulusLightSpanKm(cascade)/float(xz)+.24f)
        /(ATMOS_BOTTOM_RADIUS+cloudBaseKm)+1e-5f;
    float highHorizon=-sqrt(max(0.0f,1.0f-ATMOS_BOTTOM_RADIUS*ATMOS_BOTTOM_RADIUS/(highest*highest)));
    bool traceSun=dot(P,L)/r+angularPad+sin(SUN_ANGULAR_DEG*.5f*DEG2RAD)>=highHorizon;
    float tau=CumulusTraceOpticalDepth(P,VisibleSunDirection(P,L),20u,.5f,traceSun,false).x;

    g_cumulusLightOut[p]=float2(tau,min(skyTau,65000.0f));
}
