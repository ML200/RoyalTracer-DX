#define COMPUTE_PASS
#include "Includes_v8.hlsli"
RWTexture2D<float4> g_cumulusAmbientOut : register(u31);
[numthreads(64,1,1)]
void main(uint3 id:SV_DispatchThreadID)
{
    uint i=id.x;if(i>=CUMULUS_AMBIENT_W*CUMULUS_AMBIENT_H)return;
    uint2 p=uint2(i%CUMULUS_AMBIENT_W,i/CUMULUS_AMBIENT_W);
    // More angular resolution around twilight. Two hemispheric bands at eight altitudes.
    float x=2*(float(p.x)+.5f)/CUMULUS_AMBIENT_W-1,mu=x*abs(x);
    float altitude=cloudBaseKm+cloudThicknessKm*((float(p.y/2)+.5f)/8);
    g_skyObserverPlanet=float3(0,ATMOS_BOTTOM_RADIUS+altitude,0);
    float3 L=float3(sqrt(max(0,1-mu*mu)),mu,0),sum=0;
    [unroll] for(uint j=0;j<8;j++) {
        float y=(p.y&1u) ? .125f : .625f;
        float az=(j+.5f)*(2*PI/8),r=sqrt(1-y*y);
        float3 tr;bool hit;
        sum+=IntegrateScattering(float3(r*cos(az),y,r*sin(az)),L,tr,hit)*SKY_INTENSITY;
    }
    // Average incident radiance for the broad multiple-scattering field; no fixed blue floor.
    g_cumulusAmbientOut[p]=float4(sum/8,1);
}
