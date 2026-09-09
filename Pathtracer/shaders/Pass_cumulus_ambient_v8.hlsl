#define COMPUTE_PASS
#include "Includes_v8.hlsli"
RWTexture2D<float4> g_cumulusAmbientOut : register(u31);
[numthreads(64,1,1)]
void main(uint3 id:SV_DispatchThreadID)
{
    uint i=id.x;if(i>=CUMULUS_AMBIENT_W*CUMULUS_AMBIENT_H)return;
    uint2 p=uint2(i%CUMULUS_AMBIENT_W,i/CUMULUS_AMBIENT_W);
    // Upper and lower incident hemispheres at eight altitudes. The lower
    // hemisphere includes atmospheric in-scatter up to the planet, not a
    // duplicate of a narrow upper-horizon ring. No cloud-cache feedback here.
    float x=2*(float(p.x)+.5f)/CUMULUS_AMBIENT_W-1,mu=x*abs(x);
    float altitude=cloudBaseKm+cloudThicknessKm*((float(p.y/2)+.5f)/8);
    g_skyObserverPlanet=float3(0,ATMOS_BOTTOM_RADIUS+altitude,0);
    float3 L=float3(sqrt(max(0,1-mu*mu)),mu,0),sum=0;
    [loop] for(uint j=0;j<16;j++) {
        float y=((j/8)+.5f)*.5f*((p.y&1u)?-1.0f:1.0f);
        float az=(j+.5f)*(2*PI/8),r=sqrt(1-y*y);
        float3 tr;bool hit;
        sum+=IntegrateScattering(float3(r*cos(az),y,r*sin(az)),L,tr,hit)*SKY_INTENSITY;
    }
    // Average incident radiance for the broad multiple-scattering field; no fixed blue floor.
    g_cumulusAmbientOut[p]=float4(sum/16,1);
}
