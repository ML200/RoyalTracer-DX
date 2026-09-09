#define COMPUTE_PASS
#include "Includes_v8.hlsli"
#include "CumulusRender_v8.hlsli"
RWByteAddressBuffer probeOutput : register(u63);
[numthreads(8,8,1)]
void integrationProbe(uint3 p:SV_DispatchThreadID)
{
    if(any(p.xy>=gImageSize))return;
    const float3 cases[8]={float3(0,0,1),float3(1,1,2),float3(100,100,1),
        float3(0,4,2),float3(4,0,2),float3(.0001f,.0001f,1),float3(.1f,10,3),float3(10,.1f,3)};
    float3 c=cases[p.y%8];float opacity=1-exp(-.5f*(c.x+c.y)*c.z);
    float u=(float(p.x)+.5f)/float(IMG_W);
    float t=CumulusCellEvent(c.x,c.y,c.z,opacity,u);
    float2 m=CumulusCellMoments(c.x,c.y,c.z,opacity,0);
    uint a=(p.y*IMG_W+p.x)*64;
    probeOutput.Store4(a,asuint(float4(c,opacity)));
    probeOutput.Store4(a+16,asuint(float4(t,m,u)));
    // Include the exactly opposed sun direction: its lift vector degenerates.
    float radius=ATMOS_BOTTOM_RADIUS+cloudBaseKm+cloudThicknessKm*(float(p.y)+.5f)/IMG_H;
    float horizon=-sqrt(max(0.0f,1-ATMOS_BOTTOM_RADIUS*ATMOS_BOTTOM_RADIUS/(radius*radius)));
    float mu=horizon+(2*u-1)*2*sin(SUN_ANGULAR_DEG*.5f*DEG2RAD);
    float3 P=float3(0,radius,0),L=float3(sqrt(1-mu*mu),mu,0);
    float3 visible=VisibleSunDirection(P,L);
    float fraction=SunDiskFractionAboveHorizon(mu,horizon);
    float3 night=VisibleSunDirection(P,float3(0,-1,0));
    probeOutput.Store4(a+32,asuint(float4(fraction,visible.y,horizon,length(visible))));
    probeOutput.Store4(a+48,asuint(float4(night,mu)));
}

[numthreads(8,8,1)]
void airQuadratureProbe(uint3 p:SV_DispatchThreadID)
{
    if(any(p.xy>=gImageSize))return;
    float scale=exp2(float(p.y%8)-5);
    float3 extinction=float3(.1f,.35f,1)*scale;
    float ds=10,u=(float(p.x)+.5f)/IMG_W,distance;float3 weight;
    AtmosphereSourceQuadrature(extinction,ds,u,distance,weight);
    uint a=(p.y*IMG_W+p.x)*64;
    probeOutput.Store4(a,asuint(float4(weight,distance)));
    probeOutput.Store4(a+16,asuint(float4(extinction,ds)));
    probeOutput.Store4(a+32,0);probeOutput.Store4(a+48,0);
}

// Shoot at the actual solar disk from a grid of ground observers. A separate,
// dense midpoint integral identifies fully covered rays and clear/thin controls.
[numthreads(8,8,1)]
void sunOcclusionProbe(uint3 p:SV_DispatchThreadID)
{
    if(any(p.xy>=gImageSize))return;
    gDispatchIdx=p;
    float2 offset=((float2(p.xy)+.5f)/float2(gImageSize)-.5f)*8000.0f;
    SetSkyObserver(InitOrigin()+sceneOriginWorld+float3(offset.x,0,offset.y));
    SunState sun=ComputeSunStateInline();
    float3 O=g_skyObserverPlanet,V=sun.dirWS;
    CumulusResult r=IntegrateCumulus(V,V,-1.0f,(uint)cloudViewSteps,.5f,false,.0001f);
    float3 disk=EvaluateSunUnattenuated(V);
    float tau=0,inner0,inner1,outer0,outer1;
    if(cloudEnabled>.5f && cloudCoverage>0 &&
        RaySphereIntersect(O,V,ATMOS_BOTTOM_RADIUS+cloudBaseKm,inner0,inner1) &&
        RaySphereIntersect(O,V,ATMOS_BOTTOM_RADIUS+cloudBaseKm+cloudThicknessKm,outer0,outer1)) {
        float begin=max(0.0f,inner1),ds=max(0.0f,outer1-begin)/512.0f;
        [loop] for(uint i=0;i<512;i++) {
            float t=begin+(float(i)+.5f)*ds;
            tau+=cloudExtinction*CumulusDensity(O+V*t,max(t*.0001f,.001f),true)*ds;
        }
    }
    uint a=(p.y*IMG_W+p.x)*64;
    probeOutput.Store4(a,asuint(float4(disk*r.transmittance,tau)));
    probeOutput.Store4(a+16,asuint(float4(disk,r.cloudT)));
    probeOutput.Store4(a+32,asuint(float4(r.transmittance,0)));
    probeOutput.Store4(a+48,0);
}
