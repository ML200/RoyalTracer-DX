#ifndef CUMULUS_CACHE_V8
#define CUMULUS_CACHE_V8
#include "CumulusLayout.h"
Texture3D<float4> g_cumulusNoise : register(t52);
Texture3D<float2> g_cumulusLight : register(t53);
Texture2DArray<float4> g_cumulusEnvironment : register(t54);
Texture2D<float4> g_cumulusAmbient : register(t55);
RWByteAddressBuffer g_cumulusQueries : register(u32);

uint CumulusQueryAddress(uint2 p) { return (p.y*IMG_W+p.x)*CUMULUS_QUERY_BYTES; }
// One owner per pixel. A reservoir bounds the compute work even with many PT samples.
// The selected throughput is multiplied by count/N in the resolve, so every deferred
// miss contributes in expectation. No angular quantization or negative cache correction.
void CumulusQueueMiss(uint2 p,float3 origin,float3 direction,float3 weight,uint flags,uint seed)
{
    if(!any(weight>0.0f)) return;
    uint a=CumulusQueryAddress(p), count=g_cumulusQueries.Load(a)+1u;
    g_cumulusQueries.Store(a,count);
    if(RandomFloatSingle(seed)*float(count)>=1.0f) return;
    g_cumulusQueries.Store3(a+4,uint3(flags,0,0));
    g_cumulusQueries.Store4(a+16,asuint(float4(origin,0)));
    g_cumulusQueries.Store4(a+32,asuint(float4(direction,0)));
    g_cumulusQueries.Store4(a+48,asuint(float4(weight,0)));
}

// Shared, planet-relative cache frame. World kilometres never enter a projection matrix.
float3 CumulusCameraPlanet()
{
    return WorldToPlanet(mul(viewI, float4(0, 0, 0, 1)).xyz + sceneOriginWorld);
}
void CumulusFrame(out float3 center, out float3 up, out float3 east, out float3 north)
{
    up = normalize(CumulusCameraPlanet());
    east = normalize(cross(abs(up.z) < 0.95f ? float3(0,0,1) : float3(1,0,0), up));
    north = cross(up, east);
    center = up * (ATMOS_BOTTOM_RADIUS + cloudBaseKm);
}
float CumulusLightSpanKm()
{
    // An orbital observer needs a wider, coarser footprint, not per-sample shadow marches everywhere.
    float height=max(0.0f,length(CumulusCameraPlanet())-ATMOS_BOTTOM_RADIUS-cloudBaseKm);
    return max(CUMULUS_LIGHT_SPAN_KM,height*4.0f);
}
float3 CumulusLightPosition(uint3 cell)
{
    float3 C, U, E, N; CumulusFrame(C,U,E,N);
    float3 uv = (float3(cell) + 0.5f) / float3(CUMULUS_LIGHT_XZ,CUMULUS_LIGHT_Y,CUMULUS_LIGHT_XZ);
    float3 radial = normalize(C + ((uv.x-.5f)*E + (uv.z-.5f)*N)*CumulusLightSpanKm());
    return radial * (ATMOS_BOTTOM_RADIUS + cloudBaseKm + uv.y*cloudThicknessKm);
}
float3 CumulusLightUv(float3 P)
{
    float3 C,U,E,N; CumulusFrame(C,U,E,N);
    float3 radial=normalize(P);
    // Inverse of the gnomonic projection used when baking, including wide orbital footprints.
    float3 D = radial*((ATMOS_BOTTOM_RADIUS+cloudBaseKm)/max(dot(radial,U),.001f))-C;
    float span=CumulusLightSpanKm();
    return float3(dot(D,E)/span+.5f,
        (length(P)-ATMOS_BOTTOM_RADIUS-cloudBaseKm)/cloudThicknessKm,
        dot(D,N)/span+.5f);
}
float2 CumulusEnvUv(float3 direction)
{
    float3 C,U,E,N; CumulusFrame(C,U,E,N);
    float3 d = float3(dot(direction,E),dot(direction,U),dot(direction,N));
    float el = asin(clamp(d.y,-1.0f,1.0f));
    return float2(atan2(d.z,d.x)/(2.0f*PI)+.5f,
        .5f+.5f*sign(el)*sqrt(abs(el)*(2.0f/PI)));
}
float3 CumulusEnvDirection(float2 uv)
{
    float3 C,U,E,N; CumulusFrame(C,U,E,N);
    float az=(uv.x-.5f)*2.0f*PI, y=uv.y*2.0f-1.0f;
    float el=y*abs(y)*(.5f*PI);
    return cos(el)*(cos(az)*E+sin(az)*N)+sin(el)*U;
}
float4 CumulusEnvironment(float3 direction, uint layer)
{
    // Wrap azimuth only. Explicit four texels also avoid filtering across array layers.
    float2 xy=CumulusEnvUv(direction)*float2(CUMULUS_ENV_W,CUMULUS_ENV_H)-.5f;
    xy.y=clamp(xy.y,0.0f,float(CUMULUS_ENV_H-1));
    int2 p=int2(floor(xy)); float2 f=frac(xy);
    int x0=(p.x+int(CUMULUS_ENV_W))%int(CUMULUS_ENV_W), x1=(x0+1)%int(CUMULUS_ENV_W);
    int y1=min(p.y+1,int(CUMULUS_ENV_H)-1);
    return lerp(lerp(g_cumulusEnvironment.Load(int4(x0,p.y,layer,0)),g_cumulusEnvironment.Load(int4(x1,p.y,layer,0)),f.x),
        lerp(g_cumulusEnvironment.Load(int4(x0,y1,layer,0)),g_cumulusEnvironment.Load(int4(x1,y1,layer,0)),f.x),f.y);
}
float CumulusSunVisibility(float3 P, float3 sunDir)
{
    if (cloudEnabled < .5f) return 1.0f;
    float base=ATMOS_BOTTOM_RADIUS+cloudBaseKm;
    if (length(P)>base+cloudThicknessKm) return 1.0f;
    // Ground receivers project to the cloud base along the sun, not straight up.
    if (length(P)<base) {
        float a,b;
        if (!RaySphereIntersect(P,sunDir,base,a,b) || b<=0) return 1.0f;
        P += sunDir*b;
    }
    float3 uv=CumulusLightUv(P);
    if (any(uv.xz<0.0f)||any(uv.xz>1.0f)) return 1.0f;
    float visibility=exp(-g_cumulusLight.SampleLevel(g_sampler_LUT,saturate(uv),0).x);
    float border=min(min(uv.x,1.0f-uv.x),min(uv.z,1.0f-uv.z));
    return lerp(1.0f,visibility,smoothstep(0.0f,.08f,border));
}
#endif
