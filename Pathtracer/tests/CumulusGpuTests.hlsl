#define COMPUTE_PASS
#ifndef CUMULUS_REFERENCE
#define CUMULUS_REFERENCE 0
#endif
#include "Includes_v8.hlsli"
#include "CumulusRender_v8.hlsli"
#include "CumulusGuideMath_v8.hlsli"
RWByteAddressBuffer testOutput : register(u63);
// A deterministic material slice isolates geometric detail from radiance noise.
[numthreads(8,8,1)]
void probeMaterial(uint3 p:SV_DispatchThreadID)
{
    if(any(p.xy>=gImageSize))return;
    gDispatchIdx=p;
    float3 up=normalize(CumulusCameraPlanet());
    float3 X=normalize(cross(up,abs(up.z)<.9f ? float3(0,0,1):float3(1,0,0)));
    float3 Z=cross(X,up);
    float2 q=((float2(p.xy)+.5f)/float2(gImageSize)-.5f)*float2(8.0f,4.5f);
    float3 P=normalize(up*ATMOS_BOTTOM_RADIUS+X*q.x+Z*q.y)
        *(ATMOS_BOTTOM_RADIUS+cloudBaseKm+cloudDebugView*cloudThicknessKm);
    CumulusMaterial m=CumulusSampleMaterial(P,.002f,true);
    float farDensity=CumulusDensity(P,.1f*max(cloudScale,.2f),true);
    float shadowDensity=CumulusSampleMaterial(P,.1f,false,true).density;
    uint address=(p.y*gImageSize.x+p.x)*64u;
    testOutput.Store4(address,asuint(float4(m.density,m.profile,m.height,0)));
    testOutput.Store4(address+16,asuint(float4(farDensity,shadowDensity,0,0)));
    testOutput.Store4(address+32,0);testOutput.Store4(address+48,0);
}
[numthreads(8,8,1)]
void main(uint3 p:SV_DispatchThreadID)
{
    if(any(p.xy>=gImageSize)) return;
    gDispatchIdx=p;
    uint seed=123u; float3 O,V;
    InitCameraRayDoF(p.xy,gImageSize,seed,O,V);
    SetSkyObserver(O+sceneOriginWorld);
    float3 L=ComputeSunStateInline().dirWS;
#if !CUMULUS_REFERENCE
    if(cloudDebugView==-8.0f) {
        // Cached clouds may remove direct illumination of the air, but must
        // never change camera extinction or erase diffuse atmospheric light.
        float limit=1+199*float(p.x)/float(IMG_W-1);
        float3 ta,tb;bool ha,hb;
        float3 shadowed=IntegrateScattering(V,L,ta,ha,limit,32,true);
        float3 clear=IntegrateScattering(V,L,tb,hb,limit,32,false);
        uint addr=(p.y*IMG_W+p.x)*64;
        testOutput.Store4(addr,asuint(float4(shadowed,0)));
        testOutput.Store4(addr+16,asuint(float4(clear,0)));
        testOutput.Store4(addr+32,asuint(float4(ta,0)));
        testOutput.Store4(addr+48,asuint(float4(tb,0)));return;
    }
    if(cloudDebugView==-9.0f) {
        float3 up=normalize(g_skyObserverPlanet);
        float height=(float(p.y)+.5f)/IMG_H;
        float3 P=up*(ATMOS_BOTTOM_RADIUS+cloudBaseKm+height*cloudThicknessKm);
        CumulusMaterial m;m.density=.1f;m.profile=.1f;m.height=height;
        float3 a=CumulusSource(P,V,L,CumulusPhase(dot(V,L)),m,.5f,.01f,true);
        m.density=1;m.profile=1;
        float3 b=CumulusSource(P,V,L,CumulusPhase(dot(V,L)),m,.5f,10.0f,true);
        float mu=2*float(p.x)/float(IMG_W-1)-1;
        float radius=ATMOS_BOTTOM_RADIUS+height*(ATMOS_TOP_RADIUS-ATMOS_BOTTOM_RADIUS);
        float r1,mu1;
        MultiScatterLutRMuFromUnit(MultiScatterLutUnitFromRMu(radius,mu),r1,mu1);
        uint addr=(p.y*IMG_W+p.x)*64;
        testOutput.Store4(addr,asuint(float4(a,0)));
        testOutput.Store4(addr+16,asuint(float4(b,0)));
        testOutput.Store4(addr+32,asuint(float4(abs(radius-r1),abs(mu-mu1),0,0)));
        testOutput.Store4(addr+48,0);return;
    }
    if(cloudDebugView==-7.0f) {
        const uint nearCount=CUMULUS_LIGHT_XZ*CUMULUS_LIGHT_Y*CUMULUS_LIGHT_XZ;
        const uint farCount=CUMULUS_FAR_LIGHT_XZ*CUMULUS_FAR_LIGHT_Y*CUMULUS_FAR_LIGHT_XZ;
        uint pixel=p.y*IMG_W+p.x;
        uint i=pixel%(nearCount+farCount),cascade=i>=nearCount?1u:0u;
        i-=cascade*nearCount;
        uint xz=cascade?CUMULUS_FAR_LIGHT_XZ:CUMULUS_LIGHT_XZ,y=cascade?CUMULUS_FAR_LIGHT_Y:CUMULUS_LIGHT_Y;
        uint3 cell=uint3((i/y)%xz,i%y,i/(xz*y));
        uint3 texel=cell+uint3(0,0,cascade*CUMULUS_LIGHT_XZ);
        float2 tau=g_cumulusLight.Load(int4(texel,0));
        float above=0;
        float expected=.5f*CumulusSampleMaterial(CumulusLightPosition(cell,cascade),.1f,false,true).density;
        if(cell.y+1<y) {
            above=g_cumulusLight.Load(int4(texel+uint3(0,1,0),0)).y;
            expected+=.5f*CumulusSampleMaterial(CumulusLightPosition(cell+uint3(0,1,0),cascade),.1f,false,true).density;
        }
        expected*=cloudExtinction*cloudThicknessKm/float(y);
        testOutput.Store4(pixel*64u,asuint(float4(tau,above,expected)));
        testOutput.Store4(pixel*64u+16,0);testOutput.Store4(pixel*64u+32,0);testOutput.Store4(pixel*64u+48,0);return;
    }
    if(cloudDebugView==-6.0f) {
        // Sweep a finite solar disk across the geometric horizon at every cloud
        // altitude. The CPU uses zero extinction to detect planet occlusion being
        // accidentally stored as cloud optical depth, including in cached light.
        float height=(float(p.y)+.5f)/float(IMG_H);
        float radius=ATMOS_BOTTOM_RADIUS+cloudBaseKm+height*cloudThicknessKm;
        float3 up=normalize(g_skyObserverPlanet);
        float3 tangent=normalize(cross(up,abs(up.z)<.9f ? float3(0,0,1):float3(1,0,0)));
        float horizon=-sqrt(max(0.0f,1-ATMOS_BOTTOM_RADIUS*ATMOS_BOTTOM_RADIUS/(radius*radius)));
        float mu=horizon+(2*float(p.x)/float(IMG_W-1)-1)*2*sin(SUN_ANGULAR_DEG*.5f*DEG2RAD);
        float3 P=up*radius,sun=up*mu+tangent*sqrt(1-mu*mu);
        float3 visibleSun=VisibleSunDirection(P,sun);
        float2 tau=CumulusTraceOpticalDepth(P,visibleSun);
        float visibility=SunDiskFractionAboveHorizon(mu,horizon);
        float3 transmitted=TransmittanceToSun(P,visibleSun,ATMOS_BOTTOM_RADIUS,ATMOS_TOP_RADIUS)*visibility;
        CumulusMaterial m;m.density=1;m.profile=1;m.height=height;
        float3 denseSource=CumulusSource(P,-sun,sun,1/(4*PI),m,.5f,1.0f,true);
        uint address=(p.y*gImageSize.x+p.x)*64u;
        testOutput.Store4(address,asuint(float4(tau,visibility,transmitted.r)));
        testOutput.Store4(address+16,asuint(float4(denseSource,0)));
        testOutput.Store4(address+32,0);testOutput.Store4(address+48,0);return;
    }
#endif
    // Read the engine's sun direction for a real sun-facing test camera.
    if(cloudDebugView==-5.0f) {
        uint address=(p.y*gImageSize.x+p.x)*64u;
        testOutput.Store4(address,asuint(float4(L,0)));
        testOutput.Store4(address+16,0);testOutput.Store4(address+32,0);testOutput.Store4(address+48,0);return;
    }
    uint hash=initRandomData(p.xy,uint2(0,0),0u,97u);
    float sampleJitter=frac(float(hash&65535u)/65536.0f+(uint(time)%4096u)*.61803398875f);
    // Lighting study: hold density and shadow candidates fixed, while production
    // reservoir randomness keeps changing with the frame. This isolates its bias.
    if(cloudDebugView==-4.0f) sampleJitter=frac(float(hash&65535u)/65536.0f);
#if !CUMULUS_REFERENCE
    if(cloudDebugView==-11 || cloudDebugView==-12 || cloudDebugView==-13 || cloudDebugView==-14) {
        float3 tr;bool hit;
        bool reference=cloudDebugView==-12;
        float3 c=IntegrateScattering(V,L,tr,hit,-1,reference?64u:12u,true,(reference||cloudDebugView==-14)?-1.0f:sampleJitter);
        uint a=(p.y*IMG_W+p.x)*64;
        if(time>0 && cloudDebugView!=-13)c=lerp(asfloat(testOutput.Load3(a)),c,1/(time+1));
        testOutput.Store4(a,asuint(float4(c,0)));testOutput.Store4(a+16,0);
        testOutput.Store4(a+32,asuint(float4(tr,0)));testOutput.Store4(a+48,0);return;
    }
#endif
    // The harness uses debugView as a finite ray distance in km; -1 selects a sky miss.
    CumulusResult r=IntegrateCumulus(V,L,cloudDebugView<-1 ? -1.0f : cloudDebugView,(uint)cloudViewSteps,sampleJitter,CUMULUS_REFERENCE!=0,2.0f/(abs(projection._m11)*float(IMG_H)));
    uint offset=(p.y*gImageSize.x+p.x)*64u;
    float4 color=float4(r.radiance,1.0f-r.cloudT);
    if(time>0.0f && cloudDebugView!=-3.0f) color=lerp(asfloat(testOutput.Load4(offset)),color,1.0f/(time+1.0f));
    testOutput.Store4(offset,asuint(color));
    testOutput.Store4(offset+16u,asuint(float4(r.normal,r.depthKm)));
    testOutput.Store4(offset+32u,asuint(float4(r.transmittance,r.deviationKm)));
    float3 P=O+V*(r.depthKm*WORLD_UNITS_PER_KM);
    float2 mv=CumulusMotion(P,P-float3(cloudWindX,0,cloudWindZ)*cloudDeltaSeconds);
    testOutput.Store4(offset+48u,asuint(float4(DLSS_GuideDepthFromWorldPos(P),mv,r.guideOpacity)));
}

#if !CUMULUS_REFERENCE
[numthreads(8,8,1)]
void writeGuides(uint3 p:SV_DispatchThreadID)
{
    if(any(p.xy>=gImageSize)||cloudDebugView<-4.0f)return;
    gDispatchIdx=p;uint seed=123u;float3 O,V;
    InitCameraRayDoF(p.xy,gImageSize,seed,O,V);SetSkyObserver(O+sceneOriginWorld);
    CumulusResult r=IntegrateCumulus(V,ComputeSunStateInline().dirWS,cloudDebugView<-1 ? -1.0f:cloudDebugView,
        (uint)cloudViewSteps,.5f,true,2/(abs(projection._m11)*IMG_H));
    uint a=(p.y*IMG_W+p.x)*64;
    testOutput.Store4(a+16,asuint(float4(r.normal,r.depthKm)));
    testOutput.Store(a+44,asuint(r.deviationKm));
    float3 P=O+V*(r.depthKm*WORLD_UNITS_PER_KM);
    float2 mv=CumulusMotion(P,P-float3(cloudWindX,0,cloudWindZ)*cloudDeltaSeconds);
    testOutput.Store4(a+48,asuint(float4(DLSS_GuideDepthFromWorldPos(P),mv,r.guideOpacity)));
}
#endif

// Exercise the production deferred miss queue and resolve using actual shifted ray origins.
[numthreads(8,8,1)]
void seedSecondary(uint3 p:SV_DispatchThreadID)
{
    if(any(p.xy>=gImageSize))return;
    gDispatchIdx=p;uint address=CumulusQueryAddress(p.xy);
    g_cumulusQueries.Store4(address,0u);gScratchPing[uint3(p.xy,2)]=float4(.125f,.25f,.5f,0);
    // Empty pixels deliberately keep poisoned payloads: the resolve must honor the count.
    if(p.x%7u==0u) {g_cumulusQueries.Store4(address+16,0x7fc00000u);return;}
    uint seed=123u;float3 O,V;InitCameraRayDoF(p.xy,gImageSize,seed,O,V);
    O+=float3(1700,2100,1200);
    float roughness=saturate(cloudDebugView); // Harness case selector, unrelated to production debug views.
    [loop] for(uint i=0;i<pt_initialSamples;i++)
#if CUMULUS_REFERENCE
        CumulusQueueMiss(p.xy,O,V,float3(i+1,.5f*(pt_initialSamples-i),.25f),roughness<.1f ? 1u:0u,initRandomData(p.xy,uint2(0,0),(uint)time,100u+i));
#else
        CumulusQueueMiss(p.xy,O,V,float3(i+1,.5f*(pt_initialSamples-i),.25f),roughness<.1f ? 1u:0u,initRandomData(p.xy,uint2(0,0),(uint)time,100u+i),roughness);
#endif
}
[numthreads(8,8,1)]
void checkSecondary(uint3 p:SV_DispatchThreadID)
{
    if(any(p.xy>=gImageSize))return;
    gDispatchIdx=p;uint address=CumulusQueryAddress(p.xy),outAddress=(p.y*IMG_W+p.x)*64;
    float3 expected=float3(.125f,.25f,.5f);float error=0;
    if(p.x%7u!=0u) {
        uint raySeed=123u;float3 O,V;InitCameraRayDoF(p.xy,gImageSize,raySeed,O,V);O+=float3(1700,2100,1200);
        SetSkyObserver(O+sceneOriginWorld);SunState sun=ComputeSunStateInline();
        uint seed=initRandomData(p.xy,uint2(0,0),(uint)time,211u);
        float roughness=saturate(cloudDebugView);
#if CUMULUS_REFERENCE
        float angle=2/(abs(projection._m11)*IMG_H);
        CumulusResult r=IntegrateCumulus(V,sun.dirWS,-1,(uint)cloudReflectionSteps,RandomFloatSingle(seed),roughness<.1f,angle);
#else
        float angle=max(2/(abs(projection._m11)*IMG_H),roughness*roughness*.025f);
        uint steps=(uint)lerp(clamp(cloudReflectionSteps,4.0f,32.0f),4.0f,roughness);
        CumulusResult r=IntegrateCumulus(V,sun.dirWS,-1,steps,RandomFloatSingle(seed),roughness<.1f,angle,true);
#endif
        expected+=asfloat(g_cumulusQueries.Load3(address+48))*(r.radiance+EvaluateSkyBackgroundBehind(V,sun,r.hitPlanet,r.radiance)*r.transmittance);
        float4 hit=asfloat(g_cumulusQueries.Load4(address+16));
        error=length(hit.xyz-(O+V*r.depthKm*WORLD_UNITS_PER_KM));
        error+=abs(hit.w-r.depthKm)+abs(asfloat(g_cumulusQueries.Load(address+44))-r.guideOpacity);
#if !CUMULUS_REFERENCE
        error+=abs(asfloat(g_cumulusQueries.Load(address+8))-roughness);
#endif
        error+=g_cumulusQueries.Load(address)==pt_initialSamples && g_cumulusQueries.Load(address+4)==(roughness<.1f ? 3u:2u) ? 0 : 100;
    }
    float3 actual=gScratchPing[uint3(p.xy,2)].rgb;
    testOutput.Store4(outAddress,asuint(float4(actual,1)));
    testOutput.Store4(outAddress+16,asuint(float4(abs(actual-expected),error)));
    testOutput.Store4(outAddress+32,0);testOutput.Store4(outAddress+48,p.x%7u==0u ? 0u : g_cumulusQueries.Load4(address+48));
}
