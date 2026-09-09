#ifndef CUMULUS_RENDER_V8
#define CUMULUS_RENDER_V8
#include "CumulusDensity_v8.hlsli"
struct CumulusResult {
    float3 radiance;
    float3 transmittance;
    float cloudT;
    float guideOpacity;
    float depthKm;
    float deviationKm;
    float3 normal;
    bool hitPlanet;
};
float CumulusHG(float mu,float g)
{
    float d=max(1e-5f,1.0f+g*g-2.0f*g*mu);
    return (1.0f-g*g)/(4.0f*PI*d*sqrt(d));
}
float CumulusPhase(float mu)
{
    // HG-Draine blend, representative 20 micron liquid droplets (Jendersie/d'Eon 2023).
    const float g=.5567f, a=21.9955f, w=.4824f;
    float draine=CumulusHG(mu,g)*(1.0f+a*mu*mu)/(1.0f+a*(1.0f+2.0f*g*g)/3.0f);
    return lerp(CumulusHG(mu,.9881f),draine,w);
}
float3 CumulusAmbientBand(float sunCos,float height,uint band)
{
    float u=.5f+.5f*sign(sunCos)*sqrt(abs(sunCos));
    float y=clamp(height*8-.5f,0.0f,7.0f),lo=floor(y),hi=min(lo+1,7.0f);
    return lerp(g_cumulusAmbient.SampleLevel(g_sampler_LUT,float2(u,(2*lo+band+.5f)/CUMULUS_AMBIENT_H),0).rgb,
        g_cumulusAmbient.SampleLevel(g_sampler_LUT,float2(u,(2*hi+band+.5f)/CUMULUS_AMBIENT_H),0).rgb,frac(y));
}
float3 CumulusSource(float3 P,float3 V,float3 L,float phase,CumulusMaterial material,float shadowJitter,float inCloudKm)
{
    float2 tau=CumulusLightingDepth(P,L,shadowJitter);
    float3 sunTr=TransmittanceToSun(P,L,ATMOS_BOTTOM_RADIUS,ATMOS_TOP_RADIUS);
    float sunCos=dot(normalize(P),L);
    float cosH=-sqrt(max(0.0f,1.0f-ATMOS_BOTTOM_RADIUS*ATMOS_BOTTOM_RADIUS/dot(P,P)));
    float visible=SunDiskFractionAboveHorizon(sunCos,cosH);
    float mu=dot(V,L),direct=exp(-tau.x)*phase;
    // Nubis-inspired dimensional-profile gate and reduced extinction for higher orders.
    // This remains an approximation, but fine lobes now modulate their own light transport.
    float penetration=lerp(.25f,lerp(.25f,.055f,saturate(inCloudKm)),saturate(mu/.9f));
    float multi=2.5f*material.profile*exp(-tau.x*penetration)*CumulusHG(mu,.18f)
        +.18f*exp(-tau.x*.035f)/(4*PI);
    float height=saturate((length(P)-ATMOS_BOTTOM_RADIUS-cloudBaseKm)/cloudThicknessKm);
    float3 top=CumulusAmbientBand(sunCos,height,0u),horizon=CumulusAmbientBand(sunCos,height,1u);
    float topLum=dot(top,float3(.2126f,.7152f,.0722f)),horLum=dot(horizon,float3(.2126f,.7152f,.0722f));
    horizon*=min(1.0f,topLum/max(horLum,1e-6f));
    float3 sky=lerp(horizon*(1-.65f*cloudCoverage),top,material.height);
    float ambientGate=sqrt(saturate(1-material.profile));
    float3 ground=sunTr*(sunSunIntensity*visible*max(sunCos,0.0f)*(.2f/PI))
        *(1-.85f*cloudCoverage)*(1-material.height)*exp(-tau.y*.15f);
    float3 ambient=cloudAmbient*(sky*(.3f+.7f*exp(-tau.y*.22f))*ambientGate+ground*.25f);
    return .999f*(sunTr*(sunSunIntensity*visible)*(direct+cloudMultipleScattering*multi)+ambient);
}
void CumulusAirSegment(inout CumulusResult r,float3 O,float3 V,float3 L,float begin,float lengthKm)
{
    if(lengthKm==0.0f) return;
    g_skyObserverPlanet=O+V*begin;
    float3 tr; bool hit;
    float3 air=IntegrateScattering(V,L,tr,hit,lengthKm)*SKY_INTENSITY;
    r.radiance+=r.transmittance*air; r.transmittance*=tr; r.hitPlanet=r.hitPlanet||hit;
    g_skyObserverPlanet=O;
}
void CumulusCloudSegment(inout CumulusResult r,inout float2 moments,float3 O,float3 V,float3 L,
    float begin,float end,uint steps,float jitterValue,float pixelAngle)
{
    float phR=PhaseRayleigh(dot(V,L)),phM=PhaseMieTwoLobe(dot(V,L));
    // The cloud source uses a finite angular footprint for the very narrow forward lobe.
    float mu=dot(V,L), cone=max(.00465f,pixelAngle);
    float phase=CumulusPhase(min(mu,cos(cone*.5f)));
    // Distant, thin shell crossings need fewer samples: their projected detail is already filtered.
    uint footprintSteps=(uint)max(16.0f,(end-begin)/max(.025f,begin*pixelAngle));
    steps=min(clamp(steps,16u,160u),footprintSteps);
    float nearScale=max(2.0f,begin*.1f), spacing=log2(1.0f+(end-begin)/nearScale);
    uint randomState=asuint(jitterValue)^asuint(begin)^0x51ED270Bu;
    float inCloudKm=0;
    [loop] for(uint i=0;i<steps;i++) {
        float u0=float(i)/steps,u1=float(i+1)/steps;
        // Exponential spacing retains nearby billows even when a horizon ray spans hundreds of km.
        float s0=nearScale*(exp2(spacing*u0)-1.0f),s1=nearScale*(exp2(spacing*u1)-1.0f);
        float ds=s1-s0, t=begin+lerp(s0,s1,RandomFloatSingle(randomState));
        float3 P=O+V*t;
        float footprint=max(t*pixelAngle,.001f);
        CumulusMaterial material=CumulusSampleMaterial(P,footprint,true);
        float sigma=cloudExtinction*material.density;
        inCloudKm+=ds*material.density;
        float ct=exp(-sigma*ds), weight=r.cloudT*(1.0f-ct);
        moments+=weight*float2(t,t*t); r.cloudT*=ct;
        MediumSample med=SampleMedium(max(0.0f,length(P)-ATMOS_BOTTOM_RADIUS));
        float3 ext=med.extinction+sigma;
        float3 sunTr=TransmittanceToSun(P,L,ATMOS_BOTTOM_RADIUS,ATMOS_TOP_RADIUS);
        float sunCos=dot(normalize(P),L);
        float cosH=-sqrt(max(0.0f,1.0f-ATMOS_BOTTOM_RADIUS*ATMOS_BOTTOM_RADIUS/dot(P,P)));
        float visible=SunDiskFractionAboveHorizon(sunCos,cosH);
        float3 airRate=((med.scatterR*phR+med.scatterM*phM)*sunTr*visible*ATMOS_MULTI_SCATTER_FACTOR
            +(med.scatterR+med.scatterM)*MultiScatterPsi(length(P),sunCos))*ATMOS_SOLAR_IRRADIANCE*SKY_INTENSITY;
        float3 source=airRate;
        if(sigma>1e-5f) source+=sigma*CumulusSource(P,V,L,phase,material,RandomFloatSingle(randomState),inCloudKm);
        float3 tr=exp(-ext*ds);
        r.radiance+=r.transmittance*source*(1.0f-tr)/max(ext,1e-8f);
        r.transmittance*=tr;
        if(max(r.transmittance.x,max(r.transmittance.y,r.transmittance.z))<.001f) break;
    }
}
// Deterministic extinction moments: RR sees one representative volume surface, with
// a depth spread that exposes its uncertainty. Do not drive guides with noisy light samples.
void CumulusGuideSegment(inout float trans,inout float2 moments,float3 O,float3 V,float2 interval,float pixelAngle)
{
    if(interval.y<=interval.x) return;
    const uint steps=48;
    float scale=max(1.0f,interval.x*.1f),spacing=log2(1+(interval.y-interval.x)/scale);
    [loop] for(uint i=0;i<steps;i++) {
        float a=scale*(exp2(spacing*(float(i)/steps))-1),b=scale*(exp2(spacing*(float(i+1)/steps))-1);
        float t=interval.x+(a+b)*.5f;
        float ct=exp(-cloudExtinction*CumulusDensity(O+V*t,t*pixelAngle,true)*(b-a));
        float weight=trans*(1-ct);moments+=weight*float2(t,t*t);trans*=ct;
        if(trans<.001f) break;
    }
}
// Intersect both parts of the spherical shell. A downward view leaves the cloud at its
// base, not at the ground: spending volume samples on that empty gap causes visible bands.
CumulusResult IntegrateCumulus(float3 V,float3 L,float limitKm,uint steps,float jitterValue,bool guides,float pixelAngle)
{
    CumulusResult r=(CumulusResult)0; r.transmittance=1.0f; r.cloudT=1.0f;
    float3 O=g_skyObserverPlanet;
    float outer0,outer1;
    float top=ATMOS_BOTTOM_RADIUS+cloudBaseKm+cloudThicknessKm;
    bool layer=cloudEnabled>.5f && cloudCoverage>0.0f && RaySphereIntersect(O,V,top,outer0,outer1) && outer1>0.0f;
    if(!layer) {
        r.radiance=IntegrateScattering(V,L,r.transmittance,r.hitPlanet,limitKm)*SKY_INTENSITY;
        return r;
    }
    float begin=max(0.0f,outer0),end=outer1;
    if(limitKm>=0.0f) end=min(end,limitKm);
    float a,b;
    if(RaySphereIntersect(O,V,ATMOS_BOTTOM_RADIUS,a,b) && a>0.0f && a<end) {
        end=a; r.hitPlanet=true;
    }
    float2 segments[2]; segments[0]=float2(begin,end); segments[1]=0.0f;
    if(RaySphereIntersect(O,V,ATMOS_BOTTOM_RADIUS+cloudBaseKm,a,b) && b>begin && a<end) {
        segments[0].y=min(end,a);
        segments[1]=float2(max(begin,b),end);
    }
    float cursor=0.0f; float2 moments=0.0f;
    [unroll] for(uint s=0;s<2;s++) {
        float2 interval=segments[s];
        if(interval.y<=interval.x) continue;
        CumulusAirSegment(r,O,V,L,cursor,interval.x-cursor);
        CumulusCloudSegment(r,moments,O,V,L,interval.x,interval.y,steps,jitterValue,pixelAngle);
        cursor=interval.y;
        if(max(r.transmittance.x,max(r.transmittance.y,r.transmittance.z))<.001f) break;
    }
    if(limitKm<0.0f || limitKm>cursor)
        CumulusAirSegment(r,O,V,L,cursor,limitKm<0.0f ? -1.0f : limitKm-cursor);
    if(guides) {
        float guideT=1; moments=0;
        [unroll] for(uint g=0;g<2;g++) CumulusGuideSegment(guideT,moments,O,V,segments[g],pixelAngle);
        r.guideOpacity=1-guideT;
    } else r.guideOpacity=1-r.cloudT;
    float opacity=r.guideOpacity;
    if(opacity>1e-5f) {
        r.depthKm=moments.x/opacity;
        r.deviationKm=sqrt(max(0.0f,moments.y/opacity-r.depthKm*r.depthKm));
        if(guides) r.normal=CumulusNormal(O+V*r.depthKm,r.depthKm);
    }
    return r;
}
#endif
