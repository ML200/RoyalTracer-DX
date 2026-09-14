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
// Blend cloud phase terms for forward scattering.
float CumulusPhase(float mu)
{

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
float3 CumulusDiffuseSource(float3 P,float3 V,float3 L,float phase,out float3 directIrradiance,out float cachedDirect)
{
    float3 visibleL=VisibleSunDirection(P,L);
    float3 sunTr=TransmittanceToSun(P,visibleL,ATMOS_BOTTOM_RADIUS,ATMOS_TOP_RADIUS);
    float sunCos=dot(normalize(P),L);
    float cosH=-sqrt(max(0.0f,1.0f-ATMOS_BOTTOM_RADIUS*ATMOS_BOTTOM_RADIUS/dot(P,P)));
    float visible=SunDiskFractionAboveHorizon(sunCos,cosH);

    float2 broadTau=CumulusCachedOpticalDepth(P);
    directIrradiance=.999f*sunTr*(sunSunIntensity*visible*phase);
    cachedDirect=exp(-broadTau.x);
    float multi=0,mu=dot(V,L);
    if(visible>0.0f) {

        float attenuation=.5f,contribution=.65f,g=.35f;
        [unroll] for(uint order=0;order<5;order++) {
            multi+=contribution*exp(-broadTau.x*attenuation)*CumulusHG(mu,g);
            attenuation*=.5f;contribution*=.65f;g*=.5f;
        }
    }
    float height=saturate((length(P)-ATMOS_BOTTOM_RADIUS-cloudBaseKm)/cloudThicknessKm);
    float3 upper=CumulusAmbientBand(sunCos,height,0u),lower=CumulusAmbientBand(sunCos,height,1u);
    float columnTau=CumulusCachedOpticalDepth(normalize(P)*(ATMOS_BOTTOM_RADIUS+cloudBaseKm)).y;
    float downTau=max(0.0f,columnTau-broadTau.y);

    float2 diffuseTransmission=rcp(1.0f+sqrt(3.0f*.001f)*float2(broadTau.y,downTau));
    float3 sky=.5f*(upper*diffuseTransmission.x+lower*diffuseTransmission.y);
    float3 ground=sunTr*(sunSunIntensity*visible*max(sunCos,0.0f)*(.2f/PI))
        *(1-.85f*cloudCoverage)*(1-height)*exp(-downTau*.15f);
    float3 ambient=cloudAmbient*(sky+ground*.25f);
    return .999f*(sunTr*(sunSunIntensity*visible)*(cloudMultipleScattering*multi)+ambient);
}
float3 CumulusSource(float3 P,float3 V,float3 L,float phase,CumulusMaterial material,float shadowJitter,float inCloudKm,bool cheap)
{
    float3 directIrradiance;float cachedDirect;
    float3 diffuse=CumulusDiffuseSource(P,V,L,phase,directIrradiance,cachedDirect);
    float direct=cachedDirect;
    if(!cheap && any(directIrradiance>0))
        direct=exp(-CumulusLightingDepth(P,VisibleSunDirection(P,L),shadowJitter).x);
    return diffuse+directIrradiance*direct;
}
float3 CumulusAirSource(float3 P,float3 L,float phR,float phM,MediumSample med)
{
    float3 visibleL=VisibleSunDirection(P,L);
    float3 sunTr=TransmittanceToSun(P,visibleL,ATMOS_BOTTOM_RADIUS,ATMOS_TOP_RADIUS);
    float sunCos=dot(normalize(P),L);
    float cosH=-sqrt(max(0.0f,1.0f-ATMOS_BOTTOM_RADIUS*ATMOS_BOTTOM_RADIUS/dot(P,P)));
    float visible=SunDiskFractionAboveHorizon(sunCos,cosH);
    float cloudSun=visible>0 ? CumulusAtmosphereSunVisibility(P,visibleL):1.0f;
    float3 airRate=((med.scatterR*phR+med.scatterM*phM)*sunTr*visible*cloudSun*ATMOS_MULTI_SCATTER_FACTOR
        +(med.scatterR+med.scatterM)*MultiScatterPsi(length(P),sunCos))*ATMOS_SOLAR_IRRADIANCE*SKY_INTENSITY;
    return airRate;
}
// Integrate clear-air scattering between cloud samples.
void CumulusAirSegment(inout CumulusResult r,float3 O,float3 V,float3 L,float begin,float lengthKm,bool cheap,float jitterValue)
{
    if(lengthKm==0.0f) return;
    g_skyObserverPlanet=O+V*begin;
    float3 tr; bool hit;
    float3 air=IntegrateScattering(V,L,tr,hit,lengthKm,cheap ? 4u : (uint)ATMOS_VIEW_STEPS,true,frac(jitterValue+begin*.41421356f))*SKY_INTENSITY;
    r.radiance+=r.transmittance*air; r.transmittance*=tr; r.hitPlanet=r.hitPlanet||hit;
    g_skyObserverPlanet=O;
}

// Sample a collision distance from a linearly varying density.
float CumulusCellEvent(float sigma0,float sigma1,float ds,float opacity,float u)
{
    float tau=-log(max(1e-20f,1-u*opacity));
    float slope=(sigma1-sigma0)/max(ds,1e-8f);
    float root=sqrt(max(0.0f,sigma0*sigma0+2*slope*tau));
    return min(ds,2*tau/max(sigma0+root,1e-8f));
}
float2 CumulusCellMoments(float sigma0,float sigma1,float ds,float opacity,float start)
{

    const float u[8]={.0624886598f,.2970085304f,.6029914696f,.8375113402f,
        .9069431844f,.9330009478f,.9669990522f,.9930568156f};
    const float w[8]={.1565346803f,.2934653197f,.2934653197f,.1565346803f,
        .0173927423f,.0326072577f,.0326072577f,.0173927423f};
    float2 m=0;
    [unroll] for(uint j=0;j<8;j++) {
        float t=start+CumulusCellEvent(sigma0,sigma1,ds,opacity,u[j]);
        m+=w[j]*float2(t,t*t);
    }
    return m;
}
struct CumulusLightSample {
    float t;
    float shadowJitter;
    float3 weight;
    float mass;
};
float3 CumulusIncidentHint(float3 P,float3 L)
{
    float cosH=-sqrt(max(0.0f,1-ATMOS_BOTTOM_RADIUS*ATMOS_BOTTOM_RADIUS/dot(P,P)));
    return TransmittanceToSun(P,VisibleSunDirection(P,L),ATMOS_BOTTOM_RADIUS,ATMOS_TOP_RADIUS)
        *(sunSunIntensity*SunDiskFractionAboveHorizon(dot(normalize(P),L),cosH));
}
// March cloud density while accumulating transmittance and radiance.
void CumulusCloudSegment(inout CumulusResult r,float3 O,float3 V,float3 L,
    float begin,float end,uint steps,float jitterValue,float pixelAngle,bool cheap,float minTransmittance)
{
    float phR=PhaseRayleigh(dot(V,L)),phM=PhaseMieTwoLobe(dot(V,L));
    float mu=dot(V,L),cone=max(.00465f,pixelAngle);
    float phase=CumulusPhase(min(mu,cos(cone*.5f)));
    uint minSteps=cheap ? 4u:16u;

    float nodeCount=float(steps);
    if(!cheap)nodeCount+=32.0f*smoothstep(8.0f,40.0f,begin);
    float footprintSteps=max(float(minSteps),(end-begin)/max(.025f,begin*pixelAngle));
    nodeCount=min(clamp(nodeCount,float(minSteps),192.0f),footprintSteps);
    steps=(uint)ceil(nodeCount);

    float nodeShift=jitterValue-.5f;
    float nearScale=cheap ? max(2.0f,begin*.1f):2.0f;
    float spacing=log2(1+(end-begin)/nearScale);
    uint randomState=Hash32(asuint(jitterValue)^asuint(begin)^0x51ED270Bu);
    uint lightState=Hash32(initRandomData(gDispatchIdx.xy,uint2(0,0),(uint)time,613u)^asuint(begin));
    uint lightSamples=cheap ? 0u:(uint)clamp(cloudLightingSamples,0.0f,4.0f);

    float3 ambientHint=0,incident0=0,incident1=0,incident2=0;
    if(lightSamples>0) {
        float sunCos=dot(normalize(O+V*begin),L);
        ambientHint=.5f*cloudAmbient*(CumulusAmbientBand(sunCos,.5f,0u)+CumulusAmbientBand(sunCos,.5f,1u));
        incident0=CumulusIncidentHint(O+V*begin,L);
        incident1=CumulusIncidentHint(O+V*((begin+end)*.5f),L);
        incident2=CumulusIncidentHint(O+V*end,L);
    }
    CumulusLightSample selected[4];
    [unroll] for(uint k=0;k<4;k++)selected[k]=(CumulusLightSample)0;
    uint airState=Hash32(asuint(jitterValue)^asuint(begin)^0xA749F32Du);
    uint airSamples=cheap ? 2u:4u;
    CumulusLightSample selectedAir[4];
    [unroll] for(uint k=0;k<4;k++)selectedAir[k]=(CumulusLightSample)0;
    float t0=begin;
    float sigma0=cloudExtinction*CumulusDensity(O+V*t0,max(t0*pixelAngle,.001f),true);
    [loop] for(uint i=0;i<steps;i++) {

        float f=i+1==steps ? 1.0f:saturate((float(i+1)+nodeShift)/nodeCount);
        float t1=begin+nearScale*(exp2(spacing*f)-1);
        float ds=t1-t0;
        float sigma1=cloudExtinction*CumulusDensity(O+V*t1,max(t1*pixelAngle,.001f),true);
        float tau=.5f*(sigma0+sigma1)*ds,ct=exp(-tau),opacity=1-ct;
        float opacityBefore=1-r.cloudT;

        float mid=(t0+t1)*.5f;
        MediumSample med=SampleMedium(max(0.0f,length(O+V*mid)-ATMOS_BOTTOM_RADIUS));
        float3 ext=med.extinction+tau/max(ds,1e-8f),tr=exp(-med.extinction*ds-tau);

        float airDistance;float3 airIntegral;
        AtmosphereSourceQuadrature(ext,ds,RandomFloatPCG(airState),airDistance,airIntegral);
        float3 airWeight=r.transmittance*airIntegral;
        float3 airHint=airWeight*(med.scatterR+med.scatterM);
        float airImportance=max(airHint.x,max(airHint.y,airHint.z));
        if(airImportance>0) {
            uint band=min(airSamples-1u,(uint)((1-min(r.transmittance.x,min(r.transmittance.y,r.transmittance.z)))*airSamples));
            [unroll] for(uint k=0;k<4;k++)if(k==band) {
                selectedAir[k].mass+=airImportance;
                if(RandomFloatPCG(airState)*selectedAir[k].mass<airImportance) {
                    selectedAir[k].t=t0+airDistance;
                    selectedAir[k].weight=airWeight/airImportance;
                }
            }
        }
        if(opacity>1e-7f) {
            float u=RandomFloatPCG(randomState);
            float eventDistance=CumulusCellEvent(sigma0,sigma1,ds,opacity,u);
            float t=t0+eventDistance;
            float shadowJitter=RandomFloatPCG(randomState);

            float3 weight=r.transmittance*opacity*exp(-med.extinction*eventDistance);
            float3 P=O+V*t;
            if(lightSamples==0) {
                CumulusMaterial material=(CumulusMaterial)0;
                r.radiance+=weight*CumulusSource(P,V,L,phase,material,shadowJitter,0,cheap);
            } else {

                float f=saturate((t-begin)/max(end-begin,1e-6f))*2;
                float3 incident=f<1 ? lerp(incident0,incident1,f):lerp(incident1,incident2,f-1);
                float3 hint=ambientHint+incident*(phase+cloudMultipleScattering*.15f);
                float3 weightedHint=weight*max(hint,1e-8f);
                float importance=max(weightedHint.x,max(weightedHint.y,weightedHint.z));
                uint band=min(lightSamples-1u,(uint)(opacityBefore*float(lightSamples)));
                [unroll] for(uint k=0;k<4;k++)if(k==band) {
                    selected[k].mass+=importance;
                    if(RandomFloatPCG(lightState)*selected[k].mass<importance) {
                        selected[k].t=t;selected[k].shadowJitter=shadowJitter;
                        selected[k].weight=weight/importance;
                    }
                }
            }
        }
        r.transmittance*=tr;r.cloudT*=ct;
        t0=t1;sigma0=sigma1;
        if(max(r.transmittance.x,max(r.transmittance.y,r.transmittance.z))<minTransmittance)break;
    }
    [loop] for(uint k=0;k<lightSamples;k++)if(selected[k].mass>0) {
        float3 P=O+V*selected[k].t;
        CumulusMaterial material=(CumulusMaterial)0;

        float3 source=CumulusSource(P,V,L,phase,material,selected[k].shadowJitter,0,false);
        r.radiance+=selected[k].weight*selected[k].mass*source;
    }
    [loop] for(uint a=0;a<airSamples;a++)if(selectedAir[a].mass>0) {
        float3 P=O+V*selectedAir[a].t;
        MediumSample med=SampleMedium(max(0.0f,length(P)-ATMOS_BOTTOM_RADIUS));
        r.radiance+=selectedAir[a].weight*selectedAir[a].mass*CumulusAirSource(P,L,phR,phM,med);
    }
}

void CumulusGuideSegment(inout float trans,inout float2 moments,float3 O,float3 V,
    float2 interval,uint requestedSteps,float pixelAngle,bool cheap)
{
    if(interval.y<=interval.x)return;
    uint minSteps=16u;
    float count=float(requestedSteps);
    if(!cheap)count+=32.0f*smoothstep(8.0f,40.0f,interval.x);
    count=min(clamp(count,float(minSteps),192.0f),max(float(minSteps),(interval.y-interval.x)/max(.025f,interval.x*pixelAngle)));
    uint steps=(uint)ceil(count);
    float shift=(float(Hash32(initRandomData(gDispatchIdx.xy,uint2(0,0),0u,823u))&65535u)/65536.0f-.5f)*.8f;
    float scale=cheap ? max(2.0f,interval.x*.1f):2.0f;
    float spacing=log2(1+(interval.y-interval.x)/scale);
    float t0=interval.x,sigma0=cloudExtinction*CumulusDensity(O+V*t0,max(t0*pixelAngle,.001f),true);
    [loop] for(uint i=0;i<steps;i++) {
        float f=i+1==steps ? 1.0f:saturate((float(i+1)+shift)/count);
        float t1=interval.x+scale*(exp2(spacing*f)-1),ds=t1-t0;
        float sigma1=cloudExtinction*CumulusDensity(O+V*t1,max(t1*pixelAngle,.001f),true);
        float ct=exp(-.5f*(sigma0+sigma1)*ds),opacity=1-ct;
        if(opacity>0)moments+=trans*opacity*CumulusCellMoments(sigma0,sigma1,ds,opacity,t0);
        trans*=ct;t0=t1;sigma0=sigma1;
        if(trans<.001f)break;
    }
}

// Combine atmosphere, cloud events, and guide moments along a ray.
CumulusResult IntegrateCumulus(float3 V,float3 L,float limitKm,uint steps,float jitterValue,bool guides,float pixelAngle,bool cheap=false)
{
    CumulusResult r=(CumulusResult)0; r.transmittance=1.0f; r.cloudT=1.0f;
    float3 O=g_skyObserverPlanet;
    float outer0,outer1;
    float top=ATMOS_BOTTOM_RADIUS+cloudBaseKm+cloudThicknessKm;
    bool layer=cloudEnabled>.5f && cloudCoverage>0.0f && RaySphereIntersect(O,V,top,outer0,outer1) && outer1>0.0f;
    if(!layer) {
        r.radiance=IntegrateScattering(V,L,r.transmittance,r.hitPlanet,limitKm,cheap ? 4u : (uint)ATMOS_VIEW_STEPS,true,jitterValue)*SKY_INTENSITY;
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

    float minTransmittance=.001f;
    if(limitKm<0.0f && dot(SafeNormalize(V),L)>=cos(SUN_ANGULAR_DEG*.5f*DEG2RAD)) {
        float3 disk=EvaluateSunUnattenuated(V);
        minTransmittance/=max(1.0f,max(disk.x,max(disk.y,disk.z)));
    }
    float cursor=0.0f; float2 moments=0.0f;
    [unroll] for(uint s=0;s<2;s++) {
        float2 interval=segments[s];
        if(interval.y<=interval.x) continue;
        CumulusAirSegment(r,O,V,L,cursor,interval.x-cursor,cheap,jitterValue);
        CumulusCloudSegment(r,O,V,L,interval.x,interval.y,steps,jitterValue,pixelAngle,cheap,minTransmittance);
        cursor=interval.y;
        if(max(r.transmittance.x,max(r.transmittance.y,r.transmittance.z))<minTransmittance) break;
    }
    if(limitKm<0.0f || limitKm>cursor)
        CumulusAirSegment(r,O,V,L,cursor,limitKm<0.0f ? -1.0f : limitKm-cursor,cheap,jitterValue);
    if(guides) {
        float guideT=1;
        [unroll] for(uint g=0;g<2;g++)CumulusGuideSegment(guideT,moments,O,V,segments[g],steps,pixelAngle,cheap);
        r.guideOpacity=1-guideT;
    } else r.guideOpacity=1-r.cloudT;
    float opacity=r.guideOpacity;
    if(guides && opacity>1e-5f) {
        r.depthKm=moments.x/opacity;
        r.deviationKm=sqrt(max(0.0f,moments.y/opacity-r.depthKm*r.depthKm));

        if(!cheap) r.normal=CumulusNormal(O+V*r.depthKm,r.depthKm);
    }
    return r;
}
#endif
