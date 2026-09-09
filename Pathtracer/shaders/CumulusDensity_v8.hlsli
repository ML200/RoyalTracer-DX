#ifndef CUMULUS_DENSITY_V8
#define CUMULUS_DENSITY_V8
#include "CumulusDensityCache_v8.hlsli"
float3 CumulusMaterialPosition(float3 P)
{
    return (P-float3(cloudWindX,0,cloudWindZ)*(walltime*.001f))/max(cloudScale,.2f)
        +float3(.137f,.731f,.353f)*cloudSeed;
}
float CumulusValue(float3 q) { return g_cumulusNoiseBA.SampleLevel(g_sampler,q/48.0f,0).r; }
float CumulusWorley(float3 q) { return g_cumulusNoiseRG.SampleLevel(g_sampler,q/8.0f,0).g; }
struct CumulusMaterial { float density; float profile; float height; };
CumulusMaterial CumulusSampleMaterial(float3 P,float footprintKm,bool fine,bool shadowCoarse=false)
{
    CumulusMaterial m=(CumulusMaterial)0;
    float altitude=length(P)-ATMOS_BOTTOM_RADIUS-cloudBaseKm;
    if(altitude<=0 || altitude>=cloudThicknessKm || cloudCoverage<=0) return m;
    float3 q=CumulusMaterialPosition(P);
    float3 foot=CumulusMaterialPosition(normalize(P)*(ATMOS_BOTTOM_RADIUS+cloudBaseKm));
    float top=cloudThicknessKm*lerp(.72f,1.0f,CumulusValue(foot*.21f+17.1f));
    float h=altitude/top; m.height=saturate(h);
    if(h>=1) return m;
    float organization=CumulusValue(foot*.25f);
    float heightProfile=smoothstep(0.0f,.10f,h)*(1.0f-smoothstep(.30f,1.0f,h));
    float coverage=lerp(.12f,.64f,smoothstep(.30f,.72f,organization))*lerp(.55f,1.8f,saturate(cloudCoverage));
    if(coverage<.02f) return m;
    // Wind deformation belongs to the cloud body, not the filtered erosion layer.
    // Keep it in distant rays and shadow caches so every pass sees the silhouette.
    float deformation=clamp(cloudFineDetail,0.0f,2.0f)*cloudDetail;
    float3 warp=0;
    if(!shadowCoarse || deformation>0) warp=float3(CumulusValue(q*.7f+float3(7.7f,0,0)),
        CumulusValue(q*.7f+float3(13.1f,0,0)),CumulusValue(q*.7f+float3(23.5f,0,0)))-.5f;
    float3 shape=shadowCoarse ? q : q+warp*.32f;
    float footprint=footprintKm/max(cloudScale,.2f);
    float lobeFade=1.0f-smoothstep(.12f,.7f,footprint);
    float uplift=smoothstep(.035f,.35f,h);
    float detailFade=fine ? 1.0f-smoothstep(.012f,.09f,footprint) : 0.0f;
    float3 lobeShape=shape,wq=shape;
    if(detailFade>0 || deformation>0) {
        float3 up=normalize(P),wind=float3(cloudWindX,0,cloudWindZ);
        wind-=up*dot(wind,up);
        if(dot(wind,wind)<.01f) wind=cross(up,abs(up.z)<.9f ? float3(0,0,1):float3(1,0,0));
        wind=normalize(wind);
        float3 across=cross(up,wind);
        // Deform all body coordinates before evaluating density. Applying this
        // only to later lobes/erosion left their enclosing silhouette unchanged.
        // The recovered wavelengths stay intact; shear and bending move the form.
        float shear=.22f*altitude/max(cloudScale,.2f)+.75f*dot(warp,up);
        lobeShape-=deformation*(wind*shear+across*(.18f*dot(warp,across)));
        wq=lobeShape-wind*(dot(lobeShape,wind)*.6f);
    }
    float3 uv=lobeShape*(.65f/16.0f);
    float base=.65f*g_cumulusNoiseRG.SampleLevel(g_sampler,uv,0).r
        +.35f*g_cumulusNoiseRG.SampleLevel(g_sampler,uv*2+float3(.117f,.053f,.239f),0).r;
    // The remaining lobe noises are UNORM. Reject only when even their
    // maximum positive contribution plus every outward child billow is empty.
    // Leave the evaluated field below in its original arithmetic order.
    float upperBase=base+.45f*uplift*lobeFade*.57f
        +.32f*cloudDetail*detailFade*smoothstep(.2f,.65f,h)*.70f;
    float upperField=(saturate(upperBase)*.7f+.3f-1+coverage*heightProfile)/.3f;
    float upperDetail=shadowCoarse ? 0.0f:cloudDetail*detailFade*lerp(.4f,1.0f,uplift);
    if(upperField<-.27f*upperDetail-1e-5f)return m;
    base+=.45f*uplift*lobeFade*(CumulusWorley(lobeShape*1.45f+float3(19.3f,7.7f,41.9f))-.43f);
    if(detailFade>0) {
        // The original broad wind-stretched octave grows outward before clipping.
        // It supplies connected streaks rather than a separate high-frequency rim.
        base+=.32f*cloudDetail*detailFade*smoothstep(.2f,.65f,h)
            *(CumulusWorley(wq*2.7f+float3(53.1f,17.7f,91.3f))-.30f);
    }
    float field=(saturate(base)*.7f+.3f-1.0f+coverage*heightProfile)/.3f;
    float detailStrength=shadowCoarse ? 0.0f : cloudDetail*detailFade*lerp(.4f,1.0f,uplift);
    // Leave room for outward child billows before clipping the density. Remaining
    // positive displacement is below .260 * detailStrength for UNORM inputs.
    if(field<=-.27f*detailStrength) return m;
    if(field>=1.0f) { m.profile=1.0f; m.density=1.0f; return m; }
    float breakupFade=1.0f-smoothstep(.008f,.065f,footprint);
    float2 breakup=0;
    if(detailStrength>0) {
        float3 edgeShape=lerp(lobeShape,wq,.45f*saturate(cloudFineDetail));
        // Both lookups replace the old erosion lookups. Child billows are 2.27x
        // larger than the rejected 7.27-frequency cells, with breakup 2.82x larger
        // than the old 23.7-frequency grain. The common wind field stays intact.
        breakup=g_cumulusNoiseRG.SampleLevel(g_sampler,(edgeShape*8.4f+float3(3.7f,19.2f,7.1f))/8.0f,0);
        float3 bend=float3(breakup.r-.4631f,breakup.g-.40363f,breakup.r-breakup.g-.05947f);
        float3 cellPosition=edgeShape*3.2f+float3(17.3f,3.1f,29.7f)+.75f*breakupFade*bend;
        float2 cells=g_cumulusNoiseBA.SampleLevel(g_sampler,cellPosition/16.0f,0);
        float boundary=1.0f-smoothstep(.4f,1.0f,abs(field));
        field+=detailStrength*boundary*(.36f*(1.0f-cells.g-.4024f)
            +.045f*(cells.r-.499f)+.04f*breakupFade*(breakup.r-.4631f));
    }
    float d=saturate(field);
    if(d<=0) return m;
    // Every remaining erosion term contains (1-d). Interior density is already exact.
    if(d>=1.0f) { m.profile=1.0f; m.density=1.0f; return m; }
    if(!shadowCoarse) {
        float bil=g_cumulusNoiseBA.SampleLevel(g_sampler,(lobeShape*.67f+float3(31.7f,11.9f,23.1f))/16.0f,0).g;
        float bil2=g_cumulusNoiseBA.SampleLevel(g_sampler,(lobeShape*1.427f+float3(7.3f,27.1f,13.9f))/16.0f,0).g;
        float carve=.65f*bil+.35f*bil2;
        d=saturate(d+uplift*lobeFade*(-.4f*pow(carve,1.5f)*(1-d)*(1-d)+.18f*(1-carve)*(1-carve)*(1-abs(2*d-1))));
    }
    if(detailStrength>0) {
        // Uneven erosion follows the larger child forms; there is no separate
        // tiny-frequency rim. Its spatial footprint also filters local shadows
        // and RR gradients, which evaluate this same density field.
        float erosion=.16f*(breakup.g-.40363f)+.05f*(breakup.r-.4631f);
        d=saturate(d+detailStrength*breakupFade*erosion*4*d*(1-d));
    }
    // Detailed lobes drive the scattering model too; they are not just silhouette erosion.
    m.profile=d; m.density=pow(d,lerp(.85f,1.0f,d));
    return m;
}
float CumulusDensity(float3 P,float footprintKm,bool fine)
{
    float density;
    if(CumulusReadDensityCache(P,footprintKm,fine,density))return density;
    return CumulusSampleMaterial(P,footprintKm,fine).density;
}
float2 CumulusTraceOpticalDepth(float3 P,float3 L,uint samples=20u,float jitterValue=.5f,bool traceSun=true,bool traceSky=true)
{
    float a,b,top=ATMOS_BOTTOM_RADIUS+cloudBaseKm+cloudThicknessKm;
    if(!RaySphereIntersect(P,L,top,a,b)||b<=0) return 0;
    float begin=max(0.0f,a),end=b, groundA,groundB;
    float room=max(0.0f,top-length(P)),skyTau=0;
    if(traceSky) [unroll] for(uint j=0;j<4;j++) skyTau+=CumulusSampleMaterial(P+normalize(P)*(room*(j+.5f)/4),.1f,false,true).density*room*.25f*cloudExtinction;
    if(!traceSun) return float2(0,min(skyTau,80.0f));
    // This cache stores cloud extinction only. Planet visibility is the smooth
    // finite-sun-disk gate at the scattering point, not an interpolated tau=80
    // discontinuity. Stop at the ground instead of tracing through the planet.
    if(RaySphereIntersect(P,L,ATMOS_BOTTOM_RADIUS,groundA,groundB)&&groundA>0 && groundA<end) {
        end=groundA;
    }
    float2 intervals[2];intervals[0]=float2(begin,end);intervals[1]=0;
    if(RaySphereIntersect(P,L,ATMOS_BOTTOM_RADIUS+cloudBaseKm,a,b) && b>begin && a<end) {
        intervals[0].y=min(end,a);intervals[1]=float2(max(begin,b),end);
    }
    // Fixed-distance cells, independent of the sun-ray exit/planet intersection.
    // Previously a moving sunset changed every sample distance when the centre
    // ray touched the planet. Clip cells to both cloud intervals, skipping the
    // empty below-base gap without redistributing the remaining samples.
    float horizonSpan=2*sqrt(max(0.0f,top*top-ATMOS_BOTTOM_RADIUS*ATMOS_BOTTOM_RADIUS));
    float tau=0,spacing=log2(1+max(end,horizonSpan)/.18f);
    [loop] for(uint i=0;i<samples;i++) {
        float d0=.18f*(exp2(spacing*(float(i)/samples))-1),d1=.18f*(exp2(spacing*(float(i+1)/samples))-1);
        [unroll] for(uint k=0;k<2;k++) {
            float lo=max(d0,intervals[k].x),hi=min(d1,intervals[k].y);
            if(hi>lo)tau+=CumulusSampleMaterial(P+L*lerp(lo,hi,frac(jitterValue+i*.618034f)),min(.25f,(hi-lo)*.02f),false,true).density*(hi-lo)*cloudExtinction;
        }
        // The cached depth also drives reduced-extinction scattering orders.
        // Stopping at 40 but returning values up to 80 made the result jump
        // whenever a moving sun added/removed the first cell beyond 40. Stop
        // at the stored saturation value so that transition stays continuous.
        if(tau>=80.0f) break;
        if(d1>=end)break;
    }
    return min(float2(tau,skyTau),80.0f);
}
float2 CumulusLightingDepth(float3 P,float3 L,float shadowJitter,bool cheap=false,bool sunlight=true)
{
    if(!cheap && sunlight) {
        // Resolve local self-shadow structure with short strata, followed by
        // the continuously blended near and far caches.
        const float localLength=.24f;
        // Blend near/far broad shadow caches continuously. A separate, sparse
        // procedural horizon trace outside the near cascade undersampled whole
        // clouds and changed lighting abruptly at the cache boundary.
        float2 tau=CumulusCachedOpticalDepth(P+L*localLength);
        if(cloudFineDetail>0) {
            const float boundaries[4]={0.0f,.024f,.080f,localLength};
            [loop] for(uint j=0;j<3;j++) {
                float a=boundaries[j],b=boundaries[j+1];
                float t=lerp(a,b,frac(shadowJitter+float(j)*.61803398875f));
                tau.x+=CumulusDensity(P+L*t,.004f,true)*(b-a)*cloudExtinction;
            }
        } else tau.x+=CumulusDensity(P+L*(shadowJitter*localLength),.004f,true)*localLength*cloudExtinction;
        return tau;
    }
    // Secondary rays use only cached lighting, including the segment at the sample.
    float3 uv=CumulusLightUv(P);
    float border=min(min(uv.x,1.0f-uv.x),min(uv.z,1.0f-uv.z));
    float2 tau;
    if(border>=.02f) tau=CumulusReadLight(uv,0u);
    else {
        float3 distantUv=CumulusLightUv(P,1u);
        tau=CumulusReadLight(distantUv,1u);
        if(border>0) tau=lerp(tau,CumulusReadLight(uv,0u),smoothstep(0.0f,.02f,border));
    }
    return tau;
}
float3 CumulusNormal(float3 P,float distanceKm)
{
    float e=clamp(distanceKm*.0005f,.012f,.07f);
    // Preserve the detailed gradient even where colour/depth use cached density.
    float3 n=float3(CumulusSampleMaterial(P-float3(e,0,0),e,true).density-CumulusSampleMaterial(P+float3(e,0,0),e,true).density,
        CumulusSampleMaterial(P-float3(0,e,0),e,true).density-CumulusSampleMaterial(P+float3(0,e,0),e,true).density,
        CumulusSampleMaterial(P-float3(0,0,e),e,true).density-CumulusSampleMaterial(P+float3(0,0,e),e,true).density);
    return dot(n,n)>1e-10f ? normalize(n) : normalize(P);
}
#endif
