#ifndef CUMULUS_DENSITY_V8
#define CUMULUS_DENSITY_V8
float3 CumulusMaterialPosition(float3 P)
{
    return (P-float3(cloudWindX,0,cloudWindZ)*(walltime*.001f))/max(cloudScale,.2f)
        +float3(.137f,.731f,.353f)*cloudSeed;
}
float CumulusValue(float3 q) { return g_cumulusNoise.SampleLevel(g_sampler,q/48.0f,0).b; }
float CumulusWorley(float3 q) { return g_cumulusNoise.SampleLevel(g_sampler,q/8.0f,0).g; }
struct CumulusMaterial { float density; float profile; float height; };
CumulusMaterial CumulusSampleMaterial(float3 P,float footprintKm,bool fine,bool shadowCoarse=false)
{
    CumulusMaterial m=(CumulusMaterial)0;
    float altitude=length(P)-ATMOS_BOTTOM_RADIUS-cloudBaseKm;
    if(altitude<=0 || altitude>=cloudThicknessKm || cloudCoverage<=0) return m;
    float3 q=CumulusMaterialPosition(P);
    float3 foot=CumulusMaterialPosition(normalize(P)*(ATMOS_BOTTOM_RADIUS+cloudBaseKm));
    float organization=CumulusValue(foot*.25f);
    float top=cloudThicknessKm*lerp(.72f,1.0f,CumulusValue(foot*.21f+17.1f));
    float h=altitude/top; m.height=saturate(h);
    if(h>=1) return m;
    float heightProfile=smoothstep(0.0f,.10f,h)*(1.0f-smoothstep(.30f,1.0f,h));
    float coverage=lerp(.12f,.64f,smoothstep(.30f,.72f,organization))*lerp(.55f,1.8f,saturate(cloudCoverage));
    if(coverage<.02f) return m;
    // Shadow-cache voxels filter sub-voxel warp and seam detail. Local stochastic
    // shadow taps retain the complete field; RR guides keep the actual warped shape.
    float3 warp=0;
    if(!shadowCoarse) warp=float3(CumulusValue(q*.7f+float3(7.7f,0,0)),
        CumulusValue(q*.7f+float3(13.1f,0,0)),CumulusValue(q*.7f+float3(23.5f,0,0)))-.5f;
    float3 shape=q+warp*.32f;
    float3 uv=shape*(.65f/16.0f);
    float base=.65f*g_cumulusNoise.SampleLevel(g_sampler,uv,0).r
        +.35f*g_cumulusNoise.SampleLevel(g_sampler,uv*2+float3(.117f,.053f,.239f),0).r;
    float footprint=footprintKm/max(cloudScale,.2f);
    float lobeFade=1.0f-smoothstep(.12f,.7f,footprint);
    float uplift=smoothstep(.035f,.35f,h);
    base+=.45f*uplift*lobeFade*(CumulusWorley(shape*1.45f+float3(19.3f,7.7f,41.9f))-.43f);
    float detailFade=fine ? 1.0f-smoothstep(.012f,.09f,footprint) : 0.0f;
    if(detailFade>0) {
        float3 up=normalize(P),wind=float3(cloudWindX,0,cloudWindZ);
        wind-=up*dot(wind,up);
        if(dot(wind,wind)<.01f) wind=cross(up,abs(up.z)<.9f ? float3(0,0,1):float3(1,0,0));
        wind=normalize(wind);
        float3 wq=shape-wind*(dot(shape,wind)*.6f);
        // Add material before thresholding: this grows translucent outward filaments.
        base+=.32f*cloudDetail*detailFade*smoothstep(.2f,.65f,h)
            *(CumulusWorley(wq*2.7f+float3(53.1f,17.7f,91.3f))-.30f);
    }
    float d=saturate((saturate(base)*.7f+.3f-1.0f+coverage*heightProfile)/.3f);
    if(d<=0) return m;
    if(!shadowCoarse) {
        float bil=g_cumulusNoise.SampleLevel(g_sampler,(shape*.67f+float3(31.7f,11.9f,23.1f))/16.0f,0).a;
        float bil2=g_cumulusNoise.SampleLevel(g_sampler,(shape*1.427f+float3(7.3f,27.1f,13.9f))/16.0f,0).a;
        float carve=.65f*bil+.35f*bil2;
        d=saturate(d+uplift*lobeFade*(-.4f*pow(carve,1.5f)*(1-d)*(1-d)+.18f*(1-carve)*(1-carve)*(1-abs(2*d-1))));
    }
    if(detailFade>0) {
        float hf=CumulusWorley(shape*7.27f);
        float micro=CumulusWorley(shape*23.7f+float3(3.7f,19.2f,7.1f));
        hf=lerp(hf,1-hf,saturate(h*1.3f));
        float signedDetail=(hf-.48f)*.48f+(micro-.40f)*.20f;
        d=saturate(d+cloudDetail*detailFade*lerp(.6f,1.0f,uplift)*signedDetail*4*d*(1-d));
    }
    // Detailed lobes drive the scattering model too; they are not just silhouette erosion.
    m.profile=d; m.density=pow(d,lerp(.85f,1.0f,d));
    return m;
}
float CumulusDensity(float3 P,float footprintKm,bool fine) { return CumulusSampleMaterial(P,footprintKm,fine).density; }
float2 CumulusTraceOpticalDepth(float3 P,float3 L,uint samples=20u,float jitterValue=.5f)
{
    float a,b,top=ATMOS_BOTTOM_RADIUS+cloudBaseKm+cloudThicknessKm;
    if(!RaySphereIntersect(P,L,top,a,b)||b<=0) return 0;
    float end=b, groundA,groundB;
    float room=max(0.0f,top-length(P)),skyTau=0;
    [unroll] for(uint j=0;j<4;j++) skyTau+=CumulusSampleMaterial(P+normalize(P)*(room*(j+.5f)/4),.1f,false,true).density*room*.25f*cloudExtinction;
    if(RaySphereIntersect(P,L,ATMOS_BOTTOM_RADIUS,groundA,groundB)&&groundA>0 && groundA<end) return float2(80,min(skyTau,80));
    // Geometric spacing reaches the actual shell exit, including hundreds of km at dawn.
    // Unlike a fixed nine-tap/18 km ray this also sees the far side of a grazing shell.
    float tau=0,spacing=log2(1+end/.18f);
    [loop] for(uint i=0;i<samples;i++) {
        float d0=.18f*(exp2(spacing*(float(i)/samples))-1),d1=.18f*(exp2(spacing*(float(i+1)/samples))-1);
        tau+=CumulusSampleMaterial(P+L*lerp(d0,d1,frac(jitterValue+i*.618034f)),min(.25f,(d1-d0)*.02f),false,true).density*(d1-d0)*cloudExtinction;
        if(tau>40) break;
    }
    return min(float2(tau,skyTau),80.0f);
}
float2 CumulusLightingDepth(float3 P,float3 L,float shadowJitter)
{
    // Resolve sub-cache lobe shadows using a stratified, detailed local segment.
    const float localLength=.24f;
    float3 uv=CumulusLightUv(P+L*localLength);
    float2 tau=all(uv>=0)&&all(uv<=1) ? g_cumulusLight.SampleLevel(g_sampler_LUT,uv,0)
        : CumulusTraceOpticalDepth(P+L*localLength,L,10u,shadowJitter);
    tau.x+=CumulusDensity(P+L*(shadowJitter*localLength),.004f,true)*localLength*cloudExtinction;
    return tau;
}
float3 CumulusNormal(float3 P,float distanceKm)
{
    float e=clamp(distanceKm*.0005f,.012f,.07f);
    float3 n=float3(CumulusDensity(P-float3(e,0,0),e,true)-CumulusDensity(P+float3(e,0,0),e,true),
        CumulusDensity(P-float3(0,e,0),e,true)-CumulusDensity(P+float3(0,e,0),e,true),
        CumulusDensity(P-float3(0,0,e),e,true)-CumulusDensity(P+float3(0,0,e),e,true));
    return dot(n,n)>1e-10f ? normalize(n) : normalize(P);
}
#endif
