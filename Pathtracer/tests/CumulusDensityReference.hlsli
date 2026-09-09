// Frozen approved material before the September 8 cost-only optimizations.
// Preserve this graph: it is the regression oracle for conservative skips.
// Shared coordinate/noise helpers are intentionally unchanged by this revision.
CumulusMaterial CumulusReferenceMaterial(float3 P,float footprintKm,bool fine,bool shadowCoarse=false)
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
