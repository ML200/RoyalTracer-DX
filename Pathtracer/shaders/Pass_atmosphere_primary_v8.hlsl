#define COMPUTE_PASS
#include "Includes_v8.hlsli"
#include "CumulusRender_v8.hlsli"

//Atmospheric scattering between the camera and the first surface or atmosphere
//exit. Slots 10/11 feed the shading composite; geometry supplies its own DLSS guides.
[numthreads(8, 8, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    if (DTid.x >= gImageWidth || DTid.y >= gImageHeight) return;
    gDispatchIdx = DTid;

    const uint2 pixel = DTid.xy;
    const uint pixelIdx = MapPixelID(float2(IMG_W, IMG_H), pixel);
    gScratchPing[uint3(pixel,CUMULUS_NORMAL_SLOT)] = 0.0f;
    gScratchPing[uint3(pixel,CUMULUS_DEPTH_SLOT)] = 0.0f;
    const bool isMeshHit = load_instID(g_sample_current, pixelIdx) != 0xFFFFFFFFu;

    uint seed = initRandomData(pixel, uint2(0, 0), (uint)time, 71u);
    float3 rayOrigin, rayDir;
    InitCameraRayDoF(pixel, uint2(IMG_W, IMG_H), seed, rayOrigin, rayDir);
    SetSkyObserver(rayOrigin + sceneOriginWorld);

    //Inside the planet, preserve local mesh radiance and extinguish sky/sun.
    if (SkyObserverIsUnderground())
    {
        gScratchPing[uint3(pixel, 10)] = 0.0f;
        gScratchPing[uint3(pixel, 11)] = float4(isMeshHit ? 1.0f : 0.0f,
            isMeshHit ? 1.0f : 0.0f, isMeshHit ? 1.0f : 0.0f, 0.0f);
        return;
    }

    const SunState sun = ComputeSunState();
    const float maxDistanceKm = isMeshHit
        ? length(load_x1(g_sample_current, pixelIdx) - rayOrigin) / WORLD_UNITS_PER_KM
        : -1.0f;
    // RR receives fresh, jittered volume integration at its own input resolution.
    // Low-discrepancy temporal rotation plus a spatial hash; not advertised as STBN.
    uint spatialSeed=initRandomData(pixel,uint2(0,0),0u,97u);
    float rayJitter = frac(float(spatialSeed & 65535u)/65536.0f + (uint(time)%4096u)*.61803398875f);
    CumulusResult cloud = IntegrateCumulus(rayDir,sun.dirWS,maxDistanceKm,
        (uint)cloudViewSteps,rayJitter,false,2.0f/(abs(projection._m11)*float(IMG_H)));
    float3 viewTr = cloud.transmittance;
    bool hitPlanet = cloud.hitPlanet;
    float3 scatter = cloud.radiance;
    float3 radiance = scatter;
    if (!isMeshHit)
        radiance += EvaluateSkyBackgroundBehind(rayDir, sun, hitPlanet, scatter) * viewTr;

#if ATM_DEBUG_RING == 1
    float dbgTr = saturate(dot(viewTr, float3(0.33333f, 0.33333f, 0.33334f)));
    float dbgScatter = dot(scatter, float3(0.33333f, 0.33333f, 0.33334f));
    dbgScatter = saturate(1.0f - exp(-max(dbgScatter, 0.0f) * 4.0f));
    radiance = float3(dbgTr, dbgScatter, 0.0f) * 0.5f;
    viewTr = 0.0f;
#endif

    gScratchPing[uint3(pixel, 10)] = float4(radiance, 0.0f);
    gScratchPing[uint3(pixel, 11)] = float4(viewTr, 0.0f);
}
