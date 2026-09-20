#include "Includes_v8.hlsli"

// Atmosphere along the primary ray: in-scattering up to the primary hit, and the sky behind
// escaped pixels. Both feed the shading pass through the scratch slices 10 (radiance) and 11
// (transmittance).
[numthreads(8, 8, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    if (DTid.x >= IMG_W || DTid.y >= IMG_H) return;

    const uint2 pixel = DTid.xy;
    const uint pixelIdx = MapPixelID(float2(IMG_W, IMG_H), pixel);
    const bool isMeshHit = load_instID(g_sample_current, pixelIdx) != 0xFFFFFFFFu;

    uint seed = initRandomData(pixel, uint2(0, 0), (uint)time, 71u);
    float3 rayOrigin, rayDir;
    InitCameraRayDoF(pixel, uint2(IMG_W, IMG_H), seed, rayOrigin, rayDir);
    SetSkyObserver(rayOrigin + sceneOriginWorld);

    if (SkyObserverIsUnderground())
    {
        gScratchPing[uint3(pixel, 10)] = 0.0f;
        gScratchPing[uint3(pixel, 11)] = float4(isMeshHit ? 1.0f : 0.0f,
            isMeshHit ? 1.0f : 0.0f, isMeshHit ? 1.0f : 0.0f, 0.0f);
        return;
    }

    const SunState sun = ComputeSunState();
    float3 radiance, viewTr;
    if (isMeshHit)
    {
        // Aerial perspective: integrate only up to the surface, with the cheaper step counts.
        const float maxDistanceKm = length(load_x1(g_sample_current, pixelIdx) - rayOrigin) / WORLD_UNITS_PER_KM;
        bool hitPlanet;
        radiance = IntegrateScattering(rayDir, sun.dirWS, viewTr, hitPlanet, maxDistanceKm, ATMOS_AERIAL_VIEW_STEPS);
    }
    else
    {
        float3 scatter;
        float  hitPlanet;
        SkyAtmosphere(rayDir, sun.dirWS, scatter, viewTr, hitPlanet);
        radiance = scatter + EvaluateSkyBackgroundBehind(rayDir, sun, hitPlanet > 0.5f, scatter) * viewTr;
    }

    if (ATM_DEBUG_RING == 1u)
    {
        const float dbgTr = saturate(dot(viewTr, float3(0.33333f, 0.33333f, 0.33334f)));
        float dbgScatter = dot(radiance, float3(0.33333f, 0.33333f, 0.33334f));
        dbgScatter = saturate(1.0f - exp(-max(dbgScatter, 0.0f) * 4.0f));
        radiance = float3(dbgTr, dbgScatter, 0.0f) * 0.5f;
        viewTr = 0.0f;
    }

    gScratchPing[uint3(pixel, 10)] = float4(radiance, 0.0f);
    gScratchPing[uint3(pixel, 11)] = float4(viewTr, 0.0f);
}
