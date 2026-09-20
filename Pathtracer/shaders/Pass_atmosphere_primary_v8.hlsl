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
    // In-scattering up to the surface (or through the whole atmosphere behind an escaped pixel),
    // integrated per pixel with a jittered step schedule and scaled by SKY_INTENSITY: the same
    // estimate the clear-sky path of the previous integrator produced. The night background and
    // the stars are added behind escaped pixels only.
    const float maxDistanceKm = isMeshHit
        ? length(load_x1(g_sample_current, pixelIdx) - rayOrigin) / WORLD_UNITS_PER_KM
        : -1.0f;
    const uint  spatialSeed = initRandomData(pixel, uint2(0, 0), 0u, 97u);
    const float rayJitter   = frac(float(spatialSeed & 65535u) / 65536.0f + (uint(time) % 4096u) * .61803398875f);
    bool   hitPlanet;
    float3 viewTr;
    const float3 scatter = IntegrateScattering(rayDir, sun.dirWS, viewTr, hitPlanet, maxDistanceKm,
        (uint)ATMOS_VIEW_STEPS, rayJitter) * SKY_INTENSITY;
    float3 radiance = scatter;
    if (!isMeshHit)
        radiance += EvaluateSkyBackgroundBehind(rayDir, sun, hitPlanet, scatter) * viewTr;

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
