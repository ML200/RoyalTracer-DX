#include "Includes_v8.hlsli"

// Per-frame sun and sky-view LUT bake (Hillaire 2020); read via ComputeSunState, SkyAtmosphere.
[numthreads(SKYBAKE_THREADS, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    const uint i = tid.x;
    if (i >= SKYBAKE_LUT_W * SKYBAKE_LUT_H) return;

    // A submerged camera can be below the ground plane; bake from no lower than it.
    float3 observer = InitOrigin() + sceneOriginWorld;
    if (OCEAN_ENABLED)
        observer.y = max(observer.y, SKY_GROUND_Y);
    SetSkyObserver(observer);
    const SunState S = ComputeSunStateInline();
    if (i == 0u) SkyBakeStoreSunState(S);

    const uint2  texel = uint2(i % SKYBAKE_LUT_W, i / SKYBAKE_LUT_W);
    const float3 v = SkyBakeDirFromUv((float2(texel) + 0.5f) / float2(SKYBAKE_LUT_W, SKYBAKE_LUT_H));
    float3 scatter   = float3(0, 0, 0);
    float3 viewTr    = float3(1, 1, 1);
    float  hitPlanet = 0.0f;

    if (!SkyObserverIsUnderground())
    {
        bool hit;
        scatter   = IntegrateScattering(v, S.dirWS, viewTr, hit);
        hitPlanet = hit ? 1.0f : 0.0f;
    }
    SkyBakeStoreView(texel, scatter, viewTr, hitPlanet);

    // Sky irradiance on a horizontal plane; clears the next frame's slot.
    const uint frame = (uint)time;
    if (i == 0u)
    {
        const uint next = SkyBakeIrradianceSlot(frame + 1u);
        g_skyBake.Store<uint64_t>(next, 0u);
        g_skyBake.Store<uint64_t>(next + 8u, 0u);
        g_skyBake.Store<uint64_t>(next + 16u, 0u);
    }
    const float3 e = v.y > 0.0f ? scatter * v.y * SkyBakeTexelSolidAngle(texel.y) : 0.0f;
    const float3 wave = WaveActiveSum(e);
    if (WaveIsFirstLane())
    {
        const uint slot = SkyBakeIrradianceSlot(frame);
        g_skyBake.InterlockedAdd64(slot, (uint64_t)(max(wave.x, 0.0f) * SKYBAKE_IRRADIANCE_SCALE + 0.5f));
        g_skyBake.InterlockedAdd64(slot + 8u, (uint64_t)(max(wave.y, 0.0f) * SKYBAKE_IRRADIANCE_SCALE + 0.5f));
        g_skyBake.InterlockedAdd64(slot + 16u, (uint64_t)(max(wave.z, 0.0f) * SKYBAKE_IRRADIANCE_SCALE + 0.5f));
    }
}
