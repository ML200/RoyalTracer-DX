#include "Includes_v8.hlsli"

// Bake the sun state and the sky (in-scattering, transmittance, planet hit) for the frame into
// the sky-bake buffer; every other pass reads the baked values through ComputeSunState and
// SkyAtmosphere.
[numthreads(SKYBAKE_THREADS, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    const uint i = tid.x;
    if (i >= SKYBAKE_LUT_W * SKYBAKE_LUT_H) return;

    SetSkyObserver(InitOrigin() + sceneOriginWorld);
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
}
