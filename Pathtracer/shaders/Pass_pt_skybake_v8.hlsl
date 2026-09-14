#define COMPUTE_PASS
#include "Includes_v8.hlsli"

[numthreads(SKYBAKE_THREADS, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    gDispatchIdx = tid;
    const uint i = tid.x;
    if (i >= SKYBAKE_LUT_W * SKYBAKE_LUT_H) return;
    if (cloudEnabled>.5f && i>0u) return;

    SetSkyObserver(InitOrigin() + sceneOriginWorld);
    const SunState S = ComputeSunStateInline();
    if (i == 0u) SkyBakeStoreSunState(S);

    if (cloudEnabled>.5f) return;

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
