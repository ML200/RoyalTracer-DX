#define COMPUTE_PASS
#include "Includes_v8.hlsli"

//====================================
//PT SKY BAKE (sun state + sky-view LUT)
//====================================
//One fixed dispatch per frame under the regular path tracer, after Pass_camera
//and before the training and bounce kernels, which both read it (layout and
//rationale in SkyBake_v8.hlsli). One thread per LUT texel; thread 0 also
//writes the sun state. Everything is a function of the camera observer and
//the frame constants, so the results are the same for every pixel.
[numthreads(SKYBAKE_THREADS, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    gDispatchIdx = tid;
    const uint i = tid.x;
    if (i >= SKYBAKE_LUT_W * SKYBAKE_LUT_H) return;
    if (cloudEnabled>.5f && i>0u) return;

    //the observer every Pass_pt miss and sun sample uses
    SetSkyObserver(InitOrigin() + sceneOriginWorld);
    const SunState S = ComputeSunStateInline();
    if (i == 0u) SkyBakeStoreSunState(S);
    // Every cloudy EvaluateSky consumer reads the separate cloud environment.
    if (cloudEnabled>.5f) return;

    const uint2  texel = uint2(i % SKYBAKE_LUT_W, i / SKYBAKE_LUT_W);
    const float3 v = SkyBakeDirFromUv((float2(texel) + 0.5f) / float2(SKYBAKE_LUT_W, SKYBAKE_LUT_H));
    float3 scatter   = float3(0, 0, 0);
    float3 viewTr    = float3(1, 1, 1);
    float  hitPlanet = 0.0f;
    //an underground observer sees no sky (the consumer returns black before
    //reading); keep the texels finite anyway
    if (!SkyObserverIsUnderground())
    {
        bool hit;
        scatter   = IntegrateScattering(v, S.dirWS, viewTr, hit);
        hitPlanet = hit ? 1.0f : 0.0f;
    }
    SkyBakeStoreView(texel, scatter, viewTr, hitPlanet);
}
