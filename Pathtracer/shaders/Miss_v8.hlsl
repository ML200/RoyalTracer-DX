#include "Includes_v8.hlsli"
#include "PtVertex_v8.hlsli"

// Escaped ray of the split path tracer: sky and sun radiance with the sun MIS weight of the
// scatter the ray came from. The raygen owns the throughput and the reuse candidate.
[shader("miss")]
void Miss(inout TracePayload payload)
{
    const uint   inFlags = payload.flags;
    const float3 rayDir  = WorldRayDirection();

    const bool underground = WorldPosIsUnderground(WorldRayOrigin() + sceneOriginWorld);
    SetSkyObserver(InitOrigin() + sceneOriginWorld);
    const float  sunSAPdf   = underground ? 0.0f : GetSunPdf(rayDir);
    const float3 sunRad     = (sunSAPdf > 0.0f) ? EvaluateSun(rayDir) : float3(0, 0, 0);
    const float  sunMisBsdf = (sunSAPdf > 0.0f)
        ? ((inFlags & PV_IN_MIS_NONE) != 0u ? 1.0f : payload.pdf / max(payload.pdf + sunSAPdf, EPSILON)) : 0.0f;

    payload.flags = PV_MISS;
    payload.pdf   = sunSAPdf > 0.0f ? sunMisBsdf : 1.0f;
    payload.color = underground ? float3(0, 0, 0) : EvaluateSky(rayDir);
    payload.auxPk = PvPackRadiance(sunRad);
}
