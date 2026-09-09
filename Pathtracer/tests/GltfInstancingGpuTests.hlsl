#define COMPUTE_PASS
#include "Includes_v8.hlsli"

RWStructuredBuffer<float4> results : register(u0, space1);

[numthreads(2, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint id = tid.x;
    HitInfo hit = EvalSurfaceStateDir(id, 0, float2(0.2, 0.3), normalize(float3(0, 1, -1)), 0);
    results[3 * id] = float4(hit.rawNormal, float(hit.lightID));
    results[3 * id + 1] = float4(CandidateGeoNormalW(id, 0), 0);
    LT_Sample chosen;
    chosen.id = id;
    chosen.pdf = 0.5;
    uint rng = id + 1;
    LT_LightSampleResult light = LT_SamplePointOnLightTree(float3(0, -10, 10), chosen, rng);
    results[3 * id + 2] = float4(light.normal, float(light.objID));
}
