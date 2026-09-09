#define COMPUTE_PASS
#include "Includes_v8.hlsli"

cbuffer TestConstants : register(b0, space1) { uint triangleCount; }
RWStructuredBuffer<float4> results : register(u0, space1);

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= triangleCount) return;
    const float3 x = float3(0, 0, -10);
    // All fixtures lie behind this receiver's horizon: uniform fallback gives
    // an exact reference even for the extreme coordinates used to skew SAOH.
    const float3 n = float3(0, 0, -1);
    uint rng = tid.x + 1u;
    const LT_Sample sample = LT_SampleLight(x, n, rng);
    const float pdf = (rs_flags & RS_FLAG_NO_MESH_LIGHTS) != 0u ? 0.0f : LT_PdfSelectTriangle(x, n, tid.x);
    results[tid.x] = float4(pdf, sample.pdf,
        LT_PdfSelectTriangle(x, n, sample.id), sample.id == LT_SENTINEL ? -1.0f : float(sample.id));
}
