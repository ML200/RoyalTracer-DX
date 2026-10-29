#include "Includes_v8.hlsli"
#include "HostOutputLayout.h"

// Runs after DLSS and before the auto-exposure update, so the exposure is the one the DLSS input was scaled by.
// The depth ray goes through the pixel centre without jitter, matching a host camera that does not jitter.
[numthreads(8, 8, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    if (DTid.x >= IMG_W || DTid.y >= IMG_H) return;

    RWTexture2D<float4> hostColor = ResourceDescriptorHeap[HOST_COLOR_UAV_SLOT];
    RWTexture2D<float>  hostDepth = ResourceDescriptorHeap[HOST_DEPTH_UAV_SLOT];

    const float log2Exposure = log2(0.18f) - asfloat(gAutoExpose.Load(AE_OFFS_SMOOTHED));
    // A host blends this into its own image and exposure history, where one NaN spreads for good (DLSS resets can
    // produce some).
    const float3 color = g_dlssOutput[DTid.xy].rgb;
    hostColor[DTid.xy] = float4(all(isfinite(color)) ? color : 0.0f, isfinite(log2Exposure) ? log2Exposure : 0.0f);

    const float2 ndc    = ((float2(DTid.xy) + 0.5f) / float2(IMG_W, IMG_H)) * 2.0f - 1.0f;
    const float4 target = mul(projectionI, float4(ndc.x, -ndc.y, 1, 1));
    const float3 dir    = normalize(mul(viewI, float4(target.xyz, 0)).xyz);
    const float3 fwd    = normalize(mul(viewI, float4(0, 0, -1, 0)).xyz);

    RayDesc ray;
    ray.Origin    = mul(viewI, float4(0, 0, 0, 1)).xyz;
    ray.Direction = dir;
    ray.TMin      = 0.0f;
    ray.TMax      = 1.0e7f;

    RayQuery<RAY_FLAG_NONE, RAYQUERY_FLAG_ALLOW_OPACITY_MICROMAPS> q;
    q.TraceRayInline(SceneBVH, RAY_FLAG_NONE, INSTANCE_MASK_SURFACE, ray);
    [loop]
    for (uint i = 0u; q.Proceed() && i < 128u; ++i)
    {
        if (q.CandidateType() != CANDIDATE_NON_OPAQUE_TRIANGLE) continue;
        const uint cInstID = q.CandidateInstanceID();
        const uint cPrimID = FlatPrimID(cInstID, q.CandidateGeometryIndex(), q.CandidatePrimitiveIndex());
        const uint cMatID  = materialIDs[instanceProps[cInstID].materialBase + cPrimID];
        // Glass stays see-through, as the host draws its transparents itself.
        if (LoadIsThinGlass(cMatID) || LoadKd_w(cMatID) < 1.0f - EPSILON) continue;
        if (AlphaCandidateOccludes(cInstID, cPrimID, q.CandidateTriangleBarycentrics()))
            q.CommitNonOpaqueTriangleHit();
    }
    hostDepth[DTid.xy] = q.CommittedStatus() == COMMITTED_TRIANGLE_HIT ? q.CommittedRayT() * dot(dir, fwd) : 0.0f;
}
