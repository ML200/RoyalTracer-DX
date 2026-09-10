#define COMPUTE_PASS
#include "Includes_v8.hlsli"
RWStructuredBuffer<float4> results : register(u0, space1);
cbuffer Probe : register(b0, space1) { float3 origin; uint padding; }

bool SelfHit(float3 p)
{
    // Both the offset and the outgoing ray point away from the floor.
    // A hit can only be caused by reconstructing the origin below its surface.
    RayDesc ray;
    ray.Origin = offset_ray(p, float3(0,origin.y>1.0f ? 1.0f : -1.0f,0));
    ray.Direction = normalize(float3(1,origin.y>1.0f ? 0.0001f : -0.0001f,0.5f));
    ray.TMin = 0.00001f;
    ray.TMax = 10000.0f;
    RayQuery<RAY_FLAG_FORCE_OPAQUE> q;
    q.TraceRayInline(SceneBVH,RAY_FLAG_NONE,255,ray);
    while(q.Proceed()) {}
    return q.CommittedStatus() == COMMITTED_TRIANGLE_HIT;
}

[numthreads(8,8,1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    const float2 xz = ((float2(tid.xy)+0.5f)/512.0f*2.0f-1.0f)*249.0f;
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = normalize(float3(xz.x,1,xz.y)-origin);
    ray.TMin = 0.00001f;
    ray.TMax = 10000.0f;
    RayQuery<RAY_FLAG_FORCE_OPAQUE> q;
    q.TraceRayInline(SceneBVH,RAY_FLAG_NONE,255,ray);
    while(q.Proceed()) {}
    float4 result=float4(0,0,0,-1);
    if(q.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
    {
        const float3 pRay = ray.Origin + ray.Direction*q.CommittedRayT();
        const float2 b=q.CommittedTriangleBarycentrics();
        // The exact two cityv3 floor triangles. Match EvalSurfaceState's
        // interpolation, with the identity instance transform used by the scene.
        const float3 p0=float3(-250,1,250);
        const float3 p1=q.CommittedPrimitiveIndex()==0 ? float3(250,1,250) : float3(250,1,-250);
        const float3 p2=q.CommittedPrimitiveIndex()==0 ? float3(250,1,-250) : float3(-250,1,-250);
        const float3 pBary=p0*(1-b.x-b.y)+p1*b.x+p2*b.y;
        result=float4(pRay.y-1,pBary.y-1,SelfHit(pRay),SelfHit(pBary));
    }
    results[tid.y*512+tid.x]=result;
}
