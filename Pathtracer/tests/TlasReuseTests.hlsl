RaytracingAccelerationStructure scene : register(t0);
RWStructuredBuffer<uint2> results : register(u0);
cbuffer Probe : register(b0) { float3 origin; uint slot; }

[numthreads(1, 1, 1)]
void main()
{
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = float3(0, 0, -1);
    ray.TMin = 0;
    ray.TMax = 10;
    RayQuery<RAY_FLAG_FORCE_OPAQUE> query;
    query.TraceRayInline(scene, RAY_FLAG_NONE, 0xff, ray);
    while (query.Proceed()) {}
    results[slot] = query.CommittedStatus() == COMMITTED_TRIANGLE_HIT
        ? uint2(query.CommittedInstanceID() + 1, asuint(query.CommittedRayT())) : uint2(0, 0);
}
