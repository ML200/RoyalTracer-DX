RaytracingAccelerationStructure scene : register(t0);
RWStructuredBuffer<uint2> result : register(u0);
cbuffer Probe : register(b0) { float3 origin; uint mode; uint count; float tMax; uint seed; uint pad; }

[numthreads(64,1,1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= count) return;
    uint h = id.x + seed * 747796405u;
    h = (h ^ (h >> 16)) * 2246822519u;
    h = (h ^ (h >> 13)) * 3266489917u;
    float azimuth = (h & 65535u) * (6.28318530718 / 65536.0);
    float y = ((h >> 16) + .5) / 65536.0 * 2 - 1;
    float r = sqrt(max(0,1-y*y));
    RayDesc ray;
    ray.Origin = origin; ray.Direction = float3(r*cos(azimuth), y, r*sin(azimuth));
    ray.TMin = .02; ray.TMax = tMax;
    RayQuery<RAY_FLAG_FORCE_OPAQUE> query;
    query.TraceRayInline(scene, mode ? RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH : RAY_FLAG_NONE, 255, ray);
    while (query.Proceed()) {}
    result[id.x] = query.CommittedStatus() == COMMITTED_TRIANGLE_HIT
        ? uint2(query.CommittedInstanceID()+1, asuint(query.CommittedRayT())) : uint2(0,0);
}
