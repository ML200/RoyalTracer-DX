#include "Includes_raygen_v8.hlsli"

[shader("closesthit")] void ClosestHit(inout HitBlobPayload payload, in BuiltInTriangleIntersectionAttributes attr) {
    uint   instanceIdx = InstanceID();
    uint   primIdx     = PrimitiveIndex();
    float3 rayOrigin   = WorldRayOrigin();

    HitInfo info = EvalSurfaceState(instanceIdx, primIdx, attr.barycentrics, rayOrigin);

    CompressToPayload(info, payload);

}