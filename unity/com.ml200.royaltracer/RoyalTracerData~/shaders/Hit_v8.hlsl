#include "Includes_v8.hlsli"
#include "OceanVolume.hlsli"
#include "PtVertex_v8.hlsli"

// Path-tracer material shading for water paths and sea hits (the rest: HitLean_v8.hlsl), invoked by
// Pass_pt_trace_v8. Other passes trace with HitObject::TraceRay and never invoke it; their anyhit keeps
// the small TracePayload.
[shader("closesthit")]
void ClosestHit(inout PtVertexIO io, in BuiltInTriangleIntersectionAttributes attr)
{
    PtShadeHit(io, InstanceID(), HIT_CLUSTER_ID(), GeometryIndex(), PrimitiveIndex(), attr.barycentrics, ObjectToWorld3x4(),
        RayTCurrent(), WorldRayDirection(), DispatchRaysIndex().xy);
}
