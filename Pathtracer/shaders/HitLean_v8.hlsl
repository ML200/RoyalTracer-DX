// Lean path-tracer shading: no sea and no water medium, so the ocean and water code compiles out
// (light sampling and the SSS walk live in PtCallables_v8.hlsl). Pass_pt_trace_v8 picks it per vertex
// (PT_HIT_LEAN_RECORD, PT_MISS_PRIMARY_LEAN) and sends the water cases to Hit_v8/Miss_v8.
#define PT_HIT_LEAN 1
#include "Includes_v8.hlsli"
#include "OceanVolume.hlsli"
#include "PtVertex_v8.hlsli"

[shader("closesthit")]
void ClosestHitLean(inout PtVertexIO io, in BuiltInTriangleIntersectionAttributes attr)
{
    PtShadeHit(io, InstanceID(), GeometryIndex(), PrimitiveIndex(), attr.barycentrics, RayTCurrent(),
        WorldRayDirection(), DispatchRaysIndex().xy);
}

// The camera record's vertex off the sea, through a MakeMiss HitObject.
[shader("miss")]
void PtPrimaryShadeLean(inout PtVertexIO io)
{
    PtShadePrimary(io, WorldRayDirection(), DispatchRaysIndex().xy);
}
