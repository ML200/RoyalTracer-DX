#include "Includes_v8.hlsli"
#include "OceanVolume.hlsli"
#include "PtVertex_v8.hlsli"

// Invoked only by Pass_pt_trace_v8; miss table order is PT_MISS_SKY, PT_MISS_PRIMARY, PT_MISS_PRIMARY_LEAN.

// Sky and sun of an escaped path ray; io.pos is the vertex it left.
[shader("miss")]
void Miss(inout PtVertexIO io)
{
    PtShadeMiss(io, io.pos, WorldRayDirection());
}

// The camera record's vertex, through a MakeMiss HitObject; the ray runs from the camera to it.
[shader("miss")]
void PtPrimaryShade(inout PtVertexIO io)
{
    PtShadePrimary(io, WorldRayDirection(), DispatchRaysIndex().xy);
}
