#include "Includes_v8.hlsli"

// The trace raygen shades every vertex itself after the reorder (PtShadeHit in PtVertex_v8.hlsli);
// this closest-hit shader only exists so the hit groups have one, and is never invoked.
[shader("closesthit")]
void ClosestHit(inout TracePayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
}
