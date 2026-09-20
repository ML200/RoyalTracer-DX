#include "Includes_v8.hlsli"

// Never invoked: the raygens shade every hit themselves from the hit object (PtShadeHit in
// PtVertex_v8.hlsli) and never call HitObject::Invoke. It exists because the driver fails
// CreateStateObject for a raytracing pipeline in which no hit group has a closest-hit shader.
[shader("closesthit")]
void ClosestHit(inout TracePayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
}
