#include "Includes_v8.hlsli"

// Never invoked; CreateStateObject fails without a closest-hit shader.
[shader("closesthit")]
void ClosestHit(inout TracePayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
}
