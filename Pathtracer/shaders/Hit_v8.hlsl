#include "Includes_v8.hlsli"

//====================================================================
//CLOSEST-HIT STUB
//====================================================================
//Raygen uses SER HitObject path, closest-hit never executes.
//D3D12 still requires a valid closest-hit symbol for the hit group.
[shader("closesthit")]
void ClosestHit(inout TracePayload payload, in BuiltInTriangleIntersectionAttributes attr) {}
