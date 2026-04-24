#include "Includes_v8.hlsli"

//====================================
//CLOSEST-HIT STUB
//====================================
//raygen uses SER HitObject path, closest-hit never runs, D3D12 still requires the symbol
[shader("closesthit")]
void ClosestHit(inout TracePayload payload, in BuiltInTriangleIntersectionAttributes attr) {}
