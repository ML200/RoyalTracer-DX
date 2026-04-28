#include "Includes_v8.hlsli"

//====================================
//CLOSEST HIT STUB
//====================================
//unused, raygen uses SER HitObject, D3D12 still requires the symbol
[shader("closesthit")]
void ClosestHit(inout TracePayload payload, in BuiltInTriangleIntersectionAttributes attr) {}
