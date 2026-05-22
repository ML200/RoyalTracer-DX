#include "Includes_v8.hlsli"

//====================================
//CLOSEST HIT STUB
//====================================
//unused, raygen uses SER HitObject, D3D12 still requires the symbol
// PLANET_INTEGRATION: this stays an empty stub — NOT the terrain shading hook.
// The terrain material branch lives in Pass_raygen_v8.hlsl's HIT block.
[shader("closesthit")]
void ClosestHit(inout TracePayload payload, in BuiltInTriangleIntersectionAttributes attr) {}
