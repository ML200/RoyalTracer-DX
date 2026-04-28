#include "Includes_v8.hlsli"

//====================================
//MISS STUB
//====================================
//unused, raygen uses SER HitObject, D3D12 still requires the symbol
[shader("miss")] void Miss(inout TracePayload payload) {}
