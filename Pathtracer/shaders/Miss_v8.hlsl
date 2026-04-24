#include "Includes_v8.hlsli"

//====================================
//MISS STUB
//====================================
//raygen uses SER HitObject path, miss never runs, D3D12 still requires the symbol
[shader("miss")] void Miss(inout TracePayload payload) {}
