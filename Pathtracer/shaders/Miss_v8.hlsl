#include "Includes_v8.hlsli"

//====================================================================
//MISS STUB
//====================================================================
//Raygen uses SER HitObject path, miss never executes.
//D3D12 still requires a valid miss symbol in the raytracing pipeline.
[shader("miss")] void Miss(inout TracePayload payload) {}
