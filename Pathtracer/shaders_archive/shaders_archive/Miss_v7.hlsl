#include "Includes_v7.hlsli"

[shader("miss")] void Miss(inout HitInfo payload
                           : SV_RayPayload) {
    //uint2 launchIndex = DispatchRaysIndex().xy;
    //float2 dims = float2(DispatchRaysDimensions().xy);
    payload.materialID = 0xFFFFFFFF;
    payload.hitPosition = float3(0.4f, 0.4f, 0.5f); // We save the color of our sky into the position
}