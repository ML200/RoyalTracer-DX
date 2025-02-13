#include "Common_v6.hlsl"

[shader("miss")] void Miss(inout HitInfo payload
                           : SV_RayPayload) {
  //uint2 launchIndex = DispatchRaysIndex().xy;
  //float2 dims = float2(DispatchRaysDimensions().xy);
  payload.materialID = 4294967294; // uint_max-1, unlikely this many materials are loaded (impossible in gpu memory)
}