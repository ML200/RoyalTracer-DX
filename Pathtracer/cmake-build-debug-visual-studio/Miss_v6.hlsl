#include "Common_v6.hlsl"

[shader("miss")] void Miss(inout HitInfo payload
                           : SV_RayPayload) {
  uint2 launchIndex = DispatchRaysIndex().xy;
  float2 dims = float2(DispatchRaysDimensions().xy);
}