#include "Common.hlsl"

[shader("miss")] void Miss(inout HitInfo payload
                           : SV_RayPayload) {
  uint2 launchIndex = DispatchRaysIndex().xy;
  float2 dims = float2(DispatchRaysDimensions().xy);

  float ramp = launchIndex.y / dims.y;
  payload.colorAndDistance *= float4(1.0f, 1.0f, 1.0f /*- 0.1f * ramp*/, -1.0f); //background color
  payload.emission += float3(.0f,.0f,.0f) * payload.colorAndDistance.xyz;
  payload.util = 1.0f;
}