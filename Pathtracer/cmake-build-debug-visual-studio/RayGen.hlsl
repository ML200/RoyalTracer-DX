#include "Common.hlsl"

// Raytracing output texture, accessed as a UAV
RWTexture2D<float4> gOutput : register(u0);

// Raytracing acceleration structure, accessed as a SRV
RaytracingAccelerationStructure SceneBVH : register(t0);

// #DXR Extra: Perspective Camera
cbuffer CameraParams : register(b0)
{
  float4x4 view;
  float4x4 projection;
  float4x4 viewI;
  float4x4 projectionI;
}


[shader("raygeneration")] void RayGen() {
  // Get the location within the dispatched 2D grid of work items
  // (often maps to pixels, so this could represent a pixel coordinate).
  uint2 launchIndex = DispatchRaysIndex().xy;
  float2 dims = float2(DispatchRaysDimensions().xy);
  float2 d = (((launchIndex.xy + 0.5f) / dims.xy) * 2.f - 1.f);
  // Define a ray, consisting of origin, direction, and the min-max distance
  // values
  // #DXR Extra: Perspective Camera
  float aspectRatio = dims.x / dims.y;

    float4 init_orig = mul(viewI, float4(0, 0, 0, 1));
    float4 target = mul(projectionI, float4(d.x, -d.y, 1, 1));
    float4 init_dir = mul(viewI, float4(target.xyz, 0));

    //We have to collect the intensity and color on the path:
    float3 accumulation = float3(0,0,0);

  //Pathtracing: x samples for y bounces
  for(int x = 0; x < 3; x++){
      HitInfo payload;
      // Initialize the ray payload
      payload.colorAndDistance = float4(1, 1, 1, 0);
      payload.emission = float3(0, 0, 0);
      payload.util = float4(0, 0, 0, 0);
      payload.origin = init_orig.xyz;
      payload.direction = init_dir.xyz;

      for(int y = 0; y < 5; y++){
          RayDesc ray;
            // Use a hybrid approach for generating random seeds
          payload.util.x = HybridTaus(Hash(launchIndex.x * 1973 + launchIndex.y * 9277 + y * 26699 + x * 12289));
          payload.util.y = HybridTaus(Hash(launchIndex.x * 2749 + launchIndex.y * 5923 + y * 23893 + x * 3229));

          if(y == 0){
              ray.Origin = init_orig;
              ray.Direction = init_dir;
          }
          else{
              ray.Origin = payload.origin;
              ray.Direction = payload.direction;
          }
          ray.TMin = 0.00000001;
          ray.TMax = 100000;
          // Trace the ray
          TraceRay(SceneBVH,RAY_FLAG_NONE,0xFF,0,0,0, ray, payload);

          //If the last ray missed, terminate loop:
          if(payload.util.x == 1.0f){
            break;
          }
      }
      accumulation += payload.emission * payload.colorAndDistance.xyz;

  }
  accumulation/=3.0f;
  gOutput[launchIndex] = float4(accumulation.xyz, 1.f);
}
