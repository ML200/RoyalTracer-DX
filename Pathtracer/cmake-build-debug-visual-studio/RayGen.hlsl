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
  float time;
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

    float3 init_orig = mul(viewI, float4(0, 0, 0, 1));
    float4 target = mul(projectionI, float4(d.x, -d.y, 1, 1));
    float3 init_dir = mul(viewI, float4(target.xyz, 0));

    //We have to collect the intensity and color on the path:
    float3 accumulation = float3(0,0,0);

  //Pathtracing: x samples for y bounces
  float samples = 1;
  for(int x = 0; x < samples; x++){
      HitInfo payload;
      // Initialize the ray payload
      payload.colorAndDistance = float4(1, 1, 1, 0);
      payload.emission = float3(0, 0, 0);
      payload.util = 0;
      payload.origin = init_orig;
      payload.seed = launchIndex.x * 73856093u ^ launchIndex.y * 19349663u ^ x * 83492791u ^ uint(time) * 1859303u;
      payload.direction = init_dir;
      payload.pdf = 1.0f;

      for(int y = 0; y < 6; y++){
          RayDesc ray;
          ray.Origin = payload.origin;
          ray.Direction = payload.direction;
          ray.TMin = 0.0001;
          ray.TMax = 10000;
          // Trace the ray
          TraceRay(SceneBVH,RAY_FLAG_NONE,0xFF,0,0,0, ray, payload);

          //If the last ray missed, terminate loop:
          if(payload.util == 1.0f){
            break;
          }

        // Apply Russian Roulette after a minimum number of bounces
        //______________________________________________________________________________________________________________
        // Assuming 'throughput' is a float3 representing the accumulated light contribution (RGB)
        if(y > 3){
            float p = max(payload.emission.x, max(payload.emission.y, payload.emission.z)); // Max component of throughput

            // Ensure 'p' is within a sensible range to avoid division by zero or extremely low probabilities
            p = max(p, 0.3f); // Ensure there's at least a 5% chance to continue, adjust as needed

            float randomValue = RandomFloat(payload.seed); // Generate a random value for Russian Roulette

            // Randomly terminate the path with a probability inversely equal to 'p'
            if (randomValue > p) {
                break; // Terminate the path
            }
            // If the path continues, adjust the throughput to compensate for the paths terminated (removed for now)
            //payload.colorAndDistance.xyz /= p;
        }
        //______________________________________________________________________________________________________________
      }
      accumulation += payload.emission;
  }
  accumulation/=samples;
  gOutput[launchIndex] = float4(accumulation.xyz, 1.f);
}
