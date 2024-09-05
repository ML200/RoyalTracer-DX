#include "Common.hlsl"

// Raytracing output texture, accessed as a UAV
RWTexture2DArray<float4> gOutput : register(u0);

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
  // Define a ray, consisting of origin, direction, and the min-max distance
  // values
  // #DXR Extra: Perspective Camera
  float aspectRatio = dims.x / dims.y;

    float3 init_orig = mul(viewI, float4(0, 0, 0, 1));
    //We have to collect the intensity and color on the path:
    float3 accumulation = float3(0,0,0);

  //Pathtracing: x samples for y bounces
  float samples = 3;
  for(int x = 0; x < samples; x++){
      HitInfo payload;
      // Initialize the ray payload
      payload.colorAndDistance = float4(1, 1, 1, 0);
      payload.emission = float3(0, 0, 0);
      payload.util.x = 0;
      payload.util.y = x;
      payload.origin = init_orig;

      //SEEDING
      // Use large prime numbers to scale coordinates and the sample index for each component
      const uint prime1_x = 73856093u; // Prime for x coordinate (component 1)
      const uint prime2_x = 19349663u; // Prime for y coordinate (component 1)
      const uint prime3_x = 83492791u; // Prime for sample index (component 1)

      const uint prime1_y = 37623481u; // Prime for x coordinate (component 2)
      const uint prime2_y = 51964263u; // Prime for y coordinate (component 2)
      const uint prime3_y = 68250729u; // Prime for sample index (component 2)

      // Additional prime for time to ensure variability over time for both components
      const uint prime_time_x = 293803u;
      const uint prime_time_y = 423977u;

      payload.seed.x = launchIndex.y * prime1_x ^ launchIndex.x * prime2_x ^ x * prime3_x ^ uint(time) * prime_time_x;
      payload.seed.y = launchIndex.x * prime1_y ^ launchIndex.y * prime2_y ^ x * prime3_y ^ uint(time) * prime_time_y;

      float jitterX = RandomFloatLCG(payload.seed.x);
      float jitterY = RandomFloatLCG(payload.seed.y);

      float2 d = (((launchIndex.xy + float2(jitterX, jitterY)) / dims.xy) * 2.f - 1.f);
      float4 target = mul(projectionI, float4(d.x, -d.y, 1, 1));
      float3 init_dir = mul(viewI, float4(target.xyz, 0));

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
          if(payload.util.x == 1.0f){
            break;
          }

        // Apply Russian Roulette after a minimum number of bounces
        //______________________________________________________________________________________________________________
        // Assuming 'throughput' is a float3 representing the accumulated light contribution (RGB)
        if(y > 3){
            float p = max(payload.emission.x, max(payload.emission.y, payload.emission.z)); // Max component of throughput

            // Ensure 'p' is within a sensible range to avoid division by zero or extremely low probabilities
            p = max(p, 0.05f); // Ensure there's at least a 5% chance to continue, adjust as needed

            float randomValue = RandomFloat(payload.seed); // Generate a random value for Russian Roulette

            // Randomly terminate the path with a probability inversely equal to 'p'
            if (randomValue > p) {
                break; // Terminate the path
            }
            // If the path continues, adjust the throughput to compensate for the paths terminated (removed for now)
            payload.colorAndDistance.xyz /= p;
        }
        else{
            if (y==0){
                gOutput[uint3(launchIndex, 10)] = float4(payload.emission, 1.0f);
            }
            else if (y==1){
                gOutput[uint3(launchIndex, 11)] = float4(payload.emission-gOutput[uint3(launchIndex, 10)], 1.0f);
            }
            else if (y==2){
                gOutput[uint3(launchIndex, 12)] = float4(payload.emission-gOutput[uint3(launchIndex, 11)]-gOutput[uint3(launchIndex, 10)], 1.0f);
            }
            else if (y==3){
                gOutput[uint3(launchIndex, 13)] = float4(payload.emission-gOutput[uint3(launchIndex, 12)]-gOutput[uint3(launchIndex, 11)]-gOutput[uint3(launchIndex, 10)], 1.0f);
            }
        }
        //______________________________________________________________________________________________________________
      }
      accumulation += payload.emission;
  }
  accumulation/=samples;


    // First, read the color from the previous frame [1] before it gets overwritten and start the accumulation
    float3 temporalAccumulation = float3(0.0f,0.0f,0.0f);//accumulation;

    // Loop to shift entries one position ahead while accumulating the existing data
    // Skipping the last slot as an example, assuming we only keep a history of 9 frames
    for (int i = 8; i >= 1; i--) {
        // Read color from the next slot
        float4 prevColor = gOutput[uint3(launchIndex, i)];
        // Accumulate colors
        temporalAccumulation += prevColor.xyz;
        // Shift color to the next slot
        gOutput[uint3(launchIndex, i + 1)] = prevColor;
    }

    // Normalize the accumulated color by the number of accumulated frames
    //temporalAccumulation /= 10.0f;
    temporalAccumulation /= 9.0f;


    // Number of À-Trous wavelet iterations
    int iterations = 3;
    // Edge sensitivity (for edge-avoiding behavior)
    float sigma_color = 0.1f;
    // Perform Edge Avoiding À-Trous Wavelet Transform to denoise the image
    float3 denoisedColor = A_TrousWaveletWithHistory(launchIndex, gOutput, dims, sigma_color, iterations);


    // Write the accumulated color to the current frame's slot [1]
    gOutput[uint3(launchIndex, 1)] = float4(accumulation, 1.0f);

    // Write the normalized (averaged) accumulated color into the output frame buffer [0]
    gOutput[uint3(launchIndex, 0)] = float4(temporalAccumulation, 1.0f);
}
