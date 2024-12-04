#include "Common.hlsl"

#define kernelSize 5
#define halfKernel 2

// Raytracing output texture, accessed as a UAV
RWTexture2DArray<float4> gOutput : register(u0);
RWTexture2D<float4> gPermanentData : register(u1);

// Raytracing acceleration structure, accessed as a SRV
RaytracingAccelerationStructure SceneBVH : register(t0);

// #DXR Extra: Perspective Camera
cbuffer CameraParams : register(b0)
{
    float4x4 view;
    float4x4 projection;
    float4x4 viewI;
    float4x4 projectionI;
    float4x4 prevView;        // Previous frame's view matrix (can be removed if not used elsewhere)
    float4x4 prevProjection;  // Previous frame's projection matrix (can be removed if not used elsewhere)
    float time;
    float4 frustumLeft;       // Left frustum plane
    float4 frustumRight;      // Right frustum plane
    float4 frustumTop;        // Top frustum plane
    float4 frustumBottom;     // Bottom frustum plane
    float4 prevFrustumLeft;   // Previous frame's frustum planes (can be removed if not used elsewhere)
    float4 prevFrustumRight;
    float4 prevFrustumTop;
    float4 prevFrustumBottom;
}

// HELPERS

// Helper function to calculate the dot product between two vectors and check if the angle is less than 10 degrees
bool isSimilarNormal(float3 normalA, float3 normalB) {
    float dotProduct = dot(normalA, normalB);
    return dotProduct > cos(radians(10.0f)); // Check if the angle is less than 10 degrees
}

// Helper function to check if the reflectiveness difference is within 0.3
bool isSimilarReflectiveness(float reflectA, float reflectB) {
    return abs(reflectA - reflectB) < 0.3f;
}

// HELPERS

[shader("raygeneration")]
void RayGen() {
    // Get the location within the dispatched 2D grid of work items (often maps to pixels, so this could represent a pixel coordinate).
    uint2 launchIndex = DispatchRaysIndex().xy;
    float2 dims = float2(DispatchRaysDimensions().xy);

    // #DXR Extra: Perspective Camera
    float aspectRatio = dims.x / dims.y;

    // Initialize the ray origin and direction
    float3 init_orig = mul(viewI, float4(0, 0, 0, 1));
    float3 accumulation = float3(0, 0, 0);

    uint samples = 1;
    uint bounces = 1;
    uint rr_threshold = 3;


    // SEEDING
    const uint prime1_x = 73856093u;
    const uint prime2_x = 19349663u;
    const uint prime3_x = 83492791u;
    const uint prime1_y = 37623481u;
    const uint prime2_y = 51964263u;
    const uint prime3_y = 68250729u;
    const uint prime_time_x = 293803u;
    const uint prime_time_y = 423977u;

    HitInfo payload;

    payload.colorAndDistance = float4(1.0f, 1.0f, 1.0f, 0.0f);
    payload.emission = float3(0.0f, 0.0f, 0.0f);
    payload.origin = init_orig;

    float2 d = ((launchIndex.xy / dims.xy) * 2.f - 1.f);
    float4 target = mul(projectionI, float4(d.x, -d.y, 1, 1));
    float3 init_dir = mul(viewI, float4(target.xyz, 0));

    // Shoot the initial ray only once; significant performance increase, ca. 1/3
    // TODO: perform single sample as long as no stochastic integration point is reached (e.g. perfect reflection/refraction)


    // Path tracing: x samples for y bounces
    for (int x = 0; x < samples; x++) {
        payload.seed.x = launchIndex.y * prime1_x ^ launchIndex.x * prime2_x ^ uint(samples + 1) * prime3_x ^ uint(time) * prime_time_x;
        payload.seed.y = launchIndex.x * prime1_y ^ launchIndex.y * prime2_y ^ uint(samples + 1) * prime3_y ^ uint(time) * prime_time_y;
        // Initialize the new payload
        payload.colorAndDistance = float4(1.0f, 1.0f, 1.0f, 0.0f);
        payload.emission = float3(0.0f, 0.0f, 0.0f);
        payload.origin = init_orig;
        payload.direction = init_dir;
        payload.pdf = 1.0f;

        for (int y = 0; y < bounces; y++) {
            RayDesc ray;
            ray.Origin = payload.origin;
            ray.Direction = payload.direction;
            ray.TMin = 0.0001;
            ray.TMax = 10000;

            payload.util.x = 0;
            payload.util.y = float(y);

            // Trace the ray
            TraceRay(SceneBVH, RAY_FLAG_NONE, 0xFF, 0, 0, 0, ray, payload);

            // If the last ray missed, terminate loop
            if (payload.util.x >= 1.0f) {
                break;
            }

            // Russian roulette: terminate the ray if the further accumulation of light would yield diminishing improvements due to a dark throughput
            if(y > rr_threshold){
                float throughput = (payload.colorAndDistance.x + payload.colorAndDistance.y + payload.colorAndDistance.z)/3.0f;
                float random = RandomFloat(payload.seed);

                if(throughput < random){
                    break;
                }

                payload.colorAndDistance.xyz *= 1.0f/random;

            }


        }

        accumulation += payload.emission;
    }

    accumulation /= samples;


    //TEMPORAL ACCUMULATION  ________________________________________________________________________________________________________
    int maxFrames = 10000000;
    float frameCount = gPermanentData[uint2(launchIndex)].w;

    // Check if the frame count is zero or uninitialized
    if (frameCount <= 0.0f && !IsNaN(accumulation.x) && !IsNaN(accumulation.y) && !IsNaN(accumulation.z))
    {
        // Initialize the accumulation buffer and frame count
        gPermanentData[uint2(launchIndex)] = float4(accumulation, 1.0f);
    }
    else if (frameCount < maxFrames && !IsNaN(accumulation.x) && !IsNaN(accumulation.y) && !IsNaN(accumulation.z))
    {
        // Continue accumulating valid samples
        gPermanentData[uint2(launchIndex)].xyz += accumulation;
        gPermanentData[uint2(launchIndex)].w += 1.0f;
    }

    // Compare the view matrices and reset if different (your existing code)
    bool different = false;
    for (int row = 0; row < 4; row++)
    {
        float4 diff = abs(view[row] - prevView[row]);
        if (any(diff > s_bias))
        {
            different = true;
            break;
        }
    }

    if (different)
    {
        // Reset buffers
        gPermanentData[uint2(launchIndex)] = float4(accumulation, 1.0f);
        frameCount = 1.0f; // Update frameCount to reflect the reset
    }

    // Safely calculate the averaged color
    frameCount = max(frameCount, 1.0f); // Ensure frameCount is at least 1 to avoid division by zero
    float3 averagedColor = gPermanentData[uint2(launchIndex)].xyz / frameCount;
    //TEMPORAL ACCUMULATION  ________________________________________________________________________________________________________


    // Output the final color to layer 0
    gOutput[uint3(launchIndex, 0)] = float4(abs(averagedColor), 1.0f);
}
