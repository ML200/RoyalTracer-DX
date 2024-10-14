#include "Common.hlsl"

#define kernelSize 5
#define halfKernel 2

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
    float3 initialHit = init_orig;

    // Path tracing: x samples for y bounces
    float samples = 1;
    for (int x = 0; x < samples; x++) {
        HitInfo payload;
        payload.colorAndDistance = float4(1, 1, 1, 0);
        payload.emission = float3(0, 0, 0);
        payload.util.x = 0;
        payload.util.y = x;
        payload.origin = init_orig;

        // SEEDING
        const uint prime1_x = 73856093u;
        const uint prime2_x = 19349663u;
        const uint prime3_x = 83492791u;
        const uint prime1_y = 37623481u;
        const uint prime2_y = 51964263u;
        const uint prime3_y = 68250729u;
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

        for (int y = 0; y < 6; y++) {
            RayDesc ray;
            ray.Origin = payload.origin;
            ray.Direction = payload.direction;
            ray.TMin = 0.0001;
            ray.TMax = 10000;

            // Trace the ray
            TraceRay(SceneBVH, RAY_FLAG_NONE, 0xFF, 0, 0, 0, ray, payload);

            // If the last ray missed, terminate loop
            if (payload.util.x == 1.0f) {
                break;
            }

            if (y == 0) {
                // Store normal and reflectiveness as in the original code
                gOutput[uint3(launchIndex, 10)] = float4(payload.hitNormal, 1.0f);
                gOutput[uint3(launchIndex, 11)] = float4(payload.reflectiveness, payload.reflectiveness, payload.reflectiveness, 1.0f);
                initialHit = payload.origin;
            }

            if (y == 1) {
                gOutput[uint3(launchIndex, 12)] = float4(payload.hitNormal, 1.0f);
                gOutput[uint3(launchIndex, 13)] = float4(payload.reflectiveness, payload.reflectiveness, payload.reflectiveness, 1.0f);
            }

            if (y == 2) {
                gOutput[uint3(launchIndex, 14)] = float4(payload.hitNormal, 1.0f);
                gOutput[uint3(launchIndex, 15)] = float4(payload.reflectiveness, payload.reflectiveness, payload.reflectiveness, 1.0f);
            }
            if (y == 3) {
                gOutput[uint3(launchIndex, 16)] = float4(payload.hitNormal, 1.0f);
                gOutput[uint3(launchIndex, 17)] = float4(payload.reflectiveness, payload.reflectiveness, payload.reflectiveness, 1.0f);
            }
        }

        accumulation += payload.emission;
    }

    accumulation /= samples;

    // DENOISE ______________________________________________________________________________________________________________________

    float3 temporalAccumulation = float3(0.0f, 0.0f, 0.0f);
    int usablePixels = 0;

    // Loop to shift entries one position ahead while accumulating the existing data
    for (int i = 8; i >= 1; i--) {
        // Shift previous accumulations
        gOutput[uint3(launchIndex, i + 1)] = gOutput[uint3(launchIndex, i)];

        // Accumulate color from previous frames at the same pixel position
        float4 prevColor = gOutput[uint3(launchIndex, i)];
        temporalAccumulation += prevColor.xyz;
        usablePixels++;
    }

    // Store the current frame's accumulation in layer 1
    gOutput[uint3(launchIndex, 1)] = float4(accumulation, 1.0f);

    // Calculate the average color over the accumulated frames
    float3 averagedColor = (temporalAccumulation) / (usablePixels);

    // Output the final color to layer 0
    gOutput[uint3(launchIndex, 0)] = float4(averagedColor, 1.0f);
}
