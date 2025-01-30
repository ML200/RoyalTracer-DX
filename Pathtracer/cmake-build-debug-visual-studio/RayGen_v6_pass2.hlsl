#include "Common_v6.hlsl"
#include "GGX_v6.hlsl"
#include "Lambertian_v6.hlsl"
#include "BRDF_v6.hlsl"
#include "Reservoir_v6.hlsl"

// Raytracing output texture, accessed as a UAV
RWTexture2DArray<float4> gOutput : register(u0);
RWTexture2D<float4> gPermanentData : register(u1);

RWStructuredBuffer<Reservoir> g_Reservoirs_current : register(u2);
RWStructuredBuffer<Reservoir> g_Reservoirs_last : register(u3);

StructuredBuffer<STriVertex> BTriVertex : register(t2);
StructuredBuffer<int> indices : register(t1);
RaytracingAccelerationStructure SceneBVH : register(t0);
StructuredBuffer<InstanceProperties> instanceProps : register(t3);
StructuredBuffer<uint> materialIDs : register(t4);
StructuredBuffer<Material> materials : register(t5);
StructuredBuffer<LightTriangle> g_EmissiveTriangles : register(t6);

#include "Sampler_v6.hlsl"

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

// Second raygen shader is the ReSTIR pass. The reservoirs were filled in the first shader, now we recombine them

[shader("raygeneration")]
void RayGen2() {
    // Get the location within the dispatched 2D grid of work items (often maps to pixels, so this could represent a pixel coordinate).
    uint2 launchIndex = DispatchRaysIndex().xy;
    uint pixelIdx = launchIndex.y * DispatchRaysDimensions().x + launchIndex.x;
    float2 dims = float2(DispatchRaysDimensions().xy);

    // #DXR Extra: Perspective Camera
    float aspectRatio = dims.x / dims.y;

    // Initialize the ray origin and direction
    float3 init_orig = mul(viewI, float4(0, 0, 0, 1));


    // SEEDING
    const uint prime1_x = 73856093u;
    const uint prime2_x = 19349663u;
    const uint prime3_x = 83492791u;
    const uint prime1_y = 37623481u;
    const uint prime2_y = 51964263u;
    const uint prime3_y = 68250729u;
    const uint prime_time_x = 293803u;
    const uint prime_time_y = 423977u;

    // Initialize once, to reduce allocs with several samples per frame
    HitInfo payload;
    uint2 seed;
    float3 accumulation = float3(0, 0, 0);

    //_______________________________SETUP__________________________________
    // Every sample recieves a new seed (use samples+2 here to get different random numbers then the RayGen1 shader)
    seed.x = launchIndex.y * prime1_x ^ launchIndex.x * prime2_x ^ uint(samples + 2) * prime3_x ^ uint(time) * prime_time_x;
    seed.y = launchIndex.x * prime1_y ^ launchIndex.y * prime2_y ^ uint(samples + 2) * prime3_y ^ uint(time) * prime_time_y;

    // The current reservoir
    Reservoir reservoir_current = g_Reservoirs_current[pixelIdx];
    Reservoir reservoir_last = g_Reservoirs_last[pixelIdx];

    if(reservoir_current.finalColor.x == 0.0f && reservoir_current.finalColor.y == 0.0f && reservoir_current.finalColor.z == 0.0f){
        //_______________________________RESTIR_TEMPORAL__________________________________
        // Temporal reuse
        if((reservoir_current.p_hat > 0.0f && reservoir_current.M > 0.0f) || (reservoir_last.p_hat > 0.0f && reservoir_last.M > 0.0f)){
            Reservoir temporal_res = {
               (float3)0.0f,  // f
               0.0f,          // p_hat
               (float3)0.0f,  // direction
               0.0f,          // distance
               (float3)0.0f,   // hitPos
               (float3)0.0f,   // hitNormal
               false,         // v_eval
               1.0f,          // v
               0.0f,          // w_sum
               0.0f,          // w_i
               0.0f,           // M
               (float3)0.0f   // Final color
             };

            float M_c = min(20, reservoir_current.M);
            float M_t = min(20, reservoir_last.M);

            float mi_c = M_c / (M_c + M_t);
            float mi_t = M_t / (M_c + M_t);

            // Calculate the weight for the given sample: w * p_hat * W
            float w_c = mi_c * reservoir_current.p_hat * reservoir_current.w_i * reservoir_current.v;
            float w_t = mi_t * reservoir_last.p_hat * reservoir_last.w_i;

            UpdateReservoir(
                temporal_res,
                w_c,
                reservoir_current.M,
                seed,
                reservoir_current.f,
                reservoir_current.p_hat,
                reservoir_current.v_eval,
                reservoir_current.v,
                reservoir_current.direction,
                reservoir_current.dist,
                reservoir_current.hitPos,
                reservoir_current.hitNormal
            );
            UpdateReservoir(
                temporal_res,
                w_t,
                reservoir_last.M,
                seed,
                reservoir_last.f,
                reservoir_last.p_hat,
                reservoir_last.v_eval,
                reservoir_last.v,
                reservoir_last.direction,
                reservoir_last.dist,
                reservoir_last.hitPos,
                reservoir_last.hitNormal
            );

            reservoir_current = temporal_res;
        }

        accumulation += reservoir_current.f * reservoir_current.w_i;


        float frameCount = gPermanentData[uint2(launchIndex)].w;
        //TEMPORAL ACCUMULATION  ___________________________________________________________________________________________
        int maxFrames = 100000000;

        // Check if the frame count is zero or uninitialized
        if (frameCount <= 0.0f && !isnan(accumulation.x) && !isnan(accumulation.y) && !isnan(accumulation.z) && isfinite(accumulation.x) && isfinite(accumulation.y) && isfinite(accumulation.z))
        {
            // Initialize the accumulation buffer and frame count
            gPermanentData[uint2(launchIndex)] = float4(accumulation, 1.0f);
        }
        else if (frameCount < maxFrames && !isnan(accumulation.x) && !isnan(accumulation.y) && !isnan(accumulation.z) && isfinite(accumulation.x) && isfinite(accumulation.y) && isfinite(accumulation.z))
        {
            // Continue accumulating valid samples
            gPermanentData[uint2(launchIndex)].xyz += accumulation;
            gPermanentData[uint2(launchIndex)].w += 1.0f;
        }

        // Safely calculate the averaged color
        frameCount = max(frameCount, 1.0f); // Ensure frameCount is at least 1 to avoid division by zero
        float3 averagedColor = gPermanentData[uint2(launchIndex)].xyz / frameCount;
        //TEMPORAL ACCUMULATION  ___________________________________________________________________________________________

        //averagedColor = accumulation;


        //__________________________________________ReSTIR__________________________________________________
        // Compare the view matrices and reset if different
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

            g_Reservoirs_last[pixelIdx] = reservoir_current;
            g_Reservoirs_current[pixelIdx] = reservoir_current;
        }
        else{
            // Save the current reservoir and rotate the reservoirbuffer
            g_Reservoirs_last[pixelIdx] = reservoir_current;
        }

        // Output the final color to layer 0
        gOutput[uint3(launchIndex, 0)] = float4(averagedColor, 1.0f);
    }
    else{
        gOutput[uint3(launchIndex, 0)] = float4(reservoir_current.finalColor, 1.0f);
    }
}
