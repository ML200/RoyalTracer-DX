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

[shader("raygeneration")]
void RayGen() {
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


    //_______________________________RESERVOIRS__________________________________
    Reservoir reservoir = {
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
      0.0f           // M
    };

    // The current reservoir (offset by one frame)
    Reservoir reservoir_current = g_Reservoirs_current[pixelIdx];
    Reservoir reservoir_last = g_Reservoirs_last[pixelIdx];


    //_______________________________SETUP__________________________________
    // Every sample recieves a new seed
    seed.x = launchIndex.y * prime1_x ^ launchIndex.x * prime2_x ^ uint(samples + 1) * prime3_x ^ uint(time) * prime_time_x;
    seed.y = launchIndex.x * prime1_y ^ launchIndex.y * prime2_y ^ uint(samples + 1) * prime3_y ^ uint(time) * prime_time_y;

    // Calculate the initial ray direction. Jitter is used to randomize the rays intersection point on subpixel level
    float jitterX = RandomFloat(seed);
    float jitterY = RandomFloat(seed);
    float2 d = (((launchIndex.xy + float2(jitterX, jitterY)) / dims.xy) * 2.f - 1.f);
    float4 target = mul(projectionI, float4(d.x, -d.y, 1, 1));
    float3 init_dir = mul(viewI, float4(target.xyz, 0));


    //_________________________PATH_VARIABLES______________________________
    float3 throughput = float3(1,1,1);
    float3 emission = float3(0,0,0);
    float3 direction = init_dir;
    float3 origin = init_orig;


    //____________________________CAMERA_RAY________________________________
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = 0.0001;
    ray.TMax = 10000;

    // Trace the camera ray
    TraceRay(SceneBVH, RAY_FLAG_NONE, 0xFF, 0, 0, 0, ray, payload);
    Material hitMaterial = materials[payload.materialID];

    bool performSampling = true;
    // Check if we hit a light source
    if((hitMaterial.Ke.x + hitMaterial.Ke.y + hitMaterial.Ke.z) > 0.0f){
        emission = materials[payload.materialID].Ke;
        accumulation = emission;
        performSampling = false;
    }

    //_______________________________PATH_SAMPLING__________________________________
    // Perform y bounces
    if(performSampling){
        for (int y = 0; y < bounces; y++) {
            /*
            Pathtracing concept:
            To support advanced algorithms like ReSTIR, the path tracing will be always structured the same. Most of the work will be
            done in the raygeneration shader. Sampling works always the same:
            - Evaluate direct NEE: Perform light sampling using NEE. This can be done with or without visibility check which will significantly speed up RIS
            - Evaluate direct BSDF: Sample the BSDF to evaluate direct light contribution
            - Evaluate indirect BSDF: Sample the BSDF to trace a ray with several bounces accumulating the pdf (ReSTIR GI/PT)
            */

            //_______________________________RIS_DIRECT_ILLUMINATION__________________________________

            SampleRIS(
                10,
                1,
                -direction,
                reservoir,
                payload,
                seed
                );

            //_______________________________RESTIR_DI__________________________________
            // Fetch spatial pixel ids
            /*uint spatial_pixels[5];
            float p_sum = 0.0f;
            for(int j = 0; j<5; j++){
                spatial_pixels[j] = GetRandomPixelArea(30, DispatchRaysDimensions().x, DispatchRaysDimensions().y, launchIndex.x, launchIndex.y, seed);
                p_sum += g_Reservoirs_current[spatial_pixels[j]].p_hat;
            }

            if(p_sum > 0.0f){
                float w_c = reservoir_current.p_hat / (p_sum + reservoir_current.p_hat);
                WeightReservoir(reservoir_current, w_c);

                // Combine with spatial reservoirs
                for(int k = 0; k<5; k++){
                    Reservoir pixel_r = g_Reservoirs_current[spatial_pixels[k]];
                    if(RejectNormal(pixel_r.hitNormal, reservoir_current.hitNormal) == false && pixel_r.p_hat > 0.0f){
                        float mi = pixel_r.p_hat / p_sum;
                        // Calculate the weight for the given sample: 1/N (const) * p_hat * W
                        float w = mi * pixel_r.p_hat * pixel_r.w_i;
                        UpdateReservoir(
                            reservoir_current,
                            w,
                            pixel_r.M,
                            seed,
                            pixel_r.f,
                            pixel_r.p_hat,
                            pixel_r.v_eval,
                            pixel_r.v,
                            pixel_r.direction,
                            pixel_r.dist,
                            pixel_r.hitPos,
                            pixel_r.hitNormal
                        );
                    }
                }
            }*/

            // Combine with temporal reservoir
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
                   0.0f           // M
                 };

                float M_c = max(20, reservoir_current.M);
                float M_t = max(20, reservoir_last.M);

                // Calculate the weight for the given sample: w * p_hat * W
                //First use GMIS to weight the existing reservoir
                float mi_c = M_c / (M_c + M_t);
                float w_c = mi_c * reservoir_current.p_hat * reservoir_current.w_i;
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

                // Now add the temporal reservoir (weighted as well)
                float mi_t = M_t / (M_c + M_t);
                float w_t = mi_t * reservoir_last.p_hat * reservoir_last.w_i;
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



            float V = 1.0f;
            // Check visibility here. This ensures a sample with actual contribution is used
            /*if(reservoir_current.p_hat > 0.0f){
                if(reservoir_current.v_eval == false){
                    // If we didnt do a visibility test, do it now:
                    //Shadow ray
                    //____________________________________________________________
                    RayDesc ray;
                    ray.Origin = reservoir_current.hitPos + s_bias * reservoir_current.hitNormal; // Offset origin along the normal
                    ray.Direction = reservoir_current.direction;
                    ray.TMin = 0.0f;
                    ray.TMax = reservoir_current.dist - s_bias * 2.0f;
                    bool hit = true;
                    // Initialize the ray payload
                    ShadowHitInfo shadowPayload;
                    shadowPayload.isHit = false;
                    // Trace the ray
                    TraceRay(SceneBVH, RAY_FLAG_NONE, 0xFF, 1, 0, 1, ray, shadowPayload);
                    V = shadowPayload.isHit ? 0.0 : 1.0;
                    //____________________________________________________________
                }
            }*/

            emission += reservoir_current.f * reservoir_current.w_i;

            //_______________________________RUSSIAN_ROULETTE__________________________________

            // Russian roulette: terminate the ray if the further accumulation of light would yield diminishing improvements due to a dark throughput
            /*if(y > rr_threshold){
                float max_throughput = max(throughput.x, max(throughput.y, throughput.z));
                float q = clamp(max_throughput, 0.05f, 1.0f); // Ensures q is at least 5%
                float random = RandomFloat(seed);

                if(random > q){
                    break;
                }

                payload.colorAndDistance.xyz *= 1.0f/q;

            }*/
        }
    }
    accumulation += emission;

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

    averagedColor = accumulation;


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

        g_Reservoirs_last[pixelIdx] = reservoir;
        g_Reservoirs_current[pixelIdx] = reservoir;
    }
    else{
        // Save the current reservoir and rotate the reservoirbuffer
        g_Reservoirs_last[pixelIdx] = reservoir_current;
        g_Reservoirs_current[pixelIdx] = reservoir;
    }

    // Output the final color to layer 0
    gOutput[uint3(launchIndex, 0)] = float4(averagedColor, 1.0f);
}
