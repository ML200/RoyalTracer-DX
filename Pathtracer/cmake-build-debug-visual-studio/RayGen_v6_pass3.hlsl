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
#include "MIS_v6.hlsl"

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
void RayGen3() {
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
    uint2 seed;
    float3 accumulation = float3(0, 0, 0);

    //_______________________________SETUP__________________________________
    // Every sample recieves a new seed (use samples+2 here to get different random numbers then the RayGen1 shader)
    seed.x = launchIndex.y * prime1_x ^ launchIndex.x * prime2_x ^ uint(samples + 3) * prime3_x ^ uint(time) * prime_time_x;
    seed.y = launchIndex.x * prime1_y ^ launchIndex.y * prime2_y ^ uint(samples + 3) * prime3_y ^ uint(time) * prime_time_y;

    // The current reservoir
    Reservoir reservoir_current = g_Reservoirs_current[pixelIdx];
    Reservoir reservoir_spatial = {
        (float3)0.0f,  // x1
        (float3)0.0f,  // n1
        (float3)0.0f,  // x2
        (float3)0.0f,  // n2
        (float)0.0f,  // w_sum
        (float)0.0f,  // p_hat of the stored sample
        (float)0.0f,  // W
        (float)0.0f,  // M
        (float)0.0f,    // V
        (float3)0.0f,  // final color (L1), mostly 0
        (float3)0.0f,  // reconnection color (L2)
        (uint)0,  // s
        (float3)0.0f,  //o
        (uint)0  // mID
    };
    reservoir_spatial.x1 = reservoir_current.x1;
	reservoir_spatial.n1 = reservoir_current.n1;
	reservoir_spatial.L1 = reservoir_current.L1;
	reservoir_spatial.o = reservoir_current.o;
	reservoir_spatial.mID = reservoir_current.mID;

    if(reservoir_current.L1.x == 0.0f && reservoir_current.L1.y == 0.0f && reservoir_current.L1.z == 0.0f){
        // Spatial reuse
		// 1. Aquire 5 pixels in the vicinity
		Reservoir spatial_candidates[spatial_candidate_count];
        bool rejected[spatial_candidate_count];
        float M_sum = min(spatial_M_cap, reservoir_current.M); // c weight
        Reservoir canonical = reservoir_current; // canonical samples M value stored

		for (int v = 0; v < spatial_candidate_count; v++){
            // Get a random pixel in a 30 pixel radius arround this pixel
			uint pixel_r = GetRandomPixelCircleWeighted(30, DispatchRaysDimensions().x, DispatchRaysDimensions().y, launchIndex.x, launchIndex.y, seed);
			// Fetch the spatial candidate
            Reservoir spatial_candidate = g_Reservoirs_current[pixel_r];

            // Reject the pixel if certain conditions arent fulfilled
			if(!RejectNormal(canonical.n1, spatial_candidate.n1) && !RejectDistance(canonical.x1, spatial_candidate.x1, init_orig, 0.1f) && length(spatial_candidate.L1) == 0.0f && spatial_candidate.mID != 4294967294){
				spatial_candidates[v] = spatial_candidate;
                M_sum += min(spatial_M_cap, spatial_candidates[v].M);
                rejected[v] = false;
			}
            else
                rejected[v] = true;
		}

        float mi_c = GenPairwiseMIS_canonical(canonical, spatial_candidates, rejected, M_sum);

        float W_c = GetW(canonical);
        float w_c = mi_c * canonical.p_hat * W_c;
        //WeightReservoir(canonical, w_c);
        UpdateReservoir(
            reservoir_spatial,
            w_c,
            min(spatial_M_cap, canonical.M),
            canonical.p_hat,
            canonical.x2,
            canonical.n2,
            canonical.L2,
            canonical.s,
            seed
        );

        for(int v = 0; v < spatial_candidate_count; v++){
            if(!rejected[v] && M_sum > min(spatial_M_cap, reservoir_current.M)){
                Reservoir spatial_candidate = spatial_candidates[v];
                float mi_s = GenPairwiseMIS_noncanonical(canonical, spatial_candidate, M_sum);

                float W_s = GetW(spatial_candidate);
                float w_s = mi_s * spatial_candidate.p_hat * W_s;

                UpdateReservoir(
                    reservoir_spatial,
                    w_s,
                    min(spatial_M_cap,spatial_candidate.M),
                    spatial_candidate.p_hat,
                    spatial_candidate.x2,
                    spatial_candidate.n2,
                    spatial_candidate.L2,
                    spatial_candidate.s,
                    seed
                );
            }
        }
        reservoir_current = reservoir_spatial;
        float V = VisibilityCheck(reservoir_current.x1, reservoir_current.n1, normalize(reservoir_current.x2-reservoir_current.x1), length(reservoir_current.x2-reservoir_current.x1));

        if(V == 0.0f)
            reservoir_current.p_hat = 0.0f;

        float W = GetW(reservoir_current);
        accumulation = ReconnectDI(reservoir_current.x1,reservoir_current.n1,reservoir_current.x2,reservoir_current.n2,reservoir_current.L2, reservoir_current.o, reservoir_current.s, materials[reservoir_current.mID]) * W;

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

        //averagedColor = float3(weightsSUM, weightsSUM, weightsSUM);
        //averagedColor = float3(M_sum/6.0f, M_sum/6.0f, M_sum/6.0f);
        /*if(weightsSUM>=0.9f)
            averagedColor = float3(0,1,1);
        if(weightsSUM>1.1f)
            averagedColor = float3(1,0,0);*/
        averagedColor = accumulation;
		if(isnan(averagedColor.x) || isnan(averagedColor.y) || isnan(averagedColor.z))
			averagedColor = float3(1,0,1);


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
        }



        // Output the final color to layer 0
		g_Reservoirs_last[pixelIdx] = reservoir_current;
        gOutput[uint3(launchIndex, 0)] = float4(averagedColor, 1.0f);
    }
    else{
        gOutput[uint3(launchIndex, 0)] = float4(reservoir_current.L1, 1.0f);
    }
}
