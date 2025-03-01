#include "Common_v6.hlsl"
#include "GGX_v6.hlsl"
#include "Lambertian_v6.hlsl"
#include "BRDF_v6.hlsl"
#include "Reservoir_v6.hlsl"

// Raytracing output texture, accessed as a UAV
RWTexture2DArray<float4> gOutput : register(u0);
RWTexture2D<float4> gPermanentData : register(u1);

RWStructuredBuffer<Reservoir_DI> g_Reservoirs_current : register(u2);
RWStructuredBuffer<Reservoir_DI> g_Reservoirs_last : register(u3);
RWStructuredBuffer<Reservoir_GI> g_Reservoirs_current_gi : register(u4);
RWStructuredBuffer<Reservoir_GI> g_Reservoirs_last_gi : register(u5);
RWStructuredBuffer<SampleData> g_sample_current : register(u6);
RWStructuredBuffer<SampleData> g_sample_last : register(u7);

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
    SampleData sdata_current = g_sample_current[pixelIdx];

    if(sdata_current.L1.x == 0.0f && sdata_current.L1.y == 0.0f && sdata_current.L1.z == 0.0f){
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
        //_______________________________SETUP__________________________________
        // Every sample recieves a new seed (use samples+2 here to get different random numbers then the RayGen1 shader)
        seed.x = launchIndex.y * prime1_x ^ launchIndex.x * prime2_x ^ uint(samples + 3) * prime3_x ^ uint(time) * prime_time_x;
        seed.y = launchIndex.x * prime1_y ^ launchIndex.y * prime2_y ^ uint(samples + 3) * prime3_y ^ uint(time) * prime_time_y;

        // The current reservoir
        Reservoir_DI reservoir_current = g_Reservoirs_current[pixelIdx];
        Reservoir_GI reservoir_gi_current = g_Reservoirs_current_gi[pixelIdx];
        Reservoir_DI reservoir_spatial = {
            float3(0.0f, 0.0f, 0.0f), 0.0f,  // x2, pad0
            float3(0.0f, 0.0f, 0.0f), 0.0f,  // n2, pad1
            0.0f, 0.0f, 0.0f, 0.0f,          // w_sum, W, M, pad2
            float3(0.0f, 0.0f, 0.0f), 0.0f,  // L2, pad3
            0, 0, 0, 0                      // s, pad4, pad5, pad6
        };

        // Spatial reuse
		// 1. Aquire 5 pixels in the vicinity
		Reservoir_DI spatial_candidates[spatial_candidate_count];
		SampleData sample_candidates[spatial_candidate_count];
        bool rejected[spatial_candidate_count];

        float M_sum = min(spatial_M_cap, reservoir_current.M); // c weight
        Reservoir_DI canonical = reservoir_current; // canonical samples M value stored

        [loop]
		for (int v = 0; v < spatial_candidate_count; v++){
            // Get a random pixel in a 30 pixel radius arround this pixel
			uint pixel_r = GetRandomPixelCircleWeighted(spatial_radius, DispatchRaysDimensions().x, DispatchRaysDimensions().y, launchIndex.x, launchIndex.y, seed);
			// Fetch the spatial candidate
            Reservoir_DI spatial_candidate = g_Reservoirs_current[pixel_r];
            SampleData sdata_candidate = g_sample_current[pixel_r];

            // Evaluate your candidate predicate per thread
            bool candidateAccepted =
                !RejectNormal(sdata_current.n1, sdata_candidate.n1, canonical.s) &&
                !RejectDistance(sdata_current.x1, sdata_candidate.x1, init_orig, 0.1f) &&
                (length(sdata_candidate.L1) == 0.0f) &&
                (sdata_candidate.mID != 4294967294);// &&
                //(spatial_candidate.s == canonical.s);

            // Reject the pixel if certain conditions arent fulfilled
			if(candidateAccepted){
				spatial_candidates[v] = spatial_candidate;
				sample_candidates[v] = sdata_candidate;
                M_sum += min(spatial_M_cap, spatial_candidates[v].M);
                rejected[v] = false;
			}
            else
                rejected[v] = true;
		}

        float mi_c = GenPairwiseMIS_canonical(canonical, spatial_candidates, sdata_current, sample_candidates, rejected, M_sum, spatial_M_cap);
        float w_c = mi_c * GetP_Hat(canonical, canonical, sdata_current, false) * canonical.W;

        UpdateReservoir(
            reservoir_spatial,
            w_c,
            min(spatial_M_cap, canonical.M),
            canonical.x2,
            canonical.n2,
            canonical.L2,
            canonical.s,
            seed
        );

        [loop]
        for(int v = 0; v < spatial_candidate_count; v++){
            if(!rejected[v]){
                Reservoir_DI spatial_candidate = spatial_candidates[v];
                SampleData sdata_candidate = sample_candidates[v];
                float mi_s = GenPairwiseMIS_noncanonical(canonical, spatial_candidate, sdata_current, sdata_candidate, M_sum, spatial_M_cap);
                float w_s = mi_s * GetP_Hat(canonical, spatial_candidate, sdata_current, false) * spatial_candidate.W;

                UpdateReservoir(
                    reservoir_spatial,
                    w_s,
                    min(spatial_M_cap,spatial_candidate.M),
                    spatial_candidate.x2,
                    spatial_candidate.n2,
                    spatial_candidate.L2,
                    spatial_candidate.s,
                    seed
                );
            }
        }
        reservoir_current = reservoir_spatial;
        float p_hat = GetP_Hat(reservoir_current, reservoir_current, sdata_current, true);
        reservoir_current.W = GetW(reservoir_current, p_hat);

        float3 accumulation = float3(0, 0, 0);
        accumulation = ReconnectDI(sdata_current.x1,sdata_current.n1,reservoir_current.x2,reservoir_current.n2,reservoir_current.L2, sdata_current.o, reservoir_current.s, materials[sdata_current.mID]) * reservoir_current.W;
        accumulation += reservoir_gi_current.indirect;
        //TEMPORAL ACCUMULATION  ___________________________________________________________________________________________
        float frameCount = gPermanentData[uint2(launchIndex)].w;
        int maxFrames = 200000;

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

        // Skip temporal accumulation
        //averagedColor = accumulation;

        // DEBUG PIXEL COLORING
        // NaNs in magenta
		if(isnan(averagedColor.x) || isnan(averagedColor.y) || isnan(averagedColor.z))
			averagedColor = float3(1,0,1);
        // show p_hat
        // show x2 reconnection position


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

        // Set the last reservoir to the current one to support temp. reuse
		g_Reservoirs_last[pixelIdx] = reservoir_current;
        g_sample_last[pixelIdx] = sdata_current;
        //Gamma correction
        float3 finalColor = pow(averagedColor, (float3)(1.0f/2.2f));
        gOutput[uint3(launchIndex, 0)] = float4(finalColor, 1.0f);
    }
    else{
        gOutput[uint3(launchIndex, 0)] = float4(sdata_current.L1, 1.0f);
    }
}
