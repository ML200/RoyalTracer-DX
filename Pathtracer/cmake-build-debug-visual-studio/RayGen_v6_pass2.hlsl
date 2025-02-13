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
    uint2 seed;
    float3 accumulation = float3(0, 0, 0);

    //_______________________________SETUP__________________________________
    // Every sample recieves a new seed (use samples+2 here to get different random numbers then the RayGen1 shader)
    seed.x = launchIndex.y * prime1_x ^ launchIndex.x * prime2_x ^ uint(samples + 2) * prime3_x ^ uint(time) * prime_time_x;
    seed.y = launchIndex.x * prime1_y ^ launchIndex.y * prime2_y ^ uint(samples + 2) * prime3_y ^ uint(time) * prime_time_y;

    // The current reservoir
    Reservoir reservoir_current = g_Reservoirs_current[pixelIdx];

	// Simple motion vectors:
	int2 pixelPos = GetBestReprojectedPixel(reservoir_current.x1, prevView, prevProjection, dims);
	uint tempPixelIdx = pixelPos.y * DispatchRaysDimensions().x + pixelPos.x;
    Reservoir reservoir_last = g_Reservoirs_last[tempPixelIdx];

    if(reservoir_current.L1.x == 0.0f && reservoir_current.L1.y == 0.0f && reservoir_current.L1.z == 0.0f){
        //_______________________________RESTIR_TEMPORAL__________________________________
        // Temporal reuse
        /*if((reservoir_current.p_hat > 0.0f && reservoir_current.M > 0.0f) || (reservoir_last.p_hat > 0.0f && reservoir_last.M > 0.0f)){
            Reservoir temporal_res = {
    			reservoir_current.x1,  // x1
    			reservoir_current.n1,  // n1
    			(float3)0.0f,  // x2
    			(float3)0.0f,  // n2
    			(float)0.0f,  // w_sum
    			(float)0.0f,  // p_hat of the stored sample
    			(float)0.0f,  // W
    			(float)0.0f,  // M
    			(float)0.0f,  // V
    			(float3)0.0f,  // final color (L1), mostly 0
    			(float3)0.0f,  // reconnection color (L2)
				(uint)0.0f,  // s
        		reservoir_current.o,  //o
        		reservoir_current.mID  // mID
             };
			float normalRejection = RejectNormal(reservoir_current.n1, reservoir_last.n1)?0.0f:1.0f;

            float M_c = min(20, reservoir_current.M);
            float M_t = min(20, reservoir_last.M);

            float mi_c = M_c / (M_c + M_t);
            float mi_t = M_t / (M_c + M_t);

            // Calculate the weight for the given sample: w * p_hat * W
            float w_c = mi_c * reservoir_current.p_hat * reservoir_current.W * reservoir_current.V;
            float w_t = mi_t * reservoir_last.p_hat * reservoir_last.W * normalRejection;

            UpdateReservoir(
                temporal_res,
                w_c,
                reservoir_current.M,
                reservoir_current.p_hat,
                reservoir_current.x2,
                reservoir_current.n2,
                reservoir_current.L2,
                reservoir_current.s,
                seed
            );
			if(pixelPos.x != -1 && pixelPos.y != -1){
            	UpdateReservoir(
                	temporal_res,
                	w_t,
                	reservoir_last.M,
                	reservoir_last.p_hat,
                	reservoir_last.x2,
                	reservoir_last.n2,
                	reservoir_last.L2,
                	reservoir_last.s,
					seed
            	);
			}
            reservoir_current = temporal_res;
        }*/

		// Spatial reuse
		// 1. Aquire 5 pixels in the vicinity
		uint radius = 30; // 30 pixel radius
		Reservoir spatial_candidates[5];
		for (int v = 0; v < 5; v++){
			uint pixel = GetRandomPixelCircleWeighted(30, DispatchRaysDimensions().x, DispatchRaysDimensions().y, launchIndex.x, launchIndex.y, seed);
			Reservoir spatial_candidate = g_Reservoirs_current[pixel];
			if(!RejectNormal(reservoir_current.n1, spatial_candidate.n1)){
				spatial_candidates[v] = spatial_candidate;
			}
			else{
				spatial_candidates[v].M = 0.0f;
			}
		}
		float debug = 0.0f;
		for(int v = 0; v < 5; v++){
			Reservoir spatial_candidate = spatial_candidates[v];
			if(spatial_candidate.M >= 1.0f){
				debug+=1.0f;
				Reservoir spatial_res = {
    				reservoir_current.x1,  // x1
    				reservoir_current.n1,  // n1
    				(float3)0.0f,  // x2
    				(float3)0.0f,  // n2
    				(float)0.0f,  // w_sum
    				(float)0.0f,  // p_hat of the stored sample
    				(float)0.0f,  // W
    				(float)0.0f,  // M
    				(float)0.0f,  // V
    				(float3)0.0f,  // final color (L1), mostly 0
    				(float3)0.0f,  // reconnection color (L2)
					(uint)0.0f,  // s
        			reservoir_current.o,  //o
        			reservoir_current.mID  // mID
            	};
				float M_sum = reservoir_current.M + spatial_candidate.M;
				float mi_c = reservoir_current.M / M_sum;
				float mi_s = spatial_candidate.M / M_sum;
				float w_c = mi_c * reservoir_current.p_hat * reservoir_current.W;
				float w_s = mi_s * spatial_candidate.p_hat * spatial_candidate.W;

				UpdateReservoir(
                	spatial_res,
                	w_c,
                	reservoir_current.M,
                	reservoir_current.p_hat,
                	reservoir_current.x2,
                	reservoir_current.n2,
                	reservoir_current.L2,
                	reservoir_current.s,
                	seed
            	);
            	UpdateReservoir(
                	spatial_res,
                	w_s,
                	spatial_candidate.M,
                	spatial_candidate.p_hat,
                	spatial_candidate.x2,
                	spatial_candidate.n2,
                	spatial_candidate.L2,
                	spatial_candidate.s,
					seed
            	);
            	reservoir_current = spatial_res;
			}
		}

        accumulation = ReconnectDI(reservoir_current.x1,reservoir_current.n1,reservoir_current.x2,reservoir_current.n2,reservoir_current.L2, reservoir_current.o, reservoir_current.s, materials[reservoir_current.mID]) * reservoir_current.W;

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



		// ----- DEBUG: Output Motion Vectors -----
    	/*{
        	// Convert current pixel coordinate (launchIndex) to float2.
        	float2 currentPixel = float2(launchIndex.x, launchIndex.y);
        	// Convert reprojected pixel coordinate (from previous frame) to float2.
        	float2 previousPixel = float2(pixelPos.x, pixelPos.y);

        	// Calculate the motion vector (difference in pixel space).
        	float2 motionVector = currentPixel - previousPixel;

        	// For visualization, we scale and bias the motion vector into [0,1].
        	// Adjust the scaleFactor to suit the magnitude of motion in your scene.
        	float scaleFactor = 10.0f; // Example: change as needed.
        	float2 normalizedMotion = saturate(motionVector / scaleFactor + 0.5f);

        	// Write the normalized motion vector into the red (x) and green (y) channels.
        	// Blue is set to zero and alpha to one.
        	gOutput[uint3(launchIndex, 0)] = float4(normalizedMotion, 0.0f, 1.0f);

        	// Skip further processing to output only the debug motion vector.
        	return;
    	}*/
    	// ----- END DEBUG -----


        // Output the final color to layer 0
		g_Reservoirs_last[pixelIdx] = reservoir_current;
        gOutput[uint3(launchIndex, 0)] = float4(averagedColor, 1.0f);
    }
    else{
        gOutput[uint3(launchIndex, 0)] = float4(reservoir_current.L1, 1.0f);
    }
}
