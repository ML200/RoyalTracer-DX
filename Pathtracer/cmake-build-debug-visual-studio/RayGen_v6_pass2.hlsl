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
void RayGen2() {

    // Get the location within the dispatched 2D grid of work items (often maps to pixels, so this could represent a pixel coordinate).
    uint2 launchIndex = DispatchRaysIndex().xy;
    uint pixelIdx = launchIndex.y * DispatchRaysDimensions().x + launchIndex.x;
    float2 dims = float2(DispatchRaysDimensions().xy);

    // The current reservoir
    Reservoir reservoir_current = g_Reservoirs_current[pixelIdx];

	if(reservoir_current.L1.x == 0.0f && reservoir_current.L1.y == 0.0f && reservoir_current.L1.z == 0.0f){

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


        // Simple motion vectors:
        int2 pixelPos = GetBestReprojectedPixel(reservoir_current.x1, prevView, prevProjection, dims);
        uint tempPixelIdx = pixelPos.y * DispatchRaysDimensions().x + pixelPos.x;
        Reservoir reservoir_last = g_Reservoirs_last[tempPixelIdx];

        //_______________________________RESTIR_TEMPORAL__________________________________
        // Temporal reuse
        if((reservoir_current.p_hat > 0.0f && reservoir_current.M > 0.0f) || (reservoir_last.p_hat > 0.0f && reservoir_last.M > 0.0f)){
			float normalRejection = RejectNormal(reservoir_current.n1, reservoir_last.n1)?0.0f:1.0f;

            float M_c = min(temporal_M_cap, reservoir_current.M);
            float M_t = min(temporal_M_cap, reservoir_last.M);

            // TODO:
            float mi_c = M_c / (M_c + M_t);
            float mi_t = 1.0f - mi_c;

            float W_c = GetW(reservoir_current);
            float W_t = GetW(reservoir_last);

            // Calculate the weight for the given sample: w * p_hat * W
            float w_c = mi_c * reservoir_current.p_hat * W_c;
            float w_t = mi_t * reservoir_last.p_hat * W_t;


			if(
                pixelPos.x != -1 && pixelPos.y != -1
                && length(reservoir_last.L1) == 0.0f
                && normalRejection == 1.0f
            ){
                WeightReservoir(reservoir_current, w_c);
            	UpdateReservoir(
                	reservoir_current,
                	w_t,
                	min(temporal_M_cap, reservoir_last.M),
                	reservoir_last.p_hat,
                	reservoir_last.x2,
                	reservoir_last.n2,
                	reservoir_last.L2,
                	reservoir_last.s,
					seed
            	);
			}
        }
    }
    g_Reservoirs_current[pixelIdx] = reservoir_current;
}
