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
void RayGen2() {

    // Get the location within the dispatched 2D grid of work items (often maps to pixels, so this could represent a pixel coordinate).
    uint2 launchIndex = DispatchRaysIndex().xy;
    uint pixelIdx = launchIndex.y * DispatchRaysDimensions().x + launchIndex.x;
    float2 dims = float2(DispatchRaysDimensions().xy);
    // The current reservoir
    Reservoir_DI reservoir_current = g_Reservoirs_current[pixelIdx];
    SampleData sdata_current = g_sample_current[pixelIdx];

	if(sdata_current.L1.x == 0.0f && sdata_current.L1.y == 0.0f && sdata_current.L1.z == 0.0f){
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
        int2 pixelPos;
        pixelPos = GetBestReprojectedPixel_d(sdata_current.x1, prevView, prevProjection, dims);
        uint tempPixelIdx = pixelPos.y * DispatchRaysDimensions().x + pixelPos.x;
        Reservoir_DI reservoir_last = g_Reservoirs_last[tempPixelIdx];
        SampleData sdata_last = g_sample_last[tempPixelIdx];

        //_______________________________RESTIR_TEMPORAL__________________________________
        // Temporal reuse
        if(
            pixelPos.x != -1 && pixelPos.y != -1
            && length(sdata_last.L1) == 0.0f
            && !RejectNormal(sdata_current.n1, sdata_last.n1, reservoir_current.s)
            && !RejectLocation(tempPixelIdx, pixelIdx, reservoir_current.s, reservoir_last.s, materials[sdata_current.mID])
            && !RejectDistance(sdata_current.x1, sdata_last.x1, init_orig, 0.1f)
            && (reservoir_last.x2.x != 0.0f && reservoir_last.x2.y != 0.0f && reservoir_last.x2.z != 0.0f)
            //&& !RejectRoughness(view, prevView, tempPixelIdx, pixelIdx, reservoir_current.s, materials[reservoir_current.mID].Pr_Pm_Ps_Pc.x)
        ){
            float M_sum = min(temporal_M_cap, reservoir_current.M) + min(temporal_M_cap, reservoir_last.M);
            // Pairwise MIS
            float mi_c = GenPairwiseMIS_canonical_temporal(reservoir_current, reservoir_last, M_sum, temporal_M_cap);
            float mi_t = GenPairwiseMIS_noncanonical_temporal(reservoir_current, reservoir_last, M_sum, temporal_M_cap);

            // Calculate the weight for the given sample: w * p_hat * W
            float w_c = mi_c * GetP_Hat(reservoir_current, reservoir_current, sdata_current, false) * reservoir_current.W;
            float w_t = mi_t * GetP_Hat(reservoir_current, reservoir_last, sdata_current, false) * reservoir_last.W;
            Reservoir_DI reservoir_temporal = {
                float3(0.0f, 0.0f, 0.0f), 0.0f,  // x2, pad0
                float3(0.0f, 0.0f, 0.0f), 0.0f,  // n2, pad1
                0.0f, 0.0f, 0.0f, 0.0f,          // w_sum, W, M, pad2
                float3(0.0f, 0.0f, 0.0f), 0.0f,  // L2, pad3
                0, 0, 0, 0                      // s, pad4, pad5, pad6
            };
            UpdateReservoir(
                reservoir_temporal,
                w_c,
                min(temporal_M_cap, reservoir_current.M),
                reservoir_current.x2,
                reservoir_current.n2,
                reservoir_current.L2,
                reservoir_current.s,
				seed
            );

            UpdateReservoir(
                reservoir_temporal,
                w_t,
                min(temporal_M_cap, reservoir_last.M),
                reservoir_last.x2,
                reservoir_last.n2,
                reservoir_last.L2,
                reservoir_last.s,
				seed

            );

            float p_hat = GetP_Hat(reservoir_temporal, reservoir_temporal, sdata_current, true);
            reservoir_temporal.W = GetW(reservoir_temporal, p_hat);
            reservoir_current = reservoir_temporal;
        }
    }
    g_Reservoirs_current[pixelIdx] = reservoir_current;
}
