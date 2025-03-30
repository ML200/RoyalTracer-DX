#pragma warning(disable: 1234)

#include "Common_v6.hlsl"
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

#include "GGX_v6.hlsl"
#include "Lambertian_v6.hlsl"
#include "BRDF_v6.hlsl"
#include "Sampler_v6.hlsl"
#include "MIS_v6.hlsl"
#include "MIS_GI_v6.hlsl"

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
}

// Second raygen shader is the ReSTIR pass. The reservoirs were filled in the first shader, now we recombine them

[shader("raygeneration")]
void RayGen2() {

    // Get the location within the dispatched 2D grid of work items (often maps to pixels, so this could represent a pixel coordinate).
    uint2 launchIndex = DispatchRaysIndex().xy;
    float2 dims = float2(DispatchRaysDimensions().xy);
    uint pixelIdx = MapPixelID(dims, launchIndex);

    // The current reservoirs and sample data
    Reservoir_DI reservoir_current = g_Reservoirs_current[pixelIdx];
    Reservoir_GI reservoir_gi_current = g_Reservoirs_current_gi[pixelIdx];
    SampleData sdata_current = g_sample_current[pixelIdx];

    if(sdata_current.L1.x == 0.0f && sdata_current.L1.y == 0.0f && sdata_current.L1.z == 0.0f){
        // #DXR Extra: Perspective Camera
        float aspectRatio = dims.x / dims.y;

        // Initialize the ray origin and direction
        float3 init_orig = mul(viewI, float4(0, 0, 0, 1)).xyz;

        // SEEDING
        const uint prime1_x = 73856093u;
        const uint prime2_x = 19349663u;
        const uint prime3_x = 83492791u;
        const uint prime1_y = 37623481u;
        const uint prime2_y = 51964263u;
        const uint prime3_y = 68250729u;
        const uint prime_time_x = 293803u;
        const uint prime_time_y = 423977u;

        uint2 seed;
        // Every sample receives a new seed (use samples+2 here to get different random numbers than the RayGen1 shader)
        seed.x = launchIndex.y * prime1_x ^ launchIndex.x * prime2_x ^ uint(2) * prime3_x ^ uint(time) * prime_time_x;
        seed.y = launchIndex.x * prime1_y ^ launchIndex.y * prime2_y ^ uint(2) * prime3_y ^ uint(time) * prime_time_y;

        // Obtain the best reprojected pixel from the previous frame.
        int2 pixelPos = GetBestReprojectedPixel_d(sdata_current.x1, prevView, prevProjection, dims, sdata_current.objID);
        uint tempPixelIdx = MapPixelID(dims, pixelPos);
        Reservoir_DI reservoir_last = g_Reservoirs_last[tempPixelIdx];
        Reservoir_GI reservoir_gi_last = g_Reservoirs_last_gi[tempPixelIdx];
        SampleData sdata_last = g_sample_last[tempPixelIdx];

        // Define separate candidate acceptance criteria for DI and GI temporal reuse.
        bool candidateAcceptedDI = (pixelPos.x != -1 && pixelPos.y != -1 &&
            length(sdata_last.L1) == 0.0f &&
            !RejectNormal(sdata_current.n1, sdata_last.n1, 0.9f) &&
            //!RejectLocation(tempPixelIdx, pixelIdx, reservoir_current.s, reservoir_last.s, materials[sdata_current.mID]) &&
            IsValidReservoir(reservoir_last) &&
            !RejectDistance(sdata_current.x1, sdata_last.x1, init_orig, 0.1f) &&
            (reservoir_last.x2.x != 0.0f && reservoir_last.x2.y != 0.0f && reservoir_last.x2.z != 0.0f) &&
            (sdata_last.mID == sdata_current.mID)
        );

        bool candidateAcceptedGI = (pixelPos.x != -1 && pixelPos.y != -1 &&
            length(sdata_last.L1) == 0.0f &&
            !RejectWsum(reservoir_gi_last.w_sum, w_sum_threshold) &&
            !RejectNormal(sdata_current.n1, sdata_last.n1, 0.9f) &&
            !RejectDistance(sdata_current.x1, sdata_last.x1, init_orig, 0.1f) &&
            IsValidReservoir_GI(reservoir_gi_last) &&
            (sdata_last.mID == sdata_current.mID)
        );

        // -------------------- Temporal Reuse for DI --------------------
        if(candidateAcceptedDI)
        {
            uint mID = sdata_current.mID;
            MaterialOptimized matOpt = {
                materials[mID].Kd, materials[mID].Pr_Pm_Ps_Pc,
                materials[mID].Ks, materials[mID].Ke, mID
            };

            float M_sum = min(temporal_M_cap, reservoir_current.M) + min(temporal_M_cap, reservoir_last.M);
            float mi_c = GenPairwiseMIS_canonical_temporal(reservoir_current, reservoir_last, M_sum, temporal_M_cap);
            float mi_t = GenPairwiseMIS_noncanonical_temporal(reservoir_current, reservoir_last, M_sum, temporal_M_cap);

            if(length(reservoir_last.n2) == 0.0f)
            {
                mi_c = 1.0f;
                mi_t = 0.0f;
            }

            float w_c = mi_c * GetP_Hat(sdata_current.x1, sdata_current.n1,
                                        reservoir_current.x2, reservoir_current.n2,
                                        reservoir_current.L2, sdata_current.o, matOpt, false) * reservoir_current.W;
            float w_t = mi_t * GetP_Hat(sdata_current.x1, sdata_current.n1,
                                        reservoir_last.x2, reservoir_last.n2,
                                        reservoir_last.L2, sdata_current.o, matOpt, false) * reservoir_last.W;

            reservoir_current.M = min(temporal_M_cap, reservoir_current.M);
            reservoir_current.w_sum = w_c;

            UpdateReservoir(
                reservoir_current,
                w_t,
                min(temporal_M_cap, reservoir_last.M),
                reservoir_last.x2,
                reservoir_last.n2,
                reservoir_last.L2,
                seed
            );

            float p_hat = GetP_Hat(sdata_current.x1, sdata_current.n1,
                                   reservoir_current.x2, reservoir_current.n2,
                                   reservoir_current.L2, sdata_current.o, matOpt, false);
            reservoir_current.W = GetW(reservoir_current, p_hat);
        }

        // -------------------- Temporal Reuse for GI --------------------
        if(candidateAcceptedGI)
        {
            MaterialOptimized matOpt = {
                materials[sdata_current.mID].Kd, materials[sdata_current.mID].Pr_Pm_Ps_Pc,
                materials[sdata_current.mID].Ks, materials[sdata_current.mID].Ke, sdata_current.mID
            };

            float M_sum_gi = min(temporal_M_cap_GI, reservoir_gi_current.M) + min(temporal_M_cap_GI, reservoir_gi_last.M);

            float mi_c_gi = GenPairwiseMIS_canonical_temporal_GI(reservoir_gi_current, reservoir_gi_last, M_sum_gi, temporal_M_cap_GI);
            float mi_t_gi = GenPairwiseMIS_noncanonical_temporal_GI(reservoir_gi_current, reservoir_gi_last, M_sum_gi, temporal_M_cap_GI);

            MaterialOptimized mat_gi_c = CreateMaterialOptimized(materials[reservoir_gi_current.mID2], reservoir_gi_current.mID2);
            float3 f_c = GetP_Hat_GI(sdata_current.x1, sdata_current.n1,
                                     reservoir_gi_current.xn, reservoir_gi_current.nn,
                                     reservoir_gi_current.E3, reservoir_gi_current.Vn,
                                     sdata_current.o, matOpt, mat_gi_c, false);
            float w_c_gi = mi_c_gi * LinearizeVector(f_c) * reservoir_gi_current.W;

            MaterialOptimized mat_gi_t = CreateMaterialOptimized(materials[reservoir_gi_last.mID2], reservoir_gi_last.mID2);
            float3 f_t = GetP_Hat_GI(sdata_current.x1, sdata_current.n1,
                                     reservoir_gi_last.xn, reservoir_gi_last.nn,
                                     reservoir_gi_last.E3, reservoir_gi_last.Vn,
                                     sdata_current.o, matOpt, mat_gi_t, false);
            float w_t_gi = mi_t_gi * LinearizeVector(f_t) * reservoir_gi_last.W;

            reservoir_gi_current.M = min(temporal_M_cap_GI, reservoir_gi_current.M);
            reservoir_gi_current.w_sum = w_c_gi;

            UpdateReservoir_GI(
                reservoir_gi_current,
                w_t_gi,
                min(temporal_M_cap_GI, reservoir_gi_last.M),
                reservoir_gi_last.xn,
                reservoir_gi_last.nn,
                reservoir_gi_last.Vn,
                reservoir_gi_last.E3,
                reservoir_gi_last.mID2,
                seed
            );
            MaterialOptimized mat_gi = CreateMaterialOptimized(materials[reservoir_gi_current.mID2], reservoir_gi_current.mID2);
            reservoir_gi_current.W = GetW_GI(reservoir_gi_current, LinearizeVector(GetP_Hat_GI(sdata_current.x1, sdata_current.n1,
                                     reservoir_gi_current.xn, reservoir_gi_current.nn,
                                     reservoir_gi_current.E3, reservoir_gi_current.Vn,
                                     sdata_current.o, matOpt, mat_gi, false)));
        }
    }
    g_Reservoirs_current[pixelIdx] = reservoir_current;
    g_Reservoirs_current_gi[pixelIdx] = reservoir_gi_current;
}
