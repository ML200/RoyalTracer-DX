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
    float4x4 prevView;        // Previous frame's view matrix (if needed)
    float4x4 prevProjection;  // Previous frame's projection matrix (if needed)
    float time;
}

// Second raygen shader is the ReSTIR pass. The reservoirs were filled in the first shader, now we recombine them.

[shader("raygeneration")]
void RayGen3()
{



    // Get the location within the dispatched 2D grid of work items (often maps to pixels).
    uint2 launchIndex = DispatchRaysIndex().xy;
    float2 dims       = float2(DispatchRaysDimensions().xy);
    uint pixelIdx     = MapPixelID(dims, launchIndex);
    SampleData sdata_current = g_sample_current[pixelIdx];

    // If the sample is flagged with L1 == 0, 0, 0, proceed with ReSTIR reuse
    if (sdata_current.L1.x == 0.0f && sdata_current.L1.y == 0.0f && sdata_current.L1.z == 0.0f)
    {
        // Initialize the ray origin and direction (if needed in your pass).
        float3 init_orig = mul(viewI, float4(0, 0, 0, 1)).xyz;

        // ----------------------------------------------------------------------------
        // SEEDING
        const uint prime1_x     = 73856093u;
        const uint prime2_x     = 19349663u;
        const uint prime3_x     = 83492791u;
        const uint prime1_y     = 37623481u;
        const uint prime2_y     = 51964263u;
        const uint prime3_y     = 68250729u;
        const uint prime_time_x = 293803u;
        const uint prime_time_y = 423977u;

        // Initialize once, to reduce allocs with several samples per frame
        uint2 seed;
        // Every sample receives a new seed:
        seed.x = launchIndex.y * prime1_x ^ launchIndex.x * prime2_x ^ uint(3) * prime3_x ^ uint(time) * prime_time_x;
        seed.y = launchIndex.x * prime1_y ^ launchIndex.y * prime2_y ^ uint(3) * prime3_y ^ uint(time) * prime_time_y;
        // ----------------------------------------------------------------------------

        uint mID = sdata_current.mID;

        // Spatial reuse: build two separate candidate lists — one for DI, one for GI.
        uint spatial_candidates_DI[spatial_candidate_count];
        bool rejected_DI[spatial_candidate_count];
        uint spatial_candidates_GI[spatial_candidate_count];
        bool rejected_GI[spatial_candidate_count];

        // Start with the primary reservoir's M.
        float M_sum_DI = min(spatial_M_cap,    g_Reservoirs_current[pixelIdx].M);
        float M_sum_GI = min(spatial_M_cap_GI, g_Reservoirs_current_gi[pixelIdx].M);

        // We store how many candidates we have for each reservoir
        int candidateFoundCount_DI = 0;
        int candidateFoundCount_GI = 0;

        MaterialOptimized matOpt = {
            materials[mID].Kd,
            materials[mID].Pr_Pm_Ps_Pc,
            materials[mID].Ks,
            materials[mID].Ke,
            mID
        };

        // First loop: gather DI candidates
        [loop]
        for (int attempt = 0; attempt < spatial_max_tries && candidateFoundCount_DI < spatial_candidate_count; attempt++)
        {
            uint pixel_r = GetRandomPixelCircleWeighted(
                spatial_radius,
                DispatchRaysDimensions().x,
                DispatchRaysDimensions().y,
                launchIndex.x,
                launchIndex.y,
                seed
            );

            // Evaluate DI candidate predicate
            bool candidateAccepted =
                !RejectNormal(sdata_current.n1, g_sample_current[pixel_r].n1, 0.9f) &&
                !RejectDistance(sdata_current.x1, g_sample_current[pixel_r].x1, init_orig, 0.1f) &&
                IsValidReservoir(g_Reservoirs_current[pixel_r]) &&
                (length(g_sample_current[pixel_r].L1) == 0.0f) &&
                (g_sample_current[pixel_r].mID != 4294967294) &&
                (g_sample_current[pixel_r].mID == sdata_current.mID);

            if (candidateAccepted)
            {
                spatial_candidates_DI[candidateFoundCount_DI] = pixel_r;
                M_sum_DI += min(spatial_M_cap, g_Reservoirs_current[pixel_r].M);
                rejected_DI[candidateFoundCount_DI] = false;
                candidateFoundCount_DI++;
            }
        }

        // Mark the remaining DI slots as rejected
        [loop]
        for (int v = candidateFoundCount_DI; v < spatial_candidate_count; v++)
        {
            rejected_DI[v] = true;
        }

        // Second loop: gather GI candidates
        [loop]
        for (int attempt = 0; attempt < spatial_max_tries && candidateFoundCount_GI < spatial_candidate_count; attempt++)
        {
            uint pixel_r = GetRandomPixelCircleWeighted(
                spatial_radius,
                DispatchRaysDimensions().x,
                DispatchRaysDimensions().y,
                launchIndex.x,
                launchIndex.y,
                seed
            );

            // Evaluate GI candidate predicate
            bool candidateAcceptedGI =
                //matOpt.Pr_Pm_Ps_Pc.x > 0.3f &&
                //!RejectNormal(sdata_current.n1, g_sample_current[pixel_r].n1, 0.5f) &&
                !RejectDistance(sdata_current.x1, g_sample_current[pixel_r].x1, init_orig, 0.1f) &&
                !RejectBelowSurface(normalize(g_Reservoirs_current_gi[pixel_r].xn - sdata_current.x1), sdata_current.n1) &&
                !RejectWsum(g_Reservoirs_current_gi[pixel_r].w_sum, w_sum_threshold) &&
                IsValidReservoir_GI(g_Reservoirs_current_gi[pixel_r]) &&
                !RejectJacobian(Jacobian_Reconnection(
                    g_sample_current[pixel_r],
                    sdata_current,
                    g_Reservoirs_current_gi[pixel_r].xn,
                    g_Reservoirs_current_gi[pixel_r].nn,
                    g_Reservoirs_current_gi[pixel_r].Vn
                ), j_threshold) &&
                (length(g_sample_current[pixel_r].L1) == 0.0f) &&
                (g_sample_current[pixel_r].mID != 4294967294) &&
                (g_sample_current[pixel_r].mID == sdata_current.mID);

            if (candidateAcceptedGI)
            {
                spatial_candidates_GI[candidateFoundCount_GI] = pixel_r;
                M_sum_GI += min(spatial_M_cap_GI, g_Reservoirs_current_gi[pixel_r].M);
                rejected_GI[candidateFoundCount_GI] = false;
                candidateFoundCount_GI++;
            }
        }

        // Mark the remaining GI slots as rejected
        [loop]
        for (int v = candidateFoundCount_GI; v < spatial_candidate_count; v++)
        {
            rejected_GI[v] = true;
        }

        // --------------------------------------------------------------------
        // Get the canonical (current pixel) DI and GI reservoirs
        Reservoir_DI reservoir_current     = g_Reservoirs_current[pixelIdx];
        Reservoir_GI reservoir_current_gi  = g_Reservoirs_current_gi[pixelIdx];
        Reservoir_DI canonical            = reservoir_current;    // DI
        Reservoir_GI canonical_gi         = reservoir_current_gi; // GI

        // DI: Evaluate canonical sample
        float mi_c = GenPairwiseMIS_canonical(
            canonical,
            spatial_candidates_DI,
            sdata_current,
            rejected_DI,
            M_sum_DI,
            spatial_M_cap,
            matOpt
        );
        float w_c = mi_c *
                    GetP_Hat(
                        sdata_current.x1,
                        sdata_current.n1,
                        canonical.x2,
                        canonical.n2,
                        canonical.L2,
                        sdata_current.o,
                        matOpt,
                        false
                    ) *
                    canonical.W;

        // GI: Evaluate canonical sample
        float mi_c_gi = GenPairwiseMIS_canonical_GI(
            canonical_gi,
            spatial_candidates_GI,
            sdata_current,
            rejected_GI,
            M_sum_GI,
            spatial_M_cap_GI,
            matOpt
        );

        MaterialOptimized mat_gi_c = CreateMaterialOptimized(materials[canonical_gi.mID2], canonical_gi.mID2);
        float3 f_c = GetP_Hat_GI(sdata_current.x1, sdata_current.n1,
                                 canonical_gi.xn, canonical_gi.nn,
                                 canonical_gi.E3, canonical_gi.Vn,
                                 sdata_current.o, matOpt, mat_gi_c, false);
        float w_c_gi = mi_c_gi * LinearizeVector(f_c) * canonical_gi.W;

        reservoir_current.M = min(spatial_M_cap, canonical.M);
        reservoir_current.w_sum = w_c;

        reservoir_current_gi.M = min(spatial_M_cap_GI, canonical_gi.M);
        reservoir_current_gi.w_sum = w_c_gi;

        // Now loop over DI candidate list and incorporate them into DI reservoir
        [loop]
        for (int v = 0; v < spatial_candidate_count; v++)
        {
            if (!rejected_DI[v])
            {
                uint spatial_candidate = spatial_candidates_DI[v];
                float mi_s = GenPairwiseMIS_noncanonical(
                    canonical,
                    spatial_candidate,
                    sdata_current,
                    M_sum_DI,
                    spatial_M_cap,
                    matOpt
                );
                float w_s = mi_s *
                            GetP_Hat(
                                sdata_current.x1,
                                sdata_current.n1,
                                g_Reservoirs_current[spatial_candidate].x2,
                                g_Reservoirs_current[spatial_candidate].n2,
                                g_Reservoirs_current[spatial_candidate].L2,
                                sdata_current.o,
                                matOpt,
                                false
                            ) *
                            g_Reservoirs_current[spatial_candidate].W;

                UpdateReservoir(
                    reservoir_current,
                    w_s,
                    min(spatial_M_cap, g_Reservoirs_current[spatial_candidate].M),
                    g_Reservoirs_current[spatial_candidate].x2,
                    g_Reservoirs_current[spatial_candidate].n2,
                    g_Reservoirs_current[spatial_candidate].L2,
                    seed
                );
            }
        }

        // Now loop over GI candidate list and incorporate them into GI reservoir
        [loop]
        for (int v = 0; v < spatial_candidate_count; v++)
        {
            if (!rejected_GI[v])
            {
                uint spatial_candidate = spatial_candidates_GI[v];
                float mi_s_gi = GenPairwiseMIS_noncanonical_GI(
                    canonical_gi,
                    spatial_candidate,
                    sdata_current,
                    M_sum_GI,
                    spatial_M_cap,
                    matOpt
                );

                MaterialOptimized mat_gi_t_t = CreateMaterialOptimized(
                    materials[g_Reservoirs_current_gi[spatial_candidate].mID2],
                    g_Reservoirs_current_gi[spatial_candidate].mID2
                );

                // Jacobian for path reconnection
                float j_gi = Jacobian_Reconnection(
                    g_sample_current[spatial_candidate],
                    sdata_current,
                    g_Reservoirs_current_gi[spatial_candidate].xn,
                    g_Reservoirs_current_gi[spatial_candidate].nn,
                    g_Reservoirs_current_gi[spatial_candidate].Vn
                );

                float3 f_gi = GetP_Hat_GI(
                    sdata_current.x1,
                    sdata_current.n1,
                    g_Reservoirs_current_gi[spatial_candidate].xn,
                    g_Reservoirs_current_gi[spatial_candidate].nn,
                    g_Reservoirs_current_gi[spatial_candidate].E3,
                    g_Reservoirs_current_gi[spatial_candidate].Vn,
                    sdata_current.o,
                    matOpt,
                    mat_gi_t_t,
                    true
                );
                float w_s_gi = mi_s_gi * LinearizeVector(f_gi) * g_Reservoirs_current_gi[spatial_candidate].W * j_gi;

                if(j_gi != 0.0f){
                    UpdateReservoir_GI(
                        reservoir_current_gi,
                        w_s_gi,
                        min(spatial_M_cap_GI, g_Reservoirs_current_gi[spatial_candidate].M),
                        g_Reservoirs_current_gi[spatial_candidate].xn,
                        g_Reservoirs_current_gi[spatial_candidate].nn,
                        g_Reservoirs_current_gi[spatial_candidate].Vn,
                        g_Reservoirs_current_gi[spatial_candidate].E3,
                        g_Reservoirs_current_gi[spatial_candidate].mID2,
                        seed
                    );
                }
            }
        }

        float p_hat = GetP_Hat(
            sdata_current.x1,
            sdata_current.n1,
            reservoir_current.x2,
            reservoir_current.n2,
            reservoir_current.L2,
            sdata_current.o,
            matOpt,
            true
        );
        reservoir_current.W = GetW(reservoir_current, p_hat);

        // Compute final color from DI
        float3 accumulation = ReconnectDI(
            sdata_current.x1,
            sdata_current.n1,
            reservoir_current.x2,
            reservoir_current.n2,
            reservoir_current.L2,
            sdata_current.o,
            matOpt
        ) * reservoir_current.W;


        // GI -----------------------------------------------------------------------------------
        MaterialOptimized mat_gi_final = CreateMaterialOptimized(materials[reservoir_current_gi.mID2], reservoir_current_gi.mID2);
        float3 f_gi_final = GetP_Hat_GI(
            sdata_current.x1,
            sdata_current.n1,
            reservoir_current_gi.xn,
            reservoir_current_gi.nn,
            reservoir_current_gi.E3,
            reservoir_current_gi.Vn,
            sdata_current.o,
            matOpt,
            mat_gi_final,
            false
        );

        float p_hat_gi = LinearizeVector(f_gi_final);
        reservoir_current_gi.W = GetW_GI(reservoir_current_gi, p_hat_gi);
        accumulation += f_gi_final * reservoir_current_gi.W;


        // DEBUG-------------------------------
        /*accumulation = reservoir_current_gi.W;
        accumulation = 0.0f;
        if(length(sdata_current.debug) < 50.0f && !any(isnan(sdata_current.debug)) && !any(isinf(sdata_current.debug)))
            accumulation = sdata_current.debug;*/
        //accumulation = GetColorFromValue((float)candidateFoundCount_GI,0.0f, (float)spatial_candidate_count);

        // -----------------------------------------------------------
        // TEMPORAL ACCUMULATION
        float3 averagedColor;
        float frameCount = gPermanentData[uint2(launchIndex)].w;
        int maxFrames    = 2000000;

        if (frameCount <= 0.0f &&
            !isnan(accumulation.x) && !isnan(accumulation.y) && !isnan(accumulation.z) &&
            isfinite(accumulation.x) && isfinite(accumulation.y) && isfinite(accumulation.z))
        {
            // Initialize accumulation + frame count
            gPermanentData[uint2(launchIndex)] = float4(accumulation, 1.0f);
            frameCount+= 1.0f;
        }
        else if (frameCount < maxFrames &&
                 !isnan(accumulation.x) && !isnan(accumulation.y) && !isnan(accumulation.z) &&
                 isfinite(accumulation.x) && isfinite(accumulation.y) && isfinite(accumulation.z))
        {
            // Continue accumulating valid samples
            gPermanentData[uint2(launchIndex)].xyz += accumulation;
            gPermanentData[uint2(launchIndex)].w   += 1.0f;
            frameCount+= 1.0f;
        }
        averagedColor = gPermanentData[uint2(launchIndex)].xyz / frameCount;

        // If the view has changed significantly, reset accumulation
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
            // Reset buffer
            gPermanentData[uint2(launchIndex)] = float4(accumulation, 1.0f);
            frameCount = 1.0f;
        }

        // Optionally skip temporal accumulation if desired:
        averagedColor = accumulation;

        // Debug coloring for invalid values
        if (isnan(averagedColor.x) || isnan(averagedColor.y) || isnan(averagedColor.z))
            averagedColor = float3(1, 0, 1); // magenta for NaN
        if (isinf(averagedColor.x) || isinf(averagedColor.y) || isinf(averagedColor.z))
            averagedColor = float3(0, 1, 1); // cyan for infinity

        // Write out the final reservoir for potential temporal reuse next frame
        g_Reservoirs_last[pixelIdx]     = reservoir_current;
        g_Reservoirs_last_gi[pixelIdx]  = reservoir_current_gi;
        g_sample_last[pixelIdx]         = sdata_current;

        // Gamma correct
        float3 finalColor = sRGBGammaCorrection(averagedColor);
        gOutput[uint3(launchIndex, 0)] = float4(finalColor, 1.0f);

        bool r1 = RandomFloat(seed) < 0.0001f? true: false;
        /*if(r1){
            for(int d = 0; d<60; d++){
                uint2 pixel_r_d = GetRandomPixelCircleWeighted_d(
                        spatial_radius,
                        DispatchRaysDimensions().x,
                        DispatchRaysDimensions().y,
                        launchIndex.x,
                        launchIndex.y,
                        seed
                );
                gOutput[uint3(pixel_r_d, 0)] = float4(0,1,0,1);
            }
        }*/
    }
    else
    {
        // If L1 is non-zero, just store that color out.
        float3 finalColor = sRGBGammaCorrection(sdata_current.L1);
        gOutput[uint3(launchIndex, 0)] = float4(finalColor, 1.0f);
    }
}
