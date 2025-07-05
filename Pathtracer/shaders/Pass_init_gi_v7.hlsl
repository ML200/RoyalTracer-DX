#include "Constants_v7.hlsli"
#include "Common_v7.hlsli"
#include "Structures_misc.hlsli"
#include "Random_v7.hlsli"
#include "Compression_v7.hlsli"

RWTexture2DArray<float4> gOutput : register(u0);
RWTexture2D<float4> gPermanentData : register(u1);

RWByteAddressBuffer g_sample_current : register(u6);
RWByteAddressBuffer g_sample_last : register(u7);
RWByteAddressBuffer g_Reservoirs_current_di : register(u2);
RWByteAddressBuffer g_Reservoirs_last_di : register(u3);
RWByteAddressBuffer g_Reservoirs_current_gi : register(u4);
RWByteAddressBuffer g_Reservoirs_last_gi : register(u5);
RWByteAddressBuffer g_InitialBSDFRays : register(u9);

StructuredBuffer<STriVertex> BTriVertex : register(t2);
StructuredBuffer<int> indices : register(t1);
RaytracingAccelerationStructure SceneBVH : register(t0);
StructuredBuffer<InstanceProperties> instanceProps : register(t3);
StructuredBuffer<uint> materialIDs : register(t4);
StructuredBuffer<Material> materials : register(t5);
StructuredBuffer<LightTriangle> g_EmissiveTriangles : register(t6);
StructuredBuffer<float> g_AliasProb  : register(t7);
StructuredBuffer<uint>  g_AliasIdx   : register(t8);

// Needs access to all structured/random buffers
#include "Sample_data.hlsli"
#include "Initial_bsdf.hlsli"
#include "GGX_v7.hlsli"
#include "Lambertian_v7.hlsli"
#include "BSDF_v7.hlsli"

cbuffer CameraParams : register(b0)
{
    float4x4 view;
    float4x4 projection;
    float4x4 viewI;
    float4x4 projectionI;
    float4x4 prevView;
    float4x4 prevProjection;
    float time;
}
// These includes need access to ALL previous buffers
#include "Camera_ray_v7.hlsli"
#include "Reservoir_DI_v7.hlsli"
#include "Reservoir_GI_v7.hlsli"
#include "MIS_v7.hlsli"
#include "NEE_Sampling_v7.hlsli"
#include "BSDF_Sampling_v7.hlsli"
#include "Motion_vectors_v7.hlsli"

[shader("raygeneration")]
void Pass_init_gi_v7() {
    uint2 launchIndex = DispatchRaysIndex().xy;
    float2 dims       = float2(DispatchRaysDimensions().xy);
    uint pixelIdx     = MapPixelID(dims, launchIndex);

    // Load initial sample data
    SampleData sdata = loadSampleData(g_sample_current, pixelIdx);
    // path_x2 is x2
    float3 px2 = load_x2_init(g_InitialBSDFRays, pixelIdx);
    // path_n2 is n2
    float3 pn2 = load_n2_init(g_InitialBSDFRays, pixelIdx);
    if(sdata.matID != 4294967294 && all(sdata.L1 < EPSILON) && any(pn2 != 0.0f)){
        // Get a random seed
        uint2 seed = GetSeed(pixelIdx, time, 1);
        uint waveSeed = GetWaveSeed(pixelIdx, time, 1);

        // Initialize the GI reservoir and trace the path (2-4 consecutive bsdf rays)
        Reservoir_GI reservoir = (Reservoir_GI)0;

        // Store path variables that are used to fill the reservoirs
        float3 V2 = (float3)0;

        // Store path variables
        // full path throughput
        float3 tp_full = ReconnectDI(sdata.x1, sdata.n1, sdata.o, sdata.matID, px2, pn2, float3(1,1,1)); // reconnect without L
        // partial path throughput (from x3 onward, used to set L2 in the reservoir)
        float3 tp_partial = float3(1,1,1);

        // Full path pdf of a given subpath
        float pdf_full = load_pdfB_init(g_InitialBSDFRays, pixelIdx);
        bool requires_shadow_ray = true; // Set to false whenever a bsdf ray wins beeing added to the reservoir in the end
        float p_hat_final = 0.0f; // P hat cache from the iteration -> no recompute required.
        // Postponed shadow ray
        float3 s_x1;
        float3 s_x2;
        float3 s_n1;

        // Variables to cache path data
        float3 position = px2;
        float3 normal = pn2;
        float3 outgoing = normalize(sdata.x1 - px2);
        uint matID = load_matID_init(g_InitialBSDFRays, pixelIdx);


        for(int i = 0; i < BSDF_SAMPLES_GI; i++){
            // NEE samples
            for(int j = 0; j<NEE_SAMPLES_GI; j++){
                // Get the sample result
                SampleReturn result = SampleNEE(sdata, waveSeed, seed);
                // Calculate contribution and p_hat.
                float3 c = ReconnectDI(position, normal, outgoing, matID, result.x2, result.n2, float3(1,1,1));
                tp_full *= c;
                float p_hat = GetPHat(tp_full * result.L2);
                float pdf = result.pdf_nee * pdf_full;
                float w_mis = MIS_Initial_NEE(result.pdf_nee, result.pdf_bsdf, NEE_SAMPLES_GI, 1) * p_hat / pdf;
                if(isnan(w_mis))
                    w_mis = 0.0f;

                float3 L2 = result.L2;
                float3 V2_temp = V2;
                if(i != 0)
                    L2 *= tp_partial;
                else {
                    V2_temp = px2 - result.x2;
                    L2 *= J_term(result.n2, normalize(V2_temp), length(V2_temp)) * G_term(result.n2, normalize(-V2_temp));
                }

                // Update reservoir
                if(UpdateReservoirGI(reservoir, w_mis, 0, px2, pn2, L2, normalize(V2_temp), seed)){
                    s_x1 = position;
                    s_x2 = result.x2;
                    s_n1 = normal;
                }
            }

            // BSDF advancement
            {
                // Get a sample direction
                SampleReturn result = SampleBSDF(sdata, waveSeed, seed);
                float3 c = ReconnectDI(position, normal, outgoing, matID, result.x2, result.n2, float3(1,1,1));
                // Calculate contribution and p_hat.
                if(any(result.L2 > 0.0f)){
                    tp_full *= c;
                    float p_hat = GetPHat(tp_full * result.L2);
                    float pdf = result.pdf_bsdf * pdf_full;
                    float w_mis = MIS_Initial_BSDF(result.pdf_nee, result.pdf_bsdf, NEE_SAMPLES_DI, BSDF_SAMPLES_DI) * p_hat / pdf;
                    if(isnan(w_mis) || isinf(w_mis))
                        w_mis = 0.0f;

                    float3 L2 = result.L2;
                    float3 V2_temp = V2;
                    if(i != 0)
                        L2 *= tp_partial;
                    else {
                        V2_temp = px2 - result.x2;
                        L2 *= J_term(result.n2, normalize(V2_temp), length(V2_temp)) * G_term(result.n2, normalize(-V2_temp));
                    }

                    // Update reservoir with the sub path
                    if(UpdateReservoirGI(reservoir, w_mis, 0, px2, pn2, L2, V2_temp, seed)){
                        requires_shadow_ray = false;
                }
                else{
                    // Advance ray
                    pdf_full *= result.pdf_bsdf;
                    tp_full *= c;
                    outgoing = position - result.x2;
                    position = result.x2;
                    normal = result.n2;
                    matID = result.matID;

                    if(i == 0)
                        V2 = px2 - result.x2;
                    else
                        tp_partial *= c;
                }
            }
        }

        // Visbility check for the stored sample, if fail, set W to 0
        /*float V = 1.0f;
        if(requires_shadow_ray){
            V = VisibilityCheck(s_x1, s_x2, s_n1);
        }
        // Calculate W
        float p_hat = GetPHat(ReconnectDI(sdata.x1, sdata.n1, sdata.o, sdata.matID, reservoir.x2_di, reservoir.n2_di, reservoir.L2_di));
        float W = 0.0f;
        if (p_hat > EPSILON) {
            W = V * reservoir.w_sum_di / p_hat;
            // Protect against NaN/Inf
            if (isnan(W) || isinf(W)) {
                W = 0.0f;
            }
        }
        reservoir.W_gi = W;*/


        // Save the resulting reservoir to memory
        store_x2_gi(reservoir.x2_gi, g_Reservoirs_current_gi, pixelIdx, sdata.objID);
        store_n2_gi(reservoir.n2_gi, g_Reservoirs_current_gi, pixelIdx, sdata.objID);
        store_L2_gi(reservoir.L2_gi, g_Reservoirs_current_gi, pixelIdx);
        store_V2_gi(reservoir.V2_gi, g_Reservoirs_current_gi, pixelIdx);
        store_W_gi(reservoir.W_gi, g_Reservoirs_current_gi, pixelIdx);
        store_M_gi(1, g_Reservoirs_current_gi, pixelIdx);
        }
    }
}
