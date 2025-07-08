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

    // DEBUG PIXEL
    float3 debugPixel = float3(0,0,0);

    // Initialize the GI reservoir and trace the path (2-4 consecutive bsdf rays)
    Reservoir_GI reservoir = (Reservoir_GI)0;

    float3 L1 = load_L1(g_sample_current, pixelIdx);
    uint matIDtep = load_matID(g_sample_current, pixelIdx);
    if(matIDtep != 4294967294 && all(L1 < EPSILON) && any(load_n2_init(g_InitialBSDFRays, pixelIdx) != 0.0f)){
        // Get a random seed
        uint2 seed = GetSeed(pixelIdx, time, 1);
        uint waveSeed = GetWaveSeed(pixelIdx, time, 1);

        // Store path variables that are used to fill the reservoirs
        float3 V2 = (float3)0;

        // Store path variables
        // Variables to cache path data
        float3 position = load_x2_init(g_InitialBSDFRays, pixelIdx);
        float3 normal = normalize(load_n2_init(g_InitialBSDFRays, pixelIdx));
        float3 outgoing = normalize(load_x1(g_sample_current, pixelIdx) - position);
        uint matID = load_matID_init(g_InitialBSDFRays, pixelIdx);
        // full path throughput
        float3 tp_full = ReconnectDI(load_x1(g_sample_current, pixelIdx), load_n1(g_sample_current, pixelIdx), load_o(g_sample_current, pixelIdx), load_matID(g_sample_current, pixelIdx), position, normal, float3(1,1,1)); // reconnect without L

        // partial path throughput (from x3 onward, used to set L2 in the reservoir)
        float3 tp_partial = float3(1,1,1);

        // Full path pdf of a given subpath
        float pdf_full = load_pdfB_init(g_InitialBSDFRays, pixelIdx);
        bool requires_shadow_ray = true; // Set to false whenever a bsdf ray wins beeing added to the reservoir in the end
        float p_hat_final = 0.0f; // P hat cache from the iteration -> no recompute required.
        // Postponed shadow ray
        float3 s_x1 = (float3)0;
        float3 s_x2 = (float3)0;
        float3 s_n1 = (float3)0;

        for(int i = 0; i < BSDF_SAMPLES_GI; i++){
            // NEE samples
            for(int j = 0; j<NEE_SAMPLES_GI; j++){
                // Get the sample result
                SampleReturn result = SampleNEE_gen(position, normal, matID, outgoing, waveSeed, seed);
                // Calculate contribution and p_hat.
                float3 c = ReconnectDI(position, normal, outgoing, matID, result.x2, result.n2, float3(1,1,1));
                float p_hat = GetPHat(c * tp_full * result.L2);
                float pdf = result.pdf_nee * pdf_full;
                float w_mis = MIS_Initial_NEE(result.pdf_nee, result.pdf_bsdf, NEE_SAMPLES_GI, 1) * p_hat / pdf;
                if(isnan(w_mis))
                    w_mis = 0.0f;

                float3 L2 = result.L2;
                float3 V2_temp = V2;
                if(i != 0)
                    L2 *= c * tp_partial;
                else {
                    V2_temp = position - result.x2;
                    float3 V2_norm = normalize(V2_temp);
                    L2 *= J_term(result.n2, V2_norm, length(V2_temp)) * G_term(normal, V2_norm);
                }

                // Update reservoir
                if(UpdateReservoirGI(reservoir, w_mis, 0, load_x2_init(g_InitialBSDFRays, pixelIdx), load_n2_init(g_InitialBSDFRays, pixelIdx), L2, normalize(V2_temp), 0, 0, seed)){
                    p_hat_final = p_hat;
                    s_x1 = position;
                    s_x2 = result.x2;
                    s_n1 = normal;
                }
            }

            // BSDF advancement
            {
                // Get a sample direction
                SampleReturn result = SampleBSDF_gen(position, normal, matID, outgoing, waveSeed, seed);
                float3 c = ReconnectDI(position, normal, outgoing, matID, result.x2, result.n2, float3(1,1,1));
                // Calculate contribution and p_hat.
                if(any(result.L2 > 0.0f)){
                    tp_full *= c;
                    float p_hat = GetPHat(tp_full * result.L2);
                    float pdf = result.pdf_bsdf * pdf_full;
                    float w_mis = MIS_Initial_BSDF(result.pdf_nee, result.pdf_bsdf, NEE_SAMPLES_GI, 1) * p_hat / pdf;
                    if(isnan(w_mis) || isinf(w_mis))
                        w_mis = 0.0f;

                    float3 L2 = result.L2;
                    float3 V2_temp = V2;
                    if(i != 0)
                        L2 *= tp_partial * c;
                    else {
                        V2_temp = position - result.x2;
                        float3 V2_norm = normalize(V2_temp);
                        L2 *= J_term(result.n2, V2_norm, length(V2_temp)) * G_term(normal, V2_norm);
                    }

                    // Update reservoir with the sub path
                    if(UpdateReservoirGI(reservoir, w_mis, 0, load_x2_init(g_InitialBSDFRays, pixelIdx), load_n2_init(g_InitialBSDFRays, pixelIdx), L2, normalize(V2_temp), 0,0, seed)){
                        requires_shadow_ray = false;
                        p_hat_final = p_hat;
                    }
                    break;
                }
                else{
                    if(i == 0){
                        V2 = position - result.x2;
                        tp_partial *= J_term(result.n2, normalize(V2), length(V2)) * G_term(normal, normalize(V2));
                    }
                    else
                        tp_partial *= c;

                    // Advance ray
                    pdf_full *= result.pdf_bsdf;
                    tp_full *= c;
                    outgoing = normalize(position - result.x2);
                    position = result.x2;
                    normal = result.n2;
                    matID = result.matID;
                }
            }
        }

        // Visbility check for the stored sample, if fail, set W to 0
        float V = 1.0f;
        if(requires_shadow_ray && length(s_n1)>EPSILON){
            V = VisibilityCheck(s_x1, s_x2, s_n1);
        }
        // Calculate W
        float W = 0.0f;
        if (p_hat_final > 0.0f) {
            W = V * reservoir.w_sum_gi / p_hat_final;
            // Protect against NaN/Inf
            if (isnan(W) || isinf(W)) {
                W = 0.0f;
            }
        }
        reservoir.W_gi = W;

    }
    store_x2_gi(reservoir.x2_gi, g_Reservoirs_current_gi, pixelIdx, load_objID_init(g_InitialBSDFRays, pixelIdx));
    store_n2_gi(reservoir.n2_gi, g_Reservoirs_current_gi, pixelIdx, load_objID_init(g_InitialBSDFRays, pixelIdx));
    store_L2_gi(reservoir.L2_gi, g_Reservoirs_current_gi, pixelIdx);
    store_V2_gi(reservoir.V2_gi, g_Reservoirs_current_gi, pixelIdx);
    store_W_gi(reservoir.W_gi, g_Reservoirs_current_gi, pixelIdx);
    store_M_gi(1, g_Reservoirs_current_gi, pixelIdx);
    store_objID_gi(load_objID_init(g_InitialBSDFRays, pixelIdx), g_Reservoirs_current_gi, pixelIdx);
    store_matID_gi(load_matID_init(g_InitialBSDFRays, pixelIdx), g_Reservoirs_current_gi, pixelIdx);
}
