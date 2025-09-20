cbuffer Push : register(b1)
{
    uint2 gImageSize;
}
#define ENABLE_RAY_QUERY_INLINE // Activate support for inline ray tracing

#define gImageWidth  (gImageSize.x)
#define gImageHeight (gImageSize.y)
#define IMG_W        (gImageSize.x)
#define IMG_H        (gImageSize.y)

#define DispatchRaysDimensions() uint3(gImageWidth, gImageHeight, 1)

static uint3 gDispatchIdx;
#define DispatchRaysIndex()      gDispatchIdx

#include "Constants_v7.hlsli"
#include "Common_v7.hlsli"
#include "Structures_misc.hlsli"
#include "Random_v7.hlsli"
#include "Compression_v7.hlsli"

RWTexture2DArray<float4> gOutput             : register(u0);
RWTexture2D<float4>      gPermanentData      : register(u1);
RWTexture2DArray<half4> gScratchPing         : register(u8);

RWByteAddressBuffer g_sample_current         : register(u6);
RWByteAddressBuffer g_sample_last            : register(u7);
RWByteAddressBuffer g_Reservoirs_current_di  : register(u2);
RWByteAddressBuffer g_Reservoirs_last_di     : register(u3);
RWByteAddressBuffer g_Reservoirs_current_gi  : register(u4);
RWByteAddressBuffer g_Reservoirs_last_gi     : register(u5);
RWByteAddressBuffer g_InitialBSDFRays : register(u9);

StructuredBuffer<STriVertex>          BTriVertex        : register(t2);
StructuredBuffer<int>                 indices           : register(t1);
RaytracingAccelerationStructure       SceneBVH          : register(t0);
StructuredBuffer<InstanceProperties>  instanceProps     : register(t3);
StructuredBuffer<uint>                materialIDs       : register(t4);
StructuredBuffer<Material>            materials         : register(t5);
StructuredBuffer<LightTriangle>       g_EmissiveTriangles : register(t6);
StructuredBuffer<float>               g_AliasProb       : register(t7);
StructuredBuffer<uint>                g_AliasIdx        : register(t8);

// Light tree
StructuredBuffer<LightTLASNodeGpu> gLT_TLAS        : register(t9);
StructuredBuffer<LightBLASNodeGpu> gLT_BLAS        : register(t10);
StructuredBuffer<BlasRangeGpu>     gLT_Range       : register(t11);
Buffer<uint>                       gLT_LeafTriIndex: register(t12);
Buffer<float>                      gLT_LeafAliasProb : register(t13);
Buffer<uint>                       gLT_LeafAliasIdx  : register(t14);

// Needs access to all structured/random buffers
#include "LightTree_v7.hlsli"
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
#include "Reservoir_DI_v7.hlsli"
#include "Reservoir_GI_v7.hlsli"
#include "Inline_RT.hlsli"
#include "Camera_ray_v7.hlsli"
#include "MIS_v7.hlsli"
#include "NEE_Sampling_v7.hlsli"
#include "BSDF_Sampling_v7.hlsli"
#include "Motion_vectors_v7.hlsli"

//─────────────────────────────────────────────────────────────────────────────
//  Initial Sampling GI
//─────────────────────────────────────────────────────────────────────────────
[numthreads(16, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= IMG_W || tid.y >= IMG_H) return;
    gDispatchIdx = tid;

    uint2  launchIndex   = tid.xy;
    float2 dims = float2(IMG_W, IMG_H);
    uint   pixelIdx  = MapPixelID(dims, launchIndex);

    // Initialize the GI reservoir and trace the path (2-4 consecutive bsdf rays)
    Reservoir_GI reservoir = (Reservoir_GI)0;
    // Get a random seed
    uint2 seed = GetSeed(pixelIdx, time, 1);
    uint waveSeed = GetWaveSeed(pixelIdx, time, 1);

    // Load sample data
    SampleData sdata = loadSampleData(g_sample_current, pixelIdx);
    // We need position, normal, outgoing, matID
    // Get a sample
    SampleReturn result_init = SampleBSDF_gen(sdata.x1, sdata.n1, sdata.matID, sdata.o, waveSeed, seed);


    if(length(result_init.n2) > 0.0f){
        bool requires_shadow_ray = true; // Set to false whenever a bsdf ray wins beeing added to the reservoir in the end
        float p_hat_final = 0.0f; // P hat cache from the iteration -> no recompute required.
        // Postponed shadow ray
        float3 s_x1 = (float3)0;
        float3 s_x2 = (float3)0;
        float3 s_n1 = (float3)0;

        // Store path variables that are used to fill the reservoirs
        float3 V2 = (float3)0;

        // Store path variables
        // Variables to cache path data
        float3 position = result_init.x2/*load_x2_init(g_InitialBSDFRays, pixelIdx)*/;
        float3 normal = result_init.n2/*normalize(load_n2_init(g_InitialBSDFRays, pixelIdx))*/;
        float3 outgoing = normalize(sdata.x1 - position);
        uint matID = result_init.matID/*load_matID_init(g_InitialBSDFRays, pixelIdx)*/;
        // full path throughput
        float3 tp_full = ReconnectGISingle(sdata.x1/*load_x1(g_sample_current, pixelIdx)*/, sdata.n1/*load_n1(g_sample_current, pixelIdx)*/, sdata.o/*load_o(g_sample_current, pixelIdx)*/, sdata.matID/*load_matID(g_sample_current, pixelIdx)*/, position, normal, float3(1,1,1)); // reconnect without L

        // partial path throughput (from x3 onward, used to set L2 in the reservoir)
        float3 tp_partial = float3(1,1,1);

        // Full path pdf of a given subpath
        float3 ldir = position - sdata.x1/*load_x1(g_sample_current, pixelIdx)*/;
        float dist2 = length(ldir) * length(ldir);
        float pdf_full = result_init.pdf_bsdf;///*load_pdfB_init(g_InitialBSDFRays, pixelIdx)*/ * dist2 /dot(normalize(-ldir), normal); // Convert to solid angle space

        for(int i = 0; i < BSDF_SAMPLES_GI; i++){
            {
                // NEE samples
                for(int j = 0; j<NEE_SAMPLES_GI; j++){
                    // Get the sample result
                    SampleReturn result = SampleNEE_gen(position, normal, matID, outgoing, waveSeed, seed);
                    if(any(result.L2 > 0.0f)){
                        // Calculate contribution and p_hat.
                        float3 c = ReconnectGISingle(position, normal, outgoing, matID, result.x2, result.n2, float3(1,1,1));
                        float p_hat = GetPHat(c * tp_full * result.L2);
                        float pdf = result.pdf_nee * pdf_full;
                        float w_mis = MIS_Initial_NEE(result.pdf_nee, result.pdf_bsdf, NEE_SAMPLES_GI, 1) * p_hat / pdf;// * VisibilityCheckCP(position, result.x2, normal);
                        if(isnan(w_mis))
                            w_mis = 0.0f;

                        float3 L2 = result.L2;
                        float3 V2_temp = V2;
                        if(i != 0)
                            L2 *= c * tp_partial;
                        else {
                            V2_temp = position - result.x2;
                            float3 V2_norm = normalize(V2_temp);
                            L2 *= G_term(normal, V2_norm);
                        }
                        // Update reservoir
                        if(UpdateReservoirGI(reservoir, w_mis, 0, 0, 0, L2, normalize(V2_temp), 0, 0, 0, 0, 0, seed)){
                            p_hat_final = p_hat;
                            s_x1 = position;
                            s_x2 = result.x2;
                            s_n1 = normal;
                        }
                    }
                }
            }

            // BSDF advancement
            {
                // Get a sample direction
                SampleReturn result = SampleBSDF_gen(position, normal, matID, outgoing, waveSeed, seed);
                float3 c = ReconnectGISingle(position, normal, outgoing, matID, result.x2, result.n2, float3(1,1,1));
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
                        L2 *= G_term(normal, V2_norm);
                    }
                    // Update reservoir with the sub path
                    if(UpdateReservoirGI(reservoir, w_mis, 0, 0, 0, L2, normalize(V2_temp), 0, 0, 0, 0, 0, seed)){
                        requires_shadow_ray = false;
                        p_hat_final = p_hat;
                    }
                    break;
                }
                else{
                    if(any(result.n2 > 0.0f)){
                        if(i == 0){
                            V2 = position - result.x2;
                            tp_partial *= G_term(normal, normalize(V2));
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
        }

       // Visbility check for the stored sample, if fail, set W to 0
        float V = 1.0f;
        if(requires_shadow_ray && length(s_n1)>EPSILON){
            V = VisibilityCheckCP(s_x1, s_x2, s_n1);
        }
        reservoir.L2_gi *= V;
        // Calculate W
        float W = 0.0f;
        if (p_hat_final > EPSILON) {
            W = V * reservoir.w_sum_gi / p_hat_final;
            // Protect against NaN/Inf
            if (isnan(W) || isinf(W)) {
                reservoir.n2_gi = 0; // Dont pick this sample, its nan!
            }
        }
        reservoir.W_gi = W;

        reservoir.objID_gi = result_init.objID/*load_objID_init(g_InitialBSDFRays, pixelIdx)*/;
        reservoir.matID_gi = result_init.matID/*load_matID_init(g_InitialBSDFRays, pixelIdx)*/;
        reservoir.x2_gi = result_init.x2/*load_x2_init(g_InitialBSDFRays, pixelIdx)*/;
        reservoir.n2_gi = result_init.n2/*load_n2_init(g_InitialBSDFRays, pixelIdx)*/;
        reservoir.M_gi = 1;
    }
    storeReservoirGI(g_Reservoirs_current_gi, pixelIdx, reservoir);
}
