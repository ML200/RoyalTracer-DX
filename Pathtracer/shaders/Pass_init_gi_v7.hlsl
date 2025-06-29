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
    // Also load the initial bsdf ray direction from the DI pass. If it is 0,0,0, the pass hit a light, do nothing
    InitialBSDFRay rdata = loadInitialBSDFRay(g_InitialBSDFRays, pixelIdx);
    if(sdata.matID != 4294967294 && all(sdata.L1 < EPSILON) && any(rdata.n2 != 0.0f)){
        // Get a random seed
        uint2 seed = GetSeed(pixelIdx, time, 1);
        uint waveSeed = GetWaveSeed(pixelIdx, time, 1);

        // Initialize the GI reservoir and trace the path (2-4 consecutive bsdf rays)
        Reservoir_GI reservoir = (Reservoir_GI)0;

        // Store path variables that are used to fill the reservoirs
        // path_x2 is rdata.x2
        // path_n2 is rdata.n2
        //L2
        float3 L2 = (float3)0;
        // V2
        float3 V2 = (float3)0;

        // Store path variables
        // full path throughput
        float3 tp_full = Reconnect_partial(sdata.x1, sdata.n1, sdata.o, sdata.matID, rdata.x2);
        //float3 tp_full = ReconnectGI(sdata.x1, sdata.n1, sdata.o, sdata.matID, rdata.matID, rdata.x2, rdata.n2, float3(0,0,0), rdata.n2);
        // partial path throughput (from x3 onward, used to set L2 in the reservoir)
        float3 tp_partial = float3(1,1,1);
        // Full path pdf of a given subpath
        float pdf_full = rdata.pdf_bsdf;

        // Variables to cache path data
        float3 position = rdata.x2;
        float3 normal = rdata.n2;
        float3 outgoing = normalize(sdata.x1 - rdata.x2);
        utin matID = rdata.matID;

        for(int i = 0; i < BSDF_SAMPLES_GI; i++){
            {
                // Get a sample direction
                SampleReturn result = SampleBSDF(sdata, waveSeed, seed);
                // Calculate contribution and p_hat.
                if(any(result.L2 > 0.0f)){
                    float3 c = ReconnectDI(position, normal, outgoing, matID, result.x2, result.n2, result.L2);
                    float p_hat = GetPHat(c);
                    float w_mis = MIS_Initial_BSDF(result.pdf_nee, result.pdf_bsdf, NEE_SAMPLES_DI, BSDF_SAMPLES_DI) * p_hat / result.pdf_bsdf;
                    if(isnan(w_mis) || isinf(w_mis))
                        w_mis = 0.0f;
                    // Update reservoir with the sub path
                    //UpdateReservoirDI(reservoir, w_mis, 0, result.x2, result.n2, result.L2, seed);
                    return;
                }
                else{
                    // Advance ray

                }
            }
        }


        // Save the resulting reservoir to memory
        store_x2_gi(reservoir.x2_gi, g_Reservoirs_current_gi, pixelIdx);
        store_n2_gi(reservoir.n2_gi, g_Reservoirs_current_gi, pixelIdx);
        store_L2_gi(reservoir.L2_gi, g_Reservoirs_current_gi, pixelIdx);
        store_V2_gi(reservoir.V2_gi, g_Reservoirs_current_gi, pixelIdx);
        store_W_gi(reservoir.W_gi, g_Reservoirs_current_gi, pixelIdx);
        store_M_gi(1, g_Reservoirs_current_gi, pixelIdx);
    }
}
