#include "Constants_v7.hlsli"
#include "Common_v7.hlsli"
#include "Structures_misc.hlsli"
#include "Motion_vectors_v7.hlsli"
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
#include "MIS_v7.hlsli"
#include "NEE_Sampling_v7.hlsli"
#include "BSDF_Sampling_v7.hlsli"

[shader("raygeneration")]
void Pass_init_di_v7() {
    uint2 launchIndex = DispatchRaysIndex().xy;
    float2 dims       = float2(DispatchRaysDimensions().xy);
    uint pixelIdx     = MapPixelID(dims, launchIndex);

    SampleData sdata = SampleCameraRay(pixelIdx);

    // Get a random seed
    uint2 seed = GetSeed(pixelIdx, time, 1);
    uint waveSeed = GetWaveSeed(pixelIdx, time, 1);

    Reservoir_DI reservoir = (Reservoir_DI)0;

    // NEE sample(s)
    for(int i = 0; i<NEE_SAMPLES_DI; i++){
        // Get the sample result
        SampleReturn result = SampleNEE(sdata, waveSeed, seed);
        // Calculate contribution and p_hat.
        float3 c = ReconnectDI(sdata.x1, sdata.n1, sdata.o, sdata.matID, result.x2, result.n2, result.L2);
        float p_hat = GetPHat(c);
        float w_mis = MIS_Initial_NEE(result.pdf_nee, result.pdf_bsdf, NEE_SAMPLES_DI, BSDF_SAMPLES_DI) * p_hat / result.pdf_nee;
        if(isnan(w_mis))
            w_mis = 0.0f;
        // Update reservoir
        UpdateReservoirDI(reservoir, w_mis, 0, result.x2, result.n2, result.L2, seed);
    }
    bool requires_shadow_ray = true;
    // BSDF sample(s)
    for(int j = 0; j<BSDF_SAMPLES_DI; j++){
        // Get the sample result
        SampleReturn result = SampleBSDF(sdata, waveSeed, seed);
        // Calculate contribution and p_hat.
        if(any(result.L2 > 0.0f)){
            float3 c = ReconnectDI(sdata.x1, sdata.n1, sdata.o, sdata.matID, result.x2, result.n2, result.L2);
            float p_hat = GetPHat(c);
            float w_mis = MIS_Initial_BSDF(result.pdf_nee, result.pdf_bsdf, NEE_SAMPLES_DI, BSDF_SAMPLES_DI) * p_hat / result.pdf_bsdf;
            if(isnan(w_mis) || isinf(w_mis))
                w_mis = 0.0f;
        // Update reservoir
            if(UpdateReservoirDI(reservoir, w_mis, 0, result.x2, result.n2, result.L2, seed)){
                requires_shadow_ray = false;
            }
        }
    }

    // Visbility check for the stored sample, if fail, set W to 0
    float V = 1.0f;
    if(requires_shadow_ray){
        V = VisibilityCheck(sdata.x1, reservoir.x2_di, sdata.n1);
    }
    // Calculate W
    float p_hat = GetPHat(ReconnectDI(sdata.x1, sdata.n1, sdata.o, sdata.matID, reservoir.x2_di, reservoir.n2_di, reservoir.L2_di));
    if(p_hat > 0.0f)
        reservoir.W_di = V * reservoir.w_sum_di / p_hat;

    // Save the resulting reservoir to memory
    store_x2_di(reservoir.x2_di, g_Reservoirs_current_di, pixelIdx);
    store_n2_di(reservoir.n2_di, g_Reservoirs_current_di, pixelIdx);
    store_L2_di(reservoir.L2_di, g_Reservoirs_current_di, pixelIdx);
    store_W_di(reservoir.W_di, g_Reservoirs_current_di, pixelIdx);
    store_M_di(1, g_Reservoirs_current_di, pixelIdx);
}