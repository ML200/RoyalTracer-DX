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
#include "NEE_Sampling_v7.hlsli"
#include "Reservoir_DI_v7.hlsli"

[shader("raygeneration")]
void Pass_shading_v7() {
    uint2 launchIndex = DispatchRaysIndex().xy;
    float2 dims       = float2(DispatchRaysDimensions().xy);
    uint pixelIdx     = MapPixelID(dims, launchIndex);

    // Load most recent data
    SampleData sdata = loadSampleData(g_sample_current, pixelIdx);
    float3 accumulation = float3(0,0,0);

    if(all(sdata.L1 < EPSILON)){
        Reservoir_DI rdi = loadReservoirDI(g_Reservoirs_current_di, pixelIdx);

        float3 contribution = ReconnectDI(sdata.x1, sdata.n1, sdata.o, sdata.matID, rdi.x2_di, rdi.n2_di, rdi.L2_di) * rdi.W_di;

        accumulation = float3(contribution);
    }
    else{
        accumulation = float3(sdata.L1);
    }


    // ___ Accumulation ___
    float3 averagedColor;
    float frameCount = gPermanentData[uint2(launchIndex)].w;
    int maxFrames    = 10000000;

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
        if (any(diff > EPSILON))
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
    float3 finalColor = sRGBGammaCorrection(averagedColor);

    // Debug coloring for invalid values
    if (isnan(averagedColor.x) || isnan(finalColor.y) || isnan(finalColor.z))
        finalColor = float3(1, 0, 1); // magenta for NaN
    if (isinf(finalColor.x) || isinf(finalColor.y) || isinf(finalColor.z))
        finalColor = float3(0, 1, 1); // cyan for infinity

    gOutput[uint3(launchIndex, 0)] = float4(finalColor, 1.0f);

}
