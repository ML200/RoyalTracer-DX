cbuffer Push : register(b1)
{
    uint2 gImageSize;
}

#define gImageWidth   (gImageSize.x)
#define gImageHeight  (gImageSize.y)

#define DispatchRaysDimensions() uint3(gImageWidth, gImageHeight, 1)

static uint3 gDispatchIdx;
#define DispatchRaysIndex()      gDispatchIdx

#include "Constants_v7.hlsli"
#include "Common_v7.hlsli"
#include "Structures_misc.hlsli"
#include "Random_v7.hlsli"
#include "Compression_v7.hlsli"

RWTexture2DArray<half4> gOutput             : register(u0);
RWTexture2D<half4>      gPermanentData      : register(u1);
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
#include "NEE_Sampling_v7.hlsli"
#include "Reservoir_DI_v7.hlsli"
#include "Reservoir_GI_v7.hlsli"
#include "Motion_vectors_v7.hlsli"

// ─────────────────────────────────────────────────────────────────────────────
//  SHADING PASS
// ─────────────────────────────────────────────────────────────────────────────
[numthreads(16, 16, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    if (DTid.x >= gImageWidth || DTid.y >= gImageHeight) return;

    // Load the DI pipeline output
    float3 output_DI = gScratchPing[uint3(DTid.xy, 1)];
    // Load the GI pipeline output
    //float3 output_GI = ;

    float3 accumulation = output_DI;

    // ───────────────────────── Accumulation (capped) ───────────────────────────
    bool cameraChanged = false;
    [unroll]
    for (uint i = 0; i < 4; ++i) {
        if (any(view[i] != prevView[i])) cameraChanged = true;
    }
    static const half MAX_SAMPLES     = 500.0h;  // tune to taste

    half4 prev        = gPermanentData[DTid.xy];   // rgb = running avg, a = N
    half3 prevAvg     = prev.rgb;
    half  prevSamples = prev.a;

    // --- online running average --------------------------------------------------
    half3 newAvg;
    half  newSamples;
    if (cameraChanged)
    {
        // Camera moved: reset running average and sample count
        newAvg     = accumulation;
        newSamples = 1.0h;
    }
    else
    {
        newSamples = min(prevSamples + 1.0h, MAX_SAMPLES);   // clamp N
        half invN  = 1.0h / newSamples;
        newAvg     = mad(accumulation - prevAvg, invN, prevAvg);
    }

    // --- store back to the permanent UAV ----------------------------------------
    gPermanentData[DTid.xy] = half4(newAvg, newSamples);

    // --- display/debug -----------------------------------------------------------
    float3 fColor = sRGBGammaCorrection(newAvg);
    //gOutput[uint3(DTid.xy, 0)]  = half4(fColor, 1);

    float3 finalColor = sRGBGammaCorrection(accumulation);
    gOutput[uint3(DTid.xy, 0)] = float4(finalColor, 1);
}
