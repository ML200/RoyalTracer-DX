cbuffer Push : register(b1)
{
    uint2 gImageSize;
}

#define gImageWidth   (gImageSize.x)
#define gImageHeight  (gImageSize.y)
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

RWTexture2DArray<half4> gOutput             : register(u0);
RWTexture2D<float4>      gPermanentData      : register(u1);
RWTexture2DArray<float4> gScratchPing         : register(u8);

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
#include "NEE_Sampling_v7.hlsli"
#include "Reservoir_DI_v7.hlsli"
#include "Reservoir_GI_v7.hlsli"
#include "Inline_RT.hlsli"
#include "Motion_vectors_v7.hlsli"
#include "HashGrid_v7.hlsli"

// ─────────────────────────────────────────────────────────────────────────────
//  SHADING PASS
// ─────────────────────────────────────────────────────────────────────────────
[numthreads(16, 16, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    if (DTid.x >= gImageWidth || DTid.y >= gImageHeight) return;

    // Load the DI pipeline output
    float3 output_DI = 0.0f;//gScratchPing[uint3(DTid.xy, 1)];
    // Load the GI pipeline output
    float3 output_GI = gScratchPing[uint3(DTid.xy, 2)];

    float3 accumulation = output_DI + output_GI;

    // ───────────────────────── Accumulation (capped) ───────────────────────────
    bool cameraChanged = false;
    [unroll]
    for (uint i = 0; i < 4; ++i) {
        if (any(view[i] != prevView[i])) cameraChanged = true;
    }
    static const float MAX_SAMPLES     = 100000.0h;  // tune to taste

    float4 prev        = gPermanentData[DTid.xy];   // rgb = running avg, a = N
    float3 prevAvg     = prev.rgb;
    float  prevSamples = prev.a;

    float3 newAvg;
    float  newSamples;
    if (cameraChanged)
    {
        // Camera moved: reset running average and sample count
        newAvg     = accumulation;
        newSamples = 1.0h;
    }
    else
    {
        newSamples = min(prevSamples + 1.0h, MAX_SAMPLES);
        float invN  = 1.0h / newSamples;
        newAvg     = mad(accumulation - prevAvg, invN, prevAvg);
    }

    // --- store back to the permanent UAV ----------------------------------------
    gPermanentData[DTid.xy] = float4(newAvg, newSamples);

    // --- display/debug -----------------------------------------------------------
    float3 fColor = sRGBGammaCorrection(newAvg);
    gOutput[uint3(DTid.xy, 0)]  = float4(fColor, 1);

    float3 finalColor = sRGBGammaCorrection(accumulation);
    gOutput[uint3(DTid.xy, 0)]  = float4(finalColor, 1);

    // Denoiser buffers etc.
    /*uint2  launchIndex   = DTid.xy;
    float2 dims = float2(IMG_W, IMG_H);
    uint   pixelIdx  = MapPixelID(dims, launchIndex);
    SampleData sdata = loadSampleData(g_sample_current, pixelIdx);
    SampleData sdata_l = loadSampleData(g_sample_last, pixelIdx);

    gScratchPing[uint3(DTid.xy, 2)] = length(materials[sdata.matID].Ke) > 0.0f ? float4(materials[sdata.matID].Ke, 0) : materials[sdata.matID].Kd;
    gScratchPing[uint3(DTid.xy, 3)] = float4(length(materials[sdata.matID].Ke) > 0.0f ? 1 : 0, materials[sdata.matID].Pr_Pm_Ps_Pc.x, sdata.objID, 0);
    gScratchPing[uint3(DTid.xy, 4)] = float4(sdata.n1,0);

    Reservoir_GI rgi = loadReservoirGI(g_Reservoirs_current_gi, pixelIdx);
    gScratchPing[uint3(DTid.xy, 5)] = float4(sdata.x1, rgi.M_gi);
    gScratchPing[uint3(DTid.xy, 6)] = float4(sdata_l.x1,0);

    gScratchPing[uint3(DTid.xy, 0)] = float4(accumulation, 0);*/
}
