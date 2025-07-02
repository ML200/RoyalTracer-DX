cbuffer Push : register(b1)
{
    uint2 gImageSize;
}

#define gImageWidth   (gImageSize.x)
#define gImageHeight  (gImageSize.y)

#define DispatchRaysDimensions() uint3(gImageWidth, gImageHeight, 1)

groupshared uint3 gDispatchIdxShared[1];
static     uint3  gDispatchIdx;
#define DispatchRaysIndex()      gDispatchIdx

#include "Constants_v7.hlsli"
#include "Common_v7.hlsli"
#include "Structures_misc.hlsli"
#include "Random_v7.hlsli"
#include "Compression_v7.hlsli"

RWTexture2DArray<float4> gOutput              : register(u0);
RWTexture2D<float4>      gPermanentData       : register(u1);
RWTexture2DArray<float4>      gScratchPing         : register(u8);   // Storage for denoiser

RWByteAddressBuffer      g_sample_current     : register(u6);
RWByteAddressBuffer      g_sample_last        : register(u7);
RWByteAddressBuffer      g_Reservoirs_current_di : register(u2);
RWByteAddressBuffer      g_Reservoirs_last_di    : register(u3);
RWByteAddressBuffer      g_Reservoirs_current_gi : register(u4);
RWByteAddressBuffer      g_Reservoirs_last_gi    : register(u5);

StructuredBuffer<STriVertex>  BTriVertex        : register(t2);
StructuredBuffer<int>         indices           : register(t1);
RaytracingAccelerationStructure SceneBVH        : register(t0);
StructuredBuffer<InstanceProperties> instanceProps : register(t3);
StructuredBuffer<uint>             materialIDs     : register(t4);
StructuredBuffer<Material>         materials       : register(t5);
StructuredBuffer<LightTriangle>    g_EmissiveTriangles : register(t6);
StructuredBuffer<float>            g_AliasProb     : register(t7);
StructuredBuffer<uint>             g_AliasIdx      : register(t8);

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
    float     time;
}

#include "Camera_ray_v7.hlsli"
#include "NEE_Sampling_v7.hlsli"
#include "Reservoir_DI_v7.hlsli"
#include "Motion_vectors_v7.hlsli"

static const float3 kLUMA = float3(0.2126, 0.7152, 0.0722);

// RGB → luminance helper (rec. 709 constants are fine for HDR too)
inline float Luma(float3 rgb)
{
    return dot(rgb, float3(0.2126, 0.7152, 0.0722));
}

#define uThreshold 0.5f

[numthreads(16, 16, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    if (DTid.x >= gImageWidth || DTid.y >= gImageHeight)
        return;

    uint2 launch = DTid.xy;

    //------------------------------------------------------------------
    // 1. Fetch the centre pixel
    //------------------------------------------------------------------
    float3 centerRGB = gScratchPing[uint3(launch, 0)].rgb;

    //------------------------------------------------------------------
    // 2. Accumulate the eight neighbours (guarding image borders)
    //------------------------------------------------------------------
    float3 neighbourSum = 0.0;
    uint    neighbourCnt = 0;

    [unroll]
    for (int dy = -1; dy <= 1; ++dy)
    {
        [unroll]
        for (int dx = -1; dx <= 1; ++dx)
        {
            if (dx == 0 && dy == 0) continue;               // skip centre

            int2 n = int2(launch) + int2(dx, dy);
            if (n.x < 0 || n.y < 0 || n.x >= int(gImageWidth) || n.y >= int(gImageHeight))
                continue;                                   // clamp outside

            neighbourSum += gScratchPing[uint3(n, 0)].rgb;
            ++neighbourCnt;
        }
    }

    float3 neighbourAvg = neighbourSum / float(neighbourCnt);

    //------------------------------------------------------------------
    // 3. Decide if centre is a firefly
    //------------------------------------------------------------------
    float centerLum     = Luma(centerRGB);
    float neighbourLum  = Luma(neighbourAvg);

    if (centerLum > neighbourLum * (1.0 + uThreshold))
        centerRGB = neighbourAvg;      // replace the outlier

    //------------------------------------------------------------------
    // 4. Store to the "pong" slice
    //------------------------------------------------------------------
    gScratchPing[uint3(launch, 1)] = float4(centerRGB, 1.0);
}