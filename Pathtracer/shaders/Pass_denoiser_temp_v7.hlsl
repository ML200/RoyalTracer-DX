cbuffer Push : register(b1)
{
    uint2 gImageSize;
}

#define gImageWidth   (gImageSize.x)
#define gImageHeight  (gImageSize.y)

#define DispatchRaysDimensions() uint3(gImageWidth, gImageHeight, 1)

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
#include "Denoiser_helper_v7.hlsli"

static const half3 kLUMA = half3(0.2126h, 0.7152h, 0.0722h);
//------------------------------------------------------------------------------
//  Main kernel
//------------------------------------------------------------------------------
[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint2 launch = dispatchThreadId.xy;
    if (any(launch >= uint2(gImageWidth, gImageHeight)))
        return;   // out of bounds guard

    const half2 dimsH = half2(gImageWidth, gImageHeight);

    //----------------------------------------------------------------------
    //  G-buffer pulls (current frame)
    //----------------------------------------------------------------------
    float4 misc3     = gScratchPing[uint3(launch, 3)];
    float  roughness = misc3.y;
    uint   objIdCur  = (uint)misc3.z;
    uint   isEmissive= (uint)misc3.x;
    if (isEmissive)           // emissive pixels do not accumulate history
        return;

    half3  Ccur      = (half3)gScratchPing[uint3(launch, 0)].rgb;

    float3 x1Cur     = gScratchPing[uint3(launch, 5)].xyz;  // world-space pos
    float3 n1Cur     = gScratchPing[uint3(launch, 4)].xyz;  // world-space normal

    //----------------------------------------------------------------------
    //  1. Bilinear reprojection (4 taps)
    //----------------------------------------------------------------------
    WeightedPixel taps[4];
    GetBilinearReprojectedPixels_d(
        x1Cur, prevView, prevProjection,
        float2(dimsH), objIdCur,
        taps);

    //----------------------------------------------------------------------
    //  2. Validate taps and build weighted history colour
    //----------------------------------------------------------------------
    float  sumW    = 0.0;
    float3 histRGB = 0.0;
    bool   anyOK   = false;

    const float3 camPos = viewI[3].xyz;

    [unroll]
    for (int i = 0; i < 4; ++i)
    {
        if (taps[i].w == 0.0 || all(taps[i].pix == kInvalidPixel))
            continue;                    // rejected by helper

        int2 pix      = taps[i].pix;
        float4 g8     = gScratchPing[uint3(pix, 8)];   // emissive / objId (prev)

        if (g8.x != 0.0 || (uint)g8.z != objIdCur)     // emissive or object mismatch
            continue;

        // Geometry checks
        float3 xPrev  = gScratchPing[uint3(pix,10)].xyz;
        float3 nPrev  = gScratchPing[uint3(pix, 9)].xyz;

        if (dot(n1Cur, nPrev) <= 0.9 ||
            DDistanceWeight(x1Cur, xPrev, n1Cur) <= 0.0)
            continue;

        float4 hTap = gPermanentData[pix];     // history colour (alpha = Nprev)
        bool any_nan = isnan(hTap.r) || isnan(hTap.g) || isnan(hTap.b);
        if (hTap.a == 0.0 || any_nan)                     // no history stored
            continue;

        histRGB += taps[i].w * hTap.rgb;
        sumW    += taps[i].w;
        anyOK    = true;
    }

    //----------------------------------------------------------------------
    //  3. Choose history colour (if any survives)
    //----------------------------------------------------------------------
    float3 histCol   = Ccur;
    bool   histValid = false;

    if (anyOK && sumW > 0.0)
    {
        float invSum = rcp(sumW);
        histCol   = histRGB * invSum;
        histValid = true;
    }

    //----------------------------------------------------------------------
    //  4. Temporal accumulation (running mean, capped at 64 samples)
    //----------------------------------------------------------------------
    float4 prev = gPermanentData[launch];   // .rgb = Cprev, .a = Nprev
    float3 Cprev = histValid ? histCol           // replace with reprojected hist
                             : (prev.a == 0.0 ? 0.0.xxx : prev.rgb);
    float  Nprev = histValid ? prev.a            // keep previous count
                             : (histValid ? prev.a : 0.0);

    float Nnew   = min(Nprev + 1.0, 32);
    float alpha  = 1.0 / Nnew;                   // incremental weight
    half3 Cacc   = lerp((half3)Cprev, Ccur, (half)alpha);

    //----------------------------------------------------------------------
    //  5. Store results
    //----------------------------------------------------------------------
    gPermanentData[launch]        = float4((float3)Cacc, Nnew);
    gScratchPing[uint3(launch,1)] = float4((float3)Cacc, 0.0);
}