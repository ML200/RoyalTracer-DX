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
#include "Reservoir_DI_v7.hlsli"
#include "Reservoir_GI_v7.hlsli"
#include "Inline_RT.hlsli"
#include "Camera_ray_v7.hlsli"
#include "MIS_v7.hlsli"
#include "NEE_Sampling_v7.hlsli"
#include "BSDF_Sampling_v7.hlsli"
#include "Motion_vectors_v7.hlsli"
#include "Denoiser_helper_v7.hlsli"

static const float3 kLUMA = half3(0.2126h, 0.7152h, 0.0722h);
static const float kMaxMsum = 10000.0;

//------------------------------------------------------------------------------
//  Main kernel
//------------------------------------------------------------------------------
[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint2 launch = dispatchThreadId.xy;
    if (any(launch >= uint2(gImageWidth, gImageHeight)))
        return;   // out of bounds guard

    const float2 dimsH = float2(gImageWidth, gImageHeight);

    //----------------------------------------------------------------------
    //  G-buffer
    //----------------------------------------------------------------------
    float4 misc3     = gScratchPing[uint3(launch, 3)];
    float  roughness = misc3.y;
    uint   objIdCur  = (uint)misc3.z;
    uint   isEmissive= (uint)misc3.x;
    if (isEmissive)
        return;

    float3  Ccur      = gScratchPing[uint3(launch, 1)].rgb;
    float3  acurr      = gScratchPing[uint3(launch, 2)].rgb; // albedo

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
    //  2. Validate taps
    //----------------------------------------------------------------------
    float3 histNum   = 0.0;
    float  MprevSum  = 0.0;
    bool   anyOK     = false;

    [unroll]
    for (int i = 0; i < 4; ++i)
    {
        if (taps[i].w == 0.0 || all(taps[i].pix == kInvalidPixel))
            continue;

        int2  pix = taps[i].pix;
        float4 g8 = gScratchPing[uint3(pix, 8)];

        if (g8.x != 0.0 || (uint)g8.z != objIdCur)
            continue;

        float3 xPrev = gScratchPing[uint3(pix,10)].xyz;
        float3 nPrev = gScratchPing[uint3(pix, 9)].xyz;
        float3 aprev = gScratchPing[uint3(pix, 7)].rgb;

        if (dot(n1Cur, nPrev) <= 0.999 ||
            DDistanceWeight(x1Cur, xPrev, n1Cur) <= 0.05 ||
            (aprev.x != acurr.x || aprev.y != acurr.y || aprev.z != acurr.z))
            continue;

        float4 hTap = gPermanentData[pix];
        bool any_nan = isnan(hTap.r) || isnan(hTap.g) || isnan(hTap.b) || isnan(hTap.a);

        float tapM = max(0.0, hTap.a);
        if (any_nan || tapM <= 0.0)
            continue;

        histNum  += taps[i].w * (tapM * hTap.rgb);
        MprevSum += taps[i].w * tapM;
        anyOK     = true;
    }

    //----------------------------------------------------------------------
    //  3. Choose history
    //----------------------------------------------------------------------
    float3 histCol   = Ccur;
    float  Mprev     = 0.0;
    bool   histValid = false;

    if (anyOK && MprevSum > 0.0)
    {
        histCol   = histNum / MprevSum;
        Mprev     = MprevSum;
        histValid = true;
    }

    //----------------------------------------------------------------------
    //  4. Temporal accumulation
    //----------------------------------------------------------------------
    float  Mcur = max(0.0, gScratchPing[uint3(launch, 5)].w);
    float  Mnew = min(Mprev + Mcur, kMaxMsum);

    float  alpha = (Mnew > 0.0) ? (Mcur / Mnew) : 1.0;
    float3 Cacc  = histValid ? lerp(histCol, Ccur, alpha) : Ccur;

    //----------------------------------------------------------------------
    //  5. Store results
    //----------------------------------------------------------------------
    gScratchPing[uint3(launch, 12)] = float4(Cacc, Mnew);
    gScratchPing[uint3(launch,  0)] = float4(Cacc, 0.0);
}