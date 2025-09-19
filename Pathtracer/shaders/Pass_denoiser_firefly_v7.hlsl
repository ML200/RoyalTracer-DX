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

static const float3 kLUMA = float3(0.2126, 0.7152, 0.0722);
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

    // Feed forward emitters
    if (gScratchPing[uint3(launch, 3)].x == 1) {
        gScratchPing[uint3(launch, 1)] = gScratchPing[uint3(launch, 2)];
        return;
    }

    // Scale threshold with sample quality (w)
    float w        = gScratchPing[uint3(launch, 5)].w;
    float threshold = lerp(1.0f, uThreshold, saturate(w / 5.0f)); // keeps your behavior

    // Center signals
    float3 centerRGB  = gScratchPing[uint3(launch, 0)].rgb;
    float3 centerN    = normalize(gScratchPing[uint3(launch, 4)].xyz);
    float3 centerPos  = gScratchPing[uint3(launch, 5)].xyz;
    float  rough      = gScratchPing[uint3(launch, 3)].y;
    uint   centerObj  = asuint(gScratchPing[uint3(launch, 3)].z); // objID packed in .z

    // Geometry-aware gates (tweak if needed)
    // Slightly relax normal gate on rough surfaces
    float cosThresh   = lerp(0.92f, 0.85f, saturate(rough));   // smooth → stricter, rough → looser
    float distThresh  = 0.05f;                                 // world-space planar distance (same as your GI reuse)

    // Accumulate only geometry-consistent neighbors
    float3 neighbourSum = 0.0;
    uint   neighbourCnt = 0;

    [unroll]
    for (int dy = -1; dy <= 1; ++dy)
    {
        [unroll]
        for (int dx = -1; dx <= 1; ++dx)
        {
            if (dx == 0 && dy == 0) continue;

            int2 n = int2(launch) + int2(dx, dy);
            if (n.x < 0 || n.y < 0 || n.x >= int(gImageWidth) || n.y >= int(gImageHeight))
                continue;

            // Neighbor geometry
            float3 nRGB   = gScratchPing[uint3(n, 0)].rgb;
            float3 nN     = normalize(gScratchPing[uint3(n, 4)].xyz);
            float3 nPos   = gScratchPing[uint3(n, 5)].xyz;
            uint   nObj   = asuint(gScratchPing[uint3(n, 3)].z);

            // Geometry consistency checks
            bool sameObj     = (nObj == centerObj);
            bool normalOK    = !RejectNormal_GI(centerN, nN, cosThresh);
            bool distanceOK  = !RejectDistance_GI(centerPos, nPos, centerN, distThresh);

            if (sameObj && normalOK && distanceOK)
            {
                neighbourSum += nRGB;
                ++neighbourCnt;
            }
        }
    }

    // If no valid neighbors, do nothing (prevents bleeding across edges/thin features)
    if (neighbourCnt > 0)
    {
        float3 neighbourAvg = neighbourSum / float(neighbourCnt);

        // Luma-based clamp against *geometry-consistent* neighborhood
        float centerLum    = Luma(centerRGB);
        float neighbourLum = Luma(neighbourAvg);

        if (centerLum > neighbourLum * (1.0 + threshold))
        {
            // clamp down to the neighborhood mean; optionally lerp instead of hard replace:
            // centerRGB = lerp(centerRGB, neighbourAvg, 0.75);
            centerRGB = neighbourAvg;
        }
    }

    gScratchPing[uint3(launch, 1)] = float4(centerRGB, 1.0);
}
