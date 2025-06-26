cbuffer Push : register(b1) { uint2 gImageSize; }
#define gImageWidth   (gImageSize.x)
#define gImageHeight  (gImageSize.y)
#define DispatchRaysDimensions() uint3(gImageWidth, gImageHeight, 1)

groupshared uint3 gDispatchIdxShared[1];
static     uint3  gDispatchIdx;
#define DispatchRaysIndex() gDispatchIdx

#include "Constants_v7.hlsli"
#include "Common_v7.hlsli"
#include "Structures_misc.hlsli"
#include "Random_v7.hlsli"
#include "Compression_v7.hlsli"

RWTexture2DArray<float4> gOutput              : register(u0);
RWTexture2D<float4>      gPermanentData       : register(u1);
RWTexture2D<float4>      gScratchPing         : register(u8);

RWByteAddressBuffer      g_sample_current         : register(u6);
RWByteAddressBuffer      g_sample_last            : register(u7);
RWByteAddressBuffer      g_Reservoirs_current_di  : register(u2);
RWByteAddressBuffer      g_Reservoirs_last_di     : register(u3);
RWByteAddressBuffer      g_Reservoirs_current_gi  : register(u4);
RWByteAddressBuffer      g_Reservoirs_last_gi     : register(u5);

StructuredBuffer<STriVertex>  BTriVertex           : register(t2);
StructuredBuffer<int>         indices              : register(t1);
RaytracingAccelerationStructure SceneBVH           : register(t0);
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

// --------------------- Filter Parameters ------------------------------------
#define STEP_WIDTH            2
#define C_PHI                 0.6f
#define P_PHI                 3.0f
#define N_POWER               32.0f
#define EDGE_COLOR_CUTOFF     2.0f
#define EDGE_DEPTH_CUTOFF     2.0f
#define EDGE_NORMAL_MIN       0.5f
#define SKYBOX_MATID  4294967294u

static const float  KERNEL[9] = { 0.0625, 0.25, 0.25, 0.25, 0.375, 0.25, 0.25, 0.25, 0.0625 };
static const int2   OFFS[9]   = { int2(-2,0), int2(-1,0), int2(0,-2), int2(0,-1), int2(0,0),
                                  int2(0,1), int2(0,2),  int2(1,0),  int2(2,0) };
static const float3 LUMA = float3(0.2126, 0.7152, 0.0722);

struct SurfSample { float3 colour; float3 normal; float depth; uint matID; };

inline SurfSample GetSurf(uint2 pix, uint imgW)
{
    SurfSample s; uint idx = MapPixelID(uint2(imgW, gImageHeight), pix);
    s.colour = gScratchPing[pix].xyz;
    float3 pos = load_x1(g_sample_current, idx);
    s.depth  = length(pos - mul(viewI, float4(0,0,0,1)).xyz);
    s.normal = load_n1(g_sample_current, idx);
    s.matID  = load_matID(g_sample_current, idx);
    return s;
}

[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint2 launch = tid.xy;
    if (launch.x >= gImageWidth || launch.y >= gImageHeight) return;

    SurfSample centre = GetSurf(launch, gImageWidth);

    // If the centre is a skybox miss, just copy through.
    if (centre.matID == SKYBOX_MATID)
    {
        gOutput[uint3(launch, 0)] = float4(centre.colour, 1.0);
        return;
    }

    float sigma_c = C_PHI * (float)STEP_WIDTH;
    float sigma_z = P_PHI * (float)STEP_WIDTH;

    float3 accum = 0.0; float wSum = 0.0;

    [unroll]
    for (int i = 0; i < 9; ++i)
    {
        int2 coord = int2(launch) + OFFS[i] * STEP_WIDTH;
        if (coord.x < 0 || coord.y < 0 || coord.x >= int(gImageWidth) || coord.y >= int(gImageHeight))
            continue;

        SurfSample s = GetSurf(uint2(coord), gImageWidth);

        // Skip if material IDs differ (prevents cross‑edge bleeding)
        if (s.matID != centre.matID) continue;

        float3 dc = centre.colour - s.colour;
        float  colDist2 = dot(dc, dc);
        float  colCut2 = (EDGE_COLOR_CUTOFF * sigma_c) * (EDGE_COLOR_CUTOFF * sigma_c);
        if (colDist2 > colCut2) continue;

        float nDot = saturate(dot(centre.normal, s.normal));
        if (nDot < EDGE_NORMAL_MIN) continue;

        float dz = abs(centre.depth - s.depth);
        if (dz > EDGE_DEPTH_CUTOFF * sigma_z) continue;

        float c_w = exp(-colDist2 / (sigma_c * sigma_c));
        float n_w = pow(nDot, N_POWER);
        float p_w = exp(-dz / sigma_z);
        float w = c_w * n_w * p_w * KERNEL[i];

        accum += s.colour * w; wSum += w;
    }

    const float centreBias = KERNEL[4];
    accum += centre.colour * centreBias; wSum += centreBias;

    float3 outColour = accum / max(wSum, 1e-6);

    float lumC = dot(centre.colour, LUMA);
    float lumO = dot(outColour,     LUMA);
    if (lumO > lumC * 4.0) outColour *= (lumC * 4.0) / lumO;

    gOutput[uint3(launch, 0)] = float4(outColour, 1.0);
}
