#include "Constants_v7.hlsli"
#include "Common_v7.hlsli"
#include "Structures_misc.hlsli"
#include "Random_v7.hlsli"
#include "Compression_v7.hlsli"

RWTexture2DArray<float4> gOutput : register(u0);
RWTexture2D<float4> gPermanentData : register(u1);
RWTexture2D<float4> gScratchPing : register(u8); // Storage for denoiser

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
#include "Motion_vectors_v7.hlsli"

// ─── Spatial à-trous filter ──────────────────────────────────────────────────────
[shader("raygeneration")]
void Pass_denoiser_blur_2_v7()
{
    uint2  launch = DispatchRaysIndex().xy;
    float2 dims   = DispatchRaysDimensions().xy;

    // ── Constants -----------------------------------------------------------
    const int   STRIDE = 2;   // first à‑trous step (stride 1)
    const int   K      = 1;   // 5×5 kernel radius
    static const float kernel[5] = { 1.0/16, 4.0/16, 6.0/16, 4.0/16, 1.0/16 };

    // ── Central sample ------------------------------------------------------
    float3     c0 = gScratchPing[launch].xyz;
    SampleData s0 = loadSampleData(g_sample_current, MapPixelID(dims, launch));

    // Camera position in world space (needed for linear‑depth test)
    float3 camPos = mul(viewI, float4(0, 0, 0, 1)).xyz;
    float  d0     = length(s0.x1 - camPos);

    float3 sum  = 0;
    float  wSum = 0;

    // ── 5 × 5 separable stencil -------------------------------------------
    [unroll]
    for (int dy = -K; dy <= K; ++dy)
    [unroll]
    for (int dx = -K; dx <= K; ++dx)
    {
        int2 coord = int2(launch) + int2(dx, dy) * STRIDE;

        // bounds check
        if (coord.x < 0 || coord.y < 0 ||
            coord.x >= int(dims.x) || coord.y >= int(dims.y))
            continue;

        // central pixel already handled; include it normally
        // (skip explicit test so kW scales it like neighbours)

        // neighbour sample
        float3     c = gScratchPing[coord].xyz;
        SampleData s = loadSampleData(g_sample_current, MapPixelID(dims, coord));

        // ── Hit‑valid check (depth + normal) --------------------------------
        float d1   = length(s.x1 - camPos);
        float dz   = abs(d0 - d1) / max(d0, d1);   // relative depth diff
        float nDot = dot(s0.n1, s.n1);
        bool  hitValid = (dz < 0.2f) && (nDot > 0.999f);
        if (!hitValid)
            continue;   // completely reject this neighbour

        // ── Kernel + bilateral weights -------------------------------------
        float kW = kernel[dx + K] * kernel[dy + K];     // separable 5×5 kernel

        float wN = saturate(nDot * 16.0);               // normal term  (32→16)
        float wZ = saturate(exp(-dz * 20.0));           // depth term   (similar to 80→20)
        float wC = exp(-dot(c - c0, c - c0) * 4.0);     // colour / variance

        float w  = kW * wN * wZ * wC;
        sum  += w * c;
        wSum += w;
    }

    // ── Normalise & write ---------------------------------------------------
    float3 Cout = sum / max(wSum, 1e-4);

    // ── 3×3 neighbourhood clamp to kill edge ringing ----------------------
    float3 cMin = c0, cMax = c0;
    for (int ny = -1; ny <= 1; ++ny)
        for (int nx = -1; nx <= 1; ++nx)
        {
            int2 pc = int2(launch) + int2(nx, ny);
            if (pc.x < 0 || pc.y < 0 || pc.x >= int(dims.x) || pc.y >= int(dims.y))
                continue;
            float3 cn = gScratchPing[pc].xyz;
            cMin = min(cMin, cn);
            cMax = max(cMax, cn);
        }
    Cout = clamp(Cout, cMin, cMax);
    gOutput[uint3(launch, 0)] = float4(Cout, 1.0);
}
