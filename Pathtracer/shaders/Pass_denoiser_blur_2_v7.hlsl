// Pass_denoiser_blur_2_v7_firefly.hlsl
// Spatial à‑trous blur + strong firefly suppression
// -----------------------------------------------------------------------------
//  * Re-weights bright outliers during the gather phase (neighbour fireflies)
//  * Clamps the final colour against the local neighbourhood (central fireflies)
// -----------------------------------------------------------------------------

#include "Constants_v7.hlsli"
#include "Common_v7.hlsli"
#include "Structures_misc.hlsli"
#include "Random_v7.hlsli"
#include "Compression_v7.hlsli"

RWTexture2DArray<float4> gOutput          : register(u0);
RWTexture2D<float4>      gPermanentData   : register(u1);
RWTexture2D<float4>      gScratchPing     : register(u8); // Storage for denoiser

RWByteAddressBuffer g_sample_current       : register(u6);
RWByteAddressBuffer g_sample_last          : register(u7);
RWByteAddressBuffer g_Reservoirs_current_di : register(u2);
RWByteAddressBuffer g_Reservoirs_last_di    : register(u3);
RWByteAddressBuffer g_Reservoirs_current_gi : register(u4);
RWByteAddressBuffer g_Reservoirs_last_gi    : register(u5);

StructuredBuffer<STriVertex> BTriVertex          : register(t2);
StructuredBuffer<int>        indices            : register(t1);
RaytracingAccelerationStructure SceneBVH         : register(t0);
StructuredBuffer<InstanceProperties> instanceProps : register(t3);
StructuredBuffer<uint>        materialIDs       : register(t4);
StructuredBuffer<Material>    materials         : register(t5);
StructuredBuffer<LightTriangle> g_EmissiveTriangles : register(t6);
StructuredBuffer<float>        g_AliasProb      : register(t7);
StructuredBuffer<uint>         g_AliasIdx       : register(t8);

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

// ───────────── Utility ─────────────────────────────────────────────────────────
static const float3 LUMA  = float3(0.2126, 0.7152, 0.0722); // Rec.709 luminance
static const float  LARGE = 1e30;                            // big number (no FLT_MAX in HLSL)

inline float3 ClampFirefly(const float3 c, const float3 neighMax)
{
    // Allow the result to exceed the brightest neighbour by at most 50 %
    float lumC  = dot(c,        LUMA);
    float lumMx = dot(neighMax, LUMA) * 1.3;
    return (lumC > lumMx) ? c * (lumMx / max(lumC, 1e-4)) : c;
}

// ─── Spatial à‑trous filter with firefly suppression ──────────────────────────
[shader("raygeneration")]
void Pass_denoiser_blur_2_v7()
{
    uint2  launch = DispatchRaysIndex().xy;
    float2 dims   = DispatchRaysDimensions().xy;

    // ── Constants -----------------------------------------------------------
    const int   STRIDE = 4;   // first à‑trous step (stride 1)
    const int   K      = 2;   // 3×3 kernel radius (5‑tap separable)
    static const float kernel[5] = { 1.0/16, 4.0/16, 6.0/16, 4.0/16, 1.0/16 };

    // ── Central sample ------------------------------------------------------
    float3     c0 = gScratchPing[launch].xyz;
    uint pIdx0 = MapPixelID(dims, launch);
    float3 x10 = load_x1(g_sample_current, pIdx0);
    float3 n10 = load_n1(g_sample_current, pIdx0);

    // Camera position in world space (needed for linear‑depth test)
    float3 camPos = mul(viewI, float4(0, 0, 0, 1)).xyz;
    float  d0     = length(x10 - camPos);

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

        // neighbour sample
        float3     c = gScratchPing[coord].xyz;
        uint pIdx = MapPixelID(dims, coord);
        float3 x1 = load_x1(g_sample_current, pIdx);
        float3 n1 = load_n1(g_sample_current, pIdx);

        // ── Hit‑valid check (depth + normal) --------------------------------
        float d1   = length(x1 - camPos);
        float dz   = abs(d0 - d1) / max(d0, d1);   // relative depth diff
        float nDot = dot(n10, n1);
        bool  hitValid = (dz < 0.2f) && (nDot > 0.999f);
        if (!hitValid)
            continue;   // completely reject this neighbour

        // ── Kernel + bilateral weights -------------------------------------
        float kW = kernel[dx + K] * kernel[dy + K];     // separable 5×5 kernel

        float wN = saturate(nDot * 16.0);               // normal term
        float wZ = saturate(exp(-dz * 20.0));           // depth term
        float wC = exp(-dot(c - c0, c - c0) * 4.0);     // colour / variance

        // ── Firefly weight (penalise very bright neighbours) ---------------
        float lum0 = dot(c0, LUMA);
        float lumN = dot(c,  LUMA);
        // If neighbour is >4× brighter than centre, its weight drops to 0
        float wF = saturate(lum0 * 4.0 / max(lumN, 1e-4));

        float w  = kW * wN * wZ * wC * wF;
        sum  += w * c;
        wSum += w;
    }

    // ── Normalise ----------------------------------------------------------
    float3 Cout = sum / max(wSum, 1e-4);

    // ── 3×3 neighbourhood clamp to kill edge ringing & central fireflies ---
    float3 neighMin = float3( LARGE,  LARGE,  LARGE);
    float3 neighMax = float3(-LARGE, -LARGE, -LARGE);
    for (int ny = -1; ny <= 1; ++ny)
        for (int nx = -1; nx <= 1; ++nx)
        {
            if (nx == 0 && ny == 0) continue; // EXCLUDE centre for firefly clamp
            int2 pc = int2(launch) + int2(nx, ny);
            if (pc.x < 0 || pc.y < 0 || pc.x >= int(dims.x) || pc.y >= int(dims.y))
                continue;
            float3 cn = gScratchPing[pc].xyz;
            neighMin = min(neighMin, cn);
            neighMax = max(neighMax, cn);
        }
    Cout = clamp(Cout, neighMin, neighMax);            // ringing clamp
    Cout = ClampFirefly(Cout, neighMax);               // central‑firefly clamp

    gOutput[uint3(launch, 0)] = float4(Cout, 1.0);
}
