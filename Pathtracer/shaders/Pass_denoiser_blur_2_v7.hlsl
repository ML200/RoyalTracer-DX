// ─────────────────────────────────────────────────────────────────────────────
//  ROOT-CONSTANTS  (slot 1 : two uints = 8 bytes)
//
//  These map 1-to-1 to the `Set*Root32BitConstants` call in Renderer.cpp.
//──────────────────────────────────────────────────────────────────────────────
cbuffer Push : register(b1)
{
    uint2 gImageSize;
}

// Convenience aliases – some of the legacy headers still expect them
#define gImageWidth   (gImageSize.x)
#define gImageHeight  (gImageSize.y)

// ─────────────────────────────────────────────────────────────────────────────
//  Fake the two DXR intrinsics that many helpers rely on.
//  We just substitute the thread-ID and the constant dimensions.
//──────────────────────────────────────────────────────────────────────────────
#define DispatchRaysDimensions() uint3(gImageWidth, gImageHeight, 1)

// DTid is only visible inside `main`, so we stash a copy in a globals so that
// the macro below can see it.  Each thread overwrites its own instance, so no
// synchronisation is needed.
static uint3 gDispatchIdx;
#define DispatchRaysIndex()      gDispatchIdx


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

// ───────────── Utility ────────────────────────────────────────────────
static const float3 LUMA  = float3(0.2126, 0.7152, 0.0722);  // Rec.709
static const float  LARGE = 1e30;

inline float3 ClampFirefly(float3 c, float3 neighMax)
{
    float lumC  = dot(c,        LUMA);
    float lumMx = dot(neighMax, LUMA) * 1.5;        // allow 50 % over-bright
    return (lumC > lumMx) ? c * (lumMx / max(lumC, 1e-4)) : c;
}

//-----------------------------------------------------------------------------
//  5×5 à-trous bilateral blur   (stride = 1, kernel = {1,4,6,4,1}/16)
//-----------------------------------------------------------------------------
[numthreads(32, 8, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint2 launch = dispatchThreadId.xy;
    uint2 dims   = gImageSize;

    if (launch.x >= dims.x || launch.y >= dims.y)
        return;                           // out-of-bounds guard

    const int   STRIDE = 2;               // first à-trous step
    const int   K      = 2;               // ±2 radius → 5 × 5
    static const float kernel[5] = { 1.0/16, 4.0/16, 6.0/16, 4.0/16, 1.0/16 };

    // ── centre sample ------------------------------------------------------
    float3  c0   = gScratchPing[launch].xyz;

    uint  pIdx0  = MapPixelID(dims, launch);
    float3 x10   = load_x1(g_sample_current, pIdx0);
    float3 n10   = load_n1(g_sample_current, pIdx0);

    float3 camPos = mul(viewI, float4(0,0,0,1)).xyz;
    float  d0     = length(x10 - camPos);

    float3 sum  = 0.0;
    float  wSum = 0.0;

    // ── 5×5 separable stencil --------------------------------------------
    [unroll]
    for (int dy = -K; dy <= K; ++dy)
    {
        [unroll]
        for (int dx = -K; dx <= K; ++dx)
        {
            int2 coord = int2(launch) + int2(dx, dy) * STRIDE;

            // bounds check
            if (coord.x < 0 || coord.y < 0 ||
                coord.x >= int(dims.x) || coord.y >= int(dims.y))
                continue;

            float3 c    = gScratchPing[coord].xyz;
            uint   pIdx = MapPixelID(dims, coord);
            float3 x1   = load_x1(g_sample_current, pIdx);
            float3 n1   = load_n1(g_sample_current, pIdx);

            float d1   = length(x1 - camPos);
            float dz   = abs(d0 - d1) / max(d0, d1);    // relative depth diff
            float nDot = dot(n10, n1);

            // depth / normal rejection
            if (dz >= 0.2f || nDot <= 0.999f)
                continue;

            // weights ------------------------------------------------------
            float kW = kernel[dx + K] * kernel[dy + K];
            float wN = saturate(nDot * 16.0);
            float wZ = saturate(exp(-dz * 20.0));
            float wC = exp(-dot(c - c0, c - c0) * 4.0);

            float lum0 = dot(c0, LUMA);
            float lumN = dot(c , LUMA);
            float wF = saturate(lum0 * 4.0 / max(lumN, 1e-4)); // anti-firefly

            float w  = kW * wN * wZ * wC * wF;
            sum  += w * c;
            wSum += w;
        }
    }

    float3 Cout = sum / max(wSum, 1e-4);

    // ── 3×3 neighbourhood clamp (ringing / fireflies) ---------------------
    float3 neighMin = float3( LARGE,  LARGE,  LARGE);
    float3 neighMax = float3(-LARGE, -LARGE, -LARGE);

    for (int ny = -1; ny <= 1; ++ny)
        for (int nx = -1; nx <= 1; ++nx)
        {
            if (nx == 0 && ny == 0) continue;
            int2 pc = int2(launch) + int2(nx, ny);
            if (pc.x < 0 || pc.y < 0 || pc.x >= int(dims.x) || pc.y >= int(dims.y))
                continue;
            float3 cn = gScratchPing[pc].xyz;
            neighMin  = min(neighMin, cn);
            neighMax  = max(neighMax, cn);
        }

    Cout = clamp       (Cout, neighMin, neighMax);
    Cout = ClampFirefly(Cout, neighMax);

    gOutput[uint3(launch, 0)] = float4(Cout, 1.0);
}