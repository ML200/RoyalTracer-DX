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
#include "Denoiser_helper_v7.hlsli"

#define half  min16float
#define half2 min16float2
#define half3 min16float3
#define half4 min16float4

static const half3 kLUMA = half3(0.2126h, 0.7152h, 0.0722h);

//----------------------------------------------
//  Hash32_To_RGB – returns a vivid RGB in [0,1]
//  • id    : integer you want to visualise
//  • gamma : 1 = linear  (leave it this way if you write to HDR buffer)
//           2.2 = sRGB  (for LDR back-buffer debug)
//----------------------------------------------
float3 Hash32_To_RGB(uint id, float gamma /* = 1.0 */)
{
    // 1. Thomas Wang’s 32-bit mix
    id = (id ^ 61u) ^ (id >> 16);
    id += id << 3;
    id ^= id >> 4;
    id *= 0x27d4eb2du;
    id ^= id >> 15;

    // 2. Take the high 24 bits as R,G,B (8 bits each)
    float3 rgb = float3(
        ( (id >>  0) & 0xFFu ) / 255.0,
        ( (id >>  8) & 0xFFu ) / 255.0,
        ( (id >> 16) & 0xFFu ) / 255.0 );

    // 3. Optional gamma correction so that the colours look
    //    equally bright on an sRGB monitor.
    if (gamma != 1.0)
        rgb = pow(rgb, 1.0 / gamma);

    return rgb;
}

//----------------------------------------------------------------------------
//  Main CS kernel (4-tap upgrade)
//----------------------------------------------------------------------------
[numthreads(16, 16, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    // ── guard ──────────────────────────────────────────────────────────────
    if (any(DTid.xy >= uint2(gImageWidth, gImageHeight))) return;

    uint2  launch = DTid.xy;
    half2  dimsH  = half2(gImageWidth, gImageHeight);


    // DEBUG
    //gOutput[uint3(DTid.xy, 0)] = float4(Hash32_To_RGB((uint)gScratchPing[uint3(launch, 3)].z, 1.0f), 0);
    //return;
    // DEBUG



    // ── current-frame G-buffer pulls ───────────────────────────────────────
    float4 misc3      = gScratchPing[uint3(launch, 3)];
    float  roughness  = misc3.y;
    uint   objID_cur  = (uint)misc3.z;
    uint   emission   = (uint)misc3.x;
    if (emission == 1u) return;                       // emissive = no history

    half3 Ccur        = (half3)gScratchPing[uint3(launch, 1)].rgb;

    float4 pos_M      = gScratchPing[uint3(launch, 5)];
    float3 x1_cur     = pos_M.xyz;
    float  M_di_cur   = pos_M.w;

    float3 n1_cur     = gScratchPing[uint3(launch, 4)].xyz;

    // ── 1. 4-tap reprojection (helper already handles screen bounds) ───────
    WeightedPixel taps[4];
    GetBilinearReprojectedPixels_d(
        x1_cur, prevView, prevProjection,
        float2(dimsH), objID_cur,
        taps);

    // ── 2. Validate each tap & compose weighted history colour ─────────────
    float  sumW    = 0.0f;           // surviving-weight accumulator
    float3 histRGB = 0.0f.xxx;       // weighted history colour
    float  dz_acc  = 0.0f;           // weighted depth-difference (for motion)
    bool   anyOK   = false;

    const float3 camPos = viewI[3].xyz;

    [unroll] for (int i = 0; i < 4; ++i)
    {
        if (taps[i].w == 0.0f || all(taps[i].pix == kInvalidPixel))
            continue;   // helper rejected → skip early

        const int2 pix = taps[i].pix;

        // Fetch per-tap metadata --------------------------------------------------
        float4 g8    = gScratchPing[uint3(pix, 8)];   // emissive(.x) | objID(.z)
        if ((uint)g8.x != 0u)               // previous frame pixel became emissive
        {
            taps[i].w = 0.0f;
            continue;
        }

        const uint  objID_prev = (uint)g8.z;
        if (objID_prev != objID_cur)          // different object → reject
        {
            taps[i].w = 0.0f;
            continue;
        }

        // Geometry pulls ----------------------------------------------------------
        float3 xPrev = gScratchPing[uint3(pix,10)].xyz;
        float3 nPrev = gScratchPing[uint3(pix, 9)].xyz;

        // Normal-congruence & distance-weight rejection (same as before) ----------
        if (dot(n1_cur, nPrev) <= 0.9f ||
            DDistanceWeight(x1_cur, xPrev, n1_cur) <= 0.0f)
        {
            taps[i].w = 0.0f;
            continue;
        }

        // Tap survives –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
        float  dCur  = length(x1_cur - camPos);
        float  dPrev = length(xPrev  - camPos);
        float  dzTap = abs(dCur - dPrev) / max(dCur, dPrev);

        float4 hTap  = gPermanentData[pix];   // stored history colour (α in .a)
        if (hTap.a == 0.0f)                   // history unavailable → reject
        {
            taps[i].w = 0.0f;
            continue;
        }

        // Accumulate weighted contributions --------------------------------------
        histRGB += taps[i].w * hTap.rgb;
        dz_acc  += taps[i].w * dzTap;
        sumW    += taps[i].w;
        anyOK    = true;
    }

    // ── 3. Normalise / decide validity ─────────────────────────────────────
    float3 histCol   = Ccur;     // default: current colour
    float  dz        = 0.0f;
    bool   histValid = false;

    if (anyOK && sumW > 0.0f)
    {
        float invSum = rcp(sumW);
        histCol   = histRGB * invSum;
        dz        = dz_acc  * invSum;
        histValid = true;
    }

    // else: histValid stays false, keeping histCol = Ccur
    float  Nprev = gPermanentData[launch].a;   // 0 … nSamples-1
    float3 Cprev = gPermanentData[launch].rgb; // running mean up to frame-1

    bool   reset = !histValid;                 // disocclusion, emissive, etc.

    if (reset)
    {
        // force first-frame initialise
        Nprev = 0.0;
        Cprev = 0.0.xxx;
    }

    //--------------------------------------------------------------------
    // Incremental arithmetic mean:  Cnew = (Nprev·Cprev + Ccur) / (Nprev+1)
    //--------------------------------------------------------------------
    float Nnew   = min(Nprev + 1.0, 64);   // clamp if desired
    float alpha   = 1.0 / Nnew;                             // ← YOUR α

    half3 Cacc   = lerp( (half3)Cprev, Ccur, (half)alpha );

    gPermanentData[launch]        = float4((float3)Cacc, Nnew);
    gScratchPing   [uint3(launch,0)] = float4((float3)Cacc, 0);   // as before
}