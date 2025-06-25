// ────────────────────────────────────────────────────────────────────────────
//  Denoiser - temporal accumulation v12   •   ***COMPUTE VERSION***
// ────────────────────────────────────────────────────────────────────────────

//---------------------------------------------------------------------------
//  ROOT-CONSTANTS  (slot 1 : two uints = 8 bytes)
//---------------------------------------------------------------------------
cbuffer Push : register(b1)
{
    uint2 gImageSize;                    // ↳ filled from SetComputeRoot32BitConstants
}

#define gImageWidth   (gImageSize.x)
#define gImageHeight  (gImageSize.y)

//---------------------------------------------------------------------------
//  Fake DXR intrinsics many headers rely on
//---------------------------------------------------------------------------
#define DispatchRaysDimensions() uint3(gImageWidth, gImageHeight, 1)

// We will copy SV_DispatchThreadID into this global so that the legacy
// macros work unchanged.
groupshared uint3 gDispatchIdxShared[1];    // dummy shared so every thread gets its own storage
static     uint3  gDispatchIdx;
#define DispatchRaysIndex()      gDispatchIdx

// ────────────────────────────────────────────────────────────────────────────
//  All your regular includes – unchanged
// ────────────────────────────────────────────────────────────────────────────
#include "Constants_v7.hlsli"
#include "Common_v7.hlsli"
#include "Structures_misc.hlsli"
#include "Random_v7.hlsli"
#include "Compression_v7.hlsli"

RWTexture2DArray<float4> gOutput              : register(u0);
RWTexture2D<float4>      gPermanentData       : register(u1);
RWTexture2D<float4>      gScratchPing         : register(u8);   // Storage for denoiser

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

//---------------------------------------------------------------------------
//  Compute kernel
//  (32×8 threads -- matches the launch grid you used in C++)
//---------------------------------------------------------------------------
[numthreads(32, 8, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    // ---- emulate DispatchRaysIndex ---------------------------------------
    gDispatchIdx = DTid;

    // ---- guard against over-dispatch (optional) --------------------------
    if (DTid.x >= gImageWidth || DTid.y >= gImageHeight) return;

    uint2  launch = DTid.xy;
    uint2  dimsI  = DispatchRaysDimensions().xy;
    float2 dims   = float2(dimsI);
    uint   pIdx   = MapPixelID(dims, launch);

    float3  Ccur   = gOutput[uint3(launch, 1)].rgb;
    float3  x1_cur = load_x1(g_sample_current, pIdx);
    float3  n1_cur = load_n1(g_sample_current, pIdx);

    // ---- reprojection -----------------------------------------------------
    float4 prevClip = mul(prevProjection, mul(prevView, float4(x1_cur, 1.0)));
    if (prevClip.w <= 1e-4f)
    {
        gPermanentData[launch] = float4(Ccur, 0);
        gScratchPing  [launch] = float4(Ccur, 0);
        return;
    }

    float2 prevNDC = prevClip.xy / prevClip.w;
    float2 prevSS  = prevNDC * float2(0.5, -0.5) + 0.5;
    float2 reprojF = prevSS * dims;

    bool reprojOK = all(reprojF >= 0) && reprojF.x < dims.x && reprojF.y < dims.y;

    float  dz        = 0.0;
    float4 hist4     = float4(Ccur, 0);
    bool   histValid = false;

    if (reprojOK)
    {
        // --- bilinear fetch from history ----------------------------------
        float2 rc   = clamp(reprojF, 0, dims - 1.001);
        int2  i00   = int2(rc);
        float2 frac = rc - float2(i00);
        int2  i10   = min(i00 + int2(1,0), int2(dims)-1);
        int2  i01   = min(i00 + int2(0,1), int2(dims)-1);
        int2  i11   = min(i00 + int2(1,1), int2(dims)-1);

        float w00 = (1 - frac.x) * (1 - frac.y);
        float w10 =      frac.x  * (1 - frac.y);
        float w01 = (1 - frac.x) *      frac.y;
        float w11 =      frac.x  *      frac.y;

        hist4 = gPermanentData[i00]*w00 + gPermanentData[i10]*w10 +
                gPermanentData[i01]*w01 + gPermanentData[i11]*w11;

        // --- depth/normal validation --------------------------------------
        int2 nearestPx = clamp(int2(reprojF + 0.5), 0, int2(dims)-1);
        float3 x1_prev = load_x1(g_sample_current, MapPixelID(dims, nearestPx));
        float3 n1_prev = load_n1(g_sample_current, MapPixelID(dims, nearestPx));

        float3 camPos = mul(viewI, float4(0,0,0,1)).xyz;
        float  d0     = length(x1_cur - camPos);
        float  d1     = length(x1_prev - camPos);
        dz            = abs(d0 - d1) / max(d0, d1);
        float  nDot   = dot(n1_cur, n1_prev);

        histValid = (dz < 0.025f) && (nDot > 0.9f);
        if (!histValid) hist4 = float4(Ccur, 0);
    }

    // ---- neighbourhood clamp ---------------------------------------------
    float3 cMin = Ccur, cMax = Ccur;
    [unroll] for(int ny = -1; ny <= 1; ++ny)
    [unroll] for(int nx = -1; nx <= 1; ++nx)
    {
        int2 p = clamp(int2(launch) + int2(nx,ny), 0, int2(dims)-1);
        float3 cN = gOutput[uint3(p,1)].rgb;
        cMin = min(cMin, cN);
        cMax = max(cMax, cN);
    }
    cMin -= 0.01 * cMax;
    cMax += 0.01 * cMax;
    Ccur  = clamp(Ccur, cMin, cMax);

    // ---- confidence -------------------------------------------------------
    Reservoir_DI rdi = loadReservoirDI(g_Reservoirs_current_di, pIdx);
    float conf      = saturate(rdi.M_di / 30.0);   // 0-1
    float alphaBase = lerp(0.25, 0.03, conf);

    // ---- colour / motion / reactive gates ---------------------------------
    float3 diffRGB = abs(Ccur - hist4.rgb);
    float  err     = max(max(diffRGB.r, diffRGB.g), diffRGB.b);
    float  errFac  = (histValid) ? saturate((err - 0.06) * 5.0) : 0.0;
    errFac        *= (1.0 - conf);

    float mvFac = 0.0;
    float reactiveDepth = 0.0;
    if (histValid)
    {
        float2 curSS = (float2(launch) + 0.5) / dims;
        float2 velSS = curSS - prevSS;
        float  velPx = length(velSS * dims);
        mvFac        = saturate((velPx - 0.75) / 1.0);

        reactiveDepth = saturate((dz - 0.03) * 35.0);
    }

    float alpha = alphaBase;
    alpha = lerp(alpha, 1.0, errFac);
    alpha = lerp(alpha, 1.0, mvFac);
    alpha = lerp(alpha, 1.0, reactiveDepth);

    // ---- blend & store ----------------------------------------------------
    float3 Cacc   = lerp(hist4.rgb, Ccur, alpha);
    float  frames = clamp(hist4.a + 1.0, 1.0, 64.0);

    gPermanentData[launch] = float4(Cacc, frames);
    gScratchPing  [launch] = float4(Cacc, 0.0);
}