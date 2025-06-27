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

static const float3 kLUMA = float3(0.2126, 0.7152, 0.0722);

// Utility
uint2     MapPixelXY(float2 dims, uint id)
{
    uint y = id / uint(dims.x);
    uint x = id - y * uint(dims.x);
    return uint2(x, y);
}


[numthreads(16, 16, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    gDispatchIdx = DTid;
    if (DTid.x >= gImageWidth || DTid.y >= gImageHeight) return;

    uint2  launch = DTid.xy;
    uint2  dimsI  = DispatchRaysDimensions().xy;
    float2 dims   = float2(dimsI);
    uint   pIdx   = MapPixelID(dims, launch);

    float3  Ccur   = gOutput[uint3(launch, 1)].rgb;
    float3  x1_cur = load_x1(g_sample_current, pIdx);
    float3  n1_cur = load_n1(g_sample_current, pIdx);
    float3  objID_cur = load_objID(g_sample_current, pIdx);

    float2 reprojF = GetLastFramePixelCoordinates_Float(x1_cur, prevView, prevProjection, dims, objID_cur);

    bool reprojOK = all(reprojF >= 0) && reprojF.x < dims.x && reprojF.y < dims.y;

    float  dz        = 0.0;
    float4 hist4     = float4(Ccur, 0);
    bool   histValid = false;

    if (reprojOK)
    {
        float2 rc   = clamp(reprojF, 0, dims - 1.001);
        int2  i00   = int2(rc);
        float2 frac = rc - float2(i00);
        int2  i10   = min(i00 + int2(1,0), int2(dims)-1);
        int2  i01   = min(i00 + int2(0,1), int2(dims)-1);
        int2  i11   = min(i00 + int2(1,1), int2(dims)-1);

        float w[4]  = {
            (1 - frac.x) * (1 - frac.y),
                  frac.x  * (1 - frac.y),
            (1 - frac.x) *      frac.y ,
                  frac.x  *      frac.y
        };
        int2  taps[4] = { i00, i10, i01, i11 };

        float v[4]  = { 0,0,0,0 };
        float wSum  = 0.0;

        float3 camPos = mul(viewI, float4(0,0,0,1)).xyz;
        float  dCur   = length(x1_cur - camPos);

        [unroll]
        for (int k = 0; k < 4; ++k)
        {
            if (w[k] == 0.0) continue;

            uint  pidLast  = MapPixelID(dims, taps[k]);
            float3 xPrev   = load_x1   (g_sample_last, pidLast);
            float3 nPrev   = load_n1   (g_sample_last, pidLast);
            uint3  idPrev  = asuint(load_objID(g_sample_last, pidLast) + 0.5);

            float  dPrev   = length(xPrev - camPos);
            float  dzTap   = abs(dCur - dPrev) / max(dCur, dPrev);
            float  nDotTap = dot(n1_cur, nPrev);

            bool   idOK    = all(idPrev == asuint(objID_cur + 0.5));

            bool ok = (dzTap  < 0.025f) &&
                      (nDotTap > 0.90f) &&
                       idOK;

            if (ok)
            {
                v[k]   = w[k];
                wSum  += w[k];
            }
        }

        if (wSum > 0.0)
        {
            float invSum = rcp(wSum);
            hist4 =   gPermanentData[i00] * v[0] +
                      gPermanentData[i10] * v[1] +
                      gPermanentData[i01] * v[2] +
                      gPermanentData[i11] * v[3];
            hist4 *= invSum;

            histValid = true;
        }
        else
        {
            hist4     = float4(Ccur, 0);
            histValid = false;
        }
    }

    // neighbourhood clamp
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

    // confidence adjusted alpha -> the better the sampler, the stronger the stability
    Reservoir_DI rdi = loadReservoirDI(g_Reservoirs_current_di, pIdx);
    float conf      = saturate(rdi.M_di / 30.0);   // 0-1
    float alphaBase = lerp(0.25, 0.03, conf);

    // colour / motion / reactive gates
    float  lumCur   = dot(Ccur      , kLUMA);
    float  lumHist  = dot(hist4.rgb , kLUMA);
    float  err      = abs(lumCur - lumHist);
    float  errFac  = (histValid) ? saturate((err - 0.06) * 5.0) : 0.0;
    errFac        *= (1.0 - conf);

    float mvFac = 0.0;
    float reactiveDepth = 0.0;
    if (histValid)
    {
        float2 curSS = (float2(launch) + 0.5) / dims;
        float2 velSS = curSS - (reprojF + 0.5) / dims;
        float  velPx = length(velSS * dims);
        mvFac        = saturate((velPx - 1.0) / 4.0);

        reactiveDepth = saturate((dz - 0.03) * 35.0);
    }

    float alpha = alphaBase;
    alpha = lerp(alpha, 1.0, errFac);
    alpha = lerp(alpha, 1.0, mvFac);
    alpha = lerp(alpha, 1.0, reactiveDepth);

    float3 Cacc   = lerp(hist4.rgb, Ccur, alpha);
    float  frames = clamp(hist4.a + 1.0, 1.0, 64.0);

    gPermanentData[launch] = float4(Cacc, frames);
    gScratchPing  [launch] = float4(Cacc, 0.0);
}