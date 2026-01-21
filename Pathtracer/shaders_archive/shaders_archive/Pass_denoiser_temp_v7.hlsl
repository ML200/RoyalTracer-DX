#include "Includes_v7.hlsli"

static const float3 kLUMA = half3(0.2126h, 0.7152h, 0.0722h);
static const float kMaxMsum = 10000.0;

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint2 launch = dispatchThreadId.xy;
    if (any(launch >= uint2(gImageWidth, gImageHeight)))
        return;   // out of bounds guard

    const float2 dimsH = float2(gImageWidth, gImageHeight);

    // "G-buffer"
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

    // bilinear reprojection (4 taps)
    WeightedPixel taps[4];
    GetBilinearReprojectedPixels_d(
        x1Cur, prevView, prevProjection,
        float2(dimsH), objIdCur,
        taps);

    //  Valid bilin pixels
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

        if (dot(n1Cur, nPrev) <= 0.95 ||
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

    // Choose history
    float3 histCol   = Ccur;
    float  Mprev     = 0.0;
    bool   histValid = false;

    if (anyOK && MprevSum > 0.0)
    {
        histCol   = histNum / MprevSum;
        Mprev     = MprevSum;
        histValid = true;
    }

    // Temporal accumulation
    float  Mcur = max(0.0, gScratchPing[uint3(launch, 5)].w);
    float  Mnew = min(Mprev + Mcur, kMaxMsum);

    float  alpha = (Mnew > 0.0) ? (Mcur / Mnew) : 1.0;
    float3 Cacc  = histValid ? lerp(histCol, Ccur, alpha) : Ccur;

    // Store results
    gScratchPing[uint3(launch, 12)] = float4(Cacc, Mnew);
    gScratchPing[uint3(launch,  0)] = float4(Cacc, 0.0);
}