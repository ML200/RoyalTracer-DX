#include "Includes_v8.hlsli"

inline float3 sRGBGammaCorrection(float3 color)
{
    float3 result;

    // Red channel
    if (color.r <= 0.0031308f)
        result.r = 12.92f * color.r;
    else
        result.r = 1.055f * pow(color.r, 1.0f / 2.4f) - 0.055f;

    // Green channel
    if (color.g <= 0.0031308f)
        result.g = 12.92f * color.g;
    else
        result.g = 1.055f * pow(color.g, 1.0f / 2.4f) - 0.055f;

    // Blue channel
    if (color.b <= 0.0031308f)
        result.b = 12.92f * color.b;
    else
        result.b = 1.055f * pow(color.b, 1.0f / 2.4f) - 0.055f;

    return result;
}

float3 PBRNeutral(float3 color) {
    const float startCompression = 0.8f - 0.04f;
    const float desaturation = 0.15f;

    float x = min(color.r, min(color.g, color.b));
    float offset = x < 0.08f ? x - 6.25f * x * x : 0.04f;
    color -= offset;

    float peak = max(color.r, max(color.g, color.b));
    if (peak < startCompression) return max(color, 0.0f);

    float d = 1.0f - startCompression;
    float newPeak = 1.0f - d * d / (peak + d - startCompression);
    color *= newPeak / peak;

    float g = 1.0f - 1.0f / (desaturation * (peak - newPeak) + 1.0f);
    return lerp(color, newPeak.xxx, g);
}


// --------------------------------------------------------
// Firefly filter tuning (make these uniforms if you prefer)
// --------------------------------------------------------
static const float3 kLuma = float3(0.2126f, 0.7152f, 0.0722f);

// "Much larger than the mean" threshold (ratio).
// Typical values: 3..10 depending on how aggressive you want it.
static const float kFireflyThreshold = 4.0f;

// Avoid triggering in very dark regions where mean is ~0.
static const float kFireflyMinMeanLum = 1e-4f;

struct Sample
{
    float  lum;
    float3 rgb;
};

void Swap(inout Sample a, inout Sample b)
{
    Sample t = a; a = b; b = t;
}

// Simple selection sort for 9 samples (small + deterministic)
void Sort9ByLum(inout Sample s[9])
{
    [unroll]
    for (int i = 0; i < 8; ++i)
    {
        int minIdx = i;
        float minLum = s[i].lum;

        [unroll]
        for (int j = i + 1; j < 9; ++j)
        {
            if (s[j].lum < minLum)
            {
                minLum = s[j].lum;
                minIdx = j;
            }
        }

        if (minIdx != i)
            Swap(s[i], s[minIdx]);
    }
}

[numthreads(16, 16, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    if (DTid.x >= gImageWidth || DTid.y >= gImageHeight) return;

    int2 p0 = int2(DTid.xy);
    int2 maxP = int2((int)gImageWidth - 1, (int)gImageHeight - 1);

    // --------------------------------------------------------
    // 1. Process Main Image (with firefly filtering)
    // --------------------------------------------------------

    // 3x3 offsets
    const int2 offs[9] =
    {
        int2(-1,-1), int2( 0,-1), int2( 1,-1),
        int2(-1, 0), int2( 0, 0), int2( 1, 0),
        int2(-1, 1), int2( 0, 1), int2( 1, 1)
    };

    Sample s[9];
    float sumLum = 0.0f;

    // Gather neighborhood in *final linear* (after ML multiplier)
    [unroll]
    for (int i = 0; i < 9; ++i)
    {
        int2 pp = clamp(p0 + offs[i], int2(0, 0), maxP);

        float3 restir_out_i = gScratchPing[uint3(pp, 2)].rgb;
        float  log_adj_i    = gScratchPing[uint3(pp, 10)].x;
        float  mul_i        = exp(log_adj_i);

        float3 lin_i = restir_out_i * mul_i;

        float lum_i = dot(lin_i, kLuma);

        s[i].rgb = lin_i;
        s[i].lum = lum_i;

        sumLum += lum_i;
    }

    float meanLum = sumLum * (1.0f / 9.0f);

    // Center sample is index 4 (offset 0,0)
    float  centerLum = s[4].lum;
    float3 finalLinear = s[4].rgb;

    // Median (by luminance)
    Sort9ByLum(s);
    float3 medianRGB = s[4].rgb; // 5th element after sort (0..8)

    // Firefly condition: center is much brighter than neighborhood mean
    // (also guard against near-black mean to avoid over-triggering)
    bool isFirefly = (meanLum > kFireflyMinMeanLum) && (centerLum > meanLum * kFireflyThreshold);

    if (isFirefly)
        finalLinear = medianRGB;

    // Apply Gamma
    float3 outSRGB = sRGBGammaCorrection(saturate(finalLinear));
    gOutput[uint3(DTid.xy, 0)] = float4(outSRGB, 1.0f);

    // --------------------------------------------------------
    // 2. Process Debug Visualization (Slice 10)
    // --------------------------------------------------------
    // Use the *center* multiplier for debug, matching your old behavior
    float log_adjustment = gScratchPing[uint3(DTid.xy, 10)].x;
    float linear_multiplier = exp(log_adjustment);

    float3 debugColor;
    if (linear_multiplier < 1.0f)
    {
        debugColor = lerp(float3(0, 0, 1), float3(1, 1, 1), saturate(linear_multiplier));
    }
    else
    {
        float t = saturate(linear_multiplier - 1.0f);
        debugColor = lerp(float3(1, 1, 1), float3(1, 0, 0), t);
    }

    gOutput[uint3(DTid.xy, 10)] = float4(debugColor, 1.0f);
    gOutput[uint3(DTid.xy, 11)] = float4(sRGBGammaCorrection(saturate(gScratchPing[uint3(p0, 2)].rgb)), 1.0f);
}

