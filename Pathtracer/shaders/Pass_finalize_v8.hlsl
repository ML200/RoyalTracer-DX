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

[numthreads(16, 16, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    if (DTid.x >= IMG_W || DTid.y >= IMG_H) return;
    gDispatchIdx = DTid;

    uint2  launchIndex   = DTid.xy;
    float2 dims = float2(IMG_W, IMG_H);
    uint   pixelIdx  = MapPixelID(dims, launchIndex);
    uint seed = GetSeed(pixelIdx, time, 2).x;
    float prob = RandomFloatSingle(seed);

    // Fetch center data
    float3 restir_out = gScratchPing[uint3(DTid.xy, 2)].rgb;
    float  confidence    = saturate(abs(gScratchPing[uint3(DTid.xy, 10)].x));

    if(confidence > prob){
        //InvalidateReservoirGI_ShadingNormal(g_Reservoirs_last_gi, pixelIdx);
        //store_W_gi(g_Reservoirs_last_gi, pixelIdx, 0.0f);
        //uint M = load_M_gi(g_Reservoirs_last_gi, pixelIdx);
        store_M_gi(g_Reservoirs_last_gi, pixelIdx, 1u);
    }

    // Apply Gamma and output
    float3 outSRGB = sRGBGammaCorrection(saturate(restir_out));
    gOutput[uint3(DTid.xy, 0)] = float4(outSRGB, 1.0f);
    gOutput[uint3(DTid.xy, 10)] = float4(restir_out, 1.0f);
    gOutput[uint3(DTid.xy, 11)] = float4((float3)confidence, 1.0f);
}