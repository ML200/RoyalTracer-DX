#define COMPUTE_PASS
#include "Includes_v8.hlsli"

//====================================================================
//SRGB GAMMA CORRECTION
//====================================================================
inline float3 sRGBGammaCorrection(float3 color)
{
    float3 result;

    //Red channel
    if (color.r <= 0.0031308f)
        result.r = 12.92f * color.r;
    else
        result.r = 1.055f * pow(color.r, 1.0f / 2.4f) - 0.055f;

    //Green channel
    if (color.g <= 0.0031308f)
        result.g = 12.92f * color.g;
    else
        result.g = 1.055f * pow(color.g, 1.0f / 2.4f) - 0.055f;

    //Blue channel
    if (color.b <= 0.0031308f)
        result.b = 12.92f * color.b;
    else
        result.b = 1.055f * pow(color.b, 1.0f / 2.4f) - 0.055f;

    return result;
}

//====================================================================
//PBR NEUTRAL TONEMAP
//====================================================================
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

//====================================================================
//GAMMA CORRECTION AND POST-PROCESS PASS
//====================================================================
[numthreads(8, 4, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    if (DTid.x >= gImageWidth || DTid.y >= gImageHeight) return;

    //Load output slices
    float3 noisy = gOutput[uint3(DTid.xy, 0)].xyz;
    float3 clean = g_dlssOutput[DTid.xy].xyz;
    float3 gt = gOutput[uint3(DTid.xy, 2)].xyz;

    //Apply gamma correction
    noisy = sRGBGammaCorrection(noisy);
    clean =  sRGBGammaCorrection(clean);
    gt =  sRGBGammaCorrection(gt);

    gOutput[uint3(DTid.xy, 0)] = float4(noisy, 0.0f);
    gOutput[uint3(DTid.xy, 1)] = float4(clean, 0.0f);
    gOutput[uint3(DTid.xy, 2)] = float4(gt, 0.0f);
}
