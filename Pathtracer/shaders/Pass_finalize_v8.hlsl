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


// ─────────────────────────────────────────────────────────────────────────────
//  Finalize PASS
// ─────────────────────────────────────────────────────────────────────────────
[numthreads(16, 16, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    if (DTid.x >= gImageWidth || DTid.y >= gImageHeight) return;

    // --------------------------------------------------------
    // 1. Process Main Image (Slice 0)
    // --------------------------------------------------------

    // Load the base image (Restir result)
    float3 restir_out = gScratchPing[uint3(DTid.xy, 2)].rgb;

    // Load the ML adjustment (Log-Offset) from Slice 10
    float log_adjustment = gScratchPing[uint3(DTid.xy, 10)].x;

    // FIX: Convert Log-Offset to Linear Multiplier using exp()
    float linear_multiplier = exp(log_adjustment);

    // Apply correction
    float3 finalLinear = restir_out * linear_multiplier;

    // Apply Gamma
    float3 outSRGB = sRGBGammaCorrection(saturate(finalLinear));
    gOutput[uint3(DTid.xy, 0)] = float4(outSRGB, 1.0f);

    // --------------------------------------------------------
    // 2. Process Debug Visualization (Slice 1)
    // --------------------------------------------------------
    // Scheme: 0.0 (Blue/Darken) -> 1.0 (White/Neutral) -> 2.0 (Red/Brighten)

    float3 debugColor;

    // Visualize the MULTIPLIER, not the raw log output
    if (linear_multiplier < 1.0f)
    {
        debugColor = lerp(float3(0, 0, 1), float3(1, 1, 1), saturate(linear_multiplier));
    }
    else
    {
        // Remap 1.0..2.0 range for visualization
        float t = saturate(linear_multiplier - 1.0f);
        debugColor = lerp(float3(1, 1, 1), float3(1, 0, 0), t);
    }

    gOutput[uint3(DTid.xy, 10)] = float4(debugColor, 1.0f);
}
