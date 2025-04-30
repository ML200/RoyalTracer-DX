// Helper function to safely multiply a scalar and a float3
float3 SafeMultiply(float scalar, float3 vec)
{
    float3 result = scalar * vec;
    // Check if any component is NaN or infinity
    if (any(isnan(result)) || any(isinf(result)))
    {
        return float3(0.0, 0.0, 0.0);
    }
    return result;
}

// Helper function to safely multiply a scalar and a float3
float SafeMultiplyScalar(float scalar, float vec)
{
    float result = scalar * vec;
    // Check if any component is NaN or infinity
    if (isnan(result) || isinf(result))
    {
        return 0.0f;
    }
    return result;
}

// Conversion to scalar value used for phat
float GetPHat(float3 v){
    return length(v);
}

float3 sRGBGammaCorrection(float3 color)
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