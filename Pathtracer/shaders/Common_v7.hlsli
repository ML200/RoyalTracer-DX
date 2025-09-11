// Safe 2-D→1-D swizzle; returns 0xFFFFFFFF on invalid input
inline uint MapPixelID(uint2 dims, int2 lIndex)
{
    // ---------- 1. validate input ----------
    // negatives or out-of-range ──► sentinel
    if (lIndex.x < 0 || lIndex.y < 0 ||
        lIndex.x >= int(dims.x) || lIndex.y >= int(dims.y))
    {
        return 0xFFFFFFFF;          // invalid
    }

    // ---------- 2. original mapping ----------
    const uint tileWidth  = 4;
    const uint tileHeight = 8;

    uint2 uIndex   = uint2(lIndex);
    uint tileCountX = (dims.x + tileWidth - 1u) / tileWidth;

    uint tileX = uIndex.x / tileWidth;
    uint tileY = uIndex.y / tileHeight;

    uint localX = uIndex.x % tileWidth;
    uint localY = uIndex.y % tileHeight;

    uint tileIndex  = tileY * tileCountX + tileX;
    uint localIndex = localY * tileWidth + localX;

    return tileIndex * (tileWidth * tileHeight) + localIndex;
}


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
inline float GetPHat(float3 v){
    return 0.2126f * v.x + 0.7152f * v.y + 0.0722f * v.z;
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