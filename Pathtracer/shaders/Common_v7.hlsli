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