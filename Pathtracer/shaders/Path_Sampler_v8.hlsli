// Beer-Lambert absorption along a ray segment.
inline float3 CalculateAbsorptionThroughput(
    float3 tintColor,
    float distanceTraveled)
{
    return float3(
        exp(-tintColor.x * distanceTraveled),
        exp(-tintColor.y * distanceTraveled),
        exp(-tintColor.z * distanceTraveled)
    );
}
