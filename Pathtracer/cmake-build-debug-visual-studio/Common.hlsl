// Hit information, aka ray payload
// This sample only carries a shading color and hit distance.
// Note that the payload should be kept as small as possible,
// and that its size must be declared in the corresponding
// D3D12_RAYTRACING_SHADER_CONFIG pipeline subobject.
struct HitInfo {
  float4 colorAndDistance;
  float3 emission;
  float3 direction;
  float3 origin;
  float util; //IMPORTANT: util info: miss flag
  uint seed;
  float pdf;
};

// Attributes output by the raytracing when hitting a surface,
// here the barycentric coordinates
struct Attributes {
  float2 bary;
};

uint lcg(inout uint seed) {
    const uint LCG_A = 1664525u;
    const uint LCG_C = 1013904223u;
    seed = (LCG_A * seed + LCG_C);
    return seed;
}

float RandomFloat(inout uint seed) {
    return float(lcg(seed)) / float(0xFFFFFFFFu);
}

float3 RandomUnitVectorInHemisphere(float3 normal, inout uint seed)
{
    // Generate two random numbers
    float u1 = RandomFloat(seed);
    float u2 = RandomFloat(seed);

    // Convert uniform random numbers to cosine-weighted polar coordinates
    float r = sqrt(u1);
    float theta = 2.0 * 3.14159265358979323846 * u2;

    // Project polar coordinates to sample on the unit disk
    float x = r * cos(theta);
    float y = r * sin(theta);

    // Project up to hemisphere
    float z = sqrt(max(0.0f, 1.0f - x*x - y*y));

    // Create a local orthonormal basis centered around the normal
    float3 h = normal;
    float3 up = abs(normal.z) < 0.999 ? float3(0,0,1) : float3(1,0,0);
    float3 right = normalize(cross(up, h));
    float3 forward = cross(h, right);

    // Convert disk sample to hemisphere sample in the local basis
    float3 hemisphereSample = x * right + y * forward + z * h;

    return normalize(hemisphereSample);
}
