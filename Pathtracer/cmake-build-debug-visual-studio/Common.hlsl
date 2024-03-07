#define PI 3.1415f

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
  uint2 seed;
  float pdf;
};

struct Material
{
     float4 Kd;
     float3 Ks;
     float3 Ke;
     float4 Pr_Pm_Ps_Pc;
     float2 aniso_anisor;
};

// Attributes output by the raytracing when hitting a surface,
// here the barycentric coordinates
struct Attributes {
  float2 bary;
};

// Hash function to mix the seeds
uint hash(uint2 seed)
{
    uint h = seed.x + seed.y * 6364136223846793005u + 1442695040888963407u;
    h = (h ^ (h >> 30)) * 0xbf58476d1ce4e5b9u;
    h = (h ^ (h >> 27)) * 0x94d049bb133111ebu;
    h = h ^ (h >> 31);
    return h;
}

// Improved Random Float Generator
float RandomFloat(inout uint2 seed)
{
    uint h = hash(seed);
    seed += uint2(1,1); // Simple way to update seed to ensure different values on subsequent calls
    // Use the high-quality bits from the middle of the hashed value
    uint randomValue = (h >> 9) | 0x3F800000u;
    return asfloat(randomValue) - 1.0;
}

uint lcg(inout uint seed) {
    const uint LCG_A = 1664525u;
    const uint LCG_C = 1013904223u;
    seed = (LCG_A * seed + LCG_C);
    return seed;
}

float RandomFloatLCG(inout uint seed) {
    return float(lcg(seed)) / float(0xFFFFFFFFu);
}


float GGXDistribution(float alpha, float NoH) {
    float alphaSquared = alpha * alpha;
    float NoHSquared = NoH * NoH;
    float denom = NoHSquared * (alphaSquared - 1.0f) + 1.0f; // Can approach zero
    denom = max(denom, 1e-4f); // Safety check against division by zero
    return alphaSquared / (PI * denom * denom);
}
float GeometrySchlickGGX(float NoV, float k) {
    float denom = NoV * (1.0f - k) + k;
    return NoV / denom;
}

float GeometrySmith(float NoV, float NoL, float alpha) {
    float k = alpha * alpha / 2.0f;
    // Ensure NoV and NoL are not zero to avoid division by zero in GeometrySchlickGGX
    NoV = max(NoV, 1e-4f);
    NoL = max(NoL, 1e-4f);
    float ggx1 = GeometrySchlickGGX(NoV, k);
    float ggx2 = GeometrySchlickGGX(NoL, k);
    return ggx1 * ggx2;
}

float3 FresnelSchlick(float cosTheta, float3 F0) {
    cosTheta = clamp(cosTheta, 1e-4f, 1.0f); // Clamp to avoid pow with negative numbers
    return F0 + (1.0f - F0) * pow(1.0f - cosTheta, 5.0f);
}

float CalculateF0Scalar(float refractiveIndex) {
    return pow((refractiveIndex - 1.0f) / (refractiveIndex + 1.0f), 2.0f);
}

float3 CalculateF0Vector(float refractiveIndex) {
    float F0Scalar = CalculateF0Scalar(refractiveIndex);
    return float3(F0Scalar, F0Scalar, F0Scalar); // Uniform reflectance across RGB
}

float3 RandomUnitVectorInHemisphere(float3 normal, inout uint2 seed)
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

float3 SampleGGXVNDF(float3 N, float3 flatNormal, float3 V, float alpha, inout uint2 seed, out float pdf) {
    float alphaSquared = alpha * alpha;

    // Generate two random numbers for sampling
    float u1 = RandomFloatLCG(seed.x);
    float u2 = RandomFloatLCG(seed.y);

    // Sample theta and phi angles for H vector in spherical coordinates
    float theta = atan(alphaSquared * sqrt(u1) / sqrt(1.0 - u1));
    float phi = 2.0 * PI * u2;

    // Convert spherical coordinates to Cartesian coordinates for H
    float x = sin(theta) * cos(phi);
    float y = sin(theta) * sin(phi);
    float z = cos(theta);

    // Construct a local orthonormal basis (TBN matrix) around N
    float3 up = abs(N.z) < 0.999 ? float3(0,0,1) : float3(1,0,0);
    float3 right = normalize(cross(up, N));
    float3 forward = cross(N, right);

    // Transform H from local space to world space
    float3 H = normalize(x * right + y * forward + z * N);

    // Calculate the reflection direction L based on H and V
    float3 L = 2.0f * dot(V, H) * H - V;

    // Check if L is below the surface; if so, reflect it above the surface
    if (dot(flatNormal, L) < 0.0) {
        L = -L;
    }

    // Calculate PDF for GGX distribution using H
    float NoH = normalize(max(dot(N, H), 0.0f));
    float NoV = normalize(max(dot(N, V), 0.0f));
    float VoH = normalize(max(dot(V, H), 0.0f));
    float HoL = normalize(max(dot(H,L),0.0f));

    // PDF calculation adjusted for clarity and correctness
    pdf = GGXDistribution(alpha, NoH) * NoH / (4.0f * VoH * HoL);

    // Return the sampled vector L and the PDF by reference
    return normalize(L);
}


float3 Reflect(float3 incident, float3 normal)
{
    // Ensure normal is normalized
    normal = normalize(normal);

    // Calculate the reflection vector
    float3 reflected = incident - 2.0 * dot(incident, normal) * normal;

    return reflected;
}

float3 getPerpendicularVector(float3 v)
{
    // Find the smallest component of the input vector
    float minComponent = min(min(v.x, v.y), v.z);

    // Construct a vector that is not parallel to the input vector
    float3 nonParallelVec;
    if (minComponent == v.x)
        nonParallelVec = float3(1.0f, 0.0f, 0.0f);  // Input vector is mostly aligned with X-axis
    else if (minComponent == v.y)
        nonParallelVec = float3(0.0f, 1.0f, 0.0f);  // Input vector is mostly aligned with Y-axis
    else
        nonParallelVec = float3(0.0f, 0.0f, 1.0f);  // Input vector is mostly aligned with Z-axis

    // Find a perpendicular vector using cross product
    return cross(v, nonParallelVec);
}