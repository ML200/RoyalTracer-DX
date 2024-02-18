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
  uint seed;
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

float3 SampleCone(float3 normal, float3 target, float roughness, inout uint seed, out float pdf) {
    // Convert roughness to a cosine of the maximum angle for the cone
    float cosMaxAngle = sqrt(1.0 - roughness * roughness);

    // Generate two random numbers for sampling
    float u1 = RandomFloat(seed);
    float u2 = RandomFloat(seed);

    // Sample within the cone
    float cosTheta = (1.0 - u1) + u1 * cosMaxAngle;
    float sinTheta = sqrt(1.0 - cosTheta * cosTheta);
    float phi = 2.0 * PI * u2;

    // Convert to Cartesian coordinates
    float x = sinTheta * cos(phi);
    float y = sinTheta * sin(phi);
    float z = cosTheta;

    // Create a local orthonormal basis around the target vector
    float3 h = normalize(target);
    float3 up = abs(h.z) < 0.999 ? float3(0,0,1) : float3(1,0,0);
    float3 right = normalize(cross(up, h));
    float3 forward = cross(h, right);

    // Convert the sampled direction from local basis to world space
    float3 sampleVec = x * right + y * forward + z * h;

    // If the vector is below the surface, mirror it
    if (dot(sampleVec, normal) < 0.0) {
        sampleVec = -sampleVec;
    }

    // PDF calculation for the sampled direction
    if (roughness == 0.0) {
        pdf = 1.0;
    } else {
        pdf = (cosTheta / PI) / (1.0 - cosMaxAngle);
    }

    return normalize(sampleVec);
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