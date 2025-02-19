#define PI 3.1415f
#define s_bias 0.00002f // Shadow ray bias value
#define EPSILON 0.0001f // Floating point precision correction

#define LUT_SIZE_THETA 16

#define samples 1
#define bounces 1
#define rr_threshold 3

#define spatial_candidate_count 10
#define spatial_M_cap 20
#define temporal_M_cap 20

// Hit information, aka ray payload
// This sample only carries a shading color and hit distance.
// Note that the payload should be kept as small as possible,
// and that its size must be declared in the corresponding
// D3D12_RAYTRACING_SHADER_CONFIG pipeline subobject.
struct HitInfo {
  float area;
  uint materialID;
  float3 hitPosition;
  float3 hitNormal;
};

struct Material
{
     float4 Kd;
     float3 Ks;
     float3 Ke;
     float4 Pr_Pm_Ps_Pc;
     float2 aniso_anisor;
     float Ni;
     float LUT[32];
};

// Default background material (Ray miss)
static const Material g_DefaultMissMaterial =
{
    float4(0.0,0.0,0.0,0.0f),
    float3(0,0,0),
    float3(0.0,0.0,0.0),
    float4(0,0,0,0),
    float2(0,0),
    1.0f,
    {0,0,0,0,0,0,0,0,
     0,0,0,0,0,0,0,0,
     0,0,0,0,0,0,0,0,
     0,0,0,0,0,0,0,0}
};

struct ShadowHitInfo {
  bool isHit;
};

struct InstanceProperties
{
  float4x4 objectToWorld;
  float4x4 objectToWorldInverse;
  float4x4 prevObjectToWorld;
  float4x4 prevObjectToWorldInverse;
  float4x4 objectToWorldNormal;
  float4x4 prevObjectToWorldNormal;
};

struct LightTriangle {
    float3 x;
    float  pad0;
    float3 y;
    float  pad1;
    float3 z;
    float  pad2;
    uint   instanceID;
    float  weight;
    uint   triCount;
    float  total_weight;
    float3 emission;
    float  cdf;
};


struct STriVertex {
  float3 vertex;
  float4 normal;
};

// #DXR Extra: Per-Instance Data
cbuffer Colors : register(b0) {
  float3 A;
  float3 B;
  float3 C;
}

// Attributes output by the raytracing when hitting a surface,
// here the barycentric coordinates
struct Attributes {
  float2 bary;
};


// Random Float Generator
float RandomFloat(inout uint2 seed)
{
    uint v0 = seed.x;
    uint v1 = seed.y;
    uint sum = 0u;
    const uint delta = 0x9e3779b9u;

    for (uint i = 0u; i < 4u; i++)
    {
        sum += delta;
        v0 += ((v1 << 4u) + 0xA341316Cu) ^ (v1 + sum) ^ ((v1 >> 5u) + 0xC8013EA4u);
        v1 += ((v0 << 4u) + 0xAD90777Du) ^ (v0 + sum) ^ ((v0 >> 5u) + 0x7E95761Eu);
    }

    seed.x = v0;
    seed.y = v1;

    return float(v0) / 4294967296.0;
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

// Sample from a weighted circle disk around a given pixel, never returning (x,y)
uint GetRandomPixelCircleWeighted(uint radius, uint w, uint h, uint x, uint y, inout uint2 seed) {
    int newX, newY;
    do {
        // Get a uniform random value
        float u = RandomFloat(seed);

        // Inverse CDF for F(z) = 3z^2 - 2z^3, with z = r/float(radius)
        float z = 0.5 + cos((1.0/3.0)*acos(1.0 - 2.0*u) - 2.0943951); // 2π/3 ≈ 2.0943951

        // Compute r with bias toward 0
        float r = float(radius) * z;

        // Choose an angle uniformly from [0, 2π)
        float angle = RandomFloat(seed) * 6.2831853; // 2π

        // Compute offsets
        int offsetX = int(cos(angle) * r);
        int offsetY = int(sin(angle) * r);

        // Compute new coordinates clamped to the image bounds
        newX = int(x) + offsetX;
        newY = int(y) + offsetY;
        newX = clamp(newX, 0, int(w) - 1);
        newY = clamp(newY, 0, int(h) - 1);
    } while(newX == int(x) && newY == int(y));  // Reject the center pixel

    return newY * w + newX;
}



bool RejectNormal(float3 n1, float3 n2){
    float similarity = dot(n1, n2);
    return (similarity < 0.9f);
}

bool RejectDistance(float3 x1, float3 x2, float3 camPos, float threshold)
{
    // Calculate distances from the camera to each point.
    float d1 = length(x1 - camPos);
    float d2 = length(x2 - camPos);

    // Compute the relative difference.
    // We divide by the maximum distance to normalize the difference.
    float relativeDifference = abs(d1 - d2) / max(d1, d2);

    // Return true if the relative difference exceeds the threshold.
    return relativeDifference > threshold;
}


// Assume that GetLastFramePixelCoordinates_Float() returns the sub-pixel coordinate in reservoir space.
float2 GetLastFramePixelCoordinates_Float(
    float3 worldPos,
    float4x4 prevView,
    float4x4 prevProjection,
    float2 resolution)
{
    float4 clipPos = mul(prevProjection, mul(prevView, float4(worldPos, 1.0f)));
    if (clipPos.w <= 0.0f)
        return float2(-1.0f, -1.0f);

    float2 ndc = clipPos.xy / clipPos.w;
    float2 screenUV = ndc * 0.5f + 0.5f;
    // Flip Y if needed:
    screenUV.y = 1.0f - screenUV.y;

    // Get the full sub-pixel coordinate in pixel space:
    return screenUV * resolution;
}







