#define PI 3.1415f
#define s_bias 0.00001f // Shadow ray bias value
#define EPSILON 0.0001f // Floating point precision correction

#define LUT_SIZE_THETA 16

#define RIS_M 10
#define samples 1
#define bounces 1
#define rr_threshold 3

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

struct ShadowHitInfo {
  bool isHit;
};

struct InstanceProperties
{
  float4x4 objectToWorld;
  float4x4 prevObjectToWorld;
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


// Improved Random Float Generator using TEA
float RandomFloat(inout uint2 seed)
{
    uint v0 = seed.x;
    uint v1 = seed.y;
    uint sum = 0u;
    const uint delta = 0x9e3779b9u; // A key schedule constant

    // TEA encryption rounds (reduced to 4 for performance)
    for (uint i = 0u; i < 4u; i++)
    {
        sum += delta;
        v0 += ((v1 << 4u) + 0xA341316Cu) ^ (v1 + sum) ^ ((v1 >> 5u) + 0xC8013EA4u);
        v1 += ((v0 << 4u) + 0xAD90777Du) ^ (v0 + sum) ^ ((v0 >> 5u) + 0x7E95761Eu);
    }

    // Update the seed
    seed.x = v0;
    seed.y = v1;

    // Normalize the result to [0, 1)
    return float(v0) / 4294967296.0; // 2^32 = 4294967296
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

// Gaussian function for edge weighting (for spatial and temporal filtering)
float Gaussian(float dist2, float sigma) {
    return exp(-dist2 / (2.0f * sigma * sigma));
}

// Tonemapping function (Reinhard operator)
float3 ReinhardTonemap(float3 color)
{
    // Apply Reinhard tonemapping
    return color / (color + 1.0f);
}

// Alternatively, for a slightly brighter image, you can use a scaling factor
float3 ReinhardTonemapScaled(float3 color, float exposure)
{
    color *= exposure;
    return color / (color + 1.0f);
}


float2 ComputeMotionVector(float3 worldPos,
                           row_major float4x4 view, row_major float4x4 projection,
                           row_major float4x4 prevView, row_major float4x4 prevProjection,
                           float screenWidth, float screenHeight)
{
    // Current frame transformations
    float4 currentViewPos = mul(float4(worldPos, 1.0f), view);
    float4 currentClipPos = mul(currentViewPos, projection);

    // Previous frame transformations
    float4 prevViewPos = mul(float4(worldPos, 1.0f), prevView);
    float4 prevClipPos = mul(prevViewPos, prevProjection);

    // Perspective divide (from clip space to NDC)
    float w_current = max(currentClipPos.w, 1e-5f);
    float w_prev = max(prevClipPos.w, 1e-5f);

    float2 currentNDC = currentClipPos.xy / w_current;
    float2 prevNDC = prevClipPos.xy / w_prev;

    // Convert NDC to screen space coordinates
    float2 currentScreenPos = (currentNDC * 0.5f + 0.5f) * float2(screenWidth, screenHeight);
    float2 prevScreenPos = (prevNDC * 0.5f + 0.5f) * float2(screenWidth, screenHeight);

    // Compute the motion vector (difference between current and previous screen space positions)
    float2 motionVector = currentScreenPos - prevScreenPos;

    return motionVector;
}











