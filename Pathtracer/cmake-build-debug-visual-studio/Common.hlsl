#define PI 3.1415f
#define s_bias 0.00001f // Shadow ray bias value
#define EPSILON 0.0001f

#define LUT_SIZE_ROUGHNESS 16
#define LUT_SIZE_THETA 16

#define RIS_M 10

// Hit information, aka ray payload
// This sample only carries a shading color and hit distance.
// Note that the payload should be kept as small as possible,
// and that its size must be declared in the corresponding
// D3D12_RAYTRACING_SHADER_CONFIG pipeline subobject.
struct HitInfo {
  float4 colorAndDistance;
  float3 indirectThroughput; //Throughput after the first bounce
  float3 emission;
  float3 u_emission;
  float3 indirectEmission;
  float3 direction;
  float3 origin;
  float2 util; //IMPORTANT: util info: miss flag
  uint2 seed;
  float pdf;
  float3 hitNormal;
  float3 localHit;
  float reflectiveness;
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

// RIS reservoir for direct lighting
struct Reservoir
{
    uint Y; // index of most important light
    float W_y; // light weight
    float W_sum; // sum of all weights for all lights processed
    float M; // number of lights processed for this reservoir
};

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


// Update the reservoir with the light
bool UpdateReservoir(inout Reservoir reservoir, uint X, float w, float c, inout uint2 seed)
{
    reservoir.W_sum += w;
    reservoir.M += c;

    if ( RandomFloat(seed) < w / reservoir.W_sum  )
    {
        reservoir.Y = X;
        return true;
    }

    return false;
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











