#define PI 3.1415f
#define s_bias 0.00002f // Shadow ray bias value
#define EPSILON 0.000001f // Floating point precision correction

#define LUT_SIZE_THETA 16
#define EXPOSURE 1.0f

#define nee_samples 4
#define nee_samples_DI 4
#define bsdf_samples_DI 1
#define bounces 3
#define rr_threshold 1

#define spatial_candidate_count 3
#define spatial_max_tries 9
#define spatial_radius 20
#define spatial_exponent 1.0f
#define spatial_M_cap 128
#define spatial_M_cap_GI 128
#define temporal_M_cap 16
#define temporal_M_cap_GI 16
#define temporal_r_threshold 0.09f
#define w_sum_threshold 5.0f
#define j_threshold 5.0f

#define beta 1.0f

#define MIN_DIST   (2.0f * s_bias)      // Minimum allowed distance from shading point to light hit

// Hit information, aka ray payload
// This sample only carries a shading color and hit distance.
// Note that the payload should be kept as small as possible,
// and that its size must be declared in the corresponding
// D3D12_RAYTRACING_SHADER_CONFIG pipeline subobject.
struct [[raypayload]] HitInfo {
  float3 hitPosition : read(caller)
                       : write(anyhit,closesthit,miss);
  uint materialID : read(caller)
                       : write(anyhit,closesthit,miss);
  float3 hitNormal : read(caller)
                       : write(anyhit,closesthit,miss);
  float area: read(caller)
                       : write(anyhit,closesthit,miss);
  uint objID: read(caller)
                       : write(anyhit,closesthit,miss);
};

struct [[raypayload]] ShadowHitInfo {
  bool isHit: read(caller)
                       : write(anyhit,closesthit,miss);
};

struct Material
{
     float4 Kd;
     float3 Ks; float Ni;
     float3 Ke; float pad0;
     float4 Pr_Pm_Ps_Pc;
     float LUT[LUT_SIZE_THETA];
};

struct MaterialOptimized // Memory optimized material to reduce register pressure, aligned to 16 bytes per read
{
     half4 Kd; half4 Pr_Pm_Ps_Pc; // 8 + 8 = 16 bytes
     half3 Ks; half3 Ke; uint mID; // 6 + 6 + 4 = 16 bytes
};


// Default background material (Ray miss)
static const MaterialOptimized g_DefaultMissMaterial =
{
    half4(0,0,0,0), half4(0,0,0,0),
    half3(0,0,0), half3(0,0,0), 4294967294
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
    float cdf;       // 16 bytes
    float3 y;
    uint instanceID; // 16 bytes
    float3 z;
    float weight;       // 16 bytes
    float3 emission;
    uint triCount;   // 16 bytes
    float total_weight;
    float3 pad0;       // 16 bytes
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

    [loop]
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

float3 GetColorFromValue(float value, float minValue, float maxValue)
{
    // Compute normalized interpolation factor between 0 and 1, clamped
    float t = saturate((value - minValue) / (maxValue - minValue));

    // Lerp between red (1,0,0) and green (0,1,0) using t
    return lerp(float3(1, 0, 0), float3(0, 1, 0), t);
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

// Column Major
inline uint MapPixelID(uint2 dims, uint2 lIndex){
    return lIndex.x * dims.y + lIndex.y;
}

// Row Major
/*inline uint MapPixelID(uint2 dims, uint2 lIndex){
    return lIndex.y * dims.x + lIndex.x;
}*/

// Swizzling
/*inline uint MapPixelID(uint2 dims, uint2 lIndex)
{
    // NVIDIA’s 2D warp tiles are 8×4
    const uint tileWidth  = 4;
    const uint tileHeight = 8;

    // how many tiles fit across (ceiling)
    uint tileCountX = (dims.x + tileWidth  - 1) / tileWidth;
    // (we don’t need tileCountY explicitly for flattening)

    // which tile (x, y) this pixel lives in
    uint tileX = lIndex.x / tileWidth;
    uint tileY = lIndex.y / tileHeight;

    // local coords inside that tile
    uint localX = lIndex.x % tileWidth;
    uint localY = lIndex.y % tileHeight;

    // flatten tile index in row‑major across the grid of tiles
    uint tileIndex = tileY * tileCountX + tileX;
    // flatten local index in row‑major within the 8×4 tile
    uint localIndex = localY * tileWidth + localX;

    // combine: skip all pixels in earlier tiles, then add this one
    return tileIndex * (tileWidth * tileHeight) + localIndex;
}*/




inline uint GetRandomPixelCircleWeighted(uint radius, uint w, uint h, uint x, uint y, inout uint2 seed)
{
    int newX, newY;
    do {
        // Get a uniform random value.
        float u = RandomFloat(seed);
        // Adjust the weighting by using a power law.
        float z = pow(u, spatial_exponent);
        // Compute the radius value with the adjustable bias.
        float r = float(radius) * z;
        // Choose an angle uniformly from [0, 2π).
        float angle = RandomFloat(seed) * 6.2831853;
        // Compute offsets.
        int offsetX = int(cos(angle) * r);
        int offsetY = int(sin(angle) * r);
        // Calculate new coordinates.
        newX = int(x) + offsetX;
        newY = int(y) + offsetY;

        // Mirror newX into the [0, w-1] range.
        while(newX < 0 || newX >= int(w)) {
            if(newX < 0)
                newX = -newX;
            else // newX >= w
                newX = 2 * int(w) - newX - 2;
        }

        // Mirror newY into the [0, h-1] range.
        while(newY < 0 || newY >= int(h)) {
            if(newY < 0)
                newY = -newY;
            else // newY >= h
                newY = 2 * int(h) - newY - 2;
        }
    } while(newX == int(x) && newY == int(y));  // Reject the center pixel.

    //return newX * h + newY;
    return MapPixelID(uint2(w, h), uint2(newX,newY));
}

inline uint2 GetRandomPixelCircleWeighted_d(uint radius, uint w, uint h, uint x, uint y, inout uint2 seed)
{
    int newX, newY;
    do {
        // Get a uniform random value.
        float u = RandomFloat(seed);
        // Adjust the weighting by using a power law.
        float z = pow(u, spatial_exponent);
        // Compute the radius value with the adjustable bias.
        float r = float(radius) * z;
        // Choose an angle uniformly from [0, 2π).
        float angle = RandomFloat(seed) * 6.2831853;
        // Compute offsets.
        int offsetX = int(cos(angle) * r);
        int offsetY = int(sin(angle) * r);
        // Calculate new coordinates.
        newX = int(x) + offsetX;
        newY = int(y) + offsetY;

        // Mirror newX into the [0, w-1] range.
        while(newX < 0 || newX >= int(w)) {
            if(newX < 0)
                newX = -newX;
            else // newX >= w
                newX = 2 * int(w) - newX - 2;
        }

        // Mirror newY into the [0, h-1] range.
        while(newY < 0 || newY >= int(h)) {
            if(newY < 0)
                newY = -newY;
            else // newY >= h
                newY = 2 * int(h) - newY - 2;
        }
    } while(newX == int(x) && newY == int(y));  // Reject the center pixel.

    //return newX * h + newY;
    return uint2(newX,newY);
}


inline bool RejectLocation(uint x, uint y, uint s1, uint s2, Material mat){
    if(s1 == 1 && s2 == 1 && mat.Pr_Pm_Ps_Pc.x < temporal_r_threshold){
        if(y != x)
            return true;
    }
    return false;
}

inline bool RejectLength(float l1, float l2, float threshold){
    if(l1/l2 < threshold || l2/l1 < threshold){
        return true;
    }
    return false;
}

inline bool RejectJacobian(float J, float threshold){
    if(J > threshold || J < 1.0f/threshold || isnan(J) || isinf(J))
        return true;
    return false;
}

inline bool RejectWsum(float w_sum, float threshold){
    if(w_sum > threshold)
        return true;
    return false;
}

inline bool RejectRoughness(float4x4 view_i, float4x4 prevView_i, uint tempPixelID, uint currentPixelID, uint strategy, float roughness){
    // Compare the view matrices and reset if different
    bool different = false;
    for (int row = 0; row < 4; row++)
    {
        float4 diff = abs(view_i[row] - prevView_i[row]);
        if (any(diff > s_bias))
        {
            different = true;
            break;
        }
    }

    if(strategy == 1
       && roughness < temporal_r_threshold
       && different
    ){
        return true;
    }
    return false;
}

inline bool RejectNormal(float3 n1, float3 n2, float threshold){
    float similarity = dot(n1, n2);
    return (similarity < threshold);
}

inline bool RejectBelowSurface(float3 d, float3 n){
    float similarity = dot(d,  n);
    return (similarity < 0.0f);
}

inline bool RejectDistance(float3 x1, float3 x2, float3 camPos, float threshold)
{
    float d1 = length(x1 - camPos);
    float d2 = length(x2 - camPos);

    float relativeDifference = abs(d1 - d2) / max(d1, d2);
    return relativeDifference > threshold;
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








