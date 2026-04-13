// Common utility functions

// Estimated luminance
inline float Luma(float3 c) { return dot(c, float3(0.2126, 0.7152, 0.0722)); }
// Average
inline float Avg3(float3 c) { return dot(c, float3(0.33333f, 0.33333f, 0.33333f)); }

// Map pixel coordinate to tile-swizzled linear index
inline uint MapPixelID(uint2 dims, int2 lIndex)
{
    if (lIndex.x < 0 || lIndex.y < 0 ||
        lIndex.x >= int(dims.x) || lIndex.y >= int(dims.y))
    {
        return 0xFFFFFFFF;
    }
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

inline int2 UnmapPixelID(uint pixelID, uint2 dims)
{
    if (pixelID == 0xFFFFFFFF)
    {
        return int2(-1, -1);
    }

    const uint tileWidth  = 4;
    const uint tileHeight = 8;
    const uint tileSize   = tileWidth * tileHeight; // 32

    uint tileIndex  = pixelID / tileSize;
    uint localIndex = pixelID % tileSize;

    uint tileCountX = (dims.x + tileWidth - 1u) / tileWidth;

    uint tileY = tileIndex / tileCountX;
    uint tileX = tileIndex % tileCountX;

    uint localY = localIndex / tileWidth;
    uint localX = localIndex % tileWidth;

    uint globalX = tileX * tileWidth + localX;
    uint globalY = tileY * tileHeight + localY;

    // Padding check
    if (globalX >= dims.x || globalY >= dims.y)
    {
        return int2(-1, -1);
    }

    return int2(globalX, globalY);
}

void ApplyPermutationSampling(inout int2 prevPixelPos, uint uniformRandomNumber)
{
    int2 offset = int2(uniformRandomNumber & 3, (uniformRandomNumber >> 2) & 3);
    prevPixelPos += offset;

    prevPixelPos.x ^= 3;
    prevPixelPos.y ^= 3;

    prevPixelPos -= offset;
}

float3 EnvBRDFApprox2(float3 Kd, float Pr, float Pm, float NoV)
{
    // Compute F0 (specular albedo) from metallic workflow inputs
    float3 SpecularColor = lerp(0.04.xxx, Kd, saturate(Pm));

    // Convert perceptual roughness to GGX alpha
    float alpha = Pr * Pr;

    NoV = abs(NoV);

    // [Ray Tracing Gems, Chapter 32]
    float4 X;
    X.x = 1.f;
    X.y = NoV;
    X.z = NoV * NoV;
    X.w = NoV * X.z;

    float4 Y;
    Y.x = 1.f;
    Y.y = alpha;
    Y.z = alpha * alpha;
    Y.w = alpha * Y.z;

    float2x2 M1 = float2x2(0.99044f, -1.28514f,
                           1.29678f, -0.755907f);

    float3x3 M2 = float3x3(1.f,     2.92338f,  59.4188f,
                           20.3225f, -27.0302f, 222.592f,
                           121.563f, 626.13f,   316.627f);

    float2x2 M3 = float2x2(0.0365463f,  3.32707f,
                           9.0632f,    -9.04756f);

    float3x3 M4 = float3x3(1.f,      3.59685f, -1.36772f,
                           9.04401f, -16.3174f,  9.22949f,
                           5.56589f,  19.7886f, -20.2123f);

    float bias  = dot(mul(M1, X.xy),  Y.xy)  * rcp(dot(mul(M2, X.xyw), Y.xyw));
    float scale = dot(mul(M3, X.xy),  Y.xy)  * rcp(dot(mul(M4, X.xzw), Y.xyw));

    bias *= saturate(SpecularColor.g * 50);

    return mad(SpecularColor, max(0, scale), max(0, bias));
}

#define BOIL_GROUP_X 16
#define BOIL_GROUP_Y 16
#define BOIL_THREADS (BOIL_GROUP_X * BOIL_GROUP_Y) // 256

// Per-thread storage for the 16x16 reduction (one float per thread in the group)
groupshared float gBoilValues[BOIL_THREADS];

float BoilMultiplier(float strength)
{
    return 10.0f / clamp(strength, 1e-6f, 1.0f) - 9.0f;
}

bool BoilingFilter(
    uint2 localIndex,
    float filterStrength,
    float v,
    out float avgNonzero,
    out float threshold)
{
    uint gsIdx = localIndex.x + localIndex.y * BOIL_GROUP_X; // 0..255

    // Sum reduction over full 16x16 group
    gBoilValues[gsIdx] = v;
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint stride = 128u; stride > 0u; stride >>= 1u)
    {
        if (gsIdx < stride)
        {
            gBoilValues[gsIdx] += gBoilValues[gsIdx + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    float groupSum = gBoilValues[0];

    // Barrier: all threads must read groupSum before the array is reused
    GroupMemoryBarrierWithGroupSync();

    // Count reduction over full 16x16 group
    gBoilValues[gsIdx] = (v > 0.0f) ? 1.0f : 0.0f;
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint stride2 = 128u; stride2 > 0u; stride2 >>= 1u)
    {
        if (gsIdx < stride2)
        {
            gBoilValues[gsIdx] += gBoilValues[gsIdx + stride2];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    uint groupCnt = (uint)gBoilValues[0];

    avgNonzero = (groupCnt > 0) ? (groupSum / float(groupCnt)) : 0.0f;
    threshold  = avgNonzero * BoilMultiplier(filterStrength);

    return (v > threshold);
}

// Wave-only variant for raygen shaders (no groupshared memory)
bool BoilingFilter_Wave(
    float filterStrength,
    float v,
    out float avgNonzero,
    out float threshold)
{
    float waveSum = WaveActiveSum(v);
    uint  waveCnt = WaveActiveCountBits(v > 0.0f);

    avgNonzero = (waveCnt > 0) ? (waveSum / float(waveCnt)) : 0.0f;
    threshold  = avgNonzero * BoilMultiplier(filterStrength);

    return (v > threshold);
}

float DLSS_LinearDepthFromWorldPos(float3 worldPos)
{
    // view is world->view. With RH projection (XMMatrixPerspectiveFovRH), forward is -Z.
    float3 viewPos = mul(view, float4(worldPos, 1.0f)).xyz;
    return max(0.0f, -viewPos.z);
}
