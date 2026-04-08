/*
V8 common functions
*/

// Estimated luminance
inline float Luma(float3 c) { return dot(c, float3(0.2126, 0.7152, 0.0722)); }
// Average
inline float Avg3(float3 c) { return dot(c, float3(0.33333f, 0.33333f, 0.33333f)); }

// Swizzle for thread group
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
    // Check for the sentinel value returned by MapPixelID when out of bounds
    if (pixelID == 0xFFFFFFFF)
    {
        return int2(-1, -1);
    }

    const uint tileWidth  = 4;
    const uint tileHeight = 8;
    const uint tileSize   = tileWidth * tileHeight; // 32

    // 1. Separate the Global Tile Index from the Local Index within the tile
    uint tileIndex  = pixelID / tileSize;
    uint localIndex = pixelID % tileSize;

    // 2. Calculate the grid dimensions (how many tiles wide)
    uint tileCountX = (dims.x + tileWidth - 1u) / tileWidth;

    // 3. Resolve the 2D position of the Tile (TileX, TileY)
    uint tileY = tileIndex / tileCountX;
    uint tileX = tileIndex % tileCountX;

    // 4. Resolve the 2D position within the Tile (LocalX, LocalY)
    uint localY = localIndex / tileWidth;
    uint localX = localIndex % tileWidth;

    // 5. Combine to get Global Coordinates
    uint globalX = tileX * tileWidth + localX;
    uint globalY = tileY * tileHeight + localY;

    // 6. Final Bounds Check
    // Because tiles are padded to 4x8, a valid pixelID might map to a coordinate
    // that is technically outside the original image dimensions (padding area).
    if (globalX >= dims.x || globalY >= dims.y)
    {
        return int2(-1, -1);
    }

    return int2(globalX, globalY);
}

// --- Helper function to Push to Stack ---
void PushStack(uint stackIdx, uint2 val)
{
    uint outSlot;
    // Atomic Increment: We still add 1 item.
    // The counter tracks the count, not the byte size.
    g_GlobalCounters.InterlockedAdd(stackIdx * 4, 1, outSlot);

    if (stackIdx == 0)      g_Stack0[outSlot] = val;
    else if (stackIdx == 1) g_Stack1[outSlot] = val;
    // Add branches for stacks 2/3 if needed
}

// --- Helper function to Pop from Stack ---
uint2 PopStack(uint stackIdx, uint threadIdx)
{
    if (stackIdx == 0)      return g_Stack0[threadIdx];
    else if (stackIdx == 1) return g_Stack1[threadIdx];
    return uint2(0, 0);
}

uint2 LoadStack(uint stackIdx, uint elementIdx)
{
    // Compiler will optimize this branch out because stackIdx is a uniform constant
    if (stackIdx == 0) return g_Stack0[elementIdx];
    return g_Stack1[elementIdx];
}

void StoreStack(uint stackIdx, uint elementIdx, uint2 val)
{
    if (stackIdx == 0) g_Stack0[elementIdx] = val;
    else g_Stack1[elementIdx] = val;
}


inline uint FloatToOrderedUint(float v)
{
    uint bits = asuint(v);
    uint mask = (bits & 0x80000000u) ? 0xFFFFFFFFu : 0x80000000u;
    return bits ^ mask;
}

inline float OrderedUintToFloat(uint v)
{
    uint mask = (v & 0x80000000u) ? 0x80000000u : 0xFFFFFFFFu;
    return asfloat(v ^ mask);
}

inline void UpdateOriginBounds(float3 origin)
{
    uint ox = FloatToOrderedUint(origin.x);
    uint oy = FloatToOrderedUint(origin.y);
    uint oz = FloatToOrderedUint(origin.z);

    g_SortBounds.InterlockedMin(0,  ox);
    g_SortBounds.InterlockedMin(4,  oy);
    g_SortBounds.InterlockedMin(8,  oz);

    g_SortBounds.InterlockedMax(12, ox);
    g_SortBounds.InterlockedMax(16, oy);
    g_SortBounds.InterlockedMax(20, oz);
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

    // Hack for specular reflectance of 0 (optional; keep if you want same behavior)
    bias *= saturate(SpecularColor.g * 50);

    return mad(SpecularColor, max(0, scale), max(0, bias));
}



#define BOIL_GROUP_X 16
#define BOIL_GROUP_Y 16
#define BOIL_THREAD_COUNT (BOIL_GROUP_X * BOIL_GROUP_Y)
#define BOIL_MAX_WAVES 8

groupshared float gBoilSum[BOIL_MAX_WAVES];
groupshared uint  gBoilCnt[BOIL_MAX_WAVES];

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
    float waveSum = WaveActiveSum(v);
    uint  waveCnt = WaveActiveCountBits(v > 0.0f);

    uint linearThreadIndex = localIndex.x + localIndex.y * BOIL_GROUP_X;
    uint waveIndex         = linearThreadIndex / WaveGetLaneCount();

    if (WaveIsFirstLane())
    {
        if (waveIndex < BOIL_MAX_WAVES)
        {
            gBoilSum[waveIndex] = waveSum;
            gBoilCnt[waveIndex] = waveCnt;
        }
    }

    GroupMemoryBarrierWithGroupSync();

    uint numWaves = (BOIL_THREAD_COUNT + WaveGetLaneCount() - 1) / WaveGetLaneCount();
    numWaves = min(numWaves, (uint)BOIL_MAX_WAVES);

    if (linearThreadIndex < numWaves)
    {
        float s = gBoilSum[linearThreadIndex];
        uint  c = gBoilCnt[linearThreadIndex];

        s = WaveActiveSum(s);
        c = WaveActiveSum(c);

        if (linearThreadIndex == 0)
            gBoilSum[0] = (c > 0) ? (s / float(c)) : 0.0f;
    }

    GroupMemoryBarrierWithGroupSync();

    avgNonzero = gBoilSum[0];
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