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
