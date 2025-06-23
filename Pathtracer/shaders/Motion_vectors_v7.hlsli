// Safe 2-D→1-D swizzle; returns 0xFFFFFFFF on invalid input
inline uint MapPixelID(uint2 dims, int2 lIndex)
{
    // ---------- 1. validate input ----------
    // negatives or out-of-range ──► sentinel
    if (lIndex.x < 0 || lIndex.y < 0 ||
        lIndex.x >= int(dims.x) || lIndex.y >= int(dims.y))
    {
        return uint(-1);          // invalid
    }

    // ---------- 2. original mapping ----------
    const uint tileWidth  = 4;
    const uint tileHeight = 8;

    uint2 uIndex   = uint2(lIndex);               // now safe to cast
    uint tileCountX = (dims.x + tileWidth - 1u) / tileWidth;

    uint tileX = uIndex.x / tileWidth;
    uint tileY = uIndex.y / tileHeight;

    uint localX = uIndex.x % tileWidth;
    uint localY = uIndex.y % tileHeight;

    uint tileIndex  = tileY * tileCountX + tileX;
    uint localIndex = localY * tileWidth + localX;

    return tileIndex * (tileWidth * tileHeight) + localIndex;
}