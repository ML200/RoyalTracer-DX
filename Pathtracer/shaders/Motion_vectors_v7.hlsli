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

inline uint GetRandomPixelCircleWeighted(uint radius, uint w, uint h, uint x, uint y, inout uint2 seed)
{
    int newX, newY;
    do {
        // Get a uniform random value.
        float u = RandomFloat(seed);
        // Adjust the weighting by using a power law.
        float z = pow(u, SPAT_EXP_DI);
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