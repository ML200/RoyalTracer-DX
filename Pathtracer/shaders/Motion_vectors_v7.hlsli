// Swizzling
inline uint MapPixelID(uint2 dims, uint2 lIndex)
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
}
