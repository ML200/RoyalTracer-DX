#include "Includes_v8.hlsli"

// UNCOMMENT THIS to debug the Sort Keys!
// #define SHOW_KEYS 1

[numthreads(16, 16, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    if (dtid.x >= IMG_W || dtid.y >= IMG_H) return;

    // SPREAD OUT: Map buffer linearly across the whole screen
    uint bufferIndex = dtid.y * IMG_W + dtid.x;

    uint activeCount = g_GlobalCounters.Load(g_InputStackIdx * 4);

    // 1. Draw Inactive/Dead rays as Dark Red
    if (bufferIndex >= activeCount) {
        gOutput[uint3(dtid.xy, 0)] = float4(0.1, 0, 0, 1);
        return;
    }

    uint rayPixelIdx = LoadStack(g_InputStackIdx, bufferIndex);

    // =========================================================
    // MODE A: SHOW SORT KEYS (ENABLE THIS TO DEBUG)
    // =========================================================
#ifdef SHOW_KEYS
    // Copy the EXACT logic from your Sort Shader here to verify it
    PathState p = loadPathState(g_pathStateBuffer, rayPixelIdx);

    // 1. Direction Key (3 bits)
    uint dx = (p.s.x > 0.f) ? 1 : 0;
    uint dy = (p.s.y > 0.f) ? 1 : 0;
    uint dz = (p.s.z > 0.f) ? 1 : 0;
    uint dirKey = (dx) | (dy << 1) | (dz << 2);

    // Output Direction Key as Grayscale
    // If the screen is one solid gray color, your keys are broken.
    // You should see chunky blocks of 8 different shades of gray.
    float val = (float)dirKey / 7.0;
    gOutput[uint3(dtid.xy, 0)] = float4(val, val, val, 1);
    return;
#endif

    // =========================================================
    // MODE B: SHOW MEMORY LAYOUT (DEFAULT)
    // =========================================================
    uint2 rayCoord = UnmapPixelID(rayPixelIdx, float2(IMG_W, IMG_H));

    // Red = Screen X, Green = Screen Y
    float r = (float)rayCoord.x / (float)IMG_W;
    float g = (float)rayCoord.y / (float)IMG_H;

    gOutput[uint3(dtid.xy, 0)] = float4(r, g, 0.0, 1.0);
}