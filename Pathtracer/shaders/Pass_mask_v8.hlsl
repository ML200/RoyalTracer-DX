#define COMPUTE_PASS
#include "Includes_v8.hlsli"

// Approximate temporal-neighbor validity per pixel.
// Tests all 16 permutation-sampling offsets against the current G-buffer
// using the same rejection criteria as TestTemporalCandidate (normal + distance).
// Result in gOutput[pixel, 3]: 1 = all neighbors valid (flat surface),
//                               0 = no valid neighbors (corner / thin geometry).

[numthreads(16, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= IMG_W || tid.y >= IMG_H) return;

    const uint2  pixel    = tid.xy;
    const float2 dims     = float2(IMG_W, IMG_H);
    const uint   pixelIdx = MapPixelID(dims, pixel);

    // Sky / emitter: mask = 0
    if (load_isEmitter(g_sample_current, pixelIdx))
    {
        gOutput[uint3(pixel, 3)] = 0;
        return;
    }

    // Current pixel surface
    const uint   myInstID = load_instID(g_sample_current, pixelIdx);
    const uint   myPrimID = load_primID(g_sample_current, pixelIdx);
    const float2 myBary   = load_bary(g_sample_current, pixelIdx);
    const float3 myN1s    = load_n1_s_with_instID(g_sample_current, pixelIdx, myInstID);
    const float3 myPos    = ReconstructPosition(myInstID, myPrimID, myBary);

    uint validCount = 0;
    uint totalCount = 0;

    [unroll]
    for (uint p = 0; p < 16; ++p)
    {
        int2 nCoord = (int2)pixel;
        ApplyPermutationSampling(nCoord, p);

        // Out of bounds
        if (nCoord.x < 0 || nCoord.y < 0 ||
            nCoord.x >= (int)IMG_W || nCoord.y >= (int)IMG_H)
            continue;

        // Self
        if (all(nCoord == (int2)pixel))
            continue;

        totalCount++;

        uint nIdx = MapPixelID(dims, (uint2)nCoord);

        // Emitter / sky neighbor
        if (load_isEmitter(g_sample_current, nIdx))
            continue;

        // Normal similarity (same threshold as temporal reuse: 0.36)
        uint   nInstID = load_instID(g_sample_current, nIdx);
        float3 nN1s    = load_n1_s_with_instID(g_sample_current, nIdx, nInstID);
        if (dot(myN1s, nN1s) < 0.36f)
            continue;

        // Distance planarity (same threshold as temporal reuse: 0.4)
        uint   nPrimID = load_primID(g_sample_current, nIdx);
        float2 nBary   = load_bary(g_sample_current, nIdx);
        float3 nPos    = ReconstructPosition(nInstID, nPrimID, nBary);
        if (abs(dot(nPos - myPos, myN1s)) > 0.4f)
            continue;

        validCount++;
    }

    float mask = (totalCount > 0) ? (float)validCount / (float)totalCount : 0.0f;
    gOutput[uint3(pixel, 3)] = float4(mask, mask, mask, 1.0f);
}
