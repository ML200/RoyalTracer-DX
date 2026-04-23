//====================================================================
// Pass_nrc_debug_present_v8 — copy the cache's prediction at x1 into
// gOutput slice 3 for editor inspection.
//
// Runs AFTER cuda:nrc_inference (which fills g_NrcInferenceOut[pixel])
// and AFTER Pass_nrc_resolve_v8 (which is a no-op in debug mode). The
// value written matches paper Fig. 4's "Visualization at first
// non-specular vertex".
//====================================================================

#define COMPUTE_PASS
#include "Includes_v8.hlsli"
#include "Nrc_v8.hlsli"

[numthreads(8, 8, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    if (dtid.x >= IMG_W || dtid.y >= IMG_H) return;
    if (!NrcIsDebugView()) return;

    const uint2 pixel    = dtid.xy;
    const uint  pixelIdx = MapPixelID(gImageSize, pixel);
    if (pixelIdx == 0xFFFFFFFFu) return;

    float3 L_s = float3(0, 0, 0);
    if (!load_isEmitter(g_sample_current, pixelIdx)) {
        L_s = NrcLoadInferenceOutput(pixelIdx);
    }

    gOutput[uint3(pixel, 3)] = float4(L_s, 1.0f);
}
