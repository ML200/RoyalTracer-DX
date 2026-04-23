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
    if (!load_isEmitter(g_sample_current, pixelIdx))
    {
        const uint primID = load_primID(g_sample_current, pixelIdx);
        if (primID != 0xFFFFFFFFu)
        {
            // Reflectance factorisation: MLP predicts irradiance, recover
            // radiance by multiplying by (α+β) at the query vertex.
            const uint    instID = load_instID(g_sample_current, pixelIdx);
            const uint    matID  = GetMatIDFast(instID, primID);
            const float2  uv     = load_uv(g_sample_current, pixelIdx);
            float3 kd; float pr, pm;
            RefetchMaterial(matID, uv, kd, pr, pm, 0u);
            const float3 alpha   = kd * (1.0f - pm);
            const float3 betaC   = lerp(float3(0.04f, 0.04f, 0.04f), kd, pm);
            const float3 reflSum = alpha + betaC;

            L_s = NrcLoadInferenceOutput(pixelIdx) * reflSum;
        }
    }

    gOutput[uint3(pixel, 3)] = float4(L_s, 1.0f);
}
