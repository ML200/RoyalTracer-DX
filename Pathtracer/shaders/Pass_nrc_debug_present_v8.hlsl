//====================================
//NRC DEBUG VIEW PRESENT
//====================================
//writes cache prediction at x1 into scratchPing slot 9, postprocess reads it from
//there; gOutput is R8G8B8A8 UNORM and would clip the HDR L_s before tonemap

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
        //MLP predicts irradiance, recover radiance via alpha+beta. Material
        //is the baked G-buffer value - no per-triangle data, no texture.
        const float3 kd = load_kd(g_sample_current, pixelIdx);
        float pr, pm;
        load_prpm(g_sample_current, pixelIdx, pr, pm);
        const float3 alpha   = kd * (1.0f - pm);
        const float3 betaC   = lerp(float3(0.04f, 0.04f, 0.04f), kd, pm);
        const float3 reflSum = alpha + betaC;

        L_s = NrcLoadInferenceOutput(pixelIdx) * reflSum;
    }

    gScratchPing[uint3(pixel, 9)] = float4(L_s, 1.0f);
}
