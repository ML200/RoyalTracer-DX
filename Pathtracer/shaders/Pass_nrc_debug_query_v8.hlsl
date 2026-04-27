//====================================
//NRC DEBUG QUERY AT PRIMARY HIT
//====================================
//writes W*H inference requests for the debug view, runs after main inference consumed

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

    //sky and emitter primaries, leave slot untouched
    if (load_isEmitter(g_sample_current, pixelIdx)) return;

    const uint instID = load_instID(g_sample_current, pixelIdx);
    const uint primID = load_primID(g_sample_current, pixelIdx);
    if (primID == 0xFFFFFFFFu) return;

    const float2 bary = load_bary (g_sample_current, pixelIdx);
    const float3 n1_s = load_n1_s_with_instID(g_sample_current, pixelIdx, instID);
    const float2 uv   = load_uv  (g_sample_current, pixelIdx);

    const float3 x1    = ReconstructPosition(instID, primID, bary);
    const uint   matID = GetMatIDFast(instID, primID);

    float3 localKd;
    float  localPr, localPm;
    RefetchMaterial(matID, uv, localKd, localPr, localPm, 0u);

    //outgoing direction from x1 to camera
    const float3 camPos  = viewI[3].xyz;
    const float3 viewDir = normalize(camPos - x1);

    //reflectance factorisation matches raygen's cache-term
    const float3 alpha = localKd * (1.0f - localPm);
    const float3 betaC = lerp(float3(0.04f, 0.04f, 0.04f), localKd, localPm);

    //load the primary hit's backface flag so the debug query passes the
    //same side bit the main raygen passes for cache training/inference
    const bool backface = load_backface(g_sample_current, pixelIdx);

    float features[17];
    NrcBuildFeatures(x1, viewDir, n1_s, localPr, alpha, betaC, backface, features);

    //deterministic per-pixel slot, safe to reuse since main inference already consumed
    const uint base = pixelIdx * NRC_INFERENCE_IN_STRIDE;
    [unroll]
    for (uint i = 0; i < NRC_RAW_INPUT_DIM; ++i) {
        g_NrcInferenceIn.Store(base + i * 4u, asuint(features[i]));
    }
}
