//====================================================================
// Pass_nrc_debug_query_v8 — per-pixel cache inference query at x1.
//
// Purely additive: raygen's normal cache-termination + tail-seed
// inference runs first (resolve stitches it into the reservoir,
// train consumes the class-1 tail predictions). Only AFTER that
// does this pass write a W×H batch of camera-hit queries into the
// inference buffers for cuda:nrc_debug_inference to run, feeding
// Pass_nrc_debug_present_v8 on slice 3.
//
// Because the main inference's results have already been consumed
// by the time we reach this pass, it's safe to reuse the shared
// g_NrcInferenceIn / g_NrcInferenceOut buffers — the second
// inference call just overwrites them with per-pixel predictions.
//
// Features come from the primary-hit data raygen stashed into
// g_sample_current. Sky / emitter pixels skip the write; the
// present pass checks isEmitter too and leaves them at 0.
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

    // Sky / emitter primaries: no meaningful cache query — leave the
    // slot untouched. The present pass overrides these pixels to black.
    if (load_isEmitter(g_sample_current, pixelIdx)) return;

    const uint instID = load_instID(g_sample_current, pixelIdx);
    const uint primID = load_primID(g_sample_current, pixelIdx);
    if (primID == 0xFFFFFFFFu) return;   // primary miss

    const float2 bary = load_bary (g_sample_current, pixelIdx);
    const float3 n1_s = load_n1_s_with_instID(g_sample_current, pixelIdx, instID);
    const float2 uv   = load_uv  (g_sample_current, pixelIdx);

    const float3 x1    = ReconstructPosition(instID, primID, bary);
    const uint   matID = GetMatIDFast(instID, primID);

    float3 localKd;
    float  localPr, localPm;
    RefetchMaterial(matID, uv, localKd, localPr, localPm, 0u);

    // Outgoing direction ω for L̂_s(x1, ω) = direction from x1 toward
    // the camera.
    const float3 camPos  = viewI[3].xyz;
    const float3 viewDir = normalize(camPos - x1);

    // Reflectance factorisation (same conventions as raygen's cache-term).
    const float3 alpha = localKd * (1.0f - localPm);
    const float3 betaC = lerp(float3(0.04f, 0.04f, 0.04f), localKd, localPm);

    float features[14];
    NrcBuildFeatures(x1, viewDir, n1_s, localPr, alpha, betaC, features);

    // Deterministic per-pixel slot. Safe to reuse g_NrcInferenceIn
    // because cuda:nrc_inference + resolve + train have all already
    // consumed the main pass's contents by now.
    const uint base = pixelIdx * NRC_INFERENCE_IN_STRIDE;
    [unroll]
    for (uint i = 0; i < 14u; ++i) {
        g_NrcInferenceIn.Store(base + i * 4u, asuint(features[i]));
    }
}
