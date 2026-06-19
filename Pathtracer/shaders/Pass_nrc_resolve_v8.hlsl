//====================================
//NRC RESOLVE AND RESERVOIR FINALIZE
//====================================
//merges cache candidate into reservoir via RIS, also resolves x1 sharp reflection slot

#define COMPUTE_PASS
#include "Includes_v8.hlsli"
#include "Nrc_v8.hlsli"

[numthreads(8, 8, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    if (dtid.x >= IMG_W || dtid.y >= IMG_H) return;

    const uint2 pixel    = dtid.xy;
    const uint  pixelIdx = MapPixelID(gImageSize, pixel);
    if (pixelIdx == 0xFFFFFFFFu) return;

    //====================================
    //X1 SHARP REFLECTION RESOLVE
    //====================================
    //collapses Fresnel, NRC slot, env miss radiance into one RGB contribution at slot 8
    {
        const float4 reflPack = gScratchPing[uint3(pixel, 7)];
        const float3 fresnelP = reflPack.rgb;
        const uint   reflSlot = asuint(reflPack.w);
        float3 reflRGB = float3(0, 0, 0);
        float  reflW   = 0.0f;
        if (any(fresnelP > 0.0f))
        {
            float3 reflRad;
            if (reflSlot != NRC_INVALID_SLOT)
            {
                reflRad = NrcLoadInferenceOutput(reflSlot);
            }
            else
            {
                //raygen stored env miss radiance in slot 8.rgb, .w=1 marker
                const float4 envPack = gScratchPing[uint3(pixel, 8)];
                reflRad = (envPack.w > 0.0f) ? envPack.rgb : float3(0, 0, 0);
            }
            reflRGB = NrcCleanRadiance(fresnelP * reflRad);
            reflW   = 1.0f;
        }
        gScratchPing[uint3(pixel, 8)] = float4(reflRGB, reflW);
    }

    uint  slot;
    uint  throughputPk;
    uint  tpostPk;
    float pdfProduct;
    NrcLoadPendingGI(pixelIdx, slot, throughputPk, tpostPk, pdfProduct);
    if (slot == NRC_INVALID_SLOT) return;

    const float3 L_s        = NrcLoadInferenceOutput(slot);
    const float3 throughput = UnpackRGB9E5(throughputPk);
    const float3 tpost      = UnpackRGB9E5(tpostPk);

    //L2 and F match raygen's emitter hit convention
    const float3 L2        = L_s * tpost;

    //no MIS, cache covers all radiance at xk. This is the SOLE reservoir
    //candidate for a cache-terminated pixel: raygen breaks the bounce loop at
    //the cache fire and DEFERS W/M/F finalization here, so for a shadowed pixel
    //wsum enters as 0 and the cache is the only thing that can populate the
    //reservoir. F_contrib is pdf-scaled to match raygen's emitter convention,
    //but a tiny deep-path pdfProduct drives GetPHat(F_contrib) below
    //AddInitialCandidate's 1e-20 reject gate -> the SOLE candidate is dropped ->
    //wsum stays 0 -> W=0 -> InvalidateReservoir -> the pixel loses its only GI
    //source and renders as a BLACK screen tile. Because the cache fires at the
    //stochastic depth>=3 vertex (and clusters on coherent geometry) the black
    //set is screen-tile-aligned and varies per frame, over an otherwise-clean
    //image. FLOOR pdfProduct so it can't underflow the gate: the floor cancels
    //exactly in F*W (= throughput*L_s) so it is unbiased, it keeps the pdf-scaled
    //convention consistent with raygen's DI candidates (reuse-safe), and it caps
    //the reuse UCW (W=1/p_full) against fireflies. The gate now only drops
    //genuinely-zero radiance, not a valid-but-deep cache sample.
    const float  p_full    = max(pdfProduct, 1e-6f);
    const float3 F_contrib = throughput * L_s * p_full;
    const float  p_hat     = GetPHat(F_contrib);
    const float  wi        = p_hat / p_full;

    float wsum = load_wsum(g_Reservoirs_current, pixelIdx);
    uint  seed = initRandomData(pixel, uint2(11, 17), time, 5u);

    const PathVertexState ps = load_ps(g_pathStateBuffer, pixelIdx);

    AddInitialCandidate(
        wsum, g_Reservoirs_current, pixelIdx, wi,
        ps.x2, ps.n2_s,
        L2,    ps.v2,
        ps.Kd, ps.Pr, ps.Pm,
        ps.matID, ps.objID, ps.eta,
        F_contrib, seed);

    //finalize mirrors raygen's final resolve
    const float F_mag = GetPHat(load_F(g_Reservoirs_current, pixelIdx));
    float W = 0.0f;
    if (F_mag > 1e-6f && wsum > 0.0f)
    {
        W = wsum / F_mag;
        if (isnan(W) || isinf(W) || W < 0.0f) W = 0.0f;
    }

    store_wsum(g_Reservoirs_current, pixelIdx, wsum);
    store_W   (g_Reservoirs_current, pixelIdx, W);
    store_M   (g_Reservoirs_current, pixelIdx, 1u);
    if (W == 0.0f)
        InvalidateReservoir_ShadingNormal(g_Reservoirs_current, pixelIdx);
}
