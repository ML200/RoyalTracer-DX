//====================================
//NRC RESOLVE AND RESERVOIR FINALIZE
//====================================
//per pending cache-term pixel, merge cache candidate into reservoir via RIS
//pixels without pending record already finalized in raygen

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

    uint  slot;
    uint  throughputPk;
    uint  tpostPk;
    float pdfProduct;
    NrcLoadPendingGI(pixelIdx, slot, throughputPk, tpostPk, pdfProduct);
    if (slot == NRC_INVALID_SLOT) return;

    const float3 L_s        = NrcLoadInferenceOutput(slot);
    const float3 throughput = UnpackRGB9E5(throughputPk);
    const float3 tpost      = UnpackRGB9E5(tpostPk);

    //L2 and F match raygen's emitter-hit convention
    const float3 L2        = L_s * tpost;
    const float3 F_contrib = throughput * L_s * pdfProduct;

    //no MIS, cache covers all radiance at xk
    const float p_full = pdfProduct;
    const float p_hat  = GetPHat(F_contrib);
    const float wi     = (p_full > 1e-20f) ? (p_hat / p_full) : 0.0f;

    float wsum = load_wsum(g_Reservoirs_current, pixelIdx);
    uint  seed = initRandomData(pixel, uint2(11, 17), time, 5u);

    const PathVertexState ps = load_ps(g_pathStateBuffer, pixelIdx);

    AddInitialCandidate(
        wsum, g_Reservoirs_current, pixelIdx, wi,
        ps.x2, ps.n2_s,
        L2,    ps.v2,
        ps.uv,
        ps.matID, ps.objID, ps.eta,
        F_contrib, seed);

    //finalize mirrors raygen's old final-resolve block
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
