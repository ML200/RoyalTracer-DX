//====================================================================
// Pass_nrc_resolve_v8 — NRC cache resolution + reservoir finalization.
//
// Runs between the CUDA inference op and the ReSTIR temporal pass.
// For each pixel with a pending cache-terminated GI candidate:
//   * read L̂_s from the inference-output buffer
//   * reconstruct F = throughput · L̂_s · pdf_product and
//     L2      = L̂_s · tpost
//   * feed it through AddInitialCandidate so it competes via RIS with
//     whatever DI/emitter candidate raygen already locked in
//   * finalize W = wsum / p̂(F), store M=1, invalidate on W=0
// Pixels without a pending record (slot == NRC_INVALID_SLOT) were
// already finalized inside raygen and are left untouched here.
//====================================================================

#define COMPUTE_PASS
#include "Includes_v8.hlsli"
#include "Nrc_v8.hlsli"

[numthreads(8, 8, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    if (dtid.x >= IMG_W || dtid.y >= IMG_H) return;

    // In debug view mode raygen finalizes W for every pixel (cache or
    // not), and the debug view shows the raw prediction on slice 3 —
    // we deliberately skip the cache→reservoir stitch here so slice 0
    // stays pure ReSTIR PT for A/B comparison.
    if (NrcIsDebugView()) return;

    const uint2 pixel    = dtid.xy;
    const uint  pixelIdx = MapPixelID(gImageSize, pixel);
    if (pixelIdx == 0xFFFFFFFFu) return;

    uint  slot;
    uint  throughputPk;
    uint  tpostPk;
    float pdfProduct;
    NrcLoadPendingGI(pixelIdx, slot, throughputPk, tpostPk, pdfProduct);
    if (slot == NRC_INVALID_SLOT) return;

    // Cache-terminated pixel: build the GI candidate.
    const float3 L_s        = NrcLoadInferenceOutput(slot);
    const float3 throughput = UnpackRGB9E5(throughputPk);
    const float3 tpost      = UnpackRGB9E5(tpostPk);

    // L2 follows the emitter-hit convention in raygen (line ~322):
    //   L2 = radiance at x2 from direction V2 = L̂_s · tpost.
    // F_contrib follows the same shape as an emitter hit:
    //   F  = throughput · L̂_s · pdf_product.
    const float3 L2        = L_s * tpost;
    const float3 F_contrib = throughput * L_s * pdfProduct;

    // No MIS factor: the cache represents all radiance at xk, there's
    // no competing light-sampling strategy for it.
    const float p_full = pdfProduct;
    const float p_hat  = GetPHat(F_contrib);
    const float wi     = (p_full > 1e-20f) ? (p_hat / p_full) : 0.0f;

    // Pull raygen's running wsum, merge with the cache candidate via
    // the standard RIS update, then commit.
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

    // Finalize — mirrors raygen's old final-resolve block.
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
