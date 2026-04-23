#include "Includes_v8.hlsli"

//====================================================================
//FINALIZE STAGE
//====================================================================
//Commits the RIS wsum / W / M fields into the reservoir. Bit-identical
//to the "FINAL RESOLVE" block at the end of the monolithic Pass_raygen.
//wsum lives in PathState HOT2 during the bounce loop (stages wrote to
//it via store_ps_wsum) and gets transferred here.

[shader("raygeneration")]
void Pass_finalize_v8()
{
    const uint2 pixel    = DispatchRaysIndex().xy;
    const uint2 imgSize  = DispatchRaysDimensions().xy;
    const uint  pixelIdx = MapPixelID(imgSize, pixel);

    const float wsum = load_wsum_ps(g_pathStateBuffer, pixelIdx);

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
