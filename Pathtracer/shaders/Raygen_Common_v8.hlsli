//====================================
//RAYGEN COMMON
//====================================
//Shared between Pass_camera_v8 (primary hit) and Pass_raygen_v8 (bounce loop)
//after the camera ray was split into its own pass. Include AFTER Includes_v8.hlsli.

#ifndef RAYGEN_COMMON_V8_HLSLI
#define RAYGEN_COMMON_V8_HLSLI

//Per-vertex surface state carried across the bounce loop (loop phi). Bounded
//fields are half to shrink the carried state. Pass_camera fills it from the
//primary hit; Pass_raygen rebuilds it from the G-buffer per sample.
struct HitContext {
    float3 hitPos;
    float3 hitNormal;
    uint   matID;
    uint   instID;
    bool   backface;
    half3  hitLocalKd;      //albedo
    half   hitLocalPr;      //roughness
    half   hitLocalPm;      //metalness
    half2  iors;
    uint   mediumMatID;
    half3  absorptionTint;
};


//Finalize the pixel reservoir: W = wsum / GetPHat(F) (0 on degenerate/NaN), M=1
//(the N samples improve sample quality, not confidence — the 1/N is in wsum, so
//w_sum = W*M*p_hat holds at M=1). Invalidate on W==0. wsum is REGISTER-ONLY
//state: nothing ever re-read the old WSUM plane, so it was removed outright.
inline void FinalizeReservoir(uint pixelIdx, float wsum)
{
    const float F_mag = GetPHat(load_F(g_Reservoirs_current, pixelIdx));
    float W = (F_mag > 1e-6f && wsum > 0.0f) ? (wsum / F_mag) : 0.0f;
    if (isnan(W) || isinf(W) || W < 0.0f) W = 0.0f;
    store_W(g_Reservoirs_current, pixelIdx, W);
    store_M(g_Reservoirs_current, pixelIdx, 1u);
    if (W == 0.0f)
        InvalidateReservoir_ShadingNormal(g_Reservoirs_current, pixelIdx);
}

#endif // RAYGEN_COMMON_V8_HLSLI
