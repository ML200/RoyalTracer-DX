#ifndef RAYGEN_COMMON_V8_HLSLI
#define RAYGEN_COMMON_V8_HLSLI

struct HitContext {
    float3 hitPos;
    float3 hitNormal;
    uint   matID;
    uint   instID;
    bool   backface;
    half3  hitLocalKd;
    half   hitLocalPr;
    half   hitLocalPm;
    half2  iors;
    uint   mediumMatID;
    half3  absorptionTint;
};

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

#endif
