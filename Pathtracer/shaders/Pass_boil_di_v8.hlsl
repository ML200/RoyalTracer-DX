#include "Includes_v8.hlsli"

//─────────────────────────────────────────────────────────────────────────────
//  BOILING FILTER  DI  (groupshared post-pass for temporal DI)
//─────────────────────────────────────────────────────────────────────────────
[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID, uint2 localIdx : SV_GroupThreadID)
{
    const bool oob = (tid.x >= IMG_W || tid.y >= IMG_H);
    const uint2  launchIndex = tid.xy;
    const uint   pixelIdx = oob ? 0 : MapPixelID(float2(IMG_W, IMG_H), launchIndex);

    float boilValue = oob ? 0.0f : gScratchPing[uint3(launchIndex, 0)].x;

    float avgV, thrV;
    bool boil = BoilingFilter(localIdx, DI_BOIL_STRENGTH_TEMP, boilValue, avgV, thrV);

    if (avgV < DI_BOIL_MIN_AVG_TEMP)
        boil = false;

    if (!oob && boil && boilValue > 0.0f)
    {
        float W    = load_W_di(g_Reservoirs_current_di, pixelIdx);
        float wsum = load_wsum_di(g_Reservoirs_current_di, pixelIdx);
        float scale = thrV / boilValue;
        store_W_di(g_Reservoirs_current_di, pixelIdx, W * scale);
        store_wsum_di(g_Reservoirs_current_di, pixelIdx, wsum * scale);
        store_M_di(g_Reservoirs_current_di, pixelIdx, min(load_M_di(g_Reservoirs_current_di, pixelIdx), 1u));
    }
}
