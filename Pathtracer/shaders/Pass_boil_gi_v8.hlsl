#define COMPUTE_PASS
#include "Includes_v8.hlsli"

//─────────────────────────────────────────────────────────────────────────────
//  BOILING FILTER  GI  (groupshared post-pass for raygen temporal GI)
//─────────────────────────────────────────────────────────────────────────────
[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID, uint2 localIdx : SV_GroupThreadID)
{
    const bool oob = (tid.x >= IMG_W || tid.y >= IMG_H);
    const uint2  launchIndex = tid.xy;
    const uint   pixelIdx = oob ? 0 : MapPixelID(float2(IMG_W, IMG_H), launchIndex);

    float boilValue = oob ? 0.0f : gScratchPing[uint3(launchIndex, 5)].x;

    float avgV, thrV;
    bool boil = BoilingFilter(localIdx, GI_BOIL_STRENGTH_TEMP, boilValue, avgV, thrV);

    if (avgV < GI_BOIL_MIN_AVG_TEMP)
        boil = false;

    if (!oob && boil && boilValue > 0.0f)
    {
        float W   = load_W_gi(g_Reservoirs_current_gi, pixelIdx);
        float F   = load_F_gi(g_Reservoirs_current_gi, pixelIdx);
        float scale = thrV / boilValue;
        W *= scale;
        store_W_gi(g_Reservoirs_current_gi, pixelIdx, W);
        store_wsum_gi(g_Reservoirs_current_gi, pixelIdx, W * max(F, EPSILON));
        store_M_gi(g_Reservoirs_current_gi, pixelIdx, min(load_M_gi(g_Reservoirs_current_gi, pixelIdx), 1u));
    }
}
