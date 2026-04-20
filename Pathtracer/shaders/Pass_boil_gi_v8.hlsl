#define COMPUTE_PASS
#include "Includes_v8.hlsli"

//  BOILING FILTER
[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID, uint2 localIdx : SV_GroupThreadID)
{
    const bool oob = (tid.x >= IMG_W || tid.y >= IMG_H);
    const uint2  launchIndex = tid.xy;
    const uint   pixelIdx = oob ? 0 : MapPixelID(float2(IMG_W, IMG_H), launchIndex);

    float boilValue = oob ? 0.0f : gScratchPing[uint3(launchIndex, 5)].x;

    float avgV, thrV;
    bool boil = BoilingFilter(localIdx, BOIL_STRENGTH_TEMP, boilValue, avgV, thrV);

    if (avgV < BOIL_MIN_AVG_TEMP)
        boil = false;

    if (!oob && boil && boilValue > 0.0f)
    {
        float W   = load_W(g_Reservoirs_current, pixelIdx);
        float F   = GetPHat(load_F(g_Reservoirs_current, pixelIdx));
        float scale = thrV / boilValue;
        W *= scale;
        store_W(g_Reservoirs_current, pixelIdx, W);
        store_wsum(g_Reservoirs_current, pixelIdx, W * max(F, EPSILON));
        store_M(g_Reservoirs_current, pixelIdx, min(load_M(g_Reservoirs_current, pixelIdx), 1u));
    }
}
