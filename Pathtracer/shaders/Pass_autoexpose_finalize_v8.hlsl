#include "Includes_v8.hlsli"

static const float AE_KEY_VALUE   =  0.18f;

static const float AE_LOG_LUM_MIN = -6.0f;
static const float AE_LOG_LUM_MAX =  3.0f;
static const float AE_ADAPT_TAU   =  0.3f;
static const float AE_DT_MIN      =  0.001f;
static const float AE_DT_MAX      =  0.25f;

[numthreads(1, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    if (any(DTid != uint3(0, 0, 0))) return;

    const uint sumFixed  = gAutoExpose.Load(AE_OFFS_SUM);
    const uint tileCount = gAutoExpose.Load(AE_OFFS_TILE_COUNT);
    const uint isInit    = gAutoExpose.Load(AE_OFFS_INIT);

    float currentLogLum = 0.0f;
    if (tileCount > 0u) {
        const float meanFixed = float(sumFixed) / float(tileCount);
        currentLogLum = (meanFixed / AE_LOG_SCALE) - AE_LOG_OFFSET;
    }
    currentLogLum = clamp(currentLogLum, AE_LOG_LUM_MIN, AE_LOG_LUM_MAX);

    float smoothed;
    if (isInit == 0u) {

        smoothed = currentLogLum;
        gAutoExpose.Store(AE_OFFS_INIT, 1u);
    } else {
        const float prev     = asfloat(gAutoExpose.Load(AE_OFFS_SMOOTHED));
        const float prevTime = asfloat(gAutoExpose.Load(AE_OFFS_PREV_TIME));

        const float dt       = clamp(walltime - prevTime, AE_DT_MIN, AE_DT_MAX);
        const float alpha    = 1.0f - exp(-dt / AE_ADAPT_TAU);
        smoothed = lerp(prev, currentLogLum, alpha);
    }

    gAutoExpose.Store(AE_OFFS_SMOOTHED,   asuint(smoothed));
    gAutoExpose.Store(AE_OFFS_PREV_TIME,  asuint(walltime));

    gAutoExpose.Store(AE_OFFS_SUM,        0u);
    gAutoExpose.Store(AE_OFFS_TILE_COUNT, 0u);
}
