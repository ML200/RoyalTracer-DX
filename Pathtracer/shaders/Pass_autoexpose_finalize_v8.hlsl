#define COMPUTE_PASS
#include "Includes_v8.hlsli"

//====================================
//AUTO EXPOSURE FINALIZE
//====================================
//Single-thread dispatch (fx:1). Reads per-frame accumulator, computes mean
//log-luminance, smooths exponentially toward it, writes back, clears the
//accumulator slots.
//
//Time constant ~2 s assuming ~60 fps -> alpha ~0.0083. Scene-rate independent
//is a stretch goal; for now the lerp factor below is a constant per-frame
//step that produces eye-like adaptation in the 1-3 s range across 30-120 fps.

static const float AE_KEY_VALUE        = 0.12f;  // darker than 0.18 mid-grey, matches the postprocess key
static const float AE_LOG_LUM_MIN      = -4.67f; // max boost  ~ key * exp(4.67) ≈ 19x  (1/3 of the [-6,+4] widening)
static const float AE_LOG_LUM_MAX      =  3.33f; // max cut    ~ key * exp(-3.33) ≈ 0.006x
static const float AE_ADAPT_PER_FRAME  =  0.04f; // ~2 s @ 60 fps to converge to within 1/e

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
        //first frame after creation: snap to current, no temporal lag
        smoothed = currentLogLum;
        gAutoExpose.Store(AE_OFFS_INIT, 1u);
    } else {
        const float prev = asfloat(gAutoExpose.Load(AE_OFFS_SMOOTHED));
        smoothed = lerp(prev, currentLogLum, AE_ADAPT_PER_FRAME);
    }

    gAutoExpose.Store(AE_OFFS_SMOOTHED,   asuint(smoothed));
    //clear accumulators for next frame's reduce
    gAutoExpose.Store(AE_OFFS_SUM,        0u);
    gAutoExpose.Store(AE_OFFS_TILE_COUNT, 0u);
}
