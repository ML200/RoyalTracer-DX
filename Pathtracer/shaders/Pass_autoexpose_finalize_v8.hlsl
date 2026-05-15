#define COMPUTE_PASS
#include "Includes_v8.hlsli"

//====================================
//AUTO EXPOSURE FINALIZE
//====================================
//Single-thread dispatch (fx:1). Reads per-frame accumulator, computes mean
//log2-luminance, smooths exponentially toward it with a framerate-independent
//alpha derived from CameraParams.walltime, writes back, clears the accumulator.
//
//Time constant AE_ADAPT_TAU controls how fast the eye adapts (seconds).
//~1 s feels responsive without flickering; bump to 2-3 s for cinematic feel.
//Alpha = 1 - exp(-dt/tau) so the response is identical regardless of fps.

static const float AE_KEY_VALUE   =  0.18f;  // photographic mid-grey, matches postprocess key
//Limiting the boost is what keeps dark scenes looking dark. The previous
//AE_LOG_LUM_MIN = -9 let the AE multiply a mean luminance of 0.002 by ~92x —
//moonless night came out looking like a sunny afternoon. Eye adaptation isn't
//perfect either: a real human in a dark room still perceives it as dim, not
//"normal exposure." With MIN = -3 the maximum boost is ~1.4x, so anything
//darker than ~lum 0.125 reads progressively darker on screen.
static const float AE_LOG_LUM_MIN = -3.0f;   // max boost  ~ key * 2^3   = 1.44x  (preserves "feels dark")
static const float AE_LOG_LUM_MAX =  3.0f;   // max cut    ~ key * 2^-3  = 0.022x (sun-disc scenes still clip nicely)
static const float AE_ADAPT_TAU   =  0.3f;   // seconds to reach 1-1/e of new target; ~1 s to converge fully
static const float AE_DT_MIN      =  0.001f; // guard against hitches and clock anomalies
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
        //first frame after creation: snap to current, no temporal lag
        smoothed = currentLogLum;
        gAutoExpose.Store(AE_OFFS_INIT, 1u);
    } else {
        const float prev     = asfloat(gAutoExpose.Load(AE_OFFS_SMOOTHED));
        const float prevTime = asfloat(gAutoExpose.Load(AE_OFFS_PREV_TIME));
        //walltime is host-accumulated dt in seconds, gives true framerate
        //independence (time is a frame counter and unsuitable for this)
        const float dt       = clamp(walltime - prevTime, AE_DT_MIN, AE_DT_MAX);
        const float alpha    = 1.0f - exp(-dt / AE_ADAPT_TAU);
        smoothed = lerp(prev, currentLogLum, alpha);
    }

    gAutoExpose.Store(AE_OFFS_SMOOTHED,   asuint(smoothed));
    gAutoExpose.Store(AE_OFFS_PREV_TIME,  asuint(walltime));
    //clear accumulators for next frame's reduce
    gAutoExpose.Store(AE_OFFS_SUM,        0u);
    gAutoExpose.Store(AE_OFFS_TILE_COUNT, 0u);
}
