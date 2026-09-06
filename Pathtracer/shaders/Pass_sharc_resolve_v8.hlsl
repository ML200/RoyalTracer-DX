#ifndef SHARC_TEST
#define COMPUTE_PASS
#include "Includes_v8.hlsli"
#endif
#include "Sharc_v8.hlsli"

[numthreads(SHARC_GROUP_SIZE, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= SHARC_CAPACITY) return;
    uint state = g_sharc.Load(SharcStateAddress(tid.x));
    if (state == 0u || state == SHARC_LOCKED) return;
    uint e = SharcEntryAddress(tid.x);
    uint4 sumRG = g_sharc.Load4(e + SHARC_FRAME_RGB);
    uint4 sumBL = g_sharc.Load4(e + SHARC_FRAME_RGB + 16u);
    uint4 sumW = g_sharc.Load4(e + SHARC_FRAME_W_W2); // W, W2, positive, spare
    // Most live records receive nothing in a given frame; leave them untouched.
    if (all(sumRG == 0u) && all(sumBL == 0u) && all(sumW.xyz == 0u)) return;
    const float radianceScale = rcp(SHARC_RADIANCE_SCALE);
    const float weightScale = rcp(SHARC_WEIGHT_SCALE);
    float4 frame = float4(SharcFloat(sumRG.xy, radianceScale), SharcFloat(sumRG.zw, radianceScale),
        SharcFloat(sumBL.xy, radianceScale), (float)sumW.x * weightScale);
    float3 moments = float3(SharcFloat(sumBL.zw, radianceScale),
        (float)sumW.y * weightScale, (float)sumW.z * weightScale);
    if (frame.w > 0.0f)
    {
        float4 previous = float4(asfloat(g_sharc.Load3(e + SHARC_MEAN)), asfloat(g_sharc.Load(e + SHARC_HISTORY_W)));
        float3 oldMoments = float3(asfloat(g_sharc.Load(e + SHARC_HISTORY_L2)),
            asfloat(g_sharc.Load(e + SHARC_HISTORY_W2)), asfloat(g_sharc.Load(e + SHARC_HISTORY_POSITIVE)));
        bool validHistory = all(isfinite(previous)) && all(isfinite(oldMoments)) &&
            previous.w > 0.0f && oldMoments.y > 0.0f;
        if (!validHistory)
        {
            previous = 0.0f;
            oldMoments = 0.0f;
            g_sharc.Store(e + SHARC_FRAMES, 0u);
        }
        // Decay per OBSERVATION FRAME, not per elapsed wall frame. With one
        // visit per 60 frames, the old policy capped effective history at ~2
        // samples forever, below the confidence threshold no matter how long
        // the camera waited. Age eviction and scene-change resets remain active.
        float decay = 1.0f - rcp(max((float)sharc_historyFrames, 2.0f));
        float oldWeight = previous.w * decay;
        float weight = oldWeight + frame.w;
        float oldFraction = oldWeight / weight;
        float3 mean = previous.xyz * oldFraction + frame.xyz / weight;
        float second = oldMoments.x * oldFraction + moments.x / weight;
        float weight2 = oldMoments.y * decay * decay + moments.y;
        float positives = oldMoments.z * decay + moments.z;
        if (all(isfinite(float4(mean, weight))) && all(isfinite(float3(second, weight2, positives))))
        {
            g_sharc.Store3(e + SHARC_MEAN, asuint(mean));
            g_sharc.Store4(e + SHARC_HISTORY_W, asuint(float4(weight, second, weight2, positives)));
            g_sharc.Store(e + SHARC_FRAMES, min(g_sharc.Load(e + SHARC_FRAMES) + 1u, 65535u));
            float gap = (float)(sharc_frame - g_sharc.Load(e + SHARC_LAST_UPDATE));
            float interval = validHistory ? asfloat(g_sharc.Load(e + SHARC_INTERVAL)) : 1.0f;
            if (!isfinite(interval)) interval = 1.0f;
            interval = validHistory ? max(lerp(interval, gap, 0.125f), 0.5f * gap) : 1.0f;
            g_sharc.Store(e + SHARC_INTERVAL, asuint(interval));
            g_sharc.Store(e + SHARC_LAST_UPDATE, sharc_frame);
            // Compute statistical confidence once, rather than at every query.
            g_sharc.Store(e + SHARC_CONFIDENCE, asuint(SharcStatisticalConfidence(e)));
        }
    }
    // Even invalid sums must not poison subsequent frames. A subsequent valid
    // observation replaces corrupt history instead of lerping through NaN * 0.
    g_sharc.Store4(e + SHARC_FRAME_RGB, 0u);
    g_sharc.Store4(e + SHARC_FRAME_RGB + 16u, 0u);
    g_sharc.Store4(e + SHARC_FRAME_W_W2, 0u);
}
