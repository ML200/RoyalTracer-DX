#ifndef SHARC_TEST
#define COMPUTE_PASS
#include "Includes_v8.hlsli"
#endif
#include "SharcGuide_v8.hlsli"

void SharcResolveEntry(uint slot)
{
    uint state = g_sharc.Load(SharcStateAddress(slot));
    if (state == 0u || state == SHARC_LOCKED) return;
    uint e = SharcEntryAddress(slot);
    uint4 sumRG = g_sharc.Load4(e + SHARC_FRAME_RGB);
    uint4 sumBL = g_sharc.Load4(e + SHARC_FRAME_RGB + 16u);
    uint4 sumW = g_sharc.Load4(e + SHARC_FRAME_W_W2);

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
        // Corrupt history must not contaminate the next frame.
        if (!validHistory)
        {
            previous = 0.0f;
            oldMoments = 0.0f;
            g_sharc.Store(e + SHARC_FRAMES, 0u);
        }

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
            const uint frames = min(g_sharc.Load(e + SHARC_FRAMES) + 1u, 65535u);
            g_sharc.Store(e + SHARC_FRAMES, frames);
            float gap = (float)(sharc_frame - g_sharc.Load(e + SHARC_LAST_UPDATE));
            float interval = validHistory ? asfloat(g_sharc.Load(e + SHARC_INTERVAL)) : 1.0f;
            if (!isfinite(interval)) interval = 1.0f;
            interval = validHistory ? max(lerp(interval, gap, 0.125f), 0.5f * gap) : 1.0f;
            g_sharc.Store(e + SHARC_INTERVAL, asuint(interval));
            g_sharc.Store(e + SHARC_LAST_UPDATE, sharc_frame);

            const float confidence = SharcStatisticalConfidence(e);
            g_sharc.Store(e + SHARC_CONFIDENCE, asuint(confidence));

            SharcPublishQueryHistory(e, mean, confidence, frames, interval);
        }
    }

    g_sharc.Store4(e + SHARC_FRAME_RGB, 0u);
    g_sharc.Store4(e + SHARC_FRAME_RGB + 16u, 0u);
    g_sharc.Store4(e + SHARC_FRAME_W_W2, 0u);
}

groupshared uint sharcDirtyMask[SHARC_GROUP_SIZE / 32u];

[numthreads(SHARC_GROUP_SIZE, 1, 1)]
// Resolve dirty radiance and guide entries into reusable history.
void main(uint3 group : SV_GroupID, uint lane : SV_GroupIndex)
{
    uint baseSlot = group.x * SHARC_GROUP_SIZE;
    if (baseSlot >= SHARC_CAPACITY) return;

    if (baseSlot < GUIDE_CAPACITY)
    {
        const uint guideSlot = baseSlot + lane;
        if ((g_sharc.Load(GuideDirtyAddress(guideSlot >> 5u)) & (1u << (guideSlot & 31u))) != 0u)
            GuideResolveEntry(guideSlot);
    }

    if (lane < SHARC_GROUP_SIZE / 32u)
    {
        uint address = SharcDirtyAddress((baseSlot >> 5u) + lane);
        uint dirty = g_sharc.Load(address);
        sharcDirtyMask[lane] = dirty;

        if (dirty != 0u) g_sharc.Store(address, 0u);
    }
    GroupMemoryBarrierWithGroupSync();
    if ((sharcDirtyMask[lane >> 5u] & (1u << (lane & 31u))) != 0u)
        SharcResolveEntry(baseSlot + lane);
}
