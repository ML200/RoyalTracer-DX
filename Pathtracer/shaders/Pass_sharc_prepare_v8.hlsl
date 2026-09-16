#ifndef SHARC_TEST
#define COMPUTE_PASS
#include "Includes_v8.hlsli"
#endif
#include "SharcGuide_v8.hlsli"

#ifndef SHARC_TEST

bool SharcInstanceMoved(uint instance)
{
    if (instance == 0xffffffffu) return false;
    const float3x4 now = instanceProps[instance].objectToWorld;
    const float3x4 was = instanceProps[instance].prevObjectToWorld;
    return any(now[0] != was[0]) || any(now[1] != was[1]) || any(now[2] != was[2]);
}
#endif

[numthreads(SHARC_GROUP_SIZE, 1, 1)]
// Evict stale or moved entries and reset invalidated history.
void main(uint3 tid : SV_DispatchThreadID)
{
    if ((sharc_reset & 1u) != 0u && tid.x < SHARC_DIRTY_WORDS)
        g_sharc.Store(SharcDirtyAddress(tid.x), 0u);
    if ((sharc_reset & 1u) != 0u && tid.x < GUIDE_DIRTY_WORDS)
        g_sharc.Store(GuideDirtyAddress(tid.x), 0u);

    if (tid.x < GUIDE_CAPACITY) GuidePrepareEntry(tid.x);
    if (tid.x >= SHARC_CAPACITY) return;
    uint stateAddress = SharcStateAddress(tid.x);
    uint e = SharcEntryAddress(tid.x);
    bool evict = (sharc_reset & 1u) != 0u;
    if (!evict && g_sharc.Load(stateAddress) != 0u)
    {
        uint4 key = g_sharc.Load4(e + SHARC_NODE);
        SharcHistory h = SharcLoadHistory(e);
        uint level = key.w & 31u;
        float3 position = SharcNodeLocal(asint(key.xyz), level);

        float desired = SharcLevel(position);
        evict = (float)(sharc_frame - h.lastUpdate) > SharcAgeLimit(h) ||
            abs((float)level - desired) > 3.0f;
#ifndef SHARC_TEST
        if (!evict) evict = SharcInstanceMoved(g_sharc.Load(e + SHARC_NODE + 16u));
#endif
    }
    if (evict)
    {

        g_sharc.Store(stateAddress, 0u);
    }
}
