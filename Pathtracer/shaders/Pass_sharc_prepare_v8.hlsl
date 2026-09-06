#ifndef SHARC_TEST
#define COMPUTE_PASS
#include "Includes_v8.hlsli"
#endif
#include "SharcGuide_v8.hlsli"

#ifndef SHARC_TEST
// An instance whose transform changed since the previous frame (the instance
// table carries both). Its records describe surfaces that are no longer
// where the record says, so they are evicted here; static instances keep
// converging while others move (the host used to reset the whole cache).
// Floating-origin rebases rewrite both transforms together.
bool SharcInstanceMoved(uint instance)
{
    if (instance == 0xffffffffu) return false;
    const float3x4 now = instanceProps[instance].objectToWorld;
    const float3x4 was = instanceProps[instance].prevObjectToWorld;
    return any(now[0] != was[0]) || any(now[1] != was[1]) || any(now[2] != was[2]);
}
#endif

[numthreads(SHARC_GROUP_SIZE, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    if (sharc_reset != 0u && tid.x < SHARC_DIRTY_WORDS)
        g_sharc.Store(SharcDirtyAddress(tid.x), 0u);
    // The guide table is smaller than the cache; its first threads maintain
    // one receiver entry each before their cache entry (no extra dispatch).
    if (tid.x < GUIDE_CAPACITY) GuidePrepareEntry(tid.x);
    if (tid.x >= SHARC_CAPACITY) return;
    uint stateAddress = SharcStateAddress(tid.x);
    uint e = SharcEntryAddress(tid.x);
    bool evict = sharc_reset != 0u;
    if (!evict && g_sharc.Load(stateAddress) != 0u)
    {
        uint4 key = g_sharc.Load4(e + SHARC_NODE); // node.xyz, meta
        SharcHistory h = SharcLoadHistory(e);
        uint level = key.w & 31u;
        float3 position = SharcNodeLocal(asint(key.xyz), level);
        // Keep a generous LOD band while walking; obsolete detail and stale
        // regions are reclaimed without relocating or aliasing live world cells.
        float desired = SharcLevel(position);
        evict = (float)(sharc_frame - h.lastUpdate) > SharcAgeLimit(h) ||
            abs((float)level - desired) > 3.0f;
#ifndef SHARC_TEST
        if (!evict) evict = SharcInstanceMoved(g_sharc.Load(e + SHARC_NODE + 16u));
#endif
    }
    if (evict)
    {
        // Empty slots are never read, and allocation initializes every field
        // before publishing the key. Reset only the compact state table.
        g_sharc.Store(stateAddress, 0u);
    }
}
