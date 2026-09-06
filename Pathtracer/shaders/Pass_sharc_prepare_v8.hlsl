#ifndef SHARC_TEST
#define COMPUTE_PASS
#include "Includes_v8.hlsli"
#endif
#include "SharcGuide_v8.hlsli"

[numthreads(SHARC_GROUP_SIZE, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
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
    }
    if (evict)
    {
        g_sharc.Store(stateAddress, 0u);
        [unroll] for (uint b = 0u; b < SHARC_ENTRY_BYTES; b += 16u)
            g_sharc.Store4(e + b, 0u);
    }
}
