#include "Includes_v8.hlsli"
#include "SharcGuide_v8.hlsli"

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
