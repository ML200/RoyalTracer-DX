#include "Includes_v8.hlsli"
#include "SharcGuide_v8.hlsli"

bool SharcInstanceMoved(uint instance)
{
    if (instance == 0xffffffffu) return false;
    const float3x4 now = instanceProps[instance].objectToWorld;
    const float3x4 was = instanceProps[instance].prevObjectToWorld;
    return any(now[0] != was[0]) || any(now[1] != was[1]) || any(now[2] != was[2]);
}

[numthreads(SHARC_GROUP_SIZE, 1, 1)]
// Evict stale or moved entries and reset invalidated history.
void main(uint3 tid : SV_DispatchThreadID)
{
    const uint instance = SharcPrepareIndex(tid.x);
    if (instance != SHARC_INVALID && SharcInstanceMoved(instance))
        SharcEvictEntry(tid.x);
}
