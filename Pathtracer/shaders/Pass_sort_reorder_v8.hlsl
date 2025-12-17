#include "Includes_v8.hlsli"

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint activeCount = g_GlobalCounters.Load(g_InputStackIdx * 4);
    if (tid.x >= activeCount) return;

    // 1. Load Data
    uint2 data = LoadStack(g_InputStackIdx, tid.x);
    uint bucket = data.y & 0xFFFF; // 16-bit key

    // 2. WAVE ALLOCATION
    // We calculate offsets for the whole wave at once.

    // Find all peers in wave with same bucket
    uint4 matchMask = WaveMatch(bucket);

    // My local index within this group of peers
    uint laneIdx = WaveGetLaneIndex();
    uint4 lowerMask = (uint4)0;
    if (laneIdx < 32) lowerMask.x = (1u << laneIdx) - 1;
    else              lowerMask.y = (1u << (laneIdx - 32)) - 1;

    // Count how many peers are *before* me
    uint localOffset = countbits(matchMask.x & lowerMask.x) +
                       countbits(matchMask.y & lowerMask.y);

    // Am I the leader? (First one in the mask)
    bool isLeader = (localOffset == 0);

    uint baseOffset = 0;

    // Leader performs one atomic add for the whole group
    if (isLeader)
    {
        uint totalGroupSize = countbits(matchMask.x) + countbits(matchMask.y) +
                              countbits(matchMask.z) + countbits(matchMask.w);

        g_SortOffset.InterlockedAdd(bucket * 4, totalGroupSize, baseOffset);
    }

    // Broadcast the base offset from the leader to the rest of the group
    // WaveReadLaneFirst handles getting the value from the first active lane
    // in the wave that satisfies the condition, but we need the specific leader
    // for *my* bucket.

    // Since WaveReadLaneAt requires a scalar index, we use the leader's index.
    // Leader index is the first bit set in matchMask.
    uint leaderIdx = firstbitlow(matchMask.x);
    if (matchMask.x == 0) leaderIdx = firstbitlow(matchMask.y) + 32;

    baseOffset = WaveReadLaneAt(baseOffset, leaderIdx);

    // 3. Final Write Index
    uint writeIdx = baseOffset + localOffset;

    StoreStack(g_OutputStackIdx, writeIdx, data);

    // 4. Update Global Count (Thread 0 only)
    if (tid.x == 0)
    {
        g_GlobalCounters.Store(g_OutputStackIdx * 4, activeCount);
    }
}