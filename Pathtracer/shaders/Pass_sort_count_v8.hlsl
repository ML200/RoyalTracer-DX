#include "Includes_v8.hlsli"

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint activeCount = g_GlobalCounters.Load(g_InputStackIdx * 4);
    if (tid.x >= activeCount) return;

    // 1. Load Data
    uint2 data = LoadStack(g_InputStackIdx, tid.x);

    // 2. Get Bucket (The Key IS the bucket index now, 0..32767)
    uint bucket = data.y & 0xFFFF; // Mask just to be safe

    // 3. WAVE OPTIMIZATION
    // Count how many threads in this wave share the exact same bucket.
    // This turns 64 atomic adds into ~1-5 atomic adds on average.

    // Get mask of lanes where bucket matches the current lane's bucket
    uint4 matchMask = WaveMatch(bucket);

    // Count bits in the mask
    uint count = countbits(matchMask.x) + countbits(matchMask.y) +
                 countbits(matchMask.z) + countbits(matchMask.w);

    // Find the first lane in the mask (the "Leader")
    // We use bit scan logic or WaveReadLaneFirst logic.
    // Simple robust method:
    uint laneIdx = WaveGetLaneIndex();

    // Identify the lowest lane index that shares this bucket
    // (This works by masking out higher bits and finding the first set bit)
    // NOTE: For simplicity, checking if I am the first lane *in the match mask*:

    // Create a mask of all lanes LOWER than current
    uint4 lowerMask = (uint4)0;
    if (laneIdx < 32) lowerMask.x = (1u << laneIdx) - 1;
    else              lowerMask.y = (1u << (laneIdx - 32)) - 1;
    // (Assuming Wave64. If Wave32, only use .x)

    // If no lanes lower than me have this bucket, I am the leader.
    bool isLeader = ((matchMask.x & lowerMask.x) == 0) &&
                    ((matchMask.y & lowerMask.y) == 0);

    if (isLeader)
    {
        g_SortCount.InterlockedAdd(bucket * 4, count);
    }
}