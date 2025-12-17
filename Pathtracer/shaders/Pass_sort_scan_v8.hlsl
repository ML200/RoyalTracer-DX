#include "Includes_v8.hlsli"

// Config
#define THREADS 1024
#define VECTORS_PER_THREAD 16 // 16 * 4 uints = 64 items
#define ITEMS_PER_THREAD 64

// Shared memory only needs to store sums for each Wave.
// Max waves = 1024 threads / 32 (min wave size) = 32 waves.
// We allocate 64 just to be safe for weird architectures.
groupshared uint s_WaveSums[64];

[numthreads(THREADS, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID, uint groupIndex : SV_GroupIndex)
{
    // ------------------------------------------------------------------------
    // 1. CALCULATE LOCAL SUM
    // ------------------------------------------------------------------------
    // Load 64 items (16 vectors) and calculate the total sum for this thread.
    // We do NOT store the data in registers to avoid spilling.

    uint baseAddr = groupIndex * ITEMS_PER_THREAD * 4; // Bytes
    uint threadSum = 0;

    [unroll]
    for(uint i = 0; i < VECTORS_PER_THREAD; ++i)
    {
        uint4 v = g_SortCount.Load4(baseAddr + (i * 16));
        threadSum += (v.x + v.y + v.z + v.w);
    }

    // ------------------------------------------------------------------------
    // 2. WAVE & GROUP SCAN (The Magic)
    // ------------------------------------------------------------------------
    // Instead of a tree reduction with 20 barriers, we use Wave Intrinsics.

    // A. Scan inside the Wave
    // 'waveOffset' is the sum of 'threadSum' from lanes with lower indices.
    uint waveOffset = WavePrefixSum(threadSum);

    // B. Aggregate Wave Sums to Shared Memory
    // The last active lane in the wave holds the total sum of the wave.
    uint waveTotal = WaveActiveSum(threadSum);

    // Get the index of the wave within the group
    uint waveCount = THREADS / WaveGetLaneCount();
    uint waveIndex = groupIndex / WaveGetLaneCount();

    // Only the first lane of each wave writes to shared memory
    if (WaveIsFirstLane())
    {
        s_WaveSums[waveIndex] = waveTotal;
    }

    // BARRIER 1: Wait for all waves to write their totals
    GroupMemoryBarrierWithGroupSync();

    // C. Scan the Wave Sums (Performed by a single wave/thread)
    // Since we only have ~32 waves, the first wave can scan them all instantly.
    uint groupBaseOffset = 0;

    if (groupIndex < waveCount)
    {
        // Simple serial scan of the wave sums
        // (Fast enough because waveCount is small, usually 16-32)
        uint myWaveSum = s_WaveSums[groupIndex];

        // We need an exclusive scan of s_WaveSums.
        // There isn't a "WavePrefixSum" for shared memory arrays,
        // so we just read, scan locally, and write back.
        // Actually, we can just use WavePrefixSum again on the loaded values!

        uint wavescan_val = WavePrefixSum(myWaveSum);
        s_WaveSums[groupIndex] = wavescan_val;
    }

    // BARRIER 2: Wait for the shared memory scan to complete
    GroupMemoryBarrierWithGroupSync();

    // ------------------------------------------------------------------------
    // 3. GENERATE FINAL OFFSETS & WRITE
    // ------------------------------------------------------------------------

    // My final global offset starts at:
    // [Sum of previous waves] + [Sum of previous threads in my wave]
    uint myGlobalOffset = s_WaveSums[waveIndex] + waveOffset;

    // Re-read data, compute internal prefix sums, and write
    uint runningOffset = myGlobalOffset;

    [unroll]
    for(uint j = 0; j < VECTORS_PER_THREAD; ++j)
    {
        uint readAddr = baseAddr + (j * 16);

        // Re-read from L2 (Fast)
        uint4 counts = g_SortCount.Load4(readAddr);

        // Compute offsets for the 4 components
        uint4 offsets;
        offsets.x = runningOffset;
        offsets.y = runningOffset + counts.x;
        offsets.z = offsets.y     + counts.y;
        offsets.w = offsets.z     + counts.z;

        // Advance
        runningOffset += (counts.x + counts.y + counts.z + counts.w);

        // Write Offsets
        g_SortOffset.Store4(readAddr, offsets);

        // Clear Counts (Coalesced write)
        g_SortCount.Store4(readAddr, uint4(0, 0, 0, 0));
    }
}