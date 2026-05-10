#define COMPUTE_PASS
#include "Includes_v8.hlsli"

//====================================
//AUTO EXPOSURE REDUCTION
//====================================
//8x8 thread groups, each averages its tile's log-luminance and atomically
//adds the (fixed-point) tile mean to gAutoExpose. Finalize pass converts
//the accumulator back to a float and smooths it.
//
//We sum per-group means (not per-pixel sums) so the accumulator stays inside
//uint32 even at 4K: tilesAt4K * AE_LOG_SCALE * (clampMax + AE_LOG_OFFSET)
//= 32400 * 10 * 20 = 6.5M, well below 4.3e9.
//Finalize divides by tileCount instead of pixelCount to undo this.

#define AE_GROUP_SIZE 64u  // 8x8

groupshared float g_tileLL[AE_GROUP_SIZE];
groupshared uint  g_tileMask[AE_GROUP_SIZE];

float Luminance(float3 c) {
    return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

[numthreads(8, 8, 1)]
void main(uint3 DTid : SV_DispatchThreadID, uint GIdx : SV_GroupIndex)
{
    //per-thread log-luminance + valid-pixel mask written to groupshared
    if (DTid.x < gImageWidth && DTid.y < gImageHeight) {
        const float3 col = g_dlssOutput[DTid.xy].xyz;
        const float  lum = max(Luminance(col), 1e-6f);
        g_tileLL[GIdx]   = clamp(log(lum), -AE_LOG_OFFSET, AE_LOG_OFFSET);
        g_tileMask[GIdx] = 1u;
    } else {
        g_tileLL[GIdx]   = 0.0f;
        g_tileMask[GIdx] = 0u;
    }
    GroupMemoryBarrierWithGroupSync();

    //thread 0 sums + atomically adds the tile mean. With 8x8 a serial sum is
    //fine (64 adds), groupshared reduction would only save ~5 cycles per group
    if (GIdx == 0u) {
        float sumLL = 0.0f;
        uint  cnt   = 0u;
        [unroll] for (uint i = 0u; i < AE_GROUP_SIZE; ++i) {
            sumLL += g_tileLL[i];
            cnt   += g_tileMask[i];
        }
        if (cnt > 0u) {
            const float tileMean = sumLL / float(cnt);
            const uint  packed   = (uint)((tileMean + AE_LOG_OFFSET) * AE_LOG_SCALE);
            gAutoExpose.InterlockedAdd(AE_OFFS_SUM, packed);
            gAutoExpose.InterlockedAdd(AE_OFFS_TILE_COUNT, 1u);
        }
    }
}
