#include "Includes_v8.hlsli"

#define AE_GROUP_SIZE 64u

groupshared float g_tileLL[AE_GROUP_SIZE];
groupshared uint  g_tileMask[AE_GROUP_SIZE];

float Luminance(float3 c) {
    return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}


[numthreads(8, 8, 1)]
// Reduce luminance samples into exposure statistics.
void main(uint3 DTid : SV_DispatchThreadID, uint GIdx : SV_GroupIndex)
{

    if (DTid.x < IMG_W && DTid.y < IMG_H) {

        const float3 dlssOut = g_dlssOutput[DTid.xy].xyz;
        float3 hdr;
            const float expNow =
                0.18f / max(exp2(asfloat(gAutoExpose.Load(AE_OFFS_SMOOTHED))), 1e-6f);
            hdr = max(dlssOut, 0.0f) / max(expNow, 1e-8f);
        const float  lum     = max(Luminance(hdr), 1e-6f);
        g_tileLL[GIdx]   = clamp(log2(lum), -AE_LOG_OFFSET, AE_LOG_OFFSET);
        g_tileMask[GIdx] = 1u;
    } else {
        g_tileLL[GIdx]   = 0.0f;
        g_tileMask[GIdx] = 0u;
    }
    GroupMemoryBarrierWithGroupSync();

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
