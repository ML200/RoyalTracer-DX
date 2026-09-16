#define COMPUTE_PASS
#include "Includes_v8.hlsli"

[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    if (!SPMIS_SPATIAL_MODE) return;
    if (tid.x >= IMG_W || tid.y >= IMG_H) return;

    const uint cell = tid.y * IMG_W + tid.x;
    const uint cnt  = g_spmisBuffer.Load(SP_PIXCNT_A(cell));

    if (cnt == 0u) return;

    uint offset;
    g_spmisBuffer.InterlockedAdd(SP_CTR(), cnt, offset);
    g_spmisBuffer.Store(SP_OFF_A(cell), offset);
}
