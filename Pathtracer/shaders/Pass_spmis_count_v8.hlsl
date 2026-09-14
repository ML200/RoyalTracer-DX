#define COMPUTE_PASS
#include "Includes_v8.hlsli"

[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    if (!SPMIS_SPATIAL_MODE) return;
    if (tid.x >= IMG_W || tid.y >= IMG_H) return;

    const float2 dims     = float2(IMG_W, IMG_H);
    const uint   pixelIdx = MapPixelID(dims, tid.xy);
    const uint   cell     = g_spmisBuffer.Load(SP_A(SP_HASH, pixelIdx));
    if (cell == SP_UNDEF) return;

    const uint  M         = load_M(g_Reservoirs_current, pixelIdx);
    const float W         = load_W(g_Reservoirs_current, pixelIdx);
    const bool  important = (W > 0.0f);

    uint dummy, idx;
    g_spmisBuffer.InterlockedAdd(SP_PIXCNT_A(cell), 1u, dummy);
    g_spmisBuffer.InterlockedAdd(SP_CONF_A(cell),   M,  dummy);
    if (important) g_spmisBuffer.InterlockedAdd(SP_NZ_A(cell), 1u, idx);
    else           g_spmisBuffer.InterlockedAdd(SP_A(SP_OTHER, cell), 1u, idx);
    g_spmisBuffer.Store(SP_A(SP_IDX, pixelIdx), idx);
}
