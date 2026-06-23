#define COMPUTE_PASS
#include "Includes_v8.hlsli"

//====================================
//SPMIS COUNT CELLS  (single-pass class-split)
//====================================
//For every pixel with a resolved hash cell, accumulate the cell aggregates:
//  pixel count       += 1          (all members)
//  confidence sum    += M
//  non-zero count    += 1          (important == UCW>0)
//and record this pixel's index WITHIN ITS CLASS (SP_NZ for important, SP_OTHER for
//the rest). The classic approach runs the count twice (important pass first) so
//non-zero reservoirs get the low indices; we get the same "non-zero-at-front" ordering
//in one pass with a per-class index allocator, resolved by the sort. Confidences are
//UNCAPPED (the M-cap is applied only to the final spatial output).
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
