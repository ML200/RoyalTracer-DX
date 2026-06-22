        #define COMPUTE_PASS
#include "Includes_v8.hlsli"

//====================================
//SPMIS SORT  (with the important-first split)
//====================================
//Scatter each pixel into pixel_indices_sorted at its cell's offset + its within-class
//index. Important (UCW>0) reservoirs land in [offset, offset+nz); the rest follow in
//[offset+nz, offset+pixCount). This gives the reuse kernel's inner RIS a contiguous
//front block of non-zero reservoirs to sample uniformly. The STORED value is the
//pixel's MapPixelID reservoir index (what the reuse kernel loads).
[numthreads(8, 4, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    if (!SPMIS_SPATIAL_MODE) return;
    if (tid.x >= IMG_W || tid.y >= IMG_H) return;

    const float2 dims     = float2(IMG_W, IMG_H);
    const uint   pixelIdx = MapPixelID(dims, tid.xy);
    const uint   cell     = g_spmisBuffer.Load(SP_A(SP_HASH, pixelIdx));
    if (cell == SP_UNDEF) return;

    const float W         = load_W(g_Reservoirs_current, pixelIdx);
    const bool  important = (W > 0.0f);

    const uint base = g_spmisBuffer.Load(SP_A(SP_OFF, cell));
    const uint idx  = g_spmisBuffer.Load(SP_A(SP_IDX, pixelIdx));
    const uint nz   = g_spmisBuffer.Load(SP_A(SP_NZ,  cell));
    const uint pos  = important ? (base + idx) : (base + nz + idx);

    g_spmisBuffer.Store(SP_A(SP_SORTED, pos), pixelIdx);
}
