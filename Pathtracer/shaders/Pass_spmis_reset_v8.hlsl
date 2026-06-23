#define COMPUTE_PASS
#include "Includes_v8.hlsli"

//====================================
//SPMIS RESET  (fused buffer reset + per-cell data reset)
//====================================
//Clears the whole SPMIS hash grid for the frame. Runs BEFORE raygen (which inserts
//each pixel's hash) and before the count/offsets/sort pipeline (which fills the cell
//aggregates), so one reset covers both the hash table and the per-cell arrays.
//Per-cell arrays + the open-addressed checksum table are indexed by the DENSE hash
//cell index (tid.y*W+tid.x, covering [0, SP_NUMCELLS)); per-pixel arrays by MapPixelID.
[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    if (!SPMIS_SPATIAL_MODE) return;
    if (tid.x >= IMG_W || tid.y >= IMG_H) return;

    const float2 dims      = float2(IMG_W, IMG_H);
    const uint   denseCell = tid.y * IMG_W + tid.x;        // [0, SP_NUMCELLS)
    const uint   pixelIdx  = MapPixelID(dims, tid.xy);     // per-pixel slot

    g_spmisBuffer.Store(SP_A(SP_CHK,    denseCell), SP_UNDEF);
    g_spmisBuffer.Store4(SP_AGG(denseCell), uint4(0u, 0u, 0u, 0u));   // PIXCNT|NZ|CONF|OFF
    g_spmisBuffer.Store(SP_A(SP_OTHER,  denseCell), 0u);
    g_spmisBuffer.Store(SP_A(SP_SORTED, denseCell), SP_UNDEF);

    if (pixelIdx != SP_UNDEF)
    {
        g_spmisBuffer.Store(SP_A(SP_HASH, pixelIdx), SP_UNDEF);
        g_spmisBuffer.Store(SP_A(SP_IDX,  pixelIdx), 0u);
    }

    if (denseCell == 0u)
        g_spmisBuffer.Store(SP_CTR(), 0u);                 // global offset counter
}
