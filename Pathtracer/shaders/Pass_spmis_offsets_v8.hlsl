#define COMPUTE_PASS
#include "Includes_v8.hlsli"

//====================================
//SPMIS COMPUTE OFFSETS
//====================================
//Prefix-sum the per-cell pixel counts into contiguous start offsets via a single
//global atomic counter. Dispatched over the DENSE cell index space [0, SP_NUMCELLS)
//(tid.y*W+tid.x), so every possible hash cell index is visited exactly once. The
//ordering of cells in the packed array is nondeterministic (atomic race) but that is
//irrelevant - it only packs each cell's members contiguously.
[numthreads(8, 4, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    if (!SPMIS_SPATIAL_MODE) return;
    if (tid.x >= IMG_W || tid.y >= IMG_H) return;

    const uint cell = tid.y * IMG_W + tid.x;              // [0, SP_NUMCELLS)
    const uint cnt  = g_spmisBuffer.Load(SP_A(SP_PIXCNT, cell));

    uint offset;
    g_spmisBuffer.InterlockedAdd(SP_CTR(), cnt, offset);
    g_spmisBuffer.Store(SP_A(SP_OFF, cell), offset);
}
