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
[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    if (!SPMIS_SPATIAL_MODE) return;
    if (tid.x >= IMG_W || tid.y >= IMG_H) return;

    const uint cell = tid.y * IMG_W + tid.x;              // [0, SP_NUMCELLS)
    const uint cnt  = g_spmisBuffer.Load(SP_PIXCNT_A(cell));

    //Skip empty cells. The dispatch covers every cell in [0,SP_NUMCELLS)=IMG_W*IMG_H,
    //but cells are 32px screen tiles so the vast majority are unpopulated — without this
    //guard every one of ~W*H threads fans into the SINGLE global counter SP_CTR(),
    //fully serializing ~millions of same-address atomics (>99% of them adding 0). An
    //empty cell's OFF is never read (sort returns on cell==SP_UNDEF before touching it),
    //so leaving it at the reset-cleared 0 is safe.
    if (cnt == 0u) return;

    uint offset;
    g_spmisBuffer.InterlockedAdd(SP_CTR(), cnt, offset);
    g_spmisBuffer.Store(SP_OFF_A(cell), offset);
}
