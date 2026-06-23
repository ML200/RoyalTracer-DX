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
[numthreads(16, 16, 1)]
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

    const uint4 agg = g_spmisBuffer.Load4(SP_AGG(cell));   // PIXCNT|NZ|CONF|OFF
    const uint base = agg.w;                               // OFF
    const uint idx  = g_spmisBuffer.Load(SP_A(SP_IDX, pixelIdx));
    const uint nz   = agg.y;                               // NZ
    const uint pos  = important ? (base + idx) : (base + nz + idx);

    g_spmisBuffer.Store(SP_A(SP_SORTED, pos), pixelIdx);

    //Precompute the inner-RIS target (UCW*target*M) into a parallel dense array so the
    //reuse kernel reads ONE float per candidate instead of load_W + load_F + load_M
    //across two SoA planes (the inner RIS samples this risN*Ntilde times per pixel).
    //Only the non-zero front block [base, base+nz) is ever sampled, and that is exactly
    //the important slots written here, so no separate reset of SP_SORTEDW is needed.
    if (important)
    {
        const float tf = W * GetPHat(load_F(g_Reservoirs_current, pixelIdx))
                           * (float)load_M(g_Reservoirs_current, pixelIdx);
        g_spmisBuffer.Store(SP_A(SP_SORTEDW, pos), asuint(tf));
    }

    //Search-record fast path for Pass_spmis_select: bundle this cell's confidence sum and
    //this pixel's world position into one AoS record (npx-indexed) so each cell-search
    //probe is a single Load4 instead of a scattered SP_CONF[ncell] + neighbour-G-buffer
    //fan-out. cell==UNDEF pixels returned above; the select hash check rejects them before
    //ever reading the record, so they need no entry (hence no separate reset).
    const float  cellConf = (float)agg.z;                  // CONF (from the Load4 above)
    const float3 wpos     = load_x1(g_sample_current, pixelIdx);
    g_spmisBuffer.Store4(SP_SRCH(pixelIdx), uint4(asuint(cellConf), asuint(wpos)));
}
