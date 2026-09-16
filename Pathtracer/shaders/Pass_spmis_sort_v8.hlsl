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

    const float W         = load_W(g_Reservoirs_current, pixelIdx);
    const bool  important = (W > 0.0f);

    const uint4 agg = g_spmisBuffer.Load4(SP_AGG(cell));
    const uint base = agg.w;
    const uint idx  = g_spmisBuffer.Load(SP_A(SP_IDX, pixelIdx));
    const uint nz   = agg.y;
    const uint pos  = important ? (base + idx) : (base + nz + idx);

    g_spmisBuffer.Store(SP_A(SP_SORTED, pos), pixelIdx);

    if (important)
    {
        const float tf = W * GetPHat(load_F(g_Reservoirs_current, pixelIdx))
                           * (float)load_M(g_Reservoirs_current, pixelIdx);
        g_spmisBuffer.Store(SP_A(SP_SORTEDW, pos), asuint(tf));
    }

    const float  cellConf = (float)agg.z;
    const uint   inst     = load_instID(g_sample_current, pixelIdx);
    const float3 wpos     = load_x1_with_instID  (g_sample_current, pixelIdx, inst);
    const float3 wnrm     = load_n1_s_with_instID(g_sample_current, pixelIdx, inst);
    g_spmisBuffer.Store4(SP_SRCH(pixelIdx),       uint4(cell, asuint(wpos)));
    g_spmisBuffer.Store4(SP_SRCH(pixelIdx) + 16u, uint4(asuint(cellConf), asuint(wnrm)));
}
