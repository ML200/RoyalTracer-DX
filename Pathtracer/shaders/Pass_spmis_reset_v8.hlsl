#define COMPUTE_PASS
#include "Includes_v8.hlsli"

[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    if (!SPMIS_SPATIAL_MODE) return;
    if (tid.x >= IMG_W || tid.y >= IMG_H) return;

    const float2 dims      = float2(IMG_W, IMG_H);
    const uint   denseCell = tid.y * IMG_W + tid.x;
    const uint   pixelIdx  = MapPixelID(dims, tid.xy);

    g_spmisBuffer.Store(SP_A(SP_CHK,    denseCell), SP_UNDEF);
    g_spmisBuffer.Store4(SP_AGG(denseCell), uint4(0u, 0u, 0u, 0u));
    g_spmisBuffer.Store(SP_A(SP_OTHER,  denseCell), 0u);
    g_spmisBuffer.Store(SP_A(SP_SORTED, denseCell), SP_UNDEF);

    if (pixelIdx != SP_UNDEF)
    {
        g_spmisBuffer.Store(SP_A(SP_HASH, pixelIdx), SP_UNDEF);
        g_spmisBuffer.Store(SP_A(SP_IDX,  pixelIdx), 0u);

        g_spmisBuffer.Store(SP_SRCH(pixelIdx), SP_UNDEF);
    }

    if (denseCell == 0u)
        g_spmisBuffer.Store(SP_CTR(), 0u);
}
