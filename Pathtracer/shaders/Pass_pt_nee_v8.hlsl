#define COMPUTE_PASS

#define SPMIS_GRID_NONCOHERENT
#include "Includes_v8.hlsli"

[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    gDispatchIdx = tid;
    const uint2 imgSize = uint2(IMG_W, IMG_H);
    if (any(tid.xy >= imgSize)) return;
    const uint2 pixel    = tid.xy;
    const uint  pixelIdx = MapPixelID(imgSize, pixel);

    if (load_flagsWord(g_sample_current, pixelIdx) & SD_FLAG_NOBOUNCE) return;
    const SDRecord sd = load_SD(g_sample_current, pixelIdx);
    float2 pIors; uint pMedium; float3 pAbsorb;
    load_rg_primaryExtra(g_pathStateBuffer, pixelIdx, pIors, pMedium, pAbsorb);

    if (pMedium != MEDIUM_INVALID || LoadKd_w(sd.matID) < EPSILON) return;

    const uint seed     = initRandomData(pixel, uint2(8, 4), time, 1u);
    const uint pathSeed = Hash32(seed ^ 0x9E3779B9u);
    uint sNee = RcBounceSeed(pathSeed, 1u, RC_STREAM_NEE);
    const bool useLearnedLights=LTC_UseSurfaceLearning();
    const LT_Sample tree = LT_SampleLight(sd.x1, sd.n1_s, sNee, useLearnedLights);
    store_pt_neePrefetch(g_pathStateBuffer, pixelIdx, tree.id, tree.inst, tree.pdf, sNee, tree.learningToken);
}
