#define COMPUTE_PASS
// Reads the camera pass's finished G-buffer and primary extras behind barriers.
#define SPMIS_GRID_NONCOHERENT
#include "Includes_v8.hlsli"

//====================================
//PT PRIMARY-VERTEX LIGHT-TREE PREFETCH
//====================================
//The light-tree descent of the primary vertex's NEE sample (initial sample 0)
//is a serial chain of dependent node fetches. Inside Pass_pt it ran at the
//bounce kernel's occupancy (126 registers) and pulled the descent code through
//the instruction cache next to everything else that kernel executes per
//pixel. Here it runs alone, in a small kernel that hides the fetch latency,
//and the bounce kernel reads a 16-byte record instead (Path_State_v8.hlsli:
//store_pt_neePrefetch). Same seed derivation as Pass_pt (initRandomData,
//Hash32, RcBounceSeed), and the NEE stream state after the descent is stored
//with the sample, so the light sample Pass_pt produces is bit-identical.
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
    //Pass_pt's NEE gate at the primary vertex
    if (pMedium != MEDIUM_INVALID || LoadKd_w(sd.matID) < EPSILON) return;

    const uint seed     = initRandomData(pixel, uint2(8, 4), time, 1u); // initial sample 0
    const uint pathSeed = Hash32(seed ^ 0x9E3779B9u);
    uint sNee = RcBounceSeed(pathSeed, 1u, RC_STREAM_NEE);
    const LT_Sample tree = LT_SampleLight(sd.x1, sd.n1_s, sNee);
    store_pt_neePrefetch(g_pathStateBuffer, pixelIdx, tree.id, tree.pdf, sNee);
}
