#define COMPUTE_PASS
#define SPMIS_GRID_NONCOHERENT
#include "Includes_v8.hlsli"
#include "RestirLite_v8.hlsli"

//====================================
//DUPLICATION MAP (correlation reduction)
//====================================
// Counts, in a 17x17 window of the previous frame's final reservoirs, how
// many pixels hold the same sample as the centre. The temporal pass
// collapses that history's confidence toward one where the count is high,
// so a single sample cannot keep spreading through reuse (the deprecated
// pipeline's Pass_dup_gi, keyed to the lite sample identity).
static const uint TILE_W = 16u;
static const uint TILE_H = 16u;
static const uint WIN_R = 8u;
static const uint CACHE_W = TILE_W + 2u * WIN_R;
static const uint CACHE_H = TILE_H + 2u * WIN_R;
static const uint CACHE_N = CACHE_W * CACHE_H;
static const uint TILE_N = TILE_W * TILE_H;
static const uint LOADS_PER_THREAD = CACHE_N / TILE_N;

groupshared uint s_id[CACHE_H][CACHE_W];

// Identity of a stored sample: instance, position and radiance words. Zero
// marks "no sample" and never matches.
uint LiteSampleId(uint px)
{
    const uint a = LiteAddress(px);
    const uint4 w = g_Reservoirs_last.Load4(a); // position | instance
    if (w.w == LITE_EMPTY) return 0u;
    const uint L = g_Reservoirs_last.Load(a + 16u);
    return max(Hash32(w.w ^ Hash32(w.x ^ Hash32(w.y ^ Hash32(w.z ^ Hash32(L ^ 0x44555021u))))), 1u);
}

[numthreads(TILE_W, TILE_H, 1)]
void main(uint3 tid : SV_DispatchThreadID, uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID)
{
    gDispatchIdx = tid;
    const int2 tileOrigin = int2(gid.xy * uint2(TILE_W, TILE_H)) - int2(WIN_R, WIN_R);
    const uint tlin = ltid.y * TILE_W + ltid.x;
    [unroll]
    for (uint i = 0u; i < LOADS_PER_THREAD; ++i)
    {
        const uint lidx = tlin * LOADS_PER_THREAD + i;
        const uint ly = lidx / CACHE_W;
        const uint lx = lidx % CACHE_W;
        const int2 p = tileOrigin + int2(lx, ly);
        uint id = 0u;
        if (p.x >= 0 && p.y >= 0 && p.x < (int)IMG_W && p.y < (int)IMG_H)
        {
            const uint px = MapPixelID(uint2(IMG_W, IMG_H), p);
            if ((load_flagsWord(g_sample_last, px) & SD_FLAG_NOBOUNCE) == 0u) id = LiteSampleId(px);
        }
        s_id[ly][lx] = id;
    }
    GroupMemoryBarrierWithGroupSync();
    if (tid.x >= IMG_W || tid.y >= IMG_H) return;

    const int cx = (int)ltid.x + (int)WIN_R;
    const int cy = (int)ltid.y + (int)WIN_R;
    const uint mine = s_id[cy][cx];
    uint count = 0u;
    if (mine != 0u)
    {
        [loop]
        for (int dy = -(int)WIN_R; dy <= (int)WIN_R; ++dy)
        {
            [loop]
            for (int dx = -(int)WIN_R; dx <= (int)WIN_R; ++dx)
            {
                if (dx == 0 && dy == 0) continue;
                if (s_id[cy + dy][cx + dx] == mine) ++count;
            }
        }
    }
    gScratchPing[uint3(tid.xy, LITE_DUP_SCRATCH)] = float4((float)count * LITE_DUP_NORM, 0.0f, 0.0f, 0.0f);
}
