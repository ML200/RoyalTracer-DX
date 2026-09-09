#define COMPUTE_PASS
#include "Includes_v8.hlsli"
#define DUP_KEY uint

//====================================
//DUPLICATION MAP CORRELATION REDUCTION
//====================================
//counts shared V2 packed uints in a 17x17 neighborhood, output to scratch slot 6.x

#include "Duplication_Map_v8.hlsli"

[numthreads(TILE_W, TILE_H, 1)]
void main(
    uint3 tid  : SV_DispatchThreadID,
    uint3 gid  : SV_GroupID,
    uint3 ltid : SV_GroupThreadID)
{
    gDispatchIdx = tid;

    //tile origin shifted by WIN_R, cache covers 32x32 around tile
    const int2 tileOrigin = int2(gid.xy * uint2(TILE_W, TILE_H)) - int2(WIN_R, WIN_R);

    //256 threads fetch 4 entries each
    const uint tlin = ltid.y * TILE_W + ltid.x;
    [unroll]
    for (uint i = 0u; i < LOADS_PER_THREAD; ++i)
    {
        const uint lidx = tlin * LOADS_PER_THREAD + i;
        const uint ly   = lidx / CACHE_W;
        const uint lx   = lidx % CACHE_W;
        const int2 px   = tileOrigin + int2(lx, ly);

        DUP_KEY v = (DUP_KEY)0u;
        if (px.x >= 0 && px.y >= 0 && px.x < (int)IMG_W && px.y < (int)IMG_H)
        {
            const uint pid = MapPixelID(uint2(IMG_W, IMG_H), uint2(px));
            v = g_Reservoirs_last.Load(addr_v2(pid));
        }
        s_V2[ly][lx] = v;
    }

    GroupMemoryBarrierWithGroupSync();

    if (tid.x >= IMG_W || tid.y >= IMG_H) return;

    const float D = DuplicationFraction(tid, gid, ltid);
    gScratchPing[uint3(tid.xy, 6)] = float4(D, 0.0f, 0.0f, 0.0f);
}
