#define COMPUTE_PASS
#include "Includes_v8.hlsli"

//====================================
//DUPLICATION MAP CORRELATION REDUCTION
//====================================
//counts shared V2 packed uints in 17x17 neighborhood of g_Reservoirs_last
//next frame's temporal reuse reads map at backprojected coord to lower cCap
//Lin, Kettunen, Wyman 2026 "ReSTIR PT Enhanced" §5
//V2 preserved bit-exact under reconnection/hybrid shift, matching V2 = shifted copy
//DI samples use per-pixel diMarker as discriminator (raygen)
//output, gScratchPing slot 6 .x = D in [0,1]
//32x32 groupshared cache collapses 288 global reads into 4 per thread

static const uint TILE_W  = 16u;
static const uint TILE_H  = 16u;
static const uint WIN_R   = 8u;
static const uint CACHE_W = TILE_W + 2u * WIN_R;
static const uint CACHE_H = TILE_H + 2u * WIN_R;
static const uint CACHE_N = CACHE_W * CACHE_H;
static const uint TILE_N  = TILE_W * TILE_H;
static const uint LOADS_PER_THREAD = CACHE_N / TILE_N;

groupshared uint s_V2[CACHE_H][CACHE_W];

[numthreads(TILE_W, TILE_H, 1)]
void main(
    uint3 tid  : SV_DispatchThreadID,
    uint3 gid  : SV_GroupID,
    uint3 ltid : SV_GroupThreadID)
{
    gDispatchIdx = tid;

    //tile origin shifted by WIN_R, cache covers 32x32 around tile
    const int2 tileOrigin = int2(gid.xy * uint2(TILE_W, TILE_H)) - int2(WIN_R, WIN_R);

    //256 threads fetch 4 cache entries each
    const uint tlin = ltid.y * TILE_W + ltid.x;
    [unroll]
    for (uint i = 0u; i < LOADS_PER_THREAD; ++i)
    {
        const uint lidx = tlin * LOADS_PER_THREAD + i;
        const uint ly   = lidx / CACHE_W;
        const uint lx   = lidx % CACHE_W;
        const int2 px   = tileOrigin + int2(lx, ly);

        uint v = 0u;
        if (px.x >= 0 && px.y >= 0 && px.x < (int)IMG_W && px.y < (int)IMG_H)
        {
            const uint pid = MapPixelID(uint2(IMG_W, IMG_H), uint2(px));
            v = g_Reservoirs_last.Load(addr_v2(pid));
        }
        s_V2[ly][lx] = v;
    }

    GroupMemoryBarrierWithGroupSync();

    if (tid.x >= IMG_W || tid.y >= IMG_H) return;

    //my cache center, offset by WIN_R
    const int cx = (int)ltid.x + (int)WIN_R;
    const int cy = (int)ltid.y + (int)WIN_R;
    const uint myV2 = s_V2[cy][cx];

    //interior tile, full 17x17 in bounds, skip per-iter bounds check
    //same branch for all threads in group, no wavefront divergence
    const bool interior =
        (gid.x >= 1u) && (gid.y >= 1u) &&
        (gid.x * TILE_W + TILE_W - 1u + WIN_R < IMG_W) &&
        (gid.y * TILE_H + TILE_H - 1u + WIN_R < IMG_H);

    uint count = 0u;
    if (interior)
    {
        [loop] for (int dy = -(int)WIN_R; dy <= (int)WIN_R; ++dy)
        {
            [loop] for (int dx = -(int)WIN_R; dx <= (int)WIN_R; ++dx)
            {
                if (dx == 0 && dy == 0) continue;
                const uint nV2 = s_V2[cy + dy][cx + dx];
                if (nV2 == myV2) ++count;
            }
        }
    }
    else
    {
        [loop] for (int dy = -(int)WIN_R; dy <= (int)WIN_R; ++dy)
        {
            [loop] for (int dx = -(int)WIN_R; dx <= (int)WIN_R; ++dx)
            {
                if (dx == 0 && dy == 0) continue;

                //LDS at border holds 0, skip so invalidated canonical V2=0 doesn't inflate count
                const int2 gpx = int2(tid.xy) + int2(dx, dy);
                if (gpx.x < 0 || gpx.y < 0 || gpx.x >= (int)IMG_W || gpx.y >= (int)IMG_H)
                    continue;

                const uint nV2 = s_V2[cy + dy][cx + dx];
                if (nV2 == myV2) ++count;
            }
        }
    }

    const float D = (float)count * (1.0f / 288.0f);
    gScratchPing[uint3(tid.xy, 6)] = float4(D, 0.0f, 0.0f, 0.0f);
}
