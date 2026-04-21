#define COMPUTE_PASS
#include "Includes_v8.hlsli"

//====================================================================
//DUPLICATION MAP, CORRELATION REDUCTION
//====================================================================
//Post-spat pass that counts shared V2 packed uints in each pixel's
//17x17 neighborhood of g_Reservoirs_last. Next frame's temporal reuse
//reads this map at the backprojected coord to adaptively lower the
//temporal cCap in highly correlated regions, preventing firefly
//persistence. Lin, Kettunen, Wyman 2026, "ReSTIR PT Enhanced", §5.
//
//Sample-identifier proxy: raw packed-uint V2. Reconnection / hybrid
//shifts preserve V2 bit-for-bit, so matching V2 strongly indicates
//shifted copies of the same initial candidate. DI samples (env miss,
//emitter hit, NEE at d=0) ordinarily lack a meaningful V2; raygen
//writes a pixel-and-frame-unique unit vector into their V2 slot as a
//dup-map discriminator (see Pass_raygen_v8.hlsl:diMarker).
//
//Output: gScratchPing slot 6, .x = D in [0, 1].
//
//Uses a 32x32 groupshared cache to collapse the 288 per-thread global
//reads into 4 cooperative loads per thread, and also cuts the 288
//MapPixelID calls per thread down to 4. Compute-bound, so the inner
//loop is the hot path; interior tiles skip the per-iter bounds check
//entirely.

static const uint TILE_W  = 16u;
static const uint TILE_H  = 16u;
static const uint WIN_R   = 8u;                     //17x17 radius
static const uint CACHE_W = TILE_W + 2u * WIN_R;    //32
static const uint CACHE_H = TILE_H + 2u * WIN_R;    //32
static const uint CACHE_N = CACHE_W * CACHE_H;      //1024
static const uint TILE_N  = TILE_W * TILE_H;        //256
static const uint LOADS_PER_THREAD = CACHE_N / TILE_N; //4

groupshared uint s_V2[CACHE_H][CACHE_W];

[numthreads(TILE_W, TILE_H, 1)]
void main(
    uint3 tid  : SV_DispatchThreadID,
    uint3 gid  : SV_GroupID,
    uint3 ltid : SV_GroupThreadID)
{
    gDispatchIdx = tid;

    //Tile top-left in pixel space, shifted by WIN_R so the cache covers
    //[tile-8, tile+16+8) = 32 pixels in each axis.
    const int2 tileOrigin = int2(gid.xy * uint2(TILE_W, TILE_H)) - int2(WIN_R, WIN_R);

    //Cooperative load: 256 threads fetch 4 cache entries each
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

    //My center in the cache, offset by WIN_R
    const int cx = (int)ltid.x + (int)WIN_R;
    const int cy = (int)ltid.y + (int)WIN_R;
    const uint myV2 = s_V2[cy][cx];

    //Interior-tile detection: every thread's 17x17 window lies fully
    //in-bounds when the tile's outermost pixel's window does. Saves 4
    //compares per inner iter (~1K compares per thread) on the common
    //path. All threads in the group take the same branch, so there's
    //no wavefront divergence here.
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

                //LDS at image-border neighbors holds 0; skip so that
                //an invalidated canonical (V2 = 0) at the edge doesn't
                //get a spuriously high duplicate count.
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
