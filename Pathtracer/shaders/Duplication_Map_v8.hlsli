#ifndef DUPLICATION_MAP_V8_HLSLI
#define DUPLICATION_MAP_V8_HLSLI
// Caller defines DUP_KEY and fills the shared tile before the group barrier.
static const uint TILE_W  = 16u;
static const uint TILE_H  = 16u;
static const uint WIN_R   = 8u;
static const uint CACHE_W = TILE_W + 2u * WIN_R;
static const uint CACHE_H = TILE_H + 2u * WIN_R;
static const uint CACHE_N = CACHE_W * CACHE_H;
static const uint TILE_N  = TILE_W * TILE_H;
static const uint LOADS_PER_THREAD = CACHE_N / TILE_N;

groupshared DUP_KEY s_V2[CACHE_H][CACHE_W];

float DuplicationFraction(uint3 tid, uint3 gid, uint3 ltid)
{
    //my cache center, offset by WIN_R
    const int cx = (int)ltid.x + (int)WIN_R;
    const int cy = (int)ltid.y + (int)WIN_R;
    const DUP_KEY myV2 = s_V2[cy][cx];

    //interior tile skips per-iter bounds, group-uniform branch
    const bool interior =
        (gid.x >= 1u) && (gid.y >= 1u) &&
        (gid.x * TILE_W + TILE_W - 1u + WIN_R < IMG_W) &&
        (gid.y * TILE_H + TILE_H - 1u + WIN_R < IMG_H);

    uint count = 0u;
    if (interior)
    {
        // Unroll one row, keeping all 17 rows in a loop. HLSL unroll(4)
        // limits expansion to four iterations; it is not partial unrolling
        // and previously counted only 68 of the intended 288 neighbours.
        [loop] for (int dy = -(int)WIN_R; dy <= (int)WIN_R; ++dy)
        {
            [unroll] for (int dx = -(int)WIN_R; dx <= (int)WIN_R; ++dx)
            {
                if (dx == 0 && dy == 0) continue;
                const DUP_KEY nV2 = s_V2[cy + dy][cx + dx];
                if (all(nV2 == myV2)) ++count;
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

                //LDS border holds 0, skip so invalidated V2=0 cannot inflate count
                const int2 gpx = int2(tid.xy) + int2(dx, dy);
                if (gpx.x < 0 || gpx.y < 0 || gpx.x >= (int)IMG_W || gpx.y >= (int)IMG_H)
                    continue;

                const DUP_KEY nV2 = s_V2[cy + dy][cx + dx];
                if (all(nV2 == myV2)) ++count;
            }
        }
    }

    const float D = (float)count * (1.0f / 288.0f);
    return D;
}
#endif
