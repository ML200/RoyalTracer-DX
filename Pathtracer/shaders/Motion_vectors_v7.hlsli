//-----------------------------------------------
//  Wave-level spatial picker
//    – first k lanes create k offset vectors
//    – every lane randomly chooses one of them
//    – use the chosen offset to build a neighbour
//-----------------------------------------------
/*inline uint GetRandomPixelCircleWeighted(
    uint   radius,
    uint   w,
    uint   h,
    uint   x,
    uint   y,
    inout  uint2 threadSeed
){
    // Clamp k to a sensible range
    uint k = clamp(SPAT_WAVE_CANDIDATES_DI, 1u, WaveGetLaneCount());

    const uint lane = WaveGetLaneIndex();

    // -------------------------------------------------
    // 1.  Generate k candidate offsets  (lanes 0…k-1)
    // -------------------------------------------------
    int candX = 0, candY = 0;

    if (lane < k)
    {
        // Decorrelate the generators’ RNG streams a little
        uint2 localSeed = threadSeed;
        localSeed.x ^= lane * 0x9E3779B9u;

        do {
            float  u     = RandomFloatSingle(localSeed.x);
            float  z     = pow(u, SPAT_EXP_DI);
            float  r     = float(radius) * z;
            float  angle = RandomFloatSingle(localSeed.x) * 6.2831853;

            candX = int(cos(angle) * r);
            candY = int(sin(angle) * r);
            // repeat until offset ≠ (0,0) so nobody ever gets the centre
        } while (candX == 0 && candY == 0);
    }

    // -------------------------------------------------
    // 2.  Every lane picks one of those k candidates
    // -------------------------------------------------
    uint choice = (uint)(RandomFloat(threadSeed) * k);

    int offX = WaveReadLaneAt(candX, choice);
    int offY = WaveReadLaneAt(candY, choice);

    // -------------------------------------------------
    // 3.  Apply the offset and mirror to screen bounds
    // -------------------------------------------------
    int newX = int(x) + offX;
    int newY = int(y) + offY;

    // mirror X
    while (newX < 0 || newX >= int(w)) {
        newX = (newX < 0) ? -newX
                          : 2 * int(w) - newX - 2;
    }
    // mirror Y
    while (newY < 0 || newY >= int(h)) {
        newY = (newY < 0) ? -newY
                          : 2 * int(h) - newY - 2;
    }

    return MapPixelID(uint2(w, h), uint2(newX, newY));
}*/

//─────────────────────────────────────────────────────────────────────────────
//  GetRandomPixelCircleWeighted  – thread-local variant
//    • Picks ONE random offset inside a radius-weighted disk
//    • No wave intrinsics; each thread works independently
//    • Mirrors out-of-bounds coordinates back into the image
//─────────────────────────────────────────────────────────────────────────────
inline int MirrorCoord(int v, int extent)          // branch-free mirror
{
    v = abs(v);
    int  period = extent * 2;
    int  m      = v % period;
    return (m < extent) ? m : period - m - 1;
}

inline uint GetRandomPixelCircleWeighted(
    uint   radius,      // search radius in pixels
    uint   w,           // image width
    uint   h,           // image height
    uint   x,           // caller’s pixel x
    uint   y,           // caller’s pixel y
    inout  uint2 threadSeed)
{
    //---------------------------------------------------------------------
    // 1.  Draw ONE random offset in a radius-weighted disk
    //---------------------------------------------------------------------
    if (radius == 0)   // degenerate case → stay at centre
        return MapPixelID(uint2(w, h), uint2(x, y));

    int offX, offY;
    do
    {
        float  u     = RandomFloatSingle(threadSeed.x);     // [0,1)
        float  z     = pow(u, SPAT_EXP_DI);                 // weight fall-off
        float  r     = float(radius) * z;
        float  angle = RandomFloatSingle(threadSeed.x) * 6.2831853;

        offX = int(cos(angle) * r);
        offY = int(sin(angle) * r);
        // repeat until we do NOT pick the centre
    } while (offX == 0 && offY == 0);

    //---------------------------------------------------------------------
    // 2.  Apply offset and mirror to stay inside frame bounds
    //---------------------------------------------------------------------
    int newX = MirrorCoord(int(x) + offX, int(w));
    int newY = MirrorCoord(int(y) + offY, int(h));

    //---------------------------------------------------------------------
    // 3.  Map back to your linear pixel ID convention
    //---------------------------------------------------------------------
    return MapPixelID(uint2(w, h), uint2(newX, newY));
}


