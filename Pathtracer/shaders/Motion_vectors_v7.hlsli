//-----------------------------------------------
//  Wave-level spatial picker
//    – first k lanes create k offset vectors
//    – every lane randomly chooses one of them
//    – use the chosen offset to build a neighbour
//-----------------------------------------------
inline uint GetRandomPixelCircleWeighted(
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
            float  u     = RandomFloat(localSeed);
            float  z     = pow(u, SPAT_EXP_DI);
            float  r     = float(radius) * z;
            float  angle = RandomFloat(localSeed) * 6.2831853;

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
}


