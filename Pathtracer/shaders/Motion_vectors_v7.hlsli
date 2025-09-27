inline int MirrorCoord(int v, int extent)
{
    v = abs(v);
    int  period = extent * 2;
    int  m      = v % period;
    return (m < extent) ? m : period - m - 1;
}

inline uint GetRandomPixelCircleWeighted(
    uint   radius,
    uint   w,
    uint   h,
    uint   x,
    uint   y,
    inout  uint2 threadSeed)
{
    if (radius == 0)
        return MapPixelID(uint2(w, h), uint2(x, y));

    int offX, offY;
    do
    {
        float  u     = RandomFloatSingle(threadSeed.x);
        float  z     = pow(u, SPAT_EXP_DI);
        float  r     = float(radius) * z;
        float  angle = RandomFloatSingle(threadSeed.x) * 6.2831853;

        offX = int(cos(angle) * r);
        offY = int(sin(angle) * r);
    } while (offX == 0 && offY == 0);

    int newX = MirrorCoord(int(x) + offX, int(w));
    int newY = MirrorCoord(int(y) + offY, int(h));

    return MapPixelID(uint2(w, h), uint2(newX, newY));
}


