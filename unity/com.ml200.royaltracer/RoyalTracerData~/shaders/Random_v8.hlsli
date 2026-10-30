inline uint Hash32(uint v) {
    v ^= v >> 16; v *= 0x7feb352d; v ^= v >> 15; v *= 0x846ca68b; v ^= v >> 16;
    return v;
}

uint2 GetSeed(uint2 idx, uint t, uint c, uint2 tileSize = uint2(0,0))
{

    if (tileSize.x == 0 || tileSize.y == 0)
    {
        return
            idx.yx * uint2(73856093u, 37623481u) ^
            idx    * uint2(19349663u, 51964263u) ^
            uint2(83492791u, 68250729u) * c      ^
            uint2(293803u,    423977u)  * t;
    }

    uint2 tile = idx / tileSize;
    uint  tileKey = (tile.y << 16) | tile.x;
    uint base = tileKey ^ (0xB5297A4Du * t) ^ (0x68E31DA4u * c);
    uint s0 = Hash32(base ^ 0x1u);
    uint s1 = Hash32(base ^ 0x2u);

    return uint2(s0, s1);
}

// Uniform in [0, 1).
inline float RandomFloatSingle(inout uint s)
{
    s *= 1664525u;
    s += 1013904223u;
    uint r = (s >> 9u);
    r = 0x3F800000u | r;
    return asfloat(r) - 1.0f;
}

inline float RandomFloatPCG(inout uint s)
{
    s = s * 747796405u + 2891336453u;
    uint word = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    word = (word >> 22u) ^ word;
    uint r = 0x3F800000u | (word >> 9u);
    return asfloat(r) - 1.0f;
}

uint initRandomData(uint2 idx, uint2 tileSize, uint t, uint c){
    return GetSeed(idx, t, c).x;
}

#define RC_STREAM_BSDF 0u
#define RC_STREAM_NEE  1u
#define RC_STREAM_SSS  2u
#define RC_STREAM_RR   3u

inline uint RcBounceSeed(uint pathSeed, uint depth, uint stream)
{
    return Hash32(pathSeed ^ (depth * 0x9E3779B9u) ^ (stream * 0x85EBCA6Bu));
}
