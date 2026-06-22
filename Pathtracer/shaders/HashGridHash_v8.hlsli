//====================================
//SPMIS GLOBAL HASH GRID  (screen-space g-buffer hash + open-addressed cell table)
//====================================
//A single raw UAV (g_spmisBuffer, u25) holds ALL per-pixel + per-cell SPMIS arrays,
//sub-allocated by a fixed stride. This replaces the old tile-local cell scratch.
//
//Two index spaces:
//  * per-PIXEL arrays are indexed by MapPixelID (tile-swizzled, range [0, SP_STR)),
//    matching the reservoir / sample SoA layout. Stride = SP_STR == TileAlignedPx.
//  * per-CELL arrays + the checksum hash table are indexed by the DENSE hash cell
//    index in [0, SP_NUMCELLS) == [0, IMG_W*IMG_H), the modulo base.
//Both fit inside the SP_STR-strided arrays (SP_NUMCELLS <= SP_STR), so addressing is
//uniform: array k lives at element offset k*SP_STR.
#ifndef HASHGRIDHASH_V8_HLSLI
#define HASHGRIDHASH_V8_HLSLI

static const uint SP_UNDEF = 0xFFFFFFFFu;   // sentinel for an undefined checksum / grid index

//array slots (each SP_STR elements wide)
static const uint SP_HASH    = 0u;   // per-pixel : resolved cell index (or SP_UNDEF)
static const uint SP_CHK     = 1u;   // per-cell  : open-addressed checksum table
static const uint SP_IDX     = 2u;   // per-pixel : index of this pixel within its class in its cell
static const uint SP_SORTED  = 3u;   // dense     : pixel (reservoir) index, packed per cell, non-zero first
static const uint SP_PIXCNT  = 4u;   // per-cell  : total pixels in cell
static const uint SP_NZ      = 5u;   // per-cell  : non-zero (UCW>0) reservoir count
static const uint SP_CONF    = 6u;   // per-cell  : sum of M (confidence sum)
static const uint SP_OFF     = 7u;   // per-cell  : start offset into SP_SORTED
static const uint SP_OTHER   = 8u;   // per-cell  : non-important index allocator (scratch for the sort split)
static const uint SP_ARRAYS  = 9u;   // total arrays; the global offset counter sits right after them

//SP_STR == TileAlignedPx(IMG_W, IMG_H): tile-swizzled pixel count (8x4 tiles, see
//MapPixelID). Must equal the host's TileAlignedPx so the buffer is sized to match.
inline uint SP_STR()      { return ((IMG_W + 7u) / 8u) * ((IMG_H + 3u) / 4u) * 32u; }
inline uint SP_NUMCELLS() { return IMG_W * IMG_H; }                 // dense modulo base (pixel hash count)
inline uint SP_A(uint slot, uint i) { return (slot * SP_STR() + i) * 4u; }
inline uint SP_CTR()      { return (SP_ARRAYS * SP_STR()) * 4u; }   // global offset counter (1 uint)

//------------------------------------------------------------------
//Hash functions (PCG for the cell index, xxHash32 for the checksum)
//------------------------------------------------------------------
inline uint SP_h1_pcg(uint seed)
{
    uint state = seed * 747796405u + 2891336453u;
    uint word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}
inline uint SP_h2_xxhash32(uint seed)
{
    const uint P2 = 2246822519u, P3 = 3266489917u, P4 = 668265263u, P5 = 374761393u;
    uint h = seed + P5;
    h = P4 * ((h << 17) | (h >> 15));
    h = P2 * (h ^ (h >> 15));
    h = P3 * (h ^ (h >> 13));
    return h ^ (h >> 16);
}
inline float SP_xorshift01(uint s){ s ^= s << 13; s ^= s >> 17; s ^= s << 5; return (float)s * (1.0f / 4294967296.0f); }
inline void SP_onb(float3 n, out float3 T, out float3 B)
{
    const float sg = (n.z >= 0.0f) ? 1.0f : -1.0f;
    const float a  = -1.0f / (sg + n.z);
    const float b  = n.x * n.y * a;
    T = float3(1.0f + sg * n.x * n.x * a, sg * b, -sg * n.x);
    B = float3(b, sg + n.y * n.y * a, -n.y);
}
//quantize the normal (precision=2): map each component [-1,1] -> [0, 2^p-1]
//and pack into DISJOINT p-bit fields (x:[2p,3p), y:[p,2p), z:[0,p)). The previous
//version cast (uint)(int)(n*p) directly, so a negative component became 0xFFFF...
//and swamped the OR (the whole n.x<0 / n.y<0 / n.z<0 half-spaces each collapsed to
//one bucket) while the shifted fields also overlapped -- fragmenting coplanar
//surfaces and colliding opposite normals. p is bits-per-component AND the field
//stride, so the 2p/1p/0 shifts stay disjoint. (See the SPMIS tile-artifact fix.)
inline uint SP_quantize_normal(float3 n)
{
    const uint  p     = 2u;                          // bits per component (quantization precision)
    const float scale = (float)((1u << p) - 1u);     // 3 for p=2 -> buckets {0,1,2,3}
    const uint  x = (uint)(saturate(n.x * 0.5f + 0.5f) * scale + 0.5f) << (2u * p);
    const uint  y = (uint)(saturate(n.y * 0.5f + 0.5f) * scale + 0.5f) << (1u * p);
    const uint  z =  (uint)(saturate(n.z * 0.5f + 0.5f) * scale + 0.5f);
    return x | y | z;
}
//jitter the normal in its tangent plane (n, pos, 0.2) — deterministic per-position
//jitter so cell boundaries are fuzzed (anti-aliasing of the discrete normal buckets).
inline float3 SP_jitter_normal(float3 n, float3 pos, float fuzzy)
{
    float3 T, B; SP_onb(n, T, B);
    const float jx = SP_xorshift01(SP_h2_xxhash32(asuint(pos.x * 4294967295.0f))) * 2.0f - 1.0f;
    const float jy = SP_xorshift01(SP_h2_xxhash32(asuint(pos.y * 4294967295.0f))) * 2.0f - 1.0f;
    return normalize(n + (T * jx + B * jy) * fuzzy);
}
//screen-space g-buffer hash: cell = (pixel/tile_size, jittered-quantized normal).
//Returns the cell hash; the checksum (for collision resolution) is written to out.
inline uint SP_screen_hash(int px, int py, uint tileSize, float3 pos, float3 geomN, out uint outChecksum)
{
    const uint gx = (uint)px / tileSize;
    const uint gy = (uint)py / tileSize;
    const uint hn = SP_quantize_normal(SP_jitter_normal(geomN, pos, 0.2f));
    outChecksum = SP_h2_xxhash32(gx + SP_h2_xxhash32(gy + SP_h2_xxhash32(hn)));
    return SP_h1_pcg(gx + SP_h1_pcg(gy + SP_h1_pcg(hn)));
}

//------------------------------------------------------------------
//Open-addressed insertion (up to 256 probes) — atomic CAS into the SP_CHK checksum
//table, linear probing on collision. Returns the resolved cell index, or SP_UNDEF if
//the table is full / probing exhausted.
//------------------------------------------------------------------
inline uint SP_insert(uint cellIndex, uint checksum, uint numCells)
{
    uint prev;
    g_spmisBuffer.InterlockedCompareExchange(SP_A(SP_CHK, cellIndex), SP_UNDEF, checksum, prev);
    if (prev == SP_UNDEF || prev == checksum)
        return cellIndex;                                  // empty (we claimed it) or already ours

    const uint base = cellIndex;
    uint cur = cellIndex;
    [loop]
    for (uint i = 1u; i <= 256u; ++i)
    {
        cur = (cur + 1u) % numCells;
        if (cur == base) return SP_UNDEF;                  // wrapped the whole table
        const uint c = g_spmisBuffer.Load(SP_A(SP_CHK, cur));
        if (c == checksum) return cur;                     // found our cell
        if (c == SP_UNDEF)
        {
            g_spmisBuffer.InterlockedCompareExchange(SP_A(SP_CHK, cur), SP_UNDEF, checksum, prev);
            if (prev == SP_UNDEF || prev == checksum) return cur;
        }
    }
    return SP_UNDEF;
}

#endif
