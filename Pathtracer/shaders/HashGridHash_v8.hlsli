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
//slots 4..7 repurposed as the per-cell AGGREGATE struct (AoS, 16B = PIXCNT|NZ|CONF|OFF);
//addressed by SP_AGG / SP_*_A below instead of SP_A.
static const uint SP_OTHER   = 8u;   // per-cell  : non-important index allocator (scratch for the sort split)
static const uint SP_SORTEDW = 9u;   // dense     : inner-RIS target (UCW*target*M) precomputed at sort, parallel to SP_SORTED
static const uint SP_ARRAYS  = 10u;  // total arrays; the global offset counter sits right after them

//SP_STR == TileAlignedPx(IMG_W, IMG_H): tile-swizzled pixel count (8x4 tiles, see
//MapPixelID). Must equal the host's TileAlignedPx so the buffer is sized to match.
inline uint SP_STR()      { return ((IMG_W + 7u) / 8u) * ((IMG_H + 3u) / 4u) * 32u; }
inline uint SP_NUMCELLS() { return IMG_W * IMG_H; }                 // dense modulo base (pixel hash count)
inline uint SP_A(uint slot, uint i) { return (slot * SP_STR() + i) * 4u; }
inline uint SP_CTR()      { return (SP_ARRAYS * SP_STR()) * 4u; }   // global offset counter (1 uint)
//Per-cell AGGREGATE struct (AoS, 16B = PIXCNT|NZ|CONF|OFF) in the byte space of the former
//SoA slots 4..7 (cell*16 fits since SP_NUMCELLS <= SP_STR). One Load4(SP_AGG(cell)) fetches
//all four for the hot reuseCell read (was 4 scattered loads to 4 SP_STR-strided regions) and
//count's 3 InterlockedAdds then land in one cache line. Load4 lanes: .x=PIXCNT .y=NZ .z=CONF .w=OFF.
inline uint SP_AGG(uint cell)      { return (4u * SP_STR() + cell * 4u) * 4u; }
inline uint SP_PIXCNT_A(uint cell) { return SP_AGG(cell) +  0u; }
inline uint SP_NZ_A(uint cell)     { return SP_AGG(cell) +  4u; }
inline uint SP_CONF_A(uint cell)   { return SP_AGG(cell) +  8u; }
inline uint SP_OFF_A(uint cell)    { return SP_AGG(cell) + 12u; }
//Per-pixel SEARCH RECORD (AoS, 16B = cellConf + worldPos) for the Pass_spmis_select
//cell-search fast path: bundled so each probe is ONE Load4 (npx-indexed, framebuffer-
//local) instead of a scattered SP_CONF[ncell] + neighbour-G-buffer fan-out. Lives after
//the SoA arrays + counter, 16B-aligned (the 1-uint counter occupies the first 16B slot).
//Written by Pass_spmis_sort, read by Pass_spmis_select; host buffer sized for it in
//Renderer_Pipeline.cpp ((14*TileAlignedPx + 4) uints).
inline uint SP_SRCH(uint px) { return (SP_ARRAYS * SP_STR()) * 4u + 16u + px * 16u; }

//====================================
//SPMIS SPLIT-PASS SCRATCH  (aliases g_pathStateBuffer in SPMIS mode)
//====================================
//The reuse pass is split select -> shift -> merge to lift per-pass occupancy (the rays
//land in shift with minimal live state). Per-pixel record packs a header + up to
//SPMIS_SPLIT_MAXDRAWS non-canonical draw slots into the path-state scratch (texture
//spatial reuse is idle in SPMIS mode, so the buffer is free). Sized to the existing
//kPathStateBytesPerPx (88 B): 8 B header + 4*20 B = 88 B, so no host allocation change.
//  header w0 = (status<<28) | reuseCell[27:0]    select -> shift(reuseCell)/merge(status)
//         w1 = partnerPx (select) -> mis_c | passVis (shift) -> merge
//  draw d  +0 = zPx       select; shift clears to SP_UNDEF on gate-reject; merge reads
//          +4 = selProb (select) -> w_draw (shift) -> merge
//          +8 = c.xyz 12B (shift) -> merge   (shadowed reconnection contribution)
//status: 0 = emitter/skip, 1 = passthrough, 2 = normal.
//To raise SPMIS_SPLIT_MAXDRAWS, bump kPathStateBytesPerPx (Renderer.h) to >= 8 + N*20.
#define SPMIS_SPLIT_MAXDRAWS 4u
static const uint SPM_STATUS_SKIP = 0u;
static const uint SPM_STATUS_PASS = 1u;
static const uint SPM_STATUS_NORM = 2u;
static const uint SPM_HDR_BYTES   = 8u;
static const uint SPM_SLOT_BYTES  = 20u;
inline uint SPM_STRIDE()              { return SPM_HDR_BYTES + SPMIS_SPLIT_MAXDRAWS * SPM_SLOT_BYTES; }
inline uint SPM_w0(uint px)           { return px * SPM_STRIDE(); }
inline uint SPM_w1(uint px)           { return px * SPM_STRIDE() + 4u; }
inline uint SPM_slot(uint px, uint d) { return px * SPM_STRIDE() + SPM_HDR_BYTES + d * SPM_SLOT_BYTES; }
inline uint SPM_packHdr(uint reuseCell, uint status) { return (reuseCell & 0x0FFFFFFFu) | (status << 28u); }
inline uint SPM_hdrStatus(uint w0)    { return w0 >> 28u; }
inline uint SPM_hdrCell(uint w0)      { return w0 & 0x0FFFFFFFu; }

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
