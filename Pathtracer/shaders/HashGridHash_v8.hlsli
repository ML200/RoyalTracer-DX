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
//Per-pixel SEARCH RECORD (AoS, 32B) for the Pass_spmis_select cell-search fast path.
//One SECTOR-ALIGNED record now carries EVERYTHING a probe needs, so each probe is a
//single 32B load instead of the former scattered SP_HASH[npx] (4B) + 16B record pair
//in two regions (2 dependent sectors -> 1):
//  half A @ +0  : cell | worldPos.xyz     reject key + plane-distance test
//  half B @ +16 : conf | worldN.xyz       WRS weight + (gated) normal cone
//cell duplicates SP_HASH[px]: Pass_spmis_reset writes SP_UNDEF into it for every pixel
//(no-hit pixels never reach sort), Pass_spmis_sort overwrites it with the resolved cell
//and fills pos/conf/n. Select also reads its OWN record for cellCenter/myPos/myN, which
//kills its per-pixel instanceProps matrix fan-out. Base is padded to 32B (the counter
//slot below occupies the first 32B) so records never straddle sectors. Host buffer
//sized in Renderer_Pipeline.cpp ((18*TileAlignedPx + 8) uints).
inline uint SP_SRCH(uint px) { return (SP_ARRAYS * SP_STR()) * 4u + 32u + px * 32u; }

//====================================
//SPMIS SPLIT-PASS SCRATCH  (aliases g_pathStateBuffer in SPMIS mode)
//====================================
//The reuse pass is split select -> shift -> merge to lift per-pass occupancy (the rays
//land in shift with minimal live state). Per-pixel state packs a header + JOB SLOTS
//(one canonical + up to SPMIS_SPLIT_MAXDRAWS non-canonical) into the path-state scratch
//(texture spatial reuse is idle in SPMIS mode, so the buffer is free).
//
//LAYOUT: SoA PLANES over SP_STR() pixels. Select's stores and shift/merge's reloads are
//all own-pixel, so plane-major makes every access warp-coalesced (adjacent threads hit
//adjacent dwords) instead of scattering header+slots across multiple sectors per pixel.
//Total bytes 8 + SPMIS_TOTAL_ROLES*32 per px = kPathStateBytesPerPx (Renderer.h).
//  w0 plane   4B/px = (status<<28) | reuseCell[27:0]  select -> shift(reuseCell)/merge(status)
//  w1 plane   4B/px = partnerPx (select, canonical routing only — NEVER overwritten by
//              shift anymore; merge reads it directly for the reverse-shift MIS term)
//  per role d (JOB SLOT, generic — see "UNIFIED SHIFT JOB" below): d == DispatchRaysIndex().z
//              DIRECTLY (no remapping anywhere) — spatial d=0 is canonical, d=1..Ntn are
//              draws (draw index = d-1); temporal d=0 is forward, d=1 is reverse:
//              startPx plane 4B/px  startPx | startBufLast<<31        (select/temp_gi input)
//              resPx   plane 4B/px  SP_UNDEF (no job) | resPx | resBufLast<<31 | killPh<<30
//              prob    plane 4B/px  selProb (draws only; canonical's is unused garbage)
//              J8      plane 8B/px  cachedNew | Jn (Jn's SIGN carries preVisDead — see
//                      Pass_shift_v8) — shift output, merge input; draws' cachedNew<=0
//                      after a walk (visibility folded in) and preVisDead after a direct
//                      eval are BOTH still "dead" to merge, same asymmetry as before
//              c       plane 12B/px shift output (re-anchored shifted F, shadowed)
//status: 0 = emitter/skip, 1 = passthrough, 2 = normal.
//To raise SPMIS_SPLIT_MAXDRAWS, bump kPathStateBytesPerPx (Renderer.h) to
//>= 8 + (SPMIS_SPLIT_MAXDRAWS+1)*32, and spatial_shift's Depth (Renderer.cpp's
//Stage::RayGen dispatchTag resolution — keep it at Ntn+1).
//
//UNIFIED SHIFT JOB (Pass_shift_v8, round-8 unification): a "job" is one
//reconnection-or-replay mapping, fully described by {startPx, startBufLast,
//resPx, resBufLast, killPh} — generic across BOTH domains, addressed
//purely by d = DispatchRaysIndex().z (the shader never asks "which domain am I"):
//  spatial canonical (d=0): startPx=partnerPx, startBufLast=0, resPx=pixelIdx, resBufLast=0
//  spatial draw d (1..Ntn): startPx=pixelIdx (ALWAYS — stored anyway so the shader never
//                      special-cases domains), startBufLast=0, resPx=zPx, resBufLast=0
//  temporal forward (d=0): startPx=pixelIdx, startBufLast=0, resPx=tempPixelIdx, resBufLast=1
//  temporal reverse (d=1): startPx=tempPixelIdx, startBufLast=1, resPx=pixelIdx, resBufLast=0
//needWalk is NOT stored — Pass_shift_v8 derives it itself from RcReplayLen(rcInfo) after
//loading rcInfo from (resBufLast,resPx), so select/temp_gi only decide WHETHER a job
//exists (resPx==SP_UNDEF gates dead-on-arrival slots out before shift ever runs) and
//WHERE it points; the routing stage that used to also pick a "pending" bit is gone —
//the walk-or-direct branch inside Pass_shift_v8 is the ONLY place that decides.
//there is no more pre-visibility-ray hint (removed — held gBaseHint/a divergent
//early-out live across the reconnect block for ~30 spurious VGPRs on the
//spatial binary): every job, walked or direct, bands post-fence in its merge
//instead (Pass_spmis_merge / Pass_temp_merge), same Jn/gBase/threshold. killPh
//is the one surviving spatial-only legacy near-specular zeroing bit; temporal
//jobs never set it.
//
//TEMPORAL PHASE (before any of the above): temp_gi/Pass_shift_v8(loop:temporal_shift)/
//temp_merge run BEFORE select and give w0/w1/job-slot-0/1 their OWN meaning for that
//window — temp_merge is the last temporal reader, so select's overwrite right after is
//safe. w0 = TEMP_STATUS_* (temp_gi's pessimistic-DEAD-first write makes every
//early-return path, including the tempGI-off/emitter cases, leave a coherent status even
//though this word is otherwise spatial-only); w1 = candidate coord|dualBit (same slot
//spatial later reuses for partnerPx); job slots d=0 (forward) / d=1 (reverse) hold the
//SAME {startPx,resPx,killPh=0} input / {cachedNew,Jn,c} output shape as
//spatial's — temp_gi always writes a full job descriptor (never SP_UNDEF unless the
//direction is genuinely inactive, e.g. no valid reverse candidate), so Pass_shift_v8
//runs unconditionally for both temporal slots every OK pixel (no more per-direction
//PENDING-sentinel skip — the old "maybe temp_gi already resolved this directly" case is
//gone too: ALL direction resolution, walk or direct, now happens in Pass_shift_v8 only).
static const uint TEMP_STATUS_DEAD = 0u;   //no candidate this frame -> reservoir left untouched
static const uint TEMP_STATUS_OK   = 1u;
#define SPMIS_SPLIT_MAXDRAWS 4u
#define SPMIS_TOTAL_ROLES (SPMIS_SPLIT_MAXDRAWS + 1u)   // draws 0..3 + canonical/temporal-dir @ index 4
//job-descriptor resPx word: top bits of resPx (real pixel counts sit
//comfortably under 2^29, even at 8K) — SP_UNDEF (all-ones) can never collide
//with a valid combination of these. Used by select/temp_gi (write) and
//Pass_shift_v8/the merge passes (read) — see the "UNIFIED SHIFT JOB" note above.
#define SPM_BUF_LAST_BIT 0x80000000u
#define SPM_KILLPH_BIT   0x40000000u
#define SPM_RESPX_MASK   0x1FFFFFFFu   // bit 29 (was SPM_HINT_BIT) is unused/reserved
static const uint SPM_STATUS_SKIP = 0u;
static const uint SPM_STATUS_PASS = 1u;
static const uint SPM_STATUS_NORM = 2u;
inline uint SPM_w0(uint px)            { return px * 4u; }
inline uint SPM_w1(uint px)            { return SP_STR() * 4u + px * 4u; }
//job-slot d plane group starts after the two header planes (8B/px worth = 8*SP_STR bytes)
inline uint SPM_dBase(uint d)          { return SP_STR() * (8u + d * 32u); }
inline uint SPM_slotS(uint px, uint d) { return SPM_dBase(d) + px * 4u; }
inline uint SPM_slotZ(uint px, uint d) { return SPM_dBase(d) + SP_STR() *  4u + px * 4u; }
inline uint SPM_slotP(uint px, uint d) { return SPM_dBase(d) + SP_STR() *  8u + px * 4u; }
inline uint SPM_slotJ(uint px, uint d) { return SPM_dBase(d) + SP_STR() * 12u + px * 8u; }
inline uint SPM_slotC(uint px, uint d) { return SPM_dBase(d) + SP_STR() * 20u + px * 12u; }
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
    //bits per component (quantization precision) — editor slider, slot 25.
    //p is bits-per-component AND the field stride, so the 2p/1p/0 shifts stay
    //disjoint at every setting.
    const uint  p     = clamp(spmis_normalBits, 1u, 4u);
    const float scale = (float)((1u << p) - 1u);     // 3 for p=2 -> buckets {0,1,2,3}
    const uint  x = (uint)(saturate(n.x * 0.5f + 0.5f) * scale + 0.5f) << (2u * p);
    const uint  y = (uint)(saturate(n.y * 0.5f + 0.5f) * scale + 0.5f) << (1u * p);
    const uint  z =  (uint)(saturate(n.z * 0.5f + 0.5f) * scale + 0.5f);
    return x | y | z;
}
//jitter the normal in its tangent plane — deterministic per-position jitter so
//cell boundaries are fuzzed (anti-aliasing of the discrete normal buckets).
//fuzzy <= 0 returns n untouched (hard buckets).
inline float3 SP_jitter_normal(float3 n, float3 pos, float fuzzy)
{
    if (fuzzy <= 0.0f) return n;
    float3 T, B; SP_onb(n, T, B);
    const float jx = SP_xorshift01(SP_h2_xxhash32(asuint(pos.x * 4294967295.0f))) * 2.0f - 1.0f;
    const float jy = SP_xorshift01(SP_h2_xxhash32(asuint(pos.y * 4294967295.0f))) * 2.0f - 1.0f;
    return normalize(n + (T * jx + B * jy) * fuzzy);
}
//screen-space g-buffer hash: cell = (pixel/tile_size, jittered-quantized normal).
//Returns the cell hash; the checksum (for collision resolution) is written to out.
//Normal fuzz amplitude = spmis_normalFuzz (editor slider, slot 24; was 0.2 fixed).
inline uint SP_screen_hash(int px, int py, uint tileSize, float3 pos, float3 geomN, out uint outChecksum)
{
    const uint gx = (uint)px / tileSize;
    const uint gy = (uint)py / tileSize;
    const uint hn = SP_quantize_normal(SP_jitter_normal(geomN, pos, spmis_normalFuzz));
    outChecksum = SP_h2_xxhash32(gx + SP_h2_xxhash32(gy + SP_h2_xxhash32(hn)));
    return SP_h1_pcg(gx + SP_h1_pcg(gy + SP_h1_pcg(hn)));
}

//------------------------------------------------------------------
//SPMIS PSS draw algebra — one implementation for both mapping kinds inside
//Pass_spmis_shift (direct k==2 reconnection and k>2 replay roles), so both
//routes produce bit-identical weights.
//
//With |J| = cachedNew/cachedJac_i, the pairwise-MIS ratio is computed via
//  A = lum(F_i) * cachedJac_i      (= p_hat_from_i * cachedNew)
//  B = lum(c*vis) * Jn             (= p_hat_shifted * cachedNew)
//so the cachedNew factor cancels out of m_i entirely (numerically robust when
//the new-side bundle is tiny). The reservoir-combine weight then uses the
//shifted density against the SOURCE bundle: fwdT = lum(c*vis)*Jn/cachedJac_i.
//------------------------------------------------------------------
inline float SpmisDrawWeight(
    float  phF,          //lum(pr.F): neighbour target in its own domain
    float  cachedJac_i,  //neighbour's stored jacobian bundle
    float  phShift,      //lum(c*vis): shifted numerator luminance at the centre
    float  Jn,
    float  selProb, uint Ntn,
    float  neighbors_conf_sum, float centerConf, float c_i_scaled)
{
    if (!(cachedJac_i > 0.0f) || !(selProb > 0.0f)) return 0.0f;
    const float A     = phF * cachedJac_i;
    const float B     = phShift * Jn;
    const float denom = A * neighbors_conf_sum + B * centerConf;
    const float mi    = (denom > EPSILON) ? (A * c_i_scaled / denom) : 0.0f;
    const float fwdT  = phShift * Jn / cachedJac_i;
    const float spmis = 1.0f / ((float)Ntn * selProb);
    return spmis * mi * fwdT;
}

//canonical pairwise MIS weight (Nc=1 uniform partner pick over the cell).
//revT = the CENTRE sample's shifted density at the partner pixel
//     = lum(c_rev*vis) * Jn_rev / cachedJac_centre (0 when the shift failed).
inline float SpmisCanonicalMis(
    float revT, float p_c,
    float centerConf, float neighbors_conf_sum, float partnerConf, float pixCount)
{
    const float denom_mc = revT * neighbors_conf_sum + p_c * centerConf;
    return (denom_mc > EPSILON)
         ? pixCount * (partnerConf / neighbors_conf_sum) * (p_c * centerConf) / denom_mc
         : 0.0f;
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
