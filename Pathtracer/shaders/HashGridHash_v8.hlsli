#ifndef HASHGRIDHASH_V8_HLSLI
#define HASHGRIDHASH_V8_HLSLI

static const uint SP_UNDEF = 0xFFFFFFFFu;

// SoA planes use the tile-aligned pixel count.
static const uint SP_HASH    = 0u;
static const uint SP_CHK     = 1u;
static const uint SP_IDX     = 2u;
static const uint SP_SORTED  = 3u;

static const uint SP_OTHER   = 8u;
static const uint SP_SORTEDW = 9u;
static const uint SP_ARRAYS  = 10u;

inline uint SP_STR()      { return ((IMG_W + 7u) / 8u) * ((IMG_H + 3u) / 4u) * 32u; }
inline uint SP_NUMCELLS() { return IMG_W * IMG_H; }
inline uint SP_A(uint slot, uint i) { return (slot * SP_STR() + i) * 4u; }
inline uint SP_CTR()      { return (SP_ARRAYS * SP_STR()) * 4u; }

inline uint SP_AGG(uint cell)      { return (4u * SP_STR() + cell * 4u) * 4u; }
inline uint SP_PIXCNT_A(uint cell) { return SP_AGG(cell) +  0u; }
inline uint SP_NZ_A(uint cell)     { return SP_AGG(cell) +  4u; }
inline uint SP_CONF_A(uint cell)   { return SP_AGG(cell) +  8u; }
inline uint SP_OFF_A(uint cell)    { return SP_AGG(cell) + 12u; }

inline uint SP_SRCH(uint px) { return (SP_ARRAYS * SP_STR()) * 4u + 32u + px * 32u; }

static const uint TEMP_STATUS_DEAD = 0u;
static const uint TEMP_STATUS_OK   = 1u;
#define SPMIS_SPLIT_MAXDRAWS 4u
#define SPMIS_TOTAL_ROLES (SPMIS_SPLIT_MAXDRAWS + 1u)

#define SPM_BUF_LAST_BIT 0x80000000u
#define SPM_KILLPH_BIT   0x40000000u
#define SPM_RESPX_MASK   0x1FFFFFFFu
static const uint SPM_STATUS_SKIP = 0u;
static const uint SPM_STATUS_PASS = 1u;
static const uint SPM_STATUS_NORM = 2u;
inline uint SPM_w0(uint px)            { return px * 4u; }
inline uint SPM_w1(uint px)            { return SP_STR() * 4u + px * 4u; }

inline uint SPM_dBase(uint d)          { return SP_STR() * (8u + d * 32u); }
inline uint SPM_slotS(uint px, uint d) { return SPM_dBase(d) + px * 4u; }
inline uint SPM_slotZ(uint px, uint d) { return SPM_dBase(d) + SP_STR() *  4u + px * 4u; }
inline uint SPM_slotP(uint px, uint d) { return SPM_dBase(d) + SP_STR() *  8u + px * 4u; }
inline uint SPM_slotJ(uint px, uint d) { return SPM_dBase(d) + SP_STR() * 12u + px * 8u; }
inline uint SPM_slotC(uint px, uint d) { return SPM_dBase(d) + SP_STR() * 20u + px * 12u; }
inline uint SPM_packHdr(uint reuseCell, uint status) { return (reuseCell & 0x0FFFFFFFu) | (status << 28u); }
inline uint SPM_hdrStatus(uint w0)    { return w0 >> 28u; }
inline uint SPM_hdrCell(uint w0)      { return w0 & 0x0FFFFFFFu; }

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

inline uint SP_quantize_normal(float3 n)
{

    const uint  p     = clamp(spmis_normalBits, 1u, 4u);
    const float scale = (float)((1u << p) - 1u);
    const uint  x = (uint)(saturate(n.x * 0.5f + 0.5f) * scale + 0.5f) << (2u * p);
    const uint  y = (uint)(saturate(n.y * 0.5f + 0.5f) * scale + 0.5f) << (1u * p);
    const uint  z =  (uint)(saturate(n.z * 0.5f + 0.5f) * scale + 0.5f);
    return x | y | z;
}

inline float3 SP_jitter_normal(float3 n, float3 pos, float fuzzy)
{
    if (fuzzy <= 0.0f) return n;
    float3 T, B; SP_onb(n, T, B);
    const float jx = SP_xorshift01(SP_h2_xxhash32(asuint(pos.x * 4294967295.0f))) * 2.0f - 1.0f;
    const float jy = SP_xorshift01(SP_h2_xxhash32(asuint(pos.y * 4294967295.0f))) * 2.0f - 1.0f;
    return normalize(n + (T * jx + B * jy) * fuzzy);
}

inline uint SP_screen_hash(int px, int py, uint tileSize, float3 pos, float3 geomN, out uint outChecksum)
{
    const uint gx = (uint)px / tileSize;
    const uint gy = (uint)py / tileSize;
    const uint hn = SP_quantize_normal(SP_jitter_normal(geomN, pos, spmis_normalFuzz));
    outChecksum = SP_h2_xxhash32(gx + SP_h2_xxhash32(gy + SP_h2_xxhash32(hn)));
    return SP_h1_pcg(gx + SP_h1_pcg(gy + SP_h1_pcg(hn)));
}

inline float SpmisDrawWeight(
    float  phF,
    float  cachedJac_i,
    float  phShift,
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

inline float SpmisCanonicalMis(
    float revT, float p_c,
    float centerConf, float neighbors_conf_sum, float partnerConf, float pixCount)
{
    const float denom_mc = revT * neighbors_conf_sum + p_c * centerConf;
    return (denom_mc > EPSILON)
         ? pixCount * (partnerConf / neighbors_conf_sum) * (p_c * centerConf) / denom_mc
         : 0.0f;
}

inline uint SP_insert(uint cellIndex, uint checksum, uint numCells)
{
    uint prev;
    g_spmisBuffer.InterlockedCompareExchange(SP_A(SP_CHK, cellIndex), SP_UNDEF, checksum, prev);
    if (prev == SP_UNDEF || prev == checksum)
        return cellIndex;

    const uint base = cellIndex;
    uint cur = cellIndex;
    [loop]
    for (uint i = 1u; i <= 256u; ++i)
    {
        cur = (cur + 1u) % numCells;
        if (cur == base) return SP_UNDEF;
        const uint c = g_spmisBuffer.Load(SP_A(SP_CHK, cur));
        if (c == checksum) return cur;
        if (c == SP_UNDEF)
        {
            // Claim empty slots with an atomic checksum write.
            g_spmisBuffer.InterlockedCompareExchange(SP_A(SP_CHK, cur), SP_UNDEF, checksum, prev);
            if (prev == SP_UNDEF || prev == checksum) return cur;
        }
    }
    return SP_UNDEF;
}

#endif
