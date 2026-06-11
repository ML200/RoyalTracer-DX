//====================================
//PER-PIXEL PATH VERTEX STATE SCRATCH
//====================================
//wavefront, Pass_primary -> (Extend -> Classify -> MatEvalShade)* -> Pass_finalize
//SoA, all cross-stage per-pixel state here, overwritten by spat_gi_select later
//
//plane 0  PACK1   16B  x2 + n2_pk                 depth-1 vertex
//plane 1  PACK2   16B  uv + matID + objID + eta   depth-1 vertex
//plane 2  V2       4B  v2_pk                      depth-2 direction
//plane 3  HOT1    16B  throughput + prevNormal + prev_pdf + pdf_product
//plane 4  HOT2    16B  tpost + wsum + flags + pad
//plane 5  CLAS2    4B  Pr|Pm of the v_2 vertex (raygen-LIVE: store_ps_depth1)
//plane 6  SEED     4B
//plane 7  RAY_O   12B  rayOrigin
//plane 8  RAY_D   12B  rayDir
//plane 9  HP      16B  hitT + instID + primID + baryPk
//plane 10 CLAS1   16B  hitNormal + matID + etaOut + uv
//plane 11 CAND_WI 12B
//plane 12 CAND_M  16B  L + lightPdf + meta + lightObjID
//plane 13 CAND_LP 12B
//plane 14 CAND_LN  4B
//total 176B/pixel layout, tile-aligned via ps_numPx. The CPU allocation
//(Renderer.h kPathStateBytesPerPx = 88) backs only the raygen-live prefix
//(planes 0..5, 72B) plus headroom for the spat-select scratch that aliases
//the buffer from offset 0 (76B/px); the wavefront-only planes past 88B are
//dead code and intentionally unbacked. CLAS2 used to sit at plane offset 128
//- past the allocation, so its Pr/Pm round-trip silently read zeros (robust
//OOB access). Any plane the raygen path touches MUST stay below 88B/px.
static const uint PS_SZ_PACK1    = 16u;
static const uint PS_SZ_PACK2    = 16u;
static const uint PS_SZ_V2       =  4u;
static const uint PS_SZ_HOT1     = 16u;
static const uint PS_SZ_HOT2     = 16u;
static const uint PS_SZ_SEED     =  4u;
static const uint PS_SZ_RAY_O    = 12u;
static const uint PS_SZ_RAY_D    = 12u;
static const uint PS_SZ_HP       = 16u;
static const uint PS_SZ_CLAS1    = 16u;
static const uint PS_SZ_CLAS2    =  4u;
static const uint PS_SZ_CAND_WI  = 12u;
static const uint PS_SZ_CAND_M   = 16u;
static const uint PS_SZ_CAND_LP  = 12u;
static const uint PS_SZ_CAND_LN  =  4u;

static const uint PS_PLANE_PACK1    =   0u;
static const uint PS_PLANE_PACK2    =  16u;
static const uint PS_PLANE_V2       =  32u;
static const uint PS_PLANE_HOT1     =  36u;
static const uint PS_PLANE_HOT2     =  52u;
static const uint PS_PLANE_CLAS2    =  68u;
static const uint PS_PLANE_SEED     =  72u;
static const uint PS_PLANE_RAY_O    =  76u;
static const uint PS_PLANE_RAY_D    =  88u;
static const uint PS_PLANE_HP       = 100u;
static const uint PS_PLANE_CLAS1    = 116u;
static const uint PS_PLANE_CAND_WI  = 132u;
static const uint PS_PLANE_CAND_M   = 144u;
static const uint PS_PLANE_CAND_LP  = 160u;
static const uint PS_PLANE_CAND_LN  = 172u;

//tile-aligned pixel count, matches MapPixelID's 8-wide x 4-tall tile swizzle
//(Common_v8.hlsli) and the CPU-side TileAlignedPx (Renderer.h). The old
//4x8-shaped formula diverged from MapPixelID's actual max index at
//non-tile-divisible resolutions, overlapping the SoA planes.
uint ps_numPx() { return ((IMG_W + 7u) / 8u) * ((IMG_H + 3u) / 4u) * 32u; }

uint ps_addr_pack1   (uint px) { return px * PS_SZ_PACK1; }
uint ps_addr_pack2   (uint px) { uint N = ps_numPx(); return N * PS_PLANE_PACK2    + px * PS_SZ_PACK2; }
uint ps_addr_v2      (uint px) { uint N = ps_numPx(); return N * PS_PLANE_V2       + px * PS_SZ_V2; }
uint ps_addr_hot1    (uint px) { uint N = ps_numPx(); return N * PS_PLANE_HOT1     + px * PS_SZ_HOT1; }
uint ps_addr_hot2    (uint px) { uint N = ps_numPx(); return N * PS_PLANE_HOT2     + px * PS_SZ_HOT2; }
uint ps_addr_seed    (uint px) { uint N = ps_numPx(); return N * PS_PLANE_SEED     + px * PS_SZ_SEED; }
uint ps_addr_ray_o   (uint px) { uint N = ps_numPx(); return N * PS_PLANE_RAY_O    + px * PS_SZ_RAY_O; }
uint ps_addr_ray_d   (uint px) { uint N = ps_numPx(); return N * PS_PLANE_RAY_D    + px * PS_SZ_RAY_D; }
uint ps_addr_hp      (uint px) { uint N = ps_numPx(); return N * PS_PLANE_HP       + px * PS_SZ_HP; }
uint ps_addr_clas1   (uint px) { uint N = ps_numPx(); return N * PS_PLANE_CLAS1    + px * PS_SZ_CLAS1; }
uint ps_addr_clas2   (uint px) { uint N = ps_numPx(); return N * PS_PLANE_CLAS2    + px * PS_SZ_CLAS2; }
uint ps_addr_cand_wi (uint px) { uint N = ps_numPx(); return N * PS_PLANE_CAND_WI  + px * PS_SZ_CAND_WI; }
uint ps_addr_cand_m  (uint px) { uint N = ps_numPx(); return N * PS_PLANE_CAND_M   + px * PS_SZ_CAND_M; }
uint ps_addr_cand_lp (uint px) { uint N = ps_numPx(); return N * PS_PLANE_CAND_LP  + px * PS_SZ_CAND_LP; }
uint ps_addr_cand_ln (uint px) { uint N = ps_numPx(); return N * PS_PLANE_CAND_LN  + px * PS_SZ_CAND_LN; }


//====================================
//RAYGEN-ONLY SCRATCH (aliases HOT1 + HOT2)
//====================================
//Raygen and the wavefront variant share g_pathStateBuffer but never run on
//the same frame. The wavefront's HOT1/HOT2 planes are unused under the
//raygen path, so we reuse those 32B as a scratch region for values the
//compiler would otherwise keep live across the bounce-loop TraceRay reorder
//boundary. g_pathStateBuffer is declared globallycoherent so the spill+reload
//pattern is not defeated by store-to-load forwarding.
//
//HOT1 layout (16B):
//  bytes  0..11  tpost   (float3, full fp32 precision -- multiplicative
//                          throughput chain that must not lose bits)
//  bytes 12..15  nrcA0   (float, set once at primary hit, read each bounce
//                          inside the cache-termination check)
//HOT2 layout (16B):
//  bytes  0..3   nrcA          (accumulated area spread, RMW each bounce)
//  bytes  4..7   nrcEmitMask   (per-vertex emit-eligibility bits, OR'd in
//                                training-vertex emit, read at finalize)
//  bytes  8..15  free
static const uint PS_RG_OFF_TPOST       = 0u;
static const uint PS_RG_OFF_NRCA0       = 12u;
static const uint PS_RG2_OFF_NRCA       = 0u;
static const uint PS_RG2_OFF_EMITMASK   = 4u;

void store_rg_tpost(RWByteAddressBuffer buf, uint pixelIdx, float3 tpost)
{
    buf.Store3(ps_addr_hot1(pixelIdx) + PS_RG_OFF_TPOST, asuint(tpost));
}

float3 load_rg_tpost(RWByteAddressBuffer buf, uint pixelIdx)
{
    return asfloat(buf.Load3(ps_addr_hot1(pixelIdx) + PS_RG_OFF_TPOST));
}

void store_rg_nrcA0(RWByteAddressBuffer buf, uint pixelIdx, float a0)
{
    buf.Store(ps_addr_hot1(pixelIdx) + PS_RG_OFF_NRCA0, asuint(a0));
}

float load_rg_nrcA0(RWByteAddressBuffer buf, uint pixelIdx)
{
    return asfloat(buf.Load(ps_addr_hot1(pixelIdx) + PS_RG_OFF_NRCA0));
}

void store_rg_nrcA(RWByteAddressBuffer buf, uint pixelIdx, float a)
{
    buf.Store(ps_addr_hot2(pixelIdx) + PS_RG2_OFF_NRCA, asuint(a));
}

float load_rg_nrcA(RWByteAddressBuffer buf, uint pixelIdx)
{
    return asfloat(buf.Load(ps_addr_hot2(pixelIdx) + PS_RG2_OFF_NRCA));
}

void store_rg_nrcEmitMask(RWByteAddressBuffer buf, uint pixelIdx, uint mask)
{
    buf.Store(ps_addr_hot2(pixelIdx) + PS_RG2_OFF_EMITMASK, mask);
}

uint load_rg_nrcEmitMask(RWByteAddressBuffer buf, uint pixelIdx)
{
    return buf.Load(ps_addr_hot2(pixelIdx) + PS_RG2_OFF_EMITMASK);
}

//atomic OR forces a memory op the compiler cannot fold into a register-held
//copy. Used by the training-vertex emit-eligibility update so nrcEmitMask
//never enters the live set.
void atomic_or_rg_nrcEmitMask(RWByteAddressBuffer buf, uint pixelIdx, uint bit)
{
    buf.InterlockedOr(ps_addr_hot2(pixelIdx) + PS_RG2_OFF_EMITMASK, bit);
}


//====================================
//FLAGS BIT LAYOUT
//====================================
//bits 0-5   depth (0..63)
//bits 6-9   dsBounces (0..15)
//bit  10    TERMINATED
//bit  11    PERFORM_NEE
//bit  12    HAS_VALID_HIT
//bit  13    IS_BACKFACE
//bit  14    FLIP_IOR
//bit  15    TRANSMISSIVE
//bit  16    IS_EMITTER
#define PS_FLAG_TERMINATED     (1u << 10)
#define PS_FLAG_PERFORM_NEE    (1u << 11)
#define PS_FLAG_HAS_VALID_HIT  (1u << 12)
#define PS_FLAG_IS_BACKFACE    (1u << 13)
#define PS_FLAG_FLIP_IOR       (1u << 14)
#define PS_FLAG_TRANSMISSIVE   (1u << 15)
#define PS_FLAG_IS_EMITTER     (1u << 16)

#define PS_DEPTH_MASK      0x3Fu
#define PS_DEPTH_SHIFT     0u
#define PS_DSBOUNCES_MASK  0xFu
#define PS_DSBOUNCES_SHIFT 6u

uint ps_get_depth(uint flags)      { return (flags >> PS_DEPTH_SHIFT) & PS_DEPTH_MASK; }
uint ps_get_dsBounces(uint flags)  { return (flags >> PS_DSBOUNCES_SHIFT) & PS_DSBOUNCES_MASK; }
uint ps_set_depth(uint flags, uint d)
{
    return (flags & ~(PS_DEPTH_MASK << PS_DEPTH_SHIFT)) | ((d & PS_DEPTH_MASK) << PS_DEPTH_SHIFT);
}
uint ps_set_dsBounces(uint flags, uint n)
{
    return (flags & ~(PS_DSBOUNCES_MASK << PS_DSBOUNCES_SHIFT)) | ((n & PS_DSBOUNCES_MASK) << PS_DSBOUNCES_SHIFT);
}


//====================================
//PATH VERTEX STATE
//====================================
struct PathVertexState {
    float3 x2;
    float3 n2_s;
    uint   matID;
    uint   objID;
    float  eta;
    float3 Kd;
    float  Pr;
    float  Pm;
    float3 v2;
};


//v_2 vertex stashed at depth==1, feeds depth>=3 reservoir candidates. Carries
//the resolved material (Kd/Pr/Pm) so depth>=3 candidates need no re-fetch.
//Pr|Pm ride the CLAS2 plane, which lives in the backed raygen-live prefix of
//the buffer (see the plane table above).
void store_ps_depth1(RWByteAddressBuffer buf, uint pixelIdx,
                     float3 x2_world, float3 n2_world,
                     uint matID, uint objID, float eta,
                     float3 Kd, float Pr, float Pm)
{
    buf.Store4(ps_addr_pack1(pixelIdx), uint4(asuint(x2_world), PackNormal(n2_world)));
    buf.Store4(ps_addr_pack2(pixelIdx), uint4(matID, objID, asuint(eta), PackRGB9E5(Kd)));
    buf.Store (ps_addr_clas2(pixelIdx), PackFloat2x16(Pr, Pm));
}

void store_ps_v2(RWByteAddressBuffer buf, uint pixelIdx, float3 v2_world)
{
    buf.Store(ps_addr_v2(pixelIdx), PackNormal(v2_world));
}


PathVertexState load_ps(RWByteAddressBuffer buf, uint pixelIdx)
{
    PathVertexState s;

    const uint4 p1 = buf.Load4(ps_addr_pack1(pixelIdx));
    s.x2   = asfloat(p1.xyz);
    s.n2_s = UnpackNormal(p1.w);

    const uint4 p2 = buf.Load4(ps_addr_pack2(pixelIdx));
    s.matID = p2.x;
    s.objID = p2.y;
    s.eta   = asfloat(p2.z);
    s.Kd    = UnpackRGB9E5(p2.w);

    UnpackFloat2x16(buf.Load(ps_addr_clas2(pixelIdx)), s.Pr, s.Pm);

    s.v2 = UnpackNormal(buf.Load(ps_addr_v2(pixelIdx)));
    return s;
}


//====================================
//PATH STATE INIT
//====================================
//MATID_ENV_MISS sentinel so Reconnect doesn't read stale prior-frame spat data
void init_ps(RWByteAddressBuffer buf, uint pixelIdx)
{
    buf.Store4(ps_addr_pack1(pixelIdx),
               uint4(asuint(float3(0, 0, 0)), PackNormal(float3(0, 1, 0))));
    buf.Store4(ps_addr_pack2(pixelIdx),
               uint4(MATID_ENV_MISS, MATID_ENV_MISS, asuint(1.0f), 0u));
    buf.Store(ps_addr_clas2(pixelIdx), 0u);
    buf.Store(ps_addr_v2(pixelIdx), PackNormal(float3(0, 1, 0)));
}


//====================================
//HOT REGISTER PLANES
//====================================
struct HotState {
    uint   throughputPk;
    uint   prevNormalPk;
    float  prev_pdf;
    float  pdf_product;
    uint   tpostPk;
    float  wsum;
    uint   flags;
    uint   seed;
};

void store_hot1(RWByteAddressBuffer buf, uint pixelIdx,
                uint throughputPk, uint prevNormalPk, float prev_pdf, float pdf_product)
{
    buf.Store4(ps_addr_hot1(pixelIdx),
               uint4(throughputPk, prevNormalPk, asuint(prev_pdf), asuint(pdf_product)));
}

void store_hot2(RWByteAddressBuffer buf, uint pixelIdx,
                uint tpostPk, float wsum, uint flags)
{
    buf.Store4(ps_addr_hot2(pixelIdx),
               uint4(tpostPk, asuint(wsum), flags, 0u));
}

void store_seed(RWByteAddressBuffer buf, uint pixelIdx, uint seed)
{
    buf.Store(ps_addr_seed(pixelIdx), seed);
}

//HOT2 layout is [tpost, wsum, flags, pad], rewrite only flags word
void store_flags(RWByteAddressBuffer buf, uint pixelIdx, uint flags)
{
    buf.Store(ps_addr_hot2(pixelIdx) + 8u, flags);
}

void store_throughput_pk(RWByteAddressBuffer buf, uint pixelIdx, uint throughputPk)
{
    buf.Store(ps_addr_hot1(pixelIdx) + 0u, throughputPk);
}

void store_prev_pdf_normal(RWByteAddressBuffer buf, uint pixelIdx,
                           uint prevNormalPk, float prev_pdf)
{
    buf.Store2(ps_addr_hot1(pixelIdx) + 4u, uint2(prevNormalPk, asuint(prev_pdf)));
}

void store_pdf_product(RWByteAddressBuffer buf, uint pixelIdx, float pdf_product)
{
    buf.Store(ps_addr_hot1(pixelIdx) + 12u, asuint(pdf_product));
}

void store_tpost_pk(RWByteAddressBuffer buf, uint pixelIdx, uint tpostPk)
{
    buf.Store(ps_addr_hot2(pixelIdx) + 0u, tpostPk);
}

void store_ps_wsum(RWByteAddressBuffer buf, uint pixelIdx, float wsum)
{
    buf.Store(ps_addr_hot2(pixelIdx) + 4u, asuint(wsum));
}

HotState load_hot(RWByteAddressBuffer buf, uint pixelIdx)
{
    HotState h;
    const uint4 a = buf.Load4(ps_addr_hot1(pixelIdx));
    h.throughputPk = a.x;
    h.prevNormalPk = a.y;
    h.prev_pdf     = asfloat(a.z);
    h.pdf_product  = asfloat(a.w);

    const uint4 b = buf.Load4(ps_addr_hot2(pixelIdx));
    h.tpostPk = b.x;
    h.wsum    = asfloat(b.y);
    h.flags   = b.z;

    h.seed = buf.Load(ps_addr_seed(pixelIdx));
    return h;
}

uint load_flags(RWByteAddressBuffer buf, uint pixelIdx)
{
    return buf.Load(ps_addr_hot2(pixelIdx) + 8u);
}

float load_wsum_ps(RWByteAddressBuffer buf, uint pixelIdx)
{
    return asfloat(buf.Load(ps_addr_hot2(pixelIdx) + 4u));
}


//====================================
//RAY PLANES
//====================================
void store_ray(RWByteAddressBuffer buf, uint pixelIdx, float3 rayOrigin, float3 rayDir)
{
    buf.Store3(ps_addr_ray_o(pixelIdx), asuint(rayOrigin));
    buf.Store3(ps_addr_ray_d(pixelIdx), asuint(rayDir));
}

void store_ray_origin(RWByteAddressBuffer buf, uint pixelIdx, float3 rayOrigin)
{
    buf.Store3(ps_addr_ray_o(pixelIdx), asuint(rayOrigin));
}

float3 load_ray_origin(RWByteAddressBuffer buf, uint pixelIdx)
{
    return asfloat(buf.Load3(ps_addr_ray_o(pixelIdx)));
}

float3 load_ray_dir(RWByteAddressBuffer buf, uint pixelIdx)
{
    return asfloat(buf.Load3(ps_addr_ray_d(pixelIdx)));
}


//====================================
//HIT PACKET
//====================================
struct HitPacket {
    float  hitT;
    uint   instID;
    uint   primID;
    float2 bary;
    bool   isHit;
};

void store_hp(RWByteAddressBuffer buf, uint pixelIdx,
              float hitT, uint instID, uint primID, float2 bary)
{
    buf.Store4(ps_addr_hp(pixelIdx),
               uint4(asuint(hitT), instID, primID,
                     PackFloat2x16(bary.x, bary.y)));
}

//miss sentinel, instID = 0xFFFFFFFF, other fields don't matter
void store_hp_miss(RWByteAddressBuffer buf, uint pixelIdx)
{
    buf.Store(ps_addr_hp(pixelIdx) + 4u, 0xFFFFFFFFu);
}

HitPacket load_hp(RWByteAddressBuffer buf, uint pixelIdx)
{
    HitPacket hp;
    const uint4 w = buf.Load4(ps_addr_hp(pixelIdx));
    hp.hitT   = asfloat(w.x);
    hp.instID = w.y;
    hp.primID = w.z;
    UnpackFloat2x16(w.w, hp.bary.x, hp.bary.y);
    hp.isHit  = (hp.instID != 0xFFFFFFFFu);
    return hp;
}


//====================================
//CLASSIFY OUTPUT
//====================================
struct ClassifyState {
    uint   hitNormalPk;
    uint   matID;
    float  etaOut;
    float2 uv;
    uint   absTintPk;
};

void store_clas(RWByteAddressBuffer buf, uint pixelIdx,
                uint hitNormalPk, uint matID, float etaOut, float2 uv, uint absTintPk)
{
    buf.Store4(ps_addr_clas1(pixelIdx),
               uint4(hitNormalPk, matID, asuint(etaOut), PackFloat2x16(uv.x, uv.y)));
    buf.Store (ps_addr_clas2(pixelIdx), absTintPk);
}

ClassifyState load_clas(RWByteAddressBuffer buf, uint pixelIdx)
{
    ClassifyState c;
    const uint4 w = buf.Load4(ps_addr_clas1(pixelIdx));
    c.hitNormalPk = w.x;
    c.matID       = w.y;
    c.etaOut      = asfloat(w.z);
    UnpackFloat2x16(w.w, c.uv.x, c.uv.y);
    c.absTintPk   = buf.Load(ps_addr_clas2(pixelIdx));
    return c;
}


//====================================
//CANDIDATE DESCRIPTOR
//====================================
//meta bit 0 VALID, bits 1-2 KIND, invalid lanes early-return in MatEval
#define CAND_META_VALID      (1u << 0)
#define CAND_META_KIND_SHIFT 1u
#define CAND_META_KIND_MASK  0x3u
#define CAND_KIND_POINT_TRI  1u
#define CAND_KIND_SUN        2u

struct CandidateDesc {
    float3 wiDir;
    float3 L;
    float  lightPdf;
    uint   meta;
    uint   lightObjID;
    float3 lightPos;
    float3 lightN;
};

void store_cand_invalid(RWByteAddressBuffer buf, uint pixelIdx)
{
    //only meta word matters, rest is don't-care
    buf.Store(ps_addr_cand_m(pixelIdx) + 8u, 0u);
}

void store_cand_point_tri(RWByteAddressBuffer buf, uint pixelIdx,
                          float3 wiDir, float3 L, float lightPdf,
                          uint lightObjID,
                          float3 lightPos, float3 lightN)
{
    buf.Store3(ps_addr_cand_wi(pixelIdx), asuint(wiDir));
    const uint meta = CAND_META_VALID | (CAND_KIND_POINT_TRI << CAND_META_KIND_SHIFT);
    buf.Store4(ps_addr_cand_m(pixelIdx),
               uint4(PackRGB9E5(L), asuint(lightPdf), meta, lightObjID));
    buf.Store3(ps_addr_cand_lp(pixelIdx), asuint(lightPos));
    buf.Store (ps_addr_cand_ln(pixelIdx), PackNormal(lightN));
}

void store_cand_sun(RWByteAddressBuffer buf, uint pixelIdx,
                    float3 wiDir, float3 radiance, float lightPdf)
{
    buf.Store3(ps_addr_cand_wi(pixelIdx), asuint(wiDir));
    const uint meta = CAND_META_VALID | (CAND_KIND_SUN << CAND_META_KIND_SHIFT);
    buf.Store4(ps_addr_cand_m(pixelIdx),
               uint4(PackRGB9E5(radiance), asuint(lightPdf), meta, 0u));
    //lightPos/lightN unused for sun
}

CandidateDesc load_cand(RWByteAddressBuffer buf, uint pixelIdx)
{
    CandidateDesc c;
    c.wiDir = asfloat(buf.Load3(ps_addr_cand_wi(pixelIdx)));

    const uint4 m = buf.Load4(ps_addr_cand_m(pixelIdx));
    c.L          = UnpackRGB9E5(m.x);
    c.lightPdf   = asfloat(m.y);
    c.meta       = m.z;
    c.lightObjID = m.w;

    const uint kind = (c.meta >> CAND_META_KIND_SHIFT) & CAND_META_KIND_MASK;
    if (kind == CAND_KIND_POINT_TRI) {
        c.lightPos = asfloat(buf.Load3(ps_addr_cand_lp(pixelIdx)));
        c.lightN   = UnpackNormal(buf.Load(ps_addr_cand_ln(pixelIdx)));
    } else {
        c.lightPos = float3(0, 0, 0);
        c.lightN   = float3(0, 1, 0);
    }
    return c;
}

bool cand_valid(uint meta) { return (meta & CAND_META_VALID) != 0u; }
uint cand_kind (uint meta) { return (meta >> CAND_META_KIND_SHIFT) & CAND_META_KIND_MASK; }


//====================================
//DI MARKER
//====================================
//per-pixel unit vec for dup-map discrimination, closed-form hash, no storage
inline float3 diMarkerFor(uint pixelIdx, float frameTime)
{
    uint h = (pixelIdx * 0x9E3779B9u) ^ (asuint(frameTime) * 0x85EBCA6Bu);
    h ^= h >> 16; h *= 0xC2B2AE35u;
    h ^= h >> 13; h *= 0x27D4EB2Fu;
    h ^= h >> 16;
    const float u1 = (float)(h & 0xFFFFu) * (1.0f / 65535.0f);
    const float u2 = (float)((h >> 16) & 0xFFFFu) * (1.0f / 65535.0f);
    const float z  = 2.0f * u1 - 1.0f;
    const float r  = sqrt(max(0.0f, 1.0f - z * z));
    const float phi = 6.2831853f * u2;
    return float3(r * cos(phi), r * sin(phi), z);
}
