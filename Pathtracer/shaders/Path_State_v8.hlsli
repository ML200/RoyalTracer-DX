//====================================================================
//PER-PIXEL PATH VERTEX STATE SCRATCH
//====================================================================
//Wavefront pipeline: Pass_primary -> (Extend -> Classify -> MatEvalShade)*
//-> Pass_finalize. All per-pixel state that crosses stage boundaries
//lives here, SoA so each plane is a single contiguous range.
//
//Raygen-side usage, overwritten by Pass_spat_gi_select_v8 later in the frame.
//
//Existing stash planes (kept, also used by spatial reuse path):
//plane 0  PACK1   16B  x2.xyz + n2_pk             depth-1 vertex
//plane 1  PACK2   16B  uv_pk + matID + objID+eta  depth-1 vertex
//plane 2  V2       4B  v2_pk                      depth-2 direction
//
//Hot register spill (every stage reads, Scatter writes):
//plane 3  HOT1    16B  throughputPk + prevNormalPk + prev_pdf + pdf_product
//plane 4  HOT2    16B  tpostPk + wsum + flags + _pad
//plane 5  SEED     4B  seed
//
//Next-bounce ray (Scatter/Primary writes, Extend reads):
//plane 6  RAY_O   12B  rayOrigin
//plane 7  RAY_D   12B  rayDir
//
//Hit packet (Extend writes, Classify/MatEvalShade read):
//plane 8  HP      16B  hitT + instID + primID + baryPk
//
//Classify output (Classify writes, MatEvalShade reads):
//plane 9  CLAS1   16B  hitNormalPk + matID + etaOut + uvPk
//plane 10 CLAS2    4B  absTintPk (medium absorption for Scatter throughput)
//
//Candidate descriptor (sampling stages write, MatEval stages read):
//plane 11 CAND_WI 12B  wiDir
//plane 12 CAND_M  16B  L_pk + lightPdf + meta + lightObjID
//plane 13 CAND_LP 12B  lightPos
//plane 14 CAND_LN  4B  lightN_pk
//
//Total: 176 B/pixel (tile-aligned via ps_numPx()).
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
static const uint PS_PLANE_SEED     =  68u;
static const uint PS_PLANE_RAY_O    =  72u;
static const uint PS_PLANE_RAY_D    =  84u;
static const uint PS_PLANE_HP       =  96u;
static const uint PS_PLANE_CLAS1    = 112u;
static const uint PS_PLANE_CLAS2    = 128u;
static const uint PS_PLANE_CAND_WI  = 132u;
static const uint PS_PLANE_CAND_M   = 144u;
static const uint PS_PLANE_CAND_LP  = 160u;
static const uint PS_PLANE_CAND_LN  = 172u;

//Tile-aligned pixel count, matches MapPixelID's 4x8 tile swizzle. Host
//allocation must use the same formula to size m_pathStateBuffer.
uint ps_numPx() { return ((IMG_W + 3u) / 4u) * ((IMG_H + 7u) / 8u) * 32u; }

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


//====================================================================
//FLAGS BIT LAYOUT
//====================================================================
//bits  0-5  depth (0..63)
//bits  6-9  dsBounces (0..15)
//bit   10   TERMINATED         path killed, downstream stages skip
//bit   11   PERFORM_NEE        Classify set; MatEvalShade gates NEE
//bit   12   HAS_VALID_HIT      Classify set for real hits
//bit   13   IS_BACKFACE        for material routing
//bit   14   FLIP_IOR           backface && transmissive
//bit   15   TRANSMISSIVE       Kd.w < 1-EPS
//bit   16   IS_EMITTER         emitter hit, candidate already filed
//bits 17-31 reserved
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


//====================================================================
//PATH VERTEX STATE (existing, used by the depth-1 GI branch)
//====================================================================
struct PathVertexState {
    float3 x2;
    float3 n2_s;
    float2 uv;
    uint   matID;
    uint   objID;
    float  eta;
    float3 v2;
};


void store_ps_depth1(RWByteAddressBuffer buf, uint pixelIdx,
                     float3 x2_world, float3 n2_world,
                     float2 uv, uint matID, uint objID, float eta)
{
    buf.Store4(ps_addr_pack1(pixelIdx), uint4(asuint(x2_world), PackNormal(n2_world)));
    buf.Store4(ps_addr_pack2(pixelIdx), uint4(PackFloat2x16(uv.x, uv.y),
                                              matID, objID, asuint(eta)));
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
    UnpackFloat2x16(p2.x, s.uv.x, s.uv.y);
    s.matID = p2.y;
    s.objID = p2.z;
    s.eta   = asfloat(p2.w);

    s.v2 = UnpackNormal(buf.Load(ps_addr_v2(pixelIdx)));
    return s;
}


//====================================================================
//PATH STATE INIT (per-frame, Pass_primary)
//====================================================================
//Seeds the depth-1 stash with MATID_ENV_MISS sentinels so any sentinel-
//branch in Reconnect doesn't read prior-frame spat-pass scratch.
//Callers follow up with store_hot1 / store_hot2 / store_seed / store_ray
//to write the per-pixel initial hot state.
void init_ps(RWByteAddressBuffer buf, uint pixelIdx)
{
    buf.Store4(ps_addr_pack1(pixelIdx),
               uint4(asuint(float3(0, 0, 0)), PackNormal(float3(0, 1, 0))));
    buf.Store4(ps_addr_pack2(pixelIdx),
               uint4(PackFloat2x16(0.0f, 0.0f),
                     MATID_ENV_MISS, MATID_ENV_MISS, asuint(1.0f)));
    buf.Store(ps_addr_v2(pixelIdx), PackNormal(float3(0, 1, 0)));
}


//====================================================================
//HOT REGISTER PLANES
//====================================================================
struct HotState {
    uint   throughputPk;   //RGB9E5 throughput
    uint   prevNormalPk;   //packed normal of previous vertex (for emitter MIS)
    float  prev_pdf;       //BSDF pdf at previous vertex
    float  pdf_product;    //cumulative product of BSDF pdfs along path
    uint   tpostPk;        //RGB9E5 post-x2 integrand accumulator
    float  wsum;           //RIS running sum
    uint   flags;          //bit-packed, see PS_FLAG_* macros
    uint   seed;           //RNG state
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

void store_flags(RWByteAddressBuffer buf, uint pixelIdx, uint flags)
{
    //HOT2 layout: [tpost, wsum, flags, _pad]. Rewrite only the flags word.
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


//====================================================================
//RAY PLANES
//====================================================================
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


//====================================================================
//HIT PACKET
//====================================================================
struct HitPacket {
    float  hitT;
    uint   instID;
    uint   primID;
    float2 bary;
    bool   isHit;  //encoded via instID == 0xFFFFFFFFu
};

void store_hp(RWByteAddressBuffer buf, uint pixelIdx,
              float hitT, uint instID, uint primID, float2 bary)
{
    buf.Store4(ps_addr_hp(pixelIdx),
               uint4(asuint(hitT), instID, primID,
                     PackFloat2x16(bary.x, bary.y)));
}

//Encodes a miss by writing instID = 0xFFFFFFFF sentinel. Other fields
//don't matter; downstream reads only the sentinel on miss.
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


//====================================================================
//CLASSIFY OUTPUT
//====================================================================
struct ClassifyState {
    uint   hitNormalPk;
    uint   matID;
    float  etaOut;      //iors.y, "outgoing medium" IOR for Scatter/NEE BSDF evals
    float2 uv;          //surface uv at the hit
    uint   absTintPk;   //RGB9E5 absorption throughput for the just-traversed segment
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


//====================================================================
//CANDIDATE DESCRIPTOR (scratch, overwritten per sampling->MatEval pair)
//====================================================================
//meta: bit 0 = VALID, bits 1-2 = KIND. KIND_INVALID = 0, KIND_POINT_TRI = 1,
//KIND_SUN = 2. On invalid lanes, downstream MatEval early-returns.
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
    float3 lightPos;    //only meaningful for KIND_POINT_TRI
    float3 lightN;      //only meaningful for KIND_POINT_TRI
};

void store_cand_invalid(RWByteAddressBuffer buf, uint pixelIdx)
{
    //Only clearing the meta word is necessary; rest is don't-care.
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
    //lightPos / lightN unused for sun; leave whatever's there.
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


//====================================================================
//DI MARKER (per-pixel per-frame, unit vector, dup-map discriminator)
//====================================================================
//Closed-form hash from (pixelIdx, time) -> unit vector. Regenerated in
//any stage that needs it for DI candidate V2 payloads; no storage.
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
