// Byte offsets are shared with the host path-state buffer.
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

static const uint PS_RG_OFF_TPOST    =  0u;
static const uint PS_RG_OFF_DIRECTX1 = 12u;

void store_rg_tpost(RWByteAddressBuffer buf, uint pixelIdx, float3 tpost)
{
    buf.Store3(ps_addr_hot1(pixelIdx) + PS_RG_OFF_TPOST, asuint(tpost));
}

float3 load_rg_tpost(RWByteAddressBuffer buf, uint pixelIdx)
{
    return asfloat(buf.Load3(ps_addr_hot1(pixelIdx) + PS_RG_OFF_TPOST));
}

void store_rg_directX1(RWByteAddressBuffer buf, uint pixelIdx, float3 v)
{
    buf.Store(ps_addr_hot1(pixelIdx) + PS_RG_OFF_DIRECTX1, PackRGB9E5(v));
}

float3 load_rg_directX1(RWByteAddressBuffer buf, uint pixelIdx)
{
    return UnpackRGB9E5(buf.Load(ps_addr_hot1(pixelIdx) + PS_RG_OFF_DIRECTX1));
}

void add_rg_directX1(RWByteAddressBuffer buf, uint pixelIdx, float3 add)
{
    if (!any(add > 0.0f)) return;
    const uint a = ps_addr_hot1(pixelIdx) + PS_RG_OFF_DIRECTX1;
    buf.Store(a, PackRGB9E5(UnpackRGB9E5(buf.Load(a)) + add));
}

void store_rg_primaryExtra(RWByteAddressBuffer buf, uint pixelIdx,
                           float2 iors, uint mediumMatID, float3 absorptionTint)
{
    const uint base = ps_addr_hot2(pixelIdx);
    buf.Store(base +  0u, f32tof16(iors.x) | (f32tof16(iors.y) << 16));
    buf.Store(base +  4u, mediumMatID);
    buf.Store(base +  8u, f32tof16(absorptionTint.x) | (f32tof16(absorptionTint.y) << 16));
    buf.Store(base + 12u, f32tof16(absorptionTint.z));
}

void load_rg_primaryExtra(RWByteAddressBuffer buf, uint pixelIdx,
                          out float2 iors, out uint mediumMatID, out float3 absorptionTint)
{
    const uint base = ps_addr_hot2(pixelIdx);
    const uint pk0  = buf.Load(base + 0u);
    iors            = float2(f16tof32(pk0 & 0xFFFFu), f16tof32(pk0 >> 16));
    mediumMatID     = buf.Load(base + 4u);
    const uint pk1  = buf.Load(base + 8u);
    absorptionTint  = float3(f16tof32(pk1 & 0xFFFFu), f16tof32(pk1 >> 16),
                             f16tof32(buf.Load(base + 12u) & 0xFFFFu));
}

static const uint PT_NEE_PREFETCH_SALT = 0x4e454531u;

uint pt_neePrefetchTag() { return Hash32(asuint(time) ^ PT_NEE_PREFETCH_SALT); }

// Publish NEE samples with a frame tag and optional learning tokens.
void store_pt_neePrefetch(RWByteAddressBuffer buf, uint pixelIdx, uint tri, uint inst, float pdf, uint rng, uint2 learningToken=0u)
{
    buf.Store4(ps_addr_hot1(pixelIdx), uint4(tri, asuint(pdf), rng, pt_neePrefetchTag()));

    buf.Store(ps_addr_clas2(pixelIdx),learningToken.x);
    buf.Store(ps_addr_ray_o(pixelIdx),learningToken.y);
    buf.Store(ps_addr_ray_o(pixelIdx) + 4u, inst);
}

// Reject stale NEE data before restoring its random stream.
bool load_pt_neePrefetch(RWByteAddressBuffer buf, uint pixelIdx, out uint tri, out uint inst, out float pdf, inout uint rng, out uint2 learningToken)
{
    const uint4 w = buf.Load4(ps_addr_hot1(pixelIdx));
    tri = w.x;
    inst = 0xFFFFFFFFu;
    pdf = asfloat(w.y);
    learningToken=0u;
    if (w.w != pt_neePrefetchTag()) return false;
    rng = w.z;
    learningToken=uint2(buf.Load(ps_addr_clas2(pixelIdx)),buf.Load(ps_addr_ray_o(pixelIdx)));
    inst = buf.Load(ps_addr_ray_o(pixelIdx) + 4u);
    return true;
}

static const uint PS_GUIDE_ROOT_BASE  = 88u;
static const uint PS_GUIDE_ROOT_BYTES = 32u;
static const uint PS_GUIDE_ROOT_END   = 168u;
uint ps_addr_guideRoot(uint lane, uint r)
{
    return ps_numPx() * PS_GUIDE_ROOT_BASE + (lane * 2u + r) * PS_GUIDE_ROOT_BYTES;
}
bool ps_guideRootBacked(uint lane)
{
    return (lane + 1u) * (2u * PS_GUIDE_ROOT_BYTES) <= ps_numPx() * (PS_GUIDE_ROOT_END - PS_GUIDE_ROOT_BASE);
}

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

// Decode the reconnectable vertex state from packed planes.
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

// Clear the persistent path state before the first bounce.
void init_ps(RWByteAddressBuffer buf, uint pixelIdx)
{
    buf.Store4(ps_addr_pack1(pixelIdx),
               uint4(asuint(float3(0, 0, 0)), PackNormal(float3(0, 1, 0))));
    buf.Store4(ps_addr_pack2(pixelIdx),
               uint4(MATID_ENV_MISS, MATID_ENV_MISS, asuint(1.0f), 0u));
    buf.Store(ps_addr_clas2(pixelIdx), 0u);
    buf.Store(ps_addr_v2(pixelIdx), PackNormal(float3(0, 1, 0)));
}

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
