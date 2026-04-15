// RIS reservoir for global illumination
struct Reservoir_GI
{
    // Constant-after-hit payload
    float3 x2_gi;
    float3 n2_s_gi;
    float3 n2_g_gi;
    uint   objID_gi;
    uint   matID_gi;
    float2 uv_gi;
    float  etai_gi;
    float  etat_gi;

    // Varying payload
    float3 L2_gi;
    float3 V2_gi;
    float2 J_gi;
    uint   F_gi;        // RGB9E5-packed contribution (replaces scalar luminance)

    // Reservoir bookkeeping
    float  W_gi;
    float  w_sum_gi;
    uint   M_gi;

};


// SoA layout — each field is a contiguous plane across all pixels.
// Total buffer size per pixel is unchanged (80 bytes).
// Layout: PACK1(16) | L2(4) | V2(4) | N2G(4) | OBJID(4) | UV(4) | IOR(4) | MATID(4) |
//         J(8) | W(4) | F(4) | M(4) | WSUM(4) | VPOST(4) | TPOST(4) = 76

static const uint BYTES_GI       = 68u;
static const uint BYTES_GI_VPOST =  4u;
static const uint BYTES_GI_TPOST =  4u;
static const uint STRIDE_GI      = BYTES_GI + BYTES_GI_VPOST + BYTES_GI_TPOST; // 76

// Per-field sizes
static const uint GI_SZ_PACK1 = 16u;  // x2(12) + n2_s_packed(4)
static const uint GI_SZ_L2    =  4u;
static const uint GI_SZ_V2    =  4u;
static const uint GI_SZ_N2G   =  4u;
static const uint GI_SZ_OBJID =  4u;
static const uint GI_SZ_UV    =  4u;
static const uint GI_SZ_IOR   =  4u;  // etai(fp16) + etat(fp16)
static const uint GI_SZ_MATID =  4u;
static const uint GI_SZ_J     =  8u;  // J.x(4) + J.y(4)
static const uint GI_SZ_W     =  4u;
static const uint GI_SZ_F     =  4u;
static const uint GI_SZ_M     =  4u;
static const uint GI_SZ_WSUM  =  4u;
static const uint GI_SZ_VPOST =  4u;
static const uint GI_SZ_TPOST =  4u;

// Plane cumulative offsets (per-pixel contribution to base address)
static const uint GI_PLANE_PACK1 =  0u;
static const uint GI_PLANE_L2    = 16u;
static const uint GI_PLANE_V2    = 20u;
static const uint GI_PLANE_N2G   = 24u;
static const uint GI_PLANE_OBJID = 28u;
static const uint GI_PLANE_UV    = 32u;
static const uint GI_PLANE_IOR   = 36u;
static const uint GI_PLANE_MATID = 40u;
static const uint GI_PLANE_J     = 44u;
static const uint GI_PLANE_W     = 52u;
static const uint GI_PLANE_F     = 56u;
static const uint GI_PLANE_M     = 60u;
static const uint GI_PLANE_WSUM  = 64u;
static const uint GI_PLANE_VPOST = 68u;
static const uint GI_PLANE_TPOST = 72u;

// SoA address helpers
// Tile-aligned pixel count — must match MapPixelID's 4x8 tile swizzle.
uint gi_numPx()                       { return ((IMG_W + 3u) / 4u) * ((IMG_H + 7u) / 8u) * 32u; }
uint gi_addr_pack1(uint px)           { return px * GI_SZ_PACK1; }
uint gi_addr_l2(uint px)              { uint N = gi_numPx(); return N * GI_PLANE_L2    + px * GI_SZ_L2; }
uint gi_addr_v2(uint px)              { uint N = gi_numPx(); return N * GI_PLANE_V2    + px * GI_SZ_V2; }
uint gi_addr_n2g(uint px)             { uint N = gi_numPx(); return N * GI_PLANE_N2G   + px * GI_SZ_N2G; }
uint gi_addr_objid(uint px)           { uint N = gi_numPx(); return N * GI_PLANE_OBJID + px * GI_SZ_OBJID; }
uint gi_addr_uv(uint px)             { uint N = gi_numPx(); return N * GI_PLANE_UV    + px * GI_SZ_UV; }
uint gi_addr_ior(uint px)            { uint N = gi_numPx(); return N * GI_PLANE_IOR   + px * GI_SZ_IOR; }
uint gi_addr_matid(uint px)          { uint N = gi_numPx(); return N * GI_PLANE_MATID + px * GI_SZ_MATID; }
uint gi_addr_j(uint px)              { uint N = gi_numPx(); return N * GI_PLANE_J     + px * GI_SZ_J; }
uint gi_addr_w(uint px)              { uint N = gi_numPx(); return N * GI_PLANE_W     + px * GI_SZ_W; }
uint gi_addr_f(uint px)              { uint N = gi_numPx(); return N * GI_PLANE_F     + px * GI_SZ_F; }
uint gi_addr_m(uint px)              { uint N = gi_numPx(); return N * GI_PLANE_M     + px * GI_SZ_M; }
uint gi_addr_wsum(uint px)           { uint N = gi_numPx(); return N * GI_PLANE_WSUM  + px * GI_SZ_WSUM; }
uint gi_addr_vpost(uint px)          { uint N = gi_numPx(); return N * GI_PLANE_VPOST + px * GI_SZ_VPOST; }
uint gi_addr_tpost(uint px)          { uint N = gi_numPx(); return N * GI_PLANE_TPOST + px * GI_SZ_TPOST; }

// Store/load packed V_post (world-space)
void store_Vpost_gi(RWByteAddressBuffer b, uint pixelIdx, float3 Vpost_world)
{
    b.Store(gi_addr_vpost(pixelIdx), PackNormal(normalize(Vpost_world)));
}

float3 load_Vpost_gi(RWByteAddressBuffer b, uint pixelIdx)
{
    return UnpackNormal(b.Load(gi_addr_vpost(pixelIdx)));
}

void store_Tpost_gi(RWByteAddressBuffer b, uint pixelIdx, float3 Tpost)
{
    b.Store(gi_addr_tpost(pixelIdx), PackRGB9E5(max(Tpost, 0.0f)));
}

float3 load_Tpost_gi(RWByteAddressBuffer b, uint pixelIdx)
{
    return UnpackRGB9E5(b.Load(gi_addr_tpost(pixelIdx)));
}


void storeReservoirGI(RWByteAddressBuffer buf, uint pixelIdx, const Reservoir_GI r)
{
    float3 xO  = WorldToObjectPos (r.objID_gi, r.x2_gi);
    float3 nSO = WorldToObjectNrm(r.objID_gi, r.n2_s_gi);
    float3 nGO = WorldToObjectNrm(r.objID_gi, r.n2_g_gi);

    buf.Store4(gi_addr_pack1(pixelIdx), uint4(asuint(xO), PackNormal(normalize(nSO))));
    buf.Store (gi_addr_l2(pixelIdx),    PackRGB9E5(r.L2_gi));
    buf.Store (gi_addr_v2(pixelIdx),    PackNormal(normalize(r.V2_gi)));
    buf.Store (gi_addr_n2g(pixelIdx),   PackNormal(normalize(nGO)));
    buf.Store (gi_addr_objid(pixelIdx), r.objID_gi);
    buf.Store (gi_addr_uv(pixelIdx),    PackFloat2x16(r.uv_gi.x, r.uv_gi.y));
    buf.Store (gi_addr_ior(pixelIdx),   PackFloat2x16(r.etai_gi, r.etat_gi));
    buf.Store (gi_addr_matid(pixelIdx), r.matID_gi);
    buf.Store2(gi_addr_j(pixelIdx),     uint2(asuint(r.J_gi.x), asuint(r.J_gi.y)));
    buf.Store (gi_addr_w(pixelIdx),     asuint(r.W_gi));
    buf.Store (gi_addr_f(pixelIdx),     r.F_gi);
    buf.Store (gi_addr_m(pixelIdx),     r.M_gi);
    buf.Store (gi_addr_wsum(pixelIdx),  asuint(r.w_sum_gi));
}

Reservoir_GI loadReservoirGI(RWByteAddressBuffer buf, uint pixelIdx)
{
    Reservoir_GI r;

    uint4 p1 = buf.Load4(gi_addr_pack1(pixelIdx));

    r.objID_gi = buf.Load(gi_addr_objid(pixelIdx));
    r.matID_gi = buf.Load(gi_addr_matid(pixelIdx));

    r.x2_gi    = ObjectToWorldPos (r.objID_gi, asfloat(p1.xyz));
    r.n2_s_gi  = ObjectToWorldNrm(r.objID_gi, UnpackNormal(p1.w));
    r.n2_g_gi  = ObjectToWorldNrm(r.objID_gi, UnpackNormal(buf.Load(gi_addr_n2g(pixelIdx))));

    r.L2_gi    = UnpackRGB9E5(buf.Load(gi_addr_l2(pixelIdx)));
    r.V2_gi    = UnpackNormal(buf.Load(gi_addr_v2(pixelIdx)));

    uint ior_packed = buf.Load(gi_addr_ior(pixelIdx));
    uint uv_packed  = buf.Load(gi_addr_uv(pixelIdx));
    UnpackFloat2x16(uv_packed,  r.uv_gi.x,   r.uv_gi.y);
    UnpackFloat2x16(ior_packed, r.etai_gi,    r.etat_gi);

    uint2 j_raw = buf.Load2(gi_addr_j(pixelIdx));
    r.J_gi     = asfloat(j_raw);
    r.W_gi     = asfloat(buf.Load(gi_addr_w(pixelIdx)));
    r.F_gi     = buf.Load(gi_addr_f(pixelIdx));

    r.M_gi     = buf.Load(gi_addr_m(pixelIdx));
    r.w_sum_gi = asfloat(buf.Load(gi_addr_wsum(pixelIdx)));

    return r;
}


// -------------------------------
// Fast loaders (DI-style usage)
// -------------------------------

uint  load_objID_gi(RWByteAddressBuffer b, uint pixelIdx)
{
    return b.Load(gi_addr_objid(pixelIdx));
}

uint  load_matID_gi(RWByteAddressBuffer b, uint pixelIdx)
{
    return b.Load(gi_addr_matid(pixelIdx));
}

float3 load_x2_gi(RWByteAddressBuffer b, uint pixelIdx, uint objID)
{
    float3 xO = asfloat(b.Load4(gi_addr_pack1(pixelIdx)).xyz);
    return ObjectToWorldPos(objID, xO);
}

float3 load_n2_s_gi(RWByteAddressBuffer b, uint pixelIdx, uint objID)
{
    uint enc = b.Load4(gi_addr_pack1(pixelIdx)).w;
    return ObjectToWorldNrm(objID, UnpackNormal(enc));
}

float3 load_n2_g_gi(RWByteAddressBuffer b, uint pixelIdx, uint objID)
{
    uint enc = b.Load(gi_addr_n2g(pixelIdx));
    return ObjectToWorldNrm(objID, UnpackNormal(enc));
}

float3 load_L2_gi(RWByteAddressBuffer b, uint pixelIdx)
{
    return UnpackRGB9E5(b.Load(gi_addr_l2(pixelIdx)));
}

float3 load_V2_gi(RWByteAddressBuffer b, uint pixelIdx)
{
    return UnpackNormal(b.Load(gi_addr_v2(pixelIdx)));
}

float2 load_uv_gi(RWByteAddressBuffer b, uint pixelIdx)
{
    float2 r;
    UnpackFloat2x16(b.Load(gi_addr_uv(pixelIdx)), r.x, r.y);
    return r;
}

float  load_etai_gi(RWByteAddressBuffer b, uint pixelIdx)
{
    uint p = b.Load(gi_addr_ior(pixelIdx));
    return f16tof32_custom(p & 0xFFFFu);
}

float  load_etat_gi(RWByteAddressBuffer b, uint pixelIdx)
{
    uint p = b.Load(gi_addr_ior(pixelIdx));
    return f16tof32_custom(p >> 16);
}

float2 load_J_gi(RWByteAddressBuffer b, uint pixelIdx)
{
    return asfloat(b.Load2(gi_addr_j(pixelIdx)));
}

void store_Jy_gi(RWByteAddressBuffer b, uint pixelIdx, float Jy)
{
    b.Store(gi_addr_j(pixelIdx) + 4u, asuint(Jy));
}

float  load_W_gi(RWByteAddressBuffer b, uint pixelIdx)
{
    return asfloat(b.Load(gi_addr_w(pixelIdx)));
}

uint   load_F_gi(RWByteAddressBuffer b, uint pixelIdx)
{
    return b.Load(gi_addr_f(pixelIdx));
}


// -----------------------------------------
// Special entry accessors (fast update only)
// -----------------------------------------

float load_wsum_gi(RWByteAddressBuffer b, uint pixelIdx)
{
    return asfloat(b.Load(gi_addr_wsum(pixelIdx)));
}

void store_wsum_gi(RWByteAddressBuffer b, uint pixelIdx, float wsum)
{
    b.Store(gi_addr_wsum(pixelIdx), asuint(wsum));
}

uint load_M_gi(RWByteAddressBuffer b, uint pixelIdx)
{
    return b.Load(gi_addr_m(pixelIdx));
}

void store_M_gi(RWByteAddressBuffer b, uint pixelIdx, uint M)
{
    b.Store(gi_addr_m(pixelIdx), M);
}

void store_W_gi(RWByteAddressBuffer b, uint pixelIdx, float W)
{
    b.Store(gi_addr_w(pixelIdx), asuint(W));
}

void store_F_gi(RWByteAddressBuffer b, uint pixelIdx, uint F)
{
    b.Store(gi_addr_f(pixelIdx), F);
}


inline bool RejectNormal_GI(float3 n1, float3 n2, float threshold){
    float similarity = dot(n1, n2);
    return (similarity < threshold);
}

inline bool RejectDistance_GI(float3 x1, float3 x2, float3 normal, float threshold)
{
    float dist = abs(dot(x2 - x1, normal));
    return dist > threshold;
}

inline bool IsValidReservoir_GI(Reservoir_GI r){
    bool valid =
        any(abs(r.n2_s_gi) > 0.0f) &&
        r.M_gi > 0
        ;
    return valid;
}

inline bool IsValidReservoir_GI_opt(in float3 n2, in uint M){
    bool valid =
        any(abs(n2) > 0.0f) &&
        M > 0.0f;
    return valid;
}

inline void InvalidateReservoirGI_ShadingNormal(
    RWByteAddressBuffer buf,
    uint pixelIdx
)
{
    // n2_s_gi is stored in PACK1.w
    buf.Store(gi_addr_pack1(pixelIdx) + 12u, 0u);
}



// Calculate reconnection (two–sided)
inline float3 ReconnectGI(
    // Vertex x1 (camera path hit)
    in float3  x1,
    in float3  n1_s,
    in float3  n1_g,
    in float3  o,
    in uint    mID1,
    in float3  localKd1,
    in float   localPr1,
    in float   localPm1,
    in float   etai1,
    in float   etat1,

    // Vertex x2 (GI reservoir / reconnection vertex)
    in uint    mID2,
    in float3  x2,
    in float3  n2_s,
    in float3  n2_g,
    in float3  L2,
    in float3  V2,
    in float3  localKd2,
    in float   localPr2,
    in float   localPm2,
    in float   etai2,
    in float   etat2,

    // Jacobian / PDF plumbing
    in float   pdfx2,
    in float   Jc,      // canonical jacobian term
    in bool    applyJ,
    out float  Jn,
    out float  J
)
{
    if (length(L2) < EPSILON)
        return 0.0f;

    // Geometric prep
    float3 dir   = x2 - x1;
    float  dist  = length(dir);
    float3 ndirN = normalize(-dir); // direction from x2 to x1

    float3 F1 = BSDF_term(mID1, n1_s, n1_g, -ndirN, o,  localKd1, localPr1, localPm1, etai1, etat1);
    float3 F2 = BSDF_term(mID2, n2_s, n2_g, -V2, ndirN, localKd2, localPr2, localPm2, etai2, etat2);

    // Geometry term
    float  G1  = G_term(n1_s, -ndirN);
    float  G2  = G_term(n2_s, -V2);

    // Missing pdfs:
    // PDF at x1 always recomputed
    float PDF1 = PDF_term(mID1, n1_s, n1_g, -ndirN, o,  localKd1, localPr1, localPm1, etai1, etat1);

    // PDF at x2: reuse if provided (NEE ray etc), else recompute
    float PDF2 = pdfx2;
    if (pdfx2 == 0.0f)
        PDF2 = PDF_term(mID2, n2_s, n2_g, -V2, ndirN, localKd2, localPr2, localPm2, etai2, etat2);

    if (PDF1 <= EPSILON || PDF2 <= EPSILON)
        return 0.0f;

    // If the direction from x1 to x2 (-ndirN) points against the geometric normal, we are transmitting INTO the object.
    float3 transmittance = float3(1.0f, 1.0f, 1.0f);
    if (dot(n1_g, -ndirN) < 0.0f)
    {
        transmittance = CalculateAbsorptionThroughput(materials[mID1].Tf, dist);
    }

    // Always compute PSS jacobian for tracking through reservoir merge
    float Gj = abs(dot(ndirN, n2_s)) / (dist * dist);
    Jn = max(PDF1 * PDF2 * Gj, EPSILON);
    J  = 1.0f;

    // Throughput
    float3 r = (F1 / PDF1) * (F2 / PDF2) * L2 * G1 * G2 * transmittance;

    if (applyJ)
    {
        if(Jc < EPSILON) return 0.0f;
        J = Jn / Jc;
    }

    if (any(isnan(r)) || any(isinf(r)) || all(r < EPSILON))
        r = (float3)0.0f;

    return max(r,0.0f);
}



// Update GI reservoir
bool UpdateReservoirGI(
    inout Reservoir_GI reservoir,
    in float wi,
    in uint  M,

    in float3 x2,
    in float3 n2_s,
    in float3 n2_g,
    in float3 L2,
    in float3 V2,

    in float2 uv,
    in float  etai,
    in float  etat,

    in uint matID,
    in uint objID,

    in float2 J,
    in uint   F,

    inout uint2 seed
)
{
    reservoir.w_sum_gi += wi;
    reservoir.M_gi     += M;

    if (RandomFloatSingle(seed.x) < (wi / reservoir.w_sum_gi))
    {
        reservoir.x2_gi   = x2;
        reservoir.n2_s_gi = n2_s;
        reservoir.n2_g_gi = n2_g;
        reservoir.objID_gi = objID;
        reservoir.matID_gi = matID;

        reservoir.uv_gi      = uv;
        reservoir.etai_gi    = etai;
        reservoir.etat_gi    = etat;

        reservoir.L2_gi   = L2;
        reservoir.V2_gi   = V2;
        reservoir.J_gi    = J;
        reservoir.F_gi    = F;
        return true;
    }
    return false;
}


// Fast Update
bool UpdateReservoirGI_Fast(
    RWByteAddressBuffer buf,
    uint pixelIdx,
    float wi,

    // Varying (always written on accept)
    float3 L2_new,
    float2 J_new,
    float3 V2_new,

    inout uint2 seed
)
{
    // w_sum accumulates
    float currentWSum = asfloat(buf.Load(gi_addr_wsum(pixelIdx)));
    float newWSum     = currentWSum + wi;
    buf.Store(gi_addr_wsum(pixelIdx), asuint(newWSum));

    bool isAccepted = false;

    if (RandomFloatSingle(seed.x) < (wi / newWSum))
    {
        isAccepted = true;

        buf.Store(gi_addr_l2(pixelIdx), PackRGB9E5(L2_new));
        buf.Store(gi_addr_v2(pixelIdx), PackNormal(normalize(V2_new)));
        buf.Store2(gi_addr_j(pixelIdx), uint2(asuint(J_new.x), asuint(J_new.y)));
    }

    return isAccepted;
}


// ---------------------------------
// Constant-data setters
// ---------------------------------

// Store constant hit data for GI reconnection
void SetReservoirGI_ConstHit(
    RWByteAddressBuffer buf,
    uint pixelIdx,
    float3 x2_world,
    float3 n2s_world,
    float3 n2g_world,
    uint   matID,
    uint   objID
)
{
    float3 xO  = WorldToObjectPos (objID, x2_world);
    float3 nSO = WorldToObjectNrm(objID, n2s_world);
    float3 nGO = WorldToObjectNrm(objID, n2g_world);

    buf.Store4(gi_addr_pack1(pixelIdx), uint4(asuint(xO), PackNormal(normalize(nSO))));
    buf.Store (gi_addr_n2g(pixelIdx),   PackNormal(normalize(nGO)));
    buf.Store (gi_addr_objid(pixelIdx), objID);
    buf.Store (gi_addr_matid(pixelIdx), matID);
}

void SetReservoirGI_UVAndIOR(
    RWByteAddressBuffer buf,
    uint pixelIdx,
    float2 uv,
    float  etai,
    float  etat
)
{
    buf.Store(gi_addr_uv(pixelIdx),  PackFloat2x16(uv.x, uv.y));
    buf.Store(gi_addr_ior(pixelIdx), PackFloat2x16(etai, etat));
}
