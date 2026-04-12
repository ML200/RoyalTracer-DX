// RIS reservoir for global illumination
struct Reservoir_GI
{
    // Constant-after-hit payload
    float3 x2_gi;
    float3 n2_s_gi;
    uint   objID_gi;
    uint   matID_gi;
    float2 uv_gi;
    float  eta_gi;      // transmittance IOR at x2 (stored at path creation)

    // Varying payload
    float3 L2_gi;
    float3 V2_gi;
    uint   F_gi;        // RGB9E5-packed contribution (replaces scalar luminance)

    // Reservoir bookkeeping
    float  W_gi;
    float  w_sum_gi;    // raygen-only (merge passes overwrite before use)
    uint   M_gi;

};


// SoA layout — each field is a contiguous plane across all pixels.
// Merge-active: PACK1(16) | L2(4) | V2(4) | OBJID(4) | UV(4) | MATID(4) |
//               W(4) | F(4) | M(4) | ETA(4) = 52
// Raygen-only:  WSUM(4) | VPOST(4) | TPOST(4) = 12
// Total: 64 bytes/pixel

static const uint BYTES_GI       = 52u;
static const uint BYTES_GI_WSUM  =  4u;
static const uint BYTES_GI_VPOST =  4u;
static const uint BYTES_GI_TPOST =  4u;
static const uint STRIDE_GI      = BYTES_GI + BYTES_GI_WSUM + BYTES_GI_VPOST + BYTES_GI_TPOST; // 64

// Per-field sizes
static const uint GI_SZ_PACK1 = 16u;  // x2(12) + n2_s_packed(4)
static const uint GI_SZ_L2    =  4u;
static const uint GI_SZ_V2    =  4u;
static const uint GI_SZ_OBJID =  4u;
static const uint GI_SZ_UV    =  4u;
static const uint GI_SZ_MATID =  4u;
static const uint GI_SZ_W     =  4u;
static const uint GI_SZ_F     =  4u;
static const uint GI_SZ_M     =  4u;
static const uint GI_SZ_ETA   =  4u;
static const uint GI_SZ_WSUM  =  4u;
static const uint GI_SZ_VPOST =  4u;
static const uint GI_SZ_TPOST =  4u;

// Plane cumulative offsets (per-pixel contribution to base address)
static const uint GI_PLANE_PACK1 =  0u;
static const uint GI_PLANE_L2    = 16u;
static const uint GI_PLANE_V2    = 20u;
static const uint GI_PLANE_OBJID = 24u;
static const uint GI_PLANE_UV    = 28u;
static const uint GI_PLANE_MATID = 32u;
static const uint GI_PLANE_W     = 36u;
static const uint GI_PLANE_F     = 40u;
static const uint GI_PLANE_M     = 44u;
static const uint GI_PLANE_ETA   = 48u;
static const uint GI_PLANE_WSUM  = 52u;
static const uint GI_PLANE_VPOST = 56u;
static const uint GI_PLANE_TPOST = 60u;

// SoA address helpers
// Tile-aligned pixel count — must match MapPixelID's 4x8 tile swizzle.
uint gi_numPx()                       { return ((IMG_W + 3u) / 4u) * ((IMG_H + 7u) / 8u) * 32u; }
uint gi_addr_pack1(uint px)           { return px * GI_SZ_PACK1; }
uint gi_addr_l2(uint px)              { uint N = gi_numPx(); return N * GI_PLANE_L2    + px * GI_SZ_L2; }
uint gi_addr_v2(uint px)              { uint N = gi_numPx(); return N * GI_PLANE_V2    + px * GI_SZ_V2; }
uint gi_addr_objid(uint px)           { uint N = gi_numPx(); return N * GI_PLANE_OBJID + px * GI_SZ_OBJID; }
uint gi_addr_uv(uint px)             { uint N = gi_numPx(); return N * GI_PLANE_UV    + px * GI_SZ_UV; }
uint gi_addr_matid(uint px)          { uint N = gi_numPx(); return N * GI_PLANE_MATID + px * GI_SZ_MATID; }
uint gi_addr_w(uint px)              { uint N = gi_numPx(); return N * GI_PLANE_W     + px * GI_SZ_W; }
uint gi_addr_f(uint px)              { uint N = gi_numPx(); return N * GI_PLANE_F     + px * GI_SZ_F; }
uint gi_addr_m(uint px)              { uint N = gi_numPx(); return N * GI_PLANE_M     + px * GI_SZ_M; }
uint gi_addr_eta(uint px)            { uint N = gi_numPx(); return N * GI_PLANE_ETA   + px * GI_SZ_ETA; }
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

    buf.Store4(gi_addr_pack1(pixelIdx), uint4(asuint(xO), PackNormal(normalize(nSO))));
    buf.Store (gi_addr_l2(pixelIdx),    PackRGB9E5(r.L2_gi));
    buf.Store (gi_addr_v2(pixelIdx),    PackNormal(normalize(r.V2_gi)));
    buf.Store (gi_addr_objid(pixelIdx), r.objID_gi);
    buf.Store (gi_addr_uv(pixelIdx),    PackFloat2x16(r.uv_gi.x, r.uv_gi.y));
    buf.Store (gi_addr_matid(pixelIdx), r.matID_gi);
    buf.Store (gi_addr_w(pixelIdx),     asuint(r.W_gi));
    buf.Store (gi_addr_f(pixelIdx),     r.F_gi);
    buf.Store (gi_addr_m(pixelIdx),     r.M_gi);
    buf.Store (gi_addr_eta(pixelIdx),   asuint(r.eta_gi));
}

Reservoir_GI loadReservoirGI(RWByteAddressBuffer buf, uint pixelIdx)
{
    Reservoir_GI r;

    uint4 p1 = buf.Load4(gi_addr_pack1(pixelIdx));

    r.objID_gi = buf.Load(gi_addr_objid(pixelIdx));
    r.matID_gi = buf.Load(gi_addr_matid(pixelIdx));

    r.x2_gi    = ObjectToWorldPos (r.objID_gi, asfloat(p1.xyz));
    r.n2_s_gi  = ObjectToWorldNrm(r.objID_gi, UnpackNormal(p1.w));

    r.L2_gi    = UnpackRGB9E5(buf.Load(gi_addr_l2(pixelIdx)));
    r.V2_gi    = UnpackNormal(buf.Load(gi_addr_v2(pixelIdx)));

    uint uv_packed  = buf.Load(gi_addr_uv(pixelIdx));
    UnpackFloat2x16(uv_packed, r.uv_gi.x, r.uv_gi.y);

    r.W_gi     = asfloat(buf.Load(gi_addr_w(pixelIdx)));
    r.F_gi     = buf.Load(gi_addr_f(pixelIdx));

    r.M_gi     = buf.Load(gi_addr_m(pixelIdx));
    r.eta_gi   = asfloat(buf.Load(gi_addr_eta(pixelIdx)));
    r.w_sum_gi = 0.0f; // raygen-only; merge passes overwrite before use

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

float  load_W_gi(RWByteAddressBuffer b, uint pixelIdx)
{
    return asfloat(b.Load(gi_addr_w(pixelIdx)));
}

uint   load_F_gi(RWByteAddressBuffer b, uint pixelIdx)
{
    return b.Load(gi_addr_f(pixelIdx));
}

float  load_eta_gi(RWByteAddressBuffer b, uint pixelIdx)
{
    return asfloat(b.Load(gi_addr_eta(pixelIdx)));
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



// Geometric jacobian: recomputable from positions + shading normal
inline float ComputeJc(float3 x1, float3 x2, float3 n2_s)
{
    float3 d = x1 - x2;
    float  dist2 = dot(d, d);
    if (dist2 < EPSILON) return EPSILON;
    float  dist = sqrt(dist2);
    return max(abs(dot(d / dist, n2_s)) / dist2, EPSILON);
}

// Safe jacobian ratio: matches old behavior of zeroing when Jc is degenerate
inline float JacobianRatio(float Jn, float Jc)
{
    return (Jc > EPSILON) ? (Jn / Jc) : 0.0f;
}

// Calculate reconnection (two–sided)
inline float3 ReconnectGI(
    // Vertex x1 (camera path hit)
    in float3  x1,
    in float3  n1_s,
    in float3  o,
    in uint    mID1,
    in float3  localKd1,
    in float   localPr1,
    in float   localPm1,

    // Vertex x2 (GI reservoir / reconnection vertex)
    in uint    mID2,
    in float3  x2,
    in float3  n2_s,
    in float3  L2,
    in float3  V2,
    in float3  localKd2,
    in float   localPr2,
    in float   localPm2,
    in float   eta2,        // stored transmittance IOR at x2

    out float  Jn
)
{
    if (length(L2) < EPSILON)
        return 0.0f;

    // Geometric prep
    float3 dir   = x2 - x1;
    float  dist  = length(dir);
    float3 ndirN = normalize(-dir); // direction from x2 to x1

    // x1 IOR: always air / material (primary hit)
    float etai1 = 1.0f;
    float etat1 = materials[mID1].Ni;

    // x2 IOR
    float etai2 = 1.0f;
    float etat2 = eta2;

    if(dot(-ndirN, n1_s)<0.0f)
        etai2 = etat1;

    float3 F1 = BSDF_term(mID1, n1_s, n1_s, -ndirN, o,  localKd1, localPr1, localPm1, etai1, etat1);
    float3 F2 = BSDF_term(mID2, n2_s, n2_s, -V2, ndirN, localKd2, localPr2, localPm2, etai2, etat2);

    // Geometry term
    float  G1  = G_term(n1_s, -ndirN);
    float  G2  = G_term(n2_s, -V2);

    // Transmittance: if direction points against the shading normal, we are transmitting INTO the object
    float3 transmittance = float3(1.0f, 1.0f, 1.0f);
    if (dot(n1_s, -ndirN) < 0.0f)
    {
        transmittance = CalculateAbsorptionThroughput(materials[mID1].Tf, dist);
    }

    // Geometric jacobian at the new x1
    Jn = max(abs(dot(ndirN, n2_s)) / (dist * dist), EPSILON);

    // Throughput: no PDF division — wi divides by pdf explicitly in raygen
    float3 r = F1 * F2 * L2 * G1 * G2 * transmittance;

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
    in float3 L2,
    in float3 V2,

    in float2 uv,

    in uint matID,
    in uint objID,
    in float eta,

    in uint  F,

    inout uint2 seed
)
{
    reservoir.w_sum_gi += wi;
    reservoir.M_gi     += M;

    if (RandomFloatSingle(seed.x) < (wi / reservoir.w_sum_gi))
    {
        reservoir.x2_gi   = x2;
        reservoir.n2_s_gi = n2_s;
        reservoir.objID_gi = objID;
        reservoir.matID_gi = matID;
        reservoir.eta_gi   = eta;

        reservoir.uv_gi   = uv;

        reservoir.L2_gi   = L2;
        reservoir.V2_gi   = V2;
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
    uint   matID,
    uint   objID,
    float  eta
)
{
    float3 xO  = WorldToObjectPos (objID, x2_world);
    float3 nSO = WorldToObjectNrm(objID, n2s_world);

    buf.Store4(gi_addr_pack1(pixelIdx), uint4(asuint(xO), PackNormal(normalize(nSO))));
    buf.Store (gi_addr_objid(pixelIdx), objID);
    buf.Store (gi_addr_matid(pixelIdx), matID);
    buf.Store (gi_addr_eta(pixelIdx),   asuint(eta));
}

void SetReservoirGI_UV(
    RWByteAddressBuffer buf,
    uint pixelIdx,
    float2 uv
)
{
    buf.Store(gi_addr_uv(pixelIdx), PackFloat2x16(uv.x, uv.y));
}
