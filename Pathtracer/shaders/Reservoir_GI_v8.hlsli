// RIS reservoir for global illumination
struct Reservoir_GI
{
    // Constant-after-hit payload
    float3 x2_gi;
    float3 n2_s_gi;
    float3 n2_g_gi;
    uint   objID_gi;
    uint   matID_gi;
    float3 localKd_gi;
    float  localPr_gi;
    float  localPm_gi;
    float  etai_gi;
    float  etat_gi;

    // Varying payload
    float3 L2_gi;
    float3 V2_gi;
    float4 J_gi;
    float  F_gi;

    // Reservoir bookkeeping
    float  W_gi;
    float  w_sum_gi;
    uint   M_gi;
};


// Data management
static const uint BYTES_GI        = 80u;
static const uint BYTES_GI_VPOST  = 4u;
static const uint BYTES_GI_TPOST  = 4u;

static const uint STRIDE_GI       = BYTES_GI + BYTES_GI_VPOST + BYTES_GI_TPOST; // 88

uint pixelBaseAddrGI(uint pixelIdx) { return pixelIdx * STRIDE_GI; }

static const uint O_GI_PACK1  =  0u;
static const uint O_GI_PACK2  = 16u;
static const uint O_GI_PACK3  = 32u;
static const uint O_GI_PACK4  = 48u;
static const uint O_GI_PACK5  = 64u;

static const uint O_GI_W      = O_GI_PACK5 +  0u;
static const uint O_GI_F      = O_GI_PACK5 +  4u;
static const uint O_GI_M      = O_GI_PACK5 +  8u;
static const uint O_GI_WSUM   = O_GI_PACK5 + 12u;

static const uint O_GI_VPOST_BASE = BYTES_GI;                  // 80
static const uint O_GI_TPOST_BASE = BYTES_GI + BYTES_GI_VPOST; // 84

uint addr_Vpost(uint pixelIdx) { return pixelBaseAddrGI(pixelIdx) + O_GI_VPOST_BASE; }
uint addr_Tpost(uint pixelIdx) { return pixelBaseAddrGI(pixelIdx) + O_GI_TPOST_BASE; }

// Store/load packed V_post (world-space)
void store_Vpost_gi(RWByteAddressBuffer b, uint pixelIdx, float3 Vpost_world)
{
    b.Store(addr_Vpost(pixelIdx), PackNormal(normalize(Vpost_world)));
}

float3 load_Vpost_gi(RWByteAddressBuffer b, uint pixelIdx)
{
    return UnpackNormal(b.Load(addr_Vpost(pixelIdx)));
}

void store_Tpost_gi(RWByteAddressBuffer b, uint pixelIdx, float3 Tpost)
{
    // Throughput is non-negative; RGB9E5 is fine.
    b.Store(addr_Tpost(pixelIdx), PackRGB9E5(max(Tpost, 0.0f)));
}

float3 load_Tpost_gi(RWByteAddressBuffer b, uint pixelIdx)
{
    return UnpackRGB9E5(b.Load(addr_Tpost(pixelIdx)));
}





void storeReservoirGI(RWByteAddressBuffer buf, uint pixelIdx, const Reservoir_GI r)
{
    const uint base = pixelBaseAddrGI(pixelIdx);

    float3 xO  = WorldToObjectPos (r.objID_gi, r.x2_gi);
    float3 nSO = WorldToObjectNrm(r.objID_gi, r.n2_s_gi);
    float3 nGO = WorldToObjectNrm(r.objID_gi, r.n2_g_gi);

    buf.Store4(base + O_GI_PACK1,
               uint4(asuint(xO), PackNormal(normalize(nSO))));

    buf.Store4(base + O_GI_PACK2,
               uint4(PackRGB9E5(r.L2_gi),
                     PackNormal(normalize(r.V2_gi)),
                     PackNormal(normalize(nGO)),
                     r.objID_gi));

    buf.Store4(base + O_GI_PACK3,
               uint4(PackRGB9E5(r.localKd_gi),
                     PackFloat2x16(r.localPr_gi, r.localPm_gi),
                     PackFloat2x16(r.etai_gi,    r.etat_gi),
                     r.matID_gi));

    buf.Store4(base + O_GI_PACK4, asuint(r.J_gi));

    buf.Store4(base + O_GI_PACK5,
               uint4(asuint(r.W_gi),
                     asuint(r.F_gi),
                     r.M_gi,
                     asuint(r.w_sum_gi)));
}

Reservoir_GI loadReservoirGI(RWByteAddressBuffer buf, uint pixelIdx)
{
    const uint base = pixelBaseAddrGI(pixelIdx);

    uint4 p1 = buf.Load4(base + O_GI_PACK1);
    uint4 p2 = buf.Load4(base + O_GI_PACK2);
    uint4 p3 = buf.Load4(base + O_GI_PACK3);
    uint4 p4 = buf.Load4(base + O_GI_PACK4);
    uint4 p5 = buf.Load4(base + O_GI_PACK5);

    Reservoir_GI r;

    r.objID_gi = p2.w;
    r.matID_gi = p3.w;

    r.x2_gi    = ObjectToWorldPos (r.objID_gi, asfloat(p1.xyz));
    r.n2_s_gi  = ObjectToWorldNrm(r.objID_gi, UnpackNormal(p1.w));
    r.n2_g_gi  = ObjectToWorldNrm(r.objID_gi, UnpackNormal(p2.z));

    r.L2_gi    = UnpackRGB9E5(p2.x);
    r.V2_gi    = UnpackNormal(p2.y);

    r.localKd_gi = UnpackRGB9E5(p3.x);
    UnpackFloat2x16(p3.y, r.localPr_gi, r.localPm_gi);
    UnpackFloat2x16(p3.z, r.etai_gi,    r.etat_gi);

    r.J_gi     = asfloat(p4);

    r.W_gi     = asfloat(p5.x);
    r.F_gi     = asfloat(p5.y);
    r.M_gi     = p5.z;
    r.w_sum_gi = asfloat(p5.w);

    return r;
}


// -------------------------------
// Fast loaders (DI-style usage)
// -------------------------------

// Typically: obj = load_objID_gi(buf,id); then call load_x2/n2... with obj.
uint  load_objID_gi(RWByteAddressBuffer b, uint pixelIdx)
{
    return b.Load4(pixelBaseAddrGI(pixelIdx) + O_GI_PACK2).w;
}

uint  load_matID_gi(RWByteAddressBuffer b, uint pixelIdx)
{
    return b.Load4(pixelBaseAddrGI(pixelIdx) + O_GI_PACK3).w;
}

float3 load_x2_gi(RWByteAddressBuffer b, uint pixelIdx, uint objID)
{
    float3 xO = asfloat(b.Load4(pixelBaseAddrGI(pixelIdx) + O_GI_PACK1).xyz);
    return ObjectToWorldPos(objID, xO);
}

float3 load_n2_s_gi(RWByteAddressBuffer b, uint pixelIdx, uint objID)
{
    uint enc = b.Load4(pixelBaseAddrGI(pixelIdx) + O_GI_PACK1).w;
    return ObjectToWorldNrm(objID, UnpackNormal(enc));
}

float3 load_n2_g_gi(RWByteAddressBuffer b, uint pixelIdx, uint objID)
{
    uint enc = b.Load4(pixelBaseAddrGI(pixelIdx) + O_GI_PACK2).z;
    return ObjectToWorldNrm(objID, UnpackNormal(enc));
}

float3 load_L2_gi(RWByteAddressBuffer b, uint pixelIdx)
{
    uint enc = b.Load4(pixelBaseAddrGI(pixelIdx) + O_GI_PACK2).x;
    return UnpackRGB9E5(enc);
}

float3 load_V2_gi(RWByteAddressBuffer b, uint pixelIdx)
{
    uint enc = b.Load4(pixelBaseAddrGI(pixelIdx) + O_GI_PACK2).y;
    return UnpackNormal(enc);
}

float3 load_localKd_gi(RWByteAddressBuffer b, uint pixelIdx)
{
    uint enc = b.Load4(pixelBaseAddrGI(pixelIdx) + O_GI_PACK3).x;
    return UnpackRGB9E5(enc);
}

float  load_localPr_gi(RWByteAddressBuffer b, uint pixelIdx)
{
    uint p = b.Load4(pixelBaseAddrGI(pixelIdx) + O_GI_PACK3).y;
    return f16tof32(p & 0xFFFFu);
}

float  load_localPm_gi(RWByteAddressBuffer b, uint pixelIdx)
{
    uint p = b.Load4(pixelBaseAddrGI(pixelIdx) + O_GI_PACK3).y;
    return f16tof32(p >> 16);
}

float  load_etai_gi(RWByteAddressBuffer b, uint pixelIdx)
{
    uint p = b.Load4(pixelBaseAddrGI(pixelIdx) + O_GI_PACK3).z;
    return f16tof32(p & 0xFFFFu);
}

float  load_etat_gi(RWByteAddressBuffer b, uint pixelIdx)
{
    uint p = b.Load4(pixelBaseAddrGI(pixelIdx) + O_GI_PACK3).z;
    return f16tof32(p >> 16);
}

float4 load_J_gi(RWByteAddressBuffer b, uint pixelIdx)
{
    return asfloat(b.Load4(pixelBaseAddrGI(pixelIdx) + O_GI_PACK4));
}

float  load_W_gi(RWByteAddressBuffer b, uint pixelIdx)
{
    return asfloat(b.Load(pixelBaseAddrGI(pixelIdx) + O_GI_W));
}

float  load_F_gi(RWByteAddressBuffer b, uint pixelIdx)
{
    return asfloat(b.Load(pixelBaseAddrGI(pixelIdx) + O_GI_F));
}


// -----------------------------------------
// Special entry accessors (fast update only)
// -----------------------------------------

float load_wsum_gi(RWByteAddressBuffer b, uint pixelIdx)
{
    return asfloat(b.Load(pixelBaseAddrGI(pixelIdx) + O_GI_WSUM));
}

void store_wsum_gi(RWByteAddressBuffer b, uint pixelIdx, float wsum)
{
    b.Store(pixelBaseAddrGI(pixelIdx) + O_GI_WSUM, asuint(wsum));
}

uint load_M_gi(RWByteAddressBuffer b, uint pixelIdx)
{
    return b.Load(pixelBaseAddrGI(pixelIdx) + O_GI_M);
}

void store_M_gi(RWByteAddressBuffer b, uint pixelIdx, uint M)
{
    b.Store(pixelBaseAddrGI(pixelIdx) + O_GI_M, M);
}

void store_W_gi(RWByteAddressBuffer b, uint pixelIdx, float W)
{
    b.Store(pixelBaseAddrGI(pixelIdx) + O_GI_W, asuint(W));
}

void store_F_gi(RWByteAddressBuffer b, uint pixelIdx, float F)
{
    b.Store(pixelBaseAddrGI(pixelIdx) + O_GI_F, asuint(F));
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

inline bool RejectLength_GI(float3 x2_c, float3 n2_c,
                            float3 x1_c, float3 x1_n,
                            float  threshold)
{
    float J = JacobianDeterminantDI(x1_c, x2_c, x1_n, n2_c, 0u);
    return J <= threshold || J >= 1.0f/threshold;
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
    buf.Store(
        pixelBaseAddrGI(pixelIdx) + O_GI_PACK1 + 12u,
        0u
    );
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
    float3 ndirN = normalize(-dir); // direction from x1 to x2 (negated)

    // Terms (now require n_s + n_g + locals at both vertices)
    float3 F1 = BSDF_term(mID1, n1_s, n1_g, -ndirN, o,  localKd1, localPr1, localPm1, etai1, etat1);
    float3 F2 = BSDF_term(mID2, n2_s, n2_g, -V2, ndirN, localKd2, localPr2, localPm2, etai2, etat2);

    // Geometry term
    float  G1  = abs(G_term(n1_s, -ndirN));
    float  G2  = abs(G_term(n2_s, -V2));

    // Missing pdfs:
    // PDF at x1 always recomputed
    float PDF1 = PDF_term(mID1, n1_s, n1_g, -ndirN, o,  localKd1, localPr1, localPm1, etai1, etat1);

    // PDF at x2: reuse if provided (NEE ray etc), else recompute
    float PDF2 = pdfx2;
    if (pdfx2 == 0.0f)
        PDF2 = PDF_term(mID2, n2_s, n2_g, -V2, ndirN, localKd2, localPr2, localPm2, etai2, etat2);

    if (PDF1 <= EPSILON || PDF2 <= EPSILON)
        return 0.0f;

    // Reconnection jacobian
    Jn = Jc;
    J  = 1.0f;

    // Throughput
    float3 r = (F1 / PDF1) * (F2 / PDF2) * L2 * G1 * G2;

    if (applyJ)
    {
        // Apply reconnection jacobian (uses n2 geometric, as this is a geometric factor)
        float Gj = abs(dot(ndirN, n2_s)) / (dist * dist);

        Jn = max(PDF1 * PDF2 * Gj, EPSILON);
        if(Jc < EPSILON) return 0.0f;
        J  = Jn / Jc;
        // r *= J; // keep disabled as in your original
    }

    if (any(isnan(r)) || all(r < EPSILON))
        r = (float3)0.0f;

    return max(r,0.0f);
}

inline float PSSJacobian(
    // Vertex x1
    in float3 x1,
    in float3 n1_s,
    in float3 n1_g,
    in float3 o,
    in uint   mID1,
    in float3 localKd1,
    in float  localPr1,
    in float  localPm1,
    in float  etai1,
    in float  etat1,

    // Vertex x2
    in float3 x2,
    in float3 n2_s,
    in float3 n2_g,
    in float3 V2,
    in uint   mID2,
    in float3 localKd2,
    in float  localPr2,
    in float  localPm2,
    in float  etai2,
    in float  etat2,

    in float  pdfx2
)
{
    float3 dir   = x2 - x1;
    float  dist2 = dot(dir, dir);
    float3 ndirN = normalize(-dir); // your convention (x1 -> x2)

    // PDFs in the same measure as your BSDF_term/PDF_term (PSS)
    float PDF1 = PDF_term(mID1, n1_s, n1_g, -ndirN, o,
                          localKd1, localPr1, localPm1, etai1, etat1);
    if (PDF1 <= EPSILON) return 0.0f;

    // Keep logic identical: reuse pdfx2 if provided, else recompute.
    // Note: the "o" parameter at x2 is the direction towards x1, i.e. -ndirN (same as ReconnectGI).
    float PDF2 = (pdfx2 > 0.0f)
        ? pdfx2
        : PDF_term(mID2, n2_s, n2_g, -V2, ndirN,
                   localKd2, localPr2, localPm2, etai2, etat2);
    if (PDF2 <= EPSILON) return 0.0f;

    // Pure geometric factor used in your reconnection Jacobian (use geometric normal at x2)
    float Gj = abs(dot(ndirN, n2_s)) / dist2;

    return max(PDF1 * PDF2 * Gj, EPSILON);
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

    in float3 localKd,
    in float  localPr,
    in float  localPm,
    in float  etai,
    in float  etat,

    in uint matID,
    in uint objID,

    in float4 J,
    in float  F,

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

        reservoir.localKd_gi = localKd;
        reservoir.localPr_gi = localPr;
        reservoir.localPm_gi = localPm;
        reservoir.etai_gi    = etai;
        reservoir.etat_gi    = etat;

        reservoir.L2_gi = L2;
        reservoir.V2_gi = V2;
        reservoir.J_gi  = J;
        reservoir.F_gi  = F;
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
    float4 J_new,
    float3 V2_new,

    inout uint2 seed
)
{
    const uint base = pixelBaseAddrGI(pixelIdx);

    // w_sum accumulates
    float currentWSum = asfloat(buf.Load(base + O_GI_WSUM));
    float newWSum     = currentWSum + wi;
    buf.Store(base + O_GI_WSUM, asuint(newWSum));

    bool isAccepted = false;

    if (RandomFloatSingle(seed.x) < (wi / newWSum))
    {
        isAccepted = true;

        // Pack2: always update L2 + V2 (world packed normal)
        uint4 p2 = buf.Load4(base + O_GI_PACK2);
        p2.x = PackRGB9E5(L2_new);
        p2.y = PackNormal(normalize(V2_new));
        buf.Store4(base + O_GI_PACK2, p2);

        // Pack4: always overwrite J
        buf.Store4(base + O_GI_PACK4, asuint(J_new));
    }

    return isAccepted;
}


// ---------------------------------
// Constant-data setters
// ---------------------------------

// Minimal constant setter as requested: x2, n2s, n2g, matID, objID
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
    const uint base = pixelBaseAddrGI(pixelIdx);

    float3 xO  = WorldToObjectPos (objID, x2_world);
    float3 nSO = WorldToObjectNrm(objID, n2s_world);
    float3 nGO = WorldToObjectNrm(objID, n2g_world);

    // Pack1: x2 + n2_s
    buf.Store4(base + O_GI_PACK1,
               uint4(asuint(xO), PackNormal(normalize(nSO))));

    // Pack2: update only n2_g + objID (preserve L2/V2)
    uint4 p2 = buf.Load4(base + O_GI_PACK2);
    p2.z = PackNormal(normalize(nGO));
    p2.w = objID;
    buf.Store4(base + O_GI_PACK2, p2);

    // Pack3: matID (preserve local material packed fields)
    uint4 p3 = buf.Load4(base + O_GI_PACK3);
    p3.w = matID;
    buf.Store4(base + O_GI_PACK3, p3);
}

void SetReservoirGI_LocalMaterial(
    RWByteAddressBuffer buf,
    uint pixelIdx,
    float3 localKd,
    float  localPr,
    float  localPm,
    float  etai,
    float  etat
)
{
    const uint base = pixelBaseAddrGI(pixelIdx);
    uint4 p3 = buf.Load4(base + O_GI_PACK3);

    p3.x = PackRGB9E5(localKd);
    p3.y = PackFloat2x16(localPr, localPm);
    p3.z = PackFloat2x16(etai,    etat);

    buf.Store4(base + O_GI_PACK3, p3);
}

