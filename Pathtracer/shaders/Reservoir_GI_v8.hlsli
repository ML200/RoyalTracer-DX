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
    float  F_gi;

    // Reservoir bookkeeping
    float  W_gi;
    float  w_sum_gi;
    uint   M_gi;

    // Hybrid shift: RNG seed for path replay
    uint   seed_gi;
};


// Data management
// Pack1(16) + Pack2(16) + Pack3(12) + Pack4(16) + M+wsum+seed(12) = 72
static const uint BYTES_GI        = 72u;
static const uint BYTES_GI_VPOST  = 4u;
static const uint BYTES_GI_TPOST  = 4u;

static const uint STRIDE_GI       = BYTES_GI + BYTES_GI_VPOST + BYTES_GI_TPOST; // 80

uint pixelBaseAddrGI(uint pixelIdx) { return pixelIdx * STRIDE_GI; }

static const uint O_GI_PACK1  =  0u;   // x2(12) + n2_s(4)
static const uint O_GI_PACK2  = 16u;   // L2(4) + V2(4) + n2_g(4) + objID(4)
static const uint O_GI_PACK3  = 32u;   // uv(4) + etai_etat(4) + matID(4) = 12 bytes
static const uint O_GI_PACK4  = 44u;   // J.x(4) + J.y(4) + W(4) + F(4)

static const uint O_GI_W      = O_GI_PACK4 +  8u;
static const uint O_GI_F      = O_GI_PACK4 + 12u;
static const uint O_GI_M      = 60u;
static const uint O_GI_WSUM   = 64u;
static const uint O_GI_SEED   = 68u;

static const uint O_GI_VPOST_BASE = BYTES_GI;                  // 72
static const uint O_GI_TPOST_BASE = BYTES_GI + BYTES_GI_VPOST; // 76

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

    buf.Store3(base + O_GI_PACK3,
               uint3(PackFloat2x16(r.uv_gi.x, r.uv_gi.y),
                     PackFloat2x16(r.etai_gi,  r.etat_gi),
                     r.matID_gi));

    buf.Store4(base + O_GI_PACK4,
               uint4(asuint(r.J_gi.x),
                     asuint(r.J_gi.y),
                     asuint(r.W_gi),
                     asuint(r.F_gi)));

    buf.Store3(base + O_GI_M,
               uint3(r.M_gi,
                     asuint(r.w_sum_gi),
                     r.seed_gi));
}

Reservoir_GI loadReservoirGI(RWByteAddressBuffer buf, uint pixelIdx)
{
    const uint base = pixelBaseAddrGI(pixelIdx);

    uint4 p1 = buf.Load4(base + O_GI_PACK1);
    uint4 p2 = buf.Load4(base + O_GI_PACK2);
    uint3 p3 = buf.Load3(base + O_GI_PACK3);
    uint4 p4 = buf.Load4(base + O_GI_PACK4);
    uint3 p5 = buf.Load3(base + O_GI_M);

    Reservoir_GI r;

    r.objID_gi = p2.w;
    r.matID_gi = p3.z;

    r.x2_gi    = ObjectToWorldPos (r.objID_gi, asfloat(p1.xyz));
    r.n2_s_gi  = ObjectToWorldNrm(r.objID_gi, UnpackNormal(p1.w));
    r.n2_g_gi  = ObjectToWorldNrm(r.objID_gi, UnpackNormal(p2.z));

    r.L2_gi    = UnpackRGB9E5(p2.x);
    r.V2_gi    = UnpackNormal(p2.y);

    UnpackFloat2x16(p3.x, r.uv_gi.x, r.uv_gi.y);
    UnpackFloat2x16(p3.y, r.etai_gi,  r.etat_gi);

    r.J_gi     = asfloat(p4.xy);
    r.W_gi     = asfloat(p4.z);
    r.F_gi     = asfloat(p4.w);

    r.M_gi     = p5.x;
    r.w_sum_gi = asfloat(p5.y);
    r.seed_gi  = p5.z;

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
    return b.Load3(pixelBaseAddrGI(pixelIdx) + O_GI_PACK3).z;
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

float2 load_uv_gi(RWByteAddressBuffer b, uint pixelIdx)
{
    float2 r;
    UnpackFloat2x16(b.Load(pixelBaseAddrGI(pixelIdx) + O_GI_PACK3), r.x, r.y);
    return r;
}

float  load_etai_gi(RWByteAddressBuffer b, uint pixelIdx)
{
    uint p = b.Load3(pixelBaseAddrGI(pixelIdx) + O_GI_PACK3).y;
    return f16tof32_custom(p & 0xFFFFu);
}

float  load_etat_gi(RWByteAddressBuffer b, uint pixelIdx)
{
    uint p = b.Load3(pixelBaseAddrGI(pixelIdx) + O_GI_PACK3).y;
    return f16tof32_custom(p >> 16);
}

float2 load_J_gi(RWByteAddressBuffer b, uint pixelIdx)
{
    return asfloat(b.Load2(pixelBaseAddrGI(pixelIdx) + O_GI_PACK4));
}

void store_Jy_gi(RWByteAddressBuffer b, uint pixelIdx, float Jy)
{
    b.Store(pixelBaseAddrGI(pixelIdx) + O_GI_PACK4 + 4u, asuint(Jy));
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

uint load_seed_gi(RWByteAddressBuffer b, uint pixelIdx)
{
    return b.Load(pixelBaseAddrGI(pixelIdx) + O_GI_SEED);
}

void store_seed_gi(RWByteAddressBuffer b, uint pixelIdx, uint s)
{
    b.Store(pixelBaseAddrGI(pixelIdx) + O_GI_SEED, s);
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
    in float  F,
    in uint   pathSeed,

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
        reservoir.seed_gi = pathSeed;
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

        // J_gi: overwrite .x and .y (first 8 bytes of PACK4)
        buf.Store2(base + O_GI_PACK4, uint2(asuint(J_new.x), asuint(J_new.y)));
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

    // Pack3: matID (preserve uv + etai_etat, update matID only)
    // Pack3 is 12 bytes (3 uints): [uv_packed, etai_etat_packed, matID]
    uint3 p3 = buf.Load3(base + O_GI_PACK3);
    p3.z = matID;
    buf.Store3(base + O_GI_PACK3, p3);
}

void SetReservoirGI_UVAndIOR(
    RWByteAddressBuffer buf,
    uint pixelIdx,
    float2 uv,
    float  etai,
    float  etat
)
{
    const uint base = pixelBaseAddrGI(pixelIdx);
    // Pack3 is now 12 bytes: uv(4) + etai_etat(4) + matID(4)
    // Only update uv + etai_etat (first 8 bytes), preserve matID
    uint matID = buf.Load(base + O_GI_PACK3 + 8u);
    buf.Store3(base + O_GI_PACK3,
               uint3(PackFloat2x16(uv.x, uv.y),
                     PackFloat2x16(etai, etat),
                     matID));
}

