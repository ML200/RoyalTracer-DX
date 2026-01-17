// RIS reservoir for direct lighting
struct Reservoir_GI
{
    float3 x2_gi;
    float3 n2_s_gi; // Shading Normal
    float3 n2_g_gi; // Geometric Normal (Added)

    float  W_gi;
    float  w_sum_gi;
    float3 L2_gi;
    float3 V2_gi;

    float F_gi;
    float4 J_gi;
    uint   lobe0_gi;
    uint   lobe1_gi;

    uint   M_gi;
    uint   objID_gi;
    uint   matID_gi;
    uint   rSeed_gi;
    uint   rIndex_gi;

    float3 localKd_gi;
    float  localPr_gi;
    float  localPm_gi;

    float  etai_gi;
    float  etat_gi;
};

// Data management
// PACK1 (16) + PACK2 (16) + PACK3 (16) + PACK4 (16) + PACK5 (16) = 80 bytes
static const uint BYTES_GI    = 80u;

static const uint O_GI_PACK1  =  0u;   // float4 (x2, n2_s)
static const uint O_GI_PACK2  = 16u;   // float4
static const uint O_GI_PACK3  = 32u;   // float4
static const uint O_GI_PACK4  = 48u;   // float4
static const uint O_GI_PACK5  = 64u;   // uint4 (Kd, PrPm, Eta, n2_g)

static const uint O_GI_PACK_WSUM = 80u;   // float

uint pixelBaseAddrGI(uint pixelIdx) { return pixelIdx * BYTES_GI; }

uint  PackMatID_M(uint matID, uint M) { return (matID & 0xFFFFu) | (M << 16); }
void  UnpackMatID_M(uint v, out uint matID, out uint M)
{ matID = v & 0xFFFFu;  M = v >> 16; }

uint PackLobes16(uint a, uint b) { return ( (b & 0xFFu) << 8 ) | (a & 0xFFu); }
void UnpackLobes16(uint v16, out uint a, out uint b)
{
    a =  v16       & 0xFFu;
    b = (v16 >> 8) & 0xFFu;
}

void storeReservoirGI(RWByteAddressBuffer buf, uint pixelIdx, const Reservoir_GI r)
{
    uint base = pixelBaseAddrGI(pixelIdx);

    // Transform positions/normals to Object Space for storage
    float3 xO   = WorldToObjectPos (r.objID_gi, r.x2_gi);
    float3 nO_s = WorldToObjectNrm(r.objID_gi, r.n2_s_gi);
    float3 nO_g = WorldToObjectNrm(r.objID_gi, r.n2_g_gi);

    // Pack 1: x2 + Shading Normal
    buf.Store4(base + O_GI_PACK1, uint4(asuint(xO), PackNormal(nO_s)));

    // Pack 2: L2, V2, IDs
    buf.Store4(base + O_GI_PACK2, uint4(PackRGB9E5(r.L2_gi),
                                        PackNormal (r.V2_gi),
                                        r.objID_gi,
                                        PackMatID_M(r.matID_gi, r.M_gi)));

    // Pack 3: Jacobian
    buf.Store4(base + O_GI_PACK3, asuint(r.J_gi));

    // Pack 4: W, Seed, Lobes, F
    uint lobes16 = PackLobes16(r.lobe0_gi, r.lobe1_gi);
    uint rIndex16 = r.rIndex_gi & 0xFFFFu;
    uint rIndexAndLobes = rIndex16 | (lobes16 << 16);

    buf.Store4(base + O_GI_PACK4, uint4(asuint(r.W_gi),
                                        r.rSeed_gi,
                                        rIndexAndLobes,
                                        r.F_gi));

    // Pack 5 (Expanded): Kd, PrPm, Eta, Geometric Normal
    uint packedKd    = PackRGB9E5(r.localKd_gi);
    uint packedPrPm  = f32tof16(r.localPr_gi) | (f32tof16(r.localPm_gi) << 16);
    uint packedEta   = f32tof16(r.etai_gi)    | (f32tof16(r.etat_gi)    << 16);
    uint packedNormG = PackNormal(nO_g);

    // Store4 now (was Store3)
    buf.Store4(base + O_GI_PACK5, uint4(packedKd, packedPrPm, packedEta, packedNormG));
}

Reservoir_GI loadReservoirGI(RWByteAddressBuffer buf, uint pixelIdx)
{
    uint base = pixelBaseAddrGI(pixelIdx);

    uint4 p1 = buf.Load4(base + O_GI_PACK1);
    uint4 p2 = buf.Load4(base + O_GI_PACK2);
    uint4 p3 = buf.Load4(base + O_GI_PACK3);
    uint4 p4 = buf.Load4(base + O_GI_PACK4);

    // Load4 now for Pack 5
    uint4 p5 = buf.Load4(base + O_GI_PACK5);

    Reservoir_GI r;
    r.objID_gi = p2.z;
    UnpackMatID_M(p2.w, r.matID_gi, r.M_gi);

    // Pack 1: x2 + Shading Normal
    r.x2_gi     = ObjectToWorldPos (r.objID_gi, asfloat(p1.xyz));
    r.n2_s_gi   = ObjectToWorldNrm(r.objID_gi, UnpackNormal(p1.w));

    r.L2_gi     = UnpackRGB9E5(p2.x);
    r.V2_gi     = UnpackNormal (p2.y);

    r.J_gi      = asfloat(p3);

    r.W_gi      = asfloat(p4.x);
    r.rSeed_gi  = p4.y;

    uint rIndexAndLobes = p4.z;
    uint lobes16 = (rIndexAndLobes >> 16) & 0xFFFFu;
    r.rIndex_gi  =  rIndexAndLobes        & 0x0000FFFFu;
    UnpackLobes16(lobes16, r.lobe0_gi, r.lobe1_gi);

    r.F_gi      = p4.w;

    // Pack 5: Surface Params + Geometric Normal
    r.localKd_gi = UnpackRGB9E5(p5.x);

    r.localPr_gi = f16tof32(p5.y & 0xFFFF);
    r.localPm_gi = f16tof32(p5.y >> 16);

    r.etai_gi    = f16tof32(p5.z & 0xFFFF);
    r.etat_gi    = f16tof32(p5.z >> 16);

    r.n2_g_gi    = ObjectToWorldNrm(r.objID_gi, UnpackNormal(p5.w));

    r.w_sum_gi = 0.0f;
    return r;
}


// fast loaders
float3 load_x2_gi(RWByteAddressBuffer b, uint id, uint obj)
{ return ObjectToWorldPos(obj, asfloat(b.Load4(pixelBaseAddrGI(id)+O_GI_PACK1).xyz)); }

// Returns Shading Normal
float3 load_n2_s_gi(RWByteAddressBuffer b, uint id, uint obj)
{ return ObjectToWorldNrm(obj, UnpackNormal(b.Load4(pixelBaseAddrGI(id)+O_GI_PACK1).w)); }

// Returns Geometric Normal
float3 load_n2_g_gi(RWByteAddressBuffer b, uint id, uint obj)
{
    // 4th component of Pack 5
    uint packed = b.Load(pixelBaseAddrGI(id) + O_GI_PACK5 + 12u);
    return ObjectToWorldNrm(obj, UnpackNormal(packed));
}

float3 load_L2_gi(RWByteAddressBuffer b, uint id)
{ return UnpackRGB9E5(b.Load4(pixelBaseAddrGI(id)+O_GI_PACK2).x); }

float3 load_V2_gi(RWByteAddressBuffer b, uint id)
{ return UnpackNormal(b.Load4(pixelBaseAddrGI(id)+O_GI_PACK2).y); }

uint   load_objID_gi(RWByteAddressBuffer b, uint id)
{ return b.Load4(pixelBaseAddrGI(id)+O_GI_PACK2).z; }

uint   load_matID_gi(RWByteAddressBuffer b, uint id)
{ return b.Load4(pixelBaseAddrGI(id)+O_GI_PACK2).w & 0xFFFFu; }

uint   load_M_gi(RWByteAddressBuffer b, uint id)
{ return b.Load4(pixelBaseAddrGI(id)+O_GI_PACK2).w >> 16; }

float4 load_J_gi (RWByteAddressBuffer b, uint id)
{ return asfloat(b.Load4(pixelBaseAddrGI(id)+O_GI_PACK3)); }

float load_W_gi(RWByteAddressBuffer b, uint id)
{return asfloat(b.Load4(pixelBaseAddrGI(id) + O_GI_PACK4).x); }

inline float3 load_F_gi(RWByteAddressBuffer b, uint id)
{
    uint4 p4 = b.Load4(pixelBaseAddrGI(id) + O_GI_PACK4);
    return p4.w;
}

// Optimized fast load for reconnection: x2 + Shading Normal
inline void load_x2_n2_s_fast_gi(RWByteAddressBuffer b, uint id, uint obj,
                               out float3 x2, out float3 n2_s)
{
    uint4 p1 = b.Load4(pixelBaseAddrGI(id)+O_GI_PACK1);
    x2 = ObjectToWorldPos (obj, asfloat(p1.xyz));
    n2_s = ObjectToWorldNrm(obj, UnpackNormal(p1.w));
}

inline void load_L2_V2_fast_gi(RWByteAddressBuffer b, uint id,
                               out float3 L2, out float3 V2)
{
    uint4 p2 = b.Load4(pixelBaseAddrGI(id)+O_GI_PACK2);
    L2 = UnpackRGB9E5(p2.x);
    V2 = UnpackNormal (p2.y);
}

inline void load_IDs_fast_gi(RWByteAddressBuffer b, uint id,
                             out uint objID, out uint matID, out uint M)
{
    uint v = b.Load4(pixelBaseAddrGI(id)+O_GI_PACK2).w;
    objID = b.Load4(pixelBaseAddrGI(id)+O_GI_PACK2).z;
    UnpackMatID_M(v, matID, M);
}

void   load_WSeedIndexLobes_F(RWByteAddressBuffer b, uint id,
                              out float W, out uint rSeed, out uint rIndex,
                              out uint l0, out uint l1, out float3 F)
{
    uint4 p4 = b.Load4(pixelBaseAddrGI(id)+O_GI_PACK4);
    W      = asfloat(p4.x);
    rSeed  = p4.y;
    uint v = p4.z;
    rIndex = v & 0xFFFFu;
    uint lobes16 = (v >> 16) & 0xFFFFu;
    UnpackLobes16(lobes16, l0, l1);
    F = p4.w;
}

inline float3 load_localKd_gi(RWByteAddressBuffer b, uint id)
{
    uint packed = b.Load(pixelBaseAddrGI(id) + O_GI_PACK5);
    return UnpackRGB9E5(packed);
}

inline void load_localPrPm_gi(RWByteAddressBuffer b, uint id, out float Pr, out float Pm)
{
    uint packed = b.Load(pixelBaseAddrGI(id) + O_GI_PACK5 + 4u);
    Pr = f16tof32(packed & 0xFFFF);
    Pm = f16tof32(packed >> 16);
}

// Updated fast loader for surface + geo normal
inline void load_extended_surface_gi(RWByteAddressBuffer b, uint id, uint obj,
                                     out float3 Kd, out float Pr, out float Pm,
                                     out float etai, out float etat, out float3 n2_g)
{
    // Load4 to get Kd, PrPm, EtaI/EtaT, and n2_g
    uint4 p5 = b.Load4(pixelBaseAddrGI(id) + O_GI_PACK5);

    Kd = UnpackRGB9E5(p5.x);

    Pr = f16tof32(p5.y & 0xFFFF);
    Pm = f16tof32(p5.y >> 16);

    etai = f16tof32(p5.z & 0xFFFF);
    etat = f16tof32(p5.z >> 16);

    n2_g = ObjectToWorldNrm(obj, UnpackNormal(p5.w));
}

void store_wsum_gi(RWByteAddressBuffer buf, uint pixelIdx, float w_sum)
{
    // Stored strictly in its own memory slot at offset 80
    buf.Store(pixelBaseAddrGI(pixelIdx) + O_GI_PACK_WSUM, asuint(w_sum));
}

float load_wsum_gi(RWByteAddressBuffer buf, uint pixelIdx)
{
    return asfloat(buf.Load(pixelBaseAddrGI(pixelIdx) + O_GI_PACK_WSUM));
}

// Jacobian: Uses geometric normals for correct manifold calculations
float JacobianDeterminant( float3 x1_c,
                           float3 x2_c,
                           float3 x1_n_g, // Geometric
                           float3 n2_c_g  // Geometric
                         )
{
    float3  v_c   = x1_c - x2_c;
    float   distc = dot(v_c, v_c);
    float   cosc  = dot(normalize(v_c), n2_c_g);

    float3  v_n   = x1_n_g - x2_c;
    float   distn = dot(v_n, v_n);
    float   cosn  = dot(normalize(v_n), n2_c_g);

    float J = (cosc / cosn) * (distn / distc);

    return !isnan(J)?J:1e10;
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

inline bool RejectLength_GI(float3 x2_c, float3 n2_c_g,
                            float3 x1_c, float3 x1_n_g,
                            float  threshold)
{
    float J = JacobianDeterminant(x1_c, x2_c, x1_n_g, n2_c_g);
    return J <= threshold || J >= 1.0f/threshold;
}

inline bool IsValidReservoir_GI(Reservoir_GI r){
    bool valid =
        any(abs(r.n2_s_gi) > 0.0f) &&
        r.M_gi > 0.0f
        ;
    return valid;
}

inline bool IsValidReservoir_GI_opt(in float3 n2, in uint M){
    bool valid =
        any(abs(n2) > 0.0f) &&
        M > 0.0f;
    return valid;
}

float3 BSDF_term(
    uint   mID,
    float3 n_s,
    float3 n_g, // Added
    float3 s,
    float3 o,
    float3 localKd,
    float  localPr,
    float  localPm,
    float  etai,
    float  etat)
{
    return EvaluateBRDF_COMBINED(mID, n_s, n_g, s, o, localKd, localPr, localPm, etai, etat);
}

float PDF_term(
    uint   mID,
    float3 n_s,
    float3 n_g, // Added
    float3 s,
    float3 o,
    float3 localKd,
    float  localPr,
    float  localPm,
    float  etai,
    float  etat)
{
    SamplingP p = CalculateStrategyProbabilities(mID, s, n_s, etai, etat, localKd, localPm);
    return BRDF_PDF_COMBINED(p, mID, n_s, n_g, s, o, localKd, localPr, localPm, etai, etat);
}

// G term uses Geometric Normal
float G_term(float3 n_g, float3 s)
{
    return max(1e-15f, dot(n_g, s));
}

// J term uses Geometric Normal
float J_term(
    float3 n2_g,
    float3 s,
    float  dist)
{
    if (dot(n2_g, s) < 0.0f)
        n2_g = -n2_g;

    float cosThetaX2 = max(EPSILON, dot(n2_g, s));
    return cosThetaX2 / max(EPSILON, dist * dist);
}


// Calculate reconnection (two-sided) with split Shading/Geometric Normals
inline float3 ReconnectGI(
    // Surface 1 (Receiver / Current Pixel)
    in float3  x1,
    in float3  n1_s, // Shading
    in float3  n1_g, // Geometric
    in float3  o,
    in uint    mID1,
    in float3  localKd1, in float localPr1, in float localPm1, in float etai1, in float etat1,

    // Surface 2 (Source / Reservoir Sample)
    in float3  x2,
    in float3  n2_s, // Shading
    in float3  n2_g, // Geometric
    in float3  L2,
    in float3  V2,
    in uint    mID2,
    in float3  localKd2, in float localPr2, in float localPm2, in float etai2, in float etat2,

    // Reconnection Params
    in float pdfx2,
    in float Jc, // part of the jacobian term of the canonical path
    in bool applyJ,
    out float Jn,
    out float J)
{
    Jn = 1.0f;
    J = 1.0f;
    if (length(L2) < EPSILON)
        return 0;

    // Geometric prep
    float3 dir   = x2 - x1;
    float  dist  = length(dir);
    float3 ndirN = normalize(-dir); // Direction from x1 to x2

    // Evaluate BSDFs using Shading Normals (n1_s, n2_s) AND Geometric Normals (n1_g, n2_g)
    float3 F1 = BSDF_term(mID1, n1_s, n1_g, ndirN, o, localKd1, localPr1, localPm1, etai1, etat1);
    float3 F2 = BSDF_term(mID2, n2_s, n2_g, V2, -ndirN, localKd2, localPr2, localPm2, etai2, etat2);

    // G term uses Geometric Normal (n1_g)
    float G = G_term(n1_g, ndirN);

    // PDFs use both normals
    float PDF1 = PDF_term(mID1, n1_s, n1_g, ndirN, o, localKd1, localPr1, localPm1, etai1, etat1);
    float PDF2 = pdfx2;
    if(pdfx2 == 0.0f)
    {
        PDF2 = PDF_term(mID2, n2_s, n2_g, V2, -ndirN, localKd2, localPr2, localPm2, etai2, etat2);
    }

    // Throughput
    if(PDF1 <= 0.0f || PDF2 <= 0.0f)
        return 0.0f;

    float3 r = F1/PDF1 * F2/PDF2 * L2 * G;

    // Reconnection jacobian
    Jn = Jc;
    J = 1.0f;
    if(applyJ){
        // Jacobian uses Geometric Normal at x2 (n2_g)
        float Gj = max(0.0f, dot(-ndirN, n2_g)) / (dist * dist);
        Jn = (PDF1 * PDF2 * Gj);
        J = Jn / max(1e-20f, Jc);
    }

    if (any(isnan(r)) || all(r < EPSILON))
        r = (float3)0.0f;

    return r;
}

inline float PSSJacobian(
    // Surface 1
    in float3 x1,
    in float3 n1_s,
    in float3 n1_g,
    in float3 o,
    in uint   mID1,
    in float3 localKd1, in float localPr1, in float localPm1, in float etai1, in float etat1,

    // Surface 2
    in float3 x2,
    in float3 n2_s,
    in float3 n2_g,
    in float3 V2,
    in uint   mID2,
    in float3 localKd2, in float localPr2, in float localPm2, in float etai2, in float etat2,

    // Misc
    in float  pdfx2)
{
    float3 dir   = x2 - x1;
    float  dist2 = dot(dir, dir);
    float3 ndirN = normalize(-dir); // x1 -> x2

    // PDF at x1 (using n1_s / n1_g)
    float PDF1 = PDF_term(mID1, n1_s, n1_g, ndirN, o, localKd1, localPr1, localPm1, etai1, etat1);
    if (PDF1 <= 0.0f) return 0.0f;

    // PDF at x2 (using n2_s / n2_g)
    float PDF2 = pdfx2;
    if (PDF2 <= 0.0f)
    {
         PDF2 = PDF_term(mID2, n2_s, n2_g, V2, -ndirN, localKd2, localPr2, localPm2, etai2, etat2);
    }
    if (PDF2 <= 0.0f) return 0.0f;

    // Jacobian geometric term uses n2_g
    float Gj = max(0.0f, dot(-ndirN, n2_g)) / dist2;

    return PDF1 * PDF2 * Gj;
}


// Update GI reservoir
bool UpdateReservoirGI(
    inout Reservoir_GI reservoir,
    in float wi,
    in uint M,

    in float3 x2,
    in float3 n2_s, // Shading
    in float3 n2_g, // Geometric (Added)
    in float3 L2,
    in float3 V2,

    // Surface Params
    in float3 localKd,
    in float  localPr,
    in float  localPm,
    in float  etai,
    in float  etat,

    in uint matID,
    in uint objID,
    uint rSeed,
    float4 J,
    uint rIndex,
    in float F,
    in uint   lobe0,
    in uint   lobe1,
    inout uint2 seed
    )
{
    reservoir.w_sum_gi += wi;
    reservoir.M_gi += M;

    if (RandomFloatSingle(seed.x) < wi / reservoir.w_sum_gi)
    {
        reservoir.x2_gi      = x2;
        reservoir.n2_s_gi    = n2_s;
        reservoir.n2_g_gi    = n2_g; // Store Geometric
        reservoir.L2_gi      = L2;
        reservoir.V2_gi      = V2;

        reservoir.localKd_gi = localKd;
        reservoir.localPr_gi = localPr;
        reservoir.localPm_gi = localPm;
        reservoir.etai_gi    = etai;
        reservoir.etat_gi    = etat;

        reservoir.objID_gi   = objID;
        reservoir.matID_gi   = matID;
        reservoir.rSeed_gi   = rSeed;
        reservoir.J_gi       = J;
        reservoir.rIndex_gi  = rIndex;
        reservoir.F_gi       = F;
        reservoir.lobe0_gi   = lobe0;
        reservoir.lobe1_gi   = lobe1;
        return true;
    }
    return false;
}

bool UpdateReservoirGI_Initial(
    uint pixelIdx,

    in float wi,
    in uint M,

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
    uint rSeed,
    float4 J,
    uint rIndex,
    in float F,
    in uint   lobe0,
    in uint   lobe1,
    inout uint2 seed
)
{
    float current_w_sum = load_wsum_gi(g_Reservoirs_current_gi, pixelIdx);
    float new_w_sum = current_w_sum + wi;
    store_wsum_gi(g_Reservoirs_current_gi, pixelIdx, new_w_sum);

    if (RandomFloatSingle(seed.x) < (wi / new_w_sum))
    {
        Reservoir_GI r;

        r.x2_gi      = x2;
        r.n2_s_gi    = n2_s;
        r.n2_g_gi    = n2_g; // Store
        r.L2_gi      = L2;
        r.V2_gi      = V2;

        r.localKd_gi = localKd;
        r.localPr_gi = localPr;
        r.localPm_gi = localPm;
        r.etai_gi    = etai;
        r.etat_gi    = etat;

        r.objID_gi   = objID;
        r.matID_gi   = matID;
        r.M_gi       = M;

        r.rSeed_gi   = rSeed;
        r.J_gi       = J;
        r.rIndex_gi  = rIndex;
        r.F_gi       = F;
        r.lobe0_gi   = lobe0;
        r.lobe1_gi   = lobe1;

        r.W_gi       = 0.0f;

        r.w_sum_gi   = new_w_sum;
        storeReservoirGI(g_Reservoirs_current_gi, pixelIdx, r);

        return true;
    }

    return false;
}

// Ultra fast updater for path segments that are
bool UpdateReservoirGI_Initial_Fixed(
    uint pixelIdx,

    in float wi,
    in uint M,

    in float3 L2,
    in float  F,

    inout uint2 seed
)
{
    uint base = pixelBaseAddrGI(pixelIdx);

    float current_w_sum = asfloat(g_Reservoirs_current_gi.Load(base + O_GI_PACK_WSUM));
    float new_w_sum = current_w_sum + wi;
    g_Reservoirs_current_gi.Store(base + O_GI_PACK_WSUM, asuint(new_w_sum));

    uint  offsetM = base + O_GI_PACK2 + 12u;
    uint  packedID_M = g_Reservoirs_current_gi.Load(offsetM);

    uint  matID, currentM;
    UnpackMatID_M(packedID_M, matID, currentM);

    uint  newM = currentM + M;
    g_Reservoirs_current_gi.Store(offsetM, PackMatID_M(matID, newM));

    if (RandomFloatSingle(seed.x) < (wi / new_w_sum))
    {
        g_Reservoirs_current_gi.Store(base + O_GI_PACK2, PackRGB9E5(L2));
        g_Reservoirs_current_gi.Store(base + O_GI_PACK4 + 12u, asuint(F));

        return true;
    }

    return false;
}