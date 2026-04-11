// RIS reservoir for direct lighting
struct Reservoir_DI
{
    float3 x2_di;
    float3 n2_di;
    float W_di;
    float w_sum_di;
    float3 L2_di;
    uint M_di;
    uint objID_di;
};

// SoA layout — each field is a contiguous plane across all pixels.
// Total buffer size per pixel is unchanged (40 bytes).
static const uint BYTES_DI = 40u;

// Per-pixel field sizes (bytes)
static const uint DI_SZ_PACK1 = 16u;  // position(12) + packed normal(4)
static const uint DI_SZ_L2    =  4u;  // PackRGB9E5
static const uint DI_SZ_W     =  4u;
static const uint DI_SZ_OBJID =  4u;
static const uint DI_SZ_M     =  4u;
static const uint DI_SZ_WSUM  =  4u;
static const uint DI_SZ_PHAT  =  4u;

// Plane base offsets (multiply by totalPixels to get byte offset)
// Layout: PACK1 | L2 | W | OBJID | M | WSUM | PHAT
static const uint DI_PLANE_PACK1 =  0u;
static const uint DI_PLANE_L2    = 16u;
static const uint DI_PLANE_W     = 20u;
static const uint DI_PLANE_OBJID = 24u;
static const uint DI_PLANE_M     = 28u;
static const uint DI_PLANE_WSUM  = 32u;
static const uint DI_PLANE_PHAT  = 36u;

// SoA address helpers
// Tile-aligned pixel count: MapPixelID uses 4x8 tiles, so the max pixelID
// can exceed IMG_W*IMG_H when dimensions aren't multiples of 4/8.
// SoA planes must be sized to the aligned count to avoid cross-plane overlap.
uint di_numPx()                      { return ((IMG_W + 3u) / 4u) * ((IMG_H + 7u) / 8u) * 32u; }
uint di_addr_pack1(uint px)          { return px * DI_SZ_PACK1; }
uint di_addr_l2(uint px)             { uint N = di_numPx(); return N * DI_PLANE_L2    + px * DI_SZ_L2; }
uint di_addr_w(uint px)              { uint N = di_numPx(); return N * DI_PLANE_W     + px * DI_SZ_W; }
uint di_addr_objid(uint px)          { uint N = di_numPx(); return N * DI_PLANE_OBJID + px * DI_SZ_OBJID; }
uint di_addr_m(uint px)              { uint N = di_numPx(); return N * DI_PLANE_M     + px * DI_SZ_M; }
uint di_addr_wsum(uint px)           { uint N = di_numPx(); return N * DI_PLANE_WSUM  + px * DI_SZ_WSUM; }
uint di_addr_phat(uint px)           { uint N = di_numPx(); return N * DI_PLANE_PHAT  + px * DI_SZ_PHAT; }

// ------------------------------------------------------------------
// Store Functions
// ------------------------------------------------------------------

void storeReservoirDI(RWByteAddressBuffer buf, uint pixelIdx, const Reservoir_DI r)
{
    float3 pStore;
    uint   nStore;

    if (r.objID_di == 0xFFFFFFFFu || r.objID_di == 0xFFFFFFFEu)
    {
        pStore = r.x2_di;
        nStore = PackNormal(normalize(r.n2_di));
    }
    else
    {
        float3 xO = WorldToObjectPos (r.objID_di, r.x2_di);
        float3 nO = WorldToObjectNrm(r.objID_di, r.n2_di);
        pStore = xO;
        nStore = PackNormal(nO);
    }

    buf.Store4(di_addr_pack1(pixelIdx), uint4(asuint(pStore), nStore));
    buf.Store (di_addr_l2(pixelIdx),    PackRGB9E5(r.L2_di));
    buf.Store (di_addr_w(pixelIdx),     asuint(r.W_di));
    buf.Store (di_addr_objid(pixelIdx), r.objID_di);
    buf.Store (di_addr_m(pixelIdx),     r.M_di);
    buf.Store (di_addr_wsum(pixelIdx),  asuint(r.w_sum_di));
}

// ------------------------------------------------------------------
// Load Functions
// ------------------------------------------------------------------

Reservoir_DI loadReservoirDI(RWByteAddressBuffer buf, uint pixelIdx)
{
    Reservoir_DI r;

    uint4  p1   = buf.Load4(di_addr_pack1(pixelIdx));
    float3 pRaw = asfloat(p1.xyz);
    uint   nEnc = p1.w;

    r.W_di     = asfloat(buf.Load(di_addr_w(pixelIdx)));
    r.objID_di = buf.Load(di_addr_objid(pixelIdx));
    r.M_di     = buf.Load(di_addr_m(pixelIdx));
    r.w_sum_di = asfloat(buf.Load(di_addr_wsum(pixelIdx)));
    r.L2_di    = UnpackRGB9E5(buf.Load(di_addr_l2(pixelIdx)));

    if (r.objID_di == 0xFFFFFFFFu || r.objID_di == 0xFFFFFFFEu)
    {
        r.x2_di = pRaw;
        r.n2_di = UnpackNormal(nEnc);
    }
    else
    {
        r.x2_di = ObjectToWorldPos (r.objID_di, pRaw);
        r.n2_di = ObjectToWorldNrm(r.objID_di, UnpackNormal(nEnc));
    }

    return r;
}


// Single loaders
float3 load_x2_di(RWByteAddressBuffer buf, uint pixelIdx, uint objID)
{
    uint4 p1 = buf.Load4(di_addr_pack1(pixelIdx));
    float3 pRaw = asfloat(p1.xyz);
    return (objID == 0xFFFFFFFFu|| objID == 0xFFFFFFFEu) ? pRaw : ObjectToWorldPos(objID, pRaw);
}

float3 load_n2_di(RWByteAddressBuffer buf, uint pixelIdx, uint objID)
{
    uint  nEnc = buf.Load4(di_addr_pack1(pixelIdx)).w;
    return ObjectToWorldNrm(objID, UnpackNormal(nEnc));
}

float3 load_L2_di(RWByteAddressBuffer buf, uint pixelIdx)
{
    return UnpackRGB9E5(buf.Load(di_addr_l2(pixelIdx)));
}

float  load_W_di(RWByteAddressBuffer buf, uint pixelIdx)
{
    return asfloat(buf.Load(di_addr_w(pixelIdx)));
}

uint load_M_di(RWByteAddressBuffer buf, uint pixelIdx)
{
    return buf.Load(di_addr_m(pixelIdx));
}

uint load_objID_di(RWByteAddressBuffer buf, uint pixelIdx)
{
    return buf.Load(di_addr_objid(pixelIdx));
}

void store_W_di(RWByteAddressBuffer buf, uint pixelIdx, float W)
{
    buf.Store(di_addr_w(pixelIdx), asuint(W));
}

void store_M_di(RWByteAddressBuffer buf, uint pixelIdx, uint M)
{
    buf.Store(di_addr_m(pixelIdx), M);
}

// Load and store for p_hat caching
void store_phat_di(RWByteAddressBuffer buf, uint pixelIdx, float p_hat)
{
    buf.Store(di_addr_phat(pixelIdx), asuint(p_hat));
}

float load_phat_di(RWByteAddressBuffer buf, uint pixelIdx)
{
    return asfloat(buf.Load(di_addr_phat(pixelIdx)));
}

// Load and store w sum
void store_wsum_di(RWByteAddressBuffer buf, uint pixelIdx, float w_sum)
{
    buf.Store(di_addr_wsum(pixelIdx), asuint(w_sum));
}

float load_wsum_di(RWByteAddressBuffer buf, uint pixelIdx)
{
    return asfloat(buf.Load(di_addr_wsum(pixelIdx)));
}


//---------------------------------------------------------------------------------------
inline bool RejectNormal_DI(float3 n1, float3 n2, float threshold){
    float similarity = dot(n1, n2);
    return (similarity < threshold);
}

inline bool RejectDistance_DI(float3 x1, float3 x2, float3 normal, float threshold)
{
    float dist = abs(dot(x2 - x1, normal));
    return dist > threshold;
}

inline bool IsValidReservoir_DI(Reservoir_DI r){
    bool valid =
        any(abs(r.n2_di) > 0.0f) &&
        r.M_di > 0.0f
        ;
    return valid;
}

inline bool IsValidReservoir_DI_opt(in float3 n2, in uint M){
    bool valid =
        any(abs(n2) > 0.0f) &&
        M > 0.0f;
    return valid;
}


float3 BSDF_term(
    uint   mID,
    float3 n_s,
    float3 n_g,
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
    float3 n_g,
    float3 s,
    float3 o,
    float3 localKd,
    float  localPr,
    float  localPm,
    float  etai,
    float  etat)
{
    SamplingP p = CalculateStrategyProbabilities(mID, o, n_s, etai, etat, localKd, localPm);
    return BRDF_PDF_COMBINED(p, mID, n_s, n_g, s, o, localKd, localPr, localPm, etai, etat);
}

// G term uses Geometric Normal
float G_term(float3 n_g, float3 s)
{
    return abs(dot(n_g, s));
}

float J_term(
    float3 n2,
    float3 ndirN,
    float  dist)
{
    if (dot(n2, ndirN) < 0.0f)
        n2 = -n2;

    float cosThetaX2 = max(EPSILON, dot(n2, ndirN));
    return cosThetaX2 / max(EPSILON, dist * dist);
}

// Calculate reconnection
inline float3 ReconnectDI(
    in float3 x1,
    in float3 n1_s,
    in float3 n1_g,
    in float3 o,
    in uint   mID,
    in float3 x2,
    in float3 n2,
    in float3 L,
    float3 localKd,
    float  localPr,
    float  localPm,
    float  etai,
    float  etat,
    in uint objID_di)
{
    if (all(L < EPSILON))
        return 0;

    // Infinite light
    if (objID_di == 0xFFFFFFFFu)
    {
        float3 wi = normalize(x2);
        float3 F = BSDF_term(mID, n1_s, n1_g, wi, o, localKd, localPr, localPm, etai, etat);

        float cosTheta = max(1e-15f, dot(n1_s, wi));

        float3 r = F * L * cosTheta;
        if (any(isnan(r)))
            r = 0;
        return r;
    }

    // Sun (should not appear in DI reservoirs anymore — sun is handled
    // separately via scratch ping. Keep as fallback for safety.)
    if (objID_di == 0xFFFFFFFEu)
    {
        float3 wi = normalize(x2);
        float3 Le = EvaluateSun(wi);

        if (all(Le < EPSILON))
            return 0;

        float3 F = BSDF_term(mID, n1_s, n1_g, wi, o, localKd, localPr, localPm, etai, etat);
        float  c = max(1e-15f, dot(n1_s, wi));

        float3 r = F * Le * c;
        return any(isnan(r)) ? 0 : r;
    }

    // Geometric prep
    float3 dir   = x2 - x1;
    float  dist  = length(dir);
    float3 ndirN = normalize(-dir);     // direction from x1 to x2, negated

    // Terms
    float3 F = BSDF_term(mID, n1_s, n1_g, -ndirN, o, localKd, localPr, localPm, etai, etat);
    float   G = G_term(n1_s, -ndirN);

    // Throughput
    float3 r = F * L * G;

    if (any(isnan(r)))
        r = 0;

    return r;
}

// Update DI reservoir
bool UpdateReservoirDI(
    inout Reservoir_DI reservoir,
    float wi,
    uint M,

    float3 x2,
    float3 n2,
    float3 L2,
    uint   objID,
    inout uint2 seed
    )
{

    reservoir.w_sum_di += wi;
    reservoir.M_di += M;

    if (RandomFloatSingle(seed.x) < wi / reservoir.w_sum_di)
    {
        reservoir.x2_di = x2;
        reservoir.n2_di = n2;
        reservoir.L2_di = L2;
        reservoir.objID_di = objID;
        return true;
    }
    return false;
}

bool UpdateReservoirDI_Fast(
    RWByteAddressBuffer buf,
    uint pixelIdx,
    float wi,     // Weight of new sample
    float3 x2,    // New Sample Pos
    float3 n2,    // New Sample Normal
    float3 L2,    // New Sample Radiance
    uint objID,   // New Sample ID
    inout uint2 seed
)
{
    float currentWSum = asfloat(buf.Load(di_addr_wsum(pixelIdx)));
    float newWSum = currentWSum + wi;
    buf.Store(di_addr_wsum(pixelIdx), asuint(newWSum));

    bool isAccepted = false;

    if (RandomFloatSingle(seed.x) < (wi / newWSum))
    {
        isAccepted = true;

        float3 xO = WorldToObjectPos(objID, x2);
        float3 nO = WorldToObjectNrm(objID, n2);
        buf.Store4(di_addr_pack1(pixelIdx), uint4(asuint(xO), PackNormal(nO)));
        buf.Store (di_addr_l2(pixelIdx),    PackRGB9E5(L2));
        buf.Store (di_addr_w(pixelIdx),     asuint(0.0f));
        buf.Store (di_addr_objid(pixelIdx), objID);
        buf.Store (di_addr_m(pixelIdx),     1);
    }

    return isAccepted;
}

bool UpdateReservoirDI_Infinite(
    RWByteAddressBuffer buf,
    uint pixelIdx,
    float wi,     // Weight of new sample
    float3 dir,   // Light Direction
    float3 L2,    // New Sample Radiance
    uint objID,   // Light ID
    inout uint2 seed
)
{
    float currentWSum = asfloat(buf.Load(di_addr_wsum(pixelIdx)));
    float newWSum = currentWSum + wi;
    buf.Store(di_addr_wsum(pixelIdx), asuint(newWSum));

    bool isAccepted = false;

    if (RandomFloatSingle(seed.x) < (wi / newWSum))
    {
        isAccepted = true;

        buf.Store4(di_addr_pack1(pixelIdx), uint4(asuint(dir), PackNormal(normalize(float3(1,1,0.5f)))));
        buf.Store (di_addr_l2(pixelIdx),    PackRGB9E5(L2));
        buf.Store (di_addr_w(pixelIdx),     asuint(0.0f));
        buf.Store (di_addr_objid(pixelIdx), objID);   // 0xFFFFFFFFu now preserved
        buf.Store (di_addr_m(pixelIdx),     1);
    }

    return isAccepted;
}

// Conversion to scalar value used for phat (luminance)
inline float GetPHat(float3 v){
    return 0.2126f * v.x + 0.7152f * v.y + 0.0722f * v.z;
}


float JacobianDeterminantDI(
    float3 x1_c,
    float3 x2_c,
    float3 x1_n,
    float3 n2_c,
    uint   objID
)
{
    if (objID == 0xFFFFFFFFu || objID == 0xFFFFFFFEu)
        return 1.0f;

    float3  v_c   = x1_c - x2_c;
    float   distc = dot(v_c, v_c);
    float   cosc  = abs(dot(normalize(v_c), n2_c));

    float3  v_n   = x1_n - x2_c;
    float   distn = dot(v_n, v_n);
    float   cosn  = abs(dot(normalize(v_n), n2_c));

    float J = (cosn / max(cosc,1e-4)) * (distc / max(distn,1e-4));

    return !isnan(J)?J:0.0f;
}
