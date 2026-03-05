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

static const uint BYTES_DI   = 40u;

static const uint O_PACK1 = 0u;    // float4 (Position + Normal)
static const uint O_PACK2 = 16u;   // uint2  (Radiance + W)
static const uint O_OBJID = 24u;   // uint   (objID)
static const uint O_M     = 28u;   // uint   (M)
static const uint O_WSUM  = 32u;   // float  (w_sum)
static const uint O_PHAT  = 36u;   // float  (p_hat)

// helpers
uint pixelBaseAddr(uint pixelIdx) { return pixelIdx * BYTES_DI; }

// ------------------------------------------------------------------
// Store Functions
// ------------------------------------------------------------------


void storeReservoirDI(RWByteAddressBuffer buf, uint pixelIdx, const Reservoir_DI r)
{
    uint base = pixelBaseAddr(pixelIdx);

    float3 pStore;
    uint   nStore;

    if (r.objID_di == 0xFFFFFFFFu || r.objID_di == 0xFFFFFFFEu)
    {
        // Store direction directly (world space)
        pStore = r.x2_di;                      // already a direction
        nStore = PackNormal(normalize(r.n2_di)); // can be dummy; keep stable
    }
    else
    {
        // Store surface position/normal in object space
        float3 xO = WorldToObjectPos (r.objID_di, r.x2_di);
        float3 nO = WorldToObjectNrm(r.objID_di, r.n2_di);
        pStore = xO;
        nStore = PackNormal(nO);
    }

    buf.Store4(base + O_PACK1, uint4(asuint(pStore), nStore));
    buf.Store2(base + O_PACK2, uint2(PackRGB9E5(r.L2_di), asuint(r.W_di)));

    buf.Store(base + O_OBJID, r.objID_di);
    buf.Store(base + O_M,     r.M_di);
    buf.Store(base + O_WSUM,  asuint(r.w_sum_di));
}

// ------------------------------------------------------------------
// Load Functions
// ------------------------------------------------------------------

Reservoir_DI loadReservoirDI(RWByteAddressBuffer buf, uint pixelIdx)
{
    Reservoir_DI r;
    uint base = pixelBaseAddr(pixelIdx);

    uint4 p1 = buf.Load4(base + O_PACK1);
    float3 pRaw = asfloat(p1.xyz);
    uint   nEnc = p1.w;

    uint2 p2 = buf.Load2(base + O_PACK2);
    uint  Lenc = p2.x;

    r.W_di     = asfloat(p2.y);
    r.objID_di = buf.Load(base + O_OBJID);
    r.M_di     = buf.Load(base + O_M);
    r.w_sum_di = asfloat(buf.Load(base + O_WSUM));
    r.L2_di    = UnpackRGB9E5(Lenc);

    if (r.objID_di == 0xFFFFFFFFu || r.objID_di == 0xFFFFFFFEu)
    {
        // Interpret payload as environment direction
        r.x2_di = pRaw;                 // direction in world space
        r.n2_di = UnpackNormal(nEnc);   // unused; keep for validity if you want
    }
    else
    {
        // Interpret payload as surface point in object space
        r.x2_di = ObjectToWorldPos (r.objID_di, pRaw);
        r.n2_di = ObjectToWorldNrm(r.objID_di, UnpackNormal(nEnc));
    }

    return r;
}


// Single loaders
float3 load_x2_di(RWByteAddressBuffer buf, uint pixelIdx, uint objID)
{
    uint4 p1 = buf.Load4(pixelBaseAddr(pixelIdx) + O_PACK1);
    float3 pRaw = asfloat(p1.xyz);
    return (objID == 0xFFFFFFFFu|| objID == 0xFFFFFFFEu) ? pRaw : ObjectToWorldPos(objID, pRaw);
}

float3 load_n2_di(RWByteAddressBuffer buf, uint pixelIdx, uint objID)
{
    uint  nEnc = buf.Load4(pixelBaseAddr(pixelIdx) + O_PACK1).w;
    return ObjectToWorldNrm(objID, UnpackNormal(nEnc));
}

float3 load_L2_di(RWByteAddressBuffer buf, uint pixelIdx)
{
    uint Lenc = buf.Load2(pixelBaseAddr(pixelIdx) + O_PACK2).x;
    return UnpackRGB9E5(Lenc);
}

float  load_W_di(RWByteAddressBuffer buf, uint pixelIdx)
{
    return asfloat(buf.Load2(pixelBaseAddr(pixelIdx) + O_PACK2).y);
}

uint load_M_di(RWByteAddressBuffer buf, uint pixelIdx)
{
    return buf.Load(pixelBaseAddr(pixelIdx) + O_M);
}

uint load_objID_di(RWByteAddressBuffer buf, uint pixelIdx)
{
    return buf.Load(pixelBaseAddr(pixelIdx) + O_OBJID);
}

void store_W_di(RWByteAddressBuffer buf, uint pixelIdx, float W)
{
    buf.Store(pixelBaseAddr(pixelIdx) + O_PACK2 + 4, asuint(W));
}

void store_M_di(RWByteAddressBuffer buf, uint pixelIdx, uint M)
{
    buf.Store(pixelBaseAddr(pixelIdx) + O_M, M);
}

// Load and store for p_hat caching
void store_phat_di(RWByteAddressBuffer buf, uint pixelIdx, float p_hat)
{
    buf.Store(pixelBaseAddr(pixelIdx) + O_PHAT, asuint(p_hat));
}

float load_phat_di(RWByteAddressBuffer buf, uint pixelIdx)
{
    return asfloat(buf.Load(pixelBaseAddr(pixelIdx) + O_PHAT));
}

// Load and store w sum
void store_wsum_di(RWByteAddressBuffer buf, uint pixelIdx, float w_sum)
{
    uint base = pixelBaseAddr(pixelIdx);
    buf.Store(base + O_WSUM, asuint(w_sum));
}

float load_wsum_di(RWByteAddressBuffer buf, uint pixelIdx)
{
    return asfloat(buf.Load(pixelBaseAddr(pixelIdx) + O_WSUM));
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
        //any(r.L2_di > 0.0f) &&
        //r.W_di > 0.0f &&
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

    // Sun
    if (objID_di == 0xFFFFFFFEu)
    {
        // SUN SHIT
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
    //float   J = J_term(n2, ndirN, dist);

    // Throughput
    float3 r = F * L * G; //* J;

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
    float3 L2, // No need to update L1, as this is always 0 when the sample is processed here. Also,we dont want to reuse sample on a lights surface
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
    uint base = pixelBaseAddr(pixelIdx);
    uint wSumAddr = base + O_WSUM;
    float currentWSum = asfloat(buf.Load(wSumAddr));

    float newWSum = currentWSum + wi;

    buf.Store(wSumAddr, asuint(newWSum));

    bool isAccepted = false;

    if (RandomFloatSingle(seed.x) < (wi / newWSum))
    {
        isAccepted = true;

        float3 xO = WorldToObjectPos(objID, x2);
        float3 nO = WorldToObjectNrm(objID, n2);
        buf.Store4(base + O_PACK1, uint4(asuint(xO), PackNormal(nO)));

        buf.Store2(base + O_PACK2, uint2(PackRGB9E5(L2), asuint(0.0f)));

        buf.Store(base + O_OBJID, objID);
        buf.Store(base + O_M,     1);
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
    uint base = pixelBaseAddr(pixelIdx);

    uint wSumAddr = base + O_WSUM;
    float currentWSum = asfloat(buf.Load(wSumAddr));

    float newWSum = currentWSum + wi;

    buf.Store(wSumAddr, asuint(newWSum));

    bool isAccepted = false;

    if (RandomFloatSingle(seed.x) < (wi / newWSum))
    {
        isAccepted = true;

        buf.Store4(base + O_PACK1, uint4(asuint(dir), PackNormal(normalize(float3(1,1,0.5f)))));

        buf.Store2(base + O_PACK2, uint2(PackRGB9E5(L2), asuint(0.0f)));

        buf.Store(base + O_OBJID, objID);   // 0xFFFFFFFFu now preserved
        buf.Store(base + O_M,     1);
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
