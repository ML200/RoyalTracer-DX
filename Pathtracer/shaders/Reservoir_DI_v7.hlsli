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

inline bool RejectNormal_DI(float3 n1, float3 n2, float threshold){
    float similarity = dot(n1, n2);
    return (similarity < threshold);
}
/*inline bool RejectDistance_DI(float3 x1, float3 x2, float3 camPos, float threshold)
{
    float d1 = length(x1 - camPos);
    float d2 = length(x2 - camPos);

    float relativeDifference = abs(d1 - d2) / max(d1, d2);
    return relativeDifference > threshold;
}*/

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
    float3 n1,
    float3 ndirN,
    float3 o)
{
    float2  p      = CalculateStrategyProbabilities(mID, o, n1);
    float3  f0     = EvaluateBRDF(0, mID, n1, ndirN, o);
    float3  f1     = EvaluateBRDF(1, mID, n1, ndirN, o);

    return SafeMultiply(p.x, f0) +
           SafeMultiply(p.y, f1);
}

float PDF_term(
    uint   mID,
    float3 n1,
    float3 ndirN,
    float3 o)
{
    float2  p      = CalculateStrategyProbabilities(mID, o, n1);
    float3  pdf0     = BRDF_PDF(0, mID, n1, ndirN, o);
    float3  pdf1     = BRDF_PDF(1, mID, n1, ndirN, o);

    return SafeMultiply(p.x, pdf0) +
           SafeMultiply(p.y, pdf1);
}

float G_term(float3 n1, float3 ndirN)
{
    return max(1e-15, dot(n1, -ndirN));
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
    in float3 n1,
    in float3 o,
    in uint   mID,
    in float3 x2,
    in float3 n2,
    in float3 L)
{
    if (all(L < EPSILON))
        return 0;

    // Geometric prep
    float3 dir   = x2 - x1;
    float  dist  = length(dir);
    float3 ndirN = normalize(-dir);     // direction from x1 to x2, negated

    // Terms
    float3 F = BSDF_term(mID, n1, ndirN, o);
    float   G = G_term(n1, ndirN);
    float   J = J_term(n2, ndirN, dist);

    // Throughput
    float3 r = F * L * (G * J);

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

static const uint BYTES_DI   = 28u;

static const uint O_PACK1 = 0u;   // float4
static const uint O_PACK2 = 16u;  // float2
static const uint O_PACK3 = 24u;  // uint (via float)

// helpers
uint pixelBaseAddr(uint pixelIdx) { return pixelIdx * BYTES_DI; }

uint  PackObjID_M(uint objID, uint M)      { return (objID & 0xFFFFu) | (M << 16); }
void  UnpackObjID_M(uint v, out uint o, out uint m) { o = v & 0xFFFFu;  m = v >> 16; }

void storeReservoirDI(RWByteAddressBuffer buf, uint pixelIdx, const Reservoir_DI r)
{
    uint base = pixelBaseAddr(pixelIdx);
    float3 xO = WorldToObjectPos (r.objID_di, r.x2_di);
    float3 nO = WorldToObjectNrm(r.objID_di, r.n2_di);
    buf.Store4(base + O_PACK1, uint4(asuint(xO), PackNormal(nO)));
    buf.Store2(base + O_PACK2, uint2(PackRGB9E5(r.L2_di), asuint(r.W_di)));
    buf.Store (base + O_PACK3, PackObjID_M(r.objID_di, r.M_di));
}

Reservoir_DI loadReservoirDI(RWByteAddressBuffer buf, uint pixelIdx)
{
    Reservoir_DI r;
    uint base = pixelBaseAddr(pixelIdx);

    uint4 p1 = buf.Load4(base + O_PACK1);
    float3 xO   = asfloat(p1.xyz);
    uint   nEnc = p1.w;

    uint2 p2   = buf.Load2(base + O_PACK2);
    uint  Lenc = p2.x;
    float W    = asfloat(p2.y);

    uint  p3;
    p3 = buf.Load(base + O_PACK3);
    UnpackObjID_M(p3, r.objID_di, r.M_di);

    r.x2_di = ObjectToWorldPos (r.objID_di, xO);
    r.n2_di = ObjectToWorldNrm(r.objID_di, UnpackNormal(nEnc));
    r.L2_di = UnpackRGB9E5(Lenc);
    r.W_di  = W;
    r.w_sum_di = 0.0f;

    return r;
}

// Loader functions to load single
float3 load_x2_di(RWByteAddressBuffer buf, uint pixelIdx, uint objID)
{
    uint4 p1 = buf.Load4(pixelBaseAddr(pixelIdx) + O_PACK1);
    return ObjectToWorldPos(objID, asfloat(p1.xyz));
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

uint   load_M_di(RWByteAddressBuffer buf, uint pixelIdx)
{
    return buf.Load(pixelBaseAddr(pixelIdx) + O_PACK3) >> 16;
}

uint   load_objID_di(RWByteAddressBuffer buf, uint pixelIdx)
{
    return buf.Load(pixelBaseAddr(pixelIdx) + O_PACK3) & 0xFFFFu;
}

// Fast loaders (no speed impact)
inline void load_x2_n2_fast_di(RWByteAddressBuffer buf,
                             uint                pixelIdx,
                             uint                objID,
                             out float3          x2_world,
                             out float3          n2_world)
{
    uint4 p1 = buf.Load4(pixelBaseAddr(pixelIdx) + O_PACK1);
    float3 xO  = asfloat(p1.xyz);
    uint   nEnc= p1.w;
    x2_world   = ObjectToWorldPos (objID, xO);
    n2_world   = ObjectToWorldNrm(objID, UnpackNormal(nEnc));
}

inline void load_L2_W_fast_di(RWByteAddressBuffer buf,
                            uint               pixelIdx,
                            out float3         L2,
                            out float          W)
{
    uint2 p2 = buf.Load2(pixelBaseAddr(pixelIdx) + O_PACK2);
    L2 = UnpackRGB9E5(p2.x);
    W  = asfloat(p2.y);
}

inline void load_IDs_fast_di(RWByteAddressBuffer buf,
                           uint               pixelIdx,
                           out uint           objID,
                           out uint           M)
{
    UnpackObjID_M(buf.Load(pixelBaseAddr(pixelIdx) + O_PACK3), objID, M);
}

