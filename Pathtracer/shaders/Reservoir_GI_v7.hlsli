// RIS reservoir for direct lighting
struct Reservoir_GI
{
    float3 x2_gi;
    float3 n2_gi;
    float W_gi;
    float w_sum_gi;
    float3 L2_gi;
    float3 V2_gi;
    uint M_gi;
};

inline bool RejectNormal_gI(float3 n1, float3 n2, float threshold){
    float similarity = dot(n1, n2);
    return (similarity < threshold);
}
inline bool RejectDistance_GI(float3 x1, float3 x2, float3 camPos, float threshold)
{
    float d1 = length(x1 - camPos);
    float d2 = length(x2 - camPos);

    float relativeDifference = abs(d1 - d2) / max(d1, d2);
    return relativeDifference > threshold;
}

inline bool IsValidReservoir_GI(Reservoir_GI r){
    bool valid =
        any(abs(r.n2_gi) > 0.0f) &&
        any(r.L2_gi > 0.0f) &&
        r.W_gi > 0.0f &&
        r.M_gi > 0.0f
        ;
    return valid;
}



// Calculate reconnection (two–sided)
float3 ReconnectGI(
    float3  x1,
    float3  n1,
    float3  o,
    uint    mID1,

    uint    mID2,
    float3  x2,
    float3  n2,
    float3  L2,
    float3  V2
)
{
    if (all(L2 < EPSILON))
        return float3(0, 0, 0);

    float3 dir      = x2 - x1;
    float  dist     = length(dir);
    float3 ndir     = dir / dist;

    if (dot(n2, -ndir) < 0.0f)
        n2 = -n2;

    float  cosThetaX1 = max(EPSILON, dot(n1,  ndir));
    float  cosThetaX2 = max(EPSILON, dot(n2, -V2));

    // BRDF and MIS weights at x1
    float2 probs1 = CalculateStrategyProbabilities(mID1, o, n1);

    // incoming  = ndir (x1→x2)
    // outgoing  = o    (already given)
    float3 brdf1_0 = EvaluateBRDF(0, mID1, n1, ndir, o);
    float3 brdf1_1 = EvaluateBRDF(1, mID1, n1, ndir, o);
    float3 F1      = SafeMultiply(probs1.x, brdf1_0) +
                     SafeMultiply(probs1.y, brdf1_1);

    // BRDF and MIS weights at x2
    float2 probs2 = CalculateStrategyProbabilities(mID2, -ndir, n2);

    // outgoing  = -ndir (x2→x1)
    // incoming  = V2
    float3 brdf2_0 = EvaluateBRDF(0, mID2, n2, V2, -ndir);
    float3 brdf2_1 = EvaluateBRDF(1, mID2, n2, V2, -ndir);
    float3 F2      = SafeMultiply(probs2.x, brdf2_0) +
                     SafeMultiply(probs2.y, brdf2_1);

    // Final reconnection throughput (no geometry term because in angle space)
    float3 r = F1 * F2
             * L2
             * cosThetaX1 * cosThetaX2;

    // Guard against NaNs.
    if (any(isnan(r)))
        r = float3(0, 0, 0);

    return r;
}

// Calculate reconnection (two–sided)
float3 Reconnect_partial(
float3 x1,
float3 n1,
float3 o,
uint mID,
float3 x2
)
{
    float3 dir = x2 - x1;
    float3 ndirN = normalize(-dir);
    float dist = length(dir);

    float cosThetaX1 = max(EPSILON,dot(n1, -ndirN));

    float2 probs = CalculateStrategyProbabilities(mID, o, n1);
    float3 brdf0 = EvaluateBRDF(0, mID, n1, ndirN, o);
    float3 brdf1 = EvaluateBRDF(1, mID, n1, ndirN, o);
    float3 F1 = SafeMultiply(probs.x, brdf0);
    float3 F2 = SafeMultiply(probs.y, brdf1);
    float3 F = F1 + F2;

    float3 r = F * cosThetaX1;
    if(any(isnan(r)))
        r = float3(0,0,0);
    return r;
}


// Update DI reservoir
bool UpdateReservoirGI(
    inout Reservoir_GI reservoir,
    float wi,
    uint M,

    float3 x2,
    float3 n2,
    float3 L2, // No need to update L1, as this is always 0 when the sample is processed here. Also,we dont want to reuse sample on a lights surface
    float3 V2,
    inout uint2 seed
    )
{

    reservoir.w_sum_gi += wi;
    reservoir.M_gi += M;

    if (RandomFloatSingle(seed.x) < wi / reservoir.w_sum_gi)
    {
        reservoir.x2_gi = x2;
        reservoir.n2_gi = n2;
        reservoir.L2_gi = L2;
        reservoir.V2_gi = V2;
        return true;
    }
    return false;
}

// ── per-pixel field sizes (bytes) ──────────────────────────────────────────
static const uint B_x2_gi = 12;   // float3  – sample position
static const uint B_n2_gi =  4;   // packed  – normal at x2
static const uint B_L2_gi =  4;   // packed  – incident radiance
static const uint B_V2_gi =  4;   // packed  – V2 direction  (WAS 12)
static const uint B_W_gi  =  4;   // float   – reservoir weight
static const uint B_M_gi  =  4;   // uint    – sample count

// ── byte offsets of each field block within the buffer ─────────────────────
static const uint P_x2_gi = 0;
static const uint P_n2_gi = P_x2_gi + B_x2_gi;
static const uint P_L2_gi = P_n2_gi + B_n2_gi;
static const uint P_V2_gi = P_L2_gi + B_L2_gi;
static const uint P_W_gi  = P_V2_gi + B_V2_gi;
static const uint P_M_gi  = P_W_gi  + B_W_gi;

// helper: total number of pixels in the dispatch
#define PIXEL_COUNT (DispatchRaysDimensions().x * DispatchRaysDimensions().y)

//────────────────────────── x2 ───────────────────────────────────────────────
float3 load_x2_gi(RWByteAddressBuffer buf, uint pixelIdx, uint objID)
{
    uint addr = P_x2_gi * PIXEL_COUNT + pixelIdx * B_x2_gi;
    float3 xO = asfloat(buf.Load3(addr));
    return ObjectToWorldPos(objID, xO);
}
void store_x2_gi(float3 v, RWByteAddressBuffer buf, uint pixelIdx, uint objID)
{
    uint addr = P_x2_gi * PIXEL_COUNT + pixelIdx * B_x2_gi;
    float3 xO = WorldToObjectPos(objID, v);
    buf.Store3(addr, asuint(xO));
}

//────────────────────────── n2 ───────────────────────────────────────────────
float3 load_n2_gi(RWByteAddressBuffer buf, uint pixelIdx, uint objID)
{
    uint addr = P_n2_gi * PIXEL_COUNT + pixelIdx * B_n2_gi;
    float3 nO = UnpackNormal(buf.Load(addr));
    return ObjectToWorldNrm(objID, nO);
}
void store_n2_gi(float3 v, RWByteAddressBuffer buf, uint pixelIdx, uint objID)
{
    uint addr = P_n2_gi * PIXEL_COUNT + pixelIdx * B_n2_gi;
    float3 nO = WorldToObjectNrm(objID, v);
    buf.Store(addr, PackNormal(nO));
}

//────────────────────────── L2 ───────────────────────────────────────────────
float3 load_L2_gi(RWByteAddressBuffer buf, uint pixelIdx)
{
    uint addr = P_L2_gi * PIXEL_COUNT + pixelIdx * B_L2_gi;
    return UnpackRGB9E5(buf.Load(addr));
}
void store_L2_gi(float3 v, RWByteAddressBuffer buf, uint pixelIdx)
{
    uint addr = P_L2_gi * PIXEL_COUNT + pixelIdx * B_L2_gi;
    buf.Store(addr, PackRGB9E5(v));
}

//────────────────────────── V2 (packed) ─────────────────────────────────────
float3 load_V2_gi(RWByteAddressBuffer buf, uint pixelIdx)
{
    uint addr = P_V2_gi * PIXEL_COUNT + pixelIdx * B_V2_gi;
    return UnpackNormal(buf.Load(addr));           // 4-byte packed normal → float3
}
void store_V2_gi(float3 v, RWByteAddressBuffer buf, uint pixelIdx)
{
    uint addr = P_V2_gi * PIXEL_COUNT + pixelIdx * B_V2_gi;
    buf.Store(addr, PackNormal(v));                // float3 → 4-byte packed
}

//────────────────────────── W ────────────────────────────────────────────────
float load_W_gi(RWByteAddressBuffer buf, uint pixelIdx)
{
    uint addr = P_W_gi * PIXEL_COUNT + pixelIdx * B_W_gi;
    return asfloat(buf.Load(addr));
}
void store_W_gi(float w, RWByteAddressBuffer buf, uint pixelIdx)
{
    uint addr = P_W_gi * PIXEL_COUNT + pixelIdx * B_W_gi;
    buf.Store(addr, asuint(w));
}

//────────────────────────── M ────────────────────────────────────────────────
uint load_M_gi(RWByteAddressBuffer buf, uint pixelIdx)
{
    uint addr = P_M_gi * PIXEL_COUNT + pixelIdx * B_M_gi;
    return asuint(buf.Load(addr));
}
void store_M_gi(uint m, RWByteAddressBuffer buf, uint pixelIdx)
{
    uint addr = P_M_gi * PIXEL_COUNT + pixelIdx * B_M_gi;
    buf.Store(addr, m);
}

//──────────────────── load the complete GI reservoir ─────────────────────────
Reservoir_GI loadReservoirGI(RWByteAddressBuffer buf, uint pixelIdx)
{
    Reservoir_GI r;
    //r.x2_gi   = load_x2_gi(buf, pixelIdx);
    //r.n2_gi   = load_n2_gi(buf, pixelIdx);
    r.L2_gi   = load_L2_gi(buf, pixelIdx);
    r.V2_gi   = load_V2_gi(buf, pixelIdx);
    r.W_gi    = load_W_gi (buf, pixelIdx);
    r.M_gi    = load_M_gi (buf, pixelIdx);
    r.w_sum_gi = 0.0f;                 // always re-initialised on GPU
    return r;
}
