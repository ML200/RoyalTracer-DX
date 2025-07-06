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
inline bool RejectDistance_DI(float3 x1, float3 x2, float3 camPos, float threshold)
{
    float d1 = length(x1 - camPos);
    float d2 = length(x2 - camPos);

    float relativeDifference = abs(d1 - d2) / max(d1, d2);
    return relativeDifference > threshold;
}

inline bool IsValidReservoir_DI(Reservoir_DI r){
    bool valid =
        any(abs(r.n2_di) > 0.0f) &&
        any(r.L2_di > 0.0f) &&
        r.W_di > 0.0f &&
        r.M_di > 0.0f
        ;
    return valid;
}

// The remaining functions remain unchanged.
float VisibilityCheck(
    float3 x1,
    float3 x2,
    float3 n1
)
{
    float V = 0.0f;
    float3 dir = x2-x1;
    float dist = length(dir);
    RayDesc ray;
    ray.Origin = x1 + normalize(n1) * EPSILON;
    ray.Direction = normalize(dir);
    ray.TMin = EPSILON;
    ray.TMax = max(dist - 10.0f * EPSILON, 2.0f * EPSILON);
    ShadowHitInfo shadowPayload;
    shadowPayload.isHit = false;
    const uint flags =
        RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH;
    TraceRay(SceneBVH, flags, 0xFF, 1, 0, 1, ray, shadowPayload);
    V = shadowPayload.isHit ? 0.0f : 1.0f;
    return V;
}

#ifdef ENABLE_RAY_QUERY_INLINE
[forceinline]
float VisibilityCheckCP(float3 P, float3 L, float3 N)
{
    float3 dir = normalize(L - P);
    float  len = length(L - P);

    RayDesc ray;
    ray.Origin    = P + normalize(N) * EPSILON;            // ← offset *along ray*
    ray.Direction = dir;
    ray.TMin      = EPSILON;
    ray.TMax      = max(len - EPSILON*10, 2*EPSILON);

    RayQuery< RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH
              //|RAY_FLAG_CULL_BACK_FACING_TRIANGLES
              > rq;

    rq.TraceRayInline(SceneBVH,        // TLAS
                      RAY_FLAG_NONE,   // dynamic flags
                      0xFF,            // mask
                      ray);

    rq.Proceed();            // ← DRIVE TO COMPLETION

    return (rq.CommittedStatus() == COMMITTED_TRIANGLE_HIT) ? 0.0 : 1.0;
}
#endif // ENABLE_RAY_QUERY_INLINE


float3 BSDF_term(
    uint   mID,
    float3 n1,
    float3 ndirN,   // -wi (x1←x2) normalised
    float3 o)       // outgoing/view dir at x1
{
    float2  p      = CalculateStrategyProbabilities(mID, o, n1);
    float3  f0     = EvaluateBRDF(0, mID, n1, ndirN, o);
    float3  f1     = EvaluateBRDF(1, mID, n1, ndirN, o);

    return SafeMultiply(p.x, f0) +
           SafeMultiply(p.y, f1);          //  F
}

float G_term(float3 n1, float3 ndirN)                // ndirN = -wi
{
    return max(EPSILON, dot(n1, -ndirN));            // cos θ₁
}

float J_term(
    inout float3 n2,
    float3 ndirN,
    float  dist)        // |x₂−x₁|
{
    if (dot(n2, ndirN) < 0.0f)
        n2 = -n2;

    float cosThetaX2 = max(EPSILON, dot(n2, ndirN));
    return cosThetaX2 / max(EPSILON, dist * dist);   // cos θ₂ / d²
}

// Calculate reconnection
inline float3 ReconnectDI(
    float3 x1,
    float3 n1,
    float3 o,
    uint   mID,
    float3 x2,
    float3 n2,
    float3 L)
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

//___ Packing and saving ___
// Size constants
static const uint B_x2   = 12;  // float3
static const uint B_n2   =  4; // packed float3
static const uint B_L2   = 4;  // packed float3
static const uint B_W  =  4;
static const uint B_M  =  4;
static const uint B_objID  =  4;

// Offset constants
static const uint P_x2   = 0;
static const uint P_n2   = P_x2    + B_x2;
static const uint P_L2   = P_n2   + B_n2;
static const uint P_W    = P_L2    + B_L2;
static const uint P_M  = P_W     + B_W;
static const uint P_objID = P_M     + B_M;

//__________________________x2_____________________________
float3 load_x2_di(RWByteAddressBuffer buffer, uint pixelIdx, uint objID){
    uint addr = P_x2 * (DispatchRaysDimensions().x * DispatchRaysDimensions().y) + pixelIdx * B_x2;
    float3 xO = asfloat(buffer.Load3(addr));
    return ObjectToWorldPos(objID, xO);
}
void store_x2_di(float3 x2, RWByteAddressBuffer buffer, uint pixelIdx, uint objID){
    uint addr = P_x2 * (DispatchRaysDimensions().x * DispatchRaysDimensions().y) + pixelIdx * B_x2;
    float3 xO = WorldToObjectPos(objID, x2);
    buffer.Store3(addr, asuint(xO));
}

//__________________________n2_____________________________
float3 load_n2_di(RWByteAddressBuffer buffer, uint pixelIdx, uint objID){
    uint addr = P_n2 * (DispatchRaysDimensions().x * DispatchRaysDimensions().y) + pixelIdx * B_n2;
    float3 nO = UnpackNormal(buffer.Load(addr));
    return ObjectToWorldNrm(objID, nO);
}
void store_n2_di(float3 n2, RWByteAddressBuffer buffer, uint pixelIdx, uint objID){
    uint addr = P_n2 * (DispatchRaysDimensions().x * DispatchRaysDimensions().y) + pixelIdx * B_n2;
    float3 nO = WorldToObjectNrm(objID, n2);
    buffer.Store(addr, PackNormal(nO));
}

//__________________________L2_____________________________
float3 load_L2_di(RWByteAddressBuffer buffer, uint pixelIdx){
    uint addr = P_L2 * (DispatchRaysDimensions().x * DispatchRaysDimensions().y) + pixelIdx * B_L2;
    return UnpackRGB9E5(buffer.Load(addr));
}
void store_L2_di(float3 L2, RWByteAddressBuffer buffer, uint pixelIdx){
    uint addr = P_L2 * (DispatchRaysDimensions().x * DispatchRaysDimensions().y) + pixelIdx * B_L2;
    buffer.Store(addr, PackRGB9E5(L2));
}

//__________________________W_____________________________
float load_W_di(RWByteAddressBuffer buffer, uint pixelIdx){
    uint addr = P_W * (DispatchRaysDimensions().x * DispatchRaysDimensions().y) + pixelIdx * B_W;
    return asfloat(buffer.Load(addr));
}
void store_W_di(float W, RWByteAddressBuffer buffer, uint pixelIdx){
    uint addr = P_W * (DispatchRaysDimensions().x * DispatchRaysDimensions().y) + pixelIdx * B_W;
    buffer.Store(addr, W);
}

//__________________________M_____________________________
uint load_M_di(RWByteAddressBuffer buffer, uint pixelIdx){
    uint addr = P_M * (DispatchRaysDimensions().x * DispatchRaysDimensions().y) + pixelIdx * B_M;
    return asuint(buffer.Load(addr));
}
void store_M_di(uint M, RWByteAddressBuffer buffer, uint pixelIdx){
    uint addr = P_M * (DispatchRaysDimensions().x * DispatchRaysDimensions().y) + pixelIdx * B_M;
    buffer.Store(addr, M);
}

//__________________________objID___________________________
uint load_objID_di(RWByteAddressBuffer buf, uint pixelIdx)
{
    uint addr = P_objID * (DispatchRaysDimensions().x * DispatchRaysDimensions().y)
              + pixelIdx * B_objID;
    return buf.Load(addr);
}

void store_objID_di(uint objID, RWByteAddressBuffer buf, uint pixelIdx)
{
    uint addr = P_objID * (DispatchRaysDimensions().x * DispatchRaysDimensions().y)
              + pixelIdx * B_objID;
    buf.Store(addr, objID);
}



Reservoir_DI loadReservoirDI(RWByteAddressBuffer buffer, uint pixelIdx)
{
    Reservoir_DI r;
    r.objID_di  = load_objID_di(buffer, pixelIdx);

    // packed attributes
    r.x2_di  = load_x2_di(buffer, pixelIdx, r.objID_di);
    r.n2_di  = load_n2_di(buffer, pixelIdx, r.objID_di);
    r.L2_di  = load_L2_di(buffer, pixelIdx);

    // scalar data
    r.W_di       = load_W_di(buffer, pixelIdx);
    r.M_di       = load_M_di(buffer, pixelIdx);

    r.w_sum_di = 0.0f;

    return r;
}
