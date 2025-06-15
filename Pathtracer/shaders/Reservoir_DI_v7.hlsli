// RIS reservoir for direct lighting
struct Reservoir_DI
{
    float3 x2_di;
    float3 n2_di;
    float W_di;
    float w_sum_di;
    float3 L2_di;
    uint M_di;
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
        any(r.n2_di > 0.0f) &&
        any(r.L2_di > 0.0f) &&
        r.W_di > 0.0f &&
        r.M_di > 0.0f;
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
    ray.TMin = 0.001f;
    ray.TMax = max(dist - 3.0f * EPSILON, 2.0f * EPSILON);
    ShadowHitInfo shadowPayload;
    shadowPayload.isHit = false;
    TraceRay(SceneBVH, RAY_FLAG_NONE, 0xFF, 1, 0, 1, ray, shadowPayload);
    V = shadowPayload.isHit ? 0.0f : 1.0f;
    return V;
}



// Calculate reconnection
float3 ReconnectDI(
    float3 x1,
    float3 n1,
    float3 o,
    uint mID,
    float3 x2,
    float3 n2,
    float3 L
)
{
    if(all(L<EPSILON))
        return float3(0,0,0);
    float3 dir = x2 - x1;
    float3 ndirN = normalize(-dir);
    float dist = length(dir);

    float cosThetaX1 = max(EPSILON,dot(n1, -ndirN));
    if(dot(n2, ndirN) < 0.0f)
        n2 = -n2;
    float cosThetaX2 = max(EPSILON,dot(n2, ndirN));

    float2 probs = CalculateStrategyProbabilities(mID, o, n1);
    float3 brdf0 = EvaluateBRDF(0, mID, n1, ndirN, o);
    float3 brdf1 = EvaluateBRDF(1, mID, n1, ndirN, o);
    float3 F1 = SafeMultiply(probs.x, brdf0);
    float3 F2 = SafeMultiply(probs.y, brdf1);
    float3 F = F1 + F2;

    float3 r = F * L * cosThetaX1 * cosThetaX2 / (dist * dist);
    if(any(isnan(r)))
        r = float3(0,0,0);
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

// Offset constants
static const uint P_x2   = 0;
static const uint P_n2   = P_x2    + B_x2;
static const uint P_L2   = P_n2   + B_n2;
static const uint P_W    = P_L2    + B_L2;
static const uint P_M  = P_W     + B_W;

//__________________________x2_____________________________
float3 load_x2_di(RWByteAddressBuffer buffer, uint pixelIdx){
    uint addr = P_x2 * (DispatchRaysDimensions().x * DispatchRaysDimensions().y) + pixelIdx * B_x2;
    return asfloat(buffer.Load3(addr));
}
void store_x2_di(float3 x2, RWByteAddressBuffer buffer, uint pixelIdx){
    uint addr = P_x2 * (DispatchRaysDimensions().x * DispatchRaysDimensions().y) + pixelIdx * B_x2;
    buffer.Store3(addr, asuint(x2));
}

//__________________________n2_____________________________
float3 load_n2_di(RWByteAddressBuffer buffer, uint pixelIdx){
    uint addr = P_n2 * (DispatchRaysDimensions().x * DispatchRaysDimensions().y) + pixelIdx * B_n2;
    return UnpackNormal(buffer.Load(addr));
}
void store_n2_di(float3 n2, RWByteAddressBuffer buffer, uint pixelIdx){
    uint addr = P_n2 * (DispatchRaysDimensions().x * DispatchRaysDimensions().y) + pixelIdx * B_n2;
    buffer.Store(addr, PackNormal(n2));
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


Reservoir_DI loadReservoirDI(RWByteAddressBuffer buffer, uint pixelIdx)
{
    Reservoir_DI r;

    // packed attributes
    r.x2_di  = load_x2_di(buffer, pixelIdx);
    r.n2_di  = load_n2_di(buffer, pixelIdx);
    r.L2_di  = load_L2_di(buffer, pixelIdx);

    // scalar data
    r.W_di       = load_W_di(buffer, pixelIdx);
    r.M_di       = load_M_di(buffer, pixelIdx);

    r.w_sum_di = 0.0f;

    return r;
}
