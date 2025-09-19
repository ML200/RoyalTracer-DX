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
    uint objID_gi;
    uint matID_gi;
    uint rSeed_gi; // Random seed
    float J_gi; // Jacobian (pdf product)
    uint rIndex_gi; // Index of the reconnection vertex
};

float JacobianDeterminant( float3 x1_c,
                           float3 x2_c,
                           float3 x1_n,
                           float3 n2_c )
{
    float3  v_c   = x1_c - x2_c;
    float   distc = dot(v_c, v_c);          // ‖v_c‖²
    float   cosc  = abs(dot(normalize(v_c), n2_c));   // |cos φ2r|

    float3  v_n   = x1_n - x2_c;
    float   distn = dot(v_n, v_n);          // ‖v_n‖²
    float   cosn  = abs(dot(normalize(v_n), n2_c));   // |cos φ2q|

    float J = (cosc / cosn) * (distn / distc);

    return !isnan(J)?J:1e10;
}

inline bool RejectNormal_GI(float3 n1, float3 n2, float threshold){
    float similarity = dot(n1, n2);
    return (similarity < threshold);
}

/*inline bool RejectDistance_GI(float3 x1, float3 x2, float3 camPos, float threshold)
{
    float d1 = length(x1 - camPos);
    float d2 = length(x2 - camPos);

    float relativeDifference = abs(d1 - d2) / max(d1, d2);
    return relativeDifference > threshold;
}*/

inline bool RejectDistance_GI(float3 x1, float3 x2, float3 normal, float threshold)
{
    float dist = abs(dot(x2 - x1, normal));
    return dist > threshold;
}

inline bool RejectLength_GI(float3 x2_c, float3 n2_c,
                            float3 x1_c, float3 x1_n,
                            float  threshold)
{
    float J = JacobianDeterminant(x1_c, x2_c, x1_n, n2_c);
    return J <= threshold || J >= 1.0f/threshold;
}

inline bool IsValidReservoir_GI(Reservoir_GI r){
    bool valid =
        any(abs(r.n2_gi) > 0.0f) &&
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



// Calculate reconnection (two–sided)
inline float3 ReconnectGI(
    in float3  x1,
    in float3  n1,
    in float3  o,
    in uint    mID1,
    in uint    mID2,
    in float3  x2,
    in float3  n2,
    in float3  L2,
    in float3  V2)
{
    if (all(L2 < EPSILON))
        return 0;

    // Geometric prep
    float3 dir   = x2 - x1;
    float  dist  = length(dir);
    float3 ndirN = normalize(-dir);     // direction from x1 to x2, negated

    // Terms
    float3 F2 = BSDF_term(mID2, n2, V2, ndirN);

    float3 F1 = BSDF_term(mID1, n1, ndirN, o);
    float   G = G_term(n1, ndirN);

    // Throughput
    float3 r = F1 * F2 * L2 * G;

    if (any(isnan(r)) || all(r < EPSILON))
        r = (float3)EPSILON;

    return r;
}

// Calculate reconnection
inline float3 ReconnectGISingle(
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

    // Throughput
    float3 r = F * L * G;

    if (any(isnan(r)))
        r = 0;

    return r;
}


// Update DI reservoir
bool UpdateReservoirGI(
    inout Reservoir_GI reservoir,
    in float wi,
    in uint M,

    in float3 x2,
    in float3 n2,
    in float3 L2, // No need to update L1, as this is always 0 when the sample is processed here. Also,we dont want to reuse sample on a lights surface
    in float3 V2,
    in uint matID,
    in uint objID,
    uint rSeed, // Seed used for replaying
    float J, // Jacobian of the path (basically just the sequential pdf of replayed path)
    uint rIndex, // Index of the reconnection vertex
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
        reservoir.objID_gi = objID;
        reservoir.matID_gi = matID;
        reservoir.rSeed_gi = rSeed;
        reservoir.J_gi = J;
        reservoir.rIndex_gi = rIndex;
        return true;
    }
    return false;
}

// ── per-pixel field sizes (bytes) ──────────────────────────────────────────
/*static const uint B_x2_gi = 12;   // float3  – sample position
static const uint B_n2_gi =  4;   // packed  – normal at x2
static const uint B_L2_gi =  4;   // packed  – incident radiance
static const uint B_V2_gi =  4;   // packed  – V2 direction  (WAS 12)
static const uint B_W_gi  =  4;   // float   – reservoir weight
static const uint B_M_gi  =  4;   // uint    – sample count
static const uint B_objID_gi  =  4;
static const uint B_matID_gi  =  4;

// ── byte offsets of each field block within the buffer ─────────────────────
static const uint P_x2_gi = 0;
static const uint P_n2_gi = P_x2_gi + B_x2_gi;
static const uint P_L2_gi = P_n2_gi + B_n2_gi;
static const uint P_V2_gi = P_L2_gi + B_L2_gi;
static const uint P_W_gi  = P_V2_gi + B_V2_gi;
static const uint P_M_gi  = P_W_gi  + B_W_gi;
static const uint P_objID_gi = P_M_gi     + B_M_gi;
static const uint P_matID_gi = P_objID_gi     + B_objID_gi;

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

//__________________________objID___________________________
uint load_objID_gi(RWByteAddressBuffer buf, uint pixelIdx)
{
    uint addr = P_objID_gi * (DispatchRaysDimensions().x * DispatchRaysDimensions().y)
              + pixelIdx * B_objID_gi;
    return buf.Load(addr);
}

void store_objID_gi(uint objID, RWByteAddressBuffer buf, uint pixelIdx)
{
    uint addr = P_objID_gi * (DispatchRaysDimensions().x * DispatchRaysDimensions().y)
              + pixelIdx * B_objID_gi;
    buf.Store(addr, objID);
}

//__________________________matID___________________________
uint load_matID_gi(RWByteAddressBuffer buf, uint pixelIdx)
{
    uint addr = P_matID_gi * (DispatchRaysDimensions().x * DispatchRaysDimensions().y)
              + pixelIdx * B_matID_gi;
    return buf.Load(addr);
}

void store_matID_gi(uint matID, RWByteAddressBuffer buf, uint pixelIdx)
{
    uint addr = P_matID_gi * (DispatchRaysDimensions().x * DispatchRaysDimensions().y)
              + pixelIdx * B_matID_gi;
    buf.Store(addr, matID);
}

//──────────────────── load the complete GI reservoir ─────────────────────────
Reservoir_GI loadReservoirGI(RWByteAddressBuffer buf, uint pixelIdx)
{
    Reservoir_GI r;
    r.objID_gi  = load_objID_gi(buf, pixelIdx);
    r.matID_gi  = load_matID_gi(buf, pixelIdx);

    r.x2_gi   = load_x2_gi(buf, pixelIdx, r.objID_gi);
    r.n2_gi   = load_n2_gi(buf, pixelIdx, r.objID_gi);
    r.L2_gi   = load_L2_gi(buf, pixelIdx);
    r.V2_gi   = load_V2_gi(buf, pixelIdx);
    r.W_gi    = load_W_gi (buf, pixelIdx);
    r.M_gi    = load_M_gi (buf, pixelIdx);
    r.w_sum_gi = 0.0f;                 // always re-initialised on GPU
    return r;
}*/

/*─────────────────────────────────────────────────────────────────────────────
   COMPACT 32-BYTE LAYOUT  (per-pixel · GI reservoir)

        0–15   pack1  : { x2.xyz , n2_enc }                     float4
       16–31   pack2  : { L2_enc , V2_enc , objID , matID|M }   float4
       32–47   pack3  : { W_gi   , rSeed_gi , rIndex_gi , J_gi } float4
─────────────────────────────────────────────────────────────────────────────*/
static const uint BYTES_GI    = 48u;

static const uint O_GI_PACK1  = 0u;     // float4
static const uint O_GI_PACK2  = 16u;    // float4
static const uint O_GI_PACK3  = 32u;    // float


/* byte address of this pixel’s record */
uint pixelBaseAddrGI(uint pixelIdx) { return pixelIdx * BYTES_GI; }

/* 16-bit packing for matID ‖ M */
uint  PackMatID_M(uint matID, uint M) { return (matID & 0xFFFFu) | (M << 16); }
void  UnpackMatID_M(uint v, out uint matID, out uint M)
{ matID = v & 0xFFFFu;  M = v >> 16; }

/*─────────────────────────────────────────────────────────────────────────────
   FAST WHOLE-RESERVOIR STORE / LOAD
─────────────────────────────────────────────────────────────────────────────*/
void storeReservoirGI(RWByteAddressBuffer buf, uint pixelIdx, const Reservoir_GI r)
{
    uint base = pixelBaseAddrGI(pixelIdx);

    // pack1 : x2.xyz | n2_enc
    float3 xO = WorldToObjectPos (r.objID_gi, r.x2_gi);
    float3 nO = WorldToObjectNrm(r.objID_gi, r.n2_gi);
    buf.Store4(base + O_GI_PACK1, uint4(asuint(xO), PackNormal(nO)));

    // pack2 : L2_enc | V2_enc | objID | matID‖M
    buf.Store4(base + O_GI_PACK2, uint4(PackRGB9E5(r.L2_gi),
                                        PackNormal (r.V2_gi),
                                        r.objID_gi,
                                        PackMatID_M(r.matID_gi, r.M_gi)));

    // pack3 : W_gi | rSeed_gi | rIndex_gi | J_gi
    buf.Store4(base + O_GI_PACK3, uint4(asuint(r.W_gi),
                                        r.rSeed_gi,
                                        r.rIndex_gi,
                                        asuint(r.J_gi)));
}

Reservoir_GI loadReservoirGI(RWByteAddressBuffer buf, uint pixelIdx)
{
    uint base = pixelBaseAddrGI(pixelIdx);

    uint4 p1 = buf.Load4(base + O_GI_PACK1);
    uint4 p2 = buf.Load4(base + O_GI_PACK2);
    uint4 p3 = buf.Load4(base + O_GI_PACK3);

    Reservoir_GI r;
    r.objID_gi = p2.z;
    UnpackMatID_M(p2.w, r.matID_gi, r.M_gi);

    r.x2_gi     = ObjectToWorldPos (r.objID_gi, asfloat(p1.xyz));
    r.n2_gi     = ObjectToWorldNrm(r.objID_gi, UnpackNormal(p1.w));
    r.L2_gi     = UnpackRGB9E5(p2.x);
    r.V2_gi     = UnpackNormal (p2.y);
    r.W_gi      = asfloat(p3.x);
    r.rSeed_gi  = p3.y;
    r.rIndex_gi = p3.z;
    r.J_gi      = asfloat(p3.w);

    r.w_sum_gi = 0.0f; // transient
    return r;
}


/*─────────────────────────────────────────────────────────────────────────────
   FINE-GRAINED LOAD HELPers  (one read each)
─────────────────────────────────────────────────────────────────────────────*/
float3 load_x2_gi(RWByteAddressBuffer b, uint id, uint obj)
{ return ObjectToWorldPos(obj, asfloat(b.Load4(pixelBaseAddrGI(id)+O_GI_PACK1).xyz)); }

float3 load_n2_gi(RWByteAddressBuffer b, uint id, uint obj)
{ return ObjectToWorldNrm(obj, UnpackNormal(b.Load4(pixelBaseAddrGI(id)+O_GI_PACK1).w)); }

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

float load_W_gi (RWByteAddressBuffer b, uint id)
{ return asfloat(b.Load4(pixelBaseAddrGI(id)+O_GI_PACK3).x); }

uint  load_rSeed_gi (RWByteAddressBuffer b, uint id)
{ return b.Load4(pixelBaseAddrGI(id)+O_GI_PACK3).y; }

uint  load_rIndex_gi (RWByteAddressBuffer b, uint id)
{ return b.Load4(pixelBaseAddrGI(id)+O_GI_PACK3).z; }

float load_J_gi (RWByteAddressBuffer b, uint id)
{ return asfloat( b.Load4(pixelBaseAddrGI(id)+O_GI_PACK3).w); }

/*─────────────────────────────────────────────────────────────────────────────
   FAST GROUP LOADERS  (one read when you need both values)
─────────────────────────────────────────────────────────────────────────────*/
inline void load_x2_n2_fast_gi(RWByteAddressBuffer b, uint id, uint obj,
                               out float3 x2, out float3 n2)
{
    uint4 p1 = b.Load4(pixelBaseAddrGI(id)+O_GI_PACK1);
    x2 = ObjectToWorldPos (obj, asfloat(p1.xyz));
    n2 = ObjectToWorldNrm(obj, UnpackNormal(p1.w));
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

