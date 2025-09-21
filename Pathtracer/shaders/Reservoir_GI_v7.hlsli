// RIS reservoir for direct lighting
struct Reservoir_GI
{
    float3 x2_gi;
    float3 n2_gi;
    float  W_gi;
    float  w_sum_gi;
    float3 L2_gi;
    float3 V2_gi;

    // NEW
    float3 F_gi;          // contribution cached -> faster
    float4 J_gi;          // Jacobian (.x xk pdf, .y xk+1, .z NEE pdf if NEE, .w geometry term for k>1)
    uint   lobe0_gi;      // Lobes for k and k+1 to ensure bijectivity of shifts
    uint   lobe1_gi;

    uint   M_gi;
    uint   objID_gi;
    uint   matID_gi;
    uint   rSeed_gi;      // Random seed
    uint   rIndex_gi;     // Index of the reconnection vertex
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
    float4 J, // Jacobian of the path (basically just the sequential pdf of replayed path)
    uint rIndex, // Index of the reconnection vertex
    in float3 F,         // cached contribution for the selected sample
    in uint   lobe0,     // lobe id for k
    in uint   lobe1,     // lobe id for k+1
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
        reservoir.F_gi       = F;
        reservoir.lobe0_gi   = lobe0;
        reservoir.lobe1_gi   = lobe1;
        return true;
    }
    return false;
}

/*─────────────────────────────────────────────────────────────────────────────
   64-BYTE LAYOUT (per-pixel · GI reservoir, 4x Load4/Store4)

        0–15   pack1 : { x2.xyz , n2_enc }                     float4  (unchanged)
       16–31   pack2 : { L2_enc , V2_enc , objID , matID|M }   float4  (unchanged)
       32–47   pack3 : { J.x    , J.y    , J.z    , J.w }      float4  (NEW)
       48–63   pack4 : { W      , rSeed  , rIndex|lobes16 , F_enc }   (NEW)
                  - pack4.z low 16 bits : rIndex
                  - pack4.z high16 bits : PackLobes16(lobe0,lobe1)
─────────────────────────────────────────────────────────────────────────────*/
static const uint BYTES_GI    = 64u;

static const uint O_GI_PACK1  =  0u;   // float4
static const uint O_GI_PACK2  = 16u;   // float4
static const uint O_GI_PACK3  = 32u;   // float4  (J_gi)
static const uint O_GI_PACK4  = 48u;   // float4  (W, rSeed, rIndex|lobes, F_enc)

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

    // pack3 : J.x | J.y | J.z | J.w
    buf.Store4(base + O_GI_PACK3, asuint(r.J_gi));

    // pack4 : W | rSeed | (rIndex_low16 | lobes16<<16) | F_enc
    uint lobes16 = PackLobes16(r.lobe0_gi, r.lobe1_gi);
    uint rIndex16 = r.rIndex_gi & 0xFFFFu;
    uint rIndexAndLobes = rIndex16 | (lobes16 << 16);

    buf.Store4(base + O_GI_PACK4, uint4(asuint(r.W_gi),
                                        r.rSeed_gi,
                                        rIndexAndLobes,
                                        PackRGB9E5(r.F_gi)));
}

Reservoir_GI loadReservoirGI(RWByteAddressBuffer buf, uint pixelIdx)
{
    uint base = pixelBaseAddrGI(pixelIdx);

    uint4 p1 = buf.Load4(base + O_GI_PACK1);
    uint4 p2 = buf.Load4(base + O_GI_PACK2);
    uint4 p3 = buf.Load4(base + O_GI_PACK3);
    uint4 p4 = buf.Load4(base + O_GI_PACK4);

    Reservoir_GI r;
    r.objID_gi = p2.z;
    UnpackMatID_M(p2.w, r.matID_gi, r.M_gi);

    r.x2_gi     = ObjectToWorldPos (r.objID_gi, asfloat(p1.xyz));
    r.n2_gi     = ObjectToWorldNrm(r.objID_gi, UnpackNormal(p1.w));
    r.L2_gi     = UnpackRGB9E5(p2.x);
    r.V2_gi     = UnpackNormal (p2.y);

    r.J_gi      = asfloat(p3);

    r.W_gi      = asfloat(p4.x);
    r.rSeed_gi  = p4.y;

    uint rIndexAndLobes = p4.z;
    uint lobes16 = (rIndexAndLobes >> 16) & 0xFFFFu;
    r.rIndex_gi  =  rIndexAndLobes        & 0x0000FFFFu;
    UnpackLobes16(lobes16, r.lobe0_gi, r.lobe1_gi);

    r.F_gi      = UnpackRGB9E5(p4.w);

    r.w_sum_gi = 0.0f; // transient; not stored
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

float4 load_J_gi (RWByteAddressBuffer b, uint id)
{ return asfloat(b.Load4(pixelBaseAddrGI(id)+O_GI_PACK3)); }

float load_W_gi(RWByteAddressBuffer b, uint id)
{return asfloat(b.Load4(pixelBaseAddrGI(id) + O_GI_PACK4).x); }

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
    F = UnpackRGB9E5(p4.w);
}

