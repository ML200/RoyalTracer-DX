// RIS reservoir — unified DI + GI.
// matID acts as the path-kind discriminator:
//   matID < MATID_LIGHT_TRI : BSDF-sampled vertex at x2 with a real
//                              material (classic GI path, d >= 3).
//   matID == MATID_LIGHT_TRI: emissive-triangle NEE (d = 2). x2 is the hit
//                              position on the light; n2_s is the light's
//                              surface normal; L2 is the emission.
//   matID == MATID_ENV_MISS : environment/sky sample (d = 2). x2 is a unit
//                              DIRECTION (not a position); L2 is the radiance
//                              from that direction.
// See Constants_v8.hlsli for the sentinel values.
struct Reservoir
{
    // Constant-after-hit payload
    float3 x2;
    float3 n2_s;
    uint   objID;
    uint   matID;      // discriminates path kind (see header comment)
    float2 uv;
    float  eta;        // transmittance IOR at x2 (stored at path creation)

    // Varying payload
    float3 L2;
    float3 V2;
    uint   F;          // RGB9E5-packed normalized color (F / Luma(F))
    float  F_mag;      // scalar magnitude = Luma(F) = GetPHat(F)

    float  W;
    uint   M;
    float  w_sum;      // raygen-only (merge passes overwrite before use)
};


static const uint BYTES       = 56u;
static const uint BYTES_WSUM  =  4u;
static const uint BYTES_VPOST =  4u;
static const uint BYTES_TPOST =  4u;
static const uint STRIDE      = BYTES + BYTES_WSUM + BYTES_VPOST + BYTES_TPOST; // 68

//Per-field sizes
static const uint SZ_PACK1 = 16u;  // x2(12) + n2_s_packed(4)
static const uint SZ_L2    =  4u;
static const uint SZ_V2    =  4u;
static const uint SZ_OBJID =  4u;
static const uint SZ_UV    =  4u;
static const uint SZ_MATID =  4u;
static const uint SZ_W     =  4u;
static const uint SZ_F     =  4u;
static const uint SZ_FMAG  =  4u;
static const uint SZ_M     =  4u;
static const uint SZ_ETA   =  4u;
static const uint SZ_WSUM  =  4u;
static const uint SZ_VPOST =  4u;
static const uint SZ_TPOST =  4u;

//Plane cumulative offsets
static const uint PLANE_PACK1 =  0u;
static const uint PLANE_L2    = 16u;
static const uint PLANE_V2    = 20u;
static const uint PLANE_OBJID = 24u;
static const uint PLANE_UV    = 28u;
static const uint PLANE_MATID = 32u;
static const uint PLANE_W     = 36u;
static const uint PLANE_F     = 40u;
static const uint PLANE_FMAG  = 44u;
static const uint PLANE_M     = 48u;
static const uint PLANE_ETA   = 52u;
static const uint PLANE_WSUM  = 56u;
static const uint PLANE_VPOST = 60u;
static const uint PLANE_TPOST = 64u;

// SoA address helpers
// Tile-aligned pixel count, must match MapPixelID's 4x8 tile swizzle.
uint numPx()                       { return ((IMG_W + 3u) / 4u) * ((IMG_H + 7u) / 8u) * 32u; }
uint addr_pack1(uint px)           { return px * SZ_PACK1; }
uint addr_l2(uint px)              { uint N = numPx(); return N * PLANE_L2    + px * SZ_L2; }
uint addr_v2(uint px)              { uint N = numPx(); return N * PLANE_V2    + px * SZ_V2; }
uint addr_objid(uint px)           { uint N = numPx(); return N * PLANE_OBJID + px * SZ_OBJID; }
uint addr_uv(uint px)              { uint N = numPx(); return N * PLANE_UV    + px * SZ_UV; }
uint addr_matid(uint px)           { uint N = numPx(); return N * PLANE_MATID + px * SZ_MATID; }
uint addr_w(uint px)               { uint N = numPx(); return N * PLANE_W     + px * SZ_W; }
uint addr_f(uint px)               { uint N = numPx(); return N * PLANE_F     + px * SZ_F; }
uint addr_fmag(uint px)            { uint N = numPx(); return N * PLANE_FMAG  + px * SZ_FMAG; }
uint addr_m(uint px)               { uint N = numPx(); return N * PLANE_M     + px * SZ_M; }
uint addr_eta(uint px)             { uint N = numPx(); return N * PLANE_ETA   + px * SZ_ETA; }
uint addr_wsum(uint px)            { uint N = numPx(); return N * PLANE_WSUM  + px * SZ_WSUM; }
uint addr_vpost(uint px)           { uint N = numPx(); return N * PLANE_VPOST + px * SZ_VPOST; }
uint addr_tpost(uint px)           { uint N = numPx(); return N * PLANE_TPOST + px * SZ_TPOST; }

// Scalar magnitude used throughout (luminance).
inline float GetPHat(float3 v) {
    return 0.2126f * v.x + 0.7152f * v.y + 0.0722f * v.z;
}

// BRDF wrappers (kept as thin aliases to isolate MIS callers from the
// underlying BXDF module).
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

// Geometry term uses the shading normal (geometric normal has been retired).
float G_term(float3 n, float3 s)
{
    return abs(dot(n, s));
}

// Store/load packed V_post (world-space)
void store_Vpost(RWByteAddressBuffer b, uint pixelIdx, float3 Vpost_world)
{
    b.Store(addr_vpost(pixelIdx), PackNormal(normalize(Vpost_world)));
}

float3 load_Vpost(RWByteAddressBuffer b, uint pixelIdx)
{
    return UnpackNormal(b.Load(addr_vpost(pixelIdx)));
}

void store_Tpost(RWByteAddressBuffer b, uint pixelIdx, float3 Tpost)
{
    b.Store(addr_tpost(pixelIdx), PackRGB9E5(max(Tpost, 0.0f)));
}

float3 load_Tpost(RWByteAddressBuffer b, uint pixelIdx)
{
    return UnpackRGB9E5(b.Load(addr_tpost(pixelIdx)));
}


void storeReservoir(RWByteAddressBuffer buf, uint pixelIdx, const Reservoir r)
{
    float3 xO  = WorldToObjectPos(r.objID, r.x2);
    float3 nSO = WorldToObjectNrm(r.objID, r.n2_s);

    buf.Store4(addr_pack1(pixelIdx), uint4(asuint(xO), PackNormal(normalize(nSO))));
    buf.Store (addr_l2(pixelIdx),    PackRGB9E5(r.L2));
    buf.Store (addr_v2(pixelIdx),    PackNormal(normalize(r.V2)));
    buf.Store (addr_objid(pixelIdx), r.objID);
    buf.Store (addr_uv(pixelIdx),    PackFloat2x16(r.uv.x, r.uv.y));
    buf.Store (addr_matid(pixelIdx), r.matID);
    buf.Store (addr_w(pixelIdx),     asuint(r.W));
    buf.Store (addr_f(pixelIdx),     r.F);
    buf.Store (addr_fmag(pixelIdx),  asuint(r.F_mag));
    buf.Store (addr_m(pixelIdx),     r.M);
    buf.Store (addr_eta(pixelIdx),   asuint(r.eta));
}

Reservoir loadReservoir(RWByteAddressBuffer buf, uint pixelIdx)
{
    Reservoir r;

    uint4 p1 = buf.Load4(addr_pack1(pixelIdx));

    r.objID = buf.Load(addr_objid(pixelIdx));
    r.matID = buf.Load(addr_matid(pixelIdx));

    r.x2    = ObjectToWorldPos(r.objID, asfloat(p1.xyz));
    r.n2_s  = ObjectToWorldNrm(r.objID, UnpackNormal(p1.w));

    r.L2    = UnpackRGB9E5(buf.Load(addr_l2(pixelIdx)));
    r.V2    = UnpackNormal(buf.Load(addr_v2(pixelIdx)));

    uint uv_packed = buf.Load(addr_uv(pixelIdx));
    UnpackFloat2x16(uv_packed, r.uv.x, r.uv.y);

    r.W     = asfloat(buf.Load(addr_w(pixelIdx)));
    r.F     = buf.Load(addr_f(pixelIdx));
    r.F_mag = asfloat(buf.Load(addr_fmag(pixelIdx)));

    r.M     = buf.Load(addr_m(pixelIdx));
    r.eta   = asfloat(buf.Load(addr_eta(pixelIdx)));
    r.w_sum = 0.0f; // raygen-only; merge passes overwrite before use

    return r;
}


uint load_objID(RWByteAddressBuffer b, uint pixelIdx)
{
    return b.Load(addr_objid(pixelIdx));
}

uint load_matID(RWByteAddressBuffer b, uint pixelIdx)
{
    return b.Load(addr_matid(pixelIdx));
}

float3 load_x2(RWByteAddressBuffer b, uint pixelIdx, uint objID)
{
    float3 xO = asfloat(b.Load4(addr_pack1(pixelIdx)).xyz);
    return ObjectToWorldPos(objID, xO);
}

float3 load_n2_s(RWByteAddressBuffer b, uint pixelIdx, uint objID)
{
    uint enc = b.Load4(addr_pack1(pixelIdx)).w;
    return ObjectToWorldNrm(objID, UnpackNormal(enc));
}

float3 load_L2(RWByteAddressBuffer b, uint pixelIdx)
{
    return UnpackRGB9E5(b.Load(addr_l2(pixelIdx)));
}

float3 load_V2(RWByteAddressBuffer b, uint pixelIdx)
{
    return UnpackNormal(b.Load(addr_v2(pixelIdx)));
}

// Distinct from Sample_Data's load_uv (which reads the sample G-buffer).
float2 load_uv_res(RWByteAddressBuffer b, uint pixelIdx)
{
    float2 r;
    UnpackFloat2x16(b.Load(addr_uv(pixelIdx)), r.x, r.y);
    return r;
}

float load_W(RWByteAddressBuffer b, uint pixelIdx)
{
    return asfloat(b.Load(addr_w(pixelIdx)));
}

uint load_F(RWByteAddressBuffer b, uint pixelIdx)
{
    return b.Load(addr_f(pixelIdx));
}

float load_F_mag(RWByteAddressBuffer b, uint pixelIdx)
{
    return asfloat(b.Load(addr_fmag(pixelIdx)));
}

// Reconstruct full F = normalized_color * magnitude
float3 load_F_full(RWByteAddressBuffer b, uint pixelIdx)
{
    return UnpackRGB9E5(b.Load(addr_f(pixelIdx))) * asfloat(b.Load(addr_fmag(pixelIdx)));
}

float load_eta(RWByteAddressBuffer b, uint pixelIdx)
{
    return asfloat(b.Load(addr_eta(pixelIdx)));
}


float load_wsum(RWByteAddressBuffer b, uint pixelIdx)
{
    return asfloat(b.Load(addr_wsum(pixelIdx)));
}

void store_wsum(RWByteAddressBuffer b, uint pixelIdx, float wsum)
{
    b.Store(addr_wsum(pixelIdx), asuint(wsum));
}

uint load_M(RWByteAddressBuffer b, uint pixelIdx)
{
    return b.Load(addr_m(pixelIdx));
}

void store_M(RWByteAddressBuffer b, uint pixelIdx, uint M)
{
    b.Store(addr_m(pixelIdx), M);
}

void store_W(RWByteAddressBuffer b, uint pixelIdx, float W)
{
    b.Store(addr_w(pixelIdx), asuint(W));
}

void store_F(RWByteAddressBuffer b, uint pixelIdx, uint F)
{
    b.Store(addr_f(pixelIdx), F);
}

void store_F_mag(RWByteAddressBuffer b, uint pixelIdx, float mag)
{
    b.Store(addr_fmag(pixelIdx), asuint(mag));
}

// Combined store: split float3 F into normalized RGB9E5 color + float magnitude
void store_F_combined(RWByteAddressBuffer b, uint pixelIdx, float3 F)
{
    if (any(isnan(F)) || any(isinf(F)))
        F = float3(0, 0, 0);
    float mag = GetPHat(F);
    float3 norm = (mag > 1e-20f) ? F / mag : float3(0, 0, 0);
    b.Store(addr_f(pixelIdx), PackRGB9E5(norm));
    b.Store(addr_fmag(pixelIdx), asuint(mag));
}


inline bool RejectNormal(float3 n1, float3 n2, float threshold) {
    return dot(n1, n2) < threshold;
}

inline bool RejectDistance(float3 x1, float3 x2, float3 normal, float threshold)
{
    return abs(dot(x2 - x1, normal)) > threshold;
}

inline bool IsValidReservoir(Reservoir r) {
    return any(abs(r.n2_s) > 0.0f) && r.M > 0;
}

inline bool IsValidReservoir_opt(in float3 n2, in uint M) {
    return any(abs(n2) > 0.0f) && M > 0.0f;
}

inline void InvalidateReservoir_ShadingNormal(
    RWByteAddressBuffer buf,
    uint pixelIdx
)
{
    // n2_s is stored in PACK1.w
    buf.Store(addr_pack1(pixelIdx) + 12u, 0u);
}



// Geometric jacobian: recomputable from positions + shading normal
inline float ComputeJc(float3 x1, float3 x2, float3 n2_s)
{
    float3 d = x1 - x2;
    float  dist2 = dot(d, d);
    if (dist2 < EPSILON) return EPSILON;
    float  dist = sqrt(dist2);
    return max(abs(dot(d / dist, n2_s)) / dist2, EPSILON);
}

// Safe jacobian ratio
inline float JacobianRatio(float Jn, float Jc)
{
    return (Jc > EPSILON) ? (Jn / Jc) : 0.0f;
}

// Calculate reconnection
inline float3 Reconnect(
    // Vertex x1 (camera path hit)
    in float3  x1,
    in float3  n1_s,
    in float3  o,
    in uint    mID1,
    in float3  localKd1,
    in float   localPr1,
    in float   localPm1,

    // Vertex x2 (reservoir / reconnection vertex)
    in uint    mID2,
    in float3  x2,
    in float3  n2_s,
    in float3  L2,
    in float3  V2,
    in float3  localKd2,
    in float   localPr2,
    in float   localPm2,
    in float   eta2, // stored transmittance IOR at x2

    out float  Jn
)
{
    Jn = 1.0f;

    if (length(L2) < EPSILON)
        return 0.0f;

    //x1 IOR: always air / material (primary hit)
    const float etai1 = 1.0f;
    const float etat1 = materials[mID1].Ni;

    //───────────────────────────────────────────────────────────────────────
    // DI: environment / sky sample. x2 stores a DIRECTION. No G term, no
    // BSDF at x2, no medium transmittance. Jn = 1 (direction is preserved
    // under the reconnection shift).
    //───────────────────────────────────────────────────────────────────────
    if (mID2 == MATID_ENV_MISS)
    {
        const float3 wi  = normalize(x2);
        const float3 F1  = BSDF_term(mID1, n1_s, n1_s, wi, o,
                                     localKd1, localPr1, localPm1, etai1, etat1);
        const float  ct  = max(1e-15f, dot(n1_s, wi));
        float3 r = F1 * L2 * ct;
        if (any(isnan(r)) || any(isinf(r))) return 0.0f;
        return max(r, 0.0f);  // Jn already = 1
    }

    //───────────────────────────────────────────────────────────────────────
    // DI: emissive-triangle NEE sample. x2 is a world position on the light,
    // n2_s is the light's surface normal, L2 is emission. No BSDF at x2;
    // Jn uses the same connection-edge geometric factor as the vertex case.
    //───────────────────────────────────────────────────────────────────────
    if (mID2 == MATID_LIGHT_TRI)
    {
        const float3 dirT  = x2 - x1;
        const float  distT = length(dirT);
        if (distT < EPSILON) return 0.0f;
        const float3 ndirNT = normalize(-dirT);

        const float3 F1 = BSDF_term(mID1, n1_s, n1_s, -ndirNT, o,
                                    localKd1, localPr1, localPm1, etai1, etat1);
        const float  G1 = G_term(n1_s, -ndirNT);
        float3 r = F1 * L2 * G1;
        if (any(isnan(r)) || any(isinf(r))) r = 0.0f;

        Jn = max(abs(dot(ndirNT, n2_s)) / (distT * distT), EPSILON);
        return max(r, 0.0f);
    }

    //───────────────────────────────────────────────────────────────────────
    // GI: BSDF-sampled vertex at x2 (original path, d >= 3). Original code.
    //───────────────────────────────────────────────────────────────────────

    // Geometric prep
    float3 dir   = x2 - x1;
    float  dist  = length(dir);
    float3 ndirN = normalize(-dir); // direction from x2 to x1

    //x2 IOR
    float etai2 = 1.0f;
    float etat2 = eta2;

    if(dot(-ndirN, n1_s)<0.0f)
        etai2 = etat1;

    float3 F1 = BSDF_term(mID1, n1_s, n1_s, -ndirN, o,  localKd1, localPr1, localPm1, etai1, etat1);
    float3 F2 = BSDF_term(mID2, n2_s, n2_s, -V2, ndirN, localKd2, localPr2, localPm2, etai2, etat2);

    // Geometry term
    float  G1  = G_term(n1_s, -ndirN);
    float  G2  = G_term(n2_s, -V2);

    //Transmittance: if direction points against the shading normal, we are transmitting INTO the object
    float3 transmittance = float3(1.0f, 1.0f, 1.0f);
    if (dot(n1_s, -ndirN) < 0.0f)
    {
        transmittance = CalculateAbsorptionThroughput(materials[mID1].Tf, dist);
    }

    // Geometric jacobian at the new x1
    Jn = max(abs(dot(ndirN, n2_s)) / (dist * dist), EPSILON);
    // contribution
    float3 r = F1 * F2 * L2 * G1 * G2 * transmittance;

    if (any(isnan(r)) || any(isinf(r)) || all(r < EPSILON))
        r = (float3)0.0f;

    return max(r,0.0f);
}



// Update reservoir
bool UpdateReservoir(
    inout Reservoir reservoir,
    in float wi,
    in uint  M,

    in float3 x2,
    in float3 n2_s,
    in float3 L2,
    in float3 V2,

    in float2 uv,

    in uint matID,
    in uint objID,
    in float eta,

    in uint  F,
    in float F_mag,

    inout uint2 seed
)
{
    reservoir.w_sum += wi;
    reservoir.M     += M;

    if (RandomFloatSingle(seed.x) < (wi / reservoir.w_sum))
    {
        reservoir.x2    = x2;
        reservoir.n2_s  = n2_s;
        reservoir.objID = objID;
        reservoir.matID = matID;
        reservoir.eta   = eta;

        reservoir.uv    = uv;

        reservoir.L2    = L2;
        reservoir.V2    = V2;
        reservoir.F     = F;
        reservoir.F_mag = F_mag;
        return true;
    }
    return false;
}


// Tiny RIS accumulator kept in registers across the raygen path loop.
// Replaces a full local Reservoir (~17 scalars -> 3 scalars of live state).
// The winning candidate's payload (x2, n2, L2, V2, uv, matID, objID, eta)
// is written *directly* to the reservoir buffer in AddInitialCandidate;
// only F_pack / F_mag need to survive in registers for the final W.
struct InitialRisState
{
    float wsum;
    uint  F_pack;
    float F_mag;
};

// Initial-resampling candidate update: accumulates wsum in registers, and
// on RIS acceptance writes the reservoir payload straight to the buffer.
// Sentinel objIDs (env/miss) bypass object-space transform via the
// identity shortcut in Sample_Data_v8.
inline bool AddInitialCandidate(
    inout InitialRisState ris,
    RWByteAddressBuffer buf,
    uint pixelIdx,
    float  wi,
    float3 x2, float3 n2_s,
    float3 L2, float3 V2,
    float2 uv,
    uint   matID, uint objID, float eta,
    float3 F_contrib,
    inout uint seed)
{
    if (wi <= 0.0f || any(isnan(F_contrib)) || any(isinf(F_contrib))) return false;
    const float F_mag_new = GetPHat(F_contrib);
    if (F_mag_new <= 1e-20f) return false;
    const uint F_pack_new = PackRGB9E5(F_contrib / F_mag_new);

    ris.wsum += wi;
    if (ris.wsum > EPSILON && RandomFloatSingle(seed) < (wi / ris.wsum))
    {
        const float3 xO  = WorldToObjectPos(objID, x2);
        const float3 nSO = WorldToObjectNrm(objID, n2_s);
        buf.Store4(addr_pack1(pixelIdx), uint4(asuint(xO), PackNormal(normalize(nSO))));
        buf.Store (addr_l2   (pixelIdx), PackRGB9E5(L2));
        buf.Store (addr_v2   (pixelIdx), PackNormal(normalize(V2)));
        buf.Store (addr_objid(pixelIdx), objID);
        buf.Store (addr_matid(pixelIdx), matID);
        buf.Store (addr_eta  (pixelIdx), asuint(eta));
        buf.Store (addr_uv   (pixelIdx), PackFloat2x16(uv.x, uv.y));

        ris.F_pack = F_pack_new;
        ris.F_mag  = F_mag_new;
        return true;
    }
    return false;
}

// Legacy wrapper — kept for any caller that still expects the local-struct
// form. New raygen path uses AddInitialCandidate above.
inline void AddCandidate(
    inout Reservoir res,
    in float  wi,
    in float3 x2,
    in float3 n2_s,
    in float3 L2,
    in float3 V2,
    in float2 uv,
    in uint   matID,
    in uint   objID,
    in float  eta,
    in float3 F_contrib,
    inout uint seed)
{
    if (wi <= 0.0f || any(isnan(F_contrib)) || any(isinf(F_contrib))) return;
    const float F_mag = GetPHat(F_contrib);
    if (F_mag <= 1e-20f) return;
    const uint  F_pack = PackRGB9E5(F_contrib / F_mag);

    uint2 s2 = uint2(seed, 0u);
    UpdateReservoir(res, wi, 1u,
                    x2, n2_s, L2, V2, uv,
                    matID, objID, eta,
                    F_pack, F_mag, s2);
    seed = s2.x;
}


// Fast Update
bool UpdateReservoir_Fast(
    RWByteAddressBuffer buf,
    uint pixelIdx,
    float wi,
    float3 L2_new,
    float3 V2_new,

    inout uint2 seed
)
{
    float currentWSum = asfloat(buf.Load(addr_wsum(pixelIdx)));
    float newWSum     = currentWSum + wi;
    buf.Store(addr_wsum(pixelIdx), asuint(newWSum));

    bool isAccepted = false;

    if (RandomFloatSingle(seed.x) < (wi / newWSum))
    {
        isAccepted = true;

        buf.Store(addr_l2(pixelIdx), PackRGB9E5(L2_new));
        buf.Store(addr_v2(pixelIdx), PackNormal(normalize(V2_new)));
    }

    return isAccepted;
}


// Store constant hit data for reconnection
void SetReservoir_ConstHit(
    RWByteAddressBuffer buf,
    uint pixelIdx,
    float3 x2_world,
    float3 n2s_world,
    uint   matID,
    uint   objID,
    float  eta
)
{
    float3 xO  = WorldToObjectPos(objID, x2_world);
    float3 nSO = WorldToObjectNrm(objID, n2s_world);

    buf.Store4(addr_pack1(pixelIdx), uint4(asuint(xO), PackNormal(normalize(nSO))));
    buf.Store (addr_objid(pixelIdx), objID);
    buf.Store (addr_matid(pixelIdx), matID);
    buf.Store (addr_eta(pixelIdx),   asuint(eta));
}

void SetReservoir_UV(
    RWByteAddressBuffer buf,
    uint pixelIdx,
    float2 uv
)
{
    buf.Store(addr_uv(pixelIdx), PackFloat2x16(uv.x, uv.y));
}

inline bool TestTemporalCandidate(
    int2   coord,
    float2 dims,
    RWByteAddressBuffer sampleBuf,
    uint   myMatID,
    float3 myN1s,
    float3 myPos,
    out uint   outPixelIdx,
    out uint   outInstID,
    out uint   outPrimID,
    out float2 outBary)
{
    outPixelIdx = 0xFFFFFFFFu;
    outInstID   = 0;
    outPrimID   = 0;
    outBary     = float2(0, 0);

    if (coord.x < 0 || coord.y < 0 || coord.x >= (int)dims.x || coord.y >= (int)dims.y)
        return false;

    uint tpx = MapPixelID(dims, (uint2)coord);

    if (load_isEmitter(sampleBuf, tpx))
        return false;

    uint rI = load_instID(sampleBuf, tpx);
    uint rP = load_primID(sampleBuf, tpx);

    float3 ns = load_n1_s_with_instID(sampleBuf, tpx, rI);
    if (RejectNormal(myN1s, ns, 0.36f))
        return false;

    float2 rB = load_bary(sampleBuf, tpx);
    float3 xr = ReconstructPosition(rI, rP, rB);
    if (RejectDistance(myPos, xr, myN1s, 0.4f))
        return false;

    outPixelIdx = tpx;
    outInstID   = rI;
    outPrimID   = rP;
    outBary     = rB;
    return true;
}
