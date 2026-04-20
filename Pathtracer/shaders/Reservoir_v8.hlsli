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
    float3 F;          // full RGB contribution; GetPHat(F) IS the target magnitude

    float  W;
    uint   M;
    float  w_sum;      // raygen-only (merge passes overwrite before use)
};


//Per-field sizes (SoA layout, per-plane stride)
static const uint SZ_PACK1 = 16u;  // x2(12) + n2_s_packed(4)
static const uint SZ_4     =  4u;
static const uint SZ_12    = 12u;  // float3 F

//Plane cumulative offsets (in bytes per pixel)
static const uint PLANE_PACK1 =  0u;
static const uint PLANE_L2    = 16u;
static const uint PLANE_V2    = 20u;
static const uint PLANE_OBJID = 24u;
static const uint PLANE_UV    = 28u;
static const uint PLANE_MATID = 32u;
static const uint PLANE_W     = 36u;
static const uint PLANE_F     = 40u;   // float3 -> 12 bytes
static const uint PLANE_M     = 52u;
static const uint PLANE_ETA   = 56u;
static const uint PLANE_WSUM  = 60u;

// SoA address helpers
// Tile-aligned pixel count, must match MapPixelID's 4x8 tile swizzle.
uint numPx()                       { return ((IMG_W + 3u) / 4u) * ((IMG_H + 7u) / 8u) * 32u; }
uint addr_pack1(uint px)           { return px * SZ_PACK1; }
uint addr_l2(uint px)              { uint N = numPx(); return N * PLANE_L2    + px * SZ_4; }
uint addr_v2(uint px)              { uint N = numPx(); return N * PLANE_V2    + px * SZ_4; }
uint addr_objid(uint px)           { uint N = numPx(); return N * PLANE_OBJID + px * SZ_4; }
uint addr_uv(uint px)              { uint N = numPx(); return N * PLANE_UV    + px * SZ_4; }
uint addr_matid(uint px)           { uint N = numPx(); return N * PLANE_MATID + px * SZ_4; }
uint addr_w(uint px)               { uint N = numPx(); return N * PLANE_W     + px * SZ_4; }
uint addr_f(uint px)               { uint N = numPx(); return N * PLANE_F     + px * SZ_12; }
uint addr_m(uint px)               { uint N = numPx(); return N * PLANE_M     + px * SZ_4; }
uint addr_eta(uint px)             { uint N = numPx(); return N * PLANE_ETA   + px * SZ_4; }
uint addr_wsum(uint px)            { uint N = numPx(); return N * PLANE_WSUM  + px * SZ_4; }

// Scalar magnitude used throughout (luminance).
inline float GetPHat(float3 v) {
    return 0.2126f * v.x + 0.7152f * v.y + 0.0722f * v.z;
}

// BRDF wrappers (kept as thin aliases to isolate MIS callers from the
// underlying BXDF module).
// Computes the sampling-strategy probabilities inline so the BXDF module
// can branch on inactive lobes without burdening every caller.
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
    const SamplingP p = CalculateStrategyProbabilities(mID, o, n_s, etai, etat, localKd, localPm);
    return EvaluateBRDF_COMBINED(p, mID, n_s, n_g, s, o, localKd, localPr, localPm, etai, etat);
}

// Geometry term uses the shading normal (geometric normal has been retired).
float G_term(float3 n, float3 s)
{
    return abs(dot(n, s));
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
    buf.Store3(addr_f(pixelIdx),     asuint(r.F));
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
    r.F     = asfloat(buf.Load3(addr_f(pixelIdx)));

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

float3 load_F(RWByteAddressBuffer b, uint pixelIdx)
{
    return asfloat(b.Load3(addr_f(pixelIdx)));
}

float load_eta(RWByteAddressBuffer b, uint pixelIdx)
{
    return asfloat(b.Load(addr_eta(pixelIdx)));
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

void store_F(RWByteAddressBuffer b, uint pixelIdx, float3 F)
{
    b.Store3(addr_f(pixelIdx), asuint(F));
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

// Rejects the neighbor candidate if the neighbor→current shift Jacobian
// ratio Jn(myPos toward bX2) / Jc(bPos toward bX2) falls outside
// [rs_rejJacobianMin, rs_rejJacobianMax]. This is the ratio that
// multiplies w_n directly and can spike ReSTIR variance in corners.
// The opposite direction (canonical sample shifted into the neighbor's
// pixel) only enters mis_c's denominator and is self-bounded by pairwise
// MIS — m_c stays in [M_c/M_sum, 1] no matter how extreme that ratio
// gets — so checking it would only discard usable samples.
// Env/miss (MATID_ENV_MISS) preserves direction under shift so its ratio
// is 1 by construction.
inline bool JacobianRejected(float3 myPos,
                             float3 bPos,
                             uint   bMatRes,
                             float3 bX2res, float3 bN2res)
{
    if (bMatRes == MATID_ENV_MISS) return false;

    const float Jn = ComputeJc(myPos, bX2res, bN2res);
    const float Jc = ComputeJc(bPos,  bX2res, bN2res);
    const float ratio = Jn / max(Jc, EPSILON);

    // Negated range test also catches NaN / inf / negative.
    return !(ratio >= rs_rejJacobianMin && ratio <= rs_rejJacobianMax);
}

// Calculate reconnection.
//
// etai1/etat1 are the IOR pair at x1 relative to its (possibly flipped) shading
// normal n1_s, exactly as raygen derives them from hinfo.backface. Callers
// pass sv.etai / sv.etat from BuildVertex* — no additional buffer loads.
inline float3 Reconnect(
    // Vertex x1 (camera path hit)
    in float3  x1,
    in float3  n1_s,
    in float3  o,
    in uint    mID1,
    in float3  localKd1,
    in float   localPr1,
    in float   localPm1,
    in float   etai1,
    in float   etat1,

    // Vertex x2 (reservoir / reconnection vertex)
    in uint    mID2,
    in float3  x2,
    in float3  n2_s,
    in float3  L2,
    in float3  V2,
    in float3  localKd2,
    in float   localPr2,
    in float   localPm2,
    in float   eta2, // stored transmittance-side IOR at x2 (etat2)

    out float  Jn
)
{
    Jn = 1.0f;

    if (length(L2) < EPSILON)
        return 0.0f;

    //───────────────────────────────────────────────────────────────────────
    // DI: environment / sky sample. x2 stores a DIRECTION. No G term, no
    // BSDF at x2. Jn = 1 (direction is preserved under the reconnection
    // shift). Env is treated as infinitely far, so we don't apply medium
    // absorption here — if x1 is inside a medium, env light is effectively
    // the transmitted sky beyond the medium and user-facing absorption
    // tinting would require explicit thickness info we don't have.
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
    // n2_s is the light's surface normal, L2 is emission.
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

        // Absorption only when x1 is actually inside a transmissive medium.
        // Matches the gate in the GI branch below.
        const float rayDotN1    = dot(-ndirNT, n1_s);
        const float iorAfterX1  = (rayDotN1 >= 0.0f) ? etai1 : etat1;
        const bool  m1_inMedium = (iorAfterX1 > 1.0f + EPSILON)
                                  && (LoadKd_w(mID1) < 1.0f - EPSILON);
        float3 transmittance = float3(1.0f, 1.0f, 1.0f);
        if (m1_inMedium) {
            transmittance = CalculateAbsorptionThroughput(LoadTf(mID1), distT);
        }

        float3 r = F1 * L2 * G1 * transmittance;
        if (any(isnan(r)) || any(isinf(r))) r = 0.0f;

        Jn = max(abs(dot(ndirNT, n2_s)) / (distT * distT), EPSILON);
        return max(r, 0.0f);
    }

    //───────────────────────────────────────────────────────────────────────
    // GI: BSDF-sampled vertex at x2 (original path, d >= 3).
    //───────────────────────────────────────────────────────────────────────

    // Geometric prep
    float3 dir   = x2 - x1;
    float  dist  = length(dir);
    float3 ndirN = normalize(-dir); // direction from x2 to x1

    // Recover x2's IOR pair from stored etat (eta2) and material Ni.
    //   frontface original: (etai2, etat2) = (1, matNi2),  eta2 = matNi2
    //   backface  original: (etai2, etat2) = (matNi2, 1),  eta2 = 1
    // Disambiguate on the midpoint so tiny numerical drift doesn't flip.
    const float matNi2 = LoadNi(mID2);
    float etai2;
    float etat2 = eta2;
    if (matNi2 <= 1.0f + EPSILON) {
        etai2 = 1.0f;
        etat2 = 1.0f;
    } else {
        etai2 = (eta2 < 0.5f * (1.0f + matNi2)) ? matNi2 : 1.0f;
    }

    // "Which medium is the segment x1→x2 in?"  At x1 the ray exits toward
    // the etai1 half (dot ≥ 0) or etat1 half (dot < 0). At x2 it arrives
    // from the etai2 half (dot ≥ 0) or etat2 half (dot < 0). If either
    // side's IOR is > 1 AND the material is actually transmissive, the
    // segment is inside that side's medium. An opaque (Kd.w ≈ 1) surface
    // has an IOR boundary but no interior — don't apply Beer-Lambert there.
    const float rayDotN1 = dot(-ndirN, n1_s);
    const float rayDotN2 = dot( ndirN, n2_s);
    const float iorAfterX1  = (rayDotN1 >= 0.0f) ? etai1 : etat1;
    const float iorBeforeX2 = (rayDotN2 >= 0.0f) ? etai2 : etat2;

    const float m1_Kd_w = LoadKd_w(mID1);
    const float m2_Kd_w = LoadKd_w(mID2);
    const bool m1_transmissive = m1_Kd_w < 1.0f - EPSILON;
    const bool m2_transmissive = m2_Kd_w < 1.0f - EPSILON;

    const bool x1_inMedium = (iorAfterX1  > 1.0f + EPSILON) && m1_transmissive;
    const bool x2_inMedium = (iorBeforeX2 > 1.0f + EPSILON) && m2_transmissive;

    // If the segment is inside a medium, the incident IOR at x2 picks that
    // up instead of air. Prefer x1's medium (ray leaves x1 first).
    if      (x1_inMedium) etai2 = iorAfterX1;
    else if (x2_inMedium) etai2 = iorBeforeX2;

    float3 F1 = BSDF_term(mID1, n1_s, n1_s, -ndirN, o,  localKd1, localPr1, localPm1, etai1, etat1);
    float3 F2 = BSDF_term(mID2, n2_s, n2_s, -V2, ndirN, localKd2, localPr2, localPm2, etai2, etat2);

    // Geometry term
    float  G1  = G_term(n1_s, -ndirN);
    float  G2  = G_term(n2_s, -V2);

    // Beer-Lambert absorption for whichever medium the segment passes
    // through. Applied at most once (the normal case — both sides agreeing
    // means x1 and x2 bound the same medium, so only one factor is correct).
    float3 transmittance = float3(1.0f, 1.0f, 1.0f);
    if (x1_inMedium) {
        transmittance = CalculateAbsorptionThroughput(LoadTf(mID1), dist);
    } else if (x2_inMedium) {
        transmittance = CalculateAbsorptionThroughput(LoadTf(mID2), dist);
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

    in float3 F,

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
        return true;
    }
    return false;
}


// Initial-resampling candidate update: accumulates wsum in a register
// (the only RIS scalar live across iterations) and, on acceptance, writes
// the full reservoir payload — constant fields (x2, n2, matID, objID, eta,
// uv) plus varying fields (L2, V2, F) — straight to the SoA buffer. F is
// stored as the full RGB contribution (no normalize + magnitude split); the
// target magnitude is always GetPHat(F).
//
// Sentinel objIDs (env/miss) bypass the object-space transform via the
// identity shortcut in Sample_Data_v8.
inline bool AddInitialCandidate(
    inout float wsum,
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
    if (GetPHat(F_contrib) <= 1e-20f) return false;

    wsum += wi;
    if (wsum > EPSILON && RandomFloatSingle(seed) < (wi / wsum))
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
        buf.Store3(addr_f    (pixelIdx), asuint(F_contrib));
        return true;
    }
    return false;
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
