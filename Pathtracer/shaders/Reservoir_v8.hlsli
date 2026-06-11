//====================================
//RIS RESERVOIR UNIFIED DI+GI
//====================================
//matID discriminates path kind
//matID < MATID_LIGHT_TRI, BSDF-sampled GI vertex at x2, d>=3
//matID == MATID_LIGHT_TRI, NEE to emissive triangle, x2 is hit position, L2 is emission
//matID == MATID_ENV_MISS, env/sky, x2 is unit direction, L2 is radiance
struct Reservoir
{
    //constant-after-hit payload
    float3 x2;
    float3 n2_s;
    uint   objID;
    uint   matID;
    float  eta;
    //resolved x2 material - baked by raygen so reconnection never re-fetches
    float3 Kd;
    float  Pr;
    float  Pm;

    //varying payload
    float3 L2;
    float3 V2;
    float3 F;

    float  W;
    uint   M;
    float  w_sum;
};


//====================================
//SOA GROUP PLANES
//====================================
//64 B/px in 5 planes, grouped by co-access so scattered partner reads
//(temporal candidate, spatial shift / merge lazy accept) touch 4-5 cache
//lines instead of the former 11 single-field planes. Coalesced own-pixel
//access is unchanged (same bytes, fewer transactions + less address math).
//  PACK1 16B  x2(12) | n2pk(4)              reconnection geometry
//  PAY   16B  L2 | Kd | eta/Pr/Pm | matID   reconnection material payload
//  FW    16B  F(12) | W(4)                  RIS state, one Load4 in merge
//  V2     4B  solo plane                    dup pass streams it at 4B stride
//  MISC  12B  objID | M | wsum
static const uint PLANE_PACK1 =  0u;
static const uint PLANE_PAY   = 16u;
static const uint PLANE_FW    = 32u;
static const uint PLANE_V2    = 48u;
static const uint PLANE_MISC  = 52u;

//====================================
//SOA ADDRESS HELPERS
//====================================
//tile-aligned pixel count, must match MapPixelID's 8-wide x 4-tall tile
//swizzle (Common_v8.hlsli) and the CPU-side TileAlignedPx (Renderer.h).
//The old 4x8-shaped formula diverged from MapPixelID's actual max index at
//non-tile-divisible resolutions, overlapping the SoA planes.
uint numPx()                       { return ((IMG_W + 7u) / 8u) * ((IMG_H + 3u) / 4u) * 32u; }

//group bases
uint addr_pack1(uint px)           { return px * 16u; }
uint addr_pay  (uint px)           { uint N = numPx(); return N * PLANE_PAY  + px * 16u; }
uint addr_fw   (uint px)           { uint N = numPx(); return N * PLANE_FW   + px * 16u; }
uint addr_v2   (uint px)           { uint N = numPx(); return N * PLANE_V2   + px *  4u; }
uint addr_misc (uint px)           { uint N = numPx(); return N * PLANE_MISC + px * 12u; }

//per-field addresses inside the groups (legacy helper API preserved)
uint addr_l2(uint px)              { return addr_pay(px); }
uint addr_kd(uint px)              { return addr_pay(px)  +  4u; }
uint addr_eta(uint px)             { return addr_pay(px)  +  8u; }
uint addr_matid(uint px)           { return addr_pay(px)  + 12u; }
uint addr_f(uint px)               { return addr_fw(px); }
uint addr_w(uint px)               { return addr_fw(px)   + 12u; }
uint addr_objid(uint px)           { return addr_misc(px); }
uint addr_m(uint px)               { return addr_misc(px) +  4u; }
uint addr_wsum(uint px)            { return addr_misc(px) +  8u; }

//luminance
inline float GetPHat(float3 v) {
    return 0.2126f * v.x + 0.7152f * v.y + 0.0722f * v.z;
}

//====================================
//ETA + PR + PM PACK (one 32-bit word)
//====================================
//eta as half16 (IOR ~1..2.6, ample precision), Pr/Pm as unorm8 (material-
//native precision - g_mat stores Pr/Pm as unorm8 anyway). Lets the reservoir
//carry the resolved x2 material without growing past 64 B/pixel.
uint PackEtaPrPm(float eta, float pr, float pm)
{
    return (f32tof16(eta) & 0xFFFFu)
         | (uint(saturate(pr) * 255.0f + 0.5f) << 16)
         | (uint(saturate(pm) * 255.0f + 0.5f) << 24);
}
void UnpackEtaPrPm(uint p, out float eta, out float pr, out float pm)
{
    eta = f16tof32(p & 0xFFFFu);
    pr  = float((p >> 16) & 0xFFu) * (1.0f / 255.0f);
    pm  = float((p >> 24) & 0xFFu) * (1.0f / 255.0f);
}

//====================================
//BRDF WRAPPERS
//====================================
//thin aliases to isolate MIS callers from BXDF module
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

//geometry term uses shading normal
float G_term(float3 n, float3 s)
{
    return abs(dot(n, s));
}

//====================================
//RESERVOIR STORE AND LOAD
//====================================
void storeReservoir(RWByteAddressBuffer buf, uint pixelIdx, const Reservoir r)
{
    float3 xO  = WorldToObjectPos(r.objID, r.x2);
    float3 nSO = WorldToObjectNrm(r.objID, r.n2_s);

    //no outer normalize: PackNormal normalizes internally, and its zero/NaN
    //guard must see the raw vector so a cleared reservoir packs the
    //invalid-normal sentinel instead of garbage bits
    buf.Store4(addr_pack1(pixelIdx), uint4(asuint(xO), PackNormal(nSO)));
    buf.Store4(addr_pay(pixelIdx),
               uint4(PackRGB9E5(r.L2), PackRGB9E5(r.Kd),
                     PackEtaPrPm(r.eta, r.Pr, r.Pm), r.matID));
    buf.Store4(addr_fw(pixelIdx), uint4(asuint(r.F), asuint(r.W)));
    buf.Store (addr_v2(pixelIdx), PackNormal(r.V2));
    //objID + M only: wsum (addr_misc + 8) is raygen-owned and not part of
    //the merged record
    buf.Store2(addr_misc(pixelIdx), uint2(r.objID, r.M));
}

//payload-only load: everything Reconnect needs (x2/n2/L2/V2/Kd/Pr/Pm/eta/
//matID/objID) in 4 grouped fetches; F/W/M/w_sum of 'r' are left untouched.
//For scattered partner reads (spat shift, merge lazy accept) where the RIS
//state either is not needed or is being accumulated in place.
void loadReservoirPayload(RWByteAddressBuffer buf, uint pixelIdx, inout Reservoir r)
{
    const uint4 p1  = buf.Load4(addr_pack1(pixelIdx));
    const uint4 pay = buf.Load4(addr_pay(pixelIdx));

    r.objID = buf.Load(addr_objid(pixelIdx));
    r.matID = pay.w;

    r.x2    = ObjectToWorldPos(r.objID, asfloat(p1.xyz));
    r.n2_s  = ObjectToWorldNrm(r.objID, UnpackNormal(p1.w));

    r.L2    = UnpackRGB9E5(pay.x);
    r.Kd    = UnpackRGB9E5(pay.y);
    UnpackEtaPrPm(pay.z, r.eta, r.Pr, r.Pm);

    r.V2    = UnpackNormal(buf.Load(addr_v2(pixelIdx)));
}

Reservoir loadReservoir(RWByteAddressBuffer buf, uint pixelIdx)
{
    Reservoir r = (Reservoir)0;
    loadReservoirPayload(buf, pixelIdx, r);

    const uint4 fw = buf.Load4(addr_fw(pixelIdx));
    r.F     = asfloat(fw.xyz);
    r.W     = asfloat(fw.w);
    r.M     = buf.Load(addr_m(pixelIdx));
    r.w_sum = 0.0f;

    return r;
}


//====================================
//PER-FIELD LOADS AND STORES
//====================================
uint load_objID(RWByteAddressBuffer b, uint pixelIdx)
{
    return b.Load(addr_objid(pixelIdx));
}

//_res suffix: distinct from Sample_Data's load_matID which reads the G-buffer
uint load_matID_res(RWByteAddressBuffer b, uint pixelIdx)
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

//resolved x2 albedo
float3 load_kd_res(RWByteAddressBuffer b, uint pixelIdx)
{
    return UnpackRGB9E5(b.Load(addr_kd(pixelIdx)));
}

//resolved x2 roughness + metalness (share the eta word)
void load_prpm_res(RWByteAddressBuffer b, uint pixelIdx, out float pr, out float pm)
{
    float eta;
    UnpackEtaPrPm(b.Load(addr_eta(pixelIdx)), eta, pr, pm);
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
    return f16tof32(b.Load(addr_eta(pixelIdx)) & 0xFFFFu);
}

void store_wsum(RWByteAddressBuffer b, uint pixelIdx, float wsum)
{
    b.Store(addr_wsum(pixelIdx), asuint(wsum));
}

float load_wsum(RWByteAddressBuffer b, uint pixelIdx)
{
    return asfloat(b.Load(addr_wsum(pixelIdx)));
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

//====================================
//REJECTION AND VALIDITY
//====================================
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
    //n2_s stored in PACK1.w. Must be the zero-normal sentinel: UnpackNormal
    //decodes it to (0,0,0) so IsValidReservoir fails. A raw 0u (the old
    //value) octahedral-decodes to a VALID unit vector and never invalidated
    //anything.
    buf.Store(addr_pack1(pixelIdx) + 12u, PROBE_DI_NORMAL_ZERO_CODE);
}



//====================================
//JACOBIAN HELPERS
//====================================
//geometric jacobian, recomputable from positions and shading normal
inline float ComputeJc(float3 x1, float3 x2, float3 n2_s)
{
    float3 d = x1 - x2;
    float  dist2 = dot(d, d);
    if (dist2 < EPSILON) return EPSILON;
    float  dist = sqrt(dist2);
    return max(abs(dot(d / dist, n2_s)) / dist2, EPSILON);
}

inline float JacobianRatio(float Jn, float Jc)
{
    return (Jc > EPSILON) ? (Jn / Jc) : 0.0f;
}

//====================================
//RECONNECTION
//====================================
//etai1/etat1 are the IOR pair at x1 per raygen's hinfo.backface derivation
inline float3 Reconnect(
    //x1 camera path hit
    in float3  x1,
    in float3  n1_s,
    in float3  o,
    in uint    mID1,
    in float3  localKd1,
    in float   localPr1,
    in float   localPm1,
    in float   etai1,
    in float   etat1,

    //x2 reservoir / reconnection vertex
    in uint    mID2,
    in float3  x2,
    in float3  n2_s,
    in float3  L2,
    in float3  V2,
    in float3  localKd2,
    in float   localPr2,
    in float   localPm2,
    in float   eta2,

    out float  Jn
)
{
    Jn = 1.0f;

    if (length(L2) < EPSILON)
        return 0.0f;

    //====================================
    //DI ENV SKY
    //====================================
    //x2 is direction, no G, no BSDF at x2, Jn=1
    //env treated as infinitely far, no medium absorption applied
    if (mID2 == MATID_ENV_MISS)
    {
        const float3 wi  = normalize(x2);
        const float3 F1  = BSDF_term(mID1, n1_s, n1_s, wi, o,
                                     localKd1, localPr1, localPm1, etai1, etat1);
        const float  ct  = max(1e-15f, dot(n1_s, wi));
        float3 r = F1 * L2 * ct;
        if (any(isnan(r)) || any(isinf(r))) return 0.0f;
        return max(r, 0.0f);
    }

    //====================================
    //DI EMISSIVE TRIANGLE NEE
    //====================================
    //x2 is light position, n2_s is light normal, L2 is emission
    if (mID2 == MATID_LIGHT_TRI)
    {
        const float3 dirT  = x2 - x1;
        const float  distT = length(dirT);
        if (distT < EPSILON) return 0.0f;
        const float3 ndirNT = normalize(-dirT);

        const float3 F1 = BSDF_term(mID1, n1_s, n1_s, -ndirNT, o,
                                    localKd1, localPr1, localPm1, etai1, etat1);
        const float  G1 = G_term(n1_s, -ndirNT);

        //absorption only when x1 is inside transmissive medium
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

    //====================================
    //GI BSDF-SAMPLED VERTEX AT X2
    //====================================

    float3 dir   = x2 - x1;
    float  dist  = length(dir);
    float3 ndirN = normalize(-dir);

    //recover x2 IOR pair from stored etat and material Ni, disambiguate on midpoint
    const float matNi2 = LoadNi(mID2);
    float etai2;
    float etat2 = eta2;
    if (matNi2 <= 1.0f + EPSILON) {
        etai2 = 1.0f;
        etat2 = 1.0f;
    } else {
        etai2 = (eta2 < 0.5f * (1.0f + matNi2)) ? matNi2 : 1.0f;
    }

    //segment medium detection, x1 side preferred
    //opaque surface (Kd.w~=1) has IOR boundary but no interior, skip Beer-Lambert
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

    //if segment inside medium, override incident IOR at x2
    if      (x1_inMedium) etai2 = iorAfterX1;
    else if (x2_inMedium) etai2 = iorBeforeX2;

    float3 F1 = BSDF_term(mID1, n1_s, n1_s, -ndirN, o,  localKd1, localPr1, localPm1, etai1, etat1);
    float3 F2 = BSDF_term(mID2, n2_s, n2_s, -V2, ndirN, localKd2, localPr2, localPm2, etai2, etat2);

    float  G1  = G_term(n1_s, -ndirN);
    float  G2  = G_term(n2_s, -V2);

    //Beer-Lambert applied at most once, same medium bounds both sides
    float3 transmittance = float3(1.0f, 1.0f, 1.0f);
    if (x1_inMedium) {
        transmittance = CalculateAbsorptionThroughput(LoadTf(mID1), dist);
    } else if (x2_inMedium) {
        transmittance = CalculateAbsorptionThroughput(LoadTf(mID2), dist);
    }

    Jn = max(abs(dot(ndirN, n2_s)) / (dist * dist), EPSILON);
    float3 r = F1 * F2 * L2 * G1 * G2 * transmittance;

    if (any(isnan(r)) || any(isinf(r)) || all(r < EPSILON))
        r = (float3)0.0f;

    return max(r,0.0f);
}



//====================================
//RESERVOIR UPDATE
//====================================
bool UpdateReservoir(
    inout Reservoir reservoir,
    in float wi,
    in uint  M,

    in float3 x2,
    in float3 n2_s,
    in float3 L2,
    in float3 V2,

    in float3 Kd, in float Pr, in float Pm,

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

        reservoir.Kd    = Kd;
        reservoir.Pr    = Pr;
        reservoir.Pm    = Pm;

        reservoir.L2    = L2;
        reservoir.V2    = V2;
        reservoir.F     = F;
        return true;
    }
    return false;
}


//====================================
//INITIAL RESAMPLING CANDIDATE
//====================================
//wsum lives in a register, on acceptance writes full reservoir payload to SoA buffer
//F stored as full RGB, target magnitude is GetPHat(F)
//sentinel objIDs bypass object-space transform via Sample_Data_v8 identity shortcut
inline bool AddInitialCandidate(
    inout float wsum,
    RWByteAddressBuffer buf,
    uint pixelIdx,
    float  wi,
    float3 x2, float3 n2_s,
    float3 L2, float3 V2,
    float3 Kd, float Pr, float Pm,
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
        buf.Store4(addr_pack1(pixelIdx), uint4(asuint(xO), PackNormal(nSO)));
        buf.Store4(addr_pay  (pixelIdx),
                   uint4(PackRGB9E5(L2), PackRGB9E5(Kd),
                         PackEtaPrPm(eta, Pr, Pm), matID));
        //F only - W and M land in the FW/MISC groups at raygen finalize
        buf.Store3(addr_f    (pixelIdx), asuint(F_contrib));
        buf.Store (addr_v2   (pixelIdx), PackNormal(V2));
        buf.Store (addr_objid(pixelIdx), objID);
        return true;
    }
    return false;
}

//====================================
//TEMPORAL CANDIDATE TEST
//====================================
inline bool TestTemporalCandidate(
    int2   coord,
    float2 dims,
    RWByteAddressBuffer sampleBuf,
    out uint outPixelIdx)
{
    outPixelIdx = 0xFFFFFFFFu;

    if (coord.x < 0 || coord.y < 0 || coord.x >= (int)dims.x || coord.y >= (int)dims.y)
        return false;

    uint tpx = MapPixelID(dims, (uint2)coord);

    if (load_isEmitter(sampleBuf, tpx))
        return false;

    outPixelIdx = tpx;
    return true;
}
