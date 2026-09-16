// Packed fields and offsets must match host reservoir storage.
struct Reservoir
{

    float3 x2;
    float3 n2_s;
    uint   objID;
    uint   matID;
    float  eta;

    float3 Kd;
    float  Pr;
    float  Pm;

    float3 L2;
    float3 V2;
    float3 F;

    float  W;
    uint   M;
    float  w_sum;

    uint   rcInfo;
    uint   seed;
    float  cachedJac;
    float  gBase;
};

static const uint PLANE_FW    = 48u;
static const uint PLANE_V2    = 64u;
static const uint PLANE_HYB   = 68u;

// Address planes use the packed 8-by-4 pixel layout.
uint numPx()                       { return ((IMG_W + 7u) / 8u) * ((IMG_H + 3u) / 4u) * 32u; }

uint addr_recon(uint px)           { return px * 48u; }
uint addr_pack1(uint px)           { return addr_recon(px); }
uint addr_pay  (uint px)           { return addr_recon(px) + 16u; }
uint addr_pm   (uint px)           { return addr_recon(px) + 32u; }
uint addr_fw   (uint px)           { uint N = numPx(); return N * PLANE_FW   + px * 16u; }
uint addr_v2   (uint px)           { uint N = numPx(); return N * PLANE_V2   + px *  4u; }

uint addr_l2(uint px)              { return addr_pay(px); }
uint addr_kd(uint px)              { return addr_pay(px)  +  4u; }
uint addr_eta(uint px)             { return addr_pay(px)  +  8u; }
uint addr_matid(uint px)           { return addr_pay(px)  + 12u; }
uint addr_f(uint px)               { return addr_fw(px); }
uint addr_w(uint px)               { return addr_fw(px)   + 12u; }
uint addr_objid(uint px)           { return addr_pm(px); }
uint addr_m(uint px)               { return addr_pm(px)   +  8u; }
uint addr_rcinfo(uint px)          { return addr_pm(px)   + 12u; }
uint addr_hyb(uint px)             { uint N = numPx(); return N * PLANE_HYB  + px * 12u; }

#define RC_F_NOPK       (1u << 12)
#define RC_F_NOPPREV    (1u << 13)
#define RC_F_PAREA      (1u << 14)
#define RC_F_ENV_REPLAY (1u << 15)

#define RC_F_LOBES      (1u << 10)

inline uint RcPackInfo(uint k, uint d, uint flags) { return (k & 15u) | ((d & 63u) << 4) | flags; }
inline uint RcK(uint info)        { return info & 15u; }
inline uint RcD(uint info)        { return (info >> 4) & 63u; }
inline bool RcReusable(uint info) { return (info & 15u) >= 2u; }
inline bool RcEnvReplay(uint info){ return (info & RC_F_ENV_REPLAY) != 0u; }
inline bool RcHasLobes(uint info) { return (info & RC_F_LOBES) != 0u; }

inline uint RcLobeAt(uint info, uint vtx)
{
    return (info >> (16u + ((vtx - 1u) << 1))) & 3u;
}

inline uint RcLobesWord(uint mask) { return RC_F_LOBES | ((mask & 0xFFFFu) << 16); }

inline uint RcReplayLen(uint info)
{
    const uint k = RcK(info);
    if (k < 2u) return 0u;
    return (k - 2u) + (RcEnvReplay(info) ? 1u : 0u);
}

inline bool RcRoughPass(float Pr) { return Pr >= rs_reconnectRoughnessMin; }

inline bool RcCritPair(float PrPrev, float PrHit, bool hitIsEndOrVolume, float segDist, float distMin)
{
    if (!RcRoughPass(PrPrev)) return false;
    if (!hitIsEndOrVolume && !RcRoughPass(PrHit)) return false;
    return segDist >= distMin;
}

inline bool RcLobeProxyPass(float pdf)
{
    return pdf * pdf * rs_reconnectRoughnessMin <= 1.0f;
}

inline float RcFpThreshold(float camDist2, float cosPrim)
{
    return (rs_rcFpKappa * 0.01f) * camDist2 * (4.0f * PI) / max(cosPrim, 1e-4f);
}

inline bool RcFpDensityPass(float pdf, float G, float fpThresh)
{
    return pdf * G * fpThresh <= 1.0f;
}

inline bool RcGeomReject(float Jn, float gBase, float T)
{
    if (!(gBase > 0.0f)) return true;
    const float j = Jn / gBase;
    return (isnan(j) || isinf(j) || j > T || j < (1.0f / T));
}

inline float RcGeomClampScale(float Jn, float gBase, float T)
{
    if (!(gBase > 0.0f) || !(Jn > 0.0f)) return 0.0f;
    const float j = Jn / gBase;
    if (isnan(j) || isinf(j)) return 0.0f;
    const float t = max(T, 1.0f);
    return clamp(j, 1.0f / t, t) / j;
}

inline float GetPHat(float3 v) {
    return 0.2126f * v.x + 0.7152f * v.y + 0.0722f * v.z;
}

inline float FinalizeUCW(float w_sum, float p_hat, float wMax)
{
    if (!(p_hat > EPSILON) || !(w_sum > 0.0f)) return 0.0f;
    const float W = w_sum / p_hat;
    if (isnan(W) || isinf(W) || W < 0.0f) return 0.0f;
    return (wMax > 0.0f) ? min(W, wMax) : W;
}

// Pack optical and roughness parameters into half precision.
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

BrdfData BSDF_term_sel(
    bool   useLobe,
    uint   lobe,
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
    const SamplingP p = CalculateStrategyProbabilities(mID, o, n_s, etai, etat, localKd, localPr, localPm);
    float3 lobeVal; float lobePdf;
    const BrdfData m = EvaluateAndPdf_COMBINED_L(p, useLobe ? lobe : 0xFFFFFFFFu,
                                                 mID, n_s, n_g, s, o,
                                                 localKd, localPr, localPm, etai, etat, false,
                                                 lobeVal, lobePdf);
    BrdfData r;
    r.val = useLobe ? lobeVal : m.val;
    r.pdf = useLobe ? lobePdf : m.pdf;
    return r;
}

float G_term(float3 n, float3 s)
{
    return abs(dot(n, s));
}

// Store reservoir fields at offsets shared with neighboring passes.
void storeReservoir(RWByteAddressBuffer buf, uint pixelIdx, const Reservoir r)
{
    float3 xO  = WorldToObjectPos(r.objID, r.x2);
    float3 nSO = WorldToObjectNrm(r.objID, r.n2_s);
    const uint v2pk = PackNormal(r.V2);

    buf.Store4(addr_pack1(pixelIdx), uint4(asuint(xO), PackNormal(nSO)));
    buf.Store4(addr_pay(pixelIdx),
               uint4(PackRGB9E5(r.L2), PackRGB9E5(r.Kd),
                     PackEtaPrPm(r.eta, r.Pr, r.Pm), r.matID));
    buf.Store4(addr_pm(pixelIdx), uint4(r.objID, v2pk, r.M, r.rcInfo));
    buf.Store4(addr_fw(pixelIdx), uint4(asuint(r.F), asuint(r.W)));
    buf.Store (addr_v2(pixelIdx), v2pk);
    buf.Store3(addr_hyb(pixelIdx), uint3(r.seed, asuint(r.cachedJac), asuint(r.gBase)));

}

void loadReservoirPayload(RWByteAddressBuffer buf, uint pixelIdx, inout Reservoir r)
{
    const uint4 p1  = buf.Load4(addr_pack1(pixelIdx));
    const uint4 pay = buf.Load4(addr_pay(pixelIdx));
    const uint4 pm  = buf.Load4(addr_pm(pixelIdx));
    const uint3 hyb = buf.Load3(addr_hyb(pixelIdx));

    r.objID = pm.x;
    r.matID = pay.w;

    r.x2    = ObjectToWorldPos(r.objID, asfloat(p1.xyz));
    r.n2_s  = ObjectToWorldNrm(r.objID, UnpackNormal(p1.w));

    r.L2    = UnpackRGB9E5(pay.x);
    r.Kd    = UnpackRGB9E5(pay.y);
    UnpackEtaPrPm(pay.z, r.eta, r.Pr, r.Pm);

    r.V2    = UnpackNormal(pm.y);

    r.rcInfo    = pm.w;
    r.seed      = hyb.x;
    r.cachedJac = asfloat(hyb.y);
    r.gBase     = asfloat(hyb.z);
}

void loadReservoirPayloadSel(bool fromLast, uint pixelIdx, inout Reservoir r)
{
    if (fromLast) loadReservoirPayload(g_Reservoirs_last,    pixelIdx, r);
    else          loadReservoirPayload(g_Reservoirs_current, pixelIdx, r);
}

// Reconstruct a reservoir from its packed payload and state planes.
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

uint load_rcInfo(RWByteAddressBuffer b, uint pixelIdx)
{
    return b.Load(addr_rcinfo(pixelIdx));
}

uint load_seed(RWByteAddressBuffer b, uint pixelIdx)
{
    return b.Load(addr_hyb(pixelIdx));
}

void loadReservoirState(RWByteAddressBuffer buf, uint pixelIdx, inout Reservoir r)
{
    const uint4 fw  = buf.Load4(addr_fw(pixelIdx));
    const uint3 hyb = buf.Load3(addr_hyb(pixelIdx));
    r.F         = asfloat(fw.xyz);
    r.W         = asfloat(fw.w);
    r.M         = buf.Load(addr_m(pixelIdx));
    r.seed      = hyb.x;
    r.cachedJac = asfloat(hyb.y);
    r.gBase     = asfloat(hyb.z);
}

uint load_matID_res(RWByteAddressBuffer b, uint pixelIdx)
{
    return b.Load(addr_matid(pixelIdx));
}

float3 load_x2(RWByteAddressBuffer b, uint pixelIdx, uint objID)
{
    float3 xO = asfloat(b.Load4(addr_pack1(pixelIdx)).xyz);
    return ObjectToWorldPos(objID, xO);
}

float load_W(RWByteAddressBuffer b, uint pixelIdx)
{
    return asfloat(b.Load(addr_w(pixelIdx)));
}

float3 load_F(RWByteAddressBuffer b, uint pixelIdx)
{
    return asfloat(b.Load3(addr_f(pixelIdx)));
}

uint load_M(RWByteAddressBuffer b, uint pixelIdx)
{
    return b.Load(addr_m(pixelIdx));
}

float load_gBase(RWByteAddressBuffer b, uint pixelIdx)
{
    return asfloat(b.Load(addr_hyb(pixelIdx) + 8u));
}

void store_M(RWByteAddressBuffer b, uint pixelIdx, uint M)
{
    b.Store(addr_m(pixelIdx), M);
}

void store_W(RWByteAddressBuffer b, uint pixelIdx, float W)
{
    b.Store(addr_w(pixelIdx), asuint(W));
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

inline void InvalidateReservoir_ShadingNormal(
    RWByteAddressBuffer buf,
    uint pixelIdx
)
{

    buf.Store(addr_pack1(pixelIdx) + 12u, PROBE_DI_NORMAL_ZERO_CODE);
}

inline float ComputeJc(float3 x1, float3 x2, float3 n2_s)
{
    float3 d = x1 - x2;
    float  dist2 = dot(d, d);
    if (dist2 < EPSILON) return EPSILON;
    float  dist = sqrt(dist2);
    return max(abs(dot(d / dist, n2_s)) / dist2, EPSILON);
}

inline float3 ReconnectPSS(

    in float3  x1,
    in float3  n1_s,
    in float3  o,
    in uint    mID1,
    in float3  localKd1,
    in float   localPr1,
    in float   localPm1,
    in float   etai1,
    in float   etat1,

    in uint    mID2,
    in float3  x2,
    in float3  n2_s,
    in float3  L2,
    in float3  V2,
    in float3  localKd2,
    in float   localPr2,
    in float   localPm2,
    in float   eta2,

    in uint    rcInfo,
    in float   srcCachedJac,

    out float  Jn,
    out float  cachedNew
)
{
    Jn        = 1.0f;
    cachedNew = 0.0f;

    if (length(L2) < EPSILON)
        return 0.0f;

    const bool sss1 = LoadIsSSS(mID1);

    const bool useLobes = RcHasLobes(rcInfo);
    const uint rcKk     = RcK(rcInfo);

    if (mID2 == MATID_ENV_MISS)
    {
        const float3 wi  = normalize(x2);

        const BrdfData bd1 = BSDF_term_sel(useLobes && !(rcInfo & RC_F_PAREA),
                                           RcLobeAt(rcInfo, rcKk - 1u),
                                           mID1, n1_s, n1_s, wi, o,
                                           localKd1, localPr1, localPm1, etai1, etat1);
        const float  ct  = max(1e-15f, dot(n1_s, wi));
        float3 r = bd1.val * L2 * ct;
        if (any(isnan(r)) || any(isinf(r))) return 0.0f;
        cachedNew = (rcInfo & RC_F_PAREA) ? srcCachedJac
                                          : max(bd1.pdf, EPSILON);
        return max(r, 0.0f);
    }

    if (mID2 == MATID_LIGHT_TRI)
    {
        const float3 dirT  = x2 - x1;
        const float  distT = length(dirT);
        if (distT < EPSILON) return 0.0f;
        const float3 ndirNT = normalize(-dirT);

        const BrdfData bd1 = BSDF_term_sel(useLobes && !(rcInfo & RC_F_PAREA),
                                           RcLobeAt(rcInfo, rcKk - 1u),
                                           mID1, n1_s, n1_s, -ndirNT, o,
                                           localKd1, localPr1, localPm1, etai1, etat1);
        const float  G1 = G_term(n1_s, -ndirNT);

        const float rayDotN1    = dot(-ndirNT, n1_s);
        const float iorAfterX1  = (rayDotN1 >= 0.0f) ? etai1 : etat1;

        const bool  m1_inMedium = (iorAfterX1 > 1.0f + EPSILON)
                                  && (LoadKd_w(mID1) < 1.0f - EPSILON)
                                  && !LoadIsThinGlass(mID1);
        float3 transmittance = float3(1.0f, 1.0f, 1.0f);
        if (m1_inMedium) {
            transmittance = CalculateAbsorptionThroughput(LoadTf(mID1), distT);
        }

        float3 r = bd1.val * L2 * G1 * transmittance;
        if (any(isnan(r)) || any(isinf(r))) r = 0.0f;

        Jn = max(abs(dot(ndirNT, n2_s)) / (distT * distT), EPSILON);
        cachedNew = (rcInfo & RC_F_PAREA) ? srcCachedJac
                                          : max(bd1.pdf, EPSILON) * Jn;
        return max(r, 0.0f);
    }

    float3 dir   = x2 - x1;
    float  dist  = length(dir);
    float3 ndirN = normalize(-dir);

    const uint  baseID2 = MatIDBase(mID2);
    const bool  volume2 = IsVolumeVertex(mID2);
    const float G1      = G_term(n1_s, -ndirN);

    if (volume2)
    {
        if (!sss1)
            return 0.0f;
        const float3 F1v = localKd1 * SSS_INV_PI;

        const float  radius  = max(LoadSSSRadius(baseID2), SSS_MIN_RADIUS);
        const float  sigma_t = 1.0f / radius;
        const float3 albedo  = saturate(LoadSSSAlbedo(baseID2));
        const float  gg      = LoadPhaseG(baseID2);
        const float  cosP    = dot(-ndirN, V2);
        const float  phase   = EvaluatePhaseHG(gg, cosP);
        const float3 F2v     = sigma_t * albedo * phase;
        const float3 Tv      = (float3)exp(-sigma_t * dist);

        Jn = max(1.0f / (dist * dist), EPSILON);
        cachedNew = max((G1 * SSS_INV_PI)
                        * (sigma_t * exp(-sigma_t * dist))
                        * phase
                        * Jn, EPSILON);
        float3 rv = F1v * F2v * L2 * G1 * Tv;
        if (any(isnan(rv)) || any(isinf(rv)) || all(rv < EPSILON))
            rv = (float3)0.0f;
        return max(rv, 0.0f);
    }

    const float matNi2 = LoadNi(baseID2);
    float etai2;
    float etat2 = eta2;
    if (matNi2 <= 1.0f + EPSILON) {
        etai2 = 1.0f;
        etat2 = 1.0f;
    } else {
        etai2 = (eta2 < 0.5f * (1.0f + matNi2)) ? matNi2 : 1.0f;
    }

    const float rayDotN1 = dot(-ndirN, n1_s);
    const float rayDotN2 = dot( ndirN, n2_s);
    const float iorAfterX1  = (rayDotN1 >= 0.0f) ? etai1 : etat1;
    const float iorBeforeX2 = (rayDotN2 >= 0.0f) ? etai2 : etat2;

    const float m1_Kd_w = LoadKd_w(mID1);
    const float m2_Kd_w = LoadKd_w(baseID2);

    const bool m1_transmissive = m1_Kd_w < 1.0f - EPSILON && !LoadIsThinGlass(mID1);
    const bool m2_transmissive = m2_Kd_w < 1.0f - EPSILON && !LoadIsThinGlass(baseID2);

    const bool x1_inMedium = (iorAfterX1  > 1.0f + EPSILON) && m1_transmissive;
    const bool x2_inMedium = (iorBeforeX2 > 1.0f + EPSILON) && m2_transmissive;

    if      (x1_inMedium) etai2 = iorAfterX1;
    else if (x2_inMedium) etai2 = iorBeforeX2;

    const bool exit2 = IsSSSExitVertex(mID2);
    float  pdf1 = 0.0f;
    float3 F1;
    if (exit2) {
        F1 = localKd1 * SSS_INV_PI;
    } else {

        const BrdfData bd1 = BSDF_term_sel(useLobes && !(rcInfo & RC_F_NOPPREV),
                                           RcLobeAt(rcInfo, rcKk - 1u),
                                           mID1, n1_s, n1_s, -ndirN, o,
                                           localKd1, localPr1, localPm1, etai1, etat1);
        F1   = bd1.val;
        pdf1 = bd1.pdf;
    }

    const BrdfData bd2 = BSDF_term_sel(useLobes && !(rcInfo & RC_F_NOPK),
                                       RcLobeAt(rcInfo, rcKk),
                                       baseID2, n2_s, n2_s, -V2, ndirN,
                                       localKd2, localPr2, localPm2, etai2, etat2);
    const float3 F2 = bd2.val;

    float  G2  = G_term(n2_s, -V2);

    float3 transmittance = float3(1.0f, 1.0f, 1.0f);
    if (x1_inMedium) {
        transmittance = CalculateAbsorptionThroughput(LoadTf(mID1), dist);
    } else if (x2_inMedium) {
        transmittance = CalculateAbsorptionThroughput(LoadTf(baseID2), dist);
    }

    Jn = max(abs(dot(ndirN, n2_s)) / (dist * dist), EPSILON);

    float bundle = 1.0f;
    if (!(rcInfo & RC_F_NOPPREV)) bundle *= max(pdf1,    EPSILON);
    if (!(rcInfo & RC_F_NOPK))    bundle *= max(bd2.pdf, EPSILON);
    cachedNew = bundle * Jn;

    float3 r = F1 * F2 * L2 * G1 * G2 * transmittance;

    if (any(isnan(r)) || any(isinf(r)) || all(r < EPSILON))
        r = (float3)0.0f;

    return max(r,0.0f);
}

inline float3 ReconnectPSS_sv(in SurfaceVertex sv, in Reservoir r,
                              out float Jn, out float cachedNew)
{
    return ReconnectPSS(sv.x, sv.n_s, sv.o, sv.matID,
                        sv.Kd, sv.Pr, sv.Pm, sv.etai, sv.etat,
                        r.matID, r.x2, r.n2_s, r.L2, r.V2,
                        r.Kd, r.Pr, r.Pm, r.eta,
                        r.rcInfo, r.cachedJac,
                        Jn, cachedNew);
}

// Merge a candidate while preserving reservoir weight and replay metadata.
bool UpdateReservoir(
    inout Reservoir reservoir,
    in float wi,
    in uint  M,
    in Reservoir src,
    in float3 F_shifted,
    in float  cachedNew,
    in float  Jn,
    inout uint2 seed
)
{
    reservoir.w_sum += wi;
    reservoir.M     += M;

    if (RandomFloatSingle(seed.x) < (wi / reservoir.w_sum))
    {
        reservoir.x2    = src.x2;
        reservoir.n2_s  = src.n2_s;
        reservoir.objID = src.objID;
        reservoir.matID = src.matID;
        reservoir.eta   = src.eta;

        reservoir.Kd    = src.Kd;
        reservoir.Pr    = src.Pr;
        reservoir.Pm    = src.Pm;

        reservoir.L2    = src.L2;
        reservoir.V2    = src.V2;
        reservoir.F     = F_shifted;

        reservoir.rcInfo    = src.rcInfo;
        reservoir.seed      = src.seed;
        reservoir.cachedJac = cachedNew;
        reservoir.gBase     = Jn;
        return true;
    }
    return false;
}

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
    uint   rcInfo, uint pathSeed, float cachedJac, float gBase,
    inout uint seed)
{
    if (wi <= 0.0f || any(isnan(F_contrib)) || any(isinf(F_contrib))) return false;
    if (GetPHat(F_contrib) <= 1e-20f) return false;

    wsum += wi;
    if (wsum > EPSILON && RandomFloatSingle(seed) < (wi / wsum))
    {
        const float3 xO   = WorldToObjectPos(objID, x2);
        const float3 nSO  = WorldToObjectNrm(objID, n2_s);
        const uint   v2pk = PackNormal(V2);
        buf.Store4(addr_pack1(pixelIdx), uint4(asuint(xO), PackNormal(nSO)));
        buf.Store4(addr_pay  (pixelIdx),
                   uint4(PackRGB9E5(L2), PackRGB9E5(Kd),
                         PackEtaPrPm(eta, Pr, Pm), matID));

        buf.Store2(addr_pm   (pixelIdx), uint2(objID, v2pk));
        buf.Store (addr_rcinfo(pixelIdx), rcInfo);
        buf.Store3(addr_f    (pixelIdx), asuint(F_contrib));
        buf.Store (addr_v2   (pixelIdx), v2pk);
        buf.Store3(addr_hyb  (pixelIdx), uint3(pathSeed, asuint(cachedJac), asuint(gBase)));
        return true;
    }
    return false;
}

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
