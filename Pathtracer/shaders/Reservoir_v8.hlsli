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
//SOA GROUP PLANES  (72 B/px)
//====================================
//Round 2 of the co-access grouping (was 5 planes / 64 B): scattered partner reads
//(temporal candidate, spmis shift draws, merge lazy accept) always want the FULL
//reconnection payload, so it now lives in ONE 48B AoS record — exactly 2 sectors at
//any pixel (48B never straddles a third 32B sector at a 48B stride) instead of 4-5
//scattered plane sectors + a separate M fetch. FW keeps its own plane (merge streams
//F/W without payload); V2 is DUPLICATED into a solo 4B plane so Pass_dup_gi's
//4B-stride streaming stays cheap (writers store it twice); wsum stays raygen-owned
//in its own plane. Own-pixel access is a wash (same bytes either way).
//  RECON 48B  +0  x2(12) | n2pk(4)               reconnection geometry
//             +16 L2 | Kd | eta/Pr/Pm | matID    reconnection material payload
//             +32 objID | v2pk | M | pad         provenance + confidence
//  FW    16B  F(12) | W(4)                       RIS state, one Load4 in merge
//  V2     4B  solo DUPLICATE plane               dup pass streams it at 4B stride
//  WSUM   4B  raygen-owned
static const uint PLANE_FW    = 48u;
static const uint PLANE_V2    = 64u;
static const uint PLANE_WSUM  = 68u;

//====================================
//SOA ADDRESS HELPERS
//====================================
//tile-aligned pixel count, must match MapPixelID's 8-wide x 4-tall tile
//swizzle (Common_v8.hlsli) and the CPU-side TileAlignedPx (Renderer.h).
//The old 4x8-shaped formula diverged from MapPixelID's actual max index at
//non-tile-divisible resolutions, overlapping the SoA planes.
uint numPx()                       { return ((IMG_W + 7u) / 8u) * ((IMG_H + 3u) / 4u) * 32u; }

//group bases
uint addr_recon(uint px)           { return px * 48u; }
uint addr_pack1(uint px)           { return addr_recon(px); }
uint addr_pay  (uint px)           { return addr_recon(px) + 16u; }
uint addr_pm   (uint px)           { return addr_recon(px) + 32u; }   // objID | v2pk | M | pad
uint addr_fw   (uint px)           { uint N = numPx(); return N * PLANE_FW   + px * 16u; }
uint addr_v2   (uint px)           { uint N = numPx(); return N * PLANE_V2   + px *  4u; }

//per-field addresses inside the groups (legacy helper API preserved)
uint addr_l2(uint px)              { return addr_pay(px); }
uint addr_kd(uint px)              { return addr_pay(px)  +  4u; }
uint addr_eta(uint px)             { return addr_pay(px)  +  8u; }
uint addr_matid(uint px)           { return addr_pay(px)  + 12u; }
uint addr_f(uint px)               { return addr_fw(px); }
uint addr_w(uint px)               { return addr_fw(px)   + 12u; }
uint addr_objid(uint px)           { return addr_pm(px); }
uint addr_m(uint px)               { return addr_pm(px)   +  8u; }
uint addr_wsum(uint px)            { uint N = numPx(); return N * PLANE_WSUM + px * 4u; }

//luminance
inline float GetPHat(float3 v) {
    return 0.2126f * v.x + 0.7152f * v.y + 0.0722f * v.z;
}

//Finalize the unbiased contribution weight W = w_sum / p_hat(selected), BOUNDED.
//W is ~1/path-pdf so it can be legitimately large, but when the selected sample's
//target luminance p_hat is small at this pixel (grazing / occluded / dim — i.e.
//close to a surface) while w_sum is not, W spikes. In temporal/spatial reuse that
//spike feeds back every frame (w_n ~ neighbour.W), growing W without bound into a
//diverging firefly. Clamp to wMax to break the feedback (wMax editor-tunable;
//<= 0 disables). Raygen's single-frame UCW stays unclamped so small/far direct
//lights aren't darkened.
inline float FinalizeUCW(float w_sum, float p_hat, float wMax)
{
    if (!(p_hat > EPSILON) || !(w_sum > 0.0f)) return 0.0f;
    const float W = w_sum / p_hat;
    if (isnan(W) || isinf(W) || W < 0.0f) return 0.0f;
    return (wMax > 0.0f) ? min(W, wMax) : W;
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
    const uint v2pk = PackNormal(r.V2);

    //no outer normalize: PackNormal normalizes internally, and its zero/NaN
    //guard must see the raw vector so a cleared reservoir packs the
    //invalid-normal sentinel instead of garbage bits
    buf.Store4(addr_pack1(pixelIdx), uint4(asuint(xO), PackNormal(nSO)));
    buf.Store4(addr_pay(pixelIdx),
               uint4(PackRGB9E5(r.L2), PackRGB9E5(r.Kd),
                     PackEtaPrPm(r.eta, r.Pr, r.Pm), r.matID));
    buf.Store4(addr_pm(pixelIdx), uint4(r.objID, v2pk, r.M, 0u));
    buf.Store4(addr_fw(pixelIdx), uint4(asuint(r.F), asuint(r.W)));
    buf.Store (addr_v2(pixelIdx), v2pk);   //solo duplicate for the dup pass stream
    //wsum is raygen-owned and not part of the merged record
}

//payload-only load: everything Reconnect needs (x2/n2/L2/V2/Kd/Pr/Pm/eta/
//matID/objID) in 3 grouped fetches over the ONE 48B record (2 sectors at any
//pixel); F/W/M/w_sum of 'r' are left untouched — merge's lazy winner accept
//relies on keeping its own accumulated RIS state.
void loadReservoirPayload(RWByteAddressBuffer buf, uint pixelIdx, inout Reservoir r)
{
    const uint4 p1  = buf.Load4(addr_pack1(pixelIdx));
    const uint4 pay = buf.Load4(addr_pay(pixelIdx));
    const uint4 pm  = buf.Load4(addr_pm(pixelIdx));   // objID | v2pk | M | pad

    r.objID = pm.x;
    r.matID = pay.w;

    r.x2    = ObjectToWorldPos(r.objID, asfloat(p1.xyz));
    r.n2_s  = ObjectToWorldNrm(r.objID, UnpackNormal(p1.w));

    r.L2    = UnpackRGB9E5(pay.x);
    r.Kd    = UnpackRGB9E5(pay.y);
    UnpackEtaPrPm(pay.z, r.eta, r.Pr, r.Pm);

    r.V2    = UnpackNormal(pm.y);
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

//volume reconnection vertex (first in-medium SSS scatter): no surface normal, so the
//geometry term drops the cosine and is a pure inverse-square. Callers select this when
//IsVolumeVertex(reservoir.matID).
inline float ComputeJcVol(float3 x1, float3 x2)
{
    float3 d = x1 - x2;
    float  dist2 = dot(d, d);
    return max(1.0f / max(dist2, EPSILON), EPSILON);
}

//Reconnection-shift Jacobian ratio for temporal reuse, CLAMPED (not rejected) to
//[1/T, T]. Near a grazing corner the reconnection cos -> 0 makes ComputeJc tiny
//and Jn/Jc blows up; the clamp bounds it so the resampling weight can't spike into
//fireflies. T is editor-tunable (ReSTIRSettings::tempJacClamp): lower = stronger
//suppression / more bias. (SPMIS uses JacobianRatioRej, which rejects outliers
//instead, because it draws from a large pool; the small temporal candidate set
//prefers a bounded sample over a dropped one.)
inline float JacobianRatio(float Jn, float Jc, float T)
{
    //Jc/Jn are both floored to EPSILON in ComputeJc, so at the floor the correct
    //ratio is EPSILON/EPSILON = 1, NOT 0 — we must NOT reject there (that silently
    //killed long-range temporal reuse). Guard div-by-zero / non-finite, then clamp.
    if (Jc <= 0.0f) return 0.0f;
    const float j = Jn / Jc;
    if (isnan(j) || isinf(j)) return 0.0f;
    const float t = max(T, 1.0f);
    return clamp(j, 1.0f / t, t);
}

//Reconnection-shift Jacobian WITH outlier rejection (reject band ~[1/T, T], T~15):
//returns 0 - i.e. REJECTS the shifted sample - when Jn/Jc is extreme (> T or < 1/T)
//or non-finite. An unbounded Jn/Jc spikes the resampling weight into FIREFLIES; the
//cell path hits this constantly because it reuses from a large pool where some
//reconnection vertices are grazing (cos->0) or at a large distance ratio. Applied
//per non-canonical neighbour in the SPMIS pass; the plain JacobianRatio above does
//not reject. Slight bias for a large variance win (standard ReSTIR-PT practice).
inline float JacobianRatioRej(float Jn, float Jc, float T)
{
    if (Jc <= EPSILON) return 0.0f;
    float j = Jn / Jc;
    if (isnan(j) || isinf(j) || j > T || j < (1.0f / T)) return 0.0f;
    return j;
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

    //x1 is an SSS surface: it reflects via its regular BRDF; only when the path ENTERED
    //the medium (the volume reconnection vertex in the GI branch) is x1's coupling the
    //diffuse entry term 1/pi.
    const bool sss1 = LoadIsSSS(mID1);

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
        //thin glass is a single interface with no interior -> never apply Beer-Lambert.
        const bool  m1_inMedium = (iorAfterX1 > 1.0f + EPSILON)
                                  && (LoadKd_w(mID1) < 1.0f - EPSILON)
                                  && !LoadIsThinGlass(mID1);
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
    float3 ndirN = normalize(-dir);          //from x2 toward x1

    const uint  baseID2 = MatIDBase(mID2);
    const bool  volume2 = IsVolumeVertex(mID2);
    const float G1      = G_term(n1_s, -ndirN);

    //====================================
    //VOLUME RECONNECTION VERTEX (first in-medium SSS scatter)
    //====================================
    //F2 = scattering coefficient * Henyey-Greenstein phase; the connecting segment
    //carries Beer-Lambert transmittance; the geometry term drops the x2-side cosine
    //(no surface there). The shift never re-enters the walk.
    if (volume2)
    {
        const float3 F1v = sss1
            ? (localKd1 * SSS_INV_PI)                             //SSS entry coupling tinted by surface albedo
            : BSDF_term(mID1, n1_s, n1_s, -ndirN, o, localKd1, localPr1, localPm1, etai1, etat1);

        const float  radius  = max(LoadSSSRadius(baseID2), SSS_MIN_RADIUS);
        const float  sigma_t = 1.0f / radius;
        const float3 albedo  = saturate(LoadSSSAlbedo(baseID2));
        const float  gg      = LoadPhaseG(baseID2);
        const float  cosP    = dot(-ndirN, V2);                  //arriving (x1->S1) vs outgoing V2
        const float3 F2v     = sigma_t * albedo * EvaluatePhaseHG(gg, cosP);
        const float3 Tv      = (float3)exp(-sigma_t * dist);     //x1->S1 transmittance

        Jn = max(1.0f / (dist * dist), EPSILON);                 //volume: no cos at x2
        float3 rv = F1v * F2v * L2 * G1 * Tv;                    //G2 == 1
        if (any(isnan(rv)) || any(isinf(rv)) || all(rv < EPSILON))
            rv = (float3)0.0f;
        return max(rv, 0.0f);
    }

    //====================================
    //SURFACE GI VERTEX (incl. white-baked SSS entry surfaces)
    //====================================
    //recover x2 IOR pair from stored etat and material Ni, disambiguate on midpoint.
    //baseID2 strips any SSS provenance bit so g_mat[] indexing is in range.
    const float matNi2 = LoadNi(baseID2);
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
    const float m2_Kd_w = LoadKd_w(baseID2);
    //thin glass is a single interface with no interior -> excluded from medium / Beer-Lambert.
    const bool m1_transmissive = m1_Kd_w < 1.0f - EPSILON && !LoadIsThinGlass(mID1);
    const bool m2_transmissive = m2_Kd_w < 1.0f - EPSILON && !LoadIsThinGlass(baseID2);

    const bool x1_inMedium = (iorAfterX1  > 1.0f + EPSILON) && m1_transmissive;
    const bool x2_inMedium = (iorBeforeX2 > 1.0f + EPSILON) && m2_transmissive;

    //if segment inside medium, override incident IOR at x2
    if      (x1_inMedium) etai2 = iorAfterX1;
    else if (x2_inMedium) etai2 = iorBeforeX2;

    //A no-scatter SSS pass-through stores its exit B here with MATID_SSS_EXIT_BIT: the
    //light ENTERED at x1, so x1's coupling is the diffuse entry term 1/pi, not its
    //reflective BRDF (this is what the forward path folded into pdf_product). Without
    //this the reuse target would disagree with the stored F -> ReSTIR bias in thin areas.
    float3 F1 = IsSSSExitVertex(mID2)
        ? (localKd1 * SSS_INV_PI)                                 //entry coupling tinted by x1 surface albedo
        : BSDF_term(mID1, n1_s, n1_s, -ndirN, o,  localKd1, localPr1, localPm1, etai1, etat1);
    float3 F2 = BSDF_term(baseID2, n2_s, n2_s, -V2, ndirN, localKd2, localPr2, localPm2, etai2, etat2);

    float  G2  = G_term(n2_s, -V2);

    //Beer-Lambert applied at most once, same medium bounds both sides
    float3 transmittance = float3(1.0f, 1.0f, 1.0f);
    if (x1_inMedium) {
        transmittance = CalculateAbsorptionThroughput(LoadTf(mID1), dist);
    } else if (x2_inMedium) {
        transmittance = CalculateAbsorptionThroughput(LoadTf(baseID2), dist);
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
        const float3 xO   = WorldToObjectPos(objID, x2);
        const float3 nSO  = WorldToObjectNrm(objID, n2_s);
        const uint   v2pk = PackNormal(V2);
        buf.Store4(addr_pack1(pixelIdx), uint4(asuint(xO), PackNormal(nSO)));
        buf.Store4(addr_pay  (pixelIdx),
                   uint4(PackRGB9E5(L2), PackRGB9E5(Kd),
                         PackEtaPrPm(eta, Pr, Pm), matID));
        //objID + v2 land in the record; M is finalize-owned so Store2 only.
        //F only - W and M land at raygen finalize
        buf.Store2(addr_pm   (pixelIdx), uint2(objID, v2pk));
        buf.Store3(addr_f    (pixelIdx), asuint(F_contrib));
        buf.Store (addr_v2   (pixelIdx), v2pk);   //solo duplicate for the dup pass stream
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
