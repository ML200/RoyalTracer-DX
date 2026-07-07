//====================================
//RIS RESERVOIR UNIFIED DI+GI  (PSS / hybrid shift)
//====================================
//matID discriminates path kind
//matID < MATID_LIGHT_TRI, BSDF-sampled GI vertex at x_k, d>=3
//matID == MATID_LIGHT_TRI, NEE/emitter end, x_k is the light position, L2 is emission
//matID == MATID_ENV_MISS, env/sky, x2 is unit direction, L2 is radiance
//
//PSS SEMANTICS: F is the RR-clean PRIMARY-SAMPLE-SPACE contribution f/p_noRR at
//the owning pixel (every sampling pdf except RR survival folded in; suffix pdfs
//pre-divided inside L2). p_hat = lum(F); the pixel estimate stays F*W. The
//reconnection vertex sits at a VARIABLE index k (rcInfo), pinned by the hybrid
//shift criteria; the prefix x_1..x_{k-1} is reproducible from `seed` via the
//per-bounce RNG streams (RcBounceSeed). k==2 shifts reconnect directly from the
//receiver (no replay) — identical structure to the legacy reconnection shift.
struct Reservoir
{
    //constant-after-hit payload
    float3 x2;          //reconnection vertex x_k (world)
    float3 n2_s;
    uint   objID;
    uint   matID;
    float  eta;
    //resolved x_k material - baked by raygen so reconnection never re-fetches
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

    //hybrid-shift PSS state
    uint   rcInfo;      //k | d | RC_F_* flags (PM record pad word)
    uint   seed;        //pathSeed: prefix replay streams via RcBounceSeed
    float  cachedJac;   //base-side jacobian bundle * gBase (case-appropriate, see RcEval)
    float  gBase;       //base-side GEOMETRIC factor of the reconnection segment
};


//====================================
//SOA GROUP PLANES  (80 B/px)
//====================================
//Round 2 of the co-access grouping (was 5 planes / 64 B): scattered partner reads
//(temporal candidate, spmis shift draws, merge lazy accept) always want the FULL
//reconnection payload, so it now lives in ONE 48B AoS record — exactly 2 sectors at
//any pixel (48B never straddles a third 32B sector at a 48B stride) instead of 4-5
//scattered plane sectors + a separate M fetch. FW keeps its own plane (merge streams
//F/W without payload); V2 is DUPLICATED into a solo 4B plane so Pass_dup_gi's
//4B-stride streaming stays cheap (writers store it twice). Own-pixel access is a
//wash (same bytes either way). (The former WSUM plane was write-only — raygen's
//w_sum lives in registers through FinalizeReservoir and nothing ever read it back —
//so it was removed; keep host Reservoir_GI at 80 B, see Renderer.h.)
//  RECON 48B  +0  x2(12) | n2pk(4)               reconnection geometry
//             +16 L2 | Kd | eta/Pr/Pm | matID    reconnection material payload
//             +32 objID | v2pk | M | rcInfo      provenance + confidence + pin info
//  FW    16B  F(12) | W(4)                       RIS state, one Load4 in merge
//  V2     4B  solo DUPLICATE plane               dup pass streams it at 4B stride
//  HYB   12B  seed | cachedJac | gBase           hybrid-shift PSS state (one Load3)
static const uint PLANE_FW    = 48u;
static const uint PLANE_V2    = 64u;
static const uint PLANE_HYB   = 68u;

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
uint addr_rcinfo(uint px)          { return addr_pm(px)   + 12u; }
uint addr_hyb(uint px)             { uint N = numPx(); return N * PLANE_HYB  + px * 12u; }

//====================================
//RC INFO  (pin descriptor, PM record pad word)
//====================================
//k = reconnection-vertex index (x_1 = primary hit; k==2 is the legacy pin, k==0
//means the sample carries NO reusable pin — it still shades canonically but every
//shift to/from it is undefined). d = path length (vertex index of the last vertex).
//Replay length = k-2 bounces.
//Bundle flags say which pdf factors cachedJac carries beside gBase:
//  RC_F_NOPK       bundle lacks p(w_k)      (candidate ends at/leaves the pin via a
//                                            non-BSDF dim: at-pin NEE, k==d ends)
//  RC_F_NOPPREV    bundle lacks p(w_{k-1})  (reconnection segment is not BSDF-sampled)
//  RC_F_PAREA      NEE-end at a light / sun / any invariant-density copy: cachedJac
//                  IS the (approx. position-invariant) source density — the shift
//                  copies it (cachedNew = cachedJac of the source, |J| = 1).
//  RC_F_ENV_REPLAY glossy-tail env end: NO reconnection — the shift replays the
//                  prefix AND re-derives the final BSDF dim from the stream, the
//                  trace must MISS, sky+sun re-evaluated (pure PSS, J = 1,
//                  cachedJac = gBase = 1). Routes through the replay queues even
//                  at k==2 (RcReplayLen = k-1).
#define RC_F_NOPK       (1u << 12)
#define RC_F_NOPPREV    (1u << 13)
#define RC_F_PAREA      (1u << 14)
#define RC_F_ENV_REPLAY (1u << 15)
//RC_F_LOBES — the sample is LOBE-INDEXED (Enhanced supp §1): bits 16-31 carry
//the 2-bit sampled-lobe ids of vertices 1..8 (0 diffuse, 1 GGX, 2 coat,
//3 sheen). Shifts PRESERVE the lobe sequence — replay forces the recorded lobe
//per bounce, and the jacobian bundles/re-evaluations use the CONDITIONAL pdfs
//p(w|l) at the two reconnection slots (the pmf product P(l) lives in the
//source pdf like RR survival, NOT in the jacobian). Samples without the bit
//keep marginal semantics end to end — mixed populations are self-consistent,
//so toggling RS_FLAG_LOBE_PSS needs no reservoir reset. k is capped at 8 for
//lobe samples (mask coverage).
#define RC_F_LOBES      (1u << 10)

//layout: k:0-3 | d:4-9 | LOBES:10 | (11 spare) | RC_F_*:12-15 | lobe ids:16-31
inline uint RcPackInfo(uint k, uint d, uint flags) { return (k & 15u) | ((d & 63u) << 4) | flags; }
inline uint RcK(uint info)        { return info & 15u; }
inline uint RcD(uint info)        { return (info >> 4) & 63u; }
inline bool RcReusable(uint info) { return (info & 15u) >= 2u; }
inline bool RcEnvReplay(uint info){ return (info & RC_F_ENV_REPLAY) != 0u; }
inline bool RcHasLobes(uint info) { return (info & RC_F_LOBES) != 0u; }
//sampled-lobe id of vertex vtx (1-based; meaningful for vertices 1..8 of
//RC_F_LOBES samples — the segment lobe is RcLobeAt(k-1), the continuation
//lobe RcLobeAt(k))
inline uint RcLobeAt(uint info, uint vtx)
{
    return (info >> (16u + ((vtx - 1u) << 1))) & 3u;
}
//generation-side: the mask word a candidate ORs into rcInfo (marker + ids)
inline uint RcLobesWord(uint mask) { return RC_F_LOBES | ((mask & 0xFFFFu) << 16); }
//ROUTING metric: nonzero routes the shift through the replay queues/passes.
//Env-replay adds its re-derived final dim so it queues even at k==2; the
//ReplayPrefix WALK length is rcK-2 regardless (the tail dim belongs to
//EnvReplayEval, never the prefix loop).
inline uint RcReplayLen(uint info)
{
    const uint k = RcK(info);
    if (k < 2u) return 0u;
    return (k - 2u) + (RcEnvReplay(info) ? 1u : 0u);
}

//====================================
//HYBRID PIN CRITERIA  (generation-side ONLY)
//====================================
//Raygen alone decides the pin vertex; reuse carries it in rcInfo, and every shift
//is the FIXED map T_k for that sample — bijective without any offset-side criteria
//re-derivation, so none exists. Two criteria families, RS_FLAG_RC_FOOTPRINT picks:
//
//v1 (footprint OFF): material roughness both sides + minimum segment distance.
inline bool RcRoughPass(float Pr) { return Pr >= rs_reconnectRoughnessMin; }

inline bool RcCritPair(float PrPrev, float PrHit, bool hitIsEndOrVolume, float segDist, float distMin)
{
    if (!RcRoughPass(PrPrev)) return false;
    if (!hitIsEndOrVolume && !RcRoughPass(PrHit)) return false;
    return segDist >= distMin;
}

//Enhanced §4 (footprint ON):
//  * pdf-proxy glossiness guard (supplemental §4, Eq. 26): 1/p(w_{k-1})^2 >=
//    sigma_min — lobe-aware for multilobe/black-box materials (a diffuse-lobe
//    bounce on a plastic passes even when the material Pr is low). Reuses the
//    rs_reconnectRoughnessMin slider as sigma_min.
//  * dual footprint threshold (Eq. 5, literal): both the forward area density
//    p(w_{k-1})*G(x_{k-1}->x_k) and the inverse p(w_k)*G(x_k->x_{k-1}) must stay
//    below 1/threshold, threshold = (c/100) * ||x0-x1||^2 / (cos/(4pi)).
//    The forward half gates the pin at the hit; the inverse half is only known at
//    the NEXT BSDF sample -> raygen pins TENTATIVELY and revokes at the p(w_k)
//    fold (candidates stored before the fold keep the tentative pin — their
//    continuation is a non-BSDF dim, so the inverse test does not apply to them).
//    Subsumes the distance criterion (short segments blow up G).
inline bool RcLobeProxyPass(float pdf)
{
    return pdf * pdf * rs_reconnectRoughnessMin <= 1.0f;
}

//per-pixel Eq.5 threshold (the RHS); test densities against 1/thresh
inline float RcFpThreshold(float camDist2, float cosPrim)
{
    return (rs_rcFpKappa * 0.01f) * camDist2 * (4.0f * PI) / max(cosPrim, 1e-4f);
}

inline bool RcFpDensityPass(float pdf, float G, float fpThresh)
{
    return pdf * G * fpThresh <= 1.0f;
}

//====================================
//GEOMETRIC OUTLIER BAND  (PSS)
//====================================
//The resampling weight is lum(c*vis)*Jn/cachedJac*W; its pdf factors are exact
//and must NEVER be clamped (clamping them is a systematic energy bias — the
//force-to-1 band may only touch geometry). The band therefore acts on the
//GEOMETRIC ratio Jn/gBase alone. Reject variant for the spatial pool, scale
//variant (clamp to [1/T,T], T=temp_jacClamp) for the temporal candidate.
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

//NOTE: there is deliberately NO reuse-time roughness/distance gate under the
//hybrid shift — shifts are the fixed per-sample maps T_k (bijective without
//offset-side criteria), glossy receivers self-gate through the BSDF magnitude
//in the shifted target, and geometric singularities are handled by the band
//above. Full spatial/temporal reuse across all roughnesses is intended.

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
//FUSED lobe-or-marginal variant: ONE EvaluateAndPdf_COMBINED_L walk yields the
//marginal pair (its res) AND the recorded lobe's latched pair (documented to
//match EvaluateLobePdf_COMBINED exactly — same formulas, same EPSILON skips) —
//the output select picks per useLobe. Value-identical to the old
//BSDF_term_pdf / BSDF_term_lobe if/else pairs, but the material stack inlines
//ONCE per call site instead of twice: those pairs were the dominant code mass
//of every reconnection-bearing pass (I$ / noinstr pressure). Lobe absent at
//this vertex -> the latch never fires -> {0,0}, the shift-undefined signal,
//exactly like EvaluateLobePdf_COMBINED. (Slight ALU trade on lobe samples:
//the old lobe walk early-outed past the target lobe, the fused walk always
//evaluates every present lobe — code footprint beats the saved lanes here.)
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
    const SamplingP p = CalculateStrategyProbabilities(mID, o, n_s, etai, etat, localKd, localPm);
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
    buf.Store4(addr_pm(pixelIdx), uint4(r.objID, v2pk, r.M, r.rcInfo));
    buf.Store4(addr_fw(pixelIdx), uint4(asuint(r.F), asuint(r.W)));
    buf.Store (addr_v2(pixelIdx), v2pk);   //solo duplicate for the dup pass stream
    buf.Store3(addr_hyb(pixelIdx), uint3(r.seed, asuint(r.cachedJac), asuint(r.gBase)));
    //w_sum is transient RIS state (raygen registers / merge-local) — never stored
}

//payload-only load: everything the shift eval needs (x2/n2/L2/V2/Kd/Pr/Pm/eta/
//matID/objID + rcInfo/seed/cachedJac/gBase) in 4 grouped fetches; F/W/M/w_sum
//of 'r' are left untouched — merge's lazy winner accept relies on keeping its
//own accumulated RIS state.
void loadReservoirPayload(RWByteAddressBuffer buf, uint pixelIdx, inout Reservoir r)
{
    const uint4 p1  = buf.Load4(addr_pack1(pixelIdx));
    const uint4 pay = buf.Load4(addr_pay(pixelIdx));
    const uint4 pm  = buf.Load4(addr_pm(pixelIdx));   // objID | v2pk | M | rcInfo
    const uint3 hyb = buf.Load3(addr_hyb(pixelIdx));  // seed | cachedJac | gBase

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

//Buffer-selected payload load for the single-call-site shift eval
//(HybridShiftEval_post): the temporal pass shifts FROM g_Reservoirs_last
//(forward) and FROM g_Reservoirs_current (reverse) through one inlined post
//body — a per-call buffer PARAMETER would specialize the whole body per
//buffer and double it again. Pass literals where the buffer is fixed (the
//spatial pass always resolves against current) so the dead half folds.
void loadReservoirPayloadSel(bool fromLast, uint pixelIdx, inout Reservoir r)
{
    if (fromLast) loadReservoirPayload(g_Reservoirs_last,    pixelIdx, r);
    else          loadReservoirPayload(g_Reservoirs_current, pixelIdx, r);
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

uint load_rcInfo(RWByteAddressBuffer b, uint pixelIdx)
{
    return b.Load(addr_rcinfo(pixelIdx));
}

uint load_seed(RWByteAddressBuffer b, uint pixelIdx)
{
    return b.Load(addr_hyb(pixelIdx));
}

//RIS + hybrid PSS state WITHOUT the 48B payload record: what the replay passes
//re-load AFTER a prefix walk. HybridShiftEval consumes the payload internally
//post-walk; carrying a full Reservoir across the walk instead would ride every
//SER reorder (see Hybrid_Replay_v8.hlsli's live-state contract).
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


//====================================
//PER-FIELD LOADS AND STORES
//====================================
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

//source-side geometric factor alone (the pre-visibility reject hint of the
//direct shift paths — see HybridShiftEval_post's gBaseHint)
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

//====================================
//REJECTION AND VALIDITY
//====================================
inline bool IsValidReservoir(Reservoir r) {
    return any(abs(r.n2_s) > 0.0f) && r.M > 0;
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

//(ComputeJcVol + JacobianRatio / JacobianRatioRej removed: the volume-vertex 1/d²
//geometry is derived inline in ReconnectPSS's volume branch, and the legacy
//full-jacobian clamp/reject pair is superseded by the PSS GEOMETRIC band —
//RcGeomClampScale / RcGeomReject above — which bands Jn/gBase only and never
//touches pdf factors.)

//====================================
//RECONNECTION  (PSS shift evaluation)
//====================================
//Evaluates the reconnection of the receiving vertex (x1 = the receiver's primary
//hit for k==2, or the REPLAYED offset vertex y_{k-1} for k>2) to the reservoir's
//pin vertex. Outputs, per the uniform PSS shift rules:
//  return  c          shifted numerator: full f product around the reconnection
//                     (prefix throughput NOT included — the caller multiplies its
//                     replayed prefix), WITHOUT pdf divisions, WITHOUT Jn
//  Jn                 new-side GEOMETRIC factor of the reconnection segment
//                     (surface cos/d² | volume 1/d² | env 1 | light cosL/d²)
//  cachedNew          new-side jacobian bundle * Jn. The uniform reuse algebra:
//                       resampling weight = m * lum(c*vis) * Jn / cachedJac_src * W
//                       accept re-anchor:  F = c*vis * Jn / cachedNew
//                       outlier band:      Jn / gBase_src ONLY (pdfs never banded)
//                     Bundle contents follow rcInfo: {p(w_{k-1})}·{p(w_k)} for
//                     continuations, dropped per RC_F_NOPK/RC_F_NOPPREV; RC_F_PAREA
//                     (NEE light end) copies the source's invariant area density.
//etai1/etat1 are the IOR pair at x1 per raygen's hinfo.backface derivation
inline float3 ReconnectPSS(
    //x1 receiving vertex (receiver primary hit or replayed y_{k-1})
    in float3  x1,
    in float3  n1_s,
    in float3  o,
    in uint    mID1,
    in float3  localKd1,
    in float   localPr1,
    in float   localPm1,
    in float   etai1,
    in float   etat1,

    //reservoir pin vertex payload
    in uint    mID2,
    in float3  x2,
    in float3  n2_s,
    in float3  L2,
    in float3  V2,
    in float3  localKd2,
    in float   localPr2,
    in float   localPm2,
    in float   eta2,

    //pin descriptor (bundle shape + source-side invariant copy)
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

    //x1 is an SSS surface: it reflects via its regular BRDF; only when the path ENTERED
    //the medium (the volume reconnection vertex in the GI branch) is x1's coupling the
    //diffuse entry term 1/pi.
    const bool sss1 = LoadIsSSS(mID1);

    //supp §1 lobe preservation: RC_F_LOBES samples evaluate the RECORDED lobe
    //(single-lobe rho + conditional pdf) at the two jacobian slots — the
    //reconnection segment's lobe l_{k-1} unless the segment is a non-BSDF dim
    //(RC_F_NOPPREV / PAREA / SSS exit), the continuation's lobe l_k unless the
    //pin leaves via a non-BSDF dim (RC_F_NOPK). Those exempt dims stay
    //full-BSDF ("lobe 0 = all lobes" in the paper's indexing).
    const bool useLobes = RcHasLobes(rcInfo);
    const uint rcKk     = RcK(rcInfo);

    //====================================
    //DI ENV SKY  (direction payload: BSDF-miss dir-copy or sun-NEE copy)
    //====================================
    //x2 is direction, no G, no BSDF at x2, Jn=1.
    //  BSDF dir-copy: the bundle is the receiver-side marginal pdf of the
    //    stored direction (position-dependent).
    //  RC_F_PAREA (sun/NEE-sampled direction): the sampling density is globally
    //    position-INVARIANT — copy the source bundle, |J| = 1.
    //(RC_F_ENV_REPLAY candidates never reach this eval — they route through
    //EnvReplayEval in the replay passes.)
    //env treated as infinitely far, no medium absorption applied
    if (mID2 == MATID_ENV_MISS)
    {
        const float3 wi  = normalize(x2);
        //dir-copy (BSDF dim): lobe samples re-evaluate the RECORDED escaping
        //lobe; PAREA (sun-NEE) is an all-lobes dim -> full BSDF either way.
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

    //====================================
    //LIGHT VERTEX END (k==d at an emissive triangle)
    //====================================
    //x2 is light position, n2_s is light normal, L2 is emission.
    //  NEE end (RC_F_PAREA): the light point was area-sampled; its (approx.
    //    position-invariant) area density IS the source's cachedJac — copy it,
    //    which makes the PSS jacobian exactly 1 in area measure.
    //  BSDF end (emitter hit): the segment is a BSDF dim — bundle = p'(w)·Jn.
    if (mID2 == MATID_LIGHT_TRI)
    {
        const float3 dirT  = x2 - x1;
        const float  distT = length(dirT);
        if (distT < EPSILON) return 0.0f;
        const float3 ndirNT = normalize(-dirT);

        //BSDF-end (emitter hit): the segment lobe is recorded; NEE-end (PAREA)
        //is an all-lobes dim -> full BSDF.
        const BrdfData bd1 = BSDF_term_sel(useLobes && !(rcInfo & RC_F_PAREA),
                                           RcLobeAt(rcInfo, rcKk - 1u),
                                           mID1, n1_s, n1_s, -ndirNT, o,
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

        float3 r = bd1.val * L2 * G1 * transmittance;
        if (any(isnan(r)) || any(isinf(r))) r = 0.0f;

        Jn = max(abs(dot(ndirNT, n2_s)) / (distT * distT), EPSILON);
        cachedNew = (rcInfo & RC_F_PAREA) ? srcCachedJac
                                          : max(bd1.pdf, EPSILON) * Jn;
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
    //PSS bundle mirrors the walk's entry pdf chain exactly (raygen's pdfFac):
    //cos-hemisphere entry (G1/pi) * free flight (sigma_t*exp(-sigma_t*d)) * HG
    //phase pdf (== the phase value). A non-SSS receiver has no entry dims to map
    //-> shift undefined (the legacy build approximated this; PSS cannot).
    if (volume2)
    {
        if (!sss1)
            return 0.0f;
        const float3 F1v = localKd1 * SSS_INV_PI;                //SSS entry coupling tinted by surface albedo

        const float  radius  = max(LoadSSSRadius(baseID2), SSS_MIN_RADIUS);
        const float  sigma_t = 1.0f / radius;
        const float3 albedo  = saturate(LoadSSSAlbedo(baseID2));
        const float  gg      = LoadPhaseG(baseID2);
        const float  cosP    = dot(-ndirN, V2);                  //arriving (x1->S1) vs outgoing V2
        const float  phase   = EvaluatePhaseHG(gg, cosP);
        const float3 F2v     = sigma_t * albedo * phase;
        const float3 Tv      = (float3)exp(-sigma_t * dist);     //x1->S1 transmittance

        Jn = max(1.0f / (dist * dist), EPSILON);                 //volume: no cos at x2
        cachedNew = max((G1 * SSS_INV_PI)                        //entry-dir pdf (cos-hemisphere)
                        * (sigma_t * exp(-sigma_t * dist))       //free-flight pdf to S1
                        * phase                                  //HG phase pdf (self-normalized)
                        * Jn, EPSILON);
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
    //The exit path's walk is analog (bundle == 1); its rcInfo carries NOPK|NOPPREV so
    //the pdf factors below drop out.
    const bool exit2 = IsSSSExitVertex(mID2);
    float  pdf1 = 0.0f;
    float3 F1;
    if (exit2) {
        F1 = localKd1 * SSS_INV_PI;                               //entry coupling tinted by x1 surface albedo
    } else {
        //reconnection segment: single recorded lobe when BSDF-sampled
        const BrdfData bd1 = BSDF_term_sel(useLobes && !(rcInfo & RC_F_NOPPREV),
                                           RcLobeAt(rcInfo, rcKk - 1u),
                                           mID1, n1_s, n1_s, -ndirN, o,
                                           localKd1, localPr1, localPm1, etai1, etat1);
        F1   = bd1.val;
        pdf1 = bd1.pdf;
    }
    //continuation at the pin: single recorded lobe when it left via a BSDF dim
    //(at-pin NEE / SSS-entry pins carry RC_F_NOPK -> full BSDF toward V2)
    const BrdfData bd2 = BSDF_term_sel(useLobes && !(rcInfo & RC_F_NOPK),
                                       RcLobeAt(rcInfo, rcKk),
                                       baseID2, n2_s, n2_s, -V2, ndirN,
                                       localKd2, localPr2, localPm2, etai2, etat2);
    const float3 F2 = bd2.val;

    float  G2  = G_term(n2_s, -V2);

    //Beer-Lambert applied at most once, same medium bounds both sides
    float3 transmittance = float3(1.0f, 1.0f, 1.0f);
    if (x1_inMedium) {
        transmittance = CalculateAbsorptionThroughput(LoadTf(mID1), dist);
    } else if (x2_inMedium) {
        transmittance = CalculateAbsorptionThroughput(LoadTf(baseID2), dist);
    }

    Jn = max(abs(dot(ndirN, n2_s)) / (dist * dist), EPSILON);

    //new-side jacobian bundle per the pin descriptor: p'(w_{k-1}) for a BSDF-sampled
    //reconnection segment, p'(w_k) for a BSDF-sampled continuation at the pin.
    float bundle = 1.0f;
    if (!(rcInfo & RC_F_NOPPREV)) bundle *= max(pdf1,    EPSILON);
    if (!(rcInfo & RC_F_NOPK))    bundle *= max(bd2.pdf, EPSILON);
    cachedNew = bundle * Jn;

    float3 r = F1 * F2 * L2 * G1 * G2 * transmittance;

    if (any(isnan(r)) || any(isinf(r)) || all(r < EPSILON))
        r = (float3)0.0f;

    return max(r,0.0f);
}

//SurfaceVertex convenience wrapper — every reuse call site holds one.
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



//====================================
//RESERVOIR UPDATE  (shift accept = PSS re-anchor)
//====================================
//On accepting a shifted neighbour sample the reservoir becomes the OWNER of
//that sample at this pixel: the payload + replay identity (rcInfo/seed) copy
//over, while F re-anchors to the SHIFTED contribution and the jacobian cache
//re-anchors to the NEW side (cachedJac = cachedNew, gBase = Jn) so the next
//reuse of this reservoir measures its bundle against this pixel's prefix.
bool UpdateReservoir(
    inout Reservoir reservoir,
    in float wi,
    in uint  M,
    in Reservoir src,        //payload + rcInfo/seed source (the neighbour)
    in float3 F_shifted,     //c*vis*Jn/cachedNew — the re-anchored PSS contribution
    in float  cachedNew,     //new-side jacobian bundle*Jn
    in float  Jn,            //new-side geometric factor
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


//====================================
//INITIAL RESAMPLING CANDIDATE
//====================================
//wsum lives in a register, on acceptance writes full reservoir payload to SoA buffer
//F stored as full RGB (PSS contribution f/p_noRR), target magnitude is GetPHat(F)
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
        //objID + v2 land in the record; M is finalize-owned so Store2 only.
        //F only - W and M land at raygen finalize
        buf.Store2(addr_pm   (pixelIdx), uint2(objID, v2pk));
        buf.Store (addr_rcinfo(pixelIdx), rcInfo);
        buf.Store3(addr_f    (pixelIdx), asuint(F_contrib));
        buf.Store (addr_v2   (pixelIdx), v2pk);   //solo duplicate for the dup pass stream
        buf.Store3(addr_hyb  (pixelIdx), uint3(pathSeed, asuint(cachedJac), asuint(gBase)));
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
