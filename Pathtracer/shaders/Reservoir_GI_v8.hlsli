// ─────────────────────────────────────────────────────────────────────
//  Reconnection / path-type encoding for Reservoir_GI::method_gi
//
//  For each of the 4 terminal types (env / emitter / NEE / Sun),
//  the reservoir stores either:
//    • a path with a valid reconnection vertex x_k     (plain method)
//    • a path with NO reconnection vertex              (..._RP = "replay")
//
//  Hybrid-shift behaviour by method:
//    BSDF_ENV     : reconnect x1' → x_k, tail replays suffix to env miss
//    BSDF_EMIT    : reconnect x1' → x_k, tail replays suffix to emitter hit
//    NEE          : reconnect x1' → x_k (x_k is a surface), tail = NEE link
//    SUN          : reconnect x1' → x_k (x_k is a surface), tail = sun link
//    BSDF_ENV_RP  : replay x0→x1'→… with stored seed;  shift ok iff misses
//    BSDF_EMIT_RP : replay x0→x1'→… with stored seed;  shift ok iff hits emitter
//    NEE_RP       : replay x0→x1'→…x_{k-1}', then connect to stored emitter
//    SUN_RP       : replay x0→x1'→…x_{k-1}', then connect via stored sun dir
// ─────────────────────────────────────────────────────────────────────
static const uint RC_METHOD_INVALID      = 0u;
static const uint RC_METHOD_BSDF_ENV     = 1u;
static const uint RC_METHOD_BSDF_EMIT    = 2u;
static const uint RC_METHOD_NEE          = 3u;
static const uint RC_METHOD_SUN          = 4u;
static const uint RC_METHOD_BSDF_ENV_RP  = 5u;
static const uint RC_METHOD_BSDF_EMIT_RP = 6u;
static const uint RC_METHOD_NEE_RP       = 7u;
static const uint RC_METHOD_SUN_RP       = 8u;
// "Reconnection vertex IS the NEE source" variants.  Used when the criterion
// fires at the same iteration as the NEE/Sun event, so x_k = current vertex
// (= NEE source) rather than an earlier BSDF-walk vertex.  The tail is just
// the direct NEE link to the light, with V2 pointing toward the light and
// the light's sampling pdf stored in J_gi.x so the shift can substitute it
// for the (degenerate) BSDF pdf at x_k in ReconnectGI.  Shift replay length
// = k-2 exactly like HAS_RECON (so k=2 at depth=1 → zero replay bounces).
static const uint RC_METHOD_NEE_ATRC     = 9u;
static const uint RC_METHOD_SUN_ATRC     = 10u;

inline bool RC_HasReconVertex(uint m) { return (m >= RC_METHOD_BSDF_ENV     && m <= RC_METHOD_SUN) || m == RC_METHOD_NEE_ATRC || m == RC_METHOD_SUN_ATRC; }
inline bool RC_IsReplay      (uint m) { return m >= RC_METHOD_BSDF_ENV_RP  && m <= RC_METHOD_SUN_RP; }
inline bool RC_IsAtRc        (uint m) { return m == RC_METHOD_NEE_ATRC     || m == RC_METHOD_SUN_ATRC; }

static const float RC_ALPHA_MIN   = 0.1f;     // single-vertex roughness threshold (α_min)
static const float RC_C_OVER_100  = 0.0002f;  // c = 0.02 in the paper

// Eq. 5: dual ray-footprint + single-vertex roughness criterion.
// All quantities are at the candidate link x_{k-1} -> x_k.
//   pdf_km1  : p^x_{k-1}(ω_{k-1})    solid-angle BSDF pdf at x_{k-1} toward x_k
//   pdf_k    : p^x_k    (ω_k)        solid-angle BSDF pdf at x_k toward x_{k+1}
//                                    (for NEE/sun terminals, pass lightPdf here;
//                                     ω_k is degenerate but the footprint still bounds)
//   cos_xk   : |n_xk    · (-ω_{k-1})|
//   cos_xkm1 : |n_xkm1  · ( ω_{k-1})|
//   dist2    : ||x_k - x_{k-1}||^2
//   alpha_km1: effective GGX α at x_{k-1}.  Paper uses GGX α directly; for the
//              layered/non-parametric BSDF in this codebase the caller passes
//              max(Pr², Pdiff+Psheen) so purely-diffuse surfaces (Pr=0) read as
//              "fully rough" and pass the α≥α_min safeguard (paper §4.2 supplemental).
//   primary_fp_thresh : RHS of Eq. 5, precomputed at depth 0.
inline bool HybridReconnectionCriterion(
    float pdf_km1, float pdf_k,
    float cos_xk,  float cos_xkm1,
    float dist2,
    float alpha_km1,
    float primary_fp_thresh)
{
    if (alpha_km1 < RC_ALPHA_MIN) return false;

    float fwd_area_pdf = pdf_km1 * cos_xk   / max(dist2, 1e-20f);   // area density at x_k
    float bwd_area_pdf = pdf_k   * cos_xkm1 / max(dist2, 1e-20f);   // area density at x_{k-1}

    float fwd_fp = 1.0f / max(fwd_area_pdf, 1e-20f);
    float bwd_fp = 1.0f / max(bwd_area_pdf, 1e-20f);

    return min(fwd_fp, bwd_fp) >= primary_fp_thresh;
}

// RIS reservoir for global illumination
struct Reservoir_GI
{
    // Constant-after-hit payload
    float3 x2_gi;
    float3 n2_s_gi;
    float3 n2_g_gi;
    uint   objID_gi;
    uint   matID_gi;
    float2 uv_gi;
    float  etai_gi;
    float  etat_gi;

    // Varying payload
    float3 L2_gi;
    float3 V2_gi;
    float2 J_gi;
    uint   F_gi;        // RGB9E5-packed contribution (replaces scalar luminance)

    // Reservoir bookkeeping
    float  W_gi;
    float  w_sum_gi;
    uint   M_gi;

    // ── Hybrid-shift additions ─────────────────────────────────────
    uint   seed_gi;     // BSDF-stream seed snapshot → replay x_1..x_{k-1} in shift
    uint   k_gi;        // reconnection index (k >= 2; 0 = no recon vertex / replay-only)
    uint   method_gi;   // one of RC_METHOD_*
};


// SoA layout — each field is a contiguous plane across all pixels.
// Total buffer size per pixel is 88 bytes.
// Layout: PACK1(16) | L2(4) | V2(4) | N2G(4) | OBJID(4) | UV(4) | IOR(4) | MATID(4) |
//         J(8) | W(4) | F(4) | M(4) | WSUM(4) | VPOST(4) | TPOST(4) |
//         SEED(4) | K(4) | METHOD(4) = 88

static const uint BYTES_GI        = 68u;
static const uint BYTES_GI_VPOST  =  4u;
static const uint BYTES_GI_TPOST  =  4u;
static const uint BYTES_GI_SEED   =  4u;
static const uint BYTES_GI_K      =  4u;
static const uint BYTES_GI_METHOD =  4u;
static const uint STRIDE_GI       = BYTES_GI + BYTES_GI_VPOST + BYTES_GI_TPOST
                                  + BYTES_GI_SEED + BYTES_GI_K + BYTES_GI_METHOD; // 88

// Per-field sizes
static const uint GI_SZ_PACK1 = 16u;  // x2(12) + n2_s_packed(4)
static const uint GI_SZ_L2    =  4u;
static const uint GI_SZ_V2    =  4u;
static const uint GI_SZ_N2G   =  4u;
static const uint GI_SZ_OBJID =  4u;
static const uint GI_SZ_UV    =  4u;
static const uint GI_SZ_IOR   =  4u;  // etai(fp16) + etat(fp16)
static const uint GI_SZ_MATID =  4u;
static const uint GI_SZ_J     =  8u;  // J.x(4) + J.y(4)
static const uint GI_SZ_W     =  4u;
static const uint GI_SZ_F     =  4u;
static const uint GI_SZ_M     =  4u;
static const uint GI_SZ_WSUM  =  4u;
static const uint GI_SZ_VPOST  =  4u;
static const uint GI_SZ_TPOST  =  4u;
static const uint GI_SZ_SEED   =  4u;
static const uint GI_SZ_K      =  4u;
static const uint GI_SZ_METHOD =  4u;

// Plane cumulative offsets (per-pixel contribution to base address)
static const uint GI_PLANE_PACK1 =  0u;
static const uint GI_PLANE_L2    = 16u;
static const uint GI_PLANE_V2    = 20u;
static const uint GI_PLANE_N2G   = 24u;
static const uint GI_PLANE_OBJID = 28u;
static const uint GI_PLANE_UV    = 32u;
static const uint GI_PLANE_IOR   = 36u;
static const uint GI_PLANE_MATID = 40u;
static const uint GI_PLANE_J     = 44u;
static const uint GI_PLANE_W     = 52u;
static const uint GI_PLANE_F     = 56u;
static const uint GI_PLANE_M     = 60u;
static const uint GI_PLANE_WSUM  = 64u;
static const uint GI_PLANE_VPOST  = 68u;
static const uint GI_PLANE_TPOST  = 72u;
static const uint GI_PLANE_SEED   = 76u;
static const uint GI_PLANE_K      = 80u;
static const uint GI_PLANE_METHOD = 84u;

// SoA address helpers
// Tile-aligned pixel count — must match MapPixelID's 4x8 tile swizzle.
uint gi_numPx()                       { return ((IMG_W + 3u) / 4u) * ((IMG_H + 7u) / 8u) * 32u; }
uint gi_addr_pack1(uint px)           { return px * GI_SZ_PACK1; }
uint gi_addr_l2(uint px)              { uint N = gi_numPx(); return N * GI_PLANE_L2    + px * GI_SZ_L2; }
uint gi_addr_v2(uint px)              { uint N = gi_numPx(); return N * GI_PLANE_V2    + px * GI_SZ_V2; }
uint gi_addr_n2g(uint px)             { uint N = gi_numPx(); return N * GI_PLANE_N2G   + px * GI_SZ_N2G; }
uint gi_addr_objid(uint px)           { uint N = gi_numPx(); return N * GI_PLANE_OBJID + px * GI_SZ_OBJID; }
uint gi_addr_uv(uint px)             { uint N = gi_numPx(); return N * GI_PLANE_UV    + px * GI_SZ_UV; }
uint gi_addr_ior(uint px)            { uint N = gi_numPx(); return N * GI_PLANE_IOR   + px * GI_SZ_IOR; }
uint gi_addr_matid(uint px)          { uint N = gi_numPx(); return N * GI_PLANE_MATID + px * GI_SZ_MATID; }
uint gi_addr_j(uint px)              { uint N = gi_numPx(); return N * GI_PLANE_J     + px * GI_SZ_J; }
uint gi_addr_w(uint px)              { uint N = gi_numPx(); return N * GI_PLANE_W     + px * GI_SZ_W; }
uint gi_addr_f(uint px)              { uint N = gi_numPx(); return N * GI_PLANE_F     + px * GI_SZ_F; }
uint gi_addr_m(uint px)              { uint N = gi_numPx(); return N * GI_PLANE_M     + px * GI_SZ_M; }
uint gi_addr_wsum(uint px)           { uint N = gi_numPx(); return N * GI_PLANE_WSUM  + px * GI_SZ_WSUM; }
uint gi_addr_vpost(uint px)          { uint N = gi_numPx(); return N * GI_PLANE_VPOST  + px * GI_SZ_VPOST; }
uint gi_addr_tpost(uint px)          { uint N = gi_numPx(); return N * GI_PLANE_TPOST  + px * GI_SZ_TPOST; }
uint gi_addr_seed(uint px)           { uint N = gi_numPx(); return N * GI_PLANE_SEED   + px * GI_SZ_SEED; }
uint gi_addr_k(uint px)              { uint N = gi_numPx(); return N * GI_PLANE_K      + px * GI_SZ_K; }
uint gi_addr_method(uint px)         { uint N = gi_numPx(); return N * GI_PLANE_METHOD + px * GI_SZ_METHOD; }

// Store/load packed V_post (world-space)
void store_Vpost_gi(RWByteAddressBuffer b, uint pixelIdx, float3 Vpost_world)
{
    b.Store(gi_addr_vpost(pixelIdx), PackNormal(normalize(Vpost_world)));
}

float3 load_Vpost_gi(RWByteAddressBuffer b, uint pixelIdx)
{
    return UnpackNormal(b.Load(gi_addr_vpost(pixelIdx)));
}

void store_Tpost_gi(RWByteAddressBuffer b, uint pixelIdx, float3 Tpost)
{
    b.Store(gi_addr_tpost(pixelIdx), PackRGB9E5(max(Tpost, 0.0f)));
}

float3 load_Tpost_gi(RWByteAddressBuffer b, uint pixelIdx)
{
    return UnpackRGB9E5(b.Load(gi_addr_tpost(pixelIdx)));
}


// Raw word-for-word copy of a reservoir from src→dst.  Avoids the
// unpack→world-transform→repack roundtrip in load+store, which is wasted
// work for early-out paths that just pass the reservoir through unchanged.
void copyReservoirGI_Raw(RWByteAddressBuffer dst, RWByteAddressBuffer src, uint pixelIdx)
{
    dst.Store4(gi_addr_pack1(pixelIdx), src.Load4(gi_addr_pack1(pixelIdx)));
    dst.Store (gi_addr_l2(pixelIdx),    src.Load (gi_addr_l2(pixelIdx)));
    dst.Store (gi_addr_v2(pixelIdx),    src.Load (gi_addr_v2(pixelIdx)));
    dst.Store (gi_addr_n2g(pixelIdx),   src.Load (gi_addr_n2g(pixelIdx)));
    dst.Store (gi_addr_objid(pixelIdx), src.Load (gi_addr_objid(pixelIdx)));
    dst.Store (gi_addr_uv(pixelIdx),    src.Load (gi_addr_uv(pixelIdx)));
    dst.Store (gi_addr_ior(pixelIdx),   src.Load (gi_addr_ior(pixelIdx)));
    dst.Store (gi_addr_matid(pixelIdx), src.Load (gi_addr_matid(pixelIdx)));
    dst.Store2(gi_addr_j(pixelIdx),     src.Load2(gi_addr_j(pixelIdx)));
    dst.Store (gi_addr_w(pixelIdx),     src.Load (gi_addr_w(pixelIdx)));
    dst.Store (gi_addr_f(pixelIdx),     src.Load (gi_addr_f(pixelIdx)));
    dst.Store (gi_addr_m(pixelIdx),     src.Load (gi_addr_m(pixelIdx)));
    dst.Store (gi_addr_wsum(pixelIdx),  src.Load (gi_addr_wsum(pixelIdx)));
    dst.Store (gi_addr_seed(pixelIdx),  src.Load (gi_addr_seed(pixelIdx)));
    dst.Store (gi_addr_k(pixelIdx),     src.Load (gi_addr_k(pixelIdx)));
    dst.Store (gi_addr_method(pixelIdx),src.Load (gi_addr_method(pixelIdx)));
}

void storeReservoirGI(RWByteAddressBuffer buf, uint pixelIdx, const Reservoir_GI r)
{
    // Sentinels (sun=0xFFFFFFFEu, env=0xFFFFFFFFu) have no transform — store world values directly.
    bool sentinel = (r.objID_gi == 0xFFFFFFFFu) || (r.objID_gi == 0xFFFFFFFEu);
    float3 xO  = sentinel ? r.x2_gi   : WorldToObjectPos (r.objID_gi, r.x2_gi);
    float3 nSO = sentinel ? r.n2_s_gi : WorldToObjectNrm(r.objID_gi, r.n2_s_gi);
    float3 nGO = sentinel ? r.n2_g_gi : WorldToObjectNrm(r.objID_gi, r.n2_g_gi);

    buf.Store4(gi_addr_pack1(pixelIdx), uint4(asuint(xO), PackNormal(normalize(nSO))));
    buf.Store (gi_addr_l2(pixelIdx),    PackRGB9E5(r.L2_gi));
    buf.Store (gi_addr_v2(pixelIdx),    PackNormal(normalize(r.V2_gi)));
    buf.Store (gi_addr_n2g(pixelIdx),   PackNormal(normalize(nGO)));
    buf.Store (gi_addr_objid(pixelIdx), r.objID_gi);
    buf.Store (gi_addr_uv(pixelIdx),    PackFloat2x16(r.uv_gi.x, r.uv_gi.y));
    buf.Store (gi_addr_ior(pixelIdx),   PackFloat2x16(r.etai_gi, r.etat_gi));
    buf.Store (gi_addr_matid(pixelIdx), r.matID_gi);
    buf.Store2(gi_addr_j(pixelIdx),     uint2(asuint(r.J_gi.x), asuint(r.J_gi.y)));
    buf.Store (gi_addr_w(pixelIdx),     asuint(r.W_gi));
    buf.Store (gi_addr_f(pixelIdx),     r.F_gi);
    buf.Store (gi_addr_m(pixelIdx),     r.M_gi);
    buf.Store (gi_addr_wsum(pixelIdx),  asuint(r.w_sum_gi));
    buf.Store (gi_addr_seed(pixelIdx),  r.seed_gi);
    buf.Store (gi_addr_k(pixelIdx),     r.k_gi);
    buf.Store (gi_addr_method(pixelIdx),r.method_gi);
}

Reservoir_GI loadReservoirGI(RWByteAddressBuffer buf, uint pixelIdx)
{
    Reservoir_GI r;

    uint4 p1 = buf.Load4(gi_addr_pack1(pixelIdx));

    r.objID_gi = buf.Load(gi_addr_objid(pixelIdx));
    r.matID_gi = buf.Load(gi_addr_matid(pixelIdx));

    bool sentinel = (r.objID_gi == 0xFFFFFFFFu) || (r.objID_gi == 0xFFFFFFFEu);
    r.x2_gi    = sentinel ? asfloat(p1.xyz)       : ObjectToWorldPos (r.objID_gi, asfloat(p1.xyz));
    r.n2_s_gi  = sentinel ? UnpackNormal(p1.w)    : ObjectToWorldNrm(r.objID_gi, UnpackNormal(p1.w));
    uint n2g_enc = buf.Load(gi_addr_n2g(pixelIdx));
    r.n2_g_gi  = sentinel ? UnpackNormal(n2g_enc) : ObjectToWorldNrm(r.objID_gi, UnpackNormal(n2g_enc));

    r.L2_gi    = UnpackRGB9E5(buf.Load(gi_addr_l2(pixelIdx)));
    r.V2_gi    = UnpackNormal(buf.Load(gi_addr_v2(pixelIdx)));

    uint ior_packed = buf.Load(gi_addr_ior(pixelIdx));
    uint uv_packed  = buf.Load(gi_addr_uv(pixelIdx));
    UnpackFloat2x16(uv_packed,  r.uv_gi.x,   r.uv_gi.y);
    UnpackFloat2x16(ior_packed, r.etai_gi,    r.etat_gi);

    uint2 j_raw = buf.Load2(gi_addr_j(pixelIdx));
    r.J_gi     = asfloat(j_raw);
    r.W_gi     = asfloat(buf.Load(gi_addr_w(pixelIdx)));
    r.F_gi     = buf.Load(gi_addr_f(pixelIdx));

    r.M_gi     = buf.Load(gi_addr_m(pixelIdx));
    r.w_sum_gi = asfloat(buf.Load(gi_addr_wsum(pixelIdx)));

    r.seed_gi   = buf.Load(gi_addr_seed(pixelIdx));
    r.k_gi      = buf.Load(gi_addr_k(pixelIdx));
    r.method_gi = buf.Load(gi_addr_method(pixelIdx));

    return r;
}


// -------------------------------
// Fast loaders (DI-style usage)
// -------------------------------

uint  load_objID_gi(RWByteAddressBuffer b, uint pixelIdx)
{
    return b.Load(gi_addr_objid(pixelIdx));
}

uint  load_matID_gi(RWByteAddressBuffer b, uint pixelIdx)
{
    return b.Load(gi_addr_matid(pixelIdx));
}

float3 load_x2_gi(RWByteAddressBuffer b, uint pixelIdx, uint objID)
{
    float3 xO = asfloat(b.Load4(gi_addr_pack1(pixelIdx)).xyz);
    return (objID == 0xFFFFFFFFu || objID == 0xFFFFFFFEu) ? xO : ObjectToWorldPos(objID, xO);
}

float3 load_n2_s_gi(RWByteAddressBuffer b, uint pixelIdx, uint objID)
{
    uint enc = b.Load4(gi_addr_pack1(pixelIdx)).w;
    float3 n = UnpackNormal(enc);
    return (objID == 0xFFFFFFFFu || objID == 0xFFFFFFFEu) ? n : ObjectToWorldNrm(objID, n);
}

float3 load_n2_g_gi(RWByteAddressBuffer b, uint pixelIdx, uint objID)
{
    uint enc = b.Load(gi_addr_n2g(pixelIdx));
    float3 n = UnpackNormal(enc);
    return (objID == 0xFFFFFFFFu || objID == 0xFFFFFFFEu) ? n : ObjectToWorldNrm(objID, n);
}

float3 load_L2_gi(RWByteAddressBuffer b, uint pixelIdx)
{
    return UnpackRGB9E5(b.Load(gi_addr_l2(pixelIdx)));
}

float3 load_V2_gi(RWByteAddressBuffer b, uint pixelIdx)
{
    return UnpackNormal(b.Load(gi_addr_v2(pixelIdx)));
}

float2 load_uv_gi(RWByteAddressBuffer b, uint pixelIdx)
{
    float2 r;
    UnpackFloat2x16(b.Load(gi_addr_uv(pixelIdx)), r.x, r.y);
    return r;
}

float  load_etai_gi(RWByteAddressBuffer b, uint pixelIdx)
{
    uint p = b.Load(gi_addr_ior(pixelIdx));
    return f16tof32_custom(p & 0xFFFFu);
}

float  load_etat_gi(RWByteAddressBuffer b, uint pixelIdx)
{
    uint p = b.Load(gi_addr_ior(pixelIdx));
    return f16tof32_custom(p >> 16);
}

float2 load_J_gi(RWByteAddressBuffer b, uint pixelIdx)
{
    return asfloat(b.Load2(gi_addr_j(pixelIdx)));
}

void store_Jy_gi(RWByteAddressBuffer b, uint pixelIdx, float Jy)
{
    b.Store(gi_addr_j(pixelIdx) + 4u, asuint(Jy));
}

float  load_W_gi(RWByteAddressBuffer b, uint pixelIdx)
{
    return asfloat(b.Load(gi_addr_w(pixelIdx)));
}

uint   load_F_gi(RWByteAddressBuffer b, uint pixelIdx)
{
    return b.Load(gi_addr_f(pixelIdx));
}


// -----------------------------------------
// Special entry accessors (fast update only)
// -----------------------------------------

float load_wsum_gi(RWByteAddressBuffer b, uint pixelIdx)
{
    return asfloat(b.Load(gi_addr_wsum(pixelIdx)));
}

void store_wsum_gi(RWByteAddressBuffer b, uint pixelIdx, float wsum)
{
    b.Store(gi_addr_wsum(pixelIdx), asuint(wsum));
}

uint load_M_gi(RWByteAddressBuffer b, uint pixelIdx)
{
    return b.Load(gi_addr_m(pixelIdx));
}

void store_M_gi(RWByteAddressBuffer b, uint pixelIdx, uint M)
{
    b.Store(gi_addr_m(pixelIdx), M);
}

void store_W_gi(RWByteAddressBuffer b, uint pixelIdx, float W)
{
    b.Store(gi_addr_w(pixelIdx), asuint(W));
}

void store_F_gi(RWByteAddressBuffer b, uint pixelIdx, uint F)
{
    b.Store(gi_addr_f(pixelIdx), F);
}

// Hybrid-shift bookkeeping: seed / k / method
uint load_seed_gi  (RWByteAddressBuffer b, uint pixelIdx)         { return b.Load(gi_addr_seed(pixelIdx)); }
void store_seed_gi (RWByteAddressBuffer b, uint pixelIdx, uint s) {        b.Store(gi_addr_seed(pixelIdx), s); }

uint load_k_gi     (RWByteAddressBuffer b, uint pixelIdx)         { return b.Load(gi_addr_k(pixelIdx)); }
void store_k_gi    (RWByteAddressBuffer b, uint pixelIdx, uint k) {        b.Store(gi_addr_k(pixelIdx), k); }

uint load_method_gi (RWByteAddressBuffer b, uint pixelIdx)         { return b.Load(gi_addr_method(pixelIdx)); }
void store_method_gi(RWByteAddressBuffer b, uint pixelIdx, uint m) {        b.Store(gi_addr_method(pixelIdx), m); }


inline bool RejectNormal_GI(float3 n1, float3 n2, float threshold){
    float similarity = dot(n1, n2);
    return (similarity < threshold);
}

inline bool RejectDistance_GI(float3 x1, float3 x2, float3 normal, float threshold)
{
    float dist = abs(dot(x2 - x1, normal));
    return dist > threshold;
}

inline bool IsValidReservoir_GI(Reservoir_GI r){
    bool valid =
        any(abs(r.n2_s_gi) > 0.0f) &&
        r.M_gi > 0
        ;
    return valid;
}

inline bool IsValidReservoir_GI_opt(in float3 n2, in uint M){
    bool valid =
        any(abs(n2) > 0.0f) &&
        M > 0.0f;
    return valid;
}

inline void InvalidateReservoirGI_ShadingNormal(
    RWByteAddressBuffer buf,
    uint pixelIdx
)
{
    // n2_s_gi is stored in PACK1.w
    buf.Store(gi_addr_pack1(pixelIdx) + 12u, 0u);
}



// Calculate reconnection (two–sided)
inline float3 ReconnectGI(
    // Vertex x1 (camera path hit)
    in float3  x1,
    in float3  n1_s,
    in float3  n1_g,
    in float3  o,
    in uint    mID1,
    in float3  localKd1,
    in float   localPr1,
    in float   localPm1,
    in float   etai1,
    in float   etat1,

    // Vertex x2 (GI reservoir / reconnection vertex)
    in uint    mID2,
    in float3  x2,
    in float3  n2_s,
    in float3  n2_g,
    in float3  L2,
    in float3  V2,
    in float3  localKd2,
    in float   localPr2,
    in float   localPm2,
    in float   etai2,
    in float   etat2,

    // Jacobian / PDF plumbing
    in float   pdfx2,
    in float   Jc,      // canonical jacobian term
    in bool    applyJ,
    out float  Jn,
    out float  J
)
{
    if (length(L2) < EPSILON)
        return 0.0f;

    // Geometric prep
    float3 dir   = x2 - x1;
    float  dist  = length(dir);
    float3 ndirN = normalize(-dir); // direction from x2 to x1

    float3 F1 = BSDF_term(mID1, n1_s, n1_g, -ndirN, o,  localKd1, localPr1, localPm1, etai1, etat1);
    float3 F2 = BSDF_term(mID2, n2_s, n2_g, -V2, ndirN, localKd2, localPr2, localPm2, etai2, etat2);

    // Geometry term
    float  G1  = G_term(n1_s, -ndirN);
    float  G2  = G_term(n2_s, -V2);

    // Missing pdfs:
    // PDF at x1 always recomputed
    float PDF1 = PDF_term(mID1, n1_s, n1_g, -ndirN, o,  localKd1, localPr1, localPm1, etai1, etat1);

    // PDF at x2: reuse if provided (NEE ray etc), else recompute
    float PDF2 = pdfx2;
    if (pdfx2 == 0.0f)
        PDF2 = PDF_term(mID2, n2_s, n2_g, -V2, ndirN, localKd2, localPr2, localPm2, etai2, etat2);

    if (PDF1 <= EPSILON || PDF2 <= EPSILON)
        return 0.0f;

    // If the direction from x1 to x2 (-ndirN) points against the geometric normal, we are transmitting INTO the object.
    float3 transmittance = float3(1.0f, 1.0f, 1.0f);
    if (dot(n1_g, -ndirN) < 0.0f)
    {
        transmittance = CalculateAbsorptionThroughput(materials[mID1].Tf, dist);
    }

    // Always compute PSS jacobian for tracking through reservoir merge
    float Gj = abs(dot(ndirN, n2_s)) / (dist * dist);
    Jn = max(PDF1 * PDF2 * Gj, EPSILON);
    J  = 1.0f;

    // Throughput
    float3 r = (F1 / PDF1) * (F2 / PDF2) * L2 * G1 * G2 * transmittance;

    if (applyJ)
    {
        if(Jc < EPSILON) return 0.0f;
        J = Jn / Jc;
    }

    if (any(isnan(r)) || any(isinf(r)) || all(r < EPSILON))
        r = (float3)0.0f;

    return max(r,0.0f);
}



// Update GI reservoir
bool UpdateReservoirGI(
    inout Reservoir_GI reservoir,
    in float wi,
    in uint  M,

    in float3 x2,
    in float3 n2_s,
    in float3 n2_g,
    in float3 L2,
    in float3 V2,

    in float2 uv,
    in float  etai,
    in float  etat,

    in uint matID,
    in uint objID,

    in float2 J,
    in uint   F,

    // Hybrid-shift fields from the donor candidate — propagated on accept.
    in uint   cand_seed,
    in uint   cand_k,
    in uint   cand_method,

    inout uint2 seed
)
{
    reservoir.w_sum_gi += wi;
    reservoir.M_gi     += M;

    if (RandomFloatSingle(seed.x) < (wi / reservoir.w_sum_gi))
    {
        reservoir.x2_gi   = x2;
        reservoir.n2_s_gi = n2_s;
        reservoir.n2_g_gi = n2_g;
        reservoir.objID_gi = objID;
        reservoir.matID_gi = matID;

        reservoir.uv_gi      = uv;
        reservoir.etai_gi    = etai;
        reservoir.etat_gi    = etat;

        reservoir.L2_gi   = L2;
        reservoir.V2_gi   = V2;
        reservoir.J_gi    = J;
        reservoir.F_gi    = F;

        reservoir.seed_gi   = cand_seed;
        reservoir.k_gi      = cand_k;
        reservoir.method_gi = cand_method;
        return true;
    }
    return false;
}


// Per-candidate RIS update for raygen. Writes all varying + const-hit fields
// associated with this specific candidate if RIS accepts.
//
// x_k_* describe the reconnection vertex for THIS candidate:
//   • For BSDF_ENV / BSDF_EMIT / NEE / SUN: the reconnection vertex x_k
//   • For NEE_RP : light position / surface (still stored so shift can reconnect)
//   • For SUN_RP : sun direction in x_k slot (objID = 0xFFFFFFFEu)
//   • For BSDF_*_RP : unused — caller may pass dummies (objID=0)
bool UpdateReservoirGI_Candidate(
    RWByteAddressBuffer buf, uint pixelIdx, float wi,

    // Reconnection vertex (x_k) — written on accept
    float3 x_k_world, float3 n_k_s_world, float3 n_k_g_world,
    uint   matID_k,   uint   objID_k,
    float2 uv_k,      float  etai_k,      float etat_k,

    // Suffix / tail data
    float3 L2_new, float3 V2_new, float2 J_new, uint F_pk,

    // Path classification + replay handle
    uint method, uint k_idx, uint bsdf_seed0,

    inout uint2 seed)
{
    float currentWSum = asfloat(buf.Load(gi_addr_wsum(pixelIdx)));
    float newWSum     = currentWSum + wi;
    buf.Store(gi_addr_wsum(pixelIdx), asuint(newWSum));

    if (wi <= 0.0f || RandomFloatSingle(seed.x) >= (wi / newWSum))
        return false;

    // Const-hit — x_k varies per candidate. Sentinel objIDs (sun / env) skip the transform.
    bool sentinel = (objID_k == 0xFFFFFFFFu) || (objID_k == 0xFFFFFFFEu);
    float3 xO  = sentinel ? x_k_world    : WorldToObjectPos (objID_k, x_k_world);
    float3 nSO = sentinel ? n_k_s_world  : WorldToObjectNrm(objID_k, n_k_s_world);
    float3 nGO = sentinel ? n_k_g_world  : WorldToObjectNrm(objID_k, n_k_g_world);
    buf.Store4(gi_addr_pack1(pixelIdx), uint4(asuint(xO), PackNormal(normalize(nSO))));
    buf.Store (gi_addr_n2g  (pixelIdx), PackNormal(normalize(nGO)));
    buf.Store (gi_addr_objid(pixelIdx), objID_k);
    buf.Store (gi_addr_matid(pixelIdx), matID_k);
    buf.Store (gi_addr_uv   (pixelIdx), PackFloat2x16(uv_k.x, uv_k.y));
    buf.Store (gi_addr_ior  (pixelIdx), PackFloat2x16(etai_k, etat_k));

    // Varying
    buf.Store (gi_addr_l2(pixelIdx), PackRGB9E5(L2_new));
    buf.Store (gi_addr_v2(pixelIdx), PackNormal(normalize(V2_new)));
    buf.Store2(gi_addr_j (pixelIdx), uint2(asuint(J_new.x), asuint(J_new.y)));
    buf.Store (gi_addr_f (pixelIdx), F_pk);

    // Classification
    buf.Store (gi_addr_method(pixelIdx), method);
    buf.Store (gi_addr_k     (pixelIdx), k_idx);
    buf.Store (gi_addr_seed  (pixelIdx), bsdf_seed0);
    return true;
}

// Fast Update
bool UpdateReservoirGI_Fast(
    RWByteAddressBuffer buf,
    uint pixelIdx,
    float wi,

    // Varying (always written on accept)
    float3 L2_new,
    float2 J_new,
    float3 V2_new,

    inout uint2 seed
)
{
    // w_sum accumulates
    float currentWSum = asfloat(buf.Load(gi_addr_wsum(pixelIdx)));
    float newWSum     = currentWSum + wi;
    buf.Store(gi_addr_wsum(pixelIdx), asuint(newWSum));

    bool isAccepted = false;

    if (RandomFloatSingle(seed.x) < (wi / newWSum))
    {
        isAccepted = true;

        buf.Store(gi_addr_l2(pixelIdx), PackRGB9E5(L2_new));
        buf.Store(gi_addr_v2(pixelIdx), PackNormal(normalize(V2_new)));
        buf.Store2(gi_addr_j(pixelIdx), uint2(asuint(J_new.x), asuint(J_new.y)));
    }

    return isAccepted;
}


// ---------------------------------
// Constant-data setters
// ---------------------------------

// Store constant hit data for GI reconnection
void SetReservoirGI_ConstHit(
    RWByteAddressBuffer buf,
    uint pixelIdx,
    float3 x2_world,
    float3 n2s_world,
    float3 n2g_world,
    uint   matID,
    uint   objID
)
{
    float3 xO  = WorldToObjectPos (objID, x2_world);
    float3 nSO = WorldToObjectNrm(objID, n2s_world);
    float3 nGO = WorldToObjectNrm(objID, n2g_world);

    buf.Store4(gi_addr_pack1(pixelIdx), uint4(asuint(xO), PackNormal(normalize(nSO))));
    buf.Store (gi_addr_n2g(pixelIdx),   PackNormal(normalize(nGO)));
    buf.Store (gi_addr_objid(pixelIdx), objID);
    buf.Store (gi_addr_matid(pixelIdx), matID);
}

void SetReservoirGI_UVAndIOR(
    RWByteAddressBuffer buf,
    uint pixelIdx,
    float2 uv,
    float  etai,
    float  etat
)
{
    buf.Store(gi_addr_uv(pixelIdx),  PackFloat2x16(uv.x, uv.y));
    buf.Store(gi_addr_ior(pixelIdx), PackFloat2x16(etai, etat));
}
