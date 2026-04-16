// Hybrid shift (paper §7.4/§8.3) — connectability thresholds
static const uint  MAX_RC_INDEX        = 4u;    // cap on reconnection depth (replay bounces = rc - 1)
static const float HYBRID_ROUGH_MIN    = 0.2f;  // per-lobe GGX roughness threshold
static const float HYBRID_MAX_DIST_MIN = 0.02f; // minimum connection distance (~2% scene size)

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

    // Hybrid shift: seed for random replay + reconnection index k (1 = pure reconnection)
    uint   seed_gi;
    uint   rc_idx_gi;
};


// SoA layout — each field is a contiguous plane across all pixels.
// Layout: PACK1(16) | L2(4) | V2(4) | N2G(4) | OBJID(4) | UV(4) | IOR(4) | MATID(4) |
//         J(8) | W(4) | F(4) | M(4) | WSUM(4) | VPOST(4) | TPOST(4) | SEED(4) | RC(4) = 84

static const uint BYTES_GI       = 68u;
static const uint BYTES_GI_VPOST =  4u;
static const uint BYTES_GI_TPOST =  4u;
static const uint BYTES_GI_SEED  =  4u;
static const uint BYTES_GI_RC    =  4u;
static const uint STRIDE_GI      = BYTES_GI + BYTES_GI_VPOST + BYTES_GI_TPOST + BYTES_GI_SEED + BYTES_GI_RC; // 84

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
static const uint GI_SZ_VPOST =  4u;
static const uint GI_SZ_TPOST =  4u;
static const uint GI_SZ_SEED  =  4u;
static const uint GI_SZ_RC    =  4u;

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
static const uint GI_PLANE_VPOST = 68u;
static const uint GI_PLANE_TPOST = 72u;
static const uint GI_PLANE_SEED  = 76u;
static const uint GI_PLANE_RC    = 80u;

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
uint gi_addr_vpost(uint px)          { uint N = gi_numPx(); return N * GI_PLANE_VPOST + px * GI_SZ_VPOST; }
uint gi_addr_tpost(uint px)          { uint N = gi_numPx(); return N * GI_PLANE_TPOST + px * GI_SZ_TPOST; }
uint gi_addr_seed(uint px)           { uint N = gi_numPx(); return N * GI_PLANE_SEED  + px * GI_SZ_SEED; }
uint gi_addr_rc(uint px)             { uint N = gi_numPx(); return N * GI_PLANE_RC    + px * GI_SZ_RC; }

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

// Hybrid shift: seed / rc_idx accessors
void store_seed_gi(RWByteAddressBuffer b, uint pixelIdx, uint s) { b.Store(gi_addr_seed(pixelIdx), s); }
uint load_seed_gi (RWByteAddressBuffer b, uint pixelIdx)         { return b.Load(gi_addr_seed(pixelIdx)); }
void store_rc_gi  (RWByteAddressBuffer b, uint pixelIdx, uint k) { b.Store(gi_addr_rc(pixelIdx), k); }
uint load_rc_gi   (RWByteAddressBuffer b, uint pixelIdx)         { return b.Load(gi_addr_rc(pixelIdx)); }


void storeReservoirGI(RWByteAddressBuffer buf, uint pixelIdx, const Reservoir_GI r)
{
    float3 xO  = WorldToObjectPos (r.objID_gi, r.x2_gi);
    float3 nSO = WorldToObjectNrm(r.objID_gi, r.n2_s_gi);
    float3 nGO = WorldToObjectNrm(r.objID_gi, r.n2_g_gi);

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
    buf.Store (gi_addr_rc(pixelIdx),    r.rc_idx_gi);
}

Reservoir_GI loadReservoirGI(RWByteAddressBuffer buf, uint pixelIdx)
{
    Reservoir_GI r;

    uint4 p1 = buf.Load4(gi_addr_pack1(pixelIdx));

    r.objID_gi = buf.Load(gi_addr_objid(pixelIdx));
    r.matID_gi = buf.Load(gi_addr_matid(pixelIdx));

    r.x2_gi    = ObjectToWorldPos (r.objID_gi, asfloat(p1.xyz));
    r.n2_s_gi  = ObjectToWorldNrm(r.objID_gi, UnpackNormal(p1.w));
    r.n2_g_gi  = ObjectToWorldNrm(r.objID_gi, UnpackNormal(buf.Load(gi_addr_n2g(pixelIdx))));

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
    r.rc_idx_gi = buf.Load(gi_addr_rc(pixelIdx));

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
    return ObjectToWorldPos(objID, xO);
}

float3 load_n2_s_gi(RWByteAddressBuffer b, uint pixelIdx, uint objID)
{
    uint enc = b.Load4(gi_addr_pack1(pixelIdx)).w;
    return ObjectToWorldNrm(objID, UnpackNormal(enc));
}

float3 load_n2_g_gi(RWByteAddressBuffer b, uint pixelIdx, uint objID)
{
    uint enc = b.Load(gi_addr_n2g(pixelIdx));
    return ObjectToWorldNrm(objID, UnpackNormal(enc));
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



// ────────────────────────────────────────────────────────────────────────────
// Hybrid shift replay: inline RayQuery trace returning hit surface identifiers
// (matches the alpha-test logic in IsVisible from Inline_RT_v8.hlsli).
// ────────────────────────────────────────────────────────────────────────────
#ifdef ENABLE_RAY_QUERY_INLINE
struct HybridReplayHit
{
    bool   hit;
    uint   instID;
    uint   primID;
    float2 bary;
    float  tHit;
};

inline HybridReplayHit HybridTraceReplay(float3 origin, float3 direction)
{
    HybridReplayHit r;
    r.hit = false; r.instID = 0u; r.primID = 0u; r.bary = float2(0,0); r.tHit = 0.0f;

    RayDesc ray;
    ray.Origin    = origin;
    ray.Direction = direction;
    ray.TMin      = 0.001f;
    ray.TMax      = 10000.0f;

    RayQuery<RAY_FLAG_FORCE_OMM_2_STATE, RAYQUERY_FLAG_ALLOW_OPACITY_MICROMAPS> q;
    q.TraceRayInline(SceneBVH, RAY_FLAG_NONE, 0xFF, ray);

    while (q.Proceed())
    {
        if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
        {
            uint   iID    = q.CandidateInstanceID();
            uint   pID    = FlatPrimID(iID, q.CandidateGeometryIndex(), q.CandidatePrimitiveIndex());
            uint   mID    = materialIDs[instanceProps[iID].materialBase + pID];
            Material mat  = materials[mID];

            if (mat.albedoTexID < 0)
            {
                if (mat.Kd.w < 1.0f - EPSILON) continue;
                q.CommitNonOpaqueTriangleHit();
                continue;
            }

            uint baseI = instanceProps[iID].indexBase;
            uint i0 = indices[baseI + 3u * pID + 0u];
            uint i1 = indices[baseI + 3u * pID + 1u];
            uint i2 = indices[baseI + 3u * pID + 2u];

            float2 uv0 = (float2)BTriVertex[i0].texCoord;
            float2 uv1 = (float2)BTriVertex[i1].texCoord;
            float2 uv2 = (float2)BTriVertex[i2].texCoord;

            float2 bc  = q.CandidateTriangleBarycentrics();
            float  b0  = 1.0f - bc.x - bc.y;
            float2 uv  = uv0 * b0 + uv1 * bc.x + uv2 * bc.y;

            Texture2D<float4> tex = ResourceDescriptorHeap[mat.albedoTexID];
            float alpha = tex.SampleLevel(g_sampler, uv * mat.albedoUVScale, 0).a;

            if (alpha >= mat.alphaThreshold)
                q.CommitNonOpaqueTriangleHit();
        }
    }

    if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
    {
        r.hit    = true;
        r.instID = q.CommittedInstanceID();
        r.primID = FlatPrimID(r.instID, q.CommittedGeometryIndex(), q.CommittedPrimitiveIndex());
        r.bary   = q.CommittedTriangleBarycentrics();
        r.tHit   = q.CommittedRayT();
    }
    return r;
}
#endif // ENABLE_RAY_QUERY_INLINE


// Calculate reconnection (two–sided). For rc_idx == 1 this is the classic
// reconnection shift; for rc_idx >= 2 the function first performs (rc_idx−1)
// bounces of random replay from (x1, n1_s, …) using replaySeed, arriving at
// y_k, then reconnects y_k → x2. On any failure (replay miss, degenerate pdf,
// or failed connectability at y_k) the shift is undefined and returns 0 —
// preserving bijectivity per Lin et al. 2022 §7.4.
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

    // Vertex x_{k+1} (stored GI reservoir / reconnection vertex)
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

    // Hybrid shift: reconnection index k (≥1) and replay seed
    in uint    rc_idx,
    in uint    replaySeed,

    out float  Jn,
    out float  J
)
{
    Jn = EPSILON;
    J  = 1.0f;

    if (length(L2) < EPSILON)
        return 0.0f;

    // ─── y_k state: starts at (x1, …); mutated by replay for rc_idx ≥ 2 ───
    float3 yk_pos  = x1;
    float3 yk_ns   = n1_s;
    float3 yk_ng   = n1_g;
    float3 yk_o    = o;
    uint   yk_mID  = mID1;
    float3 yk_Kd   = localKd1;
    float  yk_Pr   = localPr1;
    float  yk_Pm   = localPm1;
    float  yk_etai = etai1;
    float  yk_etat = etat1;

    float3 replayTP = float3(1.0f, 1.0f, 1.0f);

#ifdef ENABLE_RAY_QUERY_INLINE
    // rc_idx is assumed to already be ≤ MAX_RC_INDEX (enforced in raygen commit block).
    // No upper-bound check here — we must not silently drop a valid reservoir just
    // because its rc_idx happens to be high; that would lose reconnections.
    if (rc_idx >= 2u)
    {
        uint pSeed = replaySeed;
        [loop]
        for (uint i = 1u; i < rc_idx; ++i)
        {
            // Replay bounce i: sample BSDF at y_i with the shared seed
            SamplingP sp = CalculateStrategyProbabilities(
                yk_mID, yk_o, yk_ns, yk_etai, yk_etat, yk_Kd, yk_Pm);
            float3 s = SampleBRDF(sp, yk_mID, yk_o, yk_ns, yk_ng,
                                  yk_Kd, yk_Pr, yk_Pm, pSeed,
                                  yk_etai, yk_etat, -1);
            BrdfData bd = EvaluateAndPdf_COMBINED(
                sp, yk_mID, yk_ns, yk_ng, s, yk_o,
                yk_Kd, yk_Pr, yk_Pm, yk_etai, yk_etat);

            if (bd.pdf <= EPSILON || dot(s, s) < 1e-12f)
                return 0.0f;

            float  cosTheta = abs(dot(yk_ns, s));
            float3 tpStep   = (bd.val * cosTheta) / bd.pdf;
            if (any(isnan(tpStep)) || any(isinf(tpStep)))
                return 0.0f;
            replayTP *= tpStep;

            // Trace to next vertex
            float3 offN   = (dot(s, yk_ng) >= 0.0f) ? yk_ng : -yk_ng;
            float3 origin = offset_ray(yk_pos, offN);

            HybridReplayHit rh = HybridTraceReplay(origin, s);
            if (!rh.hit)
                return 0.0f;

            HitInfo h   = EvalSurfaceState(rh.instID, rh.primID, rh.bary, origin, 0);
            uint    nMI = GetMatIDFast(rh.instID, rh.primID);
            float3  nKd; float nPr, nPm;
            RefetchMaterial(nMI, h.uv, nKd, nPr, nPm);

            yk_pos  = h.hitPos;
            yk_ns   = h.hitNormal;
            yk_ng   = h.hitGNormal;
            yk_o    = normalize(origin - h.hitPos);
            yk_mID  = nMI;
            yk_Kd   = nKd;
            yk_Pr   = nPr;
            yk_Pm   = nPm;
            // Simplification: treat inter-vertex medium as vacuum during replay.
            yk_etai = 1.0f;
            yk_etat = 1.0f;
        }

        // Connectability at (y_k, x_{k+1}) — paper §7.5.
        if (yk_Pr < HYBRID_ROUGH_MIN || localPr2 < HYBRID_ROUGH_MIN)
            return 0.0f;
        if (length(x2 - yk_pos) < HYBRID_MAX_DIST_MIN)
            return 0.0f;
    }
#else
    // No inline RT available — silently fall back to pure reconnection.
    if (rc_idx >= 2u) return 0.0f;
#endif

    // ─── Reconnection math at (y_k, x_{k+1}) ───
    float3 dir   = x2 - yk_pos;
    float  dist  = length(dir);
    if (dist < EPSILON) return 0.0f;
    float3 ndirN = normalize(-dir);

    float3 F1 = BSDF_term(yk_mID, yk_ns, yk_ng, -ndirN, yk_o,
                          yk_Kd, yk_Pr, yk_Pm, yk_etai, yk_etat);
    float3 F2 = BSDF_term(mID2,  n2_s,  n2_g,  -V2,    ndirN,
                          localKd2, localPr2, localPm2, etai2, etat2);

    float G1 = G_term(yk_ns, -ndirN);
    float G2 = G_term(n2_s, -V2);

    float PDF1 = PDF_term(yk_mID, yk_ns, yk_ng, -ndirN, yk_o,
                          yk_Kd, yk_Pr, yk_Pm, yk_etai, yk_etat);
    float PDF2 = pdfx2;
    if (pdfx2 == 0.0f)
        PDF2 = PDF_term(mID2, n2_s, n2_g, -V2, ndirN,
                        localKd2, localPr2, localPm2, etai2, etat2);

    if (PDF1 <= EPSILON || PDF2 <= EPSILON)
        return 0.0f;

    float3 transmittance = float3(1.0f, 1.0f, 1.0f);
    if (dot(yk_ng, -ndirN) < 0.0f)
        transmittance = CalculateAbsorptionThroughput(materials[yk_mID].Tf, dist);

    // PSS Jacobian denominator at the actual reconnection pair (y_k, x_{k+1}).
    // MUST use the GEOMETRIC normal n2_g — the canonical side is stored in raygen as
    //     prev_pdf * |hitGNormal · -rayDir| / dist²
    // using the geometric normal. If we used n2_s here the two sides of J = Jn/Jc
    // would be evaluated against different surfaces and the ratio collapses
    // unpredictably on any surface where shading ≠ geometric normal (normal-mapped
    // metal, etc.) — which is exactly the "mirror reflection shows no improvement"
    // symptom (the base shift stores one thing, the offset recomputes another).
    float Gj = abs(dot(ndirN, n2_g)) / (dist * dist);
    Jn = max(PDF1 * PDF2 * Gj, EPSILON);
    J  = 1.0f;

    float3 r = (F1 / PDF1) * (F2 / PDF2) * L2 * G1 * G2 * transmittance * replayTP;

    if (applyJ)
    {
        if (Jc < EPSILON) return 0.0f;
        J = Jn / Jc;
    }

    if (any(isnan(r)) || any(isinf(r)) || all(r < EPSILON))
        r = (float3)0.0f;

    return max(r, 0.0f);
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

    in uint   replaySeed,
    in uint   rcIdx,

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

        reservoir.seed_gi   = replaySeed;
        reservoir.rc_idx_gi = rcIdx;
        return true;
    }
    return false;
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
