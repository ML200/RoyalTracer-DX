// =====================================================================================================================
//  Shift_GI_v8.hlsli — hybrid shift mapping for GI reservoirs.
//
//  Replaces the plain reconnection shift with a method-aware shift that
//  supports the 8 path classifications in Reservoir_GI.method_gi:
//
//    BSDF_ENV / BSDF_EMIT / NEE / SUN
//        Replay the BSDF prefix x_0 → y_1 → … → y_{k-1} using the stored
//        bsdf_seed0, then reconnect y_{k-1} → x_k. Tail contribution (BSDF
//        bounces past x_k, NEE/sun factor) is baked into the stored L2 / V2.
//
//    BSDF_ENV_RP / BSDF_EMIT_RP
//        No reconnection vertex — replay the entire path with bsdf_seed0.
//        Shift succeeds iff the replay terminates at the same terminal type.
//
//    NEE_RP / SUN_RP
//        Replay the BSDF prefix to y_{k-1}, then connect directly to the
//        stored emitter (x_k = stored light position / sun direction, L2 =
//        stored emission). No reconnection-vertex re-evaluation at x_k.
//
//  v1 approximations (see the "Next" comments in the body):
//    • no volume / IOR-stack tracking during replay
//    • no Russian roulette during replay (fully deterministic)
//    • BSDF_EMIT_RP accepts any emitter terminal (not just the original one)
// =====================================================================================================================

#ifndef SHIFT_GI_V8_HLSLI
#define SHIFT_GI_V8_HLSLI

// Hard cap on replayed bounces. Matches raygen's MAX_BOUNCES so the shift
// can never outlast the original walk.
#ifndef SHIFT_MAX_BOUNCES
#define SHIFT_MAX_BOUNCES 10
#endif

// -------------------------------------------------------------------------
// ShiftReplayBounceEstimate
//
// Returns the number of BSDF replay bounces ShiftGI will perform for a
// reservoir described by (method, k).  Used as the SER coherence hint in
// the temp/spat reuse passes so threads with similar shift workload cluster
// together and divergence in the replay loop drops.
//
//   HAS_RECON (method 1..4) with k>=2 : prefix replay length = k-2
//   _RP       (method 5..8)            : full path replay (up to SHIFT_MAX_BOUNCES)
//   INVALID                            : 0 (no shift work)
// -------------------------------------------------------------------------
inline uint ShiftReplayBounceEstimate(uint method, uint k)
{
    if (method == RC_METHOD_INVALID) return 0u;
    if (RC_HasReconVertex(method))   return (k > 2u) ? (k - 2u) : 0u;
    return (uint)SHIFT_MAX_BOUNCES;   // _RP variants
}

// -------------------------------------------------------------------------
// ShiftTraceClosest_Inline
//
// Inline RayQuery closest-hit trace with alpha testing equivalent to the
// AnyHit shader.  Used by ShiftReplayPrefix / ShiftReplayToTerminal in lieu
// of TraceRay_Custom — avoids the SER reorder + payload roundtrip cost of
// the HitObject path, which dominated shift cost.
//
// Returns true on a committed triangle hit and writes instID, primID, bc.
// Returns false on miss.
// -------------------------------------------------------------------------
inline bool ShiftTraceClosest_Inline(
    in  RayDesc ray,
    out uint    outInstID,
    out uint    outPrimID,
    out float2  outBary)
{
    outInstID = 0u;
    outPrimID = 0u;
    outBary   = float2(0.0f, 0.0f);

    RayQuery<RAY_FLAG_FORCE_OMM_2_STATE, RAYQUERY_FLAG_ALLOW_OPACITY_MICROMAPS> q;
    q.TraceRayInline(SceneBVH, RAY_FLAG_NONE, 0xFF, ray);

    while (q.Proceed())
    {
        if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
        {
            uint cInstID = q.CandidateInstanceIndex();
            uint cPrimID = FlatPrimID(cInstID, q.CandidateGeometryIndex(), q.CandidatePrimitiveIndex());
            uint cMatID  = materialIDs[instanceProps[cInstID].materialBase + cPrimID];
            Material cMat = materials[cMatID];

            // No albedo texture → fully opaque (matches AnyHit shader).
            if (cMat.albedoTexID < 0)
            {
                q.CommitNonOpaqueTriangleHit();
                continue;
            }

            uint baseI = instanceProps[cInstID].indexBase;
            uint i0 = indices[baseI + 3u * cPrimID + 0u];
            uint i1 = indices[baseI + 3u * cPrimID + 1u];
            uint i2 = indices[baseI + 3u * cPrimID + 2u];

            float2 uv0 = (float2)BTriVertex[i0].texCoord;
            float2 uv1 = (float2)BTriVertex[i1].texCoord;
            float2 uv2 = (float2)BTriVertex[i2].texCoord;

            float2 bc = q.CandidateTriangleBarycentrics();
            float  b0 = 1.0f - bc.x - bc.y;
            float2 uv = uv0 * b0 + uv1 * bc.x + uv2 * bc.y;

            Texture2D<float4> tex = ResourceDescriptorHeap[cMat.albedoTexID];
            float alpha = tex.SampleLevel(g_sampler, uv * cMat.albedoUVScale, 0).a;

            if (alpha >= cMat.alphaThreshold)
                q.CommitNonOpaqueTriangleHit();
        }
    }

    if (q.CommittedStatus() != COMMITTED_TRIANGLE_HIT)
        return false;

    outInstID = q.CommittedInstanceIndex();
    outPrimID = FlatPrimID(outInstID, q.CommittedGeometryIndex(), q.CommittedPrimitiveIndex());
    outBary   = q.CommittedTriangleBarycentrics();
    return true;
}

// -------------------------------------------------------------------------
// ShiftReplayPrefix
//
// Replays (kTarget - 2) BSDF bounces starting at y_1, consuming bsdf_seed0.
// Returns the landing vertex y_{k-1}, the prefix throughput T_prefix =
// product of BSDF·cosθ/pdf factors along y_1..y_{k-2}  (the BSDF at y_{k-1}
// itself is NOT folded in — ShiftGI evaluates that via the reconnection BSDF).
// Also returns pdf_last = p^y_{k-2}(ω_{k-2}), i.e. the BSDF pdf of the last
// replayed bounce (used only by callers that still need it for diagnostics —
// the Jacobian numerator uses the reconnection-link pdfs, not this).
//
// Returns false on any early termination:  ray miss, emitter hit, zero pdf,
// numerically bad sample.  On false, the out-parameters are undefined.
// -------------------------------------------------------------------------
inline bool ShiftReplayPrefix(
    in  SurfaceVertex sv_y1,
    in  uint          bsdf_seed0,
    in  uint          kTarget,                 // >= 2
    out SurfaceVertex sv_last,                 // y_{k-1}
    out float3        prefixThroughput,        // T along y_1..y_{k-2}
    out float         pdf_last)
{
    sv_last          = sv_y1;
    prefixThroughput = float3(1, 1, 1);
    pdf_last         = 1.0f;

    // k == 2 → no bounces to replay; y_{k-1} = y_1 already.
    if (kTarget <= 2u)
        return true;

    uint   bseed      = bsdf_seed0;
    float3 rayDir     = float3(0, 0, 0);
    float3 rayOrigin  = float3(0, 0, 0);

    // Current vertex starts as y_1.
    SurfaceVertex sv_cur = sv_y1;

    // We need (kTarget - 2) BSDF bounces from y_1 to reach y_{k-1}.
    uint bouncesToDo = kTarget - 2u;

    [loop]
    for (uint b = 0; b < bouncesToDo; ++b)
    {
        // Sample BSDF at the current vertex.  sv_cur.o points from cur → camera (or prev vertex).
        SamplingP sp = CalculateStrategyProbabilities(
            sv_cur.matID, sv_cur.o, sv_cur.n_s, sv_cur.etai, sv_cur.etat, sv_cur.Kd, sv_cur.Pm);
        float3 s = SampleBRDF(sp, sv_cur.matID, sv_cur.o, sv_cur.n_s, sv_cur.n_g,
                              sv_cur.Kd, sv_cur.Pr, sv_cur.Pm, bseed,
                              sv_cur.etai, sv_cur.etat, /*ior_pointer*/ -1);
        BrdfData bdata = EvaluateAndPdf_COMBINED(sp, sv_cur.matID,
            sv_cur.n_s, sv_cur.n_g, s, sv_cur.o,
            sv_cur.Kd, sv_cur.Pr, sv_cur.Pm, sv_cur.etai, sv_cur.etat);

        if (bdata.pdf <= 1e-6f || dot(s, s) < 1e-12f)
            return false;

        float cosTheta = abs(dot(sv_cur.n_s, s));
        float3 w       = (bdata.val * cosTheta) / bdata.pdf;
        if (any(isnan(w)) || any(isinf(w)))
            return false;

        prefixThroughput *= w;

        // Trace next ray.
        float3 offsetN = (dot(s, sv_cur.n_g) >= 0.0f) ? sv_cur.n_g : -sv_cur.n_g;
        rayOrigin = offset_ray(sv_cur.x, offsetN);
        rayDir    = s;

        RayDesc ray;
        ray.Origin    = rayOrigin;
        ray.Direction = rayDir;
        ray.TMin      = 0.00001f;
        ray.TMax      = 10000.0f;

        uint   instID;
        uint   primID;
        float2 bcHit;
        if (!ShiftTraceClosest_Inline(ray, instID, primID, bcHit))
            return false;   // miss — prefix must land on a non-emissive surface

        float3 emission = GetEmissionFast(instID, primID);
        if (any(emission > 0.0f)) return false;   // hit emitter mid-prefix — shift fails

        // Build the next SurfaceVertex.  Note: view-origin for BuildVertex is
        // the previous vertex (the one we just bounced from).
        SurfaceVertex sv_next = BuildVertex(instID, primID, bcHit, sv_cur.x);
        // etai/etat not propagated (no volume tracking in v1); BuildVertex already
        // zeroed them to (1,1) which mirrors the common non-dielectric case.

        // Remember last pdf for diagnostics / external callers.
        pdf_last = bdata.pdf;

        sv_cur = sv_next;
    }

    sv_last = sv_cur;
    return true;
}

// -------------------------------------------------------------------------
// ShiftReplayToTerminal
//
// Fully replays the path from y_1 using bsdf_seed0 until it terminates at
// either an env miss or an emitter hit, up to SHIFT_MAX_BOUNCES bounces.
//
// On success (terminal matches expectMethod):
//   returns true and writes:
//     contrib = T_full · terminalRadiance
//   where T_full = product of BSDF·cosθ/pdf along every bounce AND the
//   terminal's sampling factor (1 for env miss, 1 for emitter hit — the
//   path-tracing factors for the terminal are already baked by convention).
//
// On failure (wrong terminal type, invalid sample, max bounces exceeded):
//   returns false and contrib = 0.
// -------------------------------------------------------------------------
inline bool ShiftReplayToTerminal(
    in  SurfaceVertex sv_y1,
    in  uint          bsdf_seed0,
    in  uint          expectMethod,     // BSDF_ENV_RP or BSDF_EMIT_RP
    out float3        contrib)
{
    contrib = float3(0, 0, 0);

    uint   bseed  = bsdf_seed0;
    float3 T_full = float3(1, 1, 1);

    SurfaceVertex sv_cur = sv_y1;

    [loop]
    for (uint b = 0; b < (uint)SHIFT_MAX_BOUNCES; ++b)
    {
        SamplingP sp = CalculateStrategyProbabilities(
            sv_cur.matID, sv_cur.o, sv_cur.n_s, sv_cur.etai, sv_cur.etat, sv_cur.Kd, sv_cur.Pm);
        float3 s = SampleBRDF(sp, sv_cur.matID, sv_cur.o, sv_cur.n_s, sv_cur.n_g,
                              sv_cur.Kd, sv_cur.Pr, sv_cur.Pm, bseed,
                              sv_cur.etai, sv_cur.etat, -1);
        BrdfData bdata = EvaluateAndPdf_COMBINED(sp, sv_cur.matID,
            sv_cur.n_s, sv_cur.n_g, s, sv_cur.o,
            sv_cur.Kd, sv_cur.Pr, sv_cur.Pm, sv_cur.etai, sv_cur.etat);

        if (bdata.pdf <= 1e-6f || dot(s, s) < 1e-12f)
            return false;

        float cosTheta = abs(dot(sv_cur.n_s, s));
        float3 w       = (bdata.val * cosTheta) / bdata.pdf;
        if (any(isnan(w)) || any(isinf(w)))
            return false;

        T_full *= w;

        // Trace.
        float3 offsetN = (dot(s, sv_cur.n_g) >= 0.0f) ? sv_cur.n_g : -sv_cur.n_g;
        RayDesc ray;
        ray.Origin    = offset_ray(sv_cur.x, offsetN);
        ray.Direction = s;
        ray.TMin      = 0.00001f;
        ray.TMax      = 10000.0f;

        uint   instID;
        uint   primID;
        float2 bcHit;
        bool   anyHit = ShiftTraceClosest_Inline(ray, instID, primID, bcHit);

        if (!anyHit)
        {
            // Env miss — accept only for BSDF_ENV_RP.
            if (expectMethod != RC_METHOD_BSDF_ENV_RP) return false;
            float3 envL = EvalMissState(s, float3(0, 0, 0));
            contrib = T_full * envL;
            return true;
        }

        float3 emission = GetEmissionFast(instID, primID);
        if (any(emission > 0.0f))
        {
            // Emitter hit — accept only for BSDF_EMIT_RP.
            if (expectMethod != RC_METHOD_BSDF_EMIT_RP) return false;
            contrib = T_full * emission;
            return true;
        }

        // Non-terminal surface — continue the walk.
        sv_cur = BuildVertex(instID, primID, bcHit, sv_cur.x);
    }

    // Exceeded SHIFT_MAX_BOUNCES without reaching the expected terminal.
    return false;
}

// -------------------------------------------------------------------------
// ShiftGI — method-aware hybrid shift.
//
// Shifts the stored sample (described by the reservoir fields) from its
// original domain into the current pixel's domain rooted at sv_y1.
//
// On success, returns the shifted radiance contribution.  On failure
// (visibility, BSDF=0, replay diverged, etc.) returns 0.
//
// Outputs:
//   Jn  — Jacobian numerator (PSS).  For HAS_RECON methods this is
//         p^y_{k-1}(ω_{k-1}) · G(y_{k-1}→x_k) · p^y_k(ω_k), matching the
//         formula in §4.  For _RP methods it is 1.
//   J   — Jn / J_can, or 1 for _RP methods.  This is what callers must
//         multiply into p_hat for correct RIS weighting.
// -------------------------------------------------------------------------
inline float3 ShiftGI(
    // Current-pixel x_1
    in SurfaceVertex sv_y1,

    // Stored-reservoir classification
    in uint   method,
    in uint   k,
    in uint   seed0,

    // Stored reconnection vertex (x_k)
    in float3 xk_pos, in float3 xk_ns, in float3 xk_ng,
    in uint   matID_k, in uint objID_k, in float2 uv_k,
    in float  etai_k, in float etat_k,
    in float3 xk_Kd,  in float xk_Pr, in float xk_Pm,

    // Stored tail
    in float3 L2, in float3 V2,
    in float  J_can,

    // For RC_METHOD_*_ATRC: stored lightPdf (= J_gi.x) substituted for the
    // (degenerate) BSDF pdf at x_k inside ReconnectGI.  Pass 0 for any other
    // method — ReconnectGI then recomputes PDF2 from the BSDF as usual.
    in float  pdfx2_atrc,

    // Outputs
    out float Jn,
    out float J)
{
    Jn = 0.0f;
    J  = 0.0f;

    if (method == RC_METHOD_INVALID) return float3(0, 0, 0);

    // ── Replay-only terminals ────────────────────────────────────────
    if (method == RC_METHOD_BSDF_ENV_RP || method == RC_METHOD_BSDF_EMIT_RP)
    {
        float3 c;
        bool ok = ShiftReplayToTerminal(sv_y1, seed0, method, c);
        if (!ok) return float3(0, 0, 0);
        Jn = 1.0f;
        J  = 1.0f;
        return c;
    }

    // ── All remaining methods: replay prefix to y_{k-1}  ─────────────
    // (k == 2 → no bounces, y_{k-1} = y_1.)
    SurfaceVertex sv_km1;
    float3        prefixT;
    float         pdfPrefixLast;
    if (!ShiftReplayPrefix(sv_y1, seed0, k, sv_km1, prefixT, pdfPrefixLast))
        return float3(0, 0, 0);

    // ── NEE_RP / SUN_RP: connect y_{k-1} → stored emitter (no recon BSDF at x_k)
    if (method == RC_METHOD_NEE_RP || method == RC_METHOD_SUN_RP)
    {
        float3 toL;
        float  dist;
        float  tMax;

        if (method == RC_METHOD_SUN_RP)
        {
            // x_k slot stores sun direction (unit-length, "from surface toward sun").
            toL  = xk_pos;
            dist = 1.0f;        // not used
            tMax = 10000.0f;
        }
        else
        {
            float3 d = xk_pos - sv_km1.x;
            dist = length(d);
            if (dist <= EPSILON) return float3(0, 0, 0);
            toL  = d / dist;
            tMax = dist * 0.999f;
        }

        float NdotL = dot(sv_km1.n_s, toL);
        if (NdotL <= 1e-6f) return float3(0, 0, 0);

        if (!IsVisible(sv_km1.x, sv_km1.n_g, toL, tMax))
            return float3(0, 0, 0);

        // BSDF at y_{k-1} toward the stored emitter direction.
        SamplingP sp = CalculateStrategyProbabilities(
            sv_km1.matID, sv_km1.o, sv_km1.n_s, sv_km1.etai, sv_km1.etat, sv_km1.Kd, sv_km1.Pm);
        BrdfData bd = EvaluateAndPdf_COMBINED(sp, sv_km1.matID,
            sv_km1.n_s, sv_km1.n_g, toL, sv_km1.o,
            sv_km1.Kd, sv_km1.Pr, sv_km1.Pm, sv_km1.etai, sv_km1.etat);

        // L2 is stored as emission / lightPdf (see raygen NEE_RP/SUN_RP blocks).
        // Shifted-domain contribution at y_{k-1} then reads:
        //   r = T_prefix · BSDF_y(y_{k-1}→light) · cos(n_{k-1}, toL) · emission / lightPdf
        // which is exactly prefixT · bd.val · NdotL · L2.
        float3 r = prefixT * bd.val * NdotL * L2;
        if (any(isnan(r)) || any(isinf(r))) return float3(0, 0, 0);

        Jn = 1.0f;
        J  = 1.0f;
        return r;
    }

    // ── HAS_RECON (BSDF_ENV / BSDF_EMIT / NEE / SUN / *_ATRC): reconnect y_{k-1} → x_k ─
    //
    // We reuse ReconnectGI's math, substituting y_{k-1} for x_1 and folding
    // the replayed prefix throughput over the result.  ReconnectGI already
    // handles the J_can ratio if applyJ = true.
    //
    // For *_ATRC methods: V2 stores -L_dir (toward the light) and pdfx2_atrc
    // holds the light's sampling pdf, so ReconnectGI's F2/G2/PDF2 evaluate
    // the BSDF/cos at x_k for the NEE link direction with PDF2 = lightPdf
    // (not BSDF pdf).  Plus we still need a shadow ray from x_k to the light
    // for sun (delta-direction at infinity) or to the stored light position.
    {
        const bool isAtRc = RC_IsAtRc(method);

        float3 c = ReconnectGI(
            sv_km1.x, sv_km1.n_s, sv_km1.n_g, sv_km1.o, sv_km1.matID,
            sv_km1.Kd, sv_km1.Pr, sv_km1.Pm, sv_km1.etai, sv_km1.etat,
            matID_k, xk_pos, xk_ns, xk_ng, L2, V2,
            xk_Kd, xk_Pr, xk_Pm, etai_k, etat_k,
            isAtRc ? pdfx2_atrc : 0.0f,   // _ATRC: pass stored lightPdf;  others: ReconnectGI recomputes
            J_can, true,
            Jn, J);

        // Visibility y_{k-1} → x_k  (skip for sun/env sentinels in HAS_RECON variants
        // that store the terminal as x_k — only the legacy SUN_RP/ENV_RP path uses
        // those sentinels and goes through the _RP branch above).
        bool sentinel = (objID_k == 0xFFFFFFFEu) || (objID_k == 0xFFFFFFFFu);
        if (!sentinel)
        {
            float3 d  = xk_pos - sv_km1.x;
            float  cd = length(d);
            float  vis = (cd > EPSILON &&
                          IsVisible(sv_km1.x, sv_km1.n_g, d / cd, cd * 0.999f)) ? 1.0f : 0.0f;
            c *= vis;
        }

        // No extra x_k → light visibility check needed for _ATRC: the light and x_k
        // are both canonical-stored values (unchanged by the shift), and visibility
        // was validated at candidate generation in raygen.

        if (any(isnan(c)) || any(isinf(c))) return float3(0, 0, 0);

        return prefixT * c;
    }
}

#endif // SHIFT_GI_V8_HLSLI
