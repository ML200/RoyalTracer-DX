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
        dx::HitObject hitObj = TraceRay_Custom(SceneBVH, ray, RAY_FLAG_NONE, 0xFF);

        // Prefix must land on a non-emissive surface to reach y_{k-1}.
        if (!hitObj.IsHit()) return false;

        uint instID = hitObj.GetInstanceIndex();
        uint primID = FlatPrimID(instID, hitObj.GetGeometryIndex(), hitObj.GetPrimitiveIndex());

        float3 emission = GetEmissionFast(instID, primID);
        if (any(emission > 0.0f)) return false;   // hit emitter mid-prefix — shift fails

        BuiltInTriangleIntersectionAttributes attr;
        hitObj.GetAttributes(attr);

        // Build the next SurfaceVertex.  Note: view-origin for BuildVertex is
        // the previous vertex (the one we just bounced from).
        SurfaceVertex sv_next = BuildVertex(instID, primID, attr.barycentrics, sv_cur.x);
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
        dx::HitObject hitObj = TraceRay_Custom(SceneBVH, ray, RAY_FLAG_NONE, 0xFF);

        if (!hitObj.IsHit())
        {
            // Env miss — accept only for BSDF_ENV_RP.
            if (expectMethod != RC_METHOD_BSDF_ENV_RP) return false;
            float3 envL = EvalMissState(s, float3(0, 0, 0));
            contrib = T_full * envL;
            return true;
        }

        uint instID = hitObj.GetInstanceIndex();
        uint primID = FlatPrimID(instID, hitObj.GetGeometryIndex(), hitObj.GetPrimitiveIndex());

        float3 emission = GetEmissionFast(instID, primID);
        if (any(emission > 0.0f))
        {
            // Emitter hit — accept only for BSDF_EMIT_RP.
            if (expectMethod != RC_METHOD_BSDF_EMIT_RP) return false;
            contrib = T_full * emission;
            return true;
        }

        // Non-terminal surface — continue the walk.
        BuiltInTriangleIntersectionAttributes attr;
        hitObj.GetAttributes(attr);
        sv_cur = BuildVertex(instID, primID, attr.barycentrics, sv_cur.x);
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

    // ── HAS_RECON (BSDF_ENV / BSDF_EMIT / NEE / SUN): reconnect y_{k-1} → x_k ─
    //
    // We reuse ReconnectGI's math, substituting y_{k-1} for x_1 and folding
    // the replayed prefix throughput over the result.  ReconnectGI already
    // handles the J_can ratio if applyJ = true.
    {
        float3 c = ReconnectGI(
            sv_km1.x, sv_km1.n_s, sv_km1.n_g, sv_km1.o, sv_km1.matID,
            sv_km1.Kd, sv_km1.Pr, sv_km1.Pm, sv_km1.etai, sv_km1.etat,
            matID_k, xk_pos, xk_ns, xk_ng, L2, V2,
            xk_Kd, xk_Pr, xk_Pm, etai_k, etat_k,
            0.0f /* pdfx2 — let ReconnectGI recompute */,
            J_can, true,
            Jn, J);

        // Visibility y_{k-1} → x_k  (skip for sun terminals at infinity).
        bool sentinel = (objID_k == 0xFFFFFFFEu) || (objID_k == 0xFFFFFFFFu);
        if (!sentinel)
        {
            float3 d  = xk_pos - sv_km1.x;
            float  cd = length(d);
            float  vis = (cd > EPSILON &&
                          IsVisible(sv_km1.x, sv_km1.n_g, d / cd, cd * 0.999f)) ? 1.0f : 0.0f;
            c *= vis;
        }

        if (any(isnan(c)) || any(isinf(c))) return float3(0, 0, 0);

        return prefixT * c;
    }
}

#endif // SHIFT_GI_V8_HLSLI
