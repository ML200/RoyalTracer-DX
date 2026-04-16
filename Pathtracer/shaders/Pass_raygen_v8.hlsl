#include "Includes_raygen_v8.hlsli"

#ifndef MAX_BOUNCES
#define MAX_BOUNCES 10
#endif

#ifndef MEDIUM_INVALID
#define MEDIUM_INVALID 0xFFFFFFFFu
#endif

// NOTE: Sun and environment/miss handling has been commented out in this
// revision to isolate a sun-bias source.  Affected sites:
//   • depth==0 miss: sky + sun primary-visibility write to scratch
//   • depth>=1 miss: env-map GI reservoir block (already #if 0'd; body still commented)
//   • Sun NEE at every depth (DI write at d=0, SUN / SUN_ATRC / SUN_RP GI writes at d>=1)
// Each disabled block is marked `// [SUN/MISS DISABLED]`.  Point-light NEE, emitter
// hits, and the BSDF walk itself are untouched.

[shader("raygeneration")]
void Pass_raygen_v8()
{
    // pixel / imgSize: recomputed from intrinsics at each use to avoid keeping them live across trace
    {
        uint2 pixel   = DispatchRaysIndex().xy;
        uint2 imgSize = DispatchRaysDimensions().xy;
        uint pixelIdx = MapPixelID(imgSize, pixel);
        storeReservoirDI(g_Reservoirs_current_di, pixelIdx, (Reservoir_DI)0);
        store_wsum_di(g_Reservoirs_current_di, pixelIdx, 0.0f);
        store_W_di(g_Reservoirs_current_di, pixelIdx, 0.0f);
        store_phat_di(g_Reservoirs_current_di, pixelIdx, 0.0f);

        storeReservoirGI(g_Reservoirs_current_gi, pixelIdx, (Reservoir_GI)0);
        store_wsum_gi(g_Reservoirs_current_gi, pixelIdx, 0.0f);
        store_W_gi(g_Reservoirs_current_gi, pixelIdx, 0.0f);
        store_F_gi(g_Reservoirs_current_gi, pixelIdx, 0u);
        store_M_gi(g_Reservoirs_current_gi, pixelIdx, 0u);
        store_Tpost_gi(g_Reservoirs_current_gi, pixelIdx, 1.0f);
        store_seed_gi  (g_Reservoirs_current_gi, pixelIdx, 0u);
        store_k_gi     (g_Reservoirs_current_gi, pixelIdx, 0u);
        store_method_gi(g_Reservoirs_current_gi, pixelIdx, RC_METHOD_INVALID);

        gScratchPing[uint3(pixel, 3)] = float4(0, 0, 0, 0);
    }

    // ── Path state ─────────────────────────────────────────────────────
    uint   seed       = initRandomData(DispatchRaysIndex().xy, uint2(8, 4), time, 1u);
    // Dedicated BSDF stream — consumed ONLY by BSDF lobe choice + direction.
    // RR, NEE/sun sampling, and the RIS acceptance random stay on `seed`.
    uint   bsdf_seed       = initRandomData(DispatchRaysIndex().xy, uint2(8, 4), time, 2u);
    const uint bsdf_seed0  = bsdf_seed;   // immutable snapshot for prefix replay
    float3 rayOrigin  = InitOrigin();
    float3 rayDir     = InitDirection(DispatchRaysIndex().xy, float2(DispatchRaysDimensions().xy), seed);
    uint   throughputPk = PackRGB9E5(float3(1, 1, 1));   // compressed: 3 floats → 1 uint
    uint   prevNormalPk = PackNormal(float3(0, 1, 0));    // compressed: 3 floats → 1 uint
    float  prev_pdf   = 1.0f;

    // ── Hybrid-shift bookkeeping ──────────────────────────────────────
    float primary_fp_thresh = 0.0f;    // RHS of Eq. 5, set at depth 0
    float prev_alpha        = 0.0f;    // α at x_{depth}  (→ α_{k-1} at next iter's criterion)
    uint  k_recon           = 0u;      // first depth at which criterion passed on BSDF walk
                                       //   (0 = not found yet)

    VolumeIOR_Packed viorP;
    VolumeAux_Packed aiorP;
    {
        VolumeIOR v0 = InitVolumeIOR();
        VolumeAux a0 = InitVolumeAux();
        viorP.raw   = PackIORStackAndPtr(v0.ior_stack, v0.pointer);
        aiorP.mat32 = PackMatStack(a0.matID_stack);
        aiorP.obj32 = PackObjStack(a0.objID_stack);
    }

    // ── Bounce loop ────────────────────────────────────────────────────
    [loop]
    for (int depth = 0; depth < MAX_BOUNCES; ++depth)
    {
        // Validate ray state
        if (any(isnan(rayDir)) || any(isinf(rayDir)) || dot(rayDir, rayDir) < 1e-12f ||
            any(isnan(rayOrigin)) || any(isinf(rayOrigin)))
            break;

        // Build RayDesc locally — TMin/TMax are constants, no need to keep them live
        RayDesc ray;
        ray.Origin    = rayOrigin;
        ray.Direction = rayDir;
        ray.TMin      = 0.00001f;
        ray.TMax      = 10000.0f;
        dx::HitObject hitObj = TraceRay_Custom(SceneBVH, ray, RAY_FLAG_NONE, 0xFF);

        // Recompute pixel/imgSize from intrinsics — free, avoids keeping them live across trace
        const uint2 pixel   = DispatchRaysIndex().xy;
        const uint2 imgSize = DispatchRaysDimensions().xy;

        // ── Miss ───────────────────────────────────────────────────────
        if (!hitObj.IsHit())
        {
            if (depth == 0)
            {
                // [SUN/MISS DISABLED] Primary-ray sky + sun-disk visibility.
                // With this commented out, background pixels display whatever
                // was in gScratchPing before this pass (typically cleared / zero).
                // --- original code ---
                // {
                //     float3 sun = EvaluateSun(rayDir);
                //     float3 skyL1 = EvalMissState(rayDir, sun);
                //     if (length(sun) > 0.0f) skyL1 = sun;
                //     gScratchPing[uint3(pixel, 1)] = float4(skyL1, 0);
                //     gScratchPing[uint3(pixel, 2)] = float4(skyL1, 0);
                //     store_sky(g_sample_current, MapPixelID(imgSize, pixel));
                // }
                break;
            }

            // [SUN/MISS DISABLED] Env-miss GI reservoir emission at depth >= 1.
            // Everything below this point in the miss branch is commented out;
            // the original already lived under `#if 0`, but leaving the body as
            // live code would keep UnpackRGB9E5 / EvalMissState / PackRGB9E5
            // resident in the function and on the compiled stack.
            // --- original code (previously #if 0'd) ---
            // float3 throughput = UnpackRGB9E5(throughputPk);
            // float3 envL   = EvalMissState(rayDir, float3(0,0,0));
            // float3 T_envL = throughput * envL;
            //
            // uint px = MapPixelID(imgSize, pixel);
            //
            // // DI reservoir: env map hit at depth 1
            // if (depth == 1)
            // {
            //     float p_hat = GetPHat(T_envL * prev_pdf);
            //     float wi    = p_hat / prev_pdf;
            //     if (UpdateReservoirDI_Infinite(g_Reservoirs_current_di, px, wi, rayDir, envL, 0xFFFFFFFFu, seed))
            //         store_phat_di(g_Reservoirs_current_di, px, p_hat);
            // }
            //
            // // GI reservoir: env map hit at depth >= 2 — BSDF_ENV or BSDF_ENV_RP.
            // if (depth >= 2)
            // {
            //     float  p_hat = GetPHat(T_envL);
            //     uint   F_pk  = PackRGB9E5(T_envL);
            //
            //     if (k_recon > 0u)
            //     {
            //         // Reconnection found along the walk → read x_k state stashed at G5-bis.
            //         uint   xk_obj  = load_objID_gi(g_Reservoirs_current_gi, px);
            //         uint   xk_mat  = load_matID_gi(g_Reservoirs_current_gi, px);
            //         float3 xk_pos  = load_x2_gi  (g_Reservoirs_current_gi, px, xk_obj);
            //         float3 xk_ns   = load_n2_s_gi(g_Reservoirs_current_gi, px, xk_obj);
            //         float3 xk_ng   = load_n2_g_gi(g_Reservoirs_current_gi, px, xk_obj);
            //         float2 xk_uv   = load_uv_gi  (g_Reservoirs_current_gi, px);
            //         float  xk_etai = load_etai_gi(g_Reservoirs_current_gi, px);
            //         float  xk_etat = load_etat_gi(g_Reservoirs_current_gi, px);
            //         float3 V2_new  = load_Vpost_gi(g_Reservoirs_current_gi, px);
            //         float3 tpost   = load_Tpost_gi(g_Reservoirs_current_gi, px);
            //         float2 J_new   = float2(0.0f, load_J_gi(g_Reservoirs_current_gi, px).y);
            //
            //         UpdateReservoirGI_Candidate(
            //             g_Reservoirs_current_gi, px, p_hat,
            //             xk_pos, xk_ns, xk_ng, xk_mat, xk_obj, xk_uv, xk_etai, xk_etat,
            //             envL * tpost, V2_new, J_new, F_pk,
            //             RC_METHOD_BSDF_ENV, k_recon, bsdf_seed0,
            //             seed);
            //     }
            //     else
            //     {
            //         // No reconnection vertex → BSDF_ENV_RP. Shift must replay entire path.
            //         UpdateReservoirGI_Candidate(
            //             g_Reservoirs_current_gi, px, p_hat,
            //             float3(0,0,0), float3(0,1,0), float3(0,1,0), 0u, 0u,
            //             float2(0,0), 1.0f, 1.0f,
            //             envL, -rayDir, float2(0.0f, 1.0f), F_pk,
            //             RC_METHOD_BSDF_ENV_RP, 0u, bsdf_seed0,
            //             seed);
            //     }
            // }
            break;
        }

        // ── Hit setup ──────────────────────────────────────────────────
        float  hitT   = hitObj.GetRayTCurrent();
        float3 hitPos = rayOrigin + rayDir * hitT;

        const uint instID = hitObj.GetInstanceIndex();
        const uint primID = FlatPrimID(instID, hitObj.GetGeometryIndex(), hitObj.GetPrimitiveIndex());
        uint matID  = GetMatIDFast(instID, primID);
        float2 iors = GetIORs_packed(viorP, aiorP, matID, instID);

        // Phantom surface: advance through null interface (consumes regular bounce budget)
        if (iors.y == 0.0f)
        {
            rayOrigin = hitPos;
            UpdateIORStack_packed(viorP, aiorP, matID, instID);
            continue;
        }

        uint mediumMatID = GetCurrentMediumMaterialID_packed(viorP, aiorP);

        BuiltInTriangleIntersectionAttributes attr;
        hitObj.GetAttributes(attr);
        HitInfo hinfo = EvalSurfaceState(instID, primID, attr.barycentrics, rayOrigin, depth);

        // Refetch material from UV
        float3 hitLocalKd; float hitLocalPr, hitLocalPm;
        RefetchMaterial(matID, hinfo.uv, hitLocalKd, hitLocalPr, hitLocalPm);

        // Volume absorption
        float3 absorptionTint = (mediumMatID != MEDIUM_INVALID)
            ? CalculateAbsorptionThroughput(materials[mediumMatID].Tf, hitT)
            : float3(1, 1, 1);

        float3 emission = GetEmissionFast(instID, primID);

        // ── Depth 0: store primary hit ─────────────────────────────────
        if (depth == 0)
        {
            uint px = MapPixelID(imgSize, pixel);
            bool isEmitter = any(emission > 0.0f);
            store_instID(g_sample_current, px, instID);
            store_primID(g_sample_current, px, primID, isEmitter);
            store_bary(g_sample_current, px, attr.barycentrics);
            store_etai_etat(g_sample_current, px, iors.x, iors.y);
            store_n1_g_world(g_sample_current, px, hinfo.hitGNormal, instID);
            store_n1_s_world(g_sample_current, px, hinfo.hitNormal, instID);
            store_uv(g_sample_current, px, hinfo.uv);

            // Hybrid-shift: RHS of Eq. 5 — constant per pixel for the whole path.
            {
                float cos_x1_cam = max(abs(dot(hinfo.hitNormal, -rayDir)), 1e-6f);
                primary_fp_thresh = RC_C_OVER_100 * 4.0f * 3.14159265f
                                  * (hitT * hitT) / cos_x1_cam;
            }

            if (isEmitter) {
                gScratchPing[uint3(pixel, 1)] = float4(emission, 0);
                gScratchPing[uint3(pixel, 2)] = float4(emission, 0);
            }

            // ── Trace perfect reflection ray for specular motion vectors ──
            // Uses inline RayQuery — no SER reorder point, avoids live-state spike.
            {
                float3 reflDir = reflect(rayDir, hinfo.hitNormal);
                float3 reflOrigin = offset_ray(hitPos, hinfo.hitGNormal);
                RayDesc reflRay;
                reflRay.Origin    = reflOrigin;
                reflRay.Direction = reflDir;
                reflRay.TMin      = 0.00001f;
                reflRay.TMax      = 10000.0f;

                RayQuery<RAY_FLAG_NONE> q;
                q.TraceRayInline(SceneBVH, RAY_FLAG_NONE, 0xFF, reflRay);
                while (q.Proceed())
                {
                    if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
                    {
                        uint cInstID = q.CandidateInstanceIndex();
                        uint cPrimID = FlatPrimID(cInstID, q.CandidateGeometryIndex(), q.CandidatePrimitiveIndex());
                        uint cMatID  = GetMatIDFast(cInstID, cPrimID);
                        float alpha  = materials[cMatID].alphaThreshold;
                        if (alpha < 1.0f)
                            q.CommitNonOpaqueTriangleHit();
                    }
                }

                if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
                {
                    float3 reflPos = reflOrigin + reflDir * q.CommittedRayT();
                    float3 virtualPos = reflPos - 2.0f * dot(reflPos - hitPos, hinfo.hitNormal) * hinfo.hitNormal;
                    gScratchPing[uint3(pixel, 4)] = float4(virtualPos, asfloat(instID));
                }
                else
                {
                    gScratchPing[uint3(pixel, 4)] = float4(0, 0, 0, asfloat(0xFFFFFFFFu));
                }
            }
        }

        // ── BSDF sample at current vertex (moved earlier so the dual-footprint
        //    criterion can run BEFORE NEE/Sun — letting depth=1 NEE/Sun emit a
        //    HAS_RECON-style _ATRC candidate with k=2 → zero replay bounces). ──
        SamplingP sp = CalculateStrategyProbabilities(matID, -rayDir, hinfo.hitNormal, iors.x, iors.y, hitLocalKd, hitLocalPm);
        float3 s = SampleBRDF(sp, matID, -rayDir, hinfo.hitNormal, hinfo.hitGNormal, hitLocalKd, hitLocalPr, hitLocalPm, bsdf_seed, iors.x, iors.y, GetVolumePtrFast_packed(viorP));
        BrdfData bdata = EvaluateAndPdf_COMBINED(sp, matID, hinfo.hitNormal, hinfo.hitGNormal, s, -rayDir, hitLocalKd, hitLocalPr, hitLocalPm, iors.x, iors.y);

        // ── Hybrid-shift criterion: BSDF link x_d → x_{d+1}.  Fire only on a
        //    non-emissive vertex (emitter terminates the path; can't be x_k). ──
        // Stash these so the NEE/Sun blocks below can recompute J_can_atrc when
        // they detect "criterion-just-fired" (k_recon == depth + 1).
        float  link_cos_xkm1      = 0.0f;
        float  link_cos_xk        = 0.0f;
        float  link_dist2         = 1.0f;

        if (k_recon == 0u && depth >= 1 && bdata.pdf > 1e-6f && !any(emission > 0.0f))
        {
            float3 prev_normal = UnpackNormal(prevNormalPk);
            link_cos_xk   = abs(dot(hinfo.hitGNormal, -rayDir));
            link_cos_xkm1 = abs(dot(prev_normal,       rayDir));
            link_dist2    = max(hitT * hitT, EPSILON);

            if (HybridReconnectionCriterion(prev_pdf, bdata.pdf,
                                            link_cos_xk, link_cos_xkm1, link_dist2,
                                            prev_alpha, primary_fp_thresh))
            {
                k_recon = (uint)(depth + 1);
                uint px = MapPixelID(imgSize, pixel);

                // Snapshot x_k surface data — same values as the legacy criterion site.
                SetReservoirGI_ConstHit (g_Reservoirs_current_gi, px, hitPos,
                                         hinfo.hitNormal, hinfo.hitGNormal, matID, instID);
                SetReservoirGI_UVAndIOR (g_Reservoirs_current_gi, px, hinfo.uv, iors.x, iors.y);

                // Default V_post / J_can for the regular HAS_RECON path.  NEE/Sun blocks
                // below overwrite these on accept of an _ATRC candidate (V_post = -L_dir,
                // J_can_atrc = prev_pdf · lightPdf · cos·cos / dist²).
                store_Vpost_gi(g_Reservoirs_current_gi, px, -s);
                float J_can_default = prev_pdf * (link_cos_xk / link_dist2) * bdata.pdf;
                store_Jy_gi(g_Reservoirs_current_gi, px, J_can_default);

                store_Tpost_gi(g_Reservoirs_current_gi, px, float3(1, 1, 1));
            }
        }

        // ── Emitter hit: BSDF-sampled light with MIS ──────────────────
        if (any(emission > 0.0f) && hinfo.lightID != 0xFFFFFFFFu)
        {
            float3 throughput = UnpackRGB9E5(throughputPk);
            float3 prevNormal = UnpackNormal(prevNormalPk);
            float lightPdfArea = LT_Pdf_LightTree_Area(rayOrigin, prevNormal, hinfo.lightID, instID);
            float cosLight     = max(dot(hinfo.hitNormal, -rayDir), 0.0f);
            float dist2        = max(hitT * hitT, EPSILON);
            float lightPdfSA   = (cosLight > EPSILON) ? (lightPdfArea * dist2 / cosLight) : 0.0f;
            float misWeight    = prev_pdf / max(prev_pdf + lightPdfSA, EPSILON);

            uint px = MapPixelID(imgSize, pixel);

            // DI: emitter at depth 1
            if (depth == 1)
            {
                float p_hat = GetPHat(throughput * emission * prev_pdf);
                float wi    = misWeight * p_hat / prev_pdf;
                if (UpdateReservoirDI_Fast(g_Reservoirs_current_di, px, wi, hitPos, hinfo.hitNormal, emission, instID, seed))
                    store_phat_di(g_Reservoirs_current_di, px, p_hat);
            }

            // GI: emitter at depth >= 2 — BSDF_EMIT or BSDF_EMIT_RP.
            if (depth >= 2)
            {
                float3 contrib_gi = throughput * emission;
                float  p_hat  = GetPHat(contrib_gi);
                uint   F_pk   = PackRGB9E5(contrib_gi);
                float  wi     = p_hat * misWeight;

                if (k_recon > 0u)
                {
                    // Reconnection set during BSDF walk → load x_k state stashed at G5-bis.
                    uint   xk_obj  = load_objID_gi(g_Reservoirs_current_gi, px);
                    uint   xk_mat  = load_matID_gi(g_Reservoirs_current_gi, px);
                    float3 xk_pos  = load_x2_gi  (g_Reservoirs_current_gi, px, xk_obj);
                    float3 xk_ns   = load_n2_s_gi(g_Reservoirs_current_gi, px, xk_obj);
                    float3 xk_ng   = load_n2_g_gi(g_Reservoirs_current_gi, px, xk_obj);
                    float2 xk_uv   = load_uv_gi  (g_Reservoirs_current_gi, px);
                    float  xk_etai = load_etai_gi(g_Reservoirs_current_gi, px);
                    float  xk_etat = load_etat_gi(g_Reservoirs_current_gi, px);
                    float3 V2_new  = load_Vpost_gi(g_Reservoirs_current_gi, px);
                    float3 tpost   = load_Tpost_gi(g_Reservoirs_current_gi, px);
                    float2 J_new   = float2(0.0f, load_J_gi(g_Reservoirs_current_gi, px).y);

                    UpdateReservoirGI_Candidate(
                        g_Reservoirs_current_gi, px, wi,
                        xk_pos, xk_ns, xk_ng, xk_mat, xk_obj, xk_uv, xk_etai, xk_etat,
                        emission * tpost, V2_new, J_new, F_pk,
                        RC_METHOD_BSDF_EMIT, k_recon, bsdf_seed0,
                        seed);
                }
                else
                {
                    // No reconnection → BSDF_EMIT_RP. Shift must replay entire path,
                    // succeeds iff replay terminates at an emitter.
                    UpdateReservoirGI_Candidate(
                        g_Reservoirs_current_gi, px, wi,
                        float3(0,0,0), float3(0,1,0), float3(0,1,0), 0u, 0u,
                        float2(0,0), 1.0f, 1.0f,
                        emission, -rayDir, float2(0.0f, 1.0f), F_pk,
                        RC_METHOD_BSDF_EMIT_RP, 0u, bsdf_seed0,
                        seed);
                }
            }
            break;
        }

        // ── NEE (point lights + sun) ──────────────────────────────────
        // Pack caller state to reduce live regs across IsVisible calls:
        //   hitLocalKd(3)+hitLocalPr(1)+hitLocalPm(1) → matPk(2 uints)
        //   hinfo.hitNormal(3) → hitNormalPk(1 uint) — IsVisible only needs hitGNormal
        //   throughput stays packed as throughputPk — decompress only after IsVisible
        bool performNEE = !(mediumMatID != MEDIUM_INVALID || materials[matID].Kd.w < EPSILON);

        uint matKdPk, matPrPmPk, hitNormalPk;
        if (performNEE)
        {
            matKdPk     = PackRGB9E5(hitLocalKd);
            matPrPmPk   = f32tof16_custom(hitLocalPr) | (f32tof16_custom(hitLocalPm) << 16u);
            hitNormalPk = PackNormal(hinfo.hitNormal);

            // Point light NEE
            {
                LT_LightSampleResult light = LT_SamplePointOnLight(hitPos, hinfo.hitNormal, seed);

                float3 toLight = light.position - hitPos;
                float  distSq  = dot(toLight, toLight);
                float  dist    = sqrt(distSq);
                float3 L       = toLight / dist;

                float cosSurf  = dot(hinfo.hitNormal, L);
                float cosLight = dot(light.normal, -L);

                if (cosSurf > 1e-6f && cosLight > 1e-6f && IsVisible(hitPos, hinfo.hitGNormal, L, dist * 0.999f))
                {
                    // Decompress after IsVisible — these were dead across the call
                    float3 lKd = UnpackRGB9E5(matKdPk);
                    float  lPr = f16tof32_custom(matPrPmPk & 0xFFFFu);
                    float  lPm = f16tof32_custom(matPrPmPk >> 16u);
                    float3 hitN = UnpackNormal(hitNormalPk);
                    float3 throughput = UnpackRGB9E5(throughputPk);

                    SamplingP sp_nee = CalculateStrategyProbabilities(matID, -rayDir, hitN, iors.x, iors.y, lKd, lPm);
                    BrdfData bdataNEE = EvaluateAndPdf_COMBINED(sp_nee, matID, hitN, hinfo.hitGNormal, L, -rayDir, lKd, lPr, lPm, iors.x, iors.y);

                    float lightPdf = light.pdfSolidAngle;
                    float bsdfPdf  = bdataNEE.pdf;

                    if (lightPdf > 0.0f && bsdfPdf > 0.0f)
                    {
                        float misWeight = lightPdf / (lightPdf + bsdfPdf);

                        uint px = MapPixelID(imgSize, pixel);

                        // DI: NEE at depth 0
                        if (depth == 0)
                        {
                            float p_hat = GetPHat(throughput * light.emission * bdataNEE.val * cosSurf);
                            float wi    = (lightPdf > 1e-20f) ? (misWeight * p_hat / lightPdf) : 0.0f;
                            if (UpdateReservoirDI_Fast(g_Reservoirs_current_di, px, wi, light.position, light.normal, light.emission, light.objID, seed))
                                store_phat_di(g_Reservoirs_current_di, px, p_hat);
                        }

                        // GI: NEE at depth >= 1 — NEE vs NEE_RP based on per-candidate criterion.
                        if (depth >= 1)
                        {
                            float3 contrib = throughput * light.emission * bdataNEE.val * cosSurf / lightPdf;
                            float  p_hat   = GetPHat(contrib);
                            uint   F_pk    = PackRGB9E5(contrib);
                            float  wi      = p_hat * misWeight;

                            if (k_recon > 0u && (uint)depth + 1u == k_recon)
                            {
                                // CASE A_ATRC: criterion fired THIS iter (above) → x_k = current
                                // vertex (= NEE source).  ReconnectGI re-evaluates BSDF at x_k for
                                // direction -V2 = toward light, with PDF2 = stored lightPdf.
                                // Storage:
                                //   x_k state was already snapshot to the reservoir at criterion fire.
                                //   V2 = -L (overrides default -s).  L2 = light.emission (no extra factors).
                                //   J.x = lightPdf,  J.y = J_can_atrc = prev_pdf · lightPdf · cos·cos / dist²
                                //   k = depth + 1 (= k_recon) → ShiftReplayPrefix does k-2 = depth-1 bounces.
                                // Single-cosine G to match ReconnectGI's Jn = PDF1·PDF2·cos_xk/dist²
                                // (so canonical J = Jn/J_can = 1 at y=x).
                                float  J_can_atrc = prev_pdf * lightPdf * link_cos_xk / link_dist2;
                                float2 J_new      = float2(lightPdf, J_can_atrc);

                                UpdateReservoirGI_Candidate(
                                    g_Reservoirs_current_gi, px, wi,
                                    hitPos, hinfo.hitNormal, hinfo.hitGNormal, matID, instID,
                                    hinfo.uv, iors.x, iors.y,
                                    light.emission, -L, J_new, F_pk,
                                    RC_METHOD_NEE_ATRC, k_recon, bsdf_seed0,
                                    seed);
                            }
                            else if (k_recon > 0u)
                            {
                                // CASE A: reconnect at earlier BSDF-walk vertex x_{k_recon}.
                                // Tail: BSDF bounces from x_k to x_{depth+1}, then NEE link (baked into L2 via tpost).
                                uint   xk_obj  = load_objID_gi(g_Reservoirs_current_gi, px);
                                uint   xk_mat  = load_matID_gi(g_Reservoirs_current_gi, px);
                                float3 xk_pos  = load_x2_gi  (g_Reservoirs_current_gi, px, xk_obj);
                                float3 xk_ns   = load_n2_s_gi(g_Reservoirs_current_gi, px, xk_obj);
                                float3 xk_ng   = load_n2_g_gi(g_Reservoirs_current_gi, px, xk_obj);
                                float2 xk_uv   = load_uv_gi  (g_Reservoirs_current_gi, px);
                                float  xk_etai = load_etai_gi(g_Reservoirs_current_gi, px);
                                float  xk_etat = load_etat_gi(g_Reservoirs_current_gi, px);
                                float3 V2_new  = load_Vpost_gi(g_Reservoirs_current_gi, px);
                                float3 tpost   = load_Tpost_gi(g_Reservoirs_current_gi, px);
                                float2 J_new   = float2(0.0f, load_J_gi(g_Reservoirs_current_gi, px).y);

                                // This NEE fires strictly past x_k → bake the NEE link into the tail.
                                float3 L2_new = light.emission * tpost * (bdataNEE.val * cosSurf / lightPdf);

                                UpdateReservoirGI_Candidate(
                                    g_Reservoirs_current_gi, px, wi,
                                    xk_pos, xk_ns, xk_ng, xk_mat, xk_obj, xk_uv, xk_etai, xk_etat,
                                    L2_new, V2_new, J_new, F_pk,
                                    RC_METHOD_NEE, k_recon, bsdf_seed0,
                                    seed);
                            }
                            else
                            {
                                // x_k is the LIGHT here (no BSDF-walk reconnection vertex was found).
                                // Shift mechanics = prefix replay + direct emitter connection.
                                // We always use NEE_RP; the per-candidate criterion (currently
                                // unused) could later parameterise the stored Jacobian.
                                //
                                // L2 is stored pre-divided by lightPdf so the shift's
                                //   r = prefixT * bd_y * NdotL_y * L2
                                // equals the expected  prefixT · BSDF · cos · emission / lightPdf.
                                float cos_xk      = max(cosLight, 0.0f);
                                float3 L2_store   = light.emission / max(lightPdf, EPSILON);
                                float3 V2_new     = -L;
                                float  J_can      = bsdfPdf * (cos_xk / max(distSq, EPSILON)) * lightPdf;
                                float2 J_new      = float2(0.0f, J_can);
                                uint   k_cand     = (uint)(depth + 2);   // replay length for the shift

                                UpdateReservoirGI_Candidate(
                                    g_Reservoirs_current_gi, px, wi,
                                    light.position, light.normal, light.normal,
                                    0u /*matID unused for light terminals*/, light.objID,
                                    float2(0, 0), 1.0f, 1.0f,
                                    L2_store, V2_new, J_new, F_pk,
                                    RC_METHOD_NEE_RP, k_cand, bsdf_seed0,
                                    seed);
                            }
                        }
                    }
                }
            }

            // [SUN/MISS DISABLED] Sun NEE block — covers DI write at depth 0 and
            // SUN / SUN_ATRC / SUN_RP GI candidates at depth >= 1.  The block still
            // advances `seed` by 2 floats in the original (via `rSun`); to preserve
            // the RIS random sequence for point-light candidates in the same frame,
            // we could leave that `seed` advance in.  For a pure bias-isolation pass
            // it's safer to keep the sequence unchanged (so point-light reservoirs
            // don't shift their acceptance randoms), so the 2-float consume is
            // preserved below; the rest of the block is gone.
            {
                // Preserve random-stream advancement so point-light candidate
                // acceptance randoms downstream in the frame are identical to
                // the enabled-sun build.  Remove these two calls too if you want
                // the sun-related RNG steps elided entirely.
                float __sun_r0_unused = RandomFloatSingle(seed); (void)__sun_r0_unused;
                float __sun_r1_unused = RandomFloatSingle(seed); (void)__sun_r1_unused;
            }
            // --- original sun-NEE code ---
            // {
            //     float2 rSun = float2(RandomFloatSingle(seed), RandomFloatSingle(seed));
            //     SunSampleResult sun = SampleSun(rSun);
            //     float3 hitN_sun = UnpackNormal(hitNormalPk);
            //     float NdotL = dot(hitN_sun, sun.direction);
            //
            //     if (NdotL > 1e-6f && IsVisible(hitPos, hinfo.hitGNormal, sun.direction, 10000.0f))
            //     {
            //         // Decompress after IsVisible
            //         float3 lKd = UnpackRGB9E5(matKdPk);
            //         float  lPr = f16tof32_custom(matPrPmPk & 0xFFFFu);
            //         float  lPm = f16tof32_custom(matPrPmPk >> 16u);
            //         float3 hitN = UnpackNormal(hitNormalPk);
            //         float3 throughput = UnpackRGB9E5(throughputPk);
            //
            //         SamplingP sp_nee = CalculateStrategyProbabilities(matID, -rayDir, hitN, iors.x, iors.y, lKd, lPm);
            //         BrdfData bdataNEE = EvaluateAndPdf_COMBINED(sp_nee, matID, hitN, hinfo.hitGNormal, sun.direction, -rayDir, lKd, lPr, lPm, iors.x, iors.y);
            //
            //         float lightPdf = sun.pdf;
            //         float bsdfPdf  = bdataNEE.pdf;
            //
            //         if (lightPdf > 0.0f && bsdfPdf > 0.0f)
            //         {
            //             float3 contrib = throughput * NdotL * sun.radiance * bdataNEE.val / lightPdf;
            //             float misWeight = lightPdf / (lightPdf + bsdfPdf);
            //
            //             // DI: sun at depth 0 — write directly, bypass ReSTIR
            //             if (depth == 0)
            //             {
            //                 gScratchPing[uint3(pixel, 3)] = float4(misWeight * contrib, 0);
            //             }
            //
            //             // GI: sun at depth >= 1 — SUN vs SUN_RP based on per-candidate criterion.
            //             if (depth >= 1)
            //             {
            //                 uint px = MapPixelID(imgSize, pixel);
            //                 float  p_hat = GetPHat(contrib);
            //                 uint   F_pk  = PackRGB9E5(contrib);
            //                 float  wi    = p_hat;
            //
            //                 if (k_recon > 0u && (uint)depth + 1u == k_recon)
            //                 {
            //                     // CASE A_ATRC: criterion fired THIS iter (above) → x_k = current
            //                     // vertex (= NEE source).  Same pattern as NEE_ATRC above; for sun,
            //                     // the "G" geometry term degenerates to just cos_xkm1 (sun is at
            //                     // infinity → cos_xk/dist² → 1 in solid-angle measure), but we keep
            //                     // both cosines in the *atrc* J_can since x_{k-1}→x_k is a real surface
            //                     // link — the "lightness" only matters for the second pdf substitution.
            //                     // Single-cosine G to match ReconnectGI's Jn = PDF1·PDF2·cos_xk/dist²
            //                     // (so canonical J = Jn/J_can = 1 at y=x).
            //                     float  J_can_atrc = prev_pdf * lightPdf * link_cos_xk / link_dist2;
            //                     float2 J_new      = float2(lightPdf, J_can_atrc);
            //
            //                     UpdateReservoirGI_Candidate(
            //                         g_Reservoirs_current_gi, px, wi,
            //                         hitPos, hinfo.hitNormal, hinfo.hitGNormal, matID, instID,
            //                         hinfo.uv, iors.x, iors.y,
            //                         sun.radiance, -sun.direction, J_new, F_pk,
            //                         RC_METHOD_SUN_ATRC, k_recon, bsdf_seed0,
            //                         seed);
            //                 }
            //                 else if (k_recon > 0u)
            //                 {
            //                     // CASE A: reconnect at earlier BSDF-walk vertex x_{k_recon}.
            //                     uint   xk_obj  = load_objID_gi(g_Reservoirs_current_gi, px);
            //                     uint   xk_mat  = load_matID_gi(g_Reservoirs_current_gi, px);
            //                     float3 xk_pos  = load_x2_gi  (g_Reservoirs_current_gi, px, xk_obj);
            //                     float3 xk_ns   = load_n2_s_gi(g_Reservoirs_current_gi, px, xk_obj);
            //                     float3 xk_ng   = load_n2_g_gi(g_Reservoirs_current_gi, px, xk_obj);
            //                     float2 xk_uv   = load_uv_gi  (g_Reservoirs_current_gi, px);
            //                     float  xk_etai = load_etai_gi(g_Reservoirs_current_gi, px);
            //                     float  xk_etat = load_etat_gi(g_Reservoirs_current_gi, px);
            //                     float3 V2_new  = load_Vpost_gi(g_Reservoirs_current_gi, px);
            //                     float3 tpost   = load_Tpost_gi(g_Reservoirs_current_gi, px);
            //                     float2 J_new   = float2(0.0f, load_J_gi(g_Reservoirs_current_gi, px).y);
            //
            //                     float3 L2_new = sun.radiance * tpost * (bdataNEE.val * NdotL / lightPdf);
            //
            //                     UpdateReservoirGI_Candidate(
            //                         g_Reservoirs_current_gi, px, wi,
            //                         xk_pos, xk_ns, xk_ng, xk_mat, xk_obj, xk_uv, xk_etai, xk_etat,
            //                         L2_new, V2_new, J_new, F_pk,
            //                         RC_METHOD_SUN, k_recon, bsdf_seed0,
            //                         seed);
            //                 }
            //                 else
            //                 {
            //                     // x_k = sun (at infinity). Shift = prefix replay + direct sun connection.
            //                     // Always SUN_RP; per-candidate criterion value unused in v1.
            //                     //
            //                     // L2 is stored pre-divided by sun.pdf so the shift produces the correctly
            //                     // scaled contribution (see NEE_RP comment above).
            //                     float3 L2_store = sun.radiance / max(lightPdf, EPSILON);
            //                     float3 V2_new   = -sun.direction;
            //                     float  J_can    = bsdfPdf * max(NdotL, 1e-6f) * lightPdf;
            //                     float2 J_new    = float2(0.0f, J_can);
            //                     uint   k_cand   = (uint)(depth + 2);
            //
            //                     // objID = 0xFFFFFFFEu is the existing "sun" sentinel (see DI side).
            //                     UpdateReservoirGI_Candidate(
            //                         g_Reservoirs_current_gi, px, wi,
            //                         sun.direction, -sun.direction, -sun.direction,
            //                         0u, 0xFFFFFFFEu, float2(0, 0), 1.0f, 1.0f,
            //                         L2_store, V2_new, J_new, F_pk,
            //                         RC_METHOD_SUN_RP, k_cand, bsdf_seed0,
            //                         seed);
            //                 }
            //             }
            //         }
            //     }
            // }

        }

        // BSDF sample + criterion already ran above (before NEE/Sun) so the
        // _ATRC variants could fire for depth=1 NEE/Sun.  Just compute the
        // throughput weight from the cached `bdata` / `s`.
        float cosTheta = abs(dot(hinfo.hitNormal, s));
        float3 updateWeight = (bdata.pdf > 1e-6f)
            ? (bdata.val * absorptionTint * cosTheta) / bdata.pdf
            : float3(0, 0, 0);

        // IOR stack update on transmission
        if (dot(hinfo.hitGNormal, s) < 0.0f)
            UpdateIORStack_packed(viorP, aiorP, matID, instID);

        // Validate before continuing
        if (dot(s, s) < 1e-12f || bdata.pdf <= 1e-6f || any(isnan(updateWeight)) || any(isinf(updateWeight)))
            break;

        // Advance path state
        prev_pdf    = bdata.pdf;
        rayDir      = s;
        float3 offsetN = dot(s, hinfo.hitGNormal) >= 0.0f ? hinfo.hitGNormal : -hinfo.hitGNormal;
        rayOrigin   = offset_ray(hitPos, offsetN);

        // Update throughput: decompress → multiply → RR → recompress
        // Tpost must track the same weight as throughput (including RR) for depth >= 1
        {
            float3 throughput = UnpackRGB9E5(throughputPk) * updateWeight;
            float3 tpostWeight = updateWeight;  // weight factor for Tpost (before RR)

            // Russian Roulette (skip depth 1 to ensure at least one bounce)
            if (depth > 1)
            {
                float survivalProb = min(1.0f, Luma(throughput));
                if (RandomFloatSingle(seed) >= survivalProb) break;
                float rrBoost = 1.0f / max(survivalProb, 0.1f);
                throughput  *= rrBoost;
                tpostWeight *= rrBoost;  // Tpost must include RR survival weight
            }

            // Update post-reconnection throughput for GI.
            // Accumulates BSDF factors at x_{k+1}, x_{k+2}, …  The shift re-evaluates
            // the BSDF at x_k itself, so we start accumulating from the iter after k_recon
            // was set (at the moment tpostWeight first describes BSDF at x_{k+1}).
            // tpostWeight at iter d = BSDF factor at x_{d+1}. We want BSDF at x_{k+1} onward,
            // i.e. d+1 >= k+1, i.e. d >= k. Guard k_recon > 0 so we never write before
            // reconnection was decided.
            if (k_recon > 0u && (uint)depth >= k_recon)
            {
                uint px = MapPixelID(imgSize, pixel);
                float3 tpost = load_Tpost_gi(g_Reservoirs_current_gi, px);
                store_Tpost_gi(g_Reservoirs_current_gi, px, tpost * tpostWeight);
            }

            throughputPk = PackRGB9E5(throughput);
        }
        prevNormalPk = PackNormal(hinfo.hitNormal);
        // Effective α at x_{depth+1} for the criterion at the next iter (where this becomes α_{k-1}).
        // Paper-faithful α = GGX α = Pr².  For non-parametric / layered materials (paper §4.2 supplemental),
        // derive α from the strategy mix: Lambertian + sheen lobes are inherently "rough" (α≈1), GGX/coat
        // contribute Pr² weighted by their lobe strength.  This fixes purely-diffuse materials whose
        // stored Pr=0 was previously failing the α≥α_min check and forcing every neighbour shift to replay.
        prev_alpha   = max(hitLocalPr * hitLocalPr, sp.Pdiff + sp.Psheen);
    }

    // ── Final reservoir weight computation ─────────────────────────────
    uint pixelIdx = MapPixelID(DispatchRaysDimensions().xy, DispatchRaysIndex().xy);

    // DI
    {
        float p_hat = load_phat_di(g_Reservoirs_current_di, pixelIdx);
        float wsum  = load_wsum_di(g_Reservoirs_current_di, pixelIdx);
        float W     = (p_hat > 1e-6f && wsum > 0.0f) ? (wsum / p_hat) : 0.0f;
        store_W_di(g_Reservoirs_current_di, pixelIdx, W);
        store_M_di(g_Reservoirs_current_di, pixelIdx, 1);
    }

    // GI
    {
        uint  method = load_method_gi(g_Reservoirs_current_gi, pixelIdx);
        float F_gi   = GetPHat(UnpackRGB9E5(load_F_gi(g_Reservoirs_current_gi, pixelIdx)));
        float wsum   = load_wsum_gi(g_Reservoirs_current_gi, pixelIdx);
        float Wgi    = 0.0f;

        if (method != RC_METHOD_INVALID && F_gi > 1e-6f && wsum > 0.0f)
        {
            Wgi = wsum / F_gi;
            if (isnan(Wgi) || isinf(Wgi)) Wgi = 0.0f;
        }

        if (Wgi == 0.0f)
        {
            InvalidateReservoirGI_ShadingNormal(g_Reservoirs_current_gi, pixelIdx);
            store_method_gi(g_Reservoirs_current_gi, pixelIdx, RC_METHOD_INVALID);
            store_k_gi     (g_Reservoirs_current_gi, pixelIdx, 0u);
            store_seed_gi  (g_Reservoirs_current_gi, pixelIdx, 0u);
        }

        store_W_gi(g_Reservoirs_current_gi, pixelIdx, Wgi);
        store_M_gi(g_Reservoirs_current_gi, pixelIdx, 1u);
    }
}
