#include "Includes_raygen_v8.hlsli"

#ifndef MAX_BOUNCES
#define MAX_BOUNCES 10
#endif

#ifndef MEDIUM_INVALID
#define MEDIUM_INVALID 0xFFFFFFFFu
#endif

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
        store_F_mag_gi(g_Reservoirs_current_gi, pixelIdx, 0.0f);
        store_M_gi(g_Reservoirs_current_gi, pixelIdx, 0u);
        store_Tpost_gi(g_Reservoirs_current_gi, pixelIdx, 1.0f);

        gScratchPing[uint3(pixel, 3)] = float4(0, 0, 0, 0);
    }

    // ── Path state ─────────────────────────────────────────────────────
    uint   seed       = initRandomData(DispatchRaysIndex().xy, uint2(8, 4), time, 1u);
    float3 rayOrigin  = InitOrigin();
    float3 rayDir     = InitDirection(DispatchRaysIndex().xy, float2(DispatchRaysDimensions().xy), seed);
    uint   throughputPk = PackRGB9E5(float3(1, 1, 1));   // compressed: 3 floats → 1 uint
    uint   prevNormalPk = PackNormal(float3(0, 1, 0));    // compressed: 3 floats → 1 uint
    float  prev_pdf   = 1.0f;
    float  gi_pdf_product = 1.0f;

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
                {
                    float3 sun = EvaluateSun(rayDir);
                    float3 skyL1 = EvalMissState(rayDir, sun);
                    if (length(sun) > 0.0f) skyL1 = sun;
                    gScratchPing[uint3(pixel, 1)] = float4(skyL1, 0);
                    gScratchPing[uint3(pixel, 2)] = float4(skyL1, 0);
                    store_sky(g_sample_current, MapPixelID(imgSize, pixel));
                }
                break;
            }

            float3 throughput = UnpackRGB9E5(throughputPk);
            float3 envL   = EvalMissState(rayDir, float3(0,0,0));
            float3 T_envL = throughput * envL;

            uint px = MapPixelID(imgSize, pixel);

            // DI reservoir: env map hit at depth 1
            if (depth == 1)
            {
                float p_hat = GetPHat(T_envL * prev_pdf);
                float wi    = p_hat / prev_pdf;
                if (UpdateReservoirDI_Infinite(g_Reservoirs_current_di, px, wi, rayDir, envL, 0xFFFFFFFFu, seed))
                    store_phat_di(g_Reservoirs_current_di, px, p_hat);
            }

            // GI reservoir: env map hit at depth >= 2
            if (depth >= 2)
            {
                float  p_hat  = GetPHat(T_envL);
                float3 F_val  = T_envL * gi_pdf_product;
                float3 V2_new = (depth > 2) ? load_Vpost_gi(g_Reservoirs_current_gi, px) : -rayDir;
                float3 tpost  = load_Tpost_gi(g_Reservoirs_current_gi, px);

                if (UpdateReservoirGI_Fast(g_Reservoirs_current_gi, px, p_hat, envL * tpost, V2_new, seed))
                    store_F_combined_gi(g_Reservoirs_current_gi, px, F_val);
            }
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
            store_n1_s_world(g_sample_current, px, hinfo.hitNormal, instID);
            store_uv(g_sample_current, px, hinfo.uv);
            if (isEmitter) {
                gScratchPing[uint3(pixel, 1)] = float4(emission, 0);
                gScratchPing[uint3(pixel, 2)] = float4(emission, 0);
            }

            // ── Trace perfect reflection ray for specular motion vectors ──
            // Uses inline RayQuery — no SER reorder point, avoids live-state spike.
            {
                float3 reflDir = reflect(rayDir, hinfo.hitNormal);
                float3 reflOrigin = offset_ray(hitPos, hinfo.hitNormal);
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

            // GI: emitter at depth >= 2
            if (depth >= 2)
            {
                float3 V2_new = (depth == 2) ? (-rayDir) : load_Vpost_gi(g_Reservoirs_current_gi, px);
                float3 tpost  = load_Tpost_gi(g_Reservoirs_current_gi, px);
                float3 contrib_gi = throughput * emission;
                float  p_hat  = GetPHat(contrib_gi);
                float3 F_val  = contrib_gi * gi_pdf_product;
                float  wi     = p_hat * misWeight;

                if (UpdateReservoirGI_Fast(g_Reservoirs_current_gi, px, wi, emission * tpost, V2_new, seed))
                    store_F_combined_gi(g_Reservoirs_current_gi, px, F_val);
            }
            break;
        }

        // ── Depth 1: store GI reconnection vertex data ────────────────
        if (depth == 1)
        {
            uint px = MapPixelID(imgSize, pixel);
            SetReservoirGI_ConstHit(g_Reservoirs_current_gi, px, hitPos, hinfo.hitNormal, matID, instID, iors.y);
            SetReservoirGI_UV(g_Reservoirs_current_gi, px, hinfo.uv);
        }

        // ── Depth 2: store post-reconnection direction ────────────────
        if (depth == 2)
        {
            store_Vpost_gi(g_Reservoirs_current_gi, MapPixelID(imgSize, pixel), -rayDir);
        }

        // ── NEE (point lights + sun) ──────────────────────────────────
        // Pack caller state to reduce live regs across IsVisible calls:
        //   hitLocalKd(3)+hitLocalPr(1)+hitLocalPm(1) → matPk(2 uints)
        //   hinfo.hitNormal(3) → hitNormalPk(1 uint) — IsVisible needs shading normal
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

                if (cosSurf > 1e-6f && cosLight > 1e-6f && IsVisible(hitPos, hinfo.hitNormal, L, dist * 0.999f))
                {
                    // Decompress after IsVisible — these were dead across the call
                    float3 lKd = UnpackRGB9E5(matKdPk);
                    float  lPr = f16tof32_custom(matPrPmPk & 0xFFFFu);
                    float  lPm = f16tof32_custom(matPrPmPk >> 16u);
                    float3 hitN = UnpackNormal(hitNormalPk);
                    float3 throughput = UnpackRGB9E5(throughputPk);

                    SamplingP sp_nee = CalculateStrategyProbabilities(matID, -rayDir, hitN, iors.x, iors.y, lKd, lPm);
                    BrdfData bdataNEE = EvaluateAndPdf_COMBINED(sp_nee, matID, hitN, hinfo.hitNormal, L, -rayDir, lKd, lPr, lPm, iors.x, iors.y);

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

                        // GI: NEE at depth >= 1
                        if (depth >= 1)
                        {
                            float3 V2_new = (depth == 1) ? (-L) : load_Vpost_gi(g_Reservoirs_current_gi, px);
                            float3 contrib = throughput * light.emission * bdataNEE.val * cosSurf;
                            float  p_hat   = GetPHat(contrib);
                            float3 F_val   = contrib * gi_pdf_product;
                            float  wi      = p_hat * misWeight / lightPdf;
                            float3 tpost   = load_Tpost_gi(g_Reservoirs_current_gi, px);
                            if (depth > 1) tpost *= bdataNEE.val * cosSurf;

                            if (UpdateReservoirGI_Fast(g_Reservoirs_current_gi, px, wi, light.emission * tpost, V2_new, seed))
                                store_F_combined_gi(g_Reservoirs_current_gi, px, F_val);
                        }
                    }
                }
            }

            // Sun NEE
            {
                float2 rSun = float2(RandomFloatSingle(seed), RandomFloatSingle(seed));
                SunSampleResult sun = SampleSun(rSun);
                float3 hitN_sun = UnpackNormal(hitNormalPk);
                float NdotL = dot(hitN_sun, sun.direction);

                if (NdotL > 1e-6f && IsVisible(hitPos, hinfo.hitNormal, sun.direction, 10000.0f))
                {
                    // Decompress after IsVisible
                    float3 lKd = UnpackRGB9E5(matKdPk);
                    float  lPr = f16tof32_custom(matPrPmPk & 0xFFFFu);
                    float  lPm = f16tof32_custom(matPrPmPk >> 16u);
                    float3 hitN = UnpackNormal(hitNormalPk);
                    float3 throughput = UnpackRGB9E5(throughputPk);

                    SamplingP sp_nee = CalculateStrategyProbabilities(matID, -rayDir, hitN, iors.x, iors.y, lKd, lPm);
                    BrdfData bdataNEE = EvaluateAndPdf_COMBINED(sp_nee, matID, hitN, hinfo.hitNormal, sun.direction, -rayDir, lKd, lPr, lPm, iors.x, iors.y);

                    float lightPdf = sun.pdf;
                    float bsdfPdf  = bdataNEE.pdf;

                    if (lightPdf > 0.0f && bsdfPdf > 0.0f)
                    {
                        float3 contrib = throughput * NdotL * sun.radiance * bdataNEE.val / lightPdf;
                        float misWeight = lightPdf / (lightPdf + bsdfPdf);

                        // DI: sun at depth 0 — write directly, bypass ReSTIR
                        if (depth == 0)
                        {
                            gScratchPing[uint3(pixel, 3)] = float4(misWeight * contrib, 0);
                        }

                        // GI: sun at depth >= 1
                        if (depth >= 2)
                        {
                            uint px = MapPixelID(imgSize, pixel);
                            float3 V2_new = (depth == 1) ? (-sun.direction) : load_Vpost_gi(g_Reservoirs_current_gi, px);
                            float3 contrib_gi = throughput * NdotL * sun.radiance * bdataNEE.val;
                            float  p_hat  = GetPHat(contrib_gi);
                            float3 F_val  = contrib_gi * gi_pdf_product;
                            float  wi     = p_hat / lightPdf;
                            float3 tpost  = load_Tpost_gi(g_Reservoirs_current_gi, px);
                            if (depth > 1) tpost *= bdataNEE.val * NdotL;

                            if (UpdateReservoirGI_Fast(g_Reservoirs_current_gi, px, wi, sun.radiance * tpost, V2_new, seed))
                                store_F_combined_gi(g_Reservoirs_current_gi, px, F_val);
                        }
                    }
                }
            }

            // Unpack for BSDF sampling below
            hitLocalKd = UnpackRGB9E5(matKdPk);
            hitLocalPr = f16tof32_custom(matPrPmPk & 0xFFFFu);
            hitLocalPm = f16tof32_custom(matPrPmPk >> 16u);
            hinfo.hitNormal = UnpackNormal(hitNormalPk);
        }

        // ── Sample next direction (BSDF) ──────────────────────────────
        SamplingP sp = CalculateStrategyProbabilities(matID, -rayDir, hinfo.hitNormal, iors.x, iors.y, hitLocalKd, hitLocalPm);
        float3 s = SampleBRDF(sp, matID, -rayDir, hinfo.hitNormal, hinfo.hitNormal, hitLocalKd, hitLocalPr, hitLocalPm, seed, iors.x, iors.y, GetVolumePtrFast_packed(viorP));
        BrdfData bdata = EvaluateAndPdf_COMBINED(sp, matID, hinfo.hitNormal, hinfo.hitNormal, s, -rayDir, hitLocalKd, hitLocalPr, hitLocalPm, iors.x, iors.y);

        float cosTheta = abs(dot(hinfo.hitNormal, s));
        float3 updateWeight = (bdata.pdf > 1e-6f)
            ? (bdata.val * absorptionTint * cosTheta) / bdata.pdf
            : float3(0, 0, 0);

        // IOR stack update on transmission (use geometric normal to detect actual surface crossing)
        if (dot(hinfo.hitGNormal, s) < 0.0f)
            UpdateIORStack_packed(viorP, aiorP, matID, instID);

        // Validate before continuing
        if (dot(s, s) < 1e-12f || bdata.pdf <= 1e-6f || any(isnan(updateWeight)) || any(isinf(updateWeight)))
            break;

        // Advance path state
        prev_pdf       = bdata.pdf;
        gi_pdf_product = min(gi_pdf_product * bdata.pdf, 1e30f);
        rayDir      = s;
        float3 offsetN = dot(s, hinfo.hitGNormal) >= 0.0f ? hinfo.hitGNormal : -hinfo.hitGNormal;
        rayOrigin   = offset_ray(hitPos, offsetN);

        // Update throughput: decompress → multiply → RR → recompress
        {
            float3 throughput = UnpackRGB9E5(throughputPk) * updateWeight;
            float3 tpostWeight = bdata.val * absorptionTint * cosTheta;

            // Russian Roulette (skip depth 0 to ensure at least one bounce)
            if (depth > 0)
            {
                float survivalProb = min(1.0f, Luma(throughput));
                if (RandomFloatSingle(seed) >= survivalProb) break;
                float rrBoost = 1.0f / max(survivalProb, 0.1f);
                throughput  *= rrBoost;
                tpostWeight *= rrBoost;  // Tpost must include RR survival weight
            }

            // Update post-reconnection throughput for GI
            if (depth >= 2)
            {
                uint px = MapPixelID(imgSize, pixel);
                float3 tpost = load_Tpost_gi(g_Reservoirs_current_gi, px);
                store_Tpost_gi(g_Reservoirs_current_gi, px, tpost * tpostWeight);
            }

            throughputPk = PackRGB9E5(throughput);
        }
        prevNormalPk = PackNormal(hinfo.hitNormal);
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
        float F_gi  = load_F_mag_gi(g_Reservoirs_current_gi, pixelIdx);
        float wsum  = load_wsum_gi(g_Reservoirs_current_gi, pixelIdx);
        float Wgi   = 0.0f;

        if (F_gi > 1e-6f && wsum > 0.0f)
        {
            Wgi = wsum / F_gi;
            if (isnan(Wgi) || isinf(Wgi)) Wgi = 0.0f;
        }

        if (Wgi == 0.0f)
            InvalidateReservoirGI_ShadingNormal(g_Reservoirs_current_gi, pixelIdx);

        store_W_gi(g_Reservoirs_current_gi, pixelIdx, Wgi);
        store_M_gi(g_Reservoirs_current_gi, pixelIdx, 1u);
    }
}
