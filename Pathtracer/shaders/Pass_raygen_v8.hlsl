#include "Includes_v8.hlsli"

#ifndef MAX_BOUNCES
#define MAX_BOUNCES 10
#endif

#ifndef MEDIUM_INVALID
#define MEDIUM_INVALID 0xFFFFFFFFu
#endif

//─────────────────────────────────────────────────────────────────────────────
//  RAYGEN (unified DI + GI reservoir)
//
//  Direct-lighting samples (d = 2: x0 -> x1 -> light) now compete with GI
//  samples (d >= 3) in the same reservoir. DI candidates carry sentinel
//  matIDs (MATID_ENV_MISS or MATID_LIGHT_TRI) and their DI-specific x2
//  payload; GI candidates reuse the stashed depth-1 vertex state.
//
//  State kept in registers across iterations:
//    (x2_world, n2_world, uv_at_x2, matID_at_x2, objID_at_x2, eta_at_x2)
//                                               — depth-1 vertex, for GI samples
//    v2_world                                    — direction x3 -> x2, set at depth=2
//    tpost                                       — post-x2 integrand accumulator
//
//  Reservoir is accumulated locally and written via storeReservoir at the
//  end; wsum is separately stored (Pass_boil_gi reads it).
//─────────────────────────────────────────────────────────────────────────────

[shader("raygeneration")]
void Pass_raygen_v8()
{
    const uint2 pixel    = DispatchRaysIndex().xy;
    const uint2 imgSize  = DispatchRaysDimensions().xy;
    const uint  pixelIdx = MapPixelID(imgSize, pixel);

    // Clear scratch slots 1 and 3. Slot 1 carries primary emitter/sky hits
    // (written below on depth-0 emission); slot 3 carries the depth-0 sun
    // NEE. Without clearing, shading reads stale prior-frame values for
    // non-emitter, non-sky pixels (visible as light trails).
    gScratchPing[uint3(pixel, 1)] = float4(0, 0, 0, 0);
    gScratchPing[uint3(pixel, 3)] = float4(0, 0, 0, 0);

    // Zero the reservoir. RIS acceptance overwrites on demand; this ensures
    // "no candidate accepted" pixels end up as empty reservoirs.
    storeReservoir(g_Reservoirs_current, pixelIdx, (Reservoir)0);

    // Compact RIS state — 3 scalars instead of a full local Reservoir
    // (~17). The winning payload is written directly to the reservoir
    // buffer on acceptance; F_pack/F_mag/wsum are tracked here for the
    // final W computation.
    InitialRisState ris;
    ris.wsum   = 0.0f;
    ris.F_pack = 0u;
    ris.F_mag  = 0.0f;

    uint   seed         = initRandomData(pixel, uint2(8, 4), time, 1u);
    float3 rayOrigin    = InitOrigin();
    float3 rayDir       = InitDirection(pixel, float2(imgSize), seed);
    uint   throughputPk = PackRGB9E5(float3(1, 1, 1));
    uint   prevNormalPk = PackNormal(float3(0, 1, 0));
    float  prev_pdf       = 1.0f;
    float  pdf_product = 1.0f;

    // Depth-1 vertex state (populated once at depth=1, reused by every
    // GI candidate that fires at depth >= 2).
    float3 x2_world    = float3(0, 0, 0);
    float3 n2_world    = float3(0, 1, 0);
    float2 uv_at_x2    = float2(0, 0);
    uint   matID_at_x2 = 0u;
    uint   objID_at_x2 = 0u;
    float  eta_at_x2   = 1.0f;

    float3 v2_world    = float3(0, 1, 0);  // set at depth=2
    float3 tpost       = float3(1, 1, 1);  // post-x2 integrand accumulator

    //════════════════════════════════════════════════════════════════════════
    // Path loop
    //════════════════════════════════════════════════════════════════════════
    [loop]
    for (int depth = 0; depth < MAX_BOUNCES; ++depth)
    {
        if (any(isnan(rayDir)) || any(isinf(rayDir)) || dot(rayDir, rayDir) < 1e-12f ||
            any(isnan(rayOrigin)) || any(isinf(rayOrigin)))
            break;

        RayDesc ray;
        ray.Origin    = rayOrigin;
        ray.Direction = rayDir;
        ray.TMin      = 0.00001f;
        ray.TMax      = 10000.0f;
        dx::HitObject hitObj = TraceRay_Custom(SceneBVH, ray, RAY_FLAG_NONE, 0xFF);

        //─────────────────── Miss ───────────────────
        if (!hitObj.IsHit())
        {
            if (depth == 0)
            {
                float3 sun   = EvaluateSun(rayDir);
                float3 skyL1 = EvalMissState(rayDir, sun);
                if (length(sun) > 0.0f) skyL1 = sun;
                gScratchPing[uint3(pixel, 1)] = float4(skyL1, 0);
                gScratchPing[uint3(pixel, 2)] = float4(skyL1, 0);
                store_sky(g_sample_current, pixelIdx);
                break;
            }

            const float3 throughput = UnpackRGB9E5(throughputPk);
            const float3 envL       = EvalMissState(rayDir, float3(0, 0, 0));

            // pdf-free contribution f = throughput * pdf_product * envL
            const float3 F_contrib = throughput * envL * pdf_product;
            const float  p_hat     = GetPHat(F_contrib);
            const float  p_full    = pdf_product;
            const float  wi        = (p_full > 1e-20f) ? (p_hat / p_full) : 0.0f;

            if (depth == 1)
            {
                // DI env candidate — d=2 path, x2 stored as DIRECTION.
                AddInitialCandidate(ris, g_Reservoirs_current, pixelIdx, wi,
                    rayDir, float3(0, 1, 0),          // x2=dir, n2=unit default
                    envL,   float3(0, 1, 0),          // L2,     V2=unit default
                    float2(0, 0),
                    MATID_ENV_MISS, MATID_ENV_MISS, 1.0f,
                    F_contrib, seed);
            }
            else // depth >= 2: GI env (x2 = stashed depth-1 vertex)
            {
                AddInitialCandidate(ris, g_Reservoirs_current, pixelIdx, wi,
                    x2_world, n2_world,
                    envL * tpost, v2_world,
                    uv_at_x2,
                    matID_at_x2, objID_at_x2, eta_at_x2,
                    F_contrib, seed);
            }
            break;
        }

        //─────────────────── Hit setup ───────────────────
        const float  hitT   = hitObj.GetRayTCurrent();
        const float3 hitPos = rayOrigin + rayDir * hitT;

        const uint instID = hitObj.GetInstanceIndex();
        const uint primID = FlatPrimID(instID, hitObj.GetGeometryIndex(), hitObj.GetPrimitiveIndex());
        const uint matID  = GetMatIDFast(instID, primID);

        BuiltInTriangleIntersectionAttributes attr;
        hitObj.GetAttributes(attr);
        HitInfo hinfo = EvalSurfaceState(instID, primID, attr.barycentrics, rayOrigin, depth);

        // Backface-derived IOR pair: entering → (air, matNi), exiting → (matNi, air).
        // Null-IOR boundary (matNi ≈ 1) = pure pass-through; skip the hit.
        const float matNi = materials[matID].Ni;
        if (matNi <= 1.0f + EPSILON)
        {
            rayOrigin = hitPos;
            continue;
        }

        const float2 iors = hinfo.backface ? float2(matNi, 1.0f) : float2(1.0f, matNi);
        const uint   mediumMatID = hinfo.backface ? matID : MEDIUM_INVALID;

        float3 hitLocalKd; float hitLocalPr, hitLocalPm;
        RefetchMaterial(matID, hinfo.uv, hitLocalKd, hitLocalPr, hitLocalPm);

        const float3 absorptionTint = (mediumMatID != MEDIUM_INVALID)
            ? CalculateAbsorptionThroughput(materials[mediumMatID].Tf, hitT)
            : float3(1, 1, 1);

        const float3 emission = GetEmissionFast(instID, primID);

        //─────────────────── Depth 0: primary hit storage ───────────────────
        if (depth == 0)
        {
            const bool isEmitter = any(emission > 0.0f);
            store_instID(g_sample_current, pixelIdx, instID);
            store_primID(g_sample_current, pixelIdx, primID, isEmitter);
            store_bary  (g_sample_current, pixelIdx, attr.barycentrics);
            store_n1_s_world(g_sample_current, pixelIdx, hinfo.hitNormal, instID);
            store_uv    (g_sample_current, pixelIdx, hinfo.uv);
            if (isEmitter) {
                gScratchPing[uint3(pixel, 1)] = float4(emission, 0);
                gScratchPing[uint3(pixel, 2)] = float4(emission, 0);
            }

            // Specular motion vector reflection probe (unchanged)
            {
                float3 reflDir    = reflect(rayDir, hinfo.hitNormal);
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
                    float3 reflPos    = reflOrigin + reflDir * q.CommittedRayT();
                    float3 virtualPos = reflPos - 2.0f * dot(reflPos - hitPos, hinfo.hitNormal) * hinfo.hitNormal;
                    gScratchPing[uint3(pixel, 4)] = float4(virtualPos, asfloat(instID));
                }
                else
                {
                    gScratchPing[uint3(pixel, 4)] = float4(0, 0, 0, asfloat(0xFFFFFFFFu));
                }
            }
        }

        //─────────────────── Emitter hit ───────────────────
        if (any(emission > 0.0f) && hinfo.lightID != 0xFFFFFFFFu)
        {
            const float3 throughput = UnpackRGB9E5(throughputPk);
            const float3 prevNormal = UnpackNormal(prevNormalPk);
            const float  lightPdfArea = LT_Pdf_LightTree_Area(rayOrigin, prevNormal, hinfo.lightID, instID);
            const float  cosLight     = max(dot(hinfo.hitNormal, -rayDir), 0.0f);
            const float  dist2        = max(hitT * hitT, EPSILON);
            const float  lightPdfSA   = (cosLight > EPSILON) ? (lightPdfArea * dist2 / cosLight) : 0.0f;
            const float  misWeight    = prev_pdf / max(prev_pdf + lightPdfSA, EPSILON);

            const float3 F_contrib = throughput * emission * pdf_product;
            const float  p_hat     = GetPHat(F_contrib);
            const float  p_full    = pdf_product;
            const float  wi        = (p_full > 1e-20f) ? (misWeight * p_hat / p_full) : 0.0f;

            if (depth == 1)
            {
                // DI triangle-emitter candidate — d=2 path.
                AddInitialCandidate(ris, g_Reservoirs_current, pixelIdx, wi,
                    hitPos, hinfo.hitNormal,
                    emission, float3(0, 1, 0),        // V2 unused for DI
                    float2(0, 0),
                    MATID_LIGHT_TRI, instID, 1.0f,
                    F_contrib, seed);
            }
            else // depth >= 2: GI emitter (x2 = stashed depth-1 vertex)
            {
                AddInitialCandidate(ris, g_Reservoirs_current, pixelIdx, wi,
                    x2_world, n2_world,
                    emission * tpost, v2_world,
                    uv_at_x2,
                    matID_at_x2, objID_at_x2, eta_at_x2,
                    F_contrib, seed);
            }
            break;
        }

        //─────────────────── Stash depth-1 vertex ───────────────────
        if (depth == 1)
        {
            x2_world    = hitPos;
            n2_world    = hinfo.hitNormal;
            uv_at_x2    = hinfo.uv;
            matID_at_x2 = matID;
            objID_at_x2 = instID;
            eta_at_x2   = iors.y;
        }

        if (depth == 2)
        {
            v2_world = -rayDir;  // direction x3 -> x2
        }

        //─────────────────── NEE ───────────────────
        const bool performNEE = !(mediumMatID != MEDIUM_INVALID || materials[matID].Kd.w < EPSILON);

        uint matKdPk, matPrPmPk, hitNormalPk;
        if (performNEE)
        {
            matKdPk     = PackRGB9E5(hitLocalKd);
            matPrPmPk   = f32tof16_custom(hitLocalPr) | (f32tof16_custom(hitLocalPm) << 16u);
            hitNormalPk = PackNormal(hinfo.hitNormal);

            //─── Point light NEE ───
            {
                LT_LightSampleResult light = LT_SamplePointOnLight(hitPos, hinfo.hitNormal, seed);

                const float3 toLight = light.position - hitPos;
                const float  distSq  = dot(toLight, toLight);
                const float  dist    = sqrt(distSq);
                const float3 L       = toLight / dist;

                const float  cosSurf  = dot(hinfo.hitNormal, L);
                const float  cosLightS = dot(light.normal, -L);

                if (cosSurf > 1e-6f && cosLightS > 1e-6f &&
                    IsVisible(hitPos, hinfo.hitNormal, L, dist * 0.999f))
                {
                    const float3 lKd  = UnpackRGB9E5(matKdPk);
                    const float  lPr  = f16tof32_custom(matPrPmPk & 0xFFFFu);
                    const float  lPm  = f16tof32_custom(matPrPmPk >> 16u);
                    const float3 hitN = UnpackNormal(hitNormalPk);
                    const float3 throughput = UnpackRGB9E5(throughputPk);

                    SamplingP sp_nee = CalculateStrategyProbabilities(matID, -rayDir, hitN, iors.x, iors.y, lKd, lPm);
                    BrdfData  bdataNEE = EvaluateAndPdf_COMBINED(sp_nee, matID, hitN, hinfo.hitNormal, L, -rayDir, lKd, lPr, lPm, iors.x, iors.y);

                    const float lightPdf = light.pdfSolidAngle;
                    const float bsdfPdf  = bdataNEE.pdf;

                    if (lightPdf > 1e-20f && bsdfPdf > 0.0f)
                    {
                        const float  misWeight       = lightPdf / (lightPdf + bsdfPdf);
                        const float3 localMeasurement = light.emission * bdataNEE.val * cosSurf;
                        const float3 F_contrib        = throughput * localMeasurement * pdf_product;
                        const float  p_hat            = GetPHat(F_contrib);
                        const float  p_full           = pdf_product * lightPdf;
                        const float  wi               = (p_full > 1e-20f) ? (misWeight * p_hat / p_full) : 0.0f;

                        if (depth == 0)
                        {
                            // DI NEE at primary vertex — d=2 path, x2 = light.
                            AddInitialCandidate(ris, g_Reservoirs_current, pixelIdx, wi,
                                light.position, light.normal,
                                light.emission, float3(0, 1, 0),
                                float2(0, 0),
                                MATID_LIGHT_TRI, light.objID, 1.0f,
                                F_contrib, seed);
                        }
                        else if (depth == 1)
                        {
                            // GI NEE at depth-1 vertex — d=3 path, x2 = current hit.
                            AddInitialCandidate(ris, g_Reservoirs_current, pixelIdx, wi,
                                hitPos, hinfo.hitNormal,
                                light.emission, -L,
                                hinfo.uv,
                                matID, instID, iors.y,
                                F_contrib, seed);
                        }
                        else // depth >= 2: GI NEE past depth-1 vertex.
                        {
                            // tpost so far excludes current BSDF; NEE adds it.
                            const float3 tpostNEE = tpost * bdataNEE.val * cosSurf;
                            AddInitialCandidate(ris, g_Reservoirs_current, pixelIdx, wi,
                                x2_world, n2_world,
                                light.emission * tpostNEE, v2_world,
                                uv_at_x2,
                                matID_at_x2, objID_at_x2, eta_at_x2,
                                F_contrib, seed);
                        }
                    }
                }
            }

            //─── Sun NEE ───
            {
                float2 rSun = float2(RandomFloatSingle(seed), RandomFloatSingle(seed));
                SunSampleResult sun = SampleSun(rSun);
                const float3 hitN_sun = UnpackNormal(hitNormalPk);
                const float  NdotL    = dot(hitN_sun, sun.direction);

                if (NdotL > 1e-6f && IsVisible(hitPos, hinfo.hitNormal, sun.direction, 10000.0f))
                {
                    const float3 lKd = UnpackRGB9E5(matKdPk);
                    const float  lPr = f16tof32_custom(matPrPmPk & 0xFFFFu);
                    const float  lPm = f16tof32_custom(matPrPmPk >> 16u);
                    const float3 hitN = UnpackNormal(hitNormalPk);
                    const float3 throughput = UnpackRGB9E5(throughputPk);

                    SamplingP sp_nee = CalculateStrategyProbabilities(matID, -rayDir, hitN, iors.x, iors.y, lKd, lPm);
                    BrdfData  bdataNEE = EvaluateAndPdf_COMBINED(sp_nee, matID, hitN, hinfo.hitNormal, sun.direction, -rayDir, lKd, lPr, lPm, iors.x, iors.y);

                    const float lightPdf = sun.pdf;
                    const float bsdfPdf  = bdataNEE.pdf;

                    if (lightPdf > 1e-20f && bsdfPdf > 0.0f)
                    {
                        const float  misWeight       = lightPdf / (lightPdf + bsdfPdf);
                        const float3 localMeasurement = sun.radiance * bdataNEE.val * NdotL;
                        const float3 F_contrib        = throughput * localMeasurement * pdf_product;
                        const float  p_hat            = GetPHat(F_contrib);
                        const float  p_full           = pdf_product * lightPdf;

                        if (depth == 0)
                        {
                            // Sun stays excluded from the unified reservoir
                            // (user decision: no sun in DI bounce). Write direct.
                            float3 contrib = throughput * NdotL * sun.radiance * bdataNEE.val / lightPdf;
                            gScratchPing[uint3(pixel, 3)] = float4(misWeight * contrib, 0);
                        }
                        else if (depth == 1)
                        {
                            const float wi = (p_full > 1e-20f) ? (misWeight * p_hat / p_full) : 0.0f;
                            AddInitialCandidate(ris, g_Reservoirs_current, pixelIdx, wi,
                                hitPos, hinfo.hitNormal,
                                sun.radiance, -sun.direction,
                                hinfo.uv,
                                matID, instID, iors.y,
                                F_contrib, seed);
                        }
                        else // depth >= 2
                        {
                            const float  wi       = (p_full > 1e-20f) ? (misWeight * p_hat / p_full) : 0.0f;
                            const float3 tpostNEE = tpost * bdataNEE.val * NdotL;
                            AddInitialCandidate(ris, g_Reservoirs_current, pixelIdx, wi,
                                x2_world, n2_world,
                                sun.radiance * tpostNEE, v2_world,
                                uv_at_x2,
                                matID_at_x2, objID_at_x2, eta_at_x2,
                                F_contrib, seed);
                        }
                    }
                }
            }

            hitLocalKd = UnpackRGB9E5(matKdPk);
            hitLocalPr = f16tof32_custom(matPrPmPk & 0xFFFFu);
            hitLocalPm = f16tof32_custom(matPrPmPk >> 16u);
            hinfo.hitNormal = UnpackNormal(hitNormalPk);
        }

        //─────────────────── Sample next BSDF direction ───────────────────
        SamplingP sp = CalculateStrategyProbabilities(matID, -rayDir, hinfo.hitNormal, iors.x, iors.y, hitLocalKd, hitLocalPm);
        float3    s  = SampleBRDF(sp, matID, -rayDir, hinfo.hitNormal, hinfo.hitNormal, hitLocalKd, hitLocalPr, hitLocalPm, seed, iors.x, iors.y);
        BrdfData  bdata = EvaluateAndPdf_COMBINED(sp, matID, hinfo.hitNormal, hinfo.hitNormal, s, -rayDir, hitLocalKd, hitLocalPr, hitLocalPm, iors.x, iors.y);

        const float  cosTheta     = abs(dot(hinfo.hitNormal, s));
        const float3 updateWeight = (bdata.pdf > 1e-6f)
            ? (bdata.val * absorptionTint * cosTheta) / bdata.pdf
            : float3(0, 0, 0);

        if (dot(s, s) < 1e-12f || bdata.pdf <= 1e-6f || any(isnan(updateWeight)) || any(isinf(updateWeight)))
            break;

        prev_pdf       = bdata.pdf;
        pdf_product = min(pdf_product * bdata.pdf, 1e30f);
        rayDir         = s;
        float3 offsetN = dot(s, hinfo.hitNormal) >= 0.0f ? hinfo.hitNormal : -hinfo.hitNormal;
        rayOrigin      = offset_ray(hitPos, offsetN);

        {
            float3 throughput  = UnpackRGB9E5(throughputPk) * updateWeight;
            float3 tpostWeight = bdata.val * absorptionTint * cosTheta;

            if (depth > 1)
            {
                const float survivalProb = min(1.0f, Luma(throughput));
                if (RandomFloatSingle(seed) >= survivalProb) break;
                const float rrBoost = 1.0f / max(survivalProb, 0.1f);
                throughput  *= rrBoost;
                tpostWeight *= rrBoost;
                // RR survival is part of the path pdf; include in product.
                pdf_product = min(pdf_product * survivalProb, 1e30f);
            }

            // Accumulate tpost locally (no reservoir roundtrip).
            if (depth >= 2)
                tpost *= tpostWeight;

            throughputPk = PackRGB9E5(throughput);
        }
        prevNormalPk = PackNormal(hinfo.hitNormal);
    }

    //════════════════════════════════════════════════════════════════════════
    // Final resolve — commit RIS state to the buffer, compute W.
    // Per-field stores only; x2/n2/matID/etc. were already written to the
    // buffer by AddInitialCandidate on the last RIS acceptance.
    //════════════════════════════════════════════════════════════════════════
    {
        float W = 0.0f;
        if (ris.F_mag > 1e-6f && ris.wsum > 0.0f)
        {
            W = ris.wsum / ris.F_mag;
            if (isnan(W) || isinf(W) || W < 0.0f) W = 0.0f;
        }

        store_F    (g_Reservoirs_current, pixelIdx, ris.F_pack);
        store_F_mag(g_Reservoirs_current, pixelIdx, ris.F_mag);
        store_wsum (g_Reservoirs_current, pixelIdx, ris.wsum);
        store_W    (g_Reservoirs_current, pixelIdx, W);
        store_M    (g_Reservoirs_current, pixelIdx, 1u);

        if (W == 0.0f)
            InvalidateReservoir_ShadingNormal(g_Reservoirs_current, pixelIdx);
    }
}
