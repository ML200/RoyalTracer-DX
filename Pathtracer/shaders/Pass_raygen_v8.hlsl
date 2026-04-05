#include "Includes_raygen_v8.hlsli"

#ifndef MAX_BOUNCES
#define MAX_BOUNCES 30
#endif

#ifndef MEDIUM_INVALID_15
#define MEDIUM_INVALID_15 0x7FFFu
#endif

[shader("raygeneration")]
void Pass_raygen_v8()
{
    const uint2 pixel    = DispatchRaysIndex().xy;
    const uint2 imgSize  = DispatchRaysDimensions().xy;
    const uint  pixelIdx = MapPixelID(imgSize, pixel);

    // ── Sun contribution (stored separately, not in DI reservoir) ──────
    float3 sunDirect = float3(0, 0, 0);

    // ── Reservoir init ─────────────────────────────────────────────────
    storeReservoirDI(g_Reservoirs_current_di, pixelIdx, (Reservoir_DI)0);
    store_wsum_di(g_Reservoirs_current_di, pixelIdx, 0.0f);
    store_W_di(g_Reservoirs_current_di, pixelIdx, 0.0f);
    store_phat_di(g_Reservoirs_current_di, pixelIdx, 0.0f);

    storeReservoirGI(g_Reservoirs_current_gi, pixelIdx, (Reservoir_GI)0);
    store_wsum_gi(g_Reservoirs_current_gi, pixelIdx, 0.0f);
    store_W_gi(g_Reservoirs_current_gi, pixelIdx, 0.0f);
    store_F_gi(g_Reservoirs_current_gi, pixelIdx, 0.0f);
    store_M_gi(g_Reservoirs_current_gi, pixelIdx, 0u);
    store_seed_gi(g_Reservoirs_current_gi, pixelIdx, initRandomData(pixel, uint2(8, 4), time, 7u));
    store_Tpost_gi(g_Reservoirs_current_gi, pixelIdx, 1.0f);

    // ── Path state ─────────────────────────────────────────────────────
    uint   seed       = initRandomData(pixel, uint2(8, 4), time, 1u);
    uint   seedBSDF   = initRandomData(pixel, uint2(8, 4), time, 7u);
    float3 rayOrigin  = InitOrigin();
    float3 rayDir     = InitDirection(pixel, float2(imgSize), seed);
    float3 throughput = float3(1.0f, 1.0f, 1.0f);
    float3 prevNormal    = float3(0.0f, 1.0f, 0.0f);
    float  prev_pdf   = 1.0f;
    float  partial_J  = 0.0f;   // PDF1 * G_x1_x2, computed at depth 1
    float  pdf2_bsdf  = 0.0f;   // BSDF pdf at x2, computed after depth 1 BSDF sampling

    VolumeIOR_Packed viorP;
    VolumeAux_Packed aiorP;
    {
        VolumeIOR v0 = InitVolumeIOR();
        VolumeAux a0 = InitVolumeAux();
        viorP.raw   = PackIORStackAndPtr(v0.ior_stack, v0.pointer);
        aiorP.mat16 = PackMatStack16(a0.matID_stack);
        aiorP.obj8  = PackPrioStack8(a0.objID_stack);
    }

    // ── Bounce loop ────────────────────────────────────────────────────
    int phantomBudget = 8; // safety cap: max phantom surfaces before they consume depth
    [loop]
    for (int depth = 0; depth < MAX_BOUNCES; ++depth)
    {
        // Validate ray state
        if (any(isnan(rayDir)) || any(isinf(rayDir)) || dot(rayDir, rayDir) < 1e-12f ||
            any(isnan(rayOrigin)) || any(isinf(rayOrigin)))
            break;

        RayDesc ray;
        ray.Origin    = rayOrigin;
        ray.Direction = rayDir;
        ray.TMin      = 0.00001f;
        ray.TMax      = 10000.0f;
        dx::HitObject hitObj = TraceRay_Custom(SceneBVH, ray, RAY_FLAG_NONE, 0xFF);

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
                    store_sky(g_sample_current, pixelIdx);
                }
                break;
            }

            float3 envL   = EvalMissState(rayDir, float3(0,0,0));
            float3 T_envL = throughput * envL;

            // DI reservoir: env map hit at depth 1
            if (depth == 1)
            {
                float p_hat = GetPHat(T_envL * prev_pdf);
                float wi    = p_hat / prev_pdf;
                if (UpdateReservoirDI_Infinite(g_Reservoirs_current_di, pixelIdx, wi, rayDir, envL, 0xFFFFFFFFu, seed))
                    store_phat_di(g_Reservoirs_current_di, pixelIdx, p_hat);
            }

            // GI reservoir: env map hit at depth >= 2
            if (depth >= 2)
            {
                float  p_hat  = GetPHat(T_envL);
                float3 V2_new = (depth > 2) ? load_Vpost_gi(g_Reservoirs_current_gi, pixelIdx) : -rayDir;
                float2 J_new  = float2(0.0f, partial_J * pdf2_bsdf);
                float3 tpost  = load_Tpost_gi(g_Reservoirs_current_gi, pixelIdx);

                if (UpdateReservoirGI_Fast(g_Reservoirs_current_gi, pixelIdx, p_hat, envL * tpost, J_new, V2_new, seed))
                    store_F_gi(g_Reservoirs_current_gi, pixelIdx, p_hat);
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

        // Phantom surface: advance through null interface
        if (iors.y == 0.0f)
        {
            rayOrigin = hitPos;
            UpdateIORStack_packed(viorP, aiorP, matID, instID);
            if (phantomBudget-- > 0) --depth; // don't consume depth for phantoms (capped to prevent GPU hang)
            continue;
        }

        uint mediumMatID = GetCurrentMediumMaterialID_packed(viorP, aiorP);
        if (mediumMatID == 0x0000FFFFu) mediumMatID = MEDIUM_INVALID_15;

        BuiltInTriangleIntersectionAttributes attr;
        hitObj.GetAttributes(attr);
        HitInfo hinfo = EvalSurfaceState(instID, primID, attr.barycentrics, rayOrigin, depth);

        // Refetch material from UV
        float3 hitLocalKd; float hitLocalPr, hitLocalPm;
        RefetchMaterial(matID, hinfo.uv, hitLocalKd, hitLocalPr, hitLocalPm);

        // Volume absorption
        float3 absorptionTint = (mediumMatID != MEDIUM_INVALID_15)
            ? CalculateAbsorptionThroughput(materials[mediumMatID].Tf, hitT)
            : float3(1, 1, 1);

        float3 emission = GetEmissionFast(instID, primID);

        // ── Depth 0: store primary hit ─────────────────────────────────
        if (depth == 0)
        {
            bool isEmitter = any(emission > 0.0f);
            store_instID(g_sample_current, pixelIdx, instID, isEmitter);
            store_primID(g_sample_current, pixelIdx, primID);
            store_bary(g_sample_current, pixelIdx, attr.barycentrics);
            store_etai_etat(g_sample_current, pixelIdx, iors.x, iors.y);
            store_n1_g_world(g_sample_current, pixelIdx, hinfo.hitGNormal, instID);
            store_n1_s_world(g_sample_current, pixelIdx, hinfo.hitNormal, instID);
            store_uv(g_sample_current, pixelIdx, hinfo.uv);
            if (isEmitter) {
                gScratchPing[uint3(pixel, 1)] = float4(emission, 0);
                gScratchPing[uint3(pixel, 2)] = float4(emission, 0);
            }
        }

        // ── Emitter hit: BSDF-sampled light with MIS ──────────────────
        if (any(emission > 0.0f) && hinfo.lightID != 0xFFFFFFFFu)
        {
            float lightPdfArea = LT_Pdf_LightTree_Area(rayOrigin, prevNormal, hinfo.lightID, instID);
            float cosLight     = max(dot(hinfo.hitNormal, -rayDir), 0.0f);
            float dist2        = max(hitT * hitT, EPSILON);
            float lightPdfSA   = (cosLight > EPSILON) ? (lightPdfArea * dist2 / cosLight) : 0.0f;
            float misWeight    = prev_pdf / max(prev_pdf + lightPdfSA, EPSILON);

            // DI: emitter at depth 1
            if (depth == 1)
            {
                float p_hat = GetPHat(throughput * emission * prev_pdf);
                float wi    = misWeight * p_hat / prev_pdf;
                if (UpdateReservoirDI_Fast(g_Reservoirs_current_di, pixelIdx, wi, hitPos, hinfo.hitNormal, emission, instID, seed))
                    store_phat_di(g_Reservoirs_current_di, pixelIdx, p_hat);
            }

            // GI: emitter at depth >= 2
            if (depth >= 2)
            {
                float3 V2_new = (depth == 2) ? (-rayDir) : load_Vpost_gi(g_Reservoirs_current_gi, pixelIdx);
                float3 tpost  = load_Tpost_gi(g_Reservoirs_current_gi, pixelIdx);
                float  p_hat  = GetPHat(throughput * emission);
                float  wi     = p_hat * misWeight;

                if (UpdateReservoirGI_Fast(g_Reservoirs_current_gi, pixelIdx, wi, emission * tpost, float2(0.0f, partial_J * pdf2_bsdf), V2_new, seed))
                    store_F_gi(g_Reservoirs_current_gi, pixelIdx, p_hat);
            }
            break;
        }

        // ── Depth 1: store GI reconnection vertex data ────────────────
        if (depth == 1)
        {
            SetReservoirGI_ConstHit(g_Reservoirs_current_gi, pixelIdx, hitPos, hinfo.hitNormal, hinfo.hitGNormal, matID, instID);
            SetReservoirGI_UVAndIOR(g_Reservoirs_current_gi, pixelIdx, hinfo.uv, iors.x, iors.y);
            float cos_x2 = abs(dot(hinfo.hitGNormal, -rayDir));
            float dist2  = max(hitT * hitT, EPSILON);
            partial_J    = prev_pdf * cos_x2 / dist2;
        }

        // ── Depth 2: store post-reconnection direction ────────────────
        if (depth == 2)
        {
            store_Vpost_gi(g_Reservoirs_current_gi, pixelIdx, -rayDir);
        }

        // ── NEE (point lights + sun) ──────────────────────────────────
        bool performNEE = !(mediumMatID != MEDIUM_INVALID_15 || materials[matID].Kd.w < EPSILON);

        if (performNEE)
        {
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
                    SamplingP sp_nee = CalculateStrategyProbabilities(matID, -rayDir, hinfo.hitNormal, iors.x, iors.y, hitLocalKd, hitLocalPm);
                    BrdfData bdataNEE = EvaluateAndPdf_COMBINED(sp_nee, matID, hinfo.hitNormal, hinfo.hitGNormal, L, -rayDir, hitLocalKd, hitLocalPr, hitLocalPm, iors.x, iors.y);

                    float lightPdf = light.pdfSolidAngle;
                    float bsdfPdf  = bdataNEE.pdf;

                    if (lightPdf > 0.0f && bsdfPdf > 0.0f)
                    {
                        float misWeight = lightPdf / (lightPdf + bsdfPdf);

                        // DI: NEE at depth 0
                        if (depth == 0)
                        {
                            float p_hat = GetPHat(throughput * light.emission * bdataNEE.val * cosSurf);
                            float wi    = (lightPdf > 1e-20f) ? (misWeight * p_hat / lightPdf) : 0.0f;
                            if (UpdateReservoirDI_Fast(g_Reservoirs_current_di, pixelIdx, wi, light.position, light.normal, light.emission, light.objID, seed))
                                store_phat_di(g_Reservoirs_current_di, pixelIdx, p_hat);
                        }

                        // GI: NEE at depth >= 1
                        if (depth >= 1)
                        {
                            float3 V2_new = (depth == 1) ? (-L) : load_Vpost_gi(g_Reservoirs_current_gi, pixelIdx);
                            float  Jy_nee = (depth == 1) ? (partial_J * lightPdf) : (partial_J * pdf2_bsdf);
                            float2 J_new  = (depth == 1) ? float2(lightPdf, Jy_nee) : float2(0.0f, Jy_nee);
                            float3 contrib = throughput * light.emission * bdataNEE.val * cosSurf / lightPdf;
                            float  p_hat   = GetPHat(contrib);
                            float  wi      = p_hat * misWeight;
                            float3 tpost   = load_Tpost_gi(g_Reservoirs_current_gi, pixelIdx);
                            if (depth > 1) tpost *= bdataNEE.val * cosSurf / lightPdf;

                            if (UpdateReservoirGI_Fast(g_Reservoirs_current_gi, pixelIdx, wi, light.emission * tpost, J_new, V2_new, seed))
                                store_F_gi(g_Reservoirs_current_gi, pixelIdx, p_hat);
                        }
                    }
                }
            }

            // Sun NEE
            {
                float2 rSun = float2(RandomFloatSingle(seed), RandomFloatSingle(seed));
                SunSampleResult sun = SampleSun(rSun);
                float NdotL = dot(hinfo.hitNormal, sun.direction);

                if (NdotL > 1e-6f && IsVisible(hitPos, hinfo.hitGNormal, sun.direction, 10000.0f))
                {
                    SamplingP sp_nee = CalculateStrategyProbabilities(matID, -rayDir, hinfo.hitNormal, iors.x, iors.y, hitLocalKd, hitLocalPm);
                    BrdfData bdataNEE = EvaluateAndPdf_COMBINED(sp_nee, matID, hinfo.hitNormal, hinfo.hitGNormal, sun.direction, -rayDir, hitLocalKd, hitLocalPr, hitLocalPm, iors.x, iors.y);

                    float lightPdf = sun.pdf;
                    float bsdfPdf  = bdataNEE.pdf;

                    if (lightPdf > 0.0f && bsdfPdf > 0.0f)
                    {
                        float3 contrib = throughput * NdotL * sun.radiance * bdataNEE.val / lightPdf;
                        float misWeight = lightPdf / (lightPdf + bsdfPdf);

                        // DI: sun at depth 0 — store directly, bypass ReSTIR
                        if (depth == 0)
                        {
                            sunDirect += misWeight * contrib;
                        }

                        // GI: sun at depth >= 1
                        if (depth >= 1)
                        {
                            float3 V2_new = (depth == 1) ? (-sun.direction) : load_Vpost_gi(g_Reservoirs_current_gi, pixelIdx);
                            float  Jy_sun = (depth == 1) ? (partial_J * lightPdf) : (partial_J * pdf2_bsdf);
                            float2 J_new  = (depth == 1) ? float2(lightPdf, Jy_sun) : float2(0.0f, Jy_sun);
                            float  p_hat  = GetPHat(contrib);
                            float  wi     = p_hat;
                            float3 tpost  = load_Tpost_gi(g_Reservoirs_current_gi, pixelIdx);
                            if (depth > 1) tpost *= bdataNEE.val * NdotL / lightPdf;

                            if (UpdateReservoirGI_Fast(g_Reservoirs_current_gi, pixelIdx, wi, sun.radiance * tpost, J_new, V2_new, seed))
                                store_F_gi(g_Reservoirs_current_gi, pixelIdx, p_hat);
                        }
                    }
                }
            }
        }

        // ── Sample next direction (BSDF) ──────────────────────────────
        SamplingP sp = CalculateStrategyProbabilities(matID, -rayDir, hinfo.hitNormal, iors.x, iors.y, hitLocalKd, hitLocalPm);
        float3 s = SampleBRDF(sp, matID, -rayDir, hinfo.hitNormal, hinfo.hitGNormal, hitLocalKd, hitLocalPr, hitLocalPm, seedBSDF, iors.x, iors.y, GetVolumePtrFast_packed(viorP));
        BrdfData bdata = EvaluateAndPdf_COMBINED(sp, matID, hinfo.hitNormal, hinfo.hitGNormal, s, -rayDir, hitLocalKd, hitLocalPr, hitLocalPm, iors.x, iors.y);

        // Track BSDF pdf at reconnection vertex for GI jacobian
        if (depth == 1)
            pdf2_bsdf = bdata.pdf;

        float cosTheta = abs(dot(hinfo.hitNormal, s));
        float3 updateWeight = (bdata.pdf > 1e-6f)
            ? (bdata.val * absorptionTint * cosTheta) / bdata.pdf
            : float3(0, 0, 0);

        // Update post-reconnection throughput for GI
        if (depth >= 1)
        {
            float3 tpost = load_Tpost_gi(g_Reservoirs_current_gi, pixelIdx);
            store_Tpost_gi(g_Reservoirs_current_gi, pixelIdx, tpost * updateWeight);
        }

        // IOR stack update on transmission
        if (dot(hinfo.hitGNormal, s) < 0.0f)
            UpdateIORStack_packed(viorP, aiorP, matID, instID);

        // Validate before continuing
        if (dot(s, s) < 1e-12f || bdata.pdf <= 1e-6f || any(isnan(updateWeight)) || any(isinf(updateWeight)))
            break;

        // Advance path state
        throughput *= updateWeight;
        prev_pdf    = bdata.pdf;
        rayDir      = s;
        prevNormal  = hinfo.hitNormal;
        // Offset origin along geometric normal in the direction the ray exits
        float3 offsetN = dot(s, hinfo.hitGNormal) >= 0.0f ? hinfo.hitGNormal : -hinfo.hitGNormal;
        rayOrigin   = offset_ray(hitPos, offsetN);

        // Russian Roulette (skip depth 0 to ensure at least one bounce)
        if (depth > 0)
        {
            float survivalProb = min(1.0f, Luma(throughput));
            if (RandomFloatSingle(seed) >= survivalProb) break;
            throughput /= max(survivalProb, 0.1f);
        }
    }

    // ── Final reservoir weight computation ─────────────────────────────
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
        float F_gi  = load_F_gi(g_Reservoirs_current_gi, pixelIdx);
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

    // Store sun direct contribution separately (not in ReSTIR DI)
    gScratchPing[uint3(pixel, 3)] = float4(sunDirect, 0);
}
