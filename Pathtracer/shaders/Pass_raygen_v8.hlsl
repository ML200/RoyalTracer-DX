//using namespace dx;

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
    // Evaluate ID once to save heavy ALU instructions throughout the loop
    uint pixelIdx_1 = MapPixelID(DispatchRaysDimensions().xy, DispatchRaysIndex().xy);

    // DI Reservoir init
    storeReservoirDI(g_Reservoirs_current_di, pixelIdx_1, (Reservoir_DI)0);
    store_wsum_di(g_Reservoirs_current_di, pixelIdx_1, 0.0f);
    store_W_di(g_Reservoirs_current_di, pixelIdx_1, 0.0f);
    store_phat_di(g_Reservoirs_current_di, pixelIdx_1, 0.0f);

    // GI Reservoir init
    storeReservoirGI(g_Reservoirs_current_gi, pixelIdx_1, (Reservoir_GI)0);
    store_wsum_gi(g_Reservoirs_current_gi, pixelIdx_1, 0.0f);
    store_W_gi(g_Reservoirs_current_gi, pixelIdx_1, 0.0f);
    store_F_gi(g_Reservoirs_current_gi, pixelIdx_1, 0.0f);
    store_M_gi(g_Reservoirs_current_gi, pixelIdx_1, 0u);
    store_Tpost_gi(g_Reservoirs_current_gi, pixelIdx_1, 1.0f);

    uint seed = initRandomData(DispatchRaysIndex().xy, uint2(8, 4), time, 1u);

    // Initialize raw path tracing state variables instead of packing
    float3 rayOrigin  = InitOrigin();
    float3 rayDir     = InitDirection(DispatchRaysIndex().xy, float2(DispatchRaysDimensions().xy), seed);
    float3 throughput = float3(1.0f, 1.0f, 1.0f);
    float3 prevNormal = float3(0.0f, 1.0f, 0.0f);
    float  prev_pdf   = 1.0f;

    // Initialize packed IOR stack (retained as some nested engine functions may rely on this)
    VolumeIOR_Packed viorP;
    VolumeAux_Packed aiorP;
    {
        VolumeIOR v0 = InitVolumeIOR();
        VolumeAux a0 = InitVolumeAux();

        viorP.raw   = PackIORStackAndPtr(v0.ior_stack, v0.pointer);
        aiorP.mat16 = PackMatStack16(a0.matID_stack);
        aiorP.obj8  = PackPrioStack8(a0.objID_stack);
    }

    [loop]
    for (int depth = 0; depth < MAX_BOUNCES; ++depth)
    {
        // Safety validation
        float d2 = dot(rayDir, rayDir);
        bool badDir = any(isnan(rayDir)) || any(isinf(rayDir)) || (d2 < 1e-12f);
        bool badOrg = any(isnan(rayOrigin)) || any(isinf(rayOrigin));
        if (badDir || badOrg) break;

        RayDesc ray;
        ray.Origin    = rayOrigin;
        ray.Direction = rayDir;
        ray.TMin      = 0.00001f;
        ray.TMax      = 10000.0f;

        dx::HitObject hitObj = TraceRay_Custom(SceneBVH, ray, RAY_FLAG_NONE, 0xFF);

        if (!hitObj.IsHit())
        {
            uint pixelIdx_2 = MapPixelID(DispatchRaysDimensions().xy, DispatchRaysIndex().xy);
            // Store sample data miss case
            if (depth == 0)
            {
                SampleData sdata = (SampleData)0;
                float3 sun = EvaluateSun(rayDir);
                sdata.L1 = EvalMissState(rayDir, sun);
                if (length(sun) > 0.0f)
                    sdata.L1 = sun;

                storeSampleData(g_sample_current, pixelIdx_2, sdata);
                break;
            }

            float3 envL = EvalMissState(rayDir, float3(0,0,0));
            float3 T_envL = throughput * envL;

            if (depth == 1)
            {
                float p_hat = GetPHat(T_envL * prev_pdf); // Cancel previous pdf by multiplying with it
                float wi = p_hat / prev_pdf;
                bool update = UpdateReservoirDI_Infinite(g_Reservoirs_current_di, pixelIdx_2, wi, rayDir, envL, 0xFFFFFFFFu, seed);
                if (update) store_phat_di(g_Reservoirs_current_di, pixelIdx_2, p_hat);
            }

            if (depth >= 2)
            {
                float p_hat = GetPHat(T_envL);
                float wi    = p_hat;

                float3 V2_new = (depth > 2) ? load_Vpost_gi(g_Reservoirs_current_gi, pixelIdx_2) : -rayDir;
                float4 J_new  = float4(0.0f, (depth > 2) ? 0.0f : 1.0f, 0.0f, 0.0f);

                float3 tpostgi = load_Tpost_gi(g_Reservoirs_current_gi, pixelIdx_2);
                bool update = UpdateReservoirGI_Fast(g_Reservoirs_current_gi, pixelIdx_2, wi, envL * tpostgi, J_new, V2_new, seed);

                if (update) store_F_gi(g_Reservoirs_current_gi, pixelIdx_2, p_hat);
            }
            break;
        }

        // --- Pre-invoke setup ---
        float hitT = hitObj.GetRayTCurrent();
        float3 hitPos = rayOrigin + rayDir * hitT;

        const uint instID = hitObj.GetInstanceIndex();
        const uint primID = FlatPrimID(instID, hitObj.GetGeometryIndex(), hitObj.GetPrimitiveIndex());
        uint matID = GetMatIDFast(instID, primID);

        float2 iorsF = GetIORs_packed(viorP, aiorP, matID, instID);

        // Phantom surface: advance and continue
        if (iorsF.y == 0.0f)
        {
            rayOrigin = hitPos;
            UpdateIORStack_packed(viorP, aiorP, matID, instID);
            continue;
        }

        uint mediumMatID = GetCurrentMediumMaterialID_packed(viorP, aiorP);
        if (mediumMatID == 0x0000FFFFu) mediumMatID = MEDIUM_INVALID_15;

        // Fetch barycentrics and evaluate surface
        BuiltInTriangleIntersectionAttributes attr;
        hitObj.GetAttributes(attr);
        HitInfo hinfo = EvalSurfaceState(instID, primID, attr.barycentrics, rayOrigin, depth);

        // --- Handle Absorption (Volume) ---
        float3 absorptionTint = float3(1, 1, 1);
        if (mediumMatID != MEDIUM_INVALID_15)
        {
            absorptionTint = CalculateAbsorptionThroughput(materials[mediumMatID].Tf, hitT);
        }

        // --- Emission (Direct Light) & MIS ---
        float3 emission = GetEmissionFast(instID, primID);

        // First bounce data storage
        if (depth == 0)
        {
            SampleData sdata = (SampleData)0;
            sdata.x1      = hitPos;
            sdata.n1_s    = hinfo.hitNormal;
            sdata.n1_g    = hinfo.hitGNormal;
            sdata.L1      = emission;
            sdata.o       = -rayDir;
            sdata.objID   = instID;
            sdata.matID   = matID;
            sdata.localKd = hinfo.localKd;
            sdata.localPr = hinfo.localPr;
            sdata.localPm = hinfo.localPm;
            sdata.etai    = iorsF.x;
            sdata.etat    = iorsF.y;
            uint pixelIdx_3 = MapPixelID(DispatchRaysDimensions().xy, DispatchRaysIndex().xy);
            storeSampleData(g_sample_current, pixelIdx_3, sdata);
        }

        // Valid light hit evaluation via MIS
        if (any(emission > 0.0f) && hinfo.lightID != 0xFFFFFFFFu)
        {
            float lightPdfArea = LT_Pdf_LightTree_Area(rayOrigin, prevNormal, hinfo.lightID, instID);

            float cosLight = max(dot(hinfo.hitNormal, -rayDir), 0.0f);
            float dist2 = max(hitT * hitT, EPSILON);
            float lightPdfSA = (cosLight > EPSILON) ? (lightPdfArea * dist2 / cosLight) : 0.0f;

            float jacobian = (dist2 > EPSILON) ? (cosLight / dist2) : 0.0f;
            float misWeight = prev_pdf / max(prev_pdf + lightPdfSA, EPSILON);

            uint pixelIdx_4 = MapPixelID(DispatchRaysDimensions().xy, DispatchRaysIndex().xy);

            if (depth == 1)
            {
                float3 targetRadiance = throughput * emission;
                float p_hat = GetPHat(targetRadiance * prev_pdf);
                float wi = misWeight * p_hat / prev_pdf;
                bool update = UpdateReservoirDI_Fast(g_Reservoirs_current_di, pixelIdx_4, wi, hitPos, hinfo.hitNormal, emission, instID, seed);
                if (update) store_phat_di(g_Reservoirs_current_di, pixelIdx_4, p_hat);
            }

            if (depth >= 2)
            {
                float3 V2_new = (depth == 2) ? (-rayDir) : load_Vpost_gi(g_Reservoirs_current_gi, pixelIdx_4);
                float4 J_new  = float4(0.0f, 0.0f, 0.0f, 0.0f);

                float3 contrib = throughput * emission;
                float  p_hat   = GetPHat(contrib);
                float  wi      = p_hat * misWeight;
                float3 tpostgi = load_Tpost_gi(g_Reservoirs_current_gi, pixelIdx_4);

                bool update = UpdateReservoirGI_Fast(g_Reservoirs_current_gi, pixelIdx_4, wi, emission * tpostgi, J_new, V2_new, seed);
                if (update) store_F_gi(g_Reservoirs_current_gi, pixelIdx_4, p_hat);
            }
            break; // Ends hit logic as path effectively terminates at emitter
        }

        // Store intermediate GI state
        uint pixelIdx_5 = MapPixelID(DispatchRaysDimensions().xy, DispatchRaysIndex().xy);
        if (depth == 1)
        {
            SetReservoirGI_ConstHit(g_Reservoirs_current_gi, pixelIdx_5, hitPos, hinfo.hitNormal, hinfo.hitGNormal, matID, instID);
            SetReservoirGI_LocalMaterial(g_Reservoirs_current_gi, pixelIdx_5, hinfo.localKd, hinfo.localPr, hinfo.localPm, iorsF.x, iorsF.y);
        }
        if (depth == 2)
        {
            store_Vpost_gi(g_Reservoirs_current_gi, pixelIdx_5, -rayDir);
        }

        // --- Next Event Estimation (NEE) ---
        bool performNEE = !(mediumMatID != MEDIUM_INVALID_15 || materials[matID].Kd.w < EPSILON);

        if (performNEE)
        {
            // --- Point Light NEE ---
            {
                LT_LightSampleResult light = LT_SamplePointOnLight(hitPos, hinfo.hitNormal, seed);

                float3 toLight = light.position - hitPos;
                float  distSq  = dot(toLight, toLight);
                float  dist    = sqrt(distSq);
                float3 L       = toLight / dist;

                float cosSurf  = dot(hinfo.hitNormal, L);
                float cosLight = dot(light.normal, -L);

                if (cosSurf > 1e-6f && cosLight > 1e-6f)
                {
                    RayDesc shadowRay;
                    shadowRay.Origin    = hitPos;
                    shadowRay.Direction = L;
                    shadowRay.TMin      = 0.00001f;
                    shadowRay.TMax      = dist - 0.001f;

                    RayQuery<RAY_FLAG_CULL_NON_OPAQUE | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> q;
                    q.TraceRayInline(SceneBVH, RAY_FLAG_NONE, 0xFF, shadowRay);
                    q.Proceed();

                    if (q.CommittedStatus() == COMMITTED_NOTHING)
                    {
                        SamplingP sp_nee = CalculateStrategyProbabilities(matID, -rayDir, hinfo.hitNormal, iorsF.x, iorsF.y, hinfo.localKd, hinfo.localPm);
                        BrdfData bdataNEE = EvaluateAndPdf_COMBINED(sp_nee, matID, hinfo.hitNormal, hinfo.hitGNormal, L, -rayDir, hinfo.localKd, hinfo.localPr, hinfo.localPm, iorsF.x, iorsF.y);

                        float lightPdf = light.pdfSolidAngle;
                        float bsdfPdf  = bdataNEE.pdf;

                        if (lightPdf > 0.0f && bsdfPdf > 0.0f)
                        {
                            uint pixelIdx_6 = MapPixelID(DispatchRaysDimensions().xy, DispatchRaysIndex().xy);
                            float misWeight = lightPdf / (lightPdf + bsdfPdf);

                            if (depth == 0)
                            {
                                float3 targetRadiance = throughput * light.emission * bdataNEE.val * cosSurf;
                                float p_hat = GetPHat(targetRadiance);
                                float wi = (lightPdf > 1e-20f) ? (misWeight * p_hat / lightPdf) : 0.0f;
                                bool update = UpdateReservoirDI_Fast(g_Reservoirs_current_di, pixelIdx_6, wi, light.position, light.normal, light.emission, light.objID, seed);
                                if (update) store_phat_di(g_Reservoirs_current_di, pixelIdx_6, p_hat);
                            }
                            if (depth >= 1)
                            {
                                float3 V2_new = (depth == 1) ? (-L) : load_Vpost_gi(g_Reservoirs_current_gi, pixelIdx_6);
                                float4 J_new  = (depth == 1) ? float4(lightPdf, 0.0f, 0.0f, 0.0f) : float4(0.0f, 0.0f, 0.0f, 0.0f);

                                float3 contrib = throughput * light.emission * bdataNEE.val * cosSurf / lightPdf;

                                float p_hat = GetPHat(contrib);
                                float wi    = p_hat * misWeight;
                                float3 tpostgi = load_Tpost_gi(g_Reservoirs_current_gi, pixelIdx_6);
                                if (depth > 1) tpostgi *= bdataNEE.val * cosSurf / lightPdf;

                                bool update = UpdateReservoirGI_Fast(g_Reservoirs_current_gi, pixelIdx_6, wi, light.emission * tpostgi, J_new, V2_new, seed);
                                if (update) store_F_gi(g_Reservoirs_current_gi, pixelIdx_6, p_hat);
                            }
                        }
                    }
                }
            }

            // --- Sun NEE ---
            {
                float2 rSun = float2(RandomFloatSingle(seed), RandomFloatSingle(seed));
                SunSampleResult sun = SampleSun(rSun);

                float NdotL = dot(hinfo.hitNormal, sun.direction);

                if (NdotL > 1e-6f)
                {
                    RayDesc shadowRay;
                    shadowRay.Origin    = hitPos;
                    shadowRay.Direction = sun.direction;
                    shadowRay.TMin      = 0.00001f;
                    shadowRay.TMax      = 10000.0f; // Effectively infinite

                    RayQuery<RAY_FLAG_CULL_NON_OPAQUE | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> q;
                    q.TraceRayInline(SceneBVH, RAY_FLAG_NONE, 0xFF, shadowRay);
                    q.Proceed();

                    if (q.CommittedStatus() == COMMITTED_NOTHING)
                    {
                        SamplingP sp_nee = CalculateStrategyProbabilities(matID, -rayDir, hinfo.hitNormal, iorsF.x, iorsF.y, hinfo.localKd, hinfo.localPm);
                        BrdfData bdataNEE = EvaluateAndPdf_COMBINED(sp_nee, matID, hinfo.hitNormal, hinfo.hitGNormal, sun.direction, -rayDir, hinfo.localKd, hinfo.localPr, hinfo.localPm, iorsF.x, iorsF.y);

                        float lightPdf = sun.pdf;
                        float bsdfPdf  = bdataNEE.pdf;

                        if (lightPdf > 0.0f && bsdfPdf > 0.0f)
                        {
                            float3 contrib = throughput * NdotL * sun.radiance * bdataNEE.val / lightPdf;
                            uint pixelIdx_7 = MapPixelID(DispatchRaysDimensions().xy, DispatchRaysIndex().xy);

                            if (depth == 0)
                            {
                                float3 target = throughput * bdataNEE.val * NdotL * sun.radiance;
                                float  p_hat  = GetPHat(target);
                                float  wi = (lightPdf > 1e-20f) ? (p_hat / lightPdf) : 0.0f;
                                bool update = UpdateReservoirDI_Infinite(g_Reservoirs_current_di, pixelIdx_7, wi, normalize(sun.direction), sun.radiance, 0xFFFFFFFEu, seed);
                                if (update) store_phat_di(g_Reservoirs_current_di, pixelIdx_7, p_hat);
                            }
                            if (depth >= 1)
                            {
                                float3 V2_new = (depth == 1) ? (-sun.direction) : load_Vpost_gi(g_Reservoirs_current_gi, pixelIdx_7);
                                float4 J_new  = (depth == 1) ? float4(lightPdf, 0.0f, 0.0f, 0.0f) : float4(0.0f, 0.0f, 0.0f, 0.0f);

                                float  p_hat   = GetPHat(contrib);
                                float  wi      = p_hat;
                                float3 tpostgi = load_Tpost_gi(g_Reservoirs_current_gi, pixelIdx_7);
                                if (depth > 1) tpostgi *= bdataNEE.val * NdotL / lightPdf;

                                bool update = UpdateReservoirGI_Fast(g_Reservoirs_current_gi, pixelIdx_7, wi, sun.radiance * tpostgi, J_new, V2_new, seed);
                                if (update) store_F_gi(g_Reservoirs_current_gi, pixelIdx_7, p_hat);
                            }
                        }
                    }
                }
            }
        }

        // --- Sample Next Direction (BSDF) ---
        SamplingP sp = CalculateStrategyProbabilities(matID, -rayDir, hinfo.hitNormal, iorsF.x, iorsF.y, hinfo.localKd, hinfo.localPm);
        float3 s = SampleBRDF(sp, matID, -rayDir, hinfo.hitNormal, hinfo.hitGNormal, hinfo.localKd, hinfo.localPr, hinfo.localPm, seed, iorsF.x, iorsF.y, GetVolumePtrFast_packed(viorP));
        BrdfData bdata = EvaluateAndPdf_COMBINED(sp, matID, hinfo.hitNormal, hinfo.hitGNormal, s, -rayDir, hinfo.localKd, hinfo.localPr, hinfo.localPm, iorsF.x, iorsF.y);

        float cosTheta = abs(dot(hinfo.hitNormal, s));
        float3 updateWeight = float3(0, 0, 0);
        if (bdata.pdf > 1e-6f)
        {
            updateWeight = (bdata.val * absorptionTint * cosTheta) / bdata.pdf;
        }

        if (depth >= 1)
        {
            uint pixelIdx_8 = MapPixelID(DispatchRaysDimensions().xy, DispatchRaysIndex().xy);
            float3 tpostgi = load_Tpost_gi(g_Reservoirs_current_gi, pixelIdx_8);
            tpostgi *= updateWeight;
            store_Tpost_gi(g_Reservoirs_current_gi, pixelIdx_8, tpostgi);
        }

        // IOR Transmission validation
        if (dot(hinfo.hitGNormal, s) < 0.0f)
        {
            UpdateIORStack_packed(viorP, aiorP, matID, instID);
        }

        // Validate state before progressing path
        if (dot(s, s) < 1e-12f || bdata.pdf <= 1e-6f || any(isnan(updateWeight)) || any(isinf(updateWeight)))
            break;

        // Apply throughput updates natively
        throughput *= updateWeight;
        prev_pdf    = bdata.pdf;
        rayDir      = s;
        prevNormal  = hinfo.hitNormal;
        rayOrigin   = hitPos;

        // --- Russian Roulette ---
        if (depth > 0)
        {
            float survivalProb = min(1.0f, Luma(throughput));
            if (RandomFloatSingle(seed) >= survivalProb) break;
            throughput /= max(survivalProb, 0.1f);
        }
    }

    // --- Final Reservoir Weights Processing ---
    uint pixelIdx_9 = MapPixelID(DispatchRaysDimensions().xy, DispatchRaysIndex().xy);

    // DI Evaluate
    float final_p_hat_di = load_phat_di(g_Reservoirs_current_di, pixelIdx_9);
    float w_sum_di       = load_wsum_di(g_Reservoirs_current_di, pixelIdx_9);
    float W_di           = 0.0f;
    if (final_p_hat_di > 1e-6f && w_sum_di > 0.0f)
    {
        W_di = w_sum_di / final_p_hat_di;
    }

    store_W_di(g_Reservoirs_current_di, pixelIdx_9, W_di);
    store_M_di(g_Reservoirs_current_di, pixelIdx_9, 1);

    // GI Evaluate
    float F_gi    = load_F_gi(g_Reservoirs_current_gi, pixelIdx_9);
    float wsum_gi = load_wsum_gi(g_Reservoirs_current_gi, pixelIdx_9);
    float Wgi     = 0.0f;

    if (F_gi > 1e-6f && wsum_gi > 0.0f)
    {
        Wgi = wsum_gi / F_gi;
        if (isnan(Wgi) || isinf(Wgi)) Wgi = 0.0f;
    }

    if (Wgi == 0) InvalidateReservoirGI_ShadingNormal(g_Reservoirs_current_gi, pixelIdx_9);

    store_W_gi(g_Reservoirs_current_gi, pixelIdx_9, Wgi);
    store_M_gi(g_Reservoirs_current_gi, pixelIdx_9, 1u);
}