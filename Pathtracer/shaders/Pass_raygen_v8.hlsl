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
    uint2  pixel   = DispatchRaysIndex().xy;
    uint2  imgSize = DispatchRaysDimensions().xy;
    uint   pixelIdx = MapPixelID(imgSize, pixel);

    // Reorder threads before bounce loop: group dead rays
    uint camInstID = load_instID(g_sample_current, pixelIdx);
    bool rayDead   = (camInstID == 0xFFFFFFFFu) || load_isEmitter(g_sample_current, pixelIdx);
    dx::MaybeReorderThread(dx::HitObject::MakeNop(), rayDead ? 0u : 1u, 1);

    // ── Early-out: sky (camera shader stored sentinel + scratch) ───────
    if (camInstID == 0xFFFFFFFFu)
    {
        store_W_di(g_Reservoirs_current_di, pixelIdx, 0.0f);
        store_M_di(g_Reservoirs_current_di, pixelIdx, 1);
        InvalidateReservoirGI_ShadingNormal(g_Reservoirs_current_gi, pixelIdx);
        store_W_gi(g_Reservoirs_current_gi, pixelIdx, 0.0f);
        store_M_gi(g_Reservoirs_current_gi, pixelIdx, 1u);
        return;
    }

    uint   seed       = initRandomData(DispatchRaysIndex().xy, uint2(8, 4), time, 1u);
    float3 rayOrigin  = InitOrigin();
    float3 rayDir     = InitDirection(DispatchRaysIndex().xy, float2(DispatchRaysDimensions().xy), seed);
    uint   throughputPk = PackRGB9E5(float3(1, 1, 1));
    uint   prevNormalPk = PackNormal(float3(0, 1, 0));
    float  prev_pdf       = 1.0f;
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

    [loop]
    for (int depth = 0; depth < MAX_BOUNCES; ++depth)
    {
        float  hitT;
        float3 hitPos;
        uint   instID, primID;
        HitInfo hinfo;

        if (depth == 0)
        {
            // ── Read primary hit from G-buffer (traced by camera shader) ──
            instID = load_instID(g_sample_current, pixelIdx);
            primID = load_primID(g_sample_current, pixelIdx);
            float2 bary = load_bary(g_sample_current, pixelIdx);
            hinfo  = EvalSurfaceState(instID, primID, bary, rayOrigin, 0);
            hitPos = hinfo.hitPos;
            hitT   = length(hitPos - rayOrigin);
        }
        else
        {
            // ── Trace bounce rays (depth >= 1) ────────────────────────────
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
                float3 throughput = UnpackRGB9E5(throughputPk);
                float3 envL = EvalMissState(rayDir, float3(0, 0, 0));

                float3 F_contrib = throughput * envL * gi_pdf_product;
                float  p_hat     = GetPHat(F_contrib);
                float  p_full    = gi_pdf_product;
                float  wi        = (p_full > 1e-20f) ? (p_hat / p_full) : 0.0f;

                uint px = MapPixelID(imgSize, pixel);

                if (depth == 1)
                {
                    if (UpdateReservoirDI_Infinite(g_Reservoirs_current_di, px, wi, rayDir, envL, 0xFFFFFFFFu, seed))
                        store_phat_di(g_Reservoirs_current_di, px, p_hat);
                }
                else if (depth >= 2)
                {
                    float3 V2_new = (depth > 2) ? load_Vpost_gi(g_Reservoirs_current_gi, px) : -rayDir;
                    float3 tpost  = load_Tpost_gi(g_Reservoirs_current_gi, px);

                    if (UpdateReservoirGI_Fast(g_Reservoirs_current_gi, px, wi, envL * tpost, V2_new, seed))
                        store_F_combined_gi(g_Reservoirs_current_gi, px, F_contrib);
                }
                break;
            }

            hitT   = hitObj.GetRayTCurrent();
            hitPos = rayOrigin + rayDir * hitT;

            instID = hitObj.GetInstanceIndex();
            primID = FlatPrimID(instID, hitObj.GetGeometryIndex(), hitObj.GetPrimitiveIndex());

            BuiltInTriangleIntersectionAttributes attr;
            hitObj.GetAttributes(attr);
            hinfo = EvalSurfaceState(instID, primID, attr.barycentrics, rayOrigin, depth);
        }

        // ── Common hit processing ──────────────────────────────────────
        uint   matID = GetMatIDFast(instID, primID);
        float2 iors  = GetIORs_packed(viorP, aiorP, matID, instID);

        if (iors.y == 0.0f)
        {
            rayOrigin = hitPos;
            UpdateIORStack_packed(viorP, aiorP, matID, instID);
            continue;
        }

        uint mediumMatID = GetCurrentMediumMaterialID_packed(viorP, aiorP);

        float3 hitLocalKd; float hitLocalPr, hitLocalPm;
        RefetchMaterial(matID, hinfo.uv, hitLocalKd, hitLocalPr, hitLocalPm);

        float3 absorptionTint = (mediumMatID != MEDIUM_INVALID)
            ? CalculateAbsorptionThroughput(materials[mediumMatID].Tf, hitT)
            : float3(1, 1, 1);

        float3 emission = GetEmissionFast(instID, primID);

        // ── Emitter hit ────────────────────────────────────────────────
        if (any(emission > 0.0f) && hinfo.lightID != 0xFFFFFFFFu)
        {
            float3 throughput = UnpackRGB9E5(throughputPk);
            float3 prevNormal = UnpackNormal(prevNormalPk);
            float lightPdfArea = LT_Pdf_LightTree_Area(rayOrigin, prevNormal, hinfo.lightID, instID);
            float cosLight     = max(dot(hinfo.hitNormal, -rayDir), 0.0f);
            float dist2        = max(hitT * hitT, EPSILON);
            float lightPdfSA   = (cosLight > EPSILON) ? (lightPdfArea * dist2 / cosLight) : 0.0f;
            float misWeight    = prev_pdf / max(prev_pdf + lightPdfSA, EPSILON);

            float3 F_contrib = throughput * emission * gi_pdf_product;
            float  p_hat     = GetPHat(F_contrib);
            float  p_full    = gi_pdf_product;
            float  wi        = (p_full > 1e-20f) ? (misWeight * p_hat / p_full) : 0.0f;

            uint px = MapPixelID(imgSize, pixel);

            if (depth == 1)
            {
                if (UpdateReservoirDI_Fast(g_Reservoirs_current_di, px, wi, hitPos, hinfo.hitNormal, emission, instID, seed))
                    store_phat_di(g_Reservoirs_current_di, px, p_hat);
            }
            else if (depth >= 2)
            {
                float3 V2_new = (depth == 2) ? (-rayDir) : load_Vpost_gi(g_Reservoirs_current_gi, px);
                float3 tpost  = load_Tpost_gi(g_Reservoirs_current_gi, px);

                if (UpdateReservoirGI_Fast(g_Reservoirs_current_gi, px, wi, emission * tpost, V2_new, seed))
                    store_F_combined_gi(g_Reservoirs_current_gi, px, F_contrib);
            }
            break;
        }

        //Depth 1: store reconnection vertex
        if (depth == 1)
        {
            uint px = MapPixelID(imgSize, pixel);
            SetReservoirGI_ConstHit(g_Reservoirs_current_gi, px, hitPos, hinfo.hitNormal, matID, instID, iors.y);
            SetReservoirGI_UV(g_Reservoirs_current_gi, px, hinfo.uv);
        }

        if (depth == 2)
        {
            store_Vpost_gi(g_Reservoirs_current_gi, MapPixelID(imgSize, pixel), -rayDir);
        }

        //NEE
        bool performNEE = !(mediumMatID != MEDIUM_INVALID || materials[matID].Kd.w < EPSILON);

        uint matKdPk, matPrPmPk, hitNormalPk;
        if (performNEE)
        {
            matKdPk     = PackRGB9E5(hitLocalKd);
            matPrPmPk   = f32tof16_custom(hitLocalPr) | (f32tof16_custom(hitLocalPm) << 16u);
            hitNormalPk = PackNormal(hinfo.hitNormal);

            //Point light NEE
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
                    float3 lKd = UnpackRGB9E5(matKdPk);
                    float  lPr = f16tof32_custom(matPrPmPk & 0xFFFFu);
                    float  lPm = f16tof32_custom(matPrPmPk >> 16u);
                    float3 hitN = UnpackNormal(hitNormalPk);
                    float3 throughput = UnpackRGB9E5(throughputPk);

                    SamplingP sp_nee = CalculateStrategyProbabilities(matID, -rayDir, hitN, iors.x, iors.y, lKd, lPm);
                    BrdfData bdataNEE = EvaluateAndPdf_COMBINED(sp_nee, matID, hitN, hinfo.hitNormal, L, -rayDir, lKd, lPr, lPm, iors.x, iors.y);

                    float lightPdf = light.pdfSolidAngle;
                    float bsdfPdf  = bdataNEE.pdf;

                    if (lightPdf > 1e-20f && bsdfPdf > 0.0f)
                    {
                        float misWeight = lightPdf / (lightPdf + bsdfPdf);

                        float3 localMeasurement = light.emission * bdataNEE.val * cosSurf;
                        float3 F_contrib = throughput * localMeasurement * gi_pdf_product;
                        float  p_hat     = GetPHat(F_contrib);
                        float  p_full    = gi_pdf_product * lightPdf;
                        float  wi        = (p_full > 1e-20f) ? (misWeight * p_hat / p_full) : 0.0f;

                        uint px = MapPixelID(imgSize, pixel);

                        if (depth == 0)
                        {
                            if (UpdateReservoirDI_Fast(g_Reservoirs_current_di, px, wi, light.position, light.normal, light.emission, light.objID, seed))
                                store_phat_di(g_Reservoirs_current_di, px, p_hat);
                        }
                        else if (depth >= 1)
                        {
                            float3 V2_new = (depth == 1) ? (-L) : load_Vpost_gi(g_Reservoirs_current_gi, px);
                            float3 tpost  = load_Tpost_gi(g_Reservoirs_current_gi, px);
                            if (depth > 1) tpost *= bdataNEE.val * cosSurf;

                            if (UpdateReservoirGI_Fast(g_Reservoirs_current_gi, px, wi, light.emission * tpost, V2_new, seed))
                                store_F_combined_gi(g_Reservoirs_current_gi, px, F_contrib);
                        }
                    }
                }
            }

            //Sun NEE
            {
                float2 rSun = float2(RandomFloatSingle(seed), RandomFloatSingle(seed));
                SunSampleResult sun = SampleSun(rSun);
                float3 hitN_sun = UnpackNormal(hitNormalPk);
                float NdotL = dot(hitN_sun, sun.direction);

                if (NdotL > 1e-6f && IsVisible(hitPos, hinfo.hitNormal, sun.direction, 10000.0f))
                {
                    float3 lKd = UnpackRGB9E5(matKdPk);
                    float  lPr = f16tof32_custom(matPrPmPk & 0xFFFFu);
                    float  lPm = f16tof32_custom(matPrPmPk >> 16u);
                    float3 hitN = UnpackNormal(hitNormalPk);
                    float3 throughput = UnpackRGB9E5(throughputPk);

                    SamplingP sp_nee = CalculateStrategyProbabilities(matID, -rayDir, hitN, iors.x, iors.y, lKd, lPm);
                    BrdfData bdataNEE = EvaluateAndPdf_COMBINED(sp_nee, matID, hitN, hinfo.hitNormal, sun.direction, -rayDir, lKd, lPr, lPm, iors.x, iors.y);

                    float lightPdf = sun.pdf;
                    float bsdfPdf  = bdataNEE.pdf;

                    if (lightPdf > 1e-20f && bsdfPdf > 0.0f)
                    {
                        float misWeight = lightPdf / (lightPdf + bsdfPdf);

                        float3 localMeasurement = sun.radiance * bdataNEE.val * NdotL;
                        float3 F_contrib = throughput * localMeasurement * gi_pdf_product;
                        float  p_hat     = GetPHat(F_contrib);
                        float  p_full    = gi_pdf_product * lightPdf;

                        // Depth 0: write final estimate directly to scratch (no ReSTIR for sun primary)
                        if (depth == 0)
                        {
                            float3 contrib = throughput * NdotL * sun.radiance * bdataNEE.val / lightPdf;
                            gScratchPing[uint3(pixel, 3)] = float4(misWeight * contrib, 0);
                        }
                        else if (depth >= 1)
                        {
                            float  wi = (p_full > 1e-20f) ? (misWeight * p_hat / p_full) : 0.0f;
                            uint px = MapPixelID(imgSize, pixel);
                            float3 V2_new = (depth == 2) ? (-sun.direction) : load_Vpost_gi(g_Reservoirs_current_gi, px);
                            float3 tpost  = load_Tpost_gi(g_Reservoirs_current_gi, px);
                            if (depth > 1) tpost *= bdataNEE.val * NdotL;

                            if (UpdateReservoirGI_Fast(g_Reservoirs_current_gi, px, wi, sun.radiance * tpost, V2_new, seed))
                                store_F_combined_gi(g_Reservoirs_current_gi, px, F_contrib);
                        }
                    }
                }
            }

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

        if (dot(hinfo.hitGNormal, s) < 0.0f)
            UpdateIORStack_packed(viorP, aiorP, matID, instID);

        if (dot(s, s) < 1e-12f || bdata.pdf <= 1e-6f || any(isnan(updateWeight)) || any(isinf(updateWeight)))
            break;

        prev_pdf       = bdata.pdf;
        gi_pdf_product = min(gi_pdf_product * bdata.pdf, 1e30f);
        rayDir      = s;
        float3 offsetN = dot(s, hinfo.hitGNormal) >= 0.0f ? hinfo.hitGNormal : -hinfo.hitGNormal;
        rayOrigin   = offset_ray(hitPos, offsetN);

        {
            float3 throughput = UnpackRGB9E5(throughputPk) * updateWeight;
            float3 tpostWeight = bdata.val * absorptionTint * cosTheta;

            if (depth > 0)
            {
                float survivalProb = min(1.0f, Luma(throughput));
                if (RandomFloatSingle(seed) >= survivalProb) break;
                float rrBoost = 1.0f / max(survivalProb, 0.1f);
                throughput  *= rrBoost;
                tpostWeight *= rrBoost;
                gi_pdf_product = min(gi_pdf_product * survivalProb, 1e30f);
            }

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

    //Final reservoir weight resolve
    uint finalPx = MapPixelID(DispatchRaysDimensions().xy, DispatchRaysIndex().xy);

    {
        float p_hat = load_phat_di(g_Reservoirs_current_di, finalPx);
        float wsum  = load_wsum_di(g_Reservoirs_current_di, finalPx);
        float W     = (p_hat > 1e-6f && wsum > 0.0f) ? (wsum / p_hat) : 0.0f;
        store_W_di(g_Reservoirs_current_di, finalPx, W);
        store_M_di(g_Reservoirs_current_di, finalPx, 1);
    }

    {
        float F_gi  = load_F_mag_gi(g_Reservoirs_current_gi, finalPx);
        float wsum  = load_wsum_gi(g_Reservoirs_current_gi, finalPx);
        float Wgi   = 0.0f;

        if (F_gi > 1e-6f && wsum > 0.0f)
        {
            Wgi = wsum / F_gi;
            if (isnan(Wgi) || isinf(Wgi)) Wgi = 0.0f;
        }

        if (Wgi == 0.0f)
            InvalidateReservoirGI_ShadingNormal(g_Reservoirs_current_gi, finalPx);

        store_W_gi(g_Reservoirs_current_gi, finalPx, Wgi);
        store_M_gi(g_Reservoirs_current_gi, finalPx, 1u);
    }
}
