#include "Includes_v8.hlsli"
#include "Raygen_Common_v8.hlsli"


//One reservoir: emissive-tri DI (d==1), first-bounce DI (d==2), GI (d>=3).
//d==1 direct sky+sun -> scratch slot 3; cold state in g_pathStateBuffer.
//
//The camera ray now lives in Pass_camera_v8: it wrote the primary G-buffer +
//scratch (slots 1/2/4), the SPMIS hash cell, the primaryExtra (iors/medium/
//absorption) HOT2 record, and finalized terminal pixels (miss / degenerate /
//direct emitter, all flagged SD_FLAG_NOBOUNCE). This pass rebuilds the primary
//ctx from that G-buffer per sample and runs the initial-sample + bounce loops.
[shader("raygeneration")]
void Pass_raygen_v8()
{
    const uint2 pixel    = DispatchRaysIndex().xy;
    const uint2 imgSize  = DispatchRaysDimensions().xy;
    const uint  pixelIdx = MapPixelID(imgSize, pixel);

    //terminal pixels (miss / degenerate / direct emitter) were finalized by
    //Pass_camera and flagged NOBOUNCE — skip them, reservoir already invalid.
    if (load_flagsWord(g_sample_current, pixelIdx) & SD_FLAG_NOBOUNCE)
        return;

    storeReservoir(g_Reservoirs_current, pixelIdx, (Reservoir)0);

    float wsum = 0.0f;

    //sky/sun observer at the camera altitude (absolute = view + sceneOrigin).
    SetSkyObserver(InitOrigin() + sceneOriginWorld);

    //directAtX1 in scratch (HOT1[12..15], RGB9E5); summed across the N samples,
    // /N at finalize.
    store_rg_directX1(g_pathStateBuffer, pixelIdx, float3(0, 0, 0));

    //Primary incoming (view) direction recovered from the camera origin + primary
    //hit. DoF lens jitter is sub-pixel and dropped here, matching the temporal
    //pass's NoV convention. Seeds the depth==1 back-edge carrier (rayDirPk0).
    const uint   primInst0 = load_instID(g_sample_current, pixelIdx);
    const float3 primPos0  = load_x1_with_instID(g_sample_current, pixelIdx, primInst0);
    const uint   rayDirPk0 = PackNormal(normalize(primPos0 - InitOrigin()));

    HitContext ctx;
    uint   seed;
    uint   throughputPk;
    uint   prevNormalPk;
    float  prev_pdf;
    float  pdf_product;
    uint   rayDirPk;
    float3 rayOrigin;

    //====================================
    //INITIAL-SAMPLE LOOP (pt_initialSamples)
    //====================================
    //Re-run the candidate bounce loop N times into ONE reservoir (wsum streams
    //all N; directAtX1 sums, /N at finalize). The shared primary hit is rebuilt
    //per-sample from the G-buffer + HOT2 extras rather than kept live across the
    //inner traces (that halves the continuation live-state). Only per-path
    //carriers reset per sample.
    [loop]
    for (uint s = 0u; s < pt_initialSamples; ++s)
    {
        //rebuild the shared primary ctx from the G-buffer + HOT2 extras (same
        //quantized surface as the reuse passes, so N==1 differs imperceptibly
        //from the pre-spill build).
        const uint primInst = load_instID(g_sample_current, pixelIdx);
        ctx.instID          = primInst;
        ctx.hitPos          = load_x1_with_instID  (g_sample_current, pixelIdx, primInst);
        ctx.hitNormal       = load_n1_s_with_instID(g_sample_current, pixelIdx, primInst);
        ctx.matID           = load_matID  (g_sample_current, pixelIdx);
        ctx.backface        = load_backface(g_sample_current, pixelIdx);
        ctx.hitLocalKd      = (half3)load_kd(g_sample_current, pixelIdx);
        float pr_, pm_; load_prpm(g_sample_current, pixelIdx, pr_, pm_);
        ctx.hitLocalPr      = (half)pr_;
        ctx.hitLocalPm      = (half)pm_;
        float2 iors_; uint medium_; float3 absorb_;
        load_rg_primaryExtra(g_pathStateBuffer, pixelIdx, iors_, medium_, absorb_);
        ctx.iors            = (half2)iors_;
        ctx.mediumMatID     = medium_;
        ctx.absorptionTint  = (half3)absorb_;

        //per-sample carrier reset. seed is re-init every sample now that the
        //camera ray lives in Pass_camera (s==0 no longer inherits a post-camera
        //seed); the depth==1 back-edge starts from the recovered view dir.
        seed = initRandomData(pixel, uint2(8, 4), time, s + 1u);
        throughputPk = PackRGB9E5(float3(1, 1, 1));
        prevNormalPk = PackNormal(float3(0, 1, 0));
        prev_pdf     = 1.0f;
        pdf_product  = 1.0f;
        store_rg_tpost(g_pathStateBuffer, pixelIdx, float3(1, 1, 1));
        rayDirPk     = rayDirPk0;

    //====================================
    //BOUNCE LOOP
    //====================================
    //Process v_depth (ctx), then trace to v_{depth+1}. NEE then BSDF at the top;
    //the bottom trace is the BSDF MIS partner and fires every iter. depth==1
    //primary (sky+sun -> directAtX1, emissive DI -> reservoir); ==2 first bounce
    //(stores ps); >=3 GI from PathVertexState, RR + tpost active.
    [loop]
    for (int16_t depth = 1; depth < (int)pt_maxBounces; ++depth)
    {
        float3 rayDir = UnpackNormal(rayDirPk);

        //Strategy probabilities are fixed for this bounce (matID, -rayDir, normal, iors,
        //Kd, Pm don't change within the iteration). Compute once and reuse for both NEE
        //techniques and the BSDF sample instead of three identical evals — the two NEE
        //sites sit behind separate IsVisible branches so DXC can't CSE them itself.
        const SamplingP sp = CalculateStrategyProbabilities(ctx.matID, -rayDir, ctx.hitNormal, ctx.iors.x, ctx.iors.y, ctx.hitLocalKd, ctx.hitLocalPm);

        //------------- NEE: point light + sun (MIS partner of the bottom trace) -------------
        const bool performNEE = !(ctx.mediumMatID != MEDIUM_INVALID || LoadKd_w(ctx.matID) < EPSILON);
        if (performNEE)
        {
            //point-light technique, visibility inline
            {
                LT_LightSampleResult light = LT_SamplePointOnLight(ctx.hitPos, ctx.hitNormal, seed);

                const float3 toLight = light.position - ctx.hitPos;
                const float  dist    = sqrt(dot(toLight, toLight));
                const float3 L       = toLight / dist;

                const float cosSurf   = dot(ctx.hitNormal, L);
                const float cosLightS = dot(light.normal, -L);

                if (cosSurf > 1e-6f && cosLightS > 1e-6f && light.pdfSolidAngle > 1e-20f)
                {
                    //visibility first: keep the strategy+BSDF eval out of the
                    //cross-ray live set; run it only if unoccluded.
                    if (IsVisible(ctx.hitPos, ctx.hitNormal, light.position, light.normal))
                    {
                        const float3 throughput = UnpackRGB9E5(throughputPk);

                        BrdfData  bdataNEE = EvaluateAndPdf_COMBINED(sp, ctx.matID, ctx.hitNormal, ctx.hitNormal, L, -rayDir, ctx.hitLocalKd, ctx.hitLocalPr, ctx.hitLocalPm, ctx.iors.x, ctx.iors.y);

                        const float lightPdf = light.pdfSolidAngle;
                        const float bsdfPdf  = bdataNEE.pdf;

                        if (bsdfPdf > 0.0f)
                        {
                            const float  misWeight        = lightPdf / (lightPdf + bsdfPdf);
                            const float3 localMeasurement = light.emission * bdataNEE.val * cosSurf;
                            const float3 F_contrib        = throughput * localMeasurement * pdf_product;
                            const float  p_hat            = GetPHat(F_contrib);
                            const float  p_full           = pdf_product * lightPdf;
                            const float  wi               = (p_full > 1e-20f) ? (misWeight * p_hat / p_full) : 0.0f;

                            if (depth == 1)
                            {
                                //primary DI, x2 is the light
                                AddInitialCandidate(wsum, g_Reservoirs_current, pixelIdx, wi,
                                    light.position, light.normal,
                                    light.emission, diMarkerFor(pixelIdx, time),
                                    float3(0,0,0), 0.0f, 0.0f,
                                    MATID_LIGHT_TRI, light.objID, 1.0f,
                                    F_contrib, seed);
                            }
                            else if (depth == 2)
                            {
                                //first-bounce DI, x2 is the surface
                                AddInitialCandidate(wsum, g_Reservoirs_current, pixelIdx, wi,
                                    ctx.hitPos, ctx.hitNormal,
                                    light.emission, -L,
                                    (float3)ctx.hitLocalKd, (float)ctx.hitLocalPr, (float)ctx.hitLocalPm,
                                    ctx.matID, ctx.instID, ctx.iors.y,
                                    F_contrib, seed);
                            }
                            else
                            {
                                const PathVertexState ps = load_ps(g_pathStateBuffer, pixelIdx);
                                const float3 tpost    = load_rg_tpost(g_pathStateBuffer, pixelIdx);
                                const float3 tpostNEE = tpost * bdataNEE.val * cosSurf;
                                AddInitialCandidate(wsum, g_Reservoirs_current, pixelIdx, wi,
                                    ps.x2, ps.n2_s,
                                    light.emission * tpostNEE, ps.v2,
                                    ps.Kd, ps.Pr, ps.Pm,
                                    ps.matID, ps.objID, ps.eta,
                                    F_contrib, seed);
                            }
                        }
                    }
                }
            }

            //sun technique
            {
                const float2 rSun = float2(RandomFloatSingle(seed), RandomFloatSingle(seed));
                SunSampleResult sun = SampleSun(rSun);
                const float  NdotL = dot(ctx.hitNormal, sun.direction);

                if (NdotL > 1e-6f && sun.pdf > 1e-20f)
                {
                    if (IsVisible(ctx.hitPos, ctx.hitNormal, ctx.hitPos + sun.direction * RAY_TMAX_PLANET, -sun.direction))
                    {
                    if (cloud_cloudShadowOnSurfaces > 0.5f)
                    {
                        float2 rCone = float2(RandomFloatSingle(seed), RandomFloatSingle(seed));
                        float  cosCone = cos(SURFACE_CLOUD_SHADOW_CONE_DEG * DEG2RAD);
                        float3 Lj = SampleConeAroundDir(sun.direction, cosCone, rCone);
                        float  vis = CloudSunVisibility(ctx.hitPos + sceneOriginWorld, Lj,
                                                        RandomFloatSingle(seed));
                        sun.radiance *= pow(max(vis, 1e-6f), SURFACE_CLOUD_SHADOW_SOFTNESS);
                    }

                    const float3 throughput = UnpackRGB9E5(throughputPk);

                    BrdfData  bdataNEE = EvaluateAndPdf_COMBINED(sp, ctx.matID, ctx.hitNormal, ctx.hitNormal, sun.direction, -rayDir, ctx.hitLocalKd, ctx.hitLocalPr, ctx.hitLocalPm, ctx.iors.x, ctx.iors.y);

                    const float lightPdf = sun.pdf;
                    const float bsdfPdf  = bdataNEE.pdf;

                    if (bsdfPdf > 0.0f)
                    {
                        const float  misWeight        = lightPdf / (lightPdf + bsdfPdf);
                        const float3 localMeasurement = sun.radiance * bdataNEE.val * NdotL;
                        const float3 F_contrib        = throughput * localMeasurement * pdf_product;
                        const float  p_hat            = GetPHat(F_contrib);
                        const float  p_full           = pdf_product * lightPdf;

                            const float wi = (p_full > 1e-20f) ? (misWeight * p_hat / p_full) : 0.0f;

                            if (depth == 1)
                            {
                                //primary direct sun -> directAtX1, out of the reservoir
                                add_rg_directX1(g_pathStateBuffer, pixelIdx, DirectContribution(wi, F_contrib));
                            }
                            else if (depth == 2)
                            {
                                AddInitialCandidate(wsum, g_Reservoirs_current, pixelIdx, wi,
                                    ctx.hitPos, ctx.hitNormal,
                                    sun.radiance, -sun.direction,
                                    (float3)ctx.hitLocalKd, (float)ctx.hitLocalPr, (float)ctx.hitLocalPm,
                                    ctx.matID, ctx.instID, ctx.iors.y,
                                    F_contrib, seed);
                            }
                            else
                            {
                                const float3 tpost    = load_rg_tpost(g_pathStateBuffer, pixelIdx);
                                const float3 tpostNEE = tpost * bdataNEE.val * NdotL;
                                const PathVertexState ps = load_ps(g_pathStateBuffer, pixelIdx);
                                AddInitialCandidate(wsum, g_Reservoirs_current, pixelIdx, wi,
                                    ps.x2, ps.n2_s,
                                    sun.radiance * tpostNEE, ps.v2,
                                    ps.Kd, ps.Pr, ps.Pm,
                                    ps.matID, ps.objID, ps.eta,
                                    F_contrib, seed);
                            }
                        }
                    }
                }
            }
        }

        //------------- BSDF sample (MIS partner for NEE; bottom trace evaluates it) -------------
        //sp hoisted to the top of the bounce (reused by both NEE techniques above).
        const float3 s       = SampleBRDF        (sp, ctx.matID, -rayDir, ctx.hitNormal, ctx.hitNormal, ctx.hitLocalKd, ctx.hitLocalPr, ctx.hitLocalPm, seed, ctx.iors.x, ctx.iors.y, false);
        const BrdfData bdata = EvaluateAndPdf_COMBINED(sp, ctx.matID, ctx.hitNormal, ctx.hitNormal, s, -rayDir, ctx.hitLocalKd, ctx.hitLocalPr, ctx.hitLocalPm, ctx.iors.x, ctx.iors.y, false);

        const float  cosTheta     = abs(dot(ctx.hitNormal, s));
        const float3 updateWeight = (bdata.pdf > 1e-6f)
            ? (bdata.val * ctx.absorptionTint * cosTheta) / bdata.pdf
            : float3(0, 0, 0);

        //state update + termination. bad BSDF sample -> no valid next direction
        if (dot(s, s) < 1e-12f || bdata.pdf <= 1e-6f ||
            any(isnan(updateWeight)) || any(isinf(updateWeight)))
            break;

        prev_pdf     = bdata.pdf;
        pdf_product  = min(pdf_product * bdata.pdf, 1e30f);
        rayDir       = s;
        rayDirPk     = PackNormal(s);  //carrier for next iter back-edge

        //v_2 reconnection dir (v_3 -> v_2 = -s), stored at the depth==2 bottom so
        //the depth==2 miss/emitter candidates read the real direction.
        if (depth == 2)
            store_ps_v2(g_pathStateBuffer, pixelIdx, -rayDir);

        const float3 offsetN = (dot(s, ctx.hitNormal) >= 0.0f) ? ctx.hitNormal : -ctx.hitNormal;
        rayOrigin    = offset_ray(ctx.hitPos, offsetN);
        //park rayOrigin in scratch (RAY_O, free under raygen) so it doesn't cross
        //the reorder; reloaded post-trace.
        store_ray_origin(g_pathStateBuffer, pixelIdx, rayOrigin);

        float3 throughput        = UnpackRGB9E5(throughputPk) * updateWeight;
        const float3 tpostWeight = bdata.val * ctx.absorptionTint * cosTheta;

        //Russian roulette (off when pt_rrStartDepth high). The 1/p boost cancels
        //in the RR-clean target F (via pdf_product) but survives in wi; must NOT
        //touch the tpost update.
        if (depth >= (int)pt_rrStartDepth)
        {
            const float survivalProb = max(min(1.0f, Luma(throughput)), 0.05f);
            if (RandomFloatSingle(seed) >= survivalProb) break;
            const float rrBoost = 1.0f / survivalProb;
            throughput  *= rrBoost;
            pdf_product  = min(pdf_product * survivalProb, 1e30f);
        }

        //tpost: post-v_2 suffix throughput across v_3..v_{D-1}. Every depth>=4 GI
        //reconnection candidate reads it (correctness), so it runs for depth>=3
        //regardless of RR; tpostWeight is the RR-unboosted f*cos.
        if (depth >= 3)
        {
            const float3 tpost = load_rg_tpost(g_pathStateBuffer, pixelIdx) * tpostWeight;
            store_rg_tpost(g_pathStateBuffer, pixelIdx, tpost);
        }

        throughputPk = PackRGB9E5(throughput);
        prevNormalPk = PackNormal(ctx.hitNormal);

        //----- TRACE v_depth -> v_{depth+1} (BSDF technique, NEE's MIS partner) -----
        if (!IsRayValid(rayOrigin, rayDir, 10000.0f))
            break;

        RayDesc rayB;
        rayB.Origin    = rayOrigin;
        rayB.Direction = rayDir;
        rayB.TMin      = 0.00001f;
        rayB.TMax      = RAY_TMAX_PLANET;
        //secondaries force 2-state to skip alpha any-hit
        dx::HitObject hitObjB = TraceRay_Custom(SceneBVH, rayB, RAY_FLAG_FORCE_OMM_2_STATE, 0xFF);

        //reload rayOrigin post-trace (globallycoherent forces the real read,
        //keeping it off the reorder — same trick as tpost).
        const float3 rayOriginR = load_ray_origin(g_pathStateBuffer, pixelIdx);

        //----- MISS: GI miss tail (sky + MIS sun disc) -----
        if (!hitObjB.IsHit())
        {
            const float3 throughputCur = UnpackRGB9E5(throughputPk);

            //evaluate sky/sun from the BOUNCE origin (observer-dependent; the
            //camera observer baked a nadir ring into indirect light).
            SetSkyObserver(rayOriginR + sceneOriginWorld);

            const float  sunSAPdf   = GetSunPdf(rayDir);
            const float3 sunRad     = (sunSAPdf > 0.0f) ? EvaluateSun(rayDir) : float3(0, 0, 0);
            const float  sunMisBsdf = (sunSAPdf > 0.0f)
                ? prev_pdf / max(prev_pdf + sunSAPdf, EPSILON) : 0.0f;
            float3 cloudTr;
            const float3 sky        = EvaluateSky(rayDir, cloudTr);
            const float3 envL       = sky + sunRad * sunMisBsdf * cloudTr;

            const float3 F_contrib = throughputCur * envL * pdf_product;
            const float  p_hat     = GetPHat(F_contrib);
            const float  wi        = (pdf_product > 1e-20f) ? (p_hat / pdf_product) : 0.0f;

            //depth==1 miss = x1 direct sky -> directAtX1; depth>=2 uses stored v_2.
            if (depth == 1)
            {
                add_rg_directX1(g_pathStateBuffer, pixelIdx, DirectContribution(wi, F_contrib));
            }
            else
            {
                const PathVertexState ps = load_ps(g_pathStateBuffer, pixelIdx);
                const float3 tpost = load_rg_tpost(g_pathStateBuffer, pixelIdx);
                AddInitialCandidate(wsum, g_Reservoirs_current, pixelIdx, wi,
                    ps.x2, ps.n2_s,
                    envL * tpost, ps.v2,
                    ps.Kd, ps.Pr, ps.Pm,
                    ps.matID, ps.objID, ps.eta,
                    F_contrib, seed);
            }
            break;
        }

        //----- HIT: extract v_{depth+1} -----
        const float  hitT_n   = hitObjB.GetRayTCurrent();
        const float3 hitPos_n = rayOriginR + rayDir * hitT_n;
        const uint   instID_n = hitObjB.GetInstanceID();
        const uint    primID_n = FlatPrimID(instID_n, hitObjB.GetGeometryIndex(), hitObjB.GetPrimitiveIndex());
        const uint    matID_n  = GetMatIDFast(instID_n, primID_n);
        BuiltInTriangleIntersectionAttributes attrB;
        hitObjB.GetAttributes(attrB);
        HitInfo hinfo_n = EvalSurfaceState(instID_n, primID_n, attrB.barycentrics, rayOriginR, (uint)depth);

        const float  matNi_n        = LoadNi(matID_n);
        const bool   transmissive_n = LoadKd_w(matID_n) < 1.0f - EPSILON;
        const bool   flipIOR_n      = hinfo_n.backface && transmissive_n;
        const float2 iors_n         = flipIOR_n ? float2(matNi_n, 1.0f) : float2(1.0f, matNi_n);
        const uint   mediumMatID_n  = flipIOR_n ? matID_n : MEDIUM_INVALID;

        float3 hitLocalKd_n; float hitLocalPr_n, hitLocalPm_n;
        RefetchMaterial(matID_n, hinfo_n.uv, hitLocalKd_n, hitLocalPr_n, hitLocalPm_n, (uint)depth);

        const float3 absorptionTint_n = (mediumMatID_n != MEDIUM_INVALID)
            ? CalculateAbsorptionThroughput(LoadTf(mediumMatID_n), hitT_n)
            : float3(1, 1, 1);

        const float3 emission_n = GetEmissionFast(instID_n, primID_n);

        //----- Emitter hit at v_{depth+1}: BSDF-technique direct emission -----
        if (any(emission_n > 0.0f) && hinfo_n.lightID != 0xFFFFFFFFu)
        {
            const float3 throughputCur = UnpackRGB9E5(throughputPk);
            const float3 prevNormalCur = UnpackNormal(prevNormalPk);
            const float  lightPdfArea  = LT_Pdf_LightTree_Area(rayOriginR, prevNormalCur, hinfo_n.lightID, instID_n);
            const float  cosLight      = max(dot(hinfo_n.hitNormal, -rayDir), 0.0f);
            const float  dist2         = max(hitT_n * hitT_n, EPSILON);
            const float  lightPdfSA    = (cosLight > EPSILON) ? (lightPdfArea * dist2 / cosLight) : 0.0f;
            const float  misWeight     = prev_pdf / max(prev_pdf + lightPdfSA, EPSILON);

            const float3 F_contrib = throughputCur * emission_n * pdf_product;
            const float  p_hat     = GetPHat(F_contrib);
            const float  wi        = (pdf_product > 1e-20f) ? (misWeight * p_hat / pdf_product) : 0.0f;

            if (depth == 1)
            {
                AddInitialCandidate(wsum, g_Reservoirs_current, pixelIdx, wi,
                    hitPos_n, hinfo_n.hitNormal,
                    emission_n, diMarkerFor(pixelIdx, time),
                    float3(0,0,0), 0.0f, 0.0f,
                    MATID_LIGHT_TRI, instID_n, 1.0f,
                    F_contrib, seed);
            }
            else
            {
                const PathVertexState ps = load_ps(g_pathStateBuffer, pixelIdx);
                const float3 tpost = load_rg_tpost(g_pathStateBuffer, pixelIdx);
                AddInitialCandidate(wsum, g_Reservoirs_current, pixelIdx, wi,
                    ps.x2, ps.n2_s,
                    emission_n * tpost, ps.v2,
                    ps.Kd, ps.Pr, ps.Pm,
                    ps.matID, ps.objID, ps.eta,
                    F_contrib, seed);
            }
            break;
        }

        //----- Non-emitter hit: stash v_2 PathVertexState at the first bounce -----
        if (depth == 1)
        {
            store_ps_depth1(g_pathStateBuffer, pixelIdx,
                            hitPos_n, hinfo_n.hitNormal,
                            matID_n, instID_n, iors_n.y,
                            hitLocalKd_n, hitLocalPr_n, hitLocalPm_n);
        }

        //----- Carry v_{depth+1} into ctx (bounded fields -> half) -----
        ctx.hitPos         = hitPos_n;
        ctx.hitNormal      = hinfo_n.hitNormal;
        ctx.matID          = matID_n;
        ctx.instID         = instID_n;
        ctx.backface       = hinfo_n.backface;
        ctx.hitLocalKd     = (half3)hitLocalKd_n;
        ctx.hitLocalPr     = (half) hitLocalPr_n;
        ctx.hitLocalPm     = (half) hitLocalPm_n;
        ctx.iors           = (half2)iors_n;
        ctx.mediumMatID    = mediumMatID_n;
        ctx.absorptionTint = (half3)absorptionTint_n;
    }

    }   // end initial-sample loop (s)

    //====================================
    //FINALIZE
    //====================================
    //x1 direct sky+sun -> scratch slot 3 (mean of the N single-sample estimates).
    const float3 directAtX1    = load_rg_directX1(g_pathStateBuffer, pixelIdx);
    const float3 directAtX1Avg = directAtX1 / max(1.0f, (float)pt_initialSamples);
    gScratchPing[uint3(pixel, 3)] = float4(X1_DIRECT_OFF ? float3(0, 0, 0) : directAtX1Avg, 0);

    //RIS-over-N: wsum sums to ~N, so /N for unbiased W (selection is scale-invariant).
    wsum /= max(1.0f, (float)pt_initialSamples);
    FinalizeReservoir(pixelIdx, wsum);
}
