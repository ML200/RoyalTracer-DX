#include "Includes_v8.hlsli"

#ifndef MAX_BOUNCES
#define MAX_BOUNCES 10
#endif

//====================================================================
//RAYGEN, UNIFIED DI + GI RESERVOIR
//====================================================================
//Direct-lighting samples, d = 2 (x0 -> x1 -> light), now compete with GI
//samples, d >= 3, in the same reservoir. DI candidates carry sentinel
//matIDs (MATID_ENV_MISS or MATID_LIGHT_TRI) and their DI-specific x2
//payload, GI candidates reuse the stashed depth-1 vertex state.
//
//Live register state (hot): seed, rayOrigin, rayDir, throughputPk,
//prevNormalPk, prev_pdf, pdf_product, tpost, wsum. That's it.
//
//Cold state pushed to memory:
//depth-1 vertex (x2, n2, uv, matID, objID, eta): write once at depth=1
//v2 direction: write once at depth=2
//RIS F (float3 contribution): written directly to the reservoir buffer on acceptance.
//All are compressed SoA in g_pathStateBuffer. Raygen-only, Pass_spat_gi_*
//overwrites the buffer later in the frame.

[shader("raygeneration")]
void Pass_raygen_v8()
{
    const uint2 pixel    = DispatchRaysIndex().xy;
    const uint2 imgSize  = DispatchRaysDimensions().xy;
    const uint  pixelIdx = MapPixelID(imgSize, pixel);

    //Clear scratch slots 1 and 3. Slot 1 carries primary emitter/sky hits,
    //written below on depth-0 emission. Slot 3 carries the depth-0 sun NEE.
    //Without clearing, shading reads stale prior-frame values for
    //non-emitter, non-sky pixels, visible as light trails.
    gScratchPing[uint3(pixel, 1)] = float4(0, 0, 0, 0);
    gScratchPing[uint3(pixel, 3)] = float4(0, 0, 0, 0);

    //Zero the reservoir. RIS acceptance overwrites on demand, this ensures
    //no-candidate-accepted pixels end up as empty reservoirs.
    storeReservoir(g_Reservoirs_current, pixelIdx, (Reservoir)0);

    //Seed the PathVertexState slot with safe sentinel defaults. Without
    //this, the pass-through continue at matNi <= 1+EPSILON, and any other
    //depth=1 early-break, leaves last frame's spatial-pass scratch in the
    //slot, which load_ps then decodes into a bogus matID/objID and hangs
    //the driver at specific camera angles.
    init_ps(g_pathStateBuffer, pixelIdx);

    //RIS state, only wsum is live across iterations. F is written
    //straight to the reservoir buffer on acceptance.
    float wsum = 0.0f;

    uint   seed         = initRandomData(pixel, uint2(8, 4), time, 1u);
    float3 rayOrigin    = InitOrigin();
    float3 rayDir       = InitDirection(pixel, float2(imgSize), seed);
    uint   throughputPk = PackRGB9E5(float3(1, 1, 1));
    uint   prevNormalPk = PackNormal(float3(0, 1, 0));
    float  prev_pdf     = 1.0f;
    float  pdf_product  = 1.0f;

    //tpost, post-x2 integrand accumulator, stays in registers. Updated
    //every bounce, so buffer RMW would be strictly worse than 3 scalars.
    float3 tpost = float3(1, 1, 1);

    //====================================================================
    //PATH LOOP
    //====================================================================
    [loop]
    for (int depth = 0; depth < MAX_BOUNCES; ++depth)
    {
        if (!IsRayValid(rayOrigin, rayDir, 10000.0f))
            break;

        RayDesc ray;
        ray.Origin    = rayOrigin;
        ray.Direction = rayDir;
        ray.TMin      = 0.00001f;
        ray.TMax      = 10000.0f;
        //Depth 0 keeps 4-state OMM, unknown states fall through to the
        //alpha any-hit. Secondary bounces force 2-state: unknown is
        //resolved by the pre-baked micromap and any-hit is skipped, which
        //is where the bulk of the alpha-test cost lives.
        const uint rayFlags = (depth == 0) ? RAY_FLAG_NONE : RAY_FLAG_FORCE_OMM_2_STATE;
        dx::HitObject hitObj = TraceRay_Custom(SceneBVH, ray, rayFlags, 0xFF);

        //====================================================================
        //MISS
        //====================================================================
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

            //pdf-free contribution f = throughput * pdf_product * envL
            const float3 F_contrib = throughput * envL * pdf_product;
            const float  p_hat     = GetPHat(F_contrib);
            const float  p_full    = pdf_product;
            const float  wi        = (p_full > 1e-20f) ? (p_hat / p_full) : 0.0f;

            bool accepted = false;
            if (depth == 1)
            {
                //DI env candidate, d=2 path, x2 stored as DIRECTION.
                accepted = AddInitialCandidate(wsum, g_Reservoirs_current, pixelIdx, wi,
                    rayDir, float3(0, 1, 0),          //x2=dir, n2=unit default
                    envL,   float3(0, 1, 0),          //L2, V2=unit default
                    float2(0, 0),
                    MATID_ENV_MISS, MATID_ENV_MISS, 1.0f,
                    F_contrib, seed);
            }
            else //depth >= 2: GI env, x2 = stashed depth-1 vertex
            {
                const PathVertexState ps = load_ps(g_pathStateBuffer, pixelIdx);
                accepted = AddInitialCandidate(wsum, g_Reservoirs_current, pixelIdx, wi,
                    ps.x2, ps.n2_s,
                    envL * tpost, ps.v2,
                    ps.uv,
                    ps.matID, ps.objID, ps.eta,
                    F_contrib, seed);
            }
            //Env miss = infinitely far but known-visible, we traced the
            //ray to get here. Clear any prior NEE-accepted shadow ray so
            //the end-of-raygen visibility trace is skipped.
            if (accepted) clear_shadow_ray(g_pathStateBuffer, pixelIdx);
            break;
        }

        //====================================================================
        //HIT SETUP
        //====================================================================
        const float  hitT   = hitObj.GetRayTCurrent();
        const float3 hitPos = rayOrigin + rayDir * hitT;

        const uint instID = hitObj.GetInstanceIndex();
        const uint primID = FlatPrimID(instID, hitObj.GetGeometryIndex(), hitObj.GetPrimitiveIndex());
        const uint matID  = GetMatIDFast(instID, primID);

        BuiltInTriangleIntersectionAttributes attr;
        hitObj.GetAttributes(attr);
        HitInfo hinfo = EvalSurfaceState(instID, primID, attr.barycentrics, rayOrigin, depth);

        //Backface-derived IOR pair: entering -> (air, matNi), exiting -> (matNi, air).
        //Null-IOR boundary, matNi ~= 1, is pure pass-through, skip the hit.
        const float matNi = LoadNi(matID);
        if (matNi <= 1.0f + EPSILON)
        {
            rayOrigin = hitPos;
            continue;
        }

        //Only flip the IOR pair, and declare an interior medium for
        //absorption, when the material is actually transmissive. An
        //opaque backface, thin single-sided geometry like leaves or paper,
        //or an inverted winding, has an IOR but no traversable inside,
        //swapping would apply a phantom Tf absorption to the incoming leg.
        const bool transmissive = LoadKd_w(matID) < 1.0f - EPSILON;
        const bool flipIOR = hinfo.backface && transmissive;
        const float2 iors = flipIOR ? float2(matNi, 1.0f) : float2(1.0f, matNi);
        const uint   mediumMatID = flipIOR ? matID : MEDIUM_INVALID;

        float3 hitLocalKd; float hitLocalPr, hitLocalPm;
        RefetchMaterial(matID, hinfo.uv, hitLocalKd, hitLocalPr, hitLocalPm, (uint)depth);

        const float3 absorptionTint = (mediumMatID != MEDIUM_INVALID)
            ? CalculateAbsorptionThroughput(LoadTf(mediumMatID), hitT)
            : float3(1, 1, 1);

        const float3 emission = GetEmissionFast(instID, primID);

        //====================================================================
        //DEPTH 0, PRIMARY HIT STORAGE
        //====================================================================
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

            //Specular motion vector reflection probe
            {
                const float3 reflDir    = reflect(rayDir, hinfo.hitNormal);
                const float3 reflOrigin = offset_ray(hitPos, hinfo.hitNormal);

                bool committed = false;
                float reflT = 0.0f;
                if (IsRayValid(reflOrigin, reflDir, 10000.0f))
                {
                    RayDesc reflRay;
                    reflRay.Origin    = reflOrigin;
                    reflRay.Direction = reflDir;
                    reflRay.TMin      = 0.00001f;
                    reflRay.TMax      = 10000.0f;

                    RayQuery<RAY_FLAG_NONE> q;
                    q.TraceRayInline(SceneBVH, RAY_FLAG_NONE, 0xFF, reflRay);
                    //Hard cap on alpha-test iterations, a grazing reflection
                    //ray through dense foliage can produce hundreds of
                    //non-opaque candidates, and certain driver states hit
                    //TDR before BVH traversal finishes. 128 is far more
                    //than any motion-vector probe needs.
                    uint alphaIter = 0;
                    while (q.Proceed() && alphaIter < 128u)
                    {
                        ++alphaIter;
                        if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
                        {
                            uint cInstID = q.CandidateInstanceIndex();
                            uint cPrimID = FlatPrimID(cInstID, q.CandidateGeometryIndex(), q.CandidatePrimitiveIndex());
                            uint cMatID  = GetMatIDFast(cInstID, cPrimID);
                            float alpha  = LoadAlphaThreshold(cMatID);
                            if (alpha < 1.0f)
                                q.CommitNonOpaqueTriangleHit();
                        }
                    }

                    committed = (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT);
                    if (committed) reflT = q.CommittedRayT();
                }

                if (committed)
                {
                    float3 reflPos    = reflOrigin + reflDir * reflT;
                    float3 virtualPos = reflPos - 2.0f * dot(reflPos - hitPos, hinfo.hitNormal) * hinfo.hitNormal;
                    gScratchPing[uint3(pixel, 4)] = float4(virtualPos, asfloat(instID));
                }
                else
                {
                    gScratchPing[uint3(pixel, 4)] = float4(0, 0, 0, asfloat(0xFFFFFFFFu));
                }
            }
        }

        //====================================================================
        //EMITTER HIT
        //====================================================================
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

            bool accepted = false;
            if (depth == 1)
            {
                //DI triangle-emitter candidate, d=2 path.
                accepted = AddInitialCandidate(wsum, g_Reservoirs_current, pixelIdx, wi,
                    hitPos, hinfo.hitNormal,
                    emission, float3(0, 1, 0),        //V2 unused for DI
                    float2(0, 0),
                    MATID_LIGHT_TRI, instID, 1.0f,
                    F_contrib, seed);
            }
            else //depth >= 2: GI emitter, x2 = stashed depth-1 vertex
            {
                const PathVertexState ps = load_ps(g_pathStateBuffer, pixelIdx);
                accepted = AddInitialCandidate(wsum, g_Reservoirs_current, pixelIdx, wi,
                    ps.x2, ps.n2_s,
                    emission * tpost, ps.v2,
                    ps.uv,
                    ps.matID, ps.objID, ps.eta,
                    F_contrib, seed);
            }
            //BSDF-sampled ray hit an emitter, visibility is implicit,
            //the path ray itself reached it. Clear deferred shadow.
            if (accepted) clear_shadow_ray(g_pathStateBuffer, pixelIdx);
            break;
        }

        //====================================================================
        //STASH DEPTH-1 VERTEX / V2
        //====================================================================
        if (depth == 1)
        {
            store_ps_depth1(g_pathStateBuffer, pixelIdx,
                            hitPos, hinfo.hitNormal,
                            hinfo.uv, matID, instID, iors.y);
        }

        if (depth == 2)
        {
            store_ps_v2(g_pathStateBuffer, pixelIdx, -rayDir);
        }

        //====================================================================
        //NEE
        //====================================================================
        const bool performNEE = !(mediumMatID != MEDIUM_INVALID || LoadKd_w(matID) < EPSILON);

        uint matKdPk, matPrPmPk, hitNormalPk;
        if (performNEE)
        {
            matKdPk     = PackRGB9E5(hitLocalKd);
            matPrPmPk   = f32tof16_custom(hitLocalPr) | (f32tof16_custom(hitLocalPm) << 16u);
            hitNormalPk = PackNormal(hinfo.hitNormal);

            //====================================================================
            //POINT LIGHT NEE
            //====================================================================
            //Visibility is DEFERRED, we add the candidate assuming it's
            //unoccluded, then if it wins RIS we stash the shadow-ray info
            //in the path-state scratch. The end-of-raygen resolve traces
            //exactly one shadow ray, for the final winning NEE sample.
            {
                LT_LightSampleResult light = LT_SamplePointOnLight(hitPos, hinfo.hitNormal, seed);

                const float3 toLight = light.position - hitPos;
                const float  distSq  = dot(toLight, toLight);
                const float  dist    = sqrt(distSq);
                const float3 L       = toLight / dist;

                const float  cosSurf  = dot(hinfo.hitNormal, L);
                const float  cosLightS = dot(light.normal, -L);

                if (cosSurf > 1e-6f && cosLightS > 1e-6f)
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

                        bool accepted = false;
                        if (depth == 0)
                        {
                            //DI NEE at primary vertex, d=2 path, x2 = light.
                            accepted = AddInitialCandidate(wsum, g_Reservoirs_current, pixelIdx, wi,
                                light.position, light.normal,
                                light.emission, float3(0, 1, 0),
                                float2(0, 0),
                                MATID_LIGHT_TRI, light.objID, 1.0f,
                                F_contrib, seed);
                        }
                        else if (depth == 1)
                        {
                            //GI NEE at depth-1 vertex, d=3 path, x2 = current hit.
                            accepted = AddInitialCandidate(wsum, g_Reservoirs_current, pixelIdx, wi,
                                hitPos, hinfo.hitNormal,
                                light.emission, -L,
                                hinfo.uv,
                                matID, instID, iors.y,
                                F_contrib, seed);
                        }
                        else //depth >= 2: GI NEE past depth-1 vertex.
                        {
                            const PathVertexState ps = load_ps(g_pathStateBuffer, pixelIdx);
                            //tpost so far excludes current BSDF, NEE adds it.
                            const float3 tpostNEE = tpost * bdataNEE.val * cosSurf;
                            accepted = AddInitialCandidate(wsum, g_Reservoirs_current, pixelIdx, wi,
                                ps.x2, ps.n2_s,
                                light.emission * tpostNEE, ps.v2,
                                ps.uv,
                                ps.matID, ps.objID, ps.eta,
                                F_contrib, seed);
                        }

                        if (accepted)
                        {
                            const float3 shadowOrigin = offset_ray(
                                hitPos,
                                dot(L, hinfo.hitNormal) >= 0.0f ? hinfo.hitNormal : -hinfo.hitNormal);
                            store_shadow_ray(g_pathStateBuffer, pixelIdx,
                                             shadowOrigin, L, dist * 0.999f);
                        }
                    }
                }
            }

            //====================================================================
            //SUN NEE
            //====================================================================
            //depth >= 1 defers visibility via the shadow-ray scratch.
            //depth == 0 bypasses the reservoir, direct scratch write, and
            //still traces its shadow ray inline, there's no RIS winner
            //to gate against.
            {
                float2 rSun = float2(RandomFloatSingle(seed), RandomFloatSingle(seed));
                SunSampleResult sun = SampleSun(rSun);
                const float3 hitN_sun = UnpackNormal(hitNormalPk);
                const float  NdotL    = dot(hitN_sun, sun.direction);

                if (NdotL > 1e-6f)
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
                            //Sun stays excluded from the unified reservoir,
                            //user decision: no sun in DI bounce. Write direct,
                            //keep inline visibility, this path doesn't feed RIS.
                            if (IsVisible(hitPos, hinfo.hitNormal, sun.direction, 10000.0f))
                            {
                                float3 contrib = throughput * NdotL * sun.radiance * bdataNEE.val / lightPdf;
                                gScratchPing[uint3(pixel, 3)] = float4(misWeight * contrib, 0);
                            }
                        }
                        else
                        {
                            const float wi = (p_full > 1e-20f) ? (misWeight * p_hat / p_full) : 0.0f;
                            bool accepted = false;
                            if (depth == 1)
                            {
                                accepted = AddInitialCandidate(wsum, g_Reservoirs_current, pixelIdx, wi,
                                    hitPos, hinfo.hitNormal,
                                    sun.radiance, -sun.direction,
                                    hinfo.uv,
                                    matID, instID, iors.y,
                                    F_contrib, seed);
                            }
                            else //depth >= 2
                            {
                                const float3 tpostNEE = tpost * bdataNEE.val * NdotL;
                                const PathVertexState ps = load_ps(g_pathStateBuffer, pixelIdx);
                                accepted = AddInitialCandidate(wsum, g_Reservoirs_current, pixelIdx, wi,
                                    ps.x2, ps.n2_s,
                                    sun.radiance * tpostNEE, ps.v2,
                                    ps.uv,
                                    ps.matID, ps.objID, ps.eta,
                                    F_contrib, seed);
                            }

                            if (accepted)
                            {
                                const float3 shadowOrigin = offset_ray(
                                    hitPos,
                                    dot(sun.direction, hinfo.hitNormal) >= 0.0f ? hinfo.hitNormal : -hinfo.hitNormal);
                                store_shadow_ray(g_pathStateBuffer, pixelIdx,
                                                 shadowOrigin, sun.direction, 10000.0f);
                            }
                        }
                    }
                }
            }

            hitLocalKd = UnpackRGB9E5(matKdPk);
            hitLocalPr = f16tof32_custom(matPrPmPk & 0xFFFFu);
            hitLocalPm = f16tof32_custom(matPrPmPk >> 16u);
            hinfo.hitNormal = UnpackNormal(hitNormalPk);
        }

        //====================================================================
        //SAMPLE NEXT BSDF DIRECTION
        //====================================================================
        SamplingP sp = CalculateStrategyProbabilities(matID, -rayDir, hinfo.hitNormal, iors.x, iors.y, hitLocalKd, hitLocalPm);
        float3    s  = SampleBRDF(sp, matID, -rayDir, hinfo.hitNormal, hinfo.hitNormal, hitLocalKd, hitLocalPr, hitLocalPm, seed, iors.x, iors.y);
        BrdfData  bdata = EvaluateAndPdf_COMBINED(sp, matID, hinfo.hitNormal, hinfo.hitNormal, s, -rayDir, hitLocalKd, hitLocalPr, hitLocalPm, iors.x, iors.y);

        const float  cosTheta     = abs(dot(hinfo.hitNormal, s));
        const float3 updateWeight = (bdata.pdf > 1e-6f)
            ? (bdata.val * absorptionTint * cosTheta) / bdata.pdf
            : float3(0, 0, 0);

        if (dot(s, s) < 1e-12f || bdata.pdf <= 1e-6f || any(isnan(updateWeight)) || any(isinf(updateWeight)))
            break;

        prev_pdf    = bdata.pdf;
        pdf_product = min(pdf_product * bdata.pdf, 1e30f);
        rayDir      = s;
        float3 offsetN = dot(s, hinfo.hitNormal) >= 0.0f ? hinfo.hitNormal : -hinfo.hitNormal;
        rayOrigin   = offset_ray(hitPos, offsetN);

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
                //RR survival is part of the path pdf, include in product.
                pdf_product = min(pdf_product * survivalProb, 1e30f);
            }

            //Accumulate tpost locally, no reservoir roundtrip.
            if (depth >= 2)
                tpost *= tpostWeight;

            throughputPk = PackRGB9E5(throughput);
        }
        prevNormalPk = PackNormal(hinfo.hitNormal);
    }

    //====================================================================
    //FINAL RESOLVE
    //====================================================================
    //Commit wsum / W / M. F was already written to the reservoir by
    //AddInitialCandidate on the last acceptance.
    //
    //Deferred shadow: NEE candidates are added without visibility, if the
    //final RIS winner is an NEE sample, the shadow-ray scratch holds its
    //origin + dir + dist (dist > 0). Trace once here, if occluded, the
    //sample's contribution is invalidated via W = 0.
    {
        const float F_mag = GetPHat(load_F(g_Reservoirs_current, pixelIdx));
        float W = 0.0f;
        if (F_mag > 1e-6f && wsum > 0.0f)
        {
            const ShadowRayInfo shadow = load_shadow_ray(g_pathStateBuffer, pixelIdx);
            const bool visible = (shadow.dist <= 0.0f) || IsVisibleOffset(shadow.origin, shadow.dir, shadow.dist);
            if (visible)
            {
                W = wsum / F_mag;
                if (isnan(W) || isinf(W) || W < 0.0f) W = 0.0f;
            }
        }

        store_wsum(g_Reservoirs_current, pixelIdx, wsum);
        store_W   (g_Reservoirs_current, pixelIdx, W);
        store_M   (g_Reservoirs_current, pixelIdx, 1u);

        if (W == 0.0f)
            InvalidateReservoir_ShadingNormal(g_Reservoirs_current, pixelIdx);
    }
}
