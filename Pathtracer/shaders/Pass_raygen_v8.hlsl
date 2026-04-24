#include "Includes_v8.hlsli"
#include "Nrc_v8.hlsli"

//safety net, RR at depth>2 is the real termination, NRC cache shortens rendering paths
#ifndef MAX_BOUNCES
#define MAX_BOUNCES 32
#endif

//====================================
//RAYGEN UNIFIED DI+GI RESERVOIR
//====================================
//DI samples (d=2) and GI samples (d>=3) compete in the same reservoir
//DI uses sentinel matIDs (MATID_ENV_MISS, MATID_LIGHT_TRI), GI reuses stashed depth-1 state
//hot state in regs: seed, rayOrigin, rayDir, throughputPk, prevNormalPk, prev_pdf, pdf_product, tpost, wsum
//cold state (x2, n2, uv, matID, objID, eta, v2) stashed in g_pathStateBuffer

[shader("raygeneration")]
void Pass_raygen_v8()
{
    const uint2 pixel    = DispatchRaysIndex().xy;
    const uint2 imgSize  = DispatchRaysDimensions().xy;
    const uint  pixelIdx = MapPixelID(imgSize, pixel);

    //====================================
    //NRC PATH CLASSIFICATION AND SER SORT
    //====================================
    //class 0 render, class 1 train biased, class 2 train unbiased
    //sort by class groups long class-2 into coherent warps
    const bool kNrcEnabled    = NrcIsEnabled();
    const bool kNrcTrainOn    = NrcIsTrainOn();
    //adaptive tile side from renderer feedback loop
    const uint  nrcTileSide   = NrcTrainingTileSide();
    const uint2 nrcTileOffset = uint2(asuint(time) % nrcTileSide,
                                      (asuint(time) / nrcTileSide) % nrcTileSide);
    bool nrcIsTraining, nrcIsUnbiased;
    uint nrcPathClass = NrcClassifyPixel(pixel, asuint(time), nrcTileSide, nrcTileOffset,
                                          nrcIsTraining, nrcIsUnbiased);
    nrcIsTraining = nrcIsTraining && kNrcTrainOn;
    if (!nrcIsTraining) nrcPathClass = NRC_CLASS_RENDER;

    {
        const uint nrcSortKey =
            (nrcPathClass << 6) |
            (((pixel.y >> 4) & 0x7u) << 3) |
            ((pixel.x >> 4) & 0x7u);
        dx::MaybeReorderThread(nrcSortKey, 8);
    }

    NrcClearPendingGI(pixelIdx);

    //render and train-biased are cache-eligible, train-unbiased runs to natural term
    const bool  nrcCacheEligible = kNrcEnabled &&
        (nrcPathClass == NRC_CLASS_TRAIN_BIASED ||
         nrcPathClass == NRC_CLASS_RENDER);
    const uint  nrcInferenceCapacity = (IMG_W * IMG_H + 255u) & ~255u;
    float nrcA0 = 0.0f;
    float nrcA  = 0.0f;
    int   nrcHitIdx = 0;
    bool  nrcCacheTerminated = false;

    uint   nrcPathId     = NRC_INVALID_PATH;
    uint   nrcTrainVIdx  = 0u;
    uint   nrcTailKind   = NRC_TAIL_INVALID;
    uint   nrcTailRadPk  = 0u;
    uint   nrcTailInfSlot = NRC_INVALID_SLOT;
    float3 nrcLNeeAccum  = float3(0, 0, 0);
    //specular BSDF samples skip area-spread, deferred termination by one vertex
    bool   nrcPrevSpecular = false;
    if (nrcIsTraining)
    {
        nrcPathId = NrcAllocateTrainingPath();
        if (nrcPathId == NRC_INVALID_PATH) { nrcIsTraining = false; nrcPathClass = NRC_CLASS_RENDER; }
    }

    //slot 1, primary emitter/sky, slot 3, former DI scratch, now always zero
    gScratchPing[uint3(pixel, 1)] = float4(0, 0, 0, 0);
    gScratchPing[uint3(pixel, 3)] = float4(0, 0, 0, 0);

    storeReservoir(g_Reservoirs_current, pixelIdx, (Reservoir)0);

    //safe defaults, without this pass-through continues leave stale PS state in the slot
    init_ps(g_pathStateBuffer, pixelIdx);

    float wsum = 0.0f;

    uint   seed         = initRandomData(pixel, uint2(8, 4), time, 1u);
    float3 rayOrigin    = InitOrigin();
    float3 rayDir       = InitDirection(pixel, float2(imgSize), seed);
    uint   throughputPk = PackRGB9E5(float3(1, 1, 1));
    uint   prevNormalPk = PackNormal(float3(0, 1, 0));
    float  prev_pdf     = 1.0f;
    float  pdf_product  = 1.0f;

    //tpost, post-x2 integrand, reg-resident
    float3 tpost = float3(1, 1, 1);

    //dup-map discriminator for DI samples, unique per pixel+frame
    float3 diMarker;
    {
        uint h = (pixelIdx * 0x9E3779B9u) ^ (asuint(time) * 0x85EBCA6Bu);
        h ^= h >> 16; h *= 0xC2B2AE35u;
        h ^= h >> 13; h *= 0x27D4EB2Fu;
        h ^= h >> 16;
        const float u1 = (float)(h & 0xFFFFu) * (1.0f / 65535.0f);
        const float u2 = (float)((h >> 16) & 0xFFFFu) * (1.0f / 65535.0f);
        const float z  = 2.0f * u1 - 1.0f;
        const float r  = sqrt(max(0.0f, 1.0f - z * z));
        const float phi = 6.2831853f * u2;
        diMarker = float3(r * cos(phi), r * sin(phi), z);
    }

    //====================================
    //PATH LOOP
    //====================================
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
        //depth 0 4-state OMM, secondary force 2-state to skip alpha any-hit
        const uint rayFlags = (depth == 0) ? RAY_FLAG_NONE : RAY_FLAG_FORCE_OMM_2_STATE;
        dx::HitObject hitObj = TraceRay_Custom(SceneBVH, ray, rayFlags, 0xFF);

        //====================================
        //MISS
        //====================================
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

            //miss on training path, backward-fill tail
            if (nrcIsTraining) {
                nrcTailKind  = NRC_TAIL_MISS;
                nrcTailRadPk = PackRGB9E5(NrcCleanRadiance(envL));
            }

            const float3 F_contrib = throughput * envL * pdf_product;
            const float  p_hat     = GetPHat(F_contrib);
            const float  p_full    = pdf_product;
            const float  wi        = (p_full > 1e-20f) ? (p_hat / p_full) : 0.0f;

            if (depth == 1)
            {
                //DI env miss, x2 stored as direction, V2 is per-pixel dup discriminator
                AddInitialCandidate(wsum, g_Reservoirs_current, pixelIdx, wi,
                    rayDir, float3(0, 1, 0),
                    envL,   diMarker,
                    float2(0, 0),
                    MATID_ENV_MISS, MATID_ENV_MISS, 1.0f,
                    F_contrib, seed);
            }
            else
            {
                //GI env, x2 is stashed depth-1 vertex
                const PathVertexState ps = load_ps(g_pathStateBuffer, pixelIdx);
                AddInitialCandidate(wsum, g_Reservoirs_current, pixelIdx, wi,
                    ps.x2, ps.n2_s,
                    envL * tpost, ps.v2,
                    ps.uv,
                    ps.matID, ps.objID, ps.eta,
                    F_contrib, seed);
            }
            break;
        }

        //====================================
        //HIT SETUP
        //====================================
        const float  hitT   = hitObj.GetRayTCurrent();
        const float3 hitPos = rayOrigin + rayDir * hitT;

        const uint instID = hitObj.GetInstanceIndex();
        const uint primID = FlatPrimID(instID, hitObj.GetGeometryIndex(), hitObj.GetPrimitiveIndex());
        const uint matID  = GetMatIDFast(instID, primID);

        BuiltInTriangleIntersectionAttributes attr;
        hitObj.GetAttributes(attr);
        HitInfo hinfo = EvalSurfaceState(instID, primID, attr.barycentrics, rayOrigin, depth);

        //null-IOR boundary is pass-through
        const float matNi = LoadNi(matID);
        if (matNi <= 1.0f + EPSILON)
        {
            rayOrigin = hitPos;
            continue;
        }

        //flip IOR pair and set interior medium only for transmissive backfaces
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

        //====================================
        //DEPTH 0 PRIMARY HIT STORAGE
        //====================================
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

            //specular motion vector probe
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
                    //cap alpha-test iterations to avoid TDR on dense foliage
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

        //====================================
        //EMITTER HIT
        //====================================
        if (any(emission > 0.0f) && hinfo.lightID != 0xFFFFFFFFu)
        {
            const float3 throughput = UnpackRGB9E5(throughputPk);
            const float3 prevNormal = UnpackNormal(prevNormalPk);
            const float  lightPdfArea = LT_Pdf_LightTree_Area(rayOrigin, prevNormal, hinfo.lightID, instID);
            const float  cosLight     = max(dot(hinfo.hitNormal, -rayDir), 0.0f);
            const float  dist2        = max(hitT * hitT, EPSILON);
            const float  lightPdfSA   = (cosLight > EPSILON) ? (lightPdfArea * dist2 / cosLight) : 0.0f;
            const float  misWeight    = prev_pdf / max(prev_pdf + lightPdfSA, EPSILON);

            //tail is MIS_BSDF-weighted, matches prev vertex's MIS_NEE-weighted L_nee
            if (nrcIsTraining) {
                nrcTailKind  = NRC_TAIL_EMITTER;
                nrcTailRadPk = PackRGB9E5(NrcCleanRadiance(emission * misWeight));
            }

            const float3 F_contrib = throughput * emission * pdf_product;
            const float  p_hat     = GetPHat(F_contrib);
            const float  p_full    = pdf_product;
            const float  wi        = (p_full > 1e-20f) ? (misWeight * p_hat / p_full) : 0.0f;

            bool accepted = false;
            if (depth == 1)
            {
                //DI triangle emitter, V2 diMarker for dup discriminator
                accepted = AddInitialCandidate(wsum, g_Reservoirs_current, pixelIdx, wi,
                    hitPos, hinfo.hitNormal,
                    emission, diMarker,
                    float2(0, 0),
                    MATID_LIGHT_TRI, instID, 1.0f,
                    F_contrib, seed);
            }
            else
            {
                //GI emitter, x2 stashed depth-1 vertex
                const PathVertexState ps = load_ps(g_pathStateBuffer, pixelIdx);
                AddInitialCandidate(wsum, g_Reservoirs_current, pixelIdx, wi,
                    ps.x2, ps.n2_s,
                    emission * tpost, ps.v2,
                    ps.uv,
                    ps.matID, ps.objID, ps.eta,
                    F_contrib, seed);
            }
            break;
        }

        //====================================
        //STASH DEPTH-1 VERTEX AND V2
        //====================================
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

        //====================================
        //NRC AREA-SPREAD AND CACHE TERMINATION
        //====================================
        //Bekaert area spread, gated at hitIdx>=3 so ReSTIR PT x2 always has real BSDF
        //specular bounces skipped via nrcPrevSpecular, class-2 runs to natural term
        ++nrcHitIdx;
        //reset per-vertex NEE accumulator, NEE blocks below add into it
        nrcLNeeAccum = float3(0, 0, 0);

        if (nrcHitIdx == 1)
        {
            const float cosPrimary = max(abs(dot(rayDir, hinfo.hitNormal)), 1e-6f);
            nrcA0 = NrcComputeA0(hitT, cosPrimary);
        }
        else
        {
            if (!nrcPrevSpecular)
            {
                const float cosHit = max(abs(dot(-rayDir, hinfo.hitNormal)), 1e-6f);
                NrcAccumulateA(nrcA, hitT, prev_pdf, cosHit);
            }

            const bool shouldFire =
                !nrcPrevSpecular &&
                NrcShouldCacheTerminate(nrcHitIdx, nrcA0, nrcA, nrcCacheEligible, nrc_area_spread_c);

            if (shouldFire)
            {
                //slot alloc fail, continue normally instead of losing the pixel
                const float3 throughput = UnpackRGB9E5(throughputPk);
                const uint nrcSlot = NrcWriteTerminationRecord(
                        pixelIdx, nrcInferenceCapacity,
                        hitPos, -rayDir, hinfo.hitNormal,
                        hitLocalPr, hitLocalKd, hitLocalPm,
                        throughput, tpost, pdf_product);
                if (nrcSlot != NRC_INVALID_SLOT)
                {
                    nrcCacheTerminated = true;
                    //class-1 uses same query as rendering tail, self-training seed
                    //pack (alpha+beta) at cache-term vertex so fill kernel recovers radiance
                    if (nrcPathClass == NRC_CLASS_TRAIN_BIASED) {
                        nrcTailKind    = NRC_TAIL_CACHE;
                        nrcTailInfSlot = nrcSlot;
                        const float3 alphaCT = hitLocalKd * (1.0f - hitLocalPm);
                        const float3 betaCT  = lerp(float3(0.04f, 0.04f, 0.04f),
                                                    hitLocalKd, hitLocalPm);
                        nrcTailRadPk = PackRGB9E5(NrcCleanRadiance(alphaCT + betaCT));
                    }
                    break;
                }
            }
        }

        //====================================
        //NEE
        //====================================
        const bool performNEE = !(mediumMatID != MEDIUM_INVALID || LoadKd_w(matID) < EPSILON);

        uint matKdPk, matPrPmPk, hitNormalPk;
        if (performNEE)
        {
            matKdPk     = PackRGB9E5(hitLocalKd);
            matPrPmPk   = f32tof16_custom(hitLocalPr) | (f32tof16_custom(hitLocalPm) << 16u);
            hitNormalPk = PackNormal(hinfo.hitNormal);

            //====================================
            //POINT LIGHT NEE
            //====================================
            //visibility inline per candidate, handles sun infinite distance correctly
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
                        const float3 shadowOrigin = offset_ray(
                            hitPos,
                            dot(L, hinfo.hitNormal) >= 0.0f ? hinfo.hitNormal : -hinfo.hitNormal);
                        if (IsVisibleOffset(shadowOrigin, L, dist * 0.999f))
                        {
                            const float  misWeight       = lightPdf / (lightPdf + bsdfPdf);
                            const float3 localMeasurement = light.emission * bdataNEE.val * cosSurf;
                            const float3 F_contrib        = throughput * localMeasurement * pdf_product;
                            const float  p_hat            = GetPHat(F_contrib);
                            const float  p_full           = pdf_product * lightPdf;
                            const float  wi               = (p_full > 1e-20f) ? (misWeight * p_hat / p_full) : 0.0f;

                            //local NEE estimator, feeds backward fill target
                            if (nrcIsTraining)
                                nrcLNeeAccum += localMeasurement * misWeight / lightPdf;

                            if (depth == 0)
                            {
                                //DI NEE at primary, x2 is light
                                AddInitialCandidate(wsum, g_Reservoirs_current, pixelIdx, wi,
                                    light.position, light.normal,
                                    light.emission, diMarker,
                                    float2(0, 0),
                                    MATID_LIGHT_TRI, light.objID, 1.0f,
                                    F_contrib, seed);
                            }
                            else if (depth == 1)
                            {
                                //GI NEE at depth-1, x2 is current hit
                                AddInitialCandidate(wsum, g_Reservoirs_current, pixelIdx, wi,
                                    hitPos, hinfo.hitNormal,
                                    light.emission, -L,
                                    hinfo.uv,
                                    matID, instID, iors.y,
                                    F_contrib, seed);
                            }
                            else
                            {
                                //GI NEE past depth-1, add BSDF*cos to tpost
                                const PathVertexState ps = load_ps(g_pathStateBuffer, pixelIdx);
                                const float3 tpostNEE = tpost * bdataNEE.val * cosSurf;
                                AddInitialCandidate(wsum, g_Reservoirs_current, pixelIdx, wi,
                                    ps.x2, ps.n2_s,
                                    light.emission * tpostNEE, ps.v2,
                                    ps.uv,
                                    ps.matID, ps.objID, ps.eta,
                                    F_contrib, seed);
                            }
                        }
                    }
                }
            }

            //====================================
            //SUN NEE
            //====================================
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

                        const float3 shadowOrigin = offset_ray(
                            hitPos,
                            dot(sun.direction, hinfo.hitNormal) >= 0.0f ? hinfo.hitNormal : -hinfo.hitNormal);
                        if (IsVisibleOffset(shadowOrigin, sun.direction, 10000.0f))
                        {
                            const float wi = (p_full > 1e-20f) ? (misWeight * p_hat / p_full) : 0.0f;

                            if (nrcIsTraining)
                                nrcLNeeAccum += localMeasurement * misWeight / lightPdf;

                            if (depth == 0)
                            {
                                //DI sun NEE, x2 is direction, sentinel preserves direction under shift
                                AddInitialCandidate(wsum, g_Reservoirs_current, pixelIdx, wi,
                                    sun.direction, float3(0, 1, 0),
                                    sun.radiance, diMarker,
                                    float2(0, 0),
                                    MATID_ENV_MISS, MATID_ENV_MISS, 1.0f,
                                    F_contrib, seed);
                            }
                            else if (depth == 1)
                            {
                                AddInitialCandidate(wsum, g_Reservoirs_current, pixelIdx, wi,
                                    hitPos, hinfo.hitNormal,
                                    sun.radiance, -sun.direction,
                                    hinfo.uv,
                                    matID, instID, iors.y,
                                    F_contrib, seed);
                            }
                            else
                            {
                                const float3 tpostNEE = tpost * bdataNEE.val * NdotL;
                                const PathVertexState ps = load_ps(g_pathStateBuffer, pixelIdx);
                                AddInitialCandidate(wsum, g_Reservoirs_current, pixelIdx, wi,
                                    ps.x2, ps.n2_s,
                                    sun.radiance * tpostNEE, ps.v2,
                                    ps.uv,
                                    ps.matID, ps.objID, ps.eta,
                                    F_contrib, seed);
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

        //====================================
        //SAMPLE NEXT BSDF DIRECTION
        //====================================
        SamplingP sp = CalculateStrategyProbabilities(matID, -rayDir, hinfo.hitNormal, iors.x, iors.y, hitLocalKd, hitLocalPm);
        float3    s  = SampleBRDF(sp, matID, -rayDir, hinfo.hitNormal, hinfo.hitNormal, hitLocalKd, hitLocalPr, hitLocalPm, seed, iors.x, iors.y);
        BrdfData  bdata = EvaluateAndPdf_COMBINED(sp, matID, hinfo.hitNormal, hinfo.hitNormal, s, -rayDir, hitLocalKd, hitLocalPr, hitLocalPm, iors.x, iors.y);

        const float  cosTheta     = abs(dot(hinfo.hitNormal, s));
        const float3 updateWeight = (bdata.pdf > 1e-6f)
            ? (bdata.val * absorptionTint * cosTheta) / bdata.pdf
            : float3(0, 0, 0);

        //emit training vertex before bad-sample break, beta=0 terminates fill
        if (nrcIsTraining && nrcTrainVIdx < NRC_MAX_VERTICES_PER_PATH)
        {
            float features[16];
            const float3 alpha = hitLocalKd * (1.0f - hitLocalPm);
            const float3 betaC = lerp(float3(0.04f, 0.04f, 0.04f), hitLocalKd, hitLocalPm);
            NrcBuildFeatures(hitPos, -rayDir, hinfo.hitNormal,
                             hitLocalPr, alpha, betaC, features);

            NrcStoreTrainingVertex(
                nrcPathId, nrcTrainVIdx, features,
                PackRGB9E5(NrcCleanRadiance(nrcLNeeAccum)),
                PackRGB9E5(NrcCleanRadiance(updateWeight)));
            ++nrcTrainVIdx;
        }

        if (dot(s, s) < 1e-12f || bdata.pdf <= 1e-6f || any(isnan(updateWeight)) || any(isinf(updateWeight)))
            break;

        prev_pdf    = bdata.pdf;
        pdf_product = min(pdf_product * bdata.pdf, 1e30f);
        //matches BSDF sampler's SMOOTH_SPECULAR_THRESHOLD
        nrcPrevSpecular = (hitLocalPr < SMOOTH_SPECULAR_THRESHOLD);
        rayDir      = s;
        float3 offsetN = dot(s, hinfo.hitNormal) >= 0.0f ? hinfo.hitNormal : -hinfo.hitNormal;
        rayOrigin   = offset_ray(hitPos, offsetN);

        {
            float3 throughput  = UnpackRGB9E5(throughputPk) * updateWeight;
            float3 tpostWeight = bdata.val * absorptionTint * cosTheta;

            //RR applies to render AND training paths, survival boost patched into stored beta
            //tpost must NOT be boosted, it is reused in L2 and extra 1/p cannot be cancelled
            if (depth > 2)
            {
                const float survivalProb = max(min(1.0f, Luma(throughput)), 0.1f);
                if (RandomFloatSingle(seed) >= survivalProb) break;
                const float rrBoost = 1.0f / survivalProb;
                throughput  *= rrBoost;
                pdf_product = min(pdf_product * survivalProb, 1e30f);

                //patch just-emitted vertex beta so E[beta_stored*tail] is unbiased over RR
                if (nrcIsTraining && nrcTrainVIdx > 0u)
                {
                    NrcUpdateTrainingVertexBeta(
                        nrcPathId, nrcTrainVIdx - 1u,
                        PackRGB9E5(NrcCleanRadiance(updateWeight * rrBoost)));
                }
            }

            //tpost accumulates locally, no reservoir RMW
            if (depth >= 2)
                tpost *= tpostWeight;

            throughputPk = PackRGB9E5(throughput);
        }
        prevNormalPk = PackNormal(hinfo.hitNormal);
    }

    //====================================
    //NRC COMMIT TRAINING PATH META
    //====================================
    //numVertices==0 paths are skipped by the fill kernel
    //untagged long paths (degenerate BSDF, MAX_BOUNCES) get NRC_TAIL_RR, tail=0 is unbiased
    if (nrcPathId != NRC_INVALID_PATH)
    {
        const uint tailKind = (nrcTailKind != NRC_TAIL_INVALID) ? nrcTailKind : NRC_TAIL_RR;
        //kTailCache uses inferenceSlot, others use tailRadPk
        const uint infSlot = (tailKind == NRC_TAIL_CACHE) ? nrcTailInfSlot : NRC_INVALID_SLOT;
        NrcStorePathMeta(nrcPathId, nrcTrainVIdx, tailKind, infSlot, nrcTailRadPk);
    }

    //====================================
    //FINAL RESOLVE
    //====================================
    //cache-terminated pixels defer W/M/invalidation to Pass_nrc_resolve_v8
    store_wsum(g_Reservoirs_current, pixelIdx, wsum);
    if (!nrcCacheTerminated)
    {
        const float F_mag = GetPHat(load_F(g_Reservoirs_current, pixelIdx));
        float W = 0.0f;
        if (F_mag > 1e-6f && wsum > 0.0f)
        {
            W = wsum / F_mag;
            if (isnan(W) || isinf(W) || W < 0.0f) W = 0.0f;
        }

        store_W(g_Reservoirs_current, pixelIdx, W);
        store_M(g_Reservoirs_current, pixelIdx, 1u);

        if (W == 0.0f)
            InvalidateReservoir_ShadingNormal(g_Reservoirs_current, pixelIdx);
    }
}
