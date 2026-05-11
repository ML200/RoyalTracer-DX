#include "Includes_v8.hlsli"
#include "Nrc_v8.hlsli"

//safety net, real termination is RR at depth>2 plus NRC cache short circuit
#ifndef MAX_BOUNCES
#define MAX_BOUNCES 32
#endif

//====================================
//PACKED NRC MUTABLE STATE
//====================================
//bits 0..5  hitIdx          (0..63)
//bits 6..9  trainVIdx       (0..15, NRC_MAX_VERTICES_PER_PATH=8)
//bits 10..12 tailKind       (0..7,  5 values used)
//bit  13    prevSpecular
//bit  14    cacheTerminated
#define NRC_PK_HITIDX_MASK     0x3Fu
#define NRC_PK_TRAINVIDX_SHIFT 6u
#define NRC_PK_TRAINVIDX_MASK  0xFu
#define NRC_PK_TAILKIND_SHIFT  10u
#define NRC_PK_TAILKIND_MASK   0x7u
#define NRC_PK_PREVSPEC_BIT    (1u << 13)
#define NRC_PK_CACHETERM_BIT   (1u << 14)

inline uint pk_hit  (uint s)        { return s & NRC_PK_HITIDX_MASK; }
inline uint pk_vidx (uint s)        { return (s >> NRC_PK_TRAINVIDX_SHIFT) & NRC_PK_TRAINVIDX_MASK; }
inline uint pk_tail (uint s)        { return (s >> NRC_PK_TAILKIND_SHIFT)  & NRC_PK_TAILKIND_MASK; }
inline bool pk_prev (uint s)        { return (s & NRC_PK_PREVSPEC_BIT)   != 0u; }
inline bool pk_cterm(uint s)        { return (s & NRC_PK_CACHETERM_BIT)  != 0u; }
inline uint pk_set_hit  (uint s, uint v) { return (s & ~NRC_PK_HITIDX_MASK) | (v & NRC_PK_HITIDX_MASK); }
inline uint pk_set_vidx (uint s, uint v) { return (s & ~(NRC_PK_TRAINVIDX_MASK << NRC_PK_TRAINVIDX_SHIFT))
                                                   | ((v & NRC_PK_TRAINVIDX_MASK) << NRC_PK_TRAINVIDX_SHIFT); }
inline uint pk_set_tail (uint s, uint v) { return (s & ~(NRC_PK_TAILKIND_MASK  << NRC_PK_TAILKIND_SHIFT))
                                                   | ((v & NRC_PK_TAILKIND_MASK)  << NRC_PK_TAILKIND_SHIFT); }
inline uint pk_set_prev (uint s, bool b) { return b ? (s | NRC_PK_PREVSPEC_BIT)  : (s & ~NRC_PK_PREVSPEC_BIT); }
inline uint pk_set_cterm(uint s)         { return s | NRC_PK_CACHETERM_BIT; }

//DI at d=2 and GI at d>=3 share one reservoir, cold state stashed in g_pathStateBuffer
[shader("raygeneration")]
void Pass_raygen_v8()
{
    const uint2 pixel    = DispatchRaysIndex().xy;
    const uint2 imgSize  = DispatchRaysDimensions().xy;
    const uint  pixelIdx = MapPixelID(imgSize, pixel);

    //class 0 render, 1 train biased, 2 train unbiased
    const bool kNrcEnabled    = NrcIsEnabled();
    const bool kNrcTrainOn    = NrcIsTrainOn();
    //x1 sharp reflection split needs NRC for the replacement radiance
    const bool kSharpRefl     = kNrcEnabled && NrcIsSharpReflectionsOn();
    //adaptive tile side from renderer feedback
    const uint  nrcTileSide   = NrcTrainingTileSide();
    const uint2 nrcTileOffset = uint2(asuint(time) % nrcTileSide,
                                      (asuint(time) / nrcTileSide) % nrcTileSide);
    bool nrcIsTraining, nrcIsUnbiased;
    uint nrcPathClass = NrcClassifyPixel(pixel, asuint(time), nrcTileSide, nrcTileOffset,
                                          nrcIsTraining, nrcIsUnbiased);
    nrcIsTraining = nrcIsTraining && kNrcTrainOn;
    if (!nrcIsTraining) nrcPathClass = NRC_CLASS_RENDER;

    NrcClearPendingGI(pixelIdx);

    //biased paths terminate into the cache, unbiased paths inject ground truth
    const bool  nrcCacheEligible = kNrcEnabled &&
        (nrcPathClass == NRC_CLASS_TRAIN_BIASED ||
         nrcPathClass == NRC_CLASS_RENDER);
    //dynamic cap from prior frame's counter, excess records fall through to MIS
    const uint  nrcInferenceCapacity = nrc_inference_capacity;

    //packed mutable NRC state: 8 small fields collapsed into one uint
    uint nrcStateA = 0u;

    float nrcA0 = 0.0f;
    float nrcA  = 0.0f;
    uint  nrcPathId      = NRC_INVALID_PATH;
    uint  nrcTailRadPk   = 0u;
    uint  nrcTailInfSlot = NRC_INVALID_SLOT;
    //NEE accumulator only lives within a single bounce iteration, never across the trace call
    float3 nrcLNeeAccum = float3(0, 0, 0);

    if (nrcIsTraining)
    {
        nrcPathId = NrcAllocateTrainingPath();
        if (nrcPathId == NRC_INVALID_PATH) { nrcIsTraining = false; nrcPathClass = NRC_CLASS_RENDER; }
    }

    //slot 1 primary emitter/sky, slot 3 unused
    gScratchPing[uint3(pixel, 1)] = float4(0, 0, 0, 0);
    gScratchPing[uint3(pixel, 3)] = float4(0, 0, 0, 0);
    //slot 7 sharp refl control rgb=Fresnel w=NRC slot, slot 8 raw env radiance
    gScratchPing[uint3(pixel, 7)] = float4(0, 0, 0, asfloat(NRC_INVALID_SLOT));
    gScratchPing[uint3(pixel, 8)] = float4(0, 0, 0, 0);

    storeReservoir(g_Reservoirs_current, pixelIdx, (Reservoir)0);

    //safe defaults, otherwise pass through continues leave stale state in the slot
    init_ps(g_pathStateBuffer, pixelIdx);

    float wsum = 0.0f;

    uint   seed         = initRandomData(pixel, uint2(8, 4), time, 1u);
    float3 rayOrigin    = InitOrigin();
    float3 rayDir       = InitDirection(pixel, float2(imgSize), seed);
    uint   throughputPk = PackRGB9E5(float3(1, 1, 1));
    uint   prevNormalPk = PackNormal(float3(0, 1, 0));
    float  prev_pdf     = 1.0f;
    float  pdf_product  = 1.0f;

    //tpost, post x2 integrand, register resident
    float3 tpost = float3(1, 1, 1);

    //dup map discriminator computed lazily at the (depth 0/1) DI write sites via diMarkerFor()

    bool primaryDone = false;

    //=====================================================================
    //PRIMARY (was depth==0)
    //null-IOR boundaries retry up to 8 times before giving up
    //=====================================================================
    [loop]
    for (uint primaryRetry = 0u; primaryRetry < 8u; ++primaryRetry)
    {
        if (!IsRayValid(rayOrigin, rayDir, 10000.0f))
        {
            primaryDone = true;
            break;
        }

        RayDesc ray;
        ray.Origin    = rayOrigin;
        ray.Direction = rayDir;
        ray.TMin      = 0.00001f;
        ray.TMax      = 100000.0f;
        //primary uses 4 state OMM
        dx::HitObject hitObj = TraceRay_Custom(SceneBVH, ray, RAY_FLAG_NONE, 0xFF);

        if (!hitObj.IsHit())
        {
            //primary miss, sky/sun
            float3 sun   = EvaluateSun(rayDir);
            float3 skyL1 = EvalMissState(rayDir, sun);
            if (length(sun) > 0.0f) skyL1 = sun;
            gScratchPing[uint3(pixel, 1)] = float4(skyL1, 0);
            gScratchPing[uint3(pixel, 2)] = float4(skyL1, 0);
            store_sky(g_sample_current, pixelIdx);
            primaryDone = true;
            break;
        }

        const float  hitT   = hitObj.GetRayTCurrent();
        const float3 hitPos = rayOrigin + rayDir * hitT;
        const uint   instID = hitObj.GetInstanceIndex();
        const uint   primID = FlatPrimID(instID, hitObj.GetGeometryIndex(), hitObj.GetPrimitiveIndex());
        const uint   matID  = GetMatIDFast(instID, primID);

        BuiltInTriangleIntersectionAttributes attr;
        hitObj.GetAttributes(attr);
        HitInfo hinfo = EvalSurfaceState(instID, primID, attr.barycentrics, rayOrigin, 0u);

        //null IOR boundary, push origin past the surface and retry.
        //gated on transmissive: tinyobj defaults mat.ior=1.0 when MTL lacks
        //'Ni', so an opaque OBJ material would otherwise hit this branch and
        //pass straight through, never writing the sample buffer. DLSS then
        //reprojects from stale instID/primID -> see-through with trails.
        const float matNi = LoadNi(matID);
        const bool transmissive = LoadKd_w(matID) < 1.0f - EPSILON;
        if (matNi <= 1.0f + EPSILON && transmissive)
        {
            const float3 offsetN = (dot(rayDir, hinfo.hitNormal) >= 0.0f)
                                   ? hinfo.hitNormal
                                   : -hinfo.hitNormal;
            rayOrigin = offset_ray(hitPos, offsetN);
            continue;
        }
        const bool flipIOR = hinfo.backface && transmissive;
        const float2 iors = flipIOR ? float2(matNi, 1.0f) : float2(1.0f, matNi);
        const uint   mediumMatID = flipIOR ? matID : MEDIUM_INVALID;

        float3 hitLocalKd; float hitLocalPr, hitLocalPm;
        RefetchMaterial(matID, hinfo.uv, hitLocalKd, hitLocalPr, hitLocalPm, 0u);

        const float3 absorptionTint = (mediumMatID != MEDIUM_INVALID)
            ? CalculateAbsorptionThroughput(LoadTf(mediumMatID), hitT)
            : float3(1, 1, 1);

        const float3 emission  = GetEmissionFast(instID, primID);
        const bool   isEmitter = any(emission > 0.0f);

        //sample buffers, drive DLSS reproject
        store_instID(g_sample_current, pixelIdx, instID);
        store_primID(g_sample_current, pixelIdx, primID, isEmitter, hinfo.backface);
        store_bary  (g_sample_current, pixelIdx, attr.barycentrics);
        store_n1_s_world(g_sample_current, pixelIdx, hinfo.hitNormal, instID);
        store_uv    (g_sample_current, pixelIdx, hinfo.uv);
        if (isEmitter) {
            gScratchPing[uint3(pixel, 1)] = float4(emission, 0);
            gScratchPing[uint3(pixel, 2)] = float4(emission, 0);
        }

        //=====================================================================
        //specular MV probe and sharp reflection NRC fire share one ray
        //heavy RayQuery state lives in this block only and dies at the closing }
        //=====================================================================
        bool nrcReflFallback = false;
        {
            const float3 reflDir    = reflect(rayDir, hinfo.hitNormal);
            const float3 reflOrigin = offset_ray(hitPos, hinfo.hitNormal);

            //gate split on roughness and kSharpRefl, keep delta lobes if NRC is off
            const bool dropGGX  = kSharpRefl && ShouldDropDeltaGGX (hitLocalPr, hitLocalPm);
            const bool dropCoat = kSharpRefl && ShouldDropDeltaCoat(matID);
            const bool splitRefl = (dropGGX || dropCoat);

            bool   committed = false;
            float  reflT     = 0.0f;
            uint   cmtInstID = 0u;
            uint   cmtGeomID = 0u;
            uint   cmtPrimID = 0u;
            float2 cmtBary   = float2(0, 0);
            if (IsRayValid(reflOrigin, reflDir, 10000.0f))
            {
                RayDesc reflRay;
                reflRay.Origin    = reflOrigin;
                reflRay.Direction = reflDir;
                reflRay.TMin      = 0.00001f;
                reflRay.TMax      = 100000.0f;

                RayQuery<RAY_FLAG_NONE> q;
                q.TraceRayInline(SceneBVH, RAY_FLAG_NONE, 0xFF, reflRay);
                //cap alpha test iterations to avoid TDR on dense foliage
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
                if (committed) {
                    reflT     = q.CommittedRayT();
                    cmtInstID = q.CommittedInstanceIndex();
                    cmtGeomID = q.CommittedGeometryIndex();
                    cmtPrimID = q.CommittedPrimitiveIndex();
                    cmtBary   = q.CommittedTriangleBarycentrics();
                }
            }

            if (committed)
            {
                const float3 reflPos    = reflOrigin + reflDir * reflT;
                const float3 virtualPos = reflPos - 2.0f * dot(reflPos - hitPos, hinfo.hitNormal) * hinfo.hitNormal;
                gScratchPing[uint3(pixel, 4)] = float4(virtualPos, asfloat(instID));
            }
            else
            {
                gScratchPing[uint3(pixel, 4)] = float4(0, 0, 0, asfloat(0xFFFFFFFFu));
            }

            //slot 7 weight, hit non emitter uses Fresnel*(alpha+beta), emitter and miss use Fresnel only
            if (splitRefl)
            {
                const float3 V_prim    = -rayDir;
                const float3 fresnelP  = ComputeSharpReflectionFresnel(
                                            matID, V_prim, hinfo.hitNormal,
                                            iors.x, iors.y, hitLocalPm,
                                            dropGGX, dropCoat);

                uint   reflSlot = NRC_INVALID_SLOT;
                float3 weight   = fresnelP;
                if (committed)
                {
                    const uint reflFlatPrim = FlatPrimID(cmtInstID, cmtGeomID, cmtPrimID);
                    const float3 reflEmission = GetEmissionFast(cmtInstID, reflFlatPrim);

                    if (any(reflEmission > 0.0f))
                    {
                        gScratchPing[uint3(pixel, 8)] =
                            float4(NrcCleanRadiance(reflEmission), 1.0f);
                    }
                    else
                    {
                        const uint reflMatID = GetMatIDFast(cmtInstID, reflFlatPrim);
                        HitInfo reflHit = EvalSurfaceState(
                            cmtInstID, reflFlatPrim, cmtBary, reflOrigin, 0u);

                        float3 reflKd; float reflPr, reflPm;
                        RefetchMaterial(reflMatID, reflHit.uv, reflKd, reflPr, reflPm, 0u);

                        const float3 reflAlpha = reflKd * (1.0f - reflPm);
                        const float3 reflBeta  = lerp(float3(0.04f, 0.04f, 0.04f), reflKd, reflPm);
                        const float3 reflSum   = reflAlpha + reflBeta;

                        const float reflAlphaL = GetPHat(reflAlpha);
                        const float reflBetaL  = GetPHat(reflBeta);
                        const float reflSpecW  = reflBetaL / (reflAlphaL + reflBetaL + EPSILON);
                        const float reflEffR   = lerp(1.0f, reflPr, reflSpecW);
                        const bool  reflInferEligible =
                            (reflEffR >= NRC_CACHE_ROUGHNESS_MIN) &&
                            (LoadKd_w(reflMatID) >= (1.0f - EPSILON));

                        if (reflInferEligible)
                        {
                            float features[17];
                            NrcBuildFeatures(reflHit.hitPos, -reflDir, reflHit.hitNormal,
                                             reflPr, reflAlpha, reflBeta, reflHit.backface, features);

                            reflSlot = NrcAppendInference(nrcInferenceCapacity, features);
                            weight   = fresnelP * reflSum;
                        }
                        else
                        {
                            nrcReflFallback = true;
                            weight = float3(0, 0, 0);
                        }
                    }
                }
                else
                {
                    const float3 sun  = EvaluateSun(reflDir);
                    float3       envL = EvalMissState(reflDir, sun);
                    if (length(sun) > 0.0f) envL = sun;
                    gScratchPing[uint3(pixel, 8)] = float4(NrcCleanRadiance(envL), 1.0f);
                }

                gScratchPing[uint3(pixel, 7)] = float4(weight, asfloat(reflSlot));
            }
        } //end sharp reflection probe scope

        //primary emitter directly hit, scratch slot 1 already holds the radiance, terminate
        if (isEmitter && hinfo.lightID != 0xFFFFFFFFu)
        {
            primaryDone = true;
            break;
        }

        //hit 1 area spread anchor
        nrcStateA = pk_set_hit(nrcStateA, 1u);
        const float cosPrimary = max(abs(dot(rayDir, hinfo.hitNormal)), 1e-6f);
        nrcA0 = NrcComputeA0(hitT, cosPrimary);
        nrcLNeeAccum = float3(0, 0, 0);

        //=====================================================================
        //NEE at primary, point light + sun techniques both run per vertex
        //=====================================================================
        const bool performNEE = !(mediumMatID != MEDIUM_INVALID || LoadKd_w(matID) < EPSILON);
        if (performNEE)
        {
            const uint matKdPk     = PackRGB9E5(hitLocalKd);
            const uint matPrPmPk   = f32tof16_custom(hitLocalPr) | (f32tof16_custom(hitLocalPm) << 16u);
            const uint hitNormalPk = PackNormal(hinfo.hitNormal);

            //point light NEE, visibility inline per candidate
            {
                LT_LightSampleResult light = LT_SamplePointOnLight(hitPos, hinfo.hitNormal, seed);

                const float3 toLight = light.position - hitPos;
                const float  distSq  = dot(toLight, toLight);
                const float  dist    = sqrt(distSq);
                const float3 L       = toLight / dist;

                const float cosSurf   = dot(hinfo.hitNormal, L);
                const float cosLightS = dot(light.normal, -L);

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
                            const float  misWeight        = lightPdf / (lightPdf + bsdfPdf);
                            const float3 localMeasurement = light.emission * bdataNEE.val * cosSurf;
                            const float3 F_contrib        = throughput * localMeasurement * pdf_product;
                            const float  p_hat            = GetPHat(F_contrib);
                            const float  p_full           = pdf_product * lightPdf;
                            const float  wi               = (p_full > 1e-20f) ? (misWeight * p_hat / p_full) : 0.0f;

                            if (nrcIsTraining)
                                nrcLNeeAccum += localMeasurement * misWeight / lightPdf;

                            //primary DI, x2 is the light
                            AddInitialCandidate(wsum, g_Reservoirs_current, pixelIdx, wi,
                                light.position, light.normal,
                                light.emission, diMarkerFor(pixelIdx, time),
                                float2(0, 0),
                                MATID_LIGHT_TRI, light.objID, 1.0f,
                                F_contrib, seed);
                        }
                    }
                }
            }

            //sun NEE
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
                        const float  misWeight        = lightPdf / (lightPdf + bsdfPdf);
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

                            AddInitialCandidate(wsum, g_Reservoirs_current, pixelIdx, wi,
                                sun.direction, float3(0, 1, 0),
                                sun.radiance, diMarkerFor(pixelIdx, time),
                                float2(0, 0),
                                MATID_ENV_MISS, MATID_ENV_MISS, 1.0f,
                                F_contrib, seed);
                        }
                    }
                }
            }
        }

        //=====================================================================
        //BSDF sample for the next bounce
        //=====================================================================
        SamplingP sp = CalculateStrategyProbabilities(matID, -rayDir, hinfo.hitNormal, iors.x, iors.y, hitLocalKd, hitLocalPm);
        bool ggxNoReflect = false;
        if (kSharpRefl && !nrcReflFallback)
        {
            const bool dropGGX0       = ShouldDropDeltaGGX (hitLocalPr, hitLocalPm);
            const bool dropCoat0      = ShouldDropDeltaCoat(matID);
            const bool transmissive_x1 = LoadKd_w(matID) < (1.0f - EPSILON);
            const bool spDropGGX      = dropGGX0 && !transmissive_x1;
            ggxNoReflect              = dropGGX0 &&  transmissive_x1;
            sp = DropDeltaLobes(sp, spDropGGX, dropCoat0);
        }

        const float3 s     = SampleBRDF(sp, matID, -rayDir, hinfo.hitNormal, hinfo.hitNormal, hitLocalKd, hitLocalPr, hitLocalPm, seed, iors.x, iors.y, ggxNoReflect);
        const BrdfData bdata = EvaluateAndPdf_COMBINED(sp, matID, hinfo.hitNormal, hinfo.hitNormal, s, -rayDir, hitLocalKd, hitLocalPr, hitLocalPm, iors.x, iors.y, ggxNoReflect);

        const float  cosTheta     = abs(dot(hinfo.hitNormal, s));
        const float3 updateWeight = (bdata.pdf > 1e-6f)
            ? (bdata.val * absorptionTint * cosTheta) / bdata.pdf
            : float3(0, 0, 0);

        //emit primary training vertex
        if (nrcIsTraining && pk_vidx(nrcStateA) < NRC_MAX_VERTICES_PER_PATH)
        {
            const uint vIdx = pk_vidx(nrcStateA);
            float features[17];
            const float3 alpha = hitLocalKd * (1.0f - hitLocalPm);
            const float3 betaC = lerp(float3(0.04f, 0.04f, 0.04f), hitLocalKd, hitLocalPm);
            NrcBuildFeatures(hitPos, -rayDir, hinfo.hitNormal,
                             hitLocalPr, alpha, betaC, hinfo.backface, features);

            NrcStoreTrainingVertex(
                nrcPathId, vIdx, features,
                PackRGB9E5(NrcCleanRadiance(nrcLNeeAccum)),
                PackRGB9E5(NrcCleanRadiance(updateWeight)));
            nrcStateA = pk_set_vidx(nrcStateA, vIdx + 1u);
        }

        if (dot(s, s) < 1e-12f || bdata.pdf <= 1e-6f || any(isnan(updateWeight)) || any(isinf(updateWeight)))
        {
            primaryDone = true;
            break;
        }

        prev_pdf    = bdata.pdf;
        pdf_product = min(pdf_product * bdata.pdf, 1e30f);
        nrcStateA   = pk_set_prev(nrcStateA, hitLocalPr < SMOOTH_SPECULAR_THRESHOLD);
        rayDir      = s;
        const float3 offsetNormPrim = (dot(s, hinfo.hitNormal) >= 0.0f) ? hinfo.hitNormal : -hinfo.hitNormal;
        rayOrigin   = offset_ray(hitPos, offsetNormPrim);

        //no RR at depth 0 (depth>1 only), no tpost update at depth 0 (depth>=2 only)
        const float3 throughputPrim = UnpackRGB9E5(throughputPk) * updateWeight;
        throughputPk = PackRGB9E5(throughputPrim);
        prevNormalPk = PackNormal(hinfo.hitNormal);

        break; //primary handled, exit retry loop
    }

    //=====================================================================
    //BOUNCE LOOP at depth >= 1
    //=====================================================================
    if (!primaryDone)
    {
        [loop]
        for (int depth = 1; depth < MAX_BOUNCES; ++depth)
        {
            if (!IsRayValid(rayOrigin, rayDir, 10000.0f))
                break;

            RayDesc ray;
            ray.Origin    = rayOrigin;
            ray.Direction = rayDir;
            ray.TMin      = 0.00001f;
            ray.TMax      = 100000.0f;
            //secondaries force 2 state to skip alpha any hit
            dx::HitObject hitObj = TraceRay_Custom(SceneBVH, ray, RAY_FLAG_FORCE_OMM_2_STATE, 0xFF);

            if (!hitObj.IsHit())
            {
                const float3 throughput = UnpackRGB9E5(throughputPk);

                //GI miss tail, sky scatter plus BSDF MIS sun disk
                const float  sunSAPdf   = GetSunPdf(rayDir);
                const float3 sunRad     = (sunSAPdf > 0.0f) ? EvaluateSun(rayDir)
                                                            : float3(0, 0, 0);
                const float  sunMisBsdf = (sunSAPdf > 0.0f)
                    ? prev_pdf / max(prev_pdf + sunSAPdf, EPSILON) : 0.0f;
                const float3 envL       = EvaluateSky(rayDir) + sunRad * sunMisBsdf;

                if (nrcIsTraining) {
                    nrcStateA   = pk_set_tail(nrcStateA, NRC_TAIL_MISS);
                    nrcTailRadPk = PackRGB9E5(NrcCleanRadiance(envL));
                }

                const float3 F_contrib = throughput * envL * pdf_product;
                const float  p_hat     = GetPHat(F_contrib);
                const float  p_full    = pdf_product;
                const float  wi        = (p_full > 1e-20f) ? (p_hat / p_full) : 0.0f;

                if (depth == 1)
                {
                    //DI env miss, x2 is direction
                    AddInitialCandidate(wsum, g_Reservoirs_current, pixelIdx, wi,
                        rayDir, float3(0, 1, 0),
                        envL,   diMarkerFor(pixelIdx, time),
                        float2(0, 0),
                        MATID_ENV_MISS, MATID_ENV_MISS, 1.0f,
                        F_contrib, seed);
                }
                else
                {
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

            const float  hitT   = hitObj.GetRayTCurrent();
            const float3 hitPos = rayOrigin + rayDir * hitT;

            const uint instID = hitObj.GetInstanceIndex();
            const uint primID = FlatPrimID(instID, hitObj.GetGeometryIndex(), hitObj.GetPrimitiveIndex());
            const uint matID  = GetMatIDFast(instID, primID);

            BuiltInTriangleIntersectionAttributes attr;
            hitObj.GetAttributes(attr);
            HitInfo hinfo = EvalSurfaceState(instID, primID, attr.barycentrics, rayOrigin, depth);

            //null IOR boundary only applies to actually transmissive surfaces
            //(see primary loop comment). Opaque Ni=1 materials must NOT pass through.
            const float matNi = LoadNi(matID);
            const bool transmissive = LoadKd_w(matID) < 1.0f - EPSILON;
            if (matNi <= 1.0f + EPSILON && transmissive)
            {
                const float3 offsetN = (dot(rayDir, hinfo.hitNormal) >= 0.0f)
                                       ? hinfo.hitNormal
                                       : -hinfo.hitNormal;
                rayOrigin = offset_ray(hitPos, offsetN);
                continue;
            }
            const bool flipIOR = hinfo.backface && transmissive;
            const float2 iors = flipIOR ? float2(matNi, 1.0f) : float2(1.0f, matNi);
            const uint   mediumMatID = flipIOR ? matID : MEDIUM_INVALID;

            float3 hitLocalKd; float hitLocalPr, hitLocalPm;
            RefetchMaterial(matID, hinfo.uv, hitLocalKd, hitLocalPr, hitLocalPm, (uint)depth);

            const float3 absorptionTint = (mediumMatID != MEDIUM_INVALID)
                ? CalculateAbsorptionThroughput(LoadTf(mediumMatID), hitT)
                : float3(1, 1, 1);

            const float3 emission = GetEmissionFast(instID, primID);

            //emitter direct hit
            if (any(emission > 0.0f) && hinfo.lightID != 0xFFFFFFFFu)
            {
                const float3 throughput = UnpackRGB9E5(throughputPk);
                const float3 prevNormal = UnpackNormal(prevNormalPk);
                const float  lightPdfArea = LT_Pdf_LightTree_Area(rayOrigin, prevNormal, hinfo.lightID, instID);
                const float  cosLight     = max(dot(hinfo.hitNormal, -rayDir), 0.0f);
                const float  dist2        = max(hitT * hitT, EPSILON);
                const float  lightPdfSA   = (cosLight > EPSILON) ? (lightPdfArea * dist2 / cosLight) : 0.0f;
                const float  misWeight    = prev_pdf / max(prev_pdf + lightPdfSA, EPSILON);

                if (nrcIsTraining) {
                    nrcStateA   = pk_set_tail(nrcStateA, NRC_TAIL_EMITTER);
                    nrcTailRadPk = PackRGB9E5(NrcCleanRadiance(emission * misWeight));
                }

                const float3 F_contrib = throughput * emission * pdf_product;
                const float  p_hat     = GetPHat(F_contrib);
                const float  p_full    = pdf_product;
                const float  wi        = (p_full > 1e-20f) ? (misWeight * p_hat / p_full) : 0.0f;

                if (depth == 1)
                {
                    AddInitialCandidate(wsum, g_Reservoirs_current, pixelIdx, wi,
                        hitPos, hinfo.hitNormal,
                        emission, diMarkerFor(pixelIdx, time),
                        float2(0, 0),
                        MATID_LIGHT_TRI, instID, 1.0f,
                        F_contrib, seed);
                }
                else
                {
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

            //++hitIdx, reset NEE accum
            const uint nrcHitIdx = pk_hit(nrcStateA) + 1u;
            nrcStateA = pk_set_hit(nrcStateA, nrcHitIdx);
            nrcLNeeAccum = float3(0, 0, 0);

            //primary already set hitIdx==1 and nrcA0; loop body always runs the >=2 path
            const bool prevSpec = pk_prev(nrcStateA);
            if (!prevSpec)
            {
                const float cosHit = max(abs(dot(-rayDir, hinfo.hitNormal)), 1e-6f);
                NrcAccumulateA(nrcA, hitT, prev_pdf, cosHit);
            }

            //multilobe roughness gate
            const float3 alphaG  = hitLocalKd * (1.0f - hitLocalPm);
            const float3 betaG   = lerp(float3(0.04f, 0.04f, 0.04f), hitLocalKd, hitLocalPm);
            const float  alphaL  = GetPHat(alphaG);
            const float  betaL   = GetPHat(betaG);
            const float  specW   = betaL / (alphaL + betaL + EPSILON);
            const float  effRough = lerp(1.0f, hitLocalPr, specW);

            const bool shouldFire =
                !prevSpec &&
                (effRough >= NRC_CACHE_ROUGHNESS_MIN) &&
                NrcShouldCacheTerminate(nrcHitIdx, nrcA0, nrcA, nrcCacheEligible, nrc_area_spread_c);

            if (shouldFire)
            {
                const float3 throughput = UnpackRGB9E5(throughputPk);
                const uint nrcSlot = NrcWriteTerminationRecord(
                        pixelIdx, nrcInferenceCapacity,
                        hitPos, -rayDir, hinfo.hitNormal,
                        hitLocalPr, hitLocalKd, hitLocalPm,
                        hinfo.backface,
                        throughput, tpost, pdf_product);
                if (nrcSlot != NRC_INVALID_SLOT)
                {
                    nrcStateA = pk_set_cterm(nrcStateA);
                    if (nrcPathClass == NRC_CLASS_TRAIN_BIASED) {
                        nrcStateA      = pk_set_tail(nrcStateA, NRC_TAIL_CACHE);
                        nrcTailInfSlot = nrcSlot;
                        const float3 alphaCT = hitLocalKd * (1.0f - hitLocalPm);
                        const float3 betaCT  = lerp(float3(0.04f, 0.04f, 0.04f), hitLocalKd, hitLocalPm);
                        nrcTailRadPk = PackRGB9E5(NrcCleanRadiance(alphaCT + betaCT));
                    }
                    break;
                }
            }

            //=====================================================================
            //NEE, point light + sun techniques both run per vertex
            //=====================================================================
            const bool performNEE = !(mediumMatID != MEDIUM_INVALID || LoadKd_w(matID) < EPSILON);
            if (performNEE)
            {
                const uint matKdPk     = PackRGB9E5(hitLocalKd);
                const uint matPrPmPk   = f32tof16_custom(hitLocalPr) | (f32tof16_custom(hitLocalPm) << 16u);
                const uint hitNormalPk = PackNormal(hinfo.hitNormal);

                //point light NEE, visibility inline per candidate
                {
                    LT_LightSampleResult light = LT_SamplePointOnLight(hitPos, hinfo.hitNormal, seed);

                    const float3 toLight = light.position - hitPos;
                    const float  distSq  = dot(toLight, toLight);
                    const float  dist    = sqrt(distSq);
                    const float3 L       = toLight / dist;

                    const float cosSurf   = dot(hinfo.hitNormal, L);
                    const float cosLightS = dot(light.normal, -L);

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
                                const float  misWeight        = lightPdf / (lightPdf + bsdfPdf);
                                const float3 localMeasurement = light.emission * bdataNEE.val * cosSurf;
                                const float3 F_contrib        = throughput * localMeasurement * pdf_product;
                                const float  p_hat            = GetPHat(F_contrib);
                                const float  p_full           = pdf_product * lightPdf;
                                const float  wi               = (p_full > 1e-20f) ? (misWeight * p_hat / p_full) : 0.0f;

                                if (nrcIsTraining)
                                    nrcLNeeAccum += localMeasurement * misWeight / lightPdf;

                                if (depth == 1)
                                {
                                    AddInitialCandidate(wsum, g_Reservoirs_current, pixelIdx, wi,
                                        hitPos, hinfo.hitNormal,
                                        light.emission, -L,
                                        hinfo.uv,
                                        matID, instID, iors.y,
                                        F_contrib, seed);
                                }
                                else
                                {
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

                //sun NEE
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
                            const float  misWeight        = lightPdf / (lightPdf + bsdfPdf);
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

                                if (depth == 1)
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

            //=====================================================================
            //BSDF sample for the next bounce (no x1 split, depth==0 hoisted)
            //=====================================================================
            SamplingP sp = CalculateStrategyProbabilities(matID, -rayDir, hinfo.hitNormal, iors.x, iors.y, hitLocalKd, hitLocalPm);
            const float3 s     = SampleBRDF(sp, matID, -rayDir, hinfo.hitNormal, hinfo.hitNormal, hitLocalKd, hitLocalPr, hitLocalPm, seed, iors.x, iors.y, false);
            const BrdfData bdata = EvaluateAndPdf_COMBINED(sp, matID, hinfo.hitNormal, hinfo.hitNormal, s, -rayDir, hitLocalKd, hitLocalPr, hitLocalPm, iors.x, iors.y, false);

            const float  cosTheta     = abs(dot(hinfo.hitNormal, s));
            const float3 updateWeight = (bdata.pdf > 1e-6f)
                ? (bdata.val * absorptionTint * cosTheta) / bdata.pdf
                : float3(0, 0, 0);

            //emit training vertex before any bad sample break, beta=0 terminates fill
            if (nrcIsTraining && pk_vidx(nrcStateA) < NRC_MAX_VERTICES_PER_PATH)
            {
                const uint vIdx = pk_vidx(nrcStateA);
                float features[17];
                const float3 alpha = hitLocalKd * (1.0f - hitLocalPm);
                const float3 betaC = lerp(float3(0.04f, 0.04f, 0.04f), hitLocalKd, hitLocalPm);
                NrcBuildFeatures(hitPos, -rayDir, hinfo.hitNormal,
                                 hitLocalPr, alpha, betaC, hinfo.backface, features);

                NrcStoreTrainingVertex(
                    nrcPathId, vIdx, features,
                    PackRGB9E5(NrcCleanRadiance(nrcLNeeAccum)),
                    PackRGB9E5(NrcCleanRadiance(updateWeight)));
                nrcStateA = pk_set_vidx(nrcStateA, vIdx + 1u);
            }

            if (dot(s, s) < 1e-12f || bdata.pdf <= 1e-6f || any(isnan(updateWeight)) || any(isinf(updateWeight)))
                break;

            prev_pdf    = bdata.pdf;
            pdf_product = min(pdf_product * bdata.pdf, 1e30f);
            nrcStateA   = pk_set_prev(nrcStateA, hitLocalPr < SMOOTH_SPECULAR_THRESHOLD);
            rayDir      = s;
            const float3 offsetN = (dot(s, hinfo.hitNormal) >= 0.0f) ? hinfo.hitNormal : -hinfo.hitNormal;
            rayOrigin   = offset_ray(hitPos, offsetN);

            {
                float3 throughput  = UnpackRGB9E5(throughputPk) * updateWeight;
                const float3 tpostWeight = bdata.val * absorptionTint * cosTheta;

                //RR applies to render and training, survival boost patched into stored beta only
                if (depth > 1)
                {
                    const float survivalProb = max(min(1.0f, Luma(throughput)), 0.1f);
                    if (RandomFloatSingle(seed) >= survivalProb) break;
                    const float rrBoost = 1.0f / survivalProb;
                    throughput  *= rrBoost;
                    pdf_product = min(pdf_product * survivalProb, 1e30f);

                    const uint vIdxAfter = pk_vidx(nrcStateA);
                    if (nrcIsTraining && vIdxAfter > 0u)
                    {
                        NrcUpdateTrainingVertexBeta(
                            nrcPathId, vIdxAfter - 1u,
                            PackRGB9E5(NrcCleanRadiance(updateWeight * rrBoost)));
                    }
                }

                if (depth >= 2)
                    tpost *= tpostWeight;

                throughputPk = PackRGB9E5(throughput);
            }
            prevNormalPk = PackNormal(hinfo.hitNormal);
        }
    }

    //=====================================================================
    //FINALIZE
    //=====================================================================
    const uint nrcTrainVIdxFinal     = pk_vidx(nrcStateA);
    const uint nrcTailKindFinal      = pk_tail(nrcStateA);
    const bool nrcCacheTerminatedFin = pk_cterm(nrcStateA);

    //numVertices=0 paths are skipped, untagged long paths get NRC_TAIL_RR, tail=0 is unbiased
    if (nrcPathId != NRC_INVALID_PATH)
    {
        const uint tailKind = (nrcTailKindFinal != NRC_TAIL_INVALID) ? nrcTailKindFinal : NRC_TAIL_RR;
        const uint infSlot  = (tailKind == NRC_TAIL_CACHE) ? nrcTailInfSlot : NRC_INVALID_SLOT;
        NrcStorePathMeta(nrcPathId, nrcTrainVIdxFinal, tailKind, infSlot, nrcTailRadPk);
    }

    //cache terminated pixels defer W/M/invalidation to Pass_nrc_resolve_v8
    store_wsum(g_Reservoirs_current, pixelIdx, wsum);
    if (!nrcCacheTerminatedFin)
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
