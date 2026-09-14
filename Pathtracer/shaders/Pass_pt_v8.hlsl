#define SPMIS_GRID_NONCOHERENT

#define SKYBAKE_CONSUMER 1
#ifndef PT_NO_DEBUG
#define PT_NO_DEBUG 0
#endif
#include "Includes_v8.hlsli"

#if PT_NO_DEBUG
#undef SHARC_DEBUG_MODE
#define SHARC_DEBUG_MODE 0u
#endif
#include "Raygen_Common_v8.hlsli"
#ifndef SHARC_UPDATE_PASS
#define SHARC_UPDATE_PASS 0
#endif
#ifndef PT_ENTRY_NAME
#define PT_ENTRY_NAME Pass_pt_v8
#endif
#if SHARC_UPDATE_PASS
#define SHARC_TRAINING_SPILL
#endif
#include "SharcPath_v8.hlsli"
#include "SharcGuide_v8.hlsli"
#if !SHARC_UPDATE_PASS
#include "SharcDebug_v8.hlsli"
#include "RestirLite_v8.hlsli"
#endif

// Update passes remap pixels into a strided training schedule.
uint2 PtPixel()
{
    uint2 pixel = DispatchRaysIndex().xy;
#if SHARC_UPDATE_PASS

    uint tileHash = Hash32(pixel.x ^ Hash32(pixel.y));
    uint phase = (sharc_frame + tileHash) % (sharc_updateStride * sharc_updateStride);
    pixel = pixel * sharc_updateStride + uint2(phase % sharc_updateStride, phase / sharc_updateStride);
#endif
    return pixel;
}

#define PtAccumulate(value) (total += (value))

// Diffuse cache hits reweight throughput before specular continuation.
bool PtPrepareVertex(HitContext ctx, float3 geometricNormal, float3 rayDir,
    bool sssEntered, uint pathSeed, uint depth, uint maxBounces, float pathSpread,
    inout SamplingP spPath, out bool cacheSurface, out bool diffuseCached, inout float3 throughput,
#if SHARC_UPDATE_PASS
    inout SharcTrainingState training, uint guideRoots
#else
    bool liteX2, bool litePath, inout float3 total, inout float3 liteL,
    inout half3 liteSuffix
#endif
)
{
    diffuseCached = false;

    cacheSurface = sharc_enabled != 0u && !sssEntered &&
        SharcMaterialEligible(ctx, spPath, geometricNormal);
    if (cacheSurface && (SHARC_UPDATE_PASS || depth > 1))
    {
        const SharcSurface surface = SharcMakeSurface(ctx, geometricNormal);
#if SHARC_UPDATE_PASS
        const float layerT = SharcLayerTransmission(spPath, ctx, -rayDir);
        const bool queryable = pathSpread > 0.0f && layerT > 1e-3f;
#else
        float layerT = 0.0f;
#endif
        uint sCache = RcBounceSeed(pathSeed, (uint)depth, 0x53484152u);
        float3 cached;
        bool hit = false;
#if SHARC_UPDATE_PASS

        [loop] for (uint attempt = 0u; attempt < 2u && !hit; ++attempt)
        {
            if (training.count == 0u) break;
            const bool forced = attempt != 0u;
            if (!forced)
            {
                if (!queryable || !SharcQueryFootprintAccepted(surface.position, pathSpread, sCache)) continue;
            }
            else if ((uint)depth + 1u < maxBounces) break;
            hit = SharcQueryStochastic(surface, forced, sCache, cached);
        }
        if (hit)
        {
            SharcTrainingRadiance(training, cached * layerT);

            GuideRootsAddSource(GuideLane(), guideRoots, cached * layerT);
        }

        else if ((uint)depth < maxBounces / 2u && layerT >= 0.2f)
            SharcTrainingVertex(training, surface, layerT,
                RcBounceSeed(pathSeed, (uint)depth, 0x53504c54u));
#else

        bool queryAccepted;
        if (liteX2)
        {

            const float ramp = SharcFootprintRamp(pathSpread, (uint)SharcLevel(ctx.hitPos));
            queryAccepted = pathSpread > 0.0f && ramp > 0.0f && RandomFloatSingle(sCache) < ramp;
        }
        else
            queryAccepted = pathSpread > 0.0f &&
                SharcQueryFootprintAccepted(surface.position, pathSpread, sCache);
        hit = queryAccepted && SharcQueryDraws(surface, sCache, cached);

        if (hit)
        {
            layerT = SharcLayerTransmission(spPath, ctx, -rayDir);
            hit = layerT > 1e-3f;
        }
        if (hit)
        {
            PtAccumulate(throughput * cached * layerT);
            if (litePath)
            {

                liteL += (float3)liteSuffix * cached * layerT;
                liteSuffix = (half3)0.0f;
                if (!any(throughput > 0.0f)) return false;
            }
        }
#endif
        if (hit)
        {

            diffuseCached = true;
            if (!DropBroadLobes(spPath, ctx.hitLocalPr)) return false;

            const float pSpecular = clamp(1.0f - layerT, 0.02f, 1.0f);
            if (RandomFloatSingle(sCache) >= pSpecular) return false;
            throughput *= rcp(pSpecular);
#if SHARC_UPDATE_PASS
            SharcTrainingScatter(training, rcp(pSpecular));
            GuideRootsScale(GuideLane(), guideRoots, rcp(pSpecular).xxx);
#endif
        }
    }
    return true;
}

#define PT_PS_DEPTH_SHIFT   0u
#define PT_PS_MIS_NONE      (1u << 7u)
#define PT_PS_DIFF_SHIFT    8u
#define PT_PS_SAMPLE_SHIFT  16u
#define PT_PS_GUIDE_SHIFT   20u

#define PT_REGULARIZE_ROUGHNESS ((float)((guide_params >> GUIDE_PARAM_REGULARIZE_SHIFT) & 63u) * 0.01f)
#define PT_PS_SSS           (1u << 24u)
#define PT_PS_LITE_VERTEX   (1u << 25u)
#define PT_PS_LITE_X2       (1u << 26u)
#define PT_PS_DIFF_CACHED   (1u << 27u)
#define PT_PS_FLIP_IOR      (1u << 28u)
#define PT_PS_SPREAD        (1u << 29u)
#define PT_PS_ROOT0         (1u << 30u)
#define PT_PS_ROOT1         (1u << 31u)
uint PtPsRoots(uint ps)     { return ps >> 30u; }

uint PtPsInit(uint s)       { return (s << PT_PS_SAMPLE_SHIFT) | (1u << PT_PS_DEPTH_SHIFT) | PT_PS_MIS_NONE; }
uint PtPsDepth(uint ps)     { return (ps >> PT_PS_DEPTH_SHIFT) & 0x7Fu; }
uint PtPsDiffDepth(uint ps) { return (ps >> PT_PS_DIFF_SHIFT) & 0x7Fu; }
uint PtPsSample(uint ps)    { return (ps >> PT_PS_SAMPLE_SHIFT) & 0xFu; }
uint PtPsGuideDepth(uint ps){ return (ps >> PT_PS_GUIDE_SHIFT) & 0xFu; }
uint PtPsWith(uint ps, uint flag, bool on) { return on ? (ps | flag) : (ps & ~flag); }

#define BN_DIMS_PER_VERTEX 16u
#define BN_DIM_STRATEGY    0u
#define BN_DIM_GUIDE_TEST  1u
#define BN_PAIR_COSINE     1u
#define BN_DIM_GUIDE_PICK  4u
#define BN_PAIR_GUIDE_CAP  3u
#define BN_PAIR_SUN        4u
#define BN1(dim)  BlueNoise(samplePixel, blueIndex, (dim) + BN_DIMS_PER_VERTEX * (depth - 1u))
#define BN2(pair) BlueNoise2(samplePixel, blueIndex, (pair) + (BN_DIMS_PER_VERTEX / 2u) * (depth - 1u))

// Each bounce combines cache, direct-light, and BSDF estimators.
[shader("raygeneration")]
void PT_ENTRY_NAME()
{
    const uint2 pixel = PtPixel();
    const uint2 imgSize = uint2(IMG_W, IMG_H);
    if (any(pixel >= imgSize)) return;
    const uint pixelIdx = MapPixelID(imgSize, pixel);

#if !SHARC_UPDATE_PASS
    if (SHARC_DEBUG_MODE != 0u)
    {
        float4 debugColor = float4(0.015f, 0.02f, 0.03f, 0.0f);
        if ((load_flagsWord(g_sample_current, pixelIdx) & SD_FLAG_NOBOUNCE) == 0u)
        {
            SDRecord primary = load_SD(g_sample_current, pixelIdx);
            float2 dIors; uint dMedium; float3 dAbsorb;
            load_rg_primaryExtra(g_pathStateBuffer, pixelIdx, dIors, dMedium, dAbsorb);
            HitContext dctx;
            dctx.hitPos         = primary.x1;
            dctx.hitNormal      = primary.n1_s;
            dctx.matID          = primary.matID;
            dctx.instID         = primary.instID;
            dctx.backface       = (primary.flags & SD_FLAG_BACKFACE) != 0u;
            dctx.hitLocalKd     = (half3)primary.Kd;
            dctx.hitLocalPr     = (half)primary.Pr;
            dctx.hitLocalPm     = (half)primary.Pm;
            dctx.iors           = (half2)dIors;
            dctx.mediumMatID    = dMedium;
            dctx.absorptionTint = (half3)dAbsorb;
            const float3 dGeometricNormal = gScratchPing[uint3(pixel, SHARC_DEBUG_SCRATCH)].xyz;
            const SamplingP dsp = CalculateStrategyProbabilities(dctx.matID, normalize(InitOrigin() - primary.x1),
                dctx.hitNormal, dctx.iors.x, dctx.iors.y, dctx.hitLocalKd, dctx.hitLocalPr, dctx.hitLocalPm);
            if (!SharcMaterialEligible(dctx, dsp, dGeometricNormal))
            {

                debugColor = float4(0.05f, 0.07f, 0.12f, 0.0f);
            }
            else if (SHARC_DEBUG_MODE == SHARC_DEBUG_GUIDING)
            {

                debugColor = GuideDebugColor(primary.x1, dGeometricNormal, dctx.hitNormal,
                    (sharc_enabled & SHARC_DEBUG_OTHER_LEVEL_BIT) != 0u);
            }
            else
            {
                SharcSurface surface = SharcMakeSurface(dctx, dGeometricNormal);
                float lod = SharcLevel(primary.x1);

                uint level = (uint)(lod + 0.5f);
                if ((sharc_enabled & SHARC_DEBUG_OTHER_LEVEL_BIT) != 0u)
                    level = frac(lod) >= 0.5f ? level - 1u : level + 1u;
                level = min(level, SHARC_MAX_LEVEL - 1u);
                debugColor = SharcDebugColor(surface, level, SHARC_DEBUG_MODE);
            }
        }

        gScratchPing[uint3(pixel, SHARC_DEBUG_SCRATCH)] = debugColor;
    }
#endif

    if (load_flagsWord(g_sample_current, pixelIdx) & SD_FLAG_NOBOUNCE) return;

    SetSkyObserver(InitOrigin() + sceneOriginWorld);

    float3 total = float3(0, 0, 0);

    const uint N = SHARC_UPDATE_PASS ? 1u : max(pt_initialSamples, 1u);

    const uint maxBounces = SHARC_UPDATE_PASS ? sharc_trainBounces : pt_maxBounces;

    uint ps = PtPsInit(0u);
    [loop]
    for (;;)
    {
        if (PtPsSample(ps) >= N) break;
        const uint  sampleIdx   = pixelIdx;
        const uint2 samplePixel = (uint2)UnmapPixelID(sampleIdx, imgSize);

        uint seed     = initRandomData(samplePixel, uint2(8, 4), time, PtPsSample(ps) + 1u);
        uint pathSeed = Hash32(seed ^ 0x9E3779B9u);
#if SHARC_UPDATE_PASS
        pathSeed = Hash32(pathSeed ^ sharc_frame ^ 0x53484152u);
        SharcTrainingState training;
        SharcTrainingInit(training, sampleIdx);
#endif

        const SDRecord sd = load_SD(g_sample_current, sampleIdx);
        float2 pIors; uint pMedium; float3 pAbsorb;
        load_rg_primaryExtra(g_pathStateBuffer, sampleIdx, pIors, pMedium, pAbsorb);

        HitContext ctx;
        ctx.hitPos         = sd.x1;
        ctx.hitNormal      = sd.n1_s;
        ctx.matID          = sd.matID;
        ctx.instID         = sd.instID;
        ctx.backface       = (sd.flags & SD_FLAG_BACKFACE) != 0u;
        ctx.hitLocalKd     = (half3)sd.Kd;
        ctx.hitLocalPr     = (half)sd.Pr;
        ctx.hitLocalPm     = (half)sd.Pm;
        ctx.iors           = (half2)pIors;
        ctx.mediumMatID    = pMedium;
        ctx.absorptionTint = (half3)pAbsorb;

        float3  throughput   = float3(1, 1, 1);
        uint    prevNormalPk = PackNormal(float3(0, 1, 0));
        float   prev_pdf     = 1.0f;
        uint    rayDirPk     = PackNormal(normalize(sd.x1 - InitOrigin()));
        float3 geometricNormal = sd.n1_s;
#if SHARC_UPDATE_PASS
        geometricNormal = gScratchPing[uint3(samplePixel, SHARC_DEBUG_SCRATCH)].xyz;
#else

        if (sharc_enabled != 0u && SHARC_DEBUG_MODE == 0u)
            geometricNormal = gScratchPing[uint3(samplePixel, SHARC_DEBUG_SCRATCH)].xyz;
#endif
        float pathSpread = 0.0f;
        g_regularizeRoughness = 0.0f;
#if !SHARC_UPDATE_PASS

        float3 liteL      = 0.0f;
        half3  liteSuffix = (half3)0.0f;
        float  liteRr     = 0.0f;
#endif

        [loop]
        for (;;)
        {
            const uint depth = PtPsDepth(ps);
            if (depth >= maxBounces) break;
            const uint s          = PtPsSample(ps);
            const bool sssEntered = (ps & PT_PS_SSS) != 0u;
            float3 rayDir = UnpackNormal(rayDirPk);

            uint sNee  = RcBounceSeed(pathSeed, (uint)depth, RC_STREAM_NEE);
            uint sBsdf = RcBounceSeed(pathSeed, (uint)depth, RC_STREAM_BSDF);
            uint sSss  = RcBounceSeed(pathSeed, (uint)depth, RC_STREAM_SSS);

            SamplingP spPath = CalculateStrategyProbabilities(ctx.matID, -rayDir, ctx.hitNormal, ctx.iors.x, ctx.iors.y, ctx.hitLocalKd, ctx.hitLocalPr, ctx.hitLocalPm);
#if !SHARC_UPDATE_PASS
            if (depth == 1u)
            {

                const bool primaryLite = LITE_ENABLED && !sssEntered &&
                    HasBroadShare(spPath, ctx.hitLocalPr, ctx.hitLocalPm) &&
                    !LoadIsSSS(ctx.matID) && ctx.mediumMatID == MEDIUM_INVALID &&
                    LoadKd_w(ctx.matID) >= EPSILON && !MaterialIsFreeBounce(ctx.matID);
                ps = PtPsWith(ps, PT_PS_LITE_VERTEX, primaryLite);
                if (LITE_ENABLED && s == 0u && !primaryLite) LiteMarkEmpty(sampleIdx);
            }

            const bool liteVertex = (ps & PT_PS_LITE_VERTEX) != 0u;
            const bool liteGen  = depth == 1u && liteVertex;
            const bool liteX2   = depth == 2u && liteVertex;
            const bool litePath = depth >= 2u && liteVertex;
#endif

            bool cacheSurface = sharc_enabled != 0u && !sssEntered &&
                SharcMaterialEligible(ctx, spPath, geometricNormal);
            bool diffuseCached = (ps & PT_PS_DIFF_CACHED) != 0u;
#if SHARC_UPDATE_PASS

            if (!PtPrepareVertex(ctx, geometricNormal, rayDir, sssEntered,
                pathSeed, depth, maxBounces, pathSpread, spPath, cacheSurface, diffuseCached,
                throughput, training, PtPsRoots(ps))) break;
#else
            if (diffuseCached) DropBroadLobes(spPath, ctx.hitLocalPr);
#endif

            const bool useLearnedLights=LTC_UseSurfaceLearning();

#if !SHARC_UPDATE_PASS

            LiteGen liteG = LiteGenEmpty();
            if (liteGen && s != 0u) liteG = LiteGenLoad(sampleIdx);
#endif

            bool enterSSS = false;
            if (!sssEntered && LoadIsSSS(ctx.matID))
            {
                const float fT     = 1.0f - FresnelDielectric(-rayDir, ctx.hitNormal, ctx.iors.x, ctx.iors.y).x;
                const float pEnter = saturate(LoadSSSWeight(ctx.matID) * fT);
                enterSSS = RandomFloatSingle(sSss) < pEnter;
            }

            const bool budgetBreak = !MaterialIsFreeBounce(ctx.matID) &&
                !SHARC_UPDATE_PASS && PtPsDiffDepth(ps) >= (uint)pt_maxDiffuseBounces;

            const bool scatterLive = !enterSSS && !budgetBreak;

            GuideSet guide = GuideEmpty();
#if SHARC_UPDATE_PASS
            uint guideEntry = GUIDE_INVALID;
            uint guideParent = GUIDE_INVALID;
            uint guideRoot = GUIDE_ROOTS;
#endif
            if (scatterLive && cacheSurface && GUIDE_ENABLED && HasBroadShare(spPath, ctx.hitLocalPr, ctx.hitLocalPm) &&
                1u + PtPsGuideDepth(ps) <= (uint)GUIDE_MAX_DEPTH)
            {
                uint sKey = RcBounceSeed(pathSeed, (uint)depth, 0x4b455953u);
                const GuideKey guideKey = GuideKeyOf(ctx.hitPos, geometricNormal, sKey);
#if SHARC_UPDATE_PASS
                guideEntry = GuideFindOrInsert(guideKey);
                {
                    const GuideKey parentKey = GuideParentKey(guideKey);
                    guideParent = GuideKeyMatches(parentKey, guideKey) ? GUIDE_INVALID : GuideFindOrInsert(parentKey);
                }
                if (GUIDE_TRAIN) guide = GuideBuildFrom(guideEntry, guideParent, ctx.hitPos, ctx.hitNormal, true);
#else
                guide = GuideBuildAt(guideKey, ctx.hitPos, ctx.hitNormal);
#endif
            }

            uint   sampledStrategy = 0u;
            float3 dir = 0.0f;
            const bool broadGGX = IsBroadGGX(ctx.hitLocalPr);
#if !SHARC_UPDATE_PASS

            const bool blue      = PtPsDiffDepth(ps) == 0u;
            const uint blueIndex = (uint)time * N + PtPsSample(ps);
#endif
            if (scatterLive)
            {
#if !SHARC_UPDATE_PASS
                if (blue)
                {
                    sampledStrategy = SelectSamplingStrategyFrom(spPath, BN1(BN_DIM_STRATEGY));
                    dir = sampledStrategy == 0u
                        ? SampleBRDF_LambertianFrom(ctx.hitNormal, ctx.hitNormal, BN2(BN_PAIR_COSINE))
                        : SampleBRDF_WithStrategy(sampledStrategy, ctx.matID, -rayDir, ctx.hitNormal, ctx.hitNormal, ctx.hitLocalKd, ctx.hitLocalPr, ctx.hitLocalPm, sBsdf, ctx.iors.x, ctx.iors.y, false);
                }
                else
#endif
                dir = SampleBRDF(spPath, ctx.matID, -rayDir, ctx.hitNormal, ctx.hitNormal, ctx.hitLocalKd, ctx.hitLocalPr, ctx.hitLocalPm, sBsdf, ctx.iors.x, ctx.iors.y, false, sampledStrategy);

                if (guide.q > 0.0f && (sampledStrategy == 0u || (sampledStrategy == 1u && broadGGX)))
                {
                    uint sGuide = RcBounceSeed(pathSeed, (uint)depth, GUIDE_STREAM);
                    float uTest = RandomFloatSingle(sGuide);
#if !SHARC_UPDATE_PASS
                    if (blue) uTest = BN1(BN_DIM_GUIDE_TEST);
#endif
                    if (uTest < guide.q)
                    {
                        uint pick;
#if !SHARC_UPDATE_PASS
                        if (blue) dir = GuideSample(guide, BN1(BN_DIM_GUIDE_PICK), BN2(BN_PAIR_GUIDE_CAP), pick);
                        else
#endif
                        dir = GuideSample(guide, sGuide, pick);
                    }
                }
            }

            BrdfData bdata = (BrdfData)0;
            float3 broadScatter = 0.0f; float broadScatterPdf = 0.0f;
            float  pdfTotal = 0.0f;

            const bool performNEE = ctx.mediumMatID == MEDIUM_INVALID &&
                (LoadKd_w(ctx.matID) >= EPSILON || (float)ctx.hitLocalPr >= SMOOTH_SPECULAR_THRESHOLD);
            float3 directSum = float3(0, 0, 0);
#if SHARC_UPDATE_PASS
            float3 directBroad = float3(0, 0, 0);
#endif

            {
                [loop]
                // Evaluate BSDF, mesh-light, and sun techniques under shared MIS.
                for (uint ev = 0u; ev < 3u; ++ev)
                {
                    if (ev == 0u ? !scatterLive : !performNEE) continue;
                    const uint tech = ev - 1u;
                    float3 L = 0.0f;
                    float3 visTarget = 0.0f;
                    float3 visTargetN = 0.0f;
                    float3 radiance = 0.0f;
                    float  lightPdf = 0.0f;
                    float  cosSurf = 0.0f;
                    uint2 trainingToken=0u;

                    float trainingReward = 0.0f;
                    bool sampled = false;

#if !SHARC_UPDATE_PASS

                    uint   liteObj   = LITE_INF;
                    float3 litePos   = 0.0f;
                    float3 liteNrm   = 0.0f;
                    float  liteDistL = RAY_TMAX_PLANET;
                    float  liteCosL  = 1.0f;
#endif
                    if (ev == 0u)
                    {

                    }
                    else if (tech == 0u)
                    {
                        if ((rs_flags & RS_FLAG_NO_MESH_LIGHTS) == 0u)
                        {

                            LT_Sample treeSample;
                            bool prefetched = false;
#if !SHARC_UPDATE_PASS
                            prefetched = depth == 1u && s == 0u &&
                                load_pt_neePrefetch(g_pathStateBuffer, sampleIdx, treeSample.id, treeSample.inst, treeSample.pdf, sNee, treeSample.learningToken);
#endif
                            if (!prefetched) treeSample = LT_SampleLight(ctx.hitPos, ctx.hitNormal, sNee, useLearnedLights);
                            trainingToken=treeSample.learningToken;
                            LT_LightSampleResult light = LT_SamplePointOnLightTree(ctx.hitPos, treeSample, sNee);

                            const float3 toLight = light.position - ctx.hitPos;
                            const float  dist    = sqrt(dot(toLight, toLight));
                            L = toLight / dist;

                            cosSurf = dot(ctx.hitNormal, L);
                            const float cosLightS = dot(light.normal, -L);
                            if (cosSurf > 1e-6f && cosLightS > 1e-6f && light.pdfSolidAngle > 1e-20f)
                            {
                                sampled    = true;
                                visTarget  = light.position;
                                visTargetN = light.normal;
                                radiance   = light.emission;
                                lightPdf   = light.pdfSolidAngle;
#if !SHARC_UPDATE_PASS
                                liteObj   = light.objID;
                                litePos   = light.position;
                                liteNrm   = light.normal;
                                liteDistL = dist;
                                liteCosL  = cosLightS;
#endif
                            }
                        }
                    }
                    else
                    {
                        float2 rSun = float2(RandomFloatSingle(sNee), RandomFloatSingle(sNee));
#if !SHARC_UPDATE_PASS
                        if (blue) rSun = BN2(BN_PAIR_SUN);
#endif
                        SunSampleResult sun = SampleSun(rSun, ctx.hitPos + sceneOriginWorld);
                        L = sun.direction;

                        cosSurf = dot(ctx.hitNormal, L);
                        if (cosSurf > 1e-6f && sun.pdf > 1e-20f)
                        {
                            sampled    = true;
                            visTarget  = ctx.hitPos + sun.direction * RAY_TMAX_PLANET;
                            visTargetN = -sun.direction;
                            radiance   = sun.radiance;
                            lightPdf   = sun.pdf;
                        }
                    }

                    float3 visT = 1.0f;
                    bool evaluate = ev == 0u;
                    if (ev == 0u) L = dir;
                    else if (sampled)
                    {

                        visT = VisibilityTransmittance(ctx.hitPos, ctx.hitNormal, visTarget, visTargetN);
                        evaluate = any(visT > 0.0f);
                    }
                    if (evaluate)
                    {

                        float3 broadNEE; float broadNeePdf;
                        BrdfData bdataNEE = EvaluateAndPdf_COMBINED_L(spPath, LOBE_BROAD, ctx.matID, ctx.hitNormal, ctx.hitNormal, L, -rayDir,
                            ctx.hitLocalKd, ctx.hitLocalPr, ctx.hitLocalPm, ctx.iors.x, ctx.iors.y, false, broadNEE, broadNeePdf);
                        if (ev == 0u)
                        {

                            bdata = bdataNEE; broadScatter = broadNEE; broadScatterPdf = broadNeePdf;
                            pdfTotal = GuideMixPdf(guide, spPath.Pdiff + (broadGGX ? spPath.Pspec : 0.0f),
                                broadScatterPdf, dir, bdata.pdf);
                        }
                        else if (bdataNEE.pdf > 0.0f)
                        {
                            trainingReward = dot(radiance*cosSurf*visT*LTC_TrainShare((float)ctx.hitLocalPr,ctx.matID,bdataNEE.val,broadNEE)/lightPdf,
                                float3(0.2126f,0.7152f,0.0722f));

                            const float  misWeight  = lightPdf / (lightPdf + bdataNEE.pdf);
                            const float3 lightScale = radiance * cosSurf * visT * (misWeight / lightPdf);
                            const float3 direct     = bdataNEE.val * lightScale;
#if !SHARC_UPDATE_PASS
                            if (liteGen)
                            {

                                PtAccumulate(throughput * (direct - broadNEE * lightScale));
                                uint sLite = RcBounceSeed(pathSeed, (uint)depth, 0x4c495445u + tech);
                                LiteSample cand;
                                LiteLink   link;
                                if (tech == 0u)
                                {
                                    cand = LiteSampleSurface(liteObj, litePos, liteNrm, radiance, LITE_KIND_LIGHT);
                                    link = LiteLinkFrom(liteDistL, cosSurf, liteCosL, false);
                                }
                                else
                                {
                                    cand = LiteSampleDirection(L, radiance);
                                    link = LiteLinkFrom(RAY_TMAX_PLANET, cosSurf, 1.0f, true);
                                }
                                LiteGenCandidate(liteG, (float3)ctx.hitLocalKd, cand, litePos, link, visT,
                                    misWeight, lightPdf, rcp((float)N), sLite);
                            }
                            else
                            {
                                PtAccumulate(throughput * direct);

                                if (litePath)
                                    liteL += (float3)liteSuffix * ((liteX2 && cacheSurface) ? broadNEE : bdataNEE.val) * lightScale;
                            }
#else
                            PtAccumulate(throughput * direct);
                            directSum += direct;
#endif
#if SHARC_UPDATE_PASS

                            if (training.fresh != SHARC_INVALID)
                                directBroad += broadNEE * lightScale;
#endif
                        }
                    }
                    if (ev != 0u) LT_TrainSample(trainingToken, trainingReward);
                }
#if SHARC_UPDATE_PASS
                if (performNEE) SharcTrainingRadianceSplit(training, directSum, directBroad);

                GuideRootsAddSource(GuideLane(), PtPsRoots(ps), directSum);
                if ((guideEntry != GUIDE_INVALID || guideParent != GUIDE_INVALID) && PtPsRoots(ps) != 3u &&
                    ps_guideRootBacked(GuideLane()))
                {
                    guideRoot = (ps & PT_PS_ROOT0) != 0u ? 1u : 0u;
                    GuideRootOpen(GuideLane(), guideRoot, guideEntry, guideParent, dir);
                    ps |= guideRoot != 0u ? PT_PS_ROOT1 : PT_PS_ROOT0;
                }
#endif
            }
#if !SHARC_UPDATE_PASS
            if (liteGen) LiteGenCommit(sampleIdx, liteG);
#endif

            if (enterSSS)
            {
                SSSWalkResult w = SubsurfaceWalk(ctx.hitPos, ctx.hitNormal, ctx.matID, sSss);
                if (!w.valid) break;

                const float3 surfKd = (float3)ctx.hitLocalKd;
                throughput *= w.wTotal * surfKd;
#if SHARC_UPDATE_PASS
                SharcTrainingAdvance(training, w.wTotal * surfKd, w.wTotal * surfKd, 1.0f);
                GuideRootsScale(GuideLane(), PtPsRoots(ps), w.wTotal * surfKd);
#else
                if (litePath) liteSuffix *= (half3)(w.wTotal * surfKd);
#endif
                ctx.hitPos         = w.exitPos;
                ctx.hitNormal      = w.exitNormal;
                ctx.hitLocalKd     = (half3)float3(1, 1, 1);
                ctx.hitLocalPr     = (half)1.0f;
                ctx.hitLocalPm     = (half)0.0f;
                ctx.iors           = (half2)float2(1.0f, 1.0f);
                ctx.mediumMatID    = MEDIUM_INVALID;
                ctx.absorptionTint = (half3)float3(1, 1, 1);
                rayDirPk           = PackNormal(-w.exitNormal);
                ps = (ps | PT_PS_SSS) & ~PT_PS_DIFF_CACHED;
                ps += 1u << PT_PS_DEPTH_SHIFT;
                if (PtPsGuideDepth(ps) < 15u) ps += 1u << PT_PS_GUIDE_SHIFT;
                continue;
            }

            if (budgetBreak) break;
            if (!MaterialIsFreeBounce(ctx.matID)) ps += 1u << PT_PS_DIFF_SHIFT;
            const float  cosTheta     = abs(dot(ctx.hitNormal, dir));
            const float3 updateWeight = (pdfTotal > 1e-6f)
                ? (bdata.val * (float3)ctx.absorptionTint * cosTheta / pdfTotal)
                : float3(0, 0, 0);
            if (dot(dir, dir) < 1e-12f || pdfTotal <= 1e-6f ||
                any(isnan(updateWeight)) || any(isinf(updateWeight)))
                break;

            if (!any(updateWeight > 0.0f))
                break;
#if !SHARC_UPDATE_PASS

            float3 liteBroad = 0.0f;
            if (liteGen)
                liteBroad = broadScatter * (float3)ctx.absorptionTint * cosTheta / pdfTotal;
#endif
#if SHARC_UPDATE_PASS

            const float3 broadWeight = broadScatter * (float3)ctx.absorptionTint * cosTheta / pdfTotal;
            const float3 updateWeightBroad = training.fresh != SHARC_INVALID ? broadWeight : updateWeight;
#endif

            const bool passThrough = sampledStrategy == 1u && dot(dir, ctx.hitNormal) < 0.0f &&
                LoadKd_w(ctx.matID) < 1.0f - EPSILON;
            if (passThrough) { }
            else if (performNEE)
            {
                prev_pdf = bdata.pdf;
                ps &= ~PT_PS_MIS_NONE;
            }
            else ps |= PT_PS_MIS_NONE;
            const bool spread = SharcScatterHasSpread(sampledStrategy, ctx.matID, ctx.hitLocalPr);
            ps = PtPsWith(ps, PT_PS_SPREAD, spread);
            if (spread && PtPsGuideDepth(ps) < 15u) ps += 1u << PT_PS_GUIDE_SHIFT;
            rayDir   = dir;
            rayDirPk = PackNormal(dir);

            // Offset the next ray toward its outgoing shading side.
            const float3 offsetN = (dot(dir, ctx.hitNormal) >= 0.0f) ? ctx.hitNormal : -ctx.hitNormal;
            const float3 rayOrigin = offset_ray(ctx.hitPos, offsetN);

            throughput *= updateWeight;
#if !SHARC_UPDATE_PASS

            if (litePath)
                liteSuffix *= (half3)((liteX2 && cacheSurface)
                    ? broadScatter * (float3)ctx.absorptionTint * cosTheta / pdfTotal
                    : updateWeight);
#endif

            float rrWeight = 1.0f;
            // Russian roulette scales surviving paths after the configured depth.
            if (depth >= (uint)(SHARC_UPDATE_PASS ? sharc_trainRrDepth : pt_rrStartDepth))
            {
                uint sRr = RcBounceSeed(pathSeed, (uint)depth, RC_STREAM_RR);
#if SHARC_UPDATE_PASS

                const float survivalProb = SharcTrainingSurvival(training, updateWeight);
#else

                const float survivalProb = max(min(1.0f, Luma(throughput) +
                    (litePath ? Luma((float3)liteSuffix) * liteRr : 0.0f)), 0.05f);
#endif
                if (RandomFloatSingle(sRr) >= survivalProb) break;
                throughput /= survivalProb;
                rrWeight = rcp(survivalProb);
#if !SHARC_UPDATE_PASS
                liteBroad /= survivalProb;
                if (litePath) liteSuffix = (half3)((float3)liteSuffix * rcp(survivalProb));
#endif
            }
#if SHARC_UPDATE_PASS
            SharcTrainingAdvance(training, updateWeight, updateWeightBroad, rrWeight);

            GuideRootsScale(GuideLane(), PtPsRoots(ps) & ~(guideRoot < GUIDE_ROOTS ? 1u << guideRoot : 0u),
                updateWeight * rrWeight);
            if (guideRoot < GUIDE_ROOTS)
                GuideRootSetWeight(GuideLane(), guideRoot,
                    broadWeight * (rrWeight * rcp(max(Luma((float3)ctx.hitLocalKd), 0.05f))),
                    broadScatterPdf / pdfTotal);
#endif

            prevNormalPk = PackNormal(ctx.hitNormal);
#if !SHARC_UPDATE_PASS

            if (liteGen) LiteParkStore(sampleIdx, liteBroad, pdfTotal);
#endif

            if (!IsRayValid(rayOrigin, rayDir, 10000.0f))
                break;

            RayDesc rayB;
            rayB.Origin    = rayOrigin;
            rayB.Direction = rayDir;
            rayB.TMin      = 0.00001f;
            rayB.TMax      = RAY_TMAX_PLANET;
            TracePayload payload = (TracePayload)0;
            dx::HitObject hitObj = dx::HitObject::TraceRay(SceneBVH, RAY_FLAG_FORCE_OMM_2_STATE,
                0xFF, 0, 1, 0, rayB, payload);
#if SHARC_UPDATE_PASS
            const uint traceSuffixHint = training.suffixLuma < 0.25f ? 1u : 0u;
#endif

            rayDir = UnpackNormal(rayDirPk);
            const bool missed = !hitObj.IsHit();

            bool    terminate = missed;
            float3  terminalL = 0.0f;
            float   hitT_n    = 0.0f;
            uint    instID_n  = 0u, primID_n = 0u, matID_n = 0u;
            float3  hitPos_n  = 0.0f;
            HitInfo hinfo_n   = (HitInfo)0;
#if !SHARC_UPDATE_PASS
            const uint litePx = sampleIdx;
            float3 liteParkBroad = 0.0f; float liteParkPdf = 0.0f;
            if (liteGen) LiteParkLoad(litePx, liteParkBroad, liteParkPdf);
            LiteSample cand = (LiteSample)0; LiteLink link = (LiteLink)0;
            float3 candPos = 0.0f; float candMis = 1.0f;
#endif
            if (missed)
            {

                const bool underground = WorldPosIsUnderground(ctx.hitPos + sceneOriginWorld);
                SetSkyObserver(cloudEnabled > .5f ? ctx.hitPos + sceneOriginWorld : InitOrigin() + sceneOriginWorld);

                const float  sunSAPdf   = underground ? 0.0f : GetSunPdf(rayDir);
                const float3 sunRad     = (sunSAPdf > 0.0f) ? EvaluateSun(rayDir) : float3(0, 0, 0);
                const float  sunMisBsdf = (sunSAPdf > 0.0f)
                    ? ((ps & PT_PS_MIS_NONE) != 0u ? 1.0f : prev_pdf / max(prev_pdf + sunSAPdf, EPSILON)) : 0.0f;
                const float3 sky = underground ? float3(0, 0, 0) : EvaluateSky(rayDir);
                const float3 envL = sky + sunRad * sunMisBsdf;
                terminalL = envL;
#if !SHARC_UPDATE_PASS

                const float cloudRoughness = sampledStrategy==0u || sampledStrategy==3u ? 1.0f
                    : sampledStrategy==2u ? (float)LoadPcr(ctx.matID) : (float)ctx.hitLocalPr;
                const uint cloudGuide = depth==1u && cloudRoughness<.25f && dot(ctx.hitNormal,rayDir)>0 ? 1u:0u;

                const float3 mainWeight = liteGen ? max(throughput-liteParkBroad,0.0f) : throughput;
                if(cloudEnabled>.5f) {
                    if(!underground) CumulusQueueMiss(samplePixel,ctx.hitPos,rayDir,mainWeight,
                        cloudGuide,RcBounceSeed(pathSeed,depth,0x434C4F55u),cloudRoughness);
                    PtAccumulate(mainWeight*sunRad*sunMisBsdf);
                } else PtAccumulate(mainWeight*envL);
                if (liteGen)
                {

                    const float liteCosX = max(dot(UnpackNormal(prevNormalPk), rayDir), 0.0f);
                    cand    = LiteSampleDirection(rayDir, sky + sunRad);
                    candPos = rayDir;
                    link    = LiteLinkFrom(RAY_TMAX_PLANET, liteCosX, 1.0f, true);
                    candMis = sunSAPdf > 0.0f ? sunMisBsdf : 1.0f;
                }
                else if (litePath) liteL += (float3)liteSuffix * envL;
#else
                PtAccumulate(throughput * envL);
                GuideRootsAddSource(GuideLane(), PtPsRoots(ps), envL);
#endif
            }
            else
            {

                hitT_n   = hitObj.GetRayTCurrent();
                instID_n = hitObj.GetInstanceID();
                primID_n = FlatPrimID(instID_n, hitObj.GetGeometryIndex(), hitObj.GetPrimitiveIndex());
                matID_n  = GetMatIDFast(instID_n, primID_n);

                float2 bary_n;
                {
                    BuiltInTriangleIntersectionAttributes attrB;
                    hitObj.GetAttributes(attrB);
                    bary_n = attrB.barycentrics;
                }
                hinfo_n  = EvalSurfaceStateDir(instID_n, primID_n, bary_n, rayDir, (uint)depth);
                hitPos_n = hinfo_n.hitPos;

                const float3 emission_n = (hinfo_n.lightID != 0xFFFFFFFFu)
                    ? g_EmissiveTriangles[hinfo_n.lightID].emission * GLOBAL_EMISSION_STRENGTH
                    : float3(0, 0, 0);
                if (any(emission_n > 0.0f))
                {
                    terminate = true;

                    const float3 prevNormalCur = ctx.hitNormal;
                    const bool   noPartner     = (ps & PT_PS_MIS_NONE) != 0u;
                    const float  lightPdfArea  = noPartner ? 0.0f
                        : LT_Pdf_LightTree_Area(ctx.hitPos, prevNormalCur, hinfo_n.lightID, instID_n, useLearnedLights);
                    const float  cosLight      = max(dot(hinfo_n.hitNormal, -rayDir), 0.0f);
                    const float  dist2         = max(hitT_n * hitT_n, EPSILON);
                    const float  lightPdfSA    = (cosLight > EPSILON) ? (lightPdfArea * dist2 / cosLight) : 0.0f;
                    const float  misWeight     = noPartner ? 1.0f : prev_pdf / max(prev_pdf + lightPdfSA, EPSILON);
                    terminalL = emission_n * misWeight;

#if !SHARC_UPDATE_PASS
                    if (liteGen)
                    {

                        PtAccumulate((throughput - liteParkBroad) * emission_n * misWeight);
                        const float  liteCosX = max(dot(prevNormalCur, rayDir), 0.0f);
                        const float3 liteNy   = dot(hinfo_n.geometricNormal, -rayDir) < 0.0f
                            ? -hinfo_n.geometricNormal : hinfo_n.geometricNormal;
                        cand    = LiteSampleSurface(instID_n, hitPos_n, liteNy, emission_n, LITE_KIND_LIGHT);
                        candPos = hitPos_n;
                        link    = LiteLinkFrom(hitT_n, liteCosX, dot(liteNy, -rayDir), false);
                        candMis = misWeight;
                    }
                    else
                    {
                        PtAccumulate(throughput * emission_n * misWeight);
                        if (litePath) liteL += (float3)liteSuffix * emission_n * misWeight;
                    }
#else
                    PtAccumulate(throughput * emission_n * misWeight);
                    GuideRootsAddSource(GuideLane(), PtPsRoots(ps), emission_n * misWeight);
#endif
                }
            }

            if (!terminate)
            {

                const float  matNi_n        = LoadNi(matID_n);
                const bool   transmissive_n = LoadKd_w(matID_n) < 1.0f - EPSILON;
                const bool   flipIOR_n      = hinfo_n.backface && transmissive_n && !LoadIsThinGlass(matID_n);
                const float2 iors_n         = flipIOR_n ? float2(matNi_n, 1.0f) : float2(1.0f, matNi_n);
                const uint   mediumMatID_n  = flipIOR_n ? matID_n : MEDIUM_INVALID;

                float3 hitLocalKd_n; float hitLocalPr_n, hitLocalPm_n;
                RefetchMaterial(matID_n, hinfo_n.uv, hitLocalKd_n, hitLocalPr_n, hitLocalPm_n, (uint)depth);

                const float3 absorptionTint_n = (mediumMatID_n != MEDIUM_INVALID)
                    ? CalculateAbsorptionThroughput(LoadTf(mediumMatID_n), hitT_n)
                    : float3(1, 1, 1);

#if !SHARC_UPDATE_PASS
                if (liteGen)
                {

                    const float3 liteNy = dot(hinfo_n.geometricNormal, -rayDir) < 0.0f
                        ? -hinfo_n.geometricNormal : hinfo_n.geometricNormal;
                    LiteParkPointStore(litePx, WorldToObjectPos(instID_n, hitPos_n), instID_n,
                        PackNormal(WorldToObjectNrm(instID_n, liteNy)), liteParkPdf);

                    liteRr     = Luma(liteParkBroad);
                    throughput = max(throughput - liteParkBroad, 0.0f);
                    liteSuffix = (half3)1.0f;
                    liteL      = 0.0f;
                    ps |= PT_PS_LITE_X2;
                }
#endif
                ctx.hitPos         = hitPos_n;
                geometricNormal    = hinfo_n.geometricNormal;

                if ((ps & PT_PS_SPREAD) != 0u)
                    pathSpread += hitT_n * sqrt(min(16.0f, rcp(max(prev_pdf *
                        abs(dot(geometricNormal, -rayDir)), 1e-6f))));
                ctx.hitNormal      = hinfo_n.hitNormal;
                ctx.matID          = matID_n;
                ctx.instID         = instID_n;
                ctx.backface       = hinfo_n.backface;
                ctx.hitLocalKd     = (half3)hitLocalKd_n;

                const float regularize = pathSpread > 0.0f ? PT_REGULARIZE_ROUGHNESS : 0.0f;
                g_regularizeRoughness = regularize;
                ctx.hitLocalPr     = (half) max(hitLocalPr_n, regularize);
                ctx.hitLocalPm     = (half) hitLocalPm_n;
                ctx.iors           = (half2)iors_n;
                ctx.mediumMatID    = mediumMatID_n;
                ctx.absorptionTint = (half3)absorptionTint_n;
                ps = PtPsWith(ps, PT_PS_FLIP_IOR, flipIOR_n);
            }

#if !SHARC_UPDATE_PASS
            if (terminate)
            {
                if (liteGen)
                {
                    uint sLite = RcBounceSeed(pathSeed, (uint)depth, 0x4c495447u);
                    LiteCandidate(litePx, load_kd(g_sample_current, litePx), cand, candPos, link,
                        (float3)1.0f, candMis, liteParkPdf, rcp((float)N), sLite);
                }
                break;
            }
#else

            if (terminate) SharcTrainingRadiance(training, terminalL);
            if (guideRoot < GUIDE_ROOTS)
                GuideRootAim(GuideLane(), guideRoot, !missed, hitPos_n, rayDir);
            if (terminate) break;
#endif

            const uint nextDepth = depth + 1u;
            if (nextDepth >= maxBounces) break;
#if !SHARC_UPDATE_PASS
            SamplingP spNext = CalculateStrategyProbabilities(ctx.matID, -rayDir, ctx.hitNormal,
                ctx.iors.x, ctx.iors.y, ctx.hitLocalKd, ctx.hitLocalPr, ctx.hitLocalPm);
            bool nextCacheSurface, nextDiffuseCached;
            if (!PtPrepareVertex(ctx, geometricNormal, rayDir, sssEntered,
                pathSeed, nextDepth, maxBounces, pathSpread, spNext, nextCacheSurface, nextDiffuseCached, throughput,
                nextDepth == 2u && liteVertex, liteVertex, total, liteL, liteSuffix)) break;
            ps = PtPsWith(ps, PT_PS_DIFF_CACHED, nextDiffuseCached);
#endif
            const uint hint = 0x40u | (ctx.instID & 0x3Fu);

            const uint hitNormalPk = PackNormal(ctx.hitNormal);
            const uint geoNormalPk = PackNormal(geometricNormal);
#if SHARC_UPDATE_PASS
            dx::MaybeReorderThread(hitObj, (hint << 1u) | traceSuffixHint, 8u);
#else
            dx::MaybeReorderThread(hitObj, hint, 7u);
#endif
            ctx.hitNormal   = UnpackNormal(hitNormalPk);
            geometricNormal = UnpackNormal(geoNormalPk);
            {
                const bool  enters = (ps & PT_PS_FLIP_IOR) != 0u;
                const float ni     = LoadNi(ctx.matID);
                ctx.iors           = (half2)(enters ? float2(ni, 1.0f) : float2(1.0f, ni));
                ctx.mediumMatID    = enters ? ctx.matID : MEDIUM_INVALID;
                ctx.absorptionTint = (half3)(enters
                    ? CalculateAbsorptionThroughput(LoadTf(ctx.matID), hitT_n) : float3(1, 1, 1));
            }
            ps += 1u << PT_PS_DEPTH_SHIFT;
        }
#if SHARC_UPDATE_PASS
        SharcTrainingCommit(training);

        GuideRootsFinalize(GuideLane(), PtPsRoots(ps), RcBounceSeed(pathSeed, 0u, GUIDE_STREAM));
#else

        if ((ps & PT_PS_LITE_X2) != 0u)
        {
            uint sLite = RcBounceSeed(pathSeed, 2u, 0x4c495448u);
            LiteCandidatePoint(sampleIdx, liteL, rcp((float)N), sLite);
        }
#endif
        ps = PtPsInit(PtPsSample(ps) + 1u);
    }

    total /= (float)N;
    if (any(isnan(total)) || any(isinf(total))) total = float3(0, 0, 0);

#if !SHARC_UPDATE_PASS
    gScratchPing[uint3((uint2)UnmapPixelID(pixelIdx, imgSize), 2)] = float4(total, 0.0f);
#endif
}
