#include "IncludesTraining_v8.hlsli"
#include "SharcPath_v8.hlsli"
#include "SharcTraining_v8.hlsli"
#include "OceanVolume.hlsli"

// Cache training: one strided path per tile that deposits its radiance into the cache entries it
// touches and observes the guide lobes along the way. Stays a single reordered kernel; the path
// tracer proper lives in Pass_pt_trace_v8.hlsl.

// Update passes remap pixels into a strided training schedule.
uint2 PtPixel()
{
    uint2 pixel = DispatchRaysIndex().xy;

    uint tileHash = Hash32(pixel.x ^ Hash32(pixel.y));
    uint phase = (sharc_frame + tileHash) % (sharc_updateStride * sharc_updateStride);
    pixel = pixel * sharc_updateStride + uint2(phase % sharc_updateStride, phase / sharc_updateStride);
    return pixel;
}

uint GuideLane() { return DispatchRaysIndex().y * DispatchRaysDimensions().x + DispatchRaysIndex().x; }

// Diffuse cache hits reweight throughput before specular continuation. The cone of the lobes that
// led here decides whether the path may end in the cache, as in the path tracer (SharcConeRamp).
bool PtPrepareVertex(HitContext ctx, float3 geometricNormal, float3 rayDir,
    bool sssEntered, uint pathSeed, uint depth, uint maxBounces, float coneWidth, float coneAngle,
    inout SamplingP spPath, out bool cacheSurface, out bool diffuseCached, inout float3 throughput,
    inout SharcTrainingState training, uint guideRoots
)
{
    diffuseCached = false;

    cacheSurface = sharc_enabled != 0u && !sssEntered &&
        SharcMaterialEligible(ctx, spPath, geometricNormal);
    if (cacheSurface)
    {
        const SharcSurface surface = SharcMakeSurface(ctx, geometricNormal);
        const float layerT = SharcLayerTransmission(spPath, ctx, -rayDir);
        const bool queryable = layerT > 1e-3f;
        uint sCache = RcBounceSeed(pathSeed, (uint)depth, 0x53484152u);
        float3 cached;
        bool hit = false;

        [loop] for (uint attempt = 0u; attempt < 2u && !hit; ++attempt)
        {
            if (training.count == 0u) break;
            const bool forced = attempt != 0u;
            if (!forced)
            {
                if (!queryable || !SharcQueryFootprintAccepted(surface.position, coneWidth, coneAngle,
                    abs(dot(geometricNormal, rayDir)), sCache)) continue;
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
        if (hit)
        {

            diffuseCached = true;
            if (!DropBroadLobes(spPath, ctx.hitLocalPr)) return false;

            const float pSpecular = clamp(1.0f - layerT, 0.02f, 1.0f);
            if (RandomFloatSingle(sCache) >= pSpecular) return false;
            throughput *= rcp(pSpecular);
            SharcTrainingScatter(training, rcp(pSpecular));
            GuideRootsScale(GuideLane(), guideRoots, rcp(pSpecular).xxx);
        }
    }
    return true;
}

// Path sampling flags of the training path (bit-compatible with the path tracer's).
static const uint PT_PS_DEPTH_SHIFT   = 0u;
static const uint PT_PS_MIS_NONE      = 1u << 7u;
static const uint PT_PS_DIFF_SHIFT    = 8u;
static const uint PT_PS_SAMPLE_SHIFT  = 16u;
static const uint PT_PS_GUIDE_SHIFT   = 20u;
static const uint PT_PS_SSS           = 1u << 24u;
static const uint PT_PS_SSS_EXIT      = 1u << 25u;   // the current vertex is a subsurface exit
static const uint PT_PS_DIFF_CACHED   = 1u << 27u;
static const uint PT_PS_FLIP_IOR      = 1u << 28u;
static const uint PT_PS_SPREAD        = 1u << 29u;
static const uint PT_PS_ROOT0         = 1u << 30u;
static const uint PT_PS_ROOT1         = 1u << 31u;
#define PT_REGULARIZE_ROUGHNESS ((float)((guide_params >> GUIDE_PARAM_REGULARIZE_SHIFT) & 63u) * 0.01f)
uint PtPsRoots(uint ps)     { return ps >> 30u; }

uint PtPsInit(uint s)       { return (s << PT_PS_SAMPLE_SHIFT) | (1u << PT_PS_DEPTH_SHIFT) | PT_PS_MIS_NONE; }
uint PtPsDepth(uint ps)     { return (ps >> PT_PS_DEPTH_SHIFT) & 0x7Fu; }
uint PtPsDiffDepth(uint ps) { return (ps >> PT_PS_DIFF_SHIFT) & 0x7Fu; }
uint PtPsSample(uint ps)    { return (ps >> PT_PS_SAMPLE_SHIFT) & 0xFu; }
uint PtPsGuideDepth(uint ps){ return (ps >> PT_PS_GUIDE_SHIFT) & 0xFu; }
uint PtPsWith(uint ps, uint flag, bool on) { return on ? (ps | flag) : (ps & ~flag); }

// Reorder key of a vertex: its instance, and whether the path behind it still carries much.
uint SharcReorderHint(uint instID, float suffixLuma)
{
    return ((0x40u | (instID & 0x3Fu)) << 1u) | (suffixLuma < 0.25f ? 1u : 0u);
}

// Each bounce combines cache, direct-light, and BSDF estimators.
[shader("raygeneration")]
void Pass_sharc_update_v8()
{
    const uint2 pixel = PtPixel();
    const uint2 imgSize = uint2(IMG_W, IMG_H);
    if (any(pixel >= imgSize)) return;
    const uint pixelIdx = MapPixelID(imgSize, pixel);

    if (load_flagsWord(g_sample_current, pixelIdx) & SD_FLAG_NOBOUNCE) return;

    SetSkyObserver(InitOrigin() + sceneOriginWorld);

    const uint maxBounces = sharc_trainBounces;

    uint ps = PtPsInit(0u);
    const uint  sampleIdx   = pixelIdx;
    const uint2 samplePixel = (uint2)UnmapPixelID(sampleIdx, imgSize);

    uint seed     = initRandomData(samplePixel, uint2(8, 4), time, PtPsSample(ps) + 1u);
    uint pathSeed = Hash32(seed ^ 0x9E3779B9u);
    pathSeed = Hash32(pathSeed ^ sharc_frame ^ 0x53484152u);
    SharcTrainingState training;
    SharcTrainingInit(training, GuideLane());

    const SDRecord sd = load_SD(g_sample_current, sampleIdx);

    // The normals and the medium state of the context are rebuilt at the top of each iteration
    // (see the reorder there); the rest carries over.
    HitContext ctx = (HitContext)0;
    ctx.hitPos         = sd.x1;
    ctx.matID          = sd.matID;
    ctx.instID         = sd.instID;
    ctx.backface       = (sd.flags & SD_FLAG_BACKFACE) != 0u;
    ctx.hitLocalKd     = (half3)sd.Kd;
    ctx.hitLocalPr     = (half)sd.Pr;
    ctx.hitLocalPm     = (half)sd.Pm;

    float3  throughput   = float3(1, 1, 1);
    float   prev_pdf     = 1.0f;
    uint    rayDirPk     = PackNormal(normalize(sd.x1 - InitOrigin()));
    uint    hitNormalPk  = PackNormal(sd.n1_s);
    uint    geoNormalPk  = PackNormal(gScratchPing[uint3(samplePixel, SHARC_DEBUG_SCRATCH)].xyz);
    float3  geometricNormal = 0.0f;
    float   hitT = 0.0f;   // length of the segment that reached the current vertex
    uint    reorderHint = SharcReorderHint(sd.instID, training.suffixLuma);
    float pathSpread = 0.0f;
    float coneWidth  = 0.0f;   // cone of the path's lobes for the cache test: width here, full angle
    float coneAngle  = 0.0f;
    float pathDist   = length(sd.x1 - InitOrigin());   // one training path stands for a tile of pixels
    bool waterDirectSegment = false;
    bool waterMedium = (sd.flags & SD_FLAG_CAMERA_WATER) != 0u;
    bool waterScatterUsed = false;
    g_regularizeRoughness = 0.0f;

    [loop]
    for (;;)
    {
        // The one reorder point, at the top of every iteration, the primary one included (sorted
        // by the instance of the camera record), as in the path tracer: behind the early exits at
        // the end of an iteration and skipped by the subsurface continuation, the device hung at
        // random. The context crosses it compact: the normals packed, the medium state rebuilt
        // below from the camera record, the subsurface exit, or the side and length of the hit.
#if SHARC_SER_REORDER
        dx::MaybeReorderThread(reorderHint, 8u);
#endif
        const uint depth = PtPsDepth(ps);
        if (depth >= maxBounces) break;
        ctx.hitNormal   = UnpackNormal(hitNormalPk);
        geometricNormal = UnpackNormal(geoNormalPk);
        if (depth == 1u)
        {
            float2 pIors; uint pMedium; float3 pAbsorb;
            load_rg_primaryExtra(sampleIdx, pIors, pMedium, pAbsorb);
            ctx.iors           = (half2)pIors;
            ctx.mediumMatID    = pMedium;
            ctx.absorptionTint = (half3)pAbsorb;
        }
        else if ((ps & PT_PS_SSS_EXIT) != 0u)
        {
            ctx.iors           = (half2)float2(1.0f, 1.0f);
            ctx.mediumMatID    = MEDIUM_INVALID;
            ctx.absorptionTint = (half3)float3(1, 1, 1);
        }
        else
        {
            const bool  enters = (ps & PT_PS_FLIP_IOR) != 0u;
            const float ni     = LoadNi(ctx.matID);
            ctx.iors           = (half2)(enters ? float2(ni, 1.0f) : float2(1.0f, ni));
            ctx.mediumMatID    = enters ? ctx.matID : MEDIUM_INVALID;
            ctx.absorptionTint = (half3)(enters && !LoadIsOceanMaterial(ctx.matID)
                ? CalculateAbsorptionThroughput(LoadTf(ctx.matID), hitT) : float3(1, 1, 1));
        }
        ps &= ~PT_PS_SSS_EXIT;
        const bool sssEntered = (ps & PT_PS_SSS) != 0u;
        float3 rayDir = UnpackNormal(rayDirPk);

        uint sNee  = RcBounceSeed(pathSeed, (uint)depth, RC_STREAM_NEE);
        uint sBsdf = RcBounceSeed(pathSeed, (uint)depth, RC_STREAM_BSDF);
        uint sSss  = RcBounceSeed(pathSeed, (uint)depth, RC_STREAM_SSS);

        SamplingP spPath = CalculateStrategyProbabilities(ctx.matID, -rayDir, ctx.hitNormal, ctx.iors.x, ctx.iors.y, ctx.hitLocalKd, ctx.hitLocalPr, ctx.hitLocalPm);

        bool cacheSurface = sharc_enabled != 0u && !sssEntered &&
            SharcMaterialEligible(ctx, spPath, geometricNormal);
        bool diffuseCached = (ps & PT_PS_DIFF_CACHED) != 0u;

        if (!PtPrepareVertex(ctx, geometricNormal, rayDir, sssEntered,
            pathSeed, depth, maxBounces, coneWidth, coneAngle, spPath, cacheSurface, diffuseCached,
            throughput, training, PtPsRoots(ps))) break;

        const bool useLearnedLights=LTC_UseSurfaceLearning();

        bool enterSSS = false;
        if (!sssEntered && LoadIsSSS(ctx.matID))
        {
            const float fT     = 1.0f - FresnelDielectric(-rayDir, ctx.hitNormal, ctx.iors.x, ctx.iors.y).x;
            const float pEnter = saturate(LoadSSSWeight(ctx.matID) * fT);
            enterSSS = RandomFloatSingle(sSss) < pEnter;
        }
        const bool scatterLive = !enterSSS;

        const bool waterDirect = LoadIsOceanMaterial(ctx.matID) && ctx.mediumMatID == MEDIUM_INVALID;
        const bool performNEE = ctx.mediumMatID == MEDIUM_INVALID &&
            (LoadKd_w(ctx.matID) >= EPSILON || !GGXUsesDeltaSampling(ctx.matID, ctx.hitLocalPr));

        // --- the lobe of this sample: one group is picked and evaluated, for the light sample
        // and the scatter alike (see EvaluateLobe). A wide pick takes the light sample, divided by
        // the probability of the pick; the water surface takes one on every pick. ---
        const uint  strategy = SelectSamplingStrategy(spPath, sBsdf);
        const uint  group    = LobeGroupOf(strategy, ctx.hitLocalPr);
        const float groupP   = LobeGroupP(spPath, group, ctx.hitLocalPr);
        const bool  neeTaken = performNEE && (waterDirect || SharcScatterHasSpread(strategy, ctx.matID, ctx.hitLocalPr));

        // --- light sampling first: both shadow traversals run before any scatter state exists ---
        float3 directSum = float3(0, 0, 0);
        float3 directBroad = float3(0, 0, 0);
        if (neeTaken)
        {
            [loop]
            for (uint tech = 0u; tech < 2u; ++tech)
            {
                float3 L = 0.0f;
                float3 visTarget = 0.0f;
                float3 visTargetN = 0.0f;
                float3 radiance = 0.0f;
                float  lightPdf = 0.0f;
                float  cosSurf = 0.0f;
                uint2  trainingToken = 0u;
                float  trainingReward = 0.0f;
                bool   sampled = false;

                // Only the sun is worth widening the water's lobe for, and only a picked GGX lobe
                // widens; see PtInlineNee.
                const bool sunTech = tech == 1u;
                const half neePr = (waterDirect && sunTech && group == LOBE_GROUP_SPEC)
                    ? (half)OceanHighlightRoughness(ctx.hitLocalPr, LoadOceanSunLobeRoughness())
                    : ctx.hitLocalPr;

                if (tech == 0u)
                {
                    if ((rs_flags & RS_FLAG_NO_MESH_LIGHTS) == 0u && !waterDirect)
                    {
                        const LT_Sample treeSample = LT_SampleLight(ctx.hitPos, ctx.hitNormal, sNee, useLearnedLights);
                        trainingToken = treeSample.learningToken;
                        const LT_LightSampleResult light = LT_SamplePointOnLightTree(ctx.hitPos, treeSample, sNee);

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
                        }
                    }
                }
                else
                {
                    float2 rSun = float2(RandomFloatSingle(sNee), RandomFloatSingle(sNee));
                    const SunSampleResult sun = SampleSun(rSun, ctx.hitPos + sceneOriginWorld);
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

                if (sampled)
                {
                    const float3 visT = VisibilityTransmittance(ctx.hitPos, ctx.hitNormal, visTarget, visTargetN, false, !waterDirect);
                    if (any(visT > 0.0f))
                    {
                        const BrdfData lobe = EvaluateLobe(spPath, group, ctx.matID, ctx.hitNormal, ctx.hitNormal, L, -rayDir,
                            ctx.hitLocalKd, neePr, ctx.hitLocalPm, ctx.iors.x, ctx.iors.y);
                        if (lobe.pdf > 0.0f)
                        {
                            const float3 broad = group == LOBE_GROUP_BROAD ? lobe.val : 0.0f;
                            trainingReward = dot(radiance*cosSurf*visT*LTC_TrainShare((float)ctx.hitLocalPr,ctx.matID,lobe.val,broad)/lightPdf,
                                float3(0.2126f,0.7152f,0.0722f));

                            const float  misWeight  = waterDirect ? 1.0f : lightPdf / (lightPdf + lobe.pdf);
                            const float3 lightScale = radiance * cosSurf * visT * (misWeight / (lightPdf * groupP));
                            directSum += lobe.val * lightScale;

                            if (training.fresh != SHARC_INVALID)
                                directBroad += broad * lightScale;
                        }
                    }
                }
                LT_TrainSample(trainingToken, trainingReward);
            }
            SharcTrainingRadianceSplit(training, directSum, directBroad);
        }
        GuideRootsAddSource(GuideLane(), PtPsRoots(ps), directSum);

        // --- the scatter: one sample of the picked lobe, guide cones for the broad group, one
        // evaluation of that group ---
        uint   guideEntry = GUIDE_INVALID;
        uint   guideParent = GUIDE_INVALID;
        uint   guideRoot = GUIDE_ROOTS;
        const uint sampledStrategy = strategy;
        float3 dir = 0.0f;
        BrdfData lobe = (BrdfData)0;
        float  pdfTotal = 0.0f;
        if (scatterLive)
        {
            dir = SampleBRDF_WithStrategy(strategy, ctx.matID, -rayDir, ctx.hitNormal, ctx.hitNormal, ctx.hitLocalKd,
                ctx.hitLocalPr, ctx.hitLocalPm, sBsdf, ctx.iors.x, ctx.iors.y, false);

            float guideQ = 0.0f, guidePdf = 0.0f;
            if (group == LOBE_GROUP_BROAD && cacheSurface && GUIDE_ENABLED &&
                HasBroadShare(spPath, ctx.hitLocalPr, ctx.hitLocalPm) && 1u + PtPsGuideDepth(ps) <= (uint)GUIDE_MAX_DEPTH)
            {
                uint sKey = RcBounceSeed(pathSeed, (uint)depth, 0x4b455953u);
                const GuideKey guideKey = GuideKeyOf(ctx.hitPos, geometricNormal, sKey);
                guideEntry = GuideFindOrInsert(guideKey);
                {
                    const GuideKey parentKey = GuideParentKey(guideKey);
                    guideParent = GuideKeyMatches(parentKey, guideKey) ? GUIDE_INVALID : GuideFindOrInsert(parentKey);
                }
                if (GUIDE_TRAIN)
                {
                    // Built after the lobe sample: the cones overlap neither the sampler nor the evaluation.
                    const GuideSet guide = GuideBuildFrom(guideEntry, guideParent, ctx.hitPos, ctx.hitNormal, true);
                    if (guide.q > 0.0f)
                    {
                        guideQ = guide.q;
                        uint sGuide = RcBounceSeed(pathSeed, (uint)depth, GUIDE_STREAM);
                        float uTest = RandomFloatSingle(sGuide);
                        if (uTest < guide.q)
                        {
                            uint pick;
                            dir = GuideSample(guide, sGuide, pick);
                        }
                        guidePdf = GuidePdf(guide, dir);
                    }
                }
            }

            // The group's density, with the guide mixed in, times the probability of the pick.
            lobe = EvaluateLobe(spPath, group, ctx.matID, ctx.hitNormal, ctx.hitNormal, dir, -rayDir,
                ctx.hitLocalKd, ctx.hitLocalPr, ctx.hitLocalPm, ctx.iors.x, ctx.iors.y);
            pdfTotal = groupP * (guideQ > 0.0f ? max(lerp(lobe.pdf, guidePdf, guideQ), 0.0f) : lobe.pdf);
        }
        if ((guideEntry != GUIDE_INVALID || guideParent != GUIDE_INVALID) && PtPsRoots(ps) != 3u &&
            ps_guideRootBacked(GuideLane()))
        {
            guideRoot = (ps & PT_PS_ROOT0) != 0u ? 1u : 0u;
            GuideRootOpen(GuideLane(), guideRoot, guideEntry, guideParent, dir);
            ps |= guideRoot != 0u ? PT_PS_ROOT1 : PT_PS_ROOT0;
        }

        if (enterSSS)
        {
            SSSWalkResult w = SubsurfaceWalk(ctx.hitPos, ctx.hitNormal, ctx.matID, sSss);
            if (!w.valid) break;

            const float3 surfKd = (float3)ctx.hitLocalKd;
            throughput *= w.wTotal * surfKd;
            SharcTrainingAdvance(training, w.wTotal * surfKd, w.wTotal * surfKd, 1.0f);
            GuideRootsScale(GuideLane(), PtPsRoots(ps), w.wTotal * surfKd);
            ctx.hitPos         = w.exitPos;
            ctx.hitLocalKd     = (half3)float3(1, 1, 1);
            ctx.hitLocalPr     = (half)1.0f;
            ctx.hitLocalPm     = (half)0.0f;
            hitNormalPk        = PackNormal(w.exitNormal);   // unit IORs, no medium: PT_PS_SSS_EXIT
            rayDirPk           = PackNormal(-w.exitNormal);
            ps = (ps | PT_PS_SSS | PT_PS_SSS_EXIT) & ~PT_PS_DIFF_CACHED;
            waterDirectSegment = false;
            ps += 1u << PT_PS_DEPTH_SHIFT;
            if (PtPsGuideDepth(ps) < 15u) ps += 1u << PT_PS_GUIDE_SHIFT;
            reorderHint = SharcReorderHint(ctx.instID, training.suffixLuma);
            continue;
        }

        if (!MaterialIsFreeBounce(ctx.matID)) ps += 1u << PT_PS_DIFF_SHIFT;
        const float  cosTheta     = abs(dot(ctx.hitNormal, dir));
        const float3 updateWeight = (pdfTotal > 1e-6f)
            ? (lobe.val * (float3)ctx.absorptionTint * cosTheta / pdfTotal)
            : float3(0, 0, 0);
        if (dot(dir, dir) < 1e-12f || pdfTotal <= 1e-6f ||
            any(isnan(updateWeight)) || any(isinf(updateWeight)))
            break;

        if (!any(updateWeight > 0.0f))
            break;

        // The broad part of this scatter: all of a broad pick, none of another.
        const float3 broadWeight = group == LOBE_GROUP_BROAD ? updateWeight : 0.0f;
        const float3 updateWeightBroad = training.fresh != SHARC_INVALID ? broadWeight : updateWeight;

        const bool passThrough = sampledStrategy == 1u && dot(dir, ctx.hitNormal) < 0.0f &&
            LoadKd_w(ctx.matID) < 1.0f - EPSILON;
        waterDirectSegment = OceanDirectLightingOwnsRay(waterDirect, dot(dir, ctx.hitNormal)) ||
            (waterDirectSegment && passThrough && LoadIsThinGlass(ctx.matID));
        const bool incomingWater = waterMedium;
        if (LoadIsOceanMaterial(ctx.matID) && !LoadIsThinGlass(ctx.matID))
            waterMedium = ctx.backface ? dot(dir,ctx.hitNormal) >= 0.0f : dot(dir,ctx.hitNormal) < 0.0f;
        waterScatterUsed = OceanScatterUsedAfterSurface(waterScatterUsed,incomingWater,waterMedium,
            LoadIsOceanMaterial(ctx.matID),passThrough && LoadIsThinGlass(ctx.matID));
        if (passThrough) { }
        else if (neeTaken)
        {
            prev_pdf = lobe.pdf;
            ps &= ~PT_PS_MIS_NONE;
        }
        else ps |= PT_PS_MIS_NONE;
        const bool spread = SharcScatterHasSpread(sampledStrategy, ctx.matID, ctx.hitLocalPr);
        ps = PtPsWith(ps, PT_PS_SPREAD, spread);
        coneAngle += SharcLobeConeAngle(sampledStrategy, ctx.matID, ctx.hitLocalPr);
        if (spread && PtPsGuideDepth(ps) < 15u) ps += 1u << PT_PS_GUIDE_SHIFT;
        rayDir   = dir;
        rayDirPk = PackNormal(dir);

        // Offset the next ray toward its outgoing shading side.
        const float3 offsetN = (dot(dir, ctx.hitNormal) >= 0.0f) ? ctx.hitNormal : -ctx.hitNormal;
        const float3 rayOrigin = offset_ray(ctx.hitPos, offsetN);

        throughput *= updateWeight;

        float rrWeight = 1.0f;
        // Russian roulette scales surviving paths after the configured depth.
        if (depth >= (uint)sharc_trainRrDepth)
        {
            uint sRr = RcBounceSeed(pathSeed, (uint)depth, RC_STREAM_RR);

            const float survivalProb = SharcTrainingSurvival(training, updateWeight);
            if (RandomFloatSingle(sRr) >= survivalProb) break;
            throughput /= survivalProb;
            rrWeight = rcp(survivalProb);
        }
        SharcTrainingAdvance(training, updateWeight, updateWeightBroad, rrWeight);

        GuideRootsScale(GuideLane(), PtPsRoots(ps) & ~(guideRoot < GUIDE_ROOTS ? 1u << guideRoot : 0u),
            updateWeight * rrWeight);
        if (guideRoot < GUIDE_ROOTS)
            GuideRootSetWeight(GuideLane(), guideRoot,
                broadWeight * (rrWeight * rcp(max(Luma((float3)ctx.hitLocalKd), 0.05f))),
                lobe.pdf / pdfTotal);

        RayDesc rayB;
        rayB.Origin    = rayOrigin;
        rayB.Direction = rayDir;
        rayB.TMin      = 0.00001f;
        rayB.TMax      = RAY_TMAX_PLANET;
        if (!IsRayDescValid(rayB))
            break;
        OceanPathHit hit = OceanTracePathHit(rayB);

        bool volumeRedirected = false;
        bool waterAbsorbed = false;
        if (waterMedium) {
            uint sVolume = RcBounceSeed(pathSeed, depth, 0x57415452u);
            // At most one new event, then extinction to its actual endpoint.
            [loop] for (uint leg=0u; leg<2u; ++leg) {
                const OceanFlight flight = OceanSampleWaterSegment(rayB.Origin,rayB.Direction,
                    hit.distance,waterScatterUsed,sVolume);
                SharcTrainingScatter(training,flight.weight);
                GuideRootsScale(GuideLane(),PtPsRoots(ps),flight.weight);
                throughput *= flight.weight;
                training.suffixLuma *= Luma(flight.weight);
                if (!any(flight.weight > 0.0f)) { waterAbsorbed = true; break; }
                if (!flight.scattered) break;
                const float3 scatterPos = rayB.Origin+rayB.Direction*flight.distance;
                if (guideRoot < GUIDE_ROOTS) GuideRootAim(GuideLane(),guideRoot,true,scatterPos,rayDir);
                pathDist += flight.distance;
                rayDir = OceanSampleScatterDirection(rayB.Direction,sVolume);
                rayDirPk = PackNormal(rayDir);
                rayB.Origin = scatterPos; rayB.Direction = rayDir;
                rayB.TMin = 0.00001f; rayB.TMax = RAY_TMAX_PLANET;
                waterScatterUsed = true;
                volumeRedirected = true;
                waterDirectSegment = false;
                ps |= PT_PS_MIS_NONE;
                prev_pdf = 1.0f;
                hit = OceanTracePathHit(rayB);
            }
        }
        if (waterAbsorbed) break;
        rayDir = UnpackNormal(rayDirPk);
        const bool missed = !hit.hit;

        bool    terminate = missed;
        float3  terminalL = 0.0f;
        float   hitT_n    = 0.0f;
        uint    instID_n  = 0u, primID_n = 0u, matID_n = 0u;
        float3  hitPos_n  = 0.0f;
        HitInfo hinfo_n   = (HitInfo)0;
        if (missed)
        {
            SetSkyObserver((OCEAN_ENABLED ? rayB.Origin : InitOrigin()) + sceneOriginWorld);
            const bool underground = WorldPosIsUnderground(rayB.Origin + sceneOriginWorld);
            const float  sunSAPdf   = underground || waterDirectSegment ? 0.0f : GetSunPdf(rayDir);
            const float3 sunRad     = (sunSAPdf > 0.0f) ? EvaluateSun(rayDir) : float3(0, 0, 0);
            const float  sunMisBsdf = (sunSAPdf > 0.0f)
                ? ((ps & PT_PS_MIS_NONE) != 0u ? 1.0f : prev_pdf / max(prev_pdf + sunSAPdf, EPSILON)) : 0.0f;
            const float3 sky = underground ? float3(0, 0, 0) : EvaluateSky(rayDir);
            const float3 envL = sky + sunRad * sunMisBsdf;
            terminalL = envL;
            GuideRootsAddSource(GuideLane(), PtPsRoots(ps), envL);
        }
        else
        {
            hitT_n   = hit.distance;
            instID_n = hit.instance;
            primID_n = FlatPrimID(instID_n,hit.geometry,hit.primitive);
            matID_n  = GetMatIDFast(instID_n, primID_n);

            const float2 bary_n = hit.barycentrics;
            {
                const float spreadHere = (ps & PT_PS_SPREAD) != 0u
                    ? hitT_n * sqrt(min(16.0f, rcp(max(prev_pdf, 1e-6f)))) : 0.0f;
                const float beam = PixelConeAngle() * float(sharc_updateStride) * (pathDist + hitT_n) + pathSpread + spreadHere;
                hinfo_n = EvalSurfaceStateDir(instID_n, primID_n, bary_n, rayDir, beam);
                pathDist += hitT_n;
            }
            hitPos_n = hinfo_n.hitPos;
            matID_n = ResolveSurfaceMaterial(matID_n, hinfo_n);

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
                const bool neeOwned = waterDirectSegment && (rs_flags & RS_FLAG_NO_MESH_LIGHTS) == 0u;
                const float  misWeight     = neeOwned ? 0.0f : (noPartner ? 1.0f : prev_pdf / max(prev_pdf + lightPdfSA, EPSILON));
                terminalL = emission_n * misWeight;
                GuideRootsAddSource(GuideLane(), PtPsRoots(ps), emission_n * misWeight);
            }
        }

        if (!terminate)
        {
            // The side of the hit decides the medium state, which the next iteration rebuilds
            // from PT_PS_FLIP_IOR and the segment length.
            const bool   transmissive_n = LoadKd_w(matID_n) < 1.0f - EPSILON;
            const bool   flipIOR_n      = hinfo_n.backface && transmissive_n && !LoadIsThinGlass(matID_n);

            float3 hitLocalKd_n; float hitLocalPr_n, hitLocalPm_n;
            RefetchMaterial(matID_n, hinfo_n, hitLocalKd_n, hitLocalPr_n, hitLocalPm_n);

            ctx.hitPos         = hitPos_n;
            geometricNormal    = hinfo_n.geometricNormal;

            if ((ps & PT_PS_SPREAD) != 0u)
                pathSpread += hitT_n * sqrt(min(16.0f, rcp(max(prev_pdf *
                    abs(dot(geometricNormal, -rayDir)), 1e-6f))));
            coneWidth += hitT_n * coneAngle;
            ctx.hitNormal      = hinfo_n.hitNormal;
            ctx.matID          = matID_n;
            ctx.instID         = instID_n;
            ctx.backface       = hinfo_n.backface;
            ctx.hitLocalKd     = (half3)hitLocalKd_n;

            const float regularize = pathSpread > 0.0f ? PT_REGULARIZE_ROUGHNESS : 0.0f;
            g_regularizeRoughness = regularize;
            ctx.hitLocalPr     = (half) max(hitLocalPr_n, regularize);
            ctx.hitLocalPm     = (half) hitLocalPm_n;
            ps = PtPsWith(ps, PT_PS_FLIP_IOR, flipIOR_n);
        }

        if (terminate) SharcTrainingRadiance(training, terminalL);
        if (guideRoot < GUIDE_ROOTS && !volumeRedirected)
            GuideRootAim(GuideLane(), guideRoot, !missed, hitPos_n, rayDir);
        if (terminate) break;

        const uint nextDepth = depth + 1u;
        if (nextDepth >= maxBounces) break;
        hitNormalPk = PackNormal(ctx.hitNormal);
        geoNormalPk = PackNormal(geometricNormal);
        hitT        = hitT_n;
        reorderHint = SharcReorderHint(ctx.instID, training.suffixLuma);
        ps += 1u << PT_PS_DEPTH_SHIFT;
    }
    SharcTrainingCommit(training);

    GuideRootsFinalize(GuideLane(), PtPsRoots(ps), RcBounceSeed(pathSeed, 0u, GUIDE_STREAM));
}
