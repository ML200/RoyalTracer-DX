#pragma once
// Per-vertex shading, run inline by the trace raygen.
#include "PtDefer_v8.hlsli"

struct PtVertexIO {
    uint   flags;
    float  pdf;      // in: previous scatter pdf; out: MIS/footprint pdf
    float  spread;   // path footprint, in/out
    float  dist;     // path length, in/out
    uint   cone;     // lobe cone for the cache test, in/out (PtConePack)
    uint   dirPk;    // out: next direction
    uint   nPk;      // out: shading normal at pos
    float3 pos;      // out: next ray's origin vertex (SSS exit included)
    float3 color;    // out: throughput multiplier (PV_CAPTURED), or sky radiance on a miss
    uint   auxPk;    // out: packed radiance or weight, per the result flags
    float3 nee;      // out: in-place light sample (PV_NEE)
    float3 neeLite;  // out: its reuse-suffix share (broad only at the park)
};

// Input flags (raygen -> vertex).
#define PV_IN_DEPTH_SHIFT   0u   // 7 bits: depth of the vertex being shaded
#define PV_IN_DIFF_SHIFT    7u   // 7 bits: diffuse bounces so far
#define PV_IN_SAMPLE_SHIFT  14u  // 3 bits
#define PV_IN_GUIDE_SHIFT   17u  // 4 bits
#define PV_IN_SSS           (1u << 21u)
#define PV_IN_LITE_VERTEX   (1u << 22u)
#define PV_IN_SPREAD        (1u << 23u)  // previous scatter had spread: grow the footprint
#define PV_IN_MIS_NONE      (1u << 24u)
#define PV_IN_PENDING       (1u << 25u)  // a deferred vertex exists: no capture
#define PV_IN_IMMEDIATE     (1u << 26u)  // this hit directly follows the deferred scatter
#define PV_IN_LAST          (1u << 27u)  // bounce limit: emission only
#define PV_IN_WATER_DIRECT  (1u << 28u)  // water NEE owns this segment's direct light
#define PV_IN_WATER_MEDIUM  (1u << 29u)
#define PV_IN_WATER_SCATTERED (1u << 30u)
#define PV_IN_SUN_OWNED     (1u << 31u)  // sun already sampled in the water for this chain

// Result flags (vertex -> raygen).
#define PV_RESULT_MASK      7u
#define PV_CONTINUE         0u
#define PV_TERMINATE        1u
#define PV_EMITTER          2u   // auxPk: emission, no light-sample partner
#define PV_EMITTER_DEFERRED 3u   // emitter record written; MIS in the deferred passes
#define PV_MISS             4u   // color: sky, auxPk: sun radiance, pdf: sun MIS weight
#define PV_CAPTURED         (1u << 3u)   // this vertex was deferred; color is the pre-capture scale
#define PV_KIND_SHIFT       4u   // 2 bits: DV_KIND_* of the captured scatter
#define PV_CACHE_HIT        (1u << 6u)   // auxPk: cache radiance, applied before the multiplier
#define PV_AUX_BROAD        (1u << 7u)   // auxPk: broad-lobe weight for the reuse suffix
#define PV_LITE_PARKED      (1u << 8u)
#define PV_SPREAD           (1u << 9u)
#define PV_MIS_NONE         (1u << 10u)
#define PV_PASS_THROUGH     (1u << 11u)
#define PV_DIFF_INC         (1u << 12u)
#define PV_STRATEGY_SHIFT   13u  // 2 bits
#define PV_PRIMARY_LITE     (1u << 15u)
#define PV_SSS_WALKED       (1u << 16u)  // the subsurface exit consumed one extra depth
#define PV_NEE              (1u << 17u)  // nee/neeLite hold this in-place vertex's light sample
#define PV_WATER_DIRECT     (1u << 18u)
#define PV_WATER_MEDIUM     (1u << 19u)
#define PV_WATER_SCATTERED  (1u << 20u)
#define PV_LITE_NEE         (1u << 21u)  // the deferred primary feeds the reuse reservoir (DVF_LITE_NEE)
#define PV_SUN_OWNED        (1u << 22u)

// Lobe cone as two f16: width here, growth per unit distance (SharcLobeConeAngle).
uint PtConePack(float width, float angle)
{
    return f32tof16(min(width, 65504.0f)) | (f32tof16(min(angle, 65504.0f)) << 16u);
}
float2 PtConeUnpack(uint cone) { return float2(f16tof32(cone & 0xffffu), f16tof32(cone >> 16u)); }

uint PvInputFlags(uint ps, bool pending, bool immediate, bool last)
{
    return (PtPsDepth(ps) << PV_IN_DEPTH_SHIFT) | (PtPsDiffDepth(ps) << PV_IN_DIFF_SHIFT) |
        (PtPsSample(ps) << PV_IN_SAMPLE_SHIFT) | (PtPsGuideDepth(ps) << PV_IN_GUIDE_SHIFT) |
        ((ps & PT_PS_SSS) != 0u ? PV_IN_SSS : 0u) | ((ps & PT_PS_LITE_VERTEX) != 0u ? PV_IN_LITE_VERTEX : 0u) |
        ((ps & PT_PS_SPREAD) != 0u ? PV_IN_SPREAD : 0u) | ((ps & PT_PS_MIS_NONE) != 0u ? PV_IN_MIS_NONE : 0u) |
        (pending ? PV_IN_PENDING : 0u) | (immediate ? PV_IN_IMMEDIATE : 0u) | (last ? PV_IN_LAST : 0u);
}

// Light-tree and sun sample of an in-place vertex, on the picked lobe group.
void PtInlineNee(HitContext ctx, SamplingP spPath, uint group, float groupP, float3 rayDir, uint pathSeed,
    uint depth, bool blue, uint2 pixel, uint blueIndex, bool liteBroadOnly, bool litePath, bool inWater,
    out float3 direct, out float3 liteDirect)
{
    direct = 0.0f;
    liteDirect = 0.0f;
    const bool waterDirect = LoadIsOceanMaterial(ctx.matID) && ctx.mediumMatID == MEDIUM_INVALID;
    const bool useLearnedLights = LTC_UseSurfaceLearning();
    uint sNee = RcBounceSeed(pathSeed, depth, RC_STREAM_NEE);
    [loop]
    for (uint tech = 0u; tech < 2u; ++tech)
    {
        // Widen the water's picked GGX lobe for the sun only.
        const bool sunTech = tech == 1u;
        const half neePr = (waterDirect && sunTech && group == LOBE_GROUP_SPEC)
            ? (half)OceanHighlightRoughness(ctx.hitLocalPr, LoadOceanSunLobeRoughness())
            : ctx.hitLocalPr;

        float3 L = 0.0f, visTarget = 0.0f, visTargetN = 0.0f, radiance = 0.0f;
        float  lightPdf = 0.0f, cosSurf = 0.0f;
        uint2  token = 0u;
        bool   sampled = false;
        OceanWaterSun waterSun = (OceanWaterSun)0;
        if (tech == 0u)
        {
            if ((rs_flags & RS_FLAG_NO_MESH_LIGHTS) != 0u || waterDirect) continue;
            const LT_Sample pick = LT_SampleLight(ctx.hitPos, ctx.hitNormal, sNee, useLearnedLights);
            token = pick.learningToken;
            const LT_LightSampleResult light = LT_SamplePointOnLightTree(ctx.hitPos, pick, sNee);
            const float3 toLight = light.position - ctx.hitPos;
            const float  dist    = sqrt(max(dot(toLight, toLight), 1e-20f));
            L = toLight / dist;
            cosSurf = dot(ctx.hitNormal, L);
            if (cosSurf > 1e-6f && dot(light.normal, -L) > 1e-6f && light.pdfSolidAngle > 1e-20f)
            {
                sampled    = true;
                visTarget  = light.position;
                visTargetN = light.normal;
                radiance   = light.emission;
                lightPdf   = light.pdfSolidAngle;
            }
        }
        else
        {
            float2 rSun = float2(RandomFloatSingle(sNee), RandomFloatSingle(sNee));
            if (blue) rSun = PtBlue2(pixel, blueIndex, depth, BN_PAIR_SUN);
            if (inWater)
            {
                if (OceanSampleWaterSun(ctx.hitPos, rSun, waterSun))
                {
                    L = waterSun.L;
                    cosSurf = dot(ctx.hitNormal, L);
                    sampled  = cosSurf > 1e-6f;
                    radiance = waterSun.radiance;
                    lightPdf = waterSun.pdf;
                }
            }
            else
            {
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
        }

        float reward = 0.0f;
        if (sampled)
        {
            const bool waterSunTech = inWater && sunTech;
            const float3 visT = waterSunTech ? OceanWaterSunReach(ctx.hitPos, ctx.hitNormal, waterSun)
                : VisibilityTransmittance(ctx.hitPos, ctx.hitNormal, visTarget, visTargetN, false, !waterDirect);
            if (any(visT > 0.0f))
            {
                const BrdfData lobe = EvaluateLobe(spPath, group, ctx.matID, ctx.hitNormal, ctx.hitNormal, L,
                    -rayDir, ctx.hitLocalKd, neePr, ctx.hitLocalPm, ctx.iors.x, ctx.iors.y);
                if (lobe.pdf > 0.0f)
                {
                    const float3 broad = group == LOBE_GROUP_BROAD ? lobe.val : 0.0f;
                    reward = dot(radiance * cosSurf * visT *
                        LTC_TrainShare((float)ctx.hitLocalPr, ctx.matID, lobe.val, group == LOBE_GROUP_BROAD) / lightPdf,
                        float3(0.2126f, 0.7152f, 0.0722f));
                    // Water direct light and the in-water sun have no BSDF partner.
                    const float  misWeight  = waterDirect || waterSunTech ? 1.0f : lightPdf / (lightPdf + lobe.pdf);
                    const float3 lightScale = radiance * cosSurf * visT * (misWeight / (lightPdf * groupP));
                    direct += lobe.val * lightScale;
                    if (litePath) liteDirect += (liteBroadOnly ? broad : lobe.val) * lightScale;
                }
            }
        }
        // Failed picks train too, with zero reward.
        if (tech == 0u) LT_TrainSample(token, reward);
    }
}

// SSS walk, cache, lobe pick, capture, continuation; ordered to cut live state.
void PtVertexShade(inout PtVertexIO io, HitContext ctx, float3 geoN, float3 dirIn, bool flipIOR,
    uint2 pixel, uint pixelIdx, uint presetOut)
{
    const uint inFlags    = io.flags;
    const uint depth      = (inFlags >> PV_IN_DEPTH_SHIFT) & 0x7Fu;
    const uint diffDepth  = (inFlags >> PV_IN_DIFF_SHIFT) & 0x7Fu;
    const uint sample     = (inFlags >> PV_IN_SAMPLE_SHIFT) & 7u;
    const uint guideDepth = (inFlags >> PV_IN_GUIDE_SHIFT) & 15u;
    const bool pending    = (inFlags & PV_IN_PENDING) != 0u;
    const bool misNoneIn  = (inFlags & PV_IN_MIS_NONE) != 0u;
    const bool inWater    = (inFlags & PV_IN_WATER_MEDIUM) != 0u;
    bool  sssEntered      = (inFlags & PV_IN_SSS) != 0u;
    bool  liteVertex      = (inFlags & PV_IN_LITE_VERTEX) != 0u;
    const uint pathSeed   = PtPathSeed(pixel, sample);
    const uint N          = PtSampleCount();
    const bool blue       = diffDepth == 0u;
    const uint blueIndex  = (uint)time * N + sample;
    const bool freeBounce = MaterialIsFreeBounce(ctx.matID);

    uint   res    = presetOut;
    float3 color  = 1.0f;
    uint   auxPk  = 0u;
    float3 dir    = 0.0f;
    float  pdfOut = io.pdf;
    bool   alive  = true;
    float3 nee = 0.0f, neeLite = 0.0f;
    g_regularizeRoughness = io.spread > 0.0f ? PT_REGULARIZE_ROUGHNESS : 0.0f;

    // Subsurface walk first; the exit is a white Lambertian vertex.
    float3 rayDir = dirIn;
    bool   walked = false;
    if (!sssEntered && LoadIsSSS(ctx.matID))
    {
        uint sSss = RcBounceSeed(pathSeed, depth, RC_STREAM_SSS);
        const float fT     = 1.0f - FresnelDielectric(-rayDir, ctx.hitNormal, ctx.iors.x, ctx.iors.y).x;
        const float pEnter = saturate(LoadSSSWeight(ctx.matID) * fT);
        if (RandomFloatSingle(sSss) < pEnter)
        {
            const SSSWalkResult w = SubsurfaceWalk(ctx.hitPos, ctx.hitNormal, ctx.matID, sSss);
            if (!w.valid) alive = false;
            else
            {
                color *= w.wTotal * (float3)ctx.hitLocalKd;
                ctx.hitPos         = w.exitPos;
                ctx.hitNormal      = w.exitNormal;
                ctx.backface       = false;
                ctx.hitLocalKd     = (half3)float3(1, 1, 1);
                ctx.hitLocalPr     = (half)1.0f;
                ctx.hitLocalPm     = (half)0.0f;
                ctx.iors           = (half2)float2(1.0f, 1.0f);
                ctx.mediumMatID    = MEDIUM_INVALID;
                ctx.absorptionTint = (half3)float3(1, 1, 1);
                geoN    = w.exitNormal;
                rayDir  = -w.exitNormal;
                flipIOR = false;
                res |= PV_SSS_WALKED;
                walked     = true;
                sssEntered = true;
            }
        }
    }
    const float3 n      = ctx.hitNormal;
    const float3 absorb = (float3)ctx.absorptionTint;
    const float3 outN   = n;

    uint sBsdf = RcBounceSeed(pathSeed, depth, RC_STREAM_BSDF);
    SamplingP spPath = CalculateStrategyProbabilities(ctx.matID, -rayDir, n, ctx.iors.x, ctx.iors.y,
        ctx.hitLocalKd, ctx.hitLocalPr, ctx.hitLocalPm);
    const bool liteX2 = depth == 2u && liteVertex;

    // Radiance cache, gated by the incoming lobe cone, not this material (SharcConeRamp).
    const bool cacheSurface = sharc_enabled != 0u && !walked && SharcMaterialEligible(ctx, spPath, geoN);
    float2 cone = PtConeUnpack(io.cone);
    float cacheScale = 1.0f;
    bool  cacheRefused = false;
    if (alive && cacheSurface && depth > 1u)
    {
        uint sCache = RcBounceSeed(pathSeed, depth, 0x53484152u);
        const float ramp = SharcConeRamp(cone.x, cone.y, ctx.hitPos);
        const bool accepted = ramp >= 1.0f || (ramp > 0.0f && RandomFloatSingle(sCache) < ramp);
        cacheRefused = !accepted;
        float3 cached = 0.0f;
        bool hit = false;
        if (accepted)
            hit = (pending && !inWater) ? SharcQueryForced(SharcMakeSurface(ctx, geoN), sCache, cached)
                                        : SharcQueryDraws(SharcMakeSurface(ctx, geoN), sCache, cached);
        float layerT = 0.0f;
        if (hit)
        {
            layerT = SharcLayerTransmission(spPath, ctx, -rayDir);
            hit = layerT > 1e-3f;
        }
        if (hit)
        {
            res |= PV_CACHE_HIT;
            auxPk = PvPackRadiance(cached * layerT);
            const float pSpecular = clamp(1.0f - layerT, 0.02f, 1.0f);
            if (!DropBroadLobes(spPath, ctx.hitLocalPr) || RandomFloatSingle(sCache) >= pSpecular) alive = false;
            else cacheScale = rcp(pSpecular);
        }
    }

    // One lobe group per sample.
    const float uStrategy = blue ? PtBlue1(pixel, blueIndex, depth, BN_DIM_STRATEGY) : RandomFloatSingle(sBsdf);
    const uint  strategy  = SelectSamplingStrategyFrom(spPath, uStrategy);
    const uint  group     = LobeGroupOf(strategy, ctx.hitLocalPr);
    const float groupP    = LobeGroupP(spPath, group, ctx.hitLocalPr);
    const bool  wide      = SharcScatterHasSpread(strategy, ctx.matID, ctx.hitLocalPr);
    const bool waterDirect = LoadIsOceanMaterial(ctx.matID) && ctx.mediumMatID == MEDIUM_INVALID;
    bool liteSurface = false;
    if (depth == 1u)
    {
        // Reuse: broad NEE of any deferring pick, continuation of broad picks only.
        liteSurface = LITE_ENABLED && !inWater && !waterDirect && !sssEntered &&
            HasBroadShare(spPath, ctx.hitLocalPr, ctx.hitLocalPm) &&
            !LoadIsSSS(ctx.matID) && ctx.mediumMatID == MEDIUM_INVALID &&
            LoadKd_w(ctx.matID) >= EPSILON && !freeBounce;
        liteVertex = liteSurface && group == LOBE_GROUP_BROAD;
        if (liteVertex) res |= PV_PRIMARY_LITE;
    }

    if (alive)
    {
        // Narrow picks bounce in place; the first wide pick is deferred.
        const bool neeBase = ctx.mediumMatID == MEDIUM_INVALID &&
            (LoadKd_w(ctx.matID) >= EPSILON || !GGXUsesDeltaSampling(ctx.matID, ctx.hitLocalPr));
        const bool capture     = !waterDirect && !inWater && wide && !pending;
        // In-place NEE: water surface, in water, or where the cache can't answer.
        const bool inlineNee   = !capture && (waterDirect ||
            (neeBase && wide && (inWater || !cacheSurface || cacheRefused)));
        const bool budgetBreak = !freeBounce && diffDepth >= (uint)pt_maxDiffuseBounces;

        if (capture)
        {
            // Defer; written first so its inputs die before sampling.
            DvVertex dv;
            dv.pos = ctx.hitPos; dv.n = n; dv.dirIn = rayDir;
            dv.matID = ctx.matID; dv.instID = ctx.instID; dv.Kd = (float3)ctx.hitLocalKd;
            dv.Pr = ctx.hitLocalPr; dv.Pm = ctx.hitLocalPm; dv.sp = spPath; dv.absorb = absorb;
            dv.flags = (ctx.backface ? DVF_BACKFACE : 0u) | (flipIOR ? DVF_FLIP_IOR : 0u) |
                (neeBase ? DVF_PERFORM_NEE : 0u) | (walked ? DVF_UNIT_IOR : 0u) |
                (group << DVF_GROUP_SHIFT) | (io.spread > 0.0f ? DVF_REGULARIZE : 0u) |
                (HasBroadShare(spPath, ctx.hitLocalPr, ctx.hitLocalPm) ? DVF_BROAD_NEE : 0u) |
                (liteSurface ? DVF_LITE_NEE : 0u);
            DvStoreVertex(pixelIdx, dv);
            DvStorePickP(pixelIdx, groupP);
            color *= cacheScale;
            cacheScale = 1.0f;
            res |= PV_CAPTURED | (liteSurface ? PV_LITE_NEE : 0u);
        }
        else if (inlineNee)
        {
            PtInlineNee(ctx, spPath, group, groupP, rayDir, pathSeed, depth, blue, pixel, blueIndex,
                liteX2 && cacheSurface, depth >= 2u && liteVertex, inWater, nee, neeLite);
            res |= PV_NEE | (inWater ? PV_SUN_OWNED : 0u);
        }

        if (budgetBreak)
        {
            if (capture)
            {
                DvStoreScatter(pixelIdx, (DvScatter)0);
                res |= (DV_KIND_NONE << PV_KIND_SHIFT) | (misNoneIn ? PV_MIS_NONE : 0u);
            }
            alive = false;
        }
        else
        {
            // Continuation; guide cones built after the lobe sample to cut live state.
            if (strategy == 0u)
            {
                const float2 u = blue ? PtBlue2(pixel, blueIndex, depth, BN_PAIR_COSINE)
                    : float2(RandomFloatSingle(sBsdf), RandomFloatSingle(sBsdf));
                dir = SampleBRDF_LambertianFrom(n, n, u);
            }
            else
                dir = SampleBRDF_WithStrategy(strategy, ctx.matID, -rayDir, n, n, ctx.hitLocalKd,
                    ctx.hitLocalPr, ctx.hitLocalPm, sBsdf, ctx.iors.x, ctx.iors.y, false);
            float guideQ = 0.0f, guidePdf = 0.0f;
            if (group == LOBE_GROUP_BROAD && cacheSurface && GUIDE_ENABLED &&
                HasBroadShare(spPath, ctx.hitLocalPr, ctx.hitLocalPm) && 1u + guideDepth <= (uint)GUIDE_MAX_DEPTH)
            {
                uint sKey = RcBounceSeed(pathSeed, depth, 0x4b455953u);
                const GuideSet guide = GuideBuildAt(GuideKeyOf(ctx.hitPos, geoN, sKey), ctx.hitPos, n);
                if (guide.q > 0.0f)
                {
                    guideQ = guide.q;
                    uint sGuide = RcBounceSeed(pathSeed, depth, GUIDE_STREAM);
                    const float uTest = blue ? PtBlue1(pixel, blueIndex, depth, BN_DIM_GUIDE_TEST) : RandomFloatSingle(sGuide);
                    if (uTest < guide.q)
                    {
                        const float  uPick = blue ? PtBlue1(pixel, blueIndex, depth, BN_DIM_GUIDE_PICK) : RandomFloatSingle(sGuide);
                        const float2 uDir  = blue ? PtBlue2(pixel, blueIndex, depth, BN_PAIR_GUIDE_CAP)
                            : float2(RandomFloatSingle(sGuide), RandomFloatSingle(sGuide));
                        uint pick;
                        dir = GuideSample(guide, uPick, uDir, pick);
                    }
                    guidePdf = GuidePdf(guide, dir);   // the cones are dead from here on
                }
            }
            res |= strategy << PV_STRATEGY_SHIFT;
            if (!freeBounce) res |= PV_DIFF_INC;
            if (wide) res |= PV_SPREAD;
            const bool passThrough = strategy == 1u && dot(dir, n) < 0.0f && LoadKd_w(ctx.matID) < 1.0f - EPSILON;
            cone.y += SharcLobeConeAngle(strategy, ctx.matID, ctx.hitLocalPr, abs(dot(n, rayDir)),
                passThrough && !LoadIsThinGlass(ctx.matID) ? (float)ctx.iors.x / (float)ctx.iors.y : 0.0f);

            // Capture: pdf-only call, so the compiler strips the value.
            BrdfData lobe;
            if (capture)
            {
                lobe.pdf = EvaluateLobe(spPath, group, ctx.matID, n, n, dir, -rayDir,
                    ctx.hitLocalKd, ctx.hitLocalPr, ctx.hitLocalPm, ctx.iors.x, ctx.iors.y).pdf;
                lobe.val = 0.0f;
            }
            else
                lobe = EvaluateLobe(spPath, group, ctx.matID, n, n, dir, -rayDir,
                    ctx.hitLocalKd, ctx.hitLocalPr, ctx.hitLocalPm, ctx.iors.x, ctx.iors.y);
            const float pdfTotal = groupP * (guideQ > 0.0f ? max(lerp(lobe.pdf, guidePdf, guideQ), 0.0f) : lobe.pdf);
            const bool  valid    = dot(dir, dir) > 1e-12f && pdfTotal > 1e-6f;
            if (OceanDirectLightingOwnsRay(waterDirect, dot(dir, n)) ||
                (passThrough && LoadIsThinGlass(ctx.matID) && (inFlags & PV_IN_WATER_DIRECT) != 0u))
                res |= PV_WATER_DIRECT;
            // The in-water sun sample owns the sun along this specular chain.
            if ((inFlags & PV_IN_SUN_OWNED) != 0u && !wide && LoadIsOceanMaterial(ctx.matID) && ctx.backface)
                res |= PV_SUN_OWNED;

            if (capture)
            {
                // Densities only; MIS uses the unguided density.
                DvScatter dsc;
                dsc.dirOut = dir; dsc.info = 0u; dsc.pdfTotal = pdfTotal; dsc.bsdfPdf = lobe.pdf;
                DvStoreScatter(pixelIdx, dsc);
                if (!valid)
                {
                    res |= (DV_KIND_NONE << PV_KIND_SHIFT) | (misNoneIn ? PV_MIS_NONE : 0u);
                    alive = false;
                }
                else
                {
                    bool misNone = misNoneIn;
                    if (passThrough) res |= PV_PASS_THROUGH;
                    else if (neeBase) { pdfOut = lobe.pdf; misNone = false; }
                    else misNone = true;
                    res |= (DV_KIND_BSDF << PV_KIND_SHIFT) | (misNone ? PV_MIS_NONE : 0u);
                }
            }
            else
            {
                // In place: narrow picks, or anything after the deferred vertex.
                const float  cosTheta = abs(dot(n, dir));
                const float3 W = valid ? lobe.val * absorb * cosTheta / pdfTotal : float3(0, 0, 0);
                if (!valid || any(isnan(W)) || any(isinf(W)) || !any(W > 0.0f)) alive = false;
                else
                {
                    if (passThrough) res |= PV_PASS_THROUGH;
                    else
                    {
                        // MIS only against an in-place light sample.
                        pdfOut = lobe.pdf;
                        if (!inlineNee) res |= PV_MIS_NONE;
                    }
                    color *= W * cacheScale;
                    if (liteX2 && cacheSurface && (res & PV_CACHE_HIT) == 0u)
                    {
                        // Reuse suffix: all of a broad pick, none of another.
                        res |= PV_AUX_BROAD;
                        auxPk = PackRGB9E5(group == LOBE_GROUP_BROAD ? W : 0.0f);
                    }
                }
            }
        }
    }

    // Only a water interface changes the medium; reflection (incl. TIR) stays.
    bool waterMedium = (inFlags & PV_IN_WATER_MEDIUM) != 0u;
    if (LoadIsOceanMaterial(ctx.matID) && !LoadIsThinGlass(ctx.matID))
        waterMedium = ctx.backface ? dot(dir,n) >= 0.0f : dot(dir,n) < 0.0f;
    if (waterMedium) res |= PV_WATER_MEDIUM;
    if (OceanScatterUsedAfterSurface((inFlags & PV_IN_WATER_SCATTERED) != 0u,
        (inFlags & PV_IN_WATER_MEDIUM) != 0u, waterMedium, LoadIsOceanMaterial(ctx.matID),
        LoadIsThinGlass(ctx.matID) && dot(dir,n) < 0.0f)) res |= PV_WATER_SCATTERED;
    if (!alive) res |= PV_TERMINATE;
    io.flags   = res;
    io.pdf     = pdfOut;
    io.dirPk   = PackNormal(dir);
    io.nPk     = PackNormal(outN);
    io.cone    = PtConePack(cone.x, cone.y);
    io.pos     = ctx.hitPos;
    io.color   = color;
    io.auxPk   = auxPk;
    io.nee     = nee;
    io.neeLite = neeLite;
}

// Primary vertex from the camera record; geoN is in scratch while the cache is on.
void PtPrimaryContext(uint2 pixel, uint pixelIdx, out HitContext ctx, out float3 geoN, out bool flipIOR)
{
    const SDRecord sd = load_SD(g_sample_current, pixelIdx);
    float2 pIors; uint pMedium; float3 pAbsorb;
    load_rg_primaryExtra(pixelIdx, pIors, pMedium, pAbsorb);
    geoN = (sharc_enabled != 0u && SHARC_DEBUG_MODE == 0u)
        ? gScratchPing[uint3(pixel, SHARC_DEBUG_SCRATCH)].xyz : sd.n1_s;
    flipIOR = pMedium != MEDIUM_INVALID;

    ctx = (HitContext)0;
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
}

// Secondary hit context; false if resolved here (emitter or bounce limit).
bool PtHitContext(inout PtVertexIO io, uint instID, uint geometryIndex, uint primitiveIndex,
    float2 barycentrics, float hitT, float3 rayDir, float3 prevPos, float3 prevN, uint pixelIdx,
    out HitContext ctx, out float3 geoN, out bool flipIOR, out uint presetOut)
{
    const uint   depth    = (io.flags >> PV_IN_DEPTH_SHIFT) & 0x7Fu;
    const uint primID = FlatPrimID(instID, geometryIndex, primitiveIndex);

    ctx       = (HitContext)0;
    geoN      = 0.0f;
    flipIOR   = false;
    presetOut = 0u;
    bool shade = true;

    // Beam width: pixel cone over the path plus rough-scatter spread.
    const float spreadHere = (io.flags & PV_IN_SPREAD) != 0u
        ? hitT * sqrt(min(16.0f, rcp(max(io.pdf, 1e-6f)))) : 0.0f;
    const float footprint = PixelConeAngle() * (io.dist + hitT) + io.spread + spreadHere;
    io.dist += hitT;
    const HitInfo hinfo = EvalSurfaceStateDir(instID, primID, barycentrics, rayDir, footprint);
    const uint matID = ResolveSurfaceMaterial(GetMatIDFast(instID, primID), hinfo);
    const float3  emission = (hinfo.lightID != 0xFFFFFFFFu)
        ? g_EmissiveTriangles[hinfo.lightID].emission * GLOBAL_EMISSION_STRENGTH
        : float3(0, 0, 0);
    if (any(emission > 0.0f))
    {
        // Water NEE already sampled this emitter (unless mesh NEE is off).
        if ((io.flags & PV_IN_WATER_DIRECT) != 0u && (rs_flags & RS_FLAG_NO_MESH_LIGHTS) == 0u)
        {
            io.flags = PV_TERMINATE;
            io.auxPk = 0u;
            return false;
        }
        const bool pending   = (io.flags & PV_IN_PENDING) != 0u;
        const bool immediate = (io.flags & PV_IN_IMMEDIATE) != 0u;
        if (pending && immediate)
        {
            DvEmitter e;
            e.lightID = hinfo.lightID;
            e.inst = instID;
            e.pos = hinfo.hitPos;
            e.n = hinfo.hitNormal;
            DvStoreEmitter(pixelIdx, e);
            io.flags = PV_EMITTER_DEFERRED;
        }
        else
        {
            // Balance heuristic vs the previous vertex's light sample (Veach 1997).
            float misWeight = 1.0f;
            if ((io.flags & PV_IN_MIS_NONE) == 0u)
            {
                const float lightPdfArea = LT_Pdf_LightTree_Area(prevPos, prevN, hinfo.lightID, instID,
                    LTC_UseSurfaceLearning());
                const float cosLight   = max(dot(hinfo.hitNormal, -rayDir), 0.0f);
                const float lightPdfSA = (cosLight > EPSILON) ? (lightPdfArea * max(hitT * hitT, EPSILON) / cosLight) : 0.0f;
                misWeight = io.pdf / max(io.pdf + lightPdfSA, EPSILON);
            }
            io.flags = PV_EMITTER;
            io.auxPk = PvPackRadiance(emission * misWeight);
        }
        shade = false;
    }
    else if ((io.flags & PV_IN_LAST) != 0u)
    {
        io.flags = PV_TERMINATE;
        shade = false;
    }
    else
    {
        const float matNi        = LoadNi(matID);
        const bool  transmissive = LoadKd_w(matID) < 1.0f - EPSILON;
        flipIOR = hinfo.backface && transmissive && !LoadIsThinGlass(matID);
        float3 hitLocalKd; float hitLocalPr, hitLocalPm;
        RefetchMaterial(matID, hinfo, hitLocalKd, hitLocalPr, hitLocalPm);

        if (depth == 2u && (io.flags & PV_IN_LITE_VERTEX) != 0u)
        {
            // Park the reuse point; the material pass applies its broad share.
            const float3 liteNy = dot(hinfo.geometricNormal, -rayDir) < 0.0f
                ? -hinfo.geometricNormal : hinfo.geometricNormal;
            LiteParkPointStore(pixelIdx, WorldToObjectPos(instID, hinfo.hitPos), instID,
                PackNormal(WorldToObjectNrm(instID, liteNy)), DvLoadScatterPdf(pixelIdx));
            presetOut |= PV_LITE_PARKED;
        }

        if ((io.flags & PV_IN_SPREAD) != 0u)
            io.spread += hitT * sqrt(min(16.0f, rcp(max(io.pdf * abs(dot(hinfo.geometricNormal, -rayDir)), 1e-6f))));
        const float2 cone = PtConeUnpack(io.cone);
        io.cone = PtConePack(cone.x + hitT * cone.y, cone.y);
        const float regularize = io.spread > 0.0f ? PT_REGULARIZE_ROUGHNESS : 0.0f;

        geoN = hinfo.geometricNormal;
        ctx.hitPos         = hinfo.hitPos;
        ctx.hitNormal      = hinfo.hitNormal;
        ctx.matID          = matID;
        ctx.instID         = instID;
        ctx.backface       = hinfo.backface;
        ctx.hitLocalKd     = (half3)hitLocalKd;
        ctx.hitLocalPr     = (half)max(hitLocalPr, regularize);
        ctx.hitLocalPm     = (half)hitLocalPm;
        ctx.iors           = (half2)(flipIOR ? float2(matNi, 1.0f) : float2(1.0f, matNi));
        ctx.mediumMatID    = flipIOR ? matID : MEDIUM_INVALID;
        ctx.absorptionTint = (half3)(flipIOR && !LoadIsOceanMaterial(matID) ? CalculateAbsorptionThroughput(LoadTf(matID), hitT) : float3(1, 1, 1));
    }
    return shade;
}

// Escaped ray: sky, sun and its MIS weight. rayOrigin only places the sky observer.
void PtShadeMiss(inout PtVertexIO io, float3 rayOrigin, float3 rayDir)
{
    const bool underground = WorldPosIsUnderground(rayOrigin + sceneOriginWorld);
    // With a sea, observe from the ray origin: the camera may be underground.
    SetSkyObserver((OCEAN_ENABLED ? rayOrigin : InitOrigin()) + sceneOriginWorld);
    const bool waterDirect = (io.flags & (PV_IN_WATER_DIRECT | PV_IN_SUN_OWNED)) != 0u;
    const float  sunSAPdf   = underground || waterDirect ? 0.0f : GetSunPdf(rayDir);
    const float3 sunRad     = (sunSAPdf > 0.0f) ? EvaluateSun(rayDir) : float3(0, 0, 0);
    const float  sunMisBsdf = (sunSAPdf > 0.0f)
        ? ((io.flags & PV_IN_MIS_NONE) != 0u ? 1.0f : io.pdf / max(io.pdf + sunSAPdf, EPSILON)) : 0.0f;

    io.flags = PV_MISS;
    io.pdf   = sunSAPdf > 0.0f ? sunMisBsdf : 1.0f;
    io.color = underground ? float3(0, 0, 0) : EvaluateSky(rayDir);
    io.auxPk = PvPackRadiance(sunRad);
}
