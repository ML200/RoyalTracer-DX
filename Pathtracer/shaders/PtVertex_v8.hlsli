#pragma once
// Per-vertex work of the split path tracer. The trace raygen reorders the secondary hits and then
// shades every vertex here itself, including the primary one and the escaped rays; the
// closest-hit and miss shaders of the pipeline are empty. The raygen builds the context of the
// vertex (PtPrimaryContext, PtHitContext) and hands it to the one PtVertexShade call, so the
// shading code is inlined once. The path loop only handles the compact result: the next
// direction, the vertex position and normal for the ray offset, a throughput multiplier, one
// packed value and flags.
#include "PtDefer_v8.hlsli"

struct PtVertexIO {
    uint   flags;
    float  pdf;      // in: pdf of the previous scatter; out: updated MIS/footprint pdf
    float  spread;   // path footprint, in/out
    float  dist;     // path length, in/out: with the pixel cone it sizes the texture footprint
    uint   cone;     // cone of the path's lobes for the cache test, in/out (PtConePack)
    uint   dirPk;    // out: next direction
    uint   nPk;      // out: shading normal of the vertex the next ray leaves
    float3 pos;      // out: position of the vertex the next ray leaves (a subsurface exit included)
    float3 color;    // out: throughput multiplier (see PV_CAPTURED), or sky radiance on a miss
    uint   auxPk;    // out: packed radiance or weight, see the result flags
    float3 nee;      // out: direct light of an in-place vertex from its own light sample (PV_NEE)
    float3 neeLite;  // out: the same for the reuse suffix (broad lobes only at the parked vertex)
};

// Input flags (raygen -> hit/miss shaders).
#define PV_IN_DEPTH_SHIFT   0u   // 7 bits: depth of the vertex being shaded
#define PV_IN_DIFF_SHIFT    7u   // 7 bits: diffuse bounces so far
#define PV_IN_SAMPLE_SHIFT  14u  // 3 bits
#define PV_IN_GUIDE_SHIFT   17u  // 4 bits
#define PV_IN_SSS           (1u << 21u)
#define PV_IN_LITE_VERTEX   (1u << 22u)
#define PV_IN_SPREAD        (1u << 23u)  // the previous scatter had spread: grow the footprint
#define PV_IN_MIS_NONE      (1u << 24u)
#define PV_IN_PENDING       (1u << 25u)  // a deferred vertex exists: no capture, no light sampling
#define PV_IN_IMMEDIATE     (1u << 26u)  // this hit directly follows the deferred scatter
#define PV_IN_LAST          (1u << 27u)  // bounce limit: emission only
#define PV_IN_WATER_DIRECT  (1u << 28u)  // water NEE already owns direct-light radiance on this segment
#define PV_IN_WATER_MEDIUM  (1u << 29u)
#define PV_IN_WATER_SCATTERED (1u << 30u)
#define PV_IN_SUN_OWNED     (1u << 31u)  // a surface in the water sampled the sun for this specular chain

// Result flags (hit/miss shaders -> raygen).
#define PV_RESULT_MASK      7u
#define PV_CONTINUE         0u
#define PV_TERMINATE        1u
#define PV_EMITTER          2u   // auxPk: emission, no light-sample partner
#define PV_EMITTER_DEFERRED 3u   // emitter record written, MIS resolved by the light/material passes
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
#define PV_NEE              (1u << 17u)  // nee/neeLite carry the light sample of this in-place vertex
#define PV_WATER_DIRECT     (1u << 18u)
#define PV_WATER_MEDIUM     (1u << 19u)
#define PV_WATER_SCATTERED  (1u << 20u)
#define PV_LITE_NEE         (1u << 21u)  // the deferred primary feeds the reuse reservoir (DVF_LITE_NEE)
#define PV_SUN_OWNED        (1u << 22u)

// The cone of the lobes picked so far (SharcLobeConeAngle), as two halves: its width at the
// current vertex and its full angle, by which the width grows per unit of distance.
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

// Light sample of an in-place vertex: a wide pick after the deferred vertex where the cache may not
// answer (the cone of the path too narrow for its cells, or a surface it does not know), a vertex
// in the water (the medium attenuation has no slot in the deferred records), or the water surface
// (its own light model). A surface the cache may answer but holds nothing for yet goes on without
// one. One light-tree pick and the sun, evaluated on the lobe group this sample picked,
// divided by the probability of that pick, and MIS-weighted against the group's own density; the
// pick also trains the light tree. liteBroadOnly restricts the reuse-suffix share to the broad
// group, as at the parked vertex. A vertex in the water takes the sun refracted by the surface
// (OceanSampleWaterSun) and owns it: the hits of its continuation out through the surface do not
// count the sun again (PV_SUN_OWNED).
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
        // Only the sun is worth widening the water's lobe for. A light-tree connection almost
        // never lands inside a lobe that narrow, so the estimator is nearly all variance and pays
        // a shadow ray for it; BSDF sampling finds those lights on its own and its MIS partner is
        // already there. Everything else keeps the authored roughness.
        // Only a picked GGX lobe widens: the broad group keeps the roughness its membership was
        // decided with.
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
                    // Water direct lighting has no BSDF-hit partner: continuation stays sharp. Nor
                    // has the sun of a vertex in the water, which its continuation leaves out.
                    const float  misWeight  = waterDirect || waterSunTech ? 1.0f : lightPdf / (lightPdf + lobe.pdf);
                    const float3 lightScale = radiance * cosSurf * visT * (misWeight / (lightPdf * groupP));
                    direct += lobe.val * lightScale;
                    if (litePath) liteDirect += (liteBroadOnly ? broad : lobe.val) * lightScale;
                }
            }
        }
        // A pick that finds no light still trains its cluster, with the zero reward it earned.
        if (tech == 0u) LT_TrainSample(token, reward);
    }
}

// Shade one vertex: subsurface walk, cache lookup, the lobe pick, capture of the first wide
// vertex, then the continuation. The phases are ordered so the inputs of each phase die before the
// next: the vertex record is written before the lobe is sampled, and the guide cones die before
// the BSDF evaluation.
//
// Each sample picks one lobe group and evaluates only that one (EvaluateLobe). A wide pick (a
// diffuse or rough lobe) defers the vertex together with its light sample, which the light and
// material passes take on the same pick, and on the broad group whatever the pick; a mirror-like
// pick bounces in place, so a reflection seen in a smooth surface is lit at the surface it shows.
// After the deferred vertex the cache ends the path wherever the cone of its lobes is wide enough
// for the cells (SharcConeRamp); where it is not, a wide pick takes its light sample in place
// (PtInlineNee). The hits of a vertex without a light sample count in full.
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

    // --- subsurface: the walk moves the vertex to its exit before anything else looks at it.
    // The exit is a white Lambertian vertex reached from inside, with the walk weight and the
    // albedo in the throughput, and is shaded like any other wide vertex from here on. ---
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

    // --- radiance cache: a diffuse hit ends the path here, or leaves a specular continuation.
    // The subsurface exit is no surface the cache knows. Only the cone of the lobes that led here
    // decides, never the material of this surface: the cache answers where one of its cells fills
    // at most a share of that cone, so the lobe reaches other cells as well (SharcConeRamp). A
    // diffuse cone is the hemisphere, which a cell never fills that much of short of contact; a
    // 0.2-roughness lobe passes at long range and not at short range. After the deferred vertex
    // the cache answers wherever it holds anything; before it (and in water) its own confidence
    // decides as well. (The training pass keeps a share of its paths going past the cache so that
    // it does not only learn from itself; this pass does not.) ---
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

    // --- the lobe of this sample: one group is picked and evaluated, a wide pick makes the vertex
    // wide and takes the light sample ---
    const float uStrategy = blue ? PtBlue1(pixel, blueIndex, depth, BN_DIM_STRATEGY) : RandomFloatSingle(sBsdf);
    const uint  strategy  = SelectSamplingStrategyFrom(spPath, uStrategy);
    const uint  group     = LobeGroupOf(strategy, ctx.hitLocalPr);
    const float groupP    = LobeGroupP(spPath, group, ctx.hitLocalPr);
    const bool  wide      = SharcScatterHasSpread(strategy, ctx.matID, ctx.hitLocalPr);
    const bool waterDirect = LoadIsOceanMaterial(ctx.matID) && ctx.mediumMatID == MEDIUM_INVALID;
    bool liteSurface = false;
    if (depth == 1u)
    {
        // The diffuse reuse takes the broad light sample of every pick that defers the vertex, and
        // the continuation of a broad pick as well; a pick that bounces in place leaves the
        // reservoir empty.
        liteSurface = LITE_ENABLED && !inWater && !waterDirect && !sssEntered &&
            HasBroadShare(spPath, ctx.hitLocalPr, ctx.hitLocalPm) &&
            !LoadIsSSS(ctx.matID) && ctx.mediumMatID == MEDIUM_INVALID &&
            LoadKd_w(ctx.matID) >= EPSILON && !freeBounce;
        liteVertex = liteSurface && group == LOBE_GROUP_BROAD;
        if (liteVertex) res |= PV_PRIMARY_LITE;
    }

    if (alive)
    {
        // --- classification: a narrow pick bounces in place, a wide one is deferred with the
        // path's light sample unless a deferred vertex already exists. ---
        const bool neeBase = ctx.mediumMatID == MEDIUM_INVALID &&
            (LoadKd_w(ctx.matID) >= EPSILON || !GGXUsesDeltaSampling(ctx.matID, ctx.hitLocalPr));
        const bool capture     = !waterDirect && !inWater && wide && !pending;
        // A wide pick lights itself in place where the cache may not answer (the cone too narrow
        // for its cells, or a surface it does not know), and anywhere in the water (the medium
        // attenuation has no slot in the deferred records). The water surface does on every pick,
        // with its own light model. A surface the cache may answer but holds nothing for yet goes
        // on without one.
        const bool inlineNee   = !capture && (waterDirect ||
            (neeBase && wide && (inWater || !cacheSurface || cacheRefused)));
        const bool budgetBreak = !freeBounce && diffDepth >= (uint)pt_maxDiffuseBounces;

        if (capture)
        {
            // Defer this vertex: the light and material passes finish it on the picked group, and
            // take the light sample on the broad group as well if the vertex has one.
            // Written first so its inputs are not kept alive through the sampling below.
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
            // --- continuation: sample the picked lobe, maybe replace it by a guide draw, then
            // evaluate that lobe group alone. The cones are built after the lobe sample, so they
            // overlap neither the samplers nor the evaluation below; only the broad group is guided.
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

            // The group's density, with the guide mixed in, times the probability of the pick. The
            // deferred vertex needs the density only (its value is the material pass's), so it
            // has a call of its own that the compiler strips down to the densities.
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
            // A surface in the water took its own sun sample, so the specular chain it continues on
            // - out through the mirror-smooth underside of the surface, or turned back by it - must
            // not find the sun a second time.
            if ((inFlags & PV_IN_SUN_OWNED) != 0u && !wide && LoadIsOceanMaterial(ctx.matID) && ctx.backface)
                res |= PV_SUN_OWNED;

            if (capture)
            {
                // Densities only: the material pass supplies the value of this scatter. The light
                // sample is weighted against the group density without the guide, on both sides.
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
                // In-place vertex: narrow picks before the deferred vertex, or anything after it.
                const float  cosTheta = abs(dot(n, dir));
                const float3 W = valid ? lobe.val * absorb * cosTheta / pdfTotal : float3(0, 0, 0);
                if (!valid || any(isnan(W)) || any(isinf(W)) || !any(W > 0.0f)) alive = false;
                else
                {
                    if (passThrough) res |= PV_PASS_THROUGH;
                    else
                    {
                        // The footprint tracks the density. Only a vertex that took its own light
                        // sample weighs the hits along this direction against it; everywhere else
                        // they count in full.
                        pdfOut = lobe.pdf;
                        if (!inlineNee) res |= PV_MIS_NONE;
                    }
                    color *= W * cacheScale;
                    if (liteX2 && cacheSurface && (res & PV_CACHE_HIT) == 0u)
                    {
                        // The reuse suffix carries the broad part: all of a broad pick, none of another.
                        res |= PV_AUX_BROAD;
                        auxPk = PackRGB9E5(group == LOBE_GROUP_BROAD ? W : 0.0f);
                    }
                }
            }
        }
    }

    // Carry the medium through submerged object hits. Only a water interface
    // changes membership; reflection (including TIR) stays on the incident side.
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

// The primary vertex from the camera record: the camera pass resolved its surface (including
// primary surface replacement) and, with the cache enabled, wrote its geometric normal to the
// scratch slice. Without the cache nothing keys on the geometric normal and the shading normal
// stands in (the inspection views overwrite the slice earlier). The raygen reads this before any
// reorder, while its threads still cover a screen tile.
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

// A secondary hit the raygen found: evaluate the surface and the material into the context of the
// vertex. An emitter hit or the bounce limit resolves the vertex right here (io carries the
// result) and returns false; otherwise the raygen hands the context to PtVertexShade. prevPos and
// prevN are the vertex the ray left, for the MIS weight of an emitter hit against that vertex's
// light sample.
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

    // The beam width at this hit: the pixel cone over the path length, plus the spread of the
    // rough scatters so far and of the one that led here.
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
        // Suppress only emitters that the preceding water NEE can sample. Keep the
        // geometry as an occluder and retain BSDF-hit emission when mesh NEE is disabled.
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
            // MIS against the deferred light sample is resolved by the light and material passes.
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
            // The hit competes with the light sample of the vertex the ray left, wherever that
            // vertex takes one on any of its lobe picks; without one it counts in full.
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
            // The first bounce parks the reuse point; its broad-lobe share is applied by the material pass.
            const float3 liteNy = dot(hinfo.geometricNormal, -rayDir) < 0.0f
                ? -hinfo.geometricNormal : hinfo.geometricNormal;
            LiteParkPointStore(pixelIdx, WorldToObjectPos(instID, hinfo.hitPos), instID,
                PackNormal(WorldToObjectNrm(instID, liteNy)), DvLoadScatterPdf(pixelIdx));
            presetOut |= PV_LITE_PARKED;
        }

        if ((io.flags & PV_IN_SPREAD) != 0u)
            io.spread += hitT * sqrt(min(16.0f, rcp(max(io.pdf * abs(dot(hinfo.geometricNormal, -rayDir)), 1e-6f))));
        // The cone of the path's lobes widens over the segment by its angle.
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

// Escaped ray: sky and sun radiance with the sun MIS weight of the scatter the ray came from.
// The raygen owns the throughput and the reuse candidate. rayOrigin only places the sky
// observer, so the vertex the ray left stands in for the offset origin.
void PtShadeMiss(inout PtVertexIO io, float3 rayOrigin, float3 rayDir)
{
    const bool underground = WorldPosIsUnderground(rayOrigin + sceneOriginWorld);
    // A camera below sea level can be underground to the atmosphere model even
    // after its path exits water. Query the sky from the escaped ray's origin.
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
