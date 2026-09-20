#pragma once
// Per-vertex work of the split path tracer. The trace raygen reorders on the hit object and then
// shades every vertex here itself, including the primary one and the escaped rays; the
// closest-hit and miss shaders of the pipeline are empty. The path loop only handles the compact
// result: the next direction, the vertex normal for the ray offset, a throughput multiplier, one
// packed value and flags.
#include "PtDefer_v8.hlsli"

struct PtVertexIO {
    uint   flags;
    float  pdf;      // in: pdf of the previous scatter; out: updated MIS/footprint pdf
    float  spread;   // path footprint, in/out
    float  dist;     // path length, in/out: with the pixel cone it sizes the texture footprint
    uint   dirPk;    // out: next direction
    uint   nPk;      // out: shading normal of the vertex the next ray leaves
    float3 color;    // out: throughput multiplier (see PV_CAPTURED), or sky radiance on a miss
    uint   auxPk;    // out: packed radiance or weight, see the result flags
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

uint PvInputFlags(uint ps, bool pending, bool immediate, bool last)
{
    return (PtPsDepth(ps) << PV_IN_DEPTH_SHIFT) | (PtPsDiffDepth(ps) << PV_IN_DIFF_SHIFT) |
        (PtPsSample(ps) << PV_IN_SAMPLE_SHIFT) | (PtPsGuideDepth(ps) << PV_IN_GUIDE_SHIFT) |
        ((ps & PT_PS_SSS) != 0u ? PV_IN_SSS : 0u) | ((ps & PT_PS_LITE_VERTEX) != 0u ? PV_IN_LITE_VERTEX : 0u) |
        ((ps & PT_PS_SPREAD) != 0u ? PV_IN_SPREAD : 0u) | ((ps & PT_PS_MIS_NONE) != 0u ? PV_IN_MIS_NONE : 0u) |
        (pending ? PV_IN_PENDING : 0u) | (immediate ? PV_IN_IMMEDIATE : 0u) | (last ? PV_IN_LAST : 0u);
}

// Shade one vertex: subsurface walk, cache lookup, the lobe pick that classifies the vertex,
// capture of the first wide vertex, then the continuation. The phases are ordered so the inputs
// of each phase die before the next: the vertex record is written before the lobe is sampled,
// and the guide cones die before the BSDF evaluation.
//
// Whether a vertex is wide is decided per sample by the lobe it picks: a diffuse or rough pick
// defers the vertex with its light sample, a mirror-like pick bounces in place and leaves the
// deferral to the next wide vertex, so a reflection seen in a smooth surface is lit at the
// surface it shows. The light sample of a mixed vertex is thus taken only on the samples that
// pick a wide lobe, and the material pass weights it up by that share.
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
                DvStoreSssExit(pixelIdx, w.exitPos);
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
    // The subsurface exit is no surface the cache knows. ---
    const bool cacheSurface = sharc_enabled != 0u && !walked && SharcMaterialEligible(ctx, spPath, geoN);
    float cacheScale = 1.0f;
    if (alive && cacheSurface && depth > 1u)
    {
        uint sCache = RcBounceSeed(pathSeed, depth, 0x53484152u);
        bool queryAccepted;
        if (liteX2)
        {
            const float ramp = SharcFootprintRamp(io.spread, (uint)SharcLevel(ctx.hitPos));
            queryAccepted = io.spread > 0.0f && ramp > 0.0f && RandomFloatSingle(sCache) < ramp;
        }
        else
            queryAccepted = io.spread > 0.0f && SharcQueryFootprintAccepted(ctx.hitPos, io.spread, sCache);
        float3 cached = 0.0f;
        bool hit = queryAccepted && SharcQueryDraws(SharcMakeSurface(ctx, geoN), sCache, cached);
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

    // --- the lobe of this sample: a wide pick makes the vertex wide ---
    const bool  broadGGX  = IsBroadGGX(ctx.hitLocalPr);
    const float uStrategy = blue ? PtBlue1(pixel, blueIndex, depth, BN_DIM_STRATEGY) : RandomFloatSingle(sBsdf);
    const uint  strategy  = SelectSamplingStrategyFrom(spPath, uStrategy);
    const bool  wide      = SharcScatterHasSpread(strategy, ctx.matID, ctx.hitLocalPr);
    if (depth == 1u)
    {
        liteVertex = LITE_ENABLED && !sssEntered && wide &&
            HasBroadShare(spPath, ctx.hitLocalPr, ctx.hitLocalPm) &&
            !LoadIsSSS(ctx.matID) && ctx.mediumMatID == MEDIUM_INVALID &&
            LoadKd_w(ctx.matID) >= EPSILON && !freeBounce;
        if (liteVertex) res |= PV_PRIMARY_LITE;
    }

    if (alive)
    {
        // --- classification: a narrow pick bounces in place, a wide one is deferred unless a
        // deferred vertex already exists ---
        const bool neeBase = ctx.mediumMatID == MEDIUM_INVALID &&
            (LoadKd_w(ctx.matID) >= EPSILON || (float)ctx.hitLocalPr >= SMOOTH_SPECULAR_THRESHOLD);
        const bool performNEE  = neeBase && wide;
        const bool capture     = wide && !pending;
        const bool budgetBreak = !freeBounce && diffDepth >= (uint)pt_maxDiffuseBounces;

        if (capture)
        {
            // Defer this vertex: the light and material passes finish it. Written first so its
            // inputs are not kept alive through the sampling below. The record carries the share
            // of the lobe picks that defer here, since only those take the light sample.
            const float narrowShare = ((float)ctx.hitLocalPr < SMOOTH_SPECULAR_THRESHOLD ? spPath.Pspec : 0.0f) +
                (LoadPcr(ctx.matID) < SMOOTH_SPECULAR_THRESHOLD ? spPath.Pcoat : 0.0f);
            DvVertex dv;
            dv.pos = ctx.hitPos; dv.n = n; dv.dirIn = rayDir;
            dv.matID = ctx.matID; dv.instID = ctx.instID; dv.Kd = (float3)ctx.hitLocalKd;
            dv.Pr = ctx.hitLocalPr; dv.Pm = ctx.hitLocalPm; dv.sp = spPath; dv.absorb = absorb;
            dv.flags = (ctx.backface ? DVF_BACKFACE : 0u) | (flipIOR ? DVF_FLIP_IOR : 0u) |
                (performNEE ? DVF_PERFORM_NEE : 0u) | (walked ? DVF_UNIT_IOR : 0u) |
                (DvPackWideShare(1.0f - narrowShare) << DVF_WIDE_SHIFT);
            DvStoreVertex(pixelIdx, dv);
            color *= cacheScale;
            cacheScale = 1.0f;
            res |= PV_CAPTURED;
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
            // evaluate the mixture once. The cones are built after the lobe sample, so they overlap
            // neither the samplers nor the evaluation below.
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
            if (cacheSurface && GUIDE_ENABLED && HasBroadShare(spPath, ctx.hitLocalPr, ctx.hitLocalPm) &&
                1u + guideDepth <= (uint)GUIDE_MAX_DEPTH)
            {
                uint sKey = RcBounceSeed(pathSeed, depth, 0x4b455953u);
                const GuideSet guide = GuideBuildAt(GuideKeyOf(ctx.hitPos, geoN, sKey), ctx.hitPos, n);
                if (guide.q > 0.0f)
                {
                    guideQ = guide.q;
                    if (strategy == 0u || (strategy == 1u && broadGGX))
                    {
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
                    }
                    guidePdf = GuidePdf(guide, dir);   // the cones are dead from here on
                }
            }
            res |= strategy << PV_STRATEGY_SHIFT;
            if (!freeBounce) res |= PV_DIFF_INC;
            if (wide) res |= PV_SPREAD;

            float3 broadVal; float broadPdf;
            const BrdfData bdata = EvaluateAndPdf_COMBINED_L(spPath, LOBE_BROAD, ctx.matID, n, n, dir, -rayDir,
                ctx.hitLocalKd, ctx.hitLocalPr, ctx.hitLocalPm, ctx.iors.x, ctx.iors.y, false, broadVal, broadPdf);
            const float pShare   = spPath.Pdiff + (broadGGX ? spPath.Pspec : 0.0f);
            const float pdfTotal = guideQ > 0.0f ? max(bdata.pdf + pShare * guideQ * (guidePdf - broadPdf), 0.0f) : bdata.pdf;
            const bool  valid    = dot(dir, dir) > 1e-12f && pdfTotal > 1e-6f;
            const bool  passThrough = strategy == 1u && dot(dir, n) < 0.0f && LoadKd_w(ctx.matID) < 1.0f - EPSILON;

            if (capture)
            {
                // Densities only: the material pass supplies the value of this scatter.
                DvScatter dsc;
                dsc.dirOut = dir; dsc.info = 0u; dsc.pdfTotal = pdfTotal; dsc.bsdfPdf = bdata.pdf;
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
                    else if (performNEE) { pdfOut = bdata.pdf; misNone = false; }
                    else misNone = true;
                    res |= (DV_KIND_BSDF << PV_KIND_SHIFT) | (misNone ? PV_MIS_NONE : 0u);
                }
            }
            else
            {
                // In-place vertex: narrow lobes before the deferred vertex, or anything after it.
                const float  cosTheta = abs(dot(n, dir));
                const float3 W = valid ? bdata.val * absorb * cosTheta / pdfTotal : float3(0, 0, 0);
                if (!valid || any(isnan(W)) || any(isinf(W)) || !any(W > 0.0f)) alive = false;
                else
                {
                    if (passThrough) res |= PV_PASS_THROUGH;
                    else
                    {
                        // No light sample competes here, but the footprint still tracks the density.
                        if (neeBase) pdfOut = bdata.pdf;
                        res |= PV_MIS_NONE;
                    }
                    color *= W * cacheScale;
                    if (liteX2 && cacheSurface && (res & PV_CACHE_HIT) == 0u)
                    {
                        res |= PV_AUX_BROAD;
                        auxPk = PackRGB9E5(broadVal * absorb * cosTheta / pdfTotal);
                    }
                }
            }
        }
    }

    if (!alive) res |= PV_TERMINATE;
    io.flags = res;
    io.pdf   = pdfOut;
    io.dirPk = PackNormal(dir);
    io.nPk   = PackNormal(outN);
    io.color = color;
    io.auxPk = auxPk;
}

// Shade the hit the raygen found. The primary vertex takes its surface from the camera record;
// every other vertex evaluates the surface and material here. Both feed the vertex routine.
void PtShadeHit(inout PtVertexIO io, uint instID, uint geometryIndex, uint primitiveIndex,
    float2 barycentrics, float hitT, float3 rayDir, uint2 pixel, uint pixelIdx)
{
    const uint   depth    = (io.flags >> PV_IN_DEPTH_SHIFT) & 0x7Fu;
    const uint primID = FlatPrimID(instID, geometryIndex, primitiveIndex);

    HitContext ctx = (HitContext)0;
    float3 geoN      = 0.0f;
    bool   flipIOR   = false;
    uint   presetOut = 0u;
    bool   shade     = true;

    if (depth == 1u)
    {
        // The camera pass resolved the primary surface (including primary surface replacement);
        // only the geometric normal comes from the retraced triangle.
        const SDRecord sd = load_SD(g_sample_current, pixelIdx);
        float2 pIors; uint pMedium; float3 pAbsorb;
        load_rg_primaryExtra(pixelIdx, pIors, pMedium, pAbsorb);
        geoN = CandidateGeoNormalW(instID, primID);
        if (dot(geoN, sd.n1_s) < 0.0f) geoN = -geoN;

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
        flipIOR = pMedium != MEDIUM_INVALID;
    }
    else
    {
        const uint  matID = GetMatIDFast(instID, primID);
        // The beam width at this hit: the pixel cone over the path length, plus the spread of the
        // rough scatters so far and of the one that led here.
        const float spreadHere = (io.flags & PV_IN_SPREAD) != 0u
            ? hitT * sqrt(min(16.0f, rcp(max(io.pdf, 1e-6f)))) : 0.0f;
        const float footprint = PixelConeAngle() * (io.dist + hitT) + io.spread + spreadHere;
        io.dist += hitT;
        const HitInfo hinfo = EvalSurfaceStateDir(instID, primID, barycentrics, rayDir, footprint);
        const float3  emission = (hinfo.lightID != 0xFFFFFFFFu)
            ? g_EmissiveTriangles[hinfo.lightID].emission * GLOBAL_EMISSION_STRENGTH
            : float3(0, 0, 0);
        if (any(emission > 0.0f))
        {
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
                io.flags = PV_EMITTER;
                io.auxPk = PvPackRadiance(emission);
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
            RefetchMaterial(matID, hinfo.uv, hitLocalKd, hitLocalPr, hitLocalPm, hinfo.uvFootprint);

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
            ctx.absorptionTint = (half3)(flipIOR ? CalculateAbsorptionThroughput(LoadTf(matID), hitT) : float3(1, 1, 1));
        }
    }

    if (shade) PtVertexShade(io, ctx, geoN, rayDir, flipIOR, pixel, pixelIdx, presetOut);
}

// Escaped ray: sky and sun radiance with the sun MIS weight of the scatter the ray came from.
// The raygen owns the throughput and the reuse candidate.
void PtShadeMiss(inout PtVertexIO io, float3 rayOrigin, float3 rayDir)
{
    const bool underground = WorldPosIsUnderground(rayOrigin + sceneOriginWorld);
    SetSkyObserver(InitOrigin() + sceneOriginWorld);
    const float  sunSAPdf   = underground ? 0.0f : GetSunPdf(rayDir);
    const float3 sunRad     = (sunSAPdf > 0.0f) ? EvaluateSun(rayDir) : float3(0, 0, 0);
    const float  sunMisBsdf = (sunSAPdf > 0.0f)
        ? ((io.flags & PV_IN_MIS_NONE) != 0u ? 1.0f : io.pdf / max(io.pdf + sunSAPdf, EPSILON)) : 0.0f;

    io.flags = PV_MISS;
    io.pdf   = sunSAPdf > 0.0f ? sunMisBsdf : 1.0f;
    io.color = underground ? float3(0, 0, 0) : EvaluateSky(rayDir);
    io.auxPk = PvPackRadiance(sunRad);
}
