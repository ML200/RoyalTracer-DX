#pragma once
// Per-vertex shading of the trace raygen, run in its hit, miss and callable shaders.
#include "PtDefer_v8.hlsli"
#include "PtNee_v8.hlsli"

// The whole raygen <-> material interface, 64 bytes; ray and hit come from system values. Only the
// 36 input bytes go in; the raygen reads the outputs only where the result flags say they are set.
#define PT_IO  read(caller, closesthit, miss) : write(caller, closesthit, miss)
#define PT_OUT read(caller) : write(closesthit, miss)
struct [raypayload] PtVertexIO {
    uint   flags     : PT_IO;
    float  pdf       : PT_IO;   // in: previous scatter pdf; out: MIS/footprint pdf
    float  spread    : PT_IO;   // path footprint, in/out
    float  dist      : PT_IO;   // path length, in/out
    uint   cone      : PT_IO;   // lobe cone for the cache test, in/out (PtConePack)
    uint   dirPk     : PT_OUT;  // next direction; on a miss the escaped one
    uint   nPk       : PT_IO;   // in: previous vertex's shading normal; out: this vertex's
    float3 pos       : PT_IO;   // in: previous vertex; out: next ray's origin vertex (SSS exit included)
    float3 color     : PT_OUT;  // throughput multiplier (PV_CAPTURED), or sky radiance on a miss
    uint   auxPk     : PT_OUT;  // packed radiance or weight, per the result flags
    uint   neePk     : PT_OUT;  // water light sample, PvPackRadiance (PV_NEE); queue slot (PV_NEE_QUEUED)
    uint   neeLitePk : PT_OUT;  // the water sample's reuse-suffix share (broad only at the park)
};
#undef PT_IO
#undef PT_OUT

// Miss table order (Renderer::CreateShaderBindingTable).
static const uint PT_MISS_SKY     = 0u;  // Miss: sky and sun
static const uint PT_MISS_PRIMARY = 1u;  // PtPrimaryShade: the camera record's vertex
static const uint PT_MISS_PRIMARY_LEAN = 2u;  // PtPrimaryShadeLean: the same off the sea

// Callable table order (PtCallables_v8.hlsl): rare or heavy work kept out of the raygen and the
// material shaders, so their code and registers stay small. Other light samples run deferred
// (PtNee_v8.hlsli, Pass_pt_nee_v8).
static const uint PT_CALLABLE_LIGHT_PDF     = 0u;
static const uint PT_CALLABLE_SSS           = 1u;
static const uint PT_CALLABLE_WATER_SEGMENT = 2u;
static const uint PT_CALLABLE_NEE_WATER     = 3u;  // water light samples (PT_NEE_IN_WATER/WATER_DIRECT)

// Subsurface random walk (PtSss), 36 bytes.
struct PtSssIO {
    float3 pos;     // in: entry; out: exit
    uint   nPk;     // in: entry normal; out: exit normal (PackNormal)
    uint   matID;
    uint   seed;    // in: the walk's seed; out: 1 if it exited
    float3 weight;  // out: walk throughput
};

// One water segment of a path (PtWaterSegment), 24 bytes.
struct PtWaterSegmentIO {
    float3 origin;  // in: ray origin; out: segment transmittance
    uint   dirPk;   // in: ray direction (PackNormal); out: in-scattered radiance, PvPackRadiance
    float  hitDist; // ray distance to the hit, TMax on a miss
    uint   bits;    // depth | PT_WATER_SAMPLED
};
#define PT_WATER_SAMPLED (1u << 31u)  // sample the sun and lights in the segment

// Emitter MIS density of the previous vertex's light sample (PtLightPdf), 24 bytes.
struct PtLightPdfIO {
    float3 pos;      // the previous vertex
    uint   nPk;      // its shading normal (PackNormal)
    uint   lightID;  // in: the emitter record; out: area density (asuint)
    uint   instID;
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
// Water state only the full hit shader handles; Pass_pt_trace keeps it away from HitLean_v8.hlsl.
#define PV_IN_WATER_ANY     (PV_IN_WATER_DIRECT | PV_IN_WATER_MEDIUM | PV_IN_WATER_SCATTERED | PV_IN_SUN_OWNED)

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
#define PV_NEE              (1u << 17u)  // neePk/neeLitePk hold this in-place vertex's water light sample
#define PV_WATER_DIRECT     (1u << 18u)
#define PV_WATER_MEDIUM     (1u << 19u)
#define PV_WATER_SCATTERED  (1u << 20u)
#define PV_LITE_NEE         (1u << 21u)  // the deferred primary feeds the reuse reservoir (DVF_LITE_NEE)
#define PV_SUN_OWNED        (1u << 22u)
#define PV_LAST_NEXT        (1u << 23u)  // over the bounce budget, unresolved: the next vertex is emission only
#define PV_NEE_QUEUED       (1u << 24u)  // the light sample is deferred: neePk is its queue slot (PtNee_v8.hlsli)

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

// Callee side of PtLightPdfIO (PtCallables_v8.hlsl).
void PtLightPdfEval(inout PtLightPdfIO p)
{
    p.lightID = asuint(LT_Pdf_LightTree_Area(p.pos, UnpackNormal(p.nPk), p.lightID, p.instID,
        LTC_UseSurfaceLearning()));
}

// Callee side of PtSssIO (PtCallables_v8.hlsl).
void PtSssWalk(inout PtSssIO p)
{
    uint seed = p.seed;
    const SSSWalkResult w = SubsurfaceWalk(p.pos, UnpackNormal(p.nPk), p.matID, seed);
    p.pos    = w.exitPos;
    p.nPk    = PackNormal(w.exitNormal);
    p.seed   = w.valid ? 1u : 0u;
    p.weight = w.wTotal;
}

// Callee side of PtWaterSegmentIO (PtCallables_v8.hlsl): in-scatter and attenuation up to the hit.
void PtWaterSegmentShade(inout PtWaterSegmentIO p, uint2 pixel)
{
    const float3 origin = p.origin;
    const float3 dir    = UnpackNormal(p.dirPk);
    uint sVolume = RcBounceSeed(PtPathSeed(pixel, PT_SAMPLE_INDEX), p.bits & ~PT_WATER_SAMPLED, 0x57415452u);
    const float segment = OceanBoundaryDistance(origin, dir, p.hitDist);
    const float3 inScatter = OceanSegmentInScatter(origin, dir, segment, (p.bits & PT_WATER_SAMPLED) != 0u, sVolume);
    float3 sigmaA, sigmaS; float phaseG;
    OceanMediumCoefficients(sigmaA, sigmaS, phaseG);
    p.origin = OceanMediumTransmittance(sigmaA + sigmaS, segment);
    p.dirPk  = PvPackRadiance(inScatter);
}

// Caller side of a water light sample (PtNeeWater), packed like PtVertexIO.neePk / neeLitePk.
void PtVertexWaterNee(PtNeeIO p, out uint neePk, out uint neeLitePk)
{
    CallShader(PT_CALLABLE_NEE_WATER, p);
    neePk     = p.kdRG;
    neeLitePk = p.kdBPr;
}

// Caller side: area density of the previous vertex's light sample for this emitter.
float PtLightPdfArea(float3 prevPos, uint prevNPk, uint lightID, uint instID)
{
    PtLightPdfIO p;
    p.pos     = prevPos;
    p.nPk     = prevNPk;
    p.lightID = lightID;
    p.instID  = instID;
    CallShader(PT_CALLABLE_LIGHT_PDF, p);
    return asfloat(p.lightID);
}

// SSS walk, cache, lobe pick, capture, continuation; ordered to cut live state. primary: the camera
// record (a compile-time constant at both call sites; only it has depth-1 work).
void PtVertexShade(inout PtVertexIO io, HitContext ctx, float3 geoN, float3 dirIn, bool flipIOR,
    uint2 pixel, uint pixelIdx, uint presetOut, bool primary)
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
    bool   doNee  = false;            // a water light sample (PtNeeWater)
    uint   neeSlot = PT_NEE_NO_SLOT;  // a deferred one (Pass_pt_nee_v8)
    uint   neeBits = 0u;
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
            PtSssIO w;
            w.pos   = ctx.hitPos;
            w.nPk   = PackNormal(ctx.hitNormal);
            w.matID = ctx.matID;
            w.seed  = sSss;
            CallShader(PT_CALLABLE_SSS, w);
            if (w.seed == 0u) alive = false;
            else
            {
                const float3 exitN = UnpackNormal(w.nPk);
                color *= w.weight * (float3)ctx.hitLocalKd;
                ctx.hitPos         = w.pos;
                ctx.hitNormal      = exitN;
                ctx.backface       = false;
                ctx.hitLocalKd     = (half3)float3(1, 1, 1);
                ctx.hitLocalPr     = (half)1.0f;
                ctx.hitLocalPm     = (half)0.0f;
                ctx.iors           = (half2)float2(1.0f, 1.0f);
                ctx.mediumMatID    = MEDIUM_INVALID;
                ctx.absorptionTint = (half3)float3(1, 1, 1);
                geoN    = exitN;
                rayDir  = -exitN;
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
        // One query site: forced (SharcQueryForced) behind a deferred vertex, else drawn (SharcQueryDraws).
        if (accepted)
            hit = SharcQueryStochastic(SharcMakeSurface(ctx, geoN), pending && !inWater, sCache, cached);
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
    if (primary)
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
        // In-place NEE: water surface, in water, or where the cache can't answer. Water samples run here
        // (PtNeeWater; they may precede the deferred vertex); the rest, always after it, are queued for
        // Pass_pt_nee_v8. Without one (or with the queue full) MIS_NONE keeps light hits unweighted.
        const bool waterNee = !capture && (waterDirect || (neeBase && wide && inWater));
        if (!waterDirect && !inWater && pending && neeBase && wide && (!cacheSurface || cacheRefused))
            neeSlot = PtNeeQueueAppend();
        const bool inlineNee = waterNee || neeSlot != PT_NEE_NO_SLOT;
        // Over budget, stop where the deferred pass, a light sample or the cache resolved the vertex;
        // otherwise its direct light needs one more, emission-only segment (as at the depth limit).
        const bool overBudget  = !freeBounce && diffDepth >= (uint)pt_maxDiffuseBounces;
        const bool budgetBreak = overBudget && (capture || inlineNee || (res & PV_CACHE_HIT) != 0u);
        if (overBudget && !budgetBreak) res |= PV_LAST_NEXT;

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
            // Requested at the end, where little else is live.
            doNee = waterNee;
            neeBits = (blue ? PT_NEE_BLUE : 0u) | (liteX2 && cacheSurface ? PT_NEE_LITE_BROAD : 0u) |
                (depth >= 2u && liteVertex ? PT_NEE_LITE_PATH : 0u) | (inWater ? PT_NEE_IN_WATER : 0u) |
                (waterDirect ? PT_NEE_WATER_DIRECT : 0u);
            res |= waterNee ? PV_NEE | (inWater ? PV_SUN_OWNED : 0u) : PV_NEE_QUEUED;
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

            // One evaluation site (instruction fetch bound); a capture keeps only the density.
            const BrdfData lobe = EvaluateLobe(spPath, group, ctx.matID, n, n, dir, -rayDir,
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
                    // Pass-through keeps the previous vertex's MIS, MIS_NONE included.
                    if (passThrough) res |= PV_PASS_THROUGH | (misNoneIn ? PV_MIS_NONE : 0u);
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
    io.neePk     = 0u;
    io.neeLitePk = 0u;
    if (neeSlot != PT_NEE_NO_SLOT)
    {
        // The raygen adds the path's weights to the slot (PtNeeQueueStoreWeights).
        PtNeeQueueStoreRequest(neeSlot, PtNeeRequest(ctx, io.nPk, rayDir, group, groupP, depth, sample, neeBits),
            pixel);
        io.neePk = neeSlot;
    }
    else if (doNee)
        PtVertexWaterNee(PtNeeRequest(ctx, io.nPk, rayDir, group, groupP, depth, sample, neeBits), io.neePk,
            io.neeLitePk);
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
    float2 barycentrics, float hitT, float3 rayDir, float3 prevPos, uint prevNPk, uint pixelIdx,
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
        ? LightRecordRadiance(hinfo.lightID, abs(dot(hinfo.geometricNormal, rayDir)))
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
                const float lightPdfArea = PtLightPdfArea(prevPos, prevNPk, hinfo.lightID, instID);
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
    io.dirPk = PackNormal(rayDir);  // the miss record's direction, so the raygen need not keep it
    io.color = underground ? float3(0, 0, 0) : EvaluateSky(rayDir);
    io.auxPk = PvPackRadiance(sunRad);
}

// Shader entries (Hit_v8.hlsl, HitLean_v8.hlsl, Miss_v8.hlsl); one PtVertexShade site each.
// Lean: the water bits are known clear, so the water code compiles out.
void PtShadeHit(inout PtVertexIO io, uint instID, uint geometryIndex, uint primitiveIndex, float2 barycentrics,
    float hitT, float3 rayDir, uint2 pixel)
{
    if (PT_HIT_LEAN) io.flags &= ~PV_IN_WATER_ANY;
    const uint pixelIdx = MapPixelID(uint2(IMG_W, IMG_H), pixel);
    HitContext ctx;
    float3 geoN;
    bool   flipIOR;
    uint   presetOut;
    if (PtHitContext(io, instID, geometryIndex, primitiveIndex, barycentrics, hitT, rayDir, io.pos, io.nPk,
            pixelIdx, ctx, geoN, flipIOR, presetOut))
        PtVertexShade(io, ctx, geoN, rayDir, flipIOR, pixel, pixelIdx, presetOut, false);
}

// The camera record starts every path (Pass_pt_trace_v8: PvInputFlags(PtPsInit(s), ...), unit pdf, no
// spread or cone, never in water), so all but the sample index is restated as constants and it folds.
void PtShadePrimary(inout PtVertexIO io, float3 rayDir, uint2 pixel)
{
    // PtPsSample is 4 bits wide, so keep its top bit too (it lands on bit 17).
    io.flags  = PvInputFlags(PtPsInit(0u), false, false, false) | (io.flags & (0xFu << PV_IN_SAMPLE_SHIFT));
    io.pdf    = 1.0f;
    io.spread = 0.0f;
    io.cone   = 0u;
    const uint pixelIdx = MapPixelID(uint2(IMG_W, IMG_H), pixel);
    HitContext ctx;
    float3 geoN;
    bool   flipIOR;
    PtPrimaryContext(pixel, pixelIdx, ctx, geoN, flipIOR);
    PtVertexShade(io, ctx, geoN, rayDir, flipIOR, pixel, pixelIdx, 0u, true);
}
