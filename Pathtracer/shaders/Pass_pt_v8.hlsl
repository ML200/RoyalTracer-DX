// PT only reads the primary extras after the camera barrier; it does not use
// ReSTIR's scratch/reload scheme that requires globally coherent accesses.
#define SPMIS_GRID_NONCOHERENT
// Per-frame sun state and sky-view LUT (Pass_pt_skybake_v8.hlsl, SkyBake_v8.hlsli):
// ComputeSunState and the bounce-miss atmosphere read them instead of marching
// the atmosphere per call. Every miss and sun sample here uses the camera observer.
#define SKYBAKE_CONSUMER 1
#ifndef PT_NO_DEBUG
#define PT_NO_DEBUG 0
#endif
#include "Includes_v8.hlsli"
// The host selects this specialization only with cache debug views off.
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
#define SHARC_TRAINING_SPILL // per-vertex training radiance/weights live in the path-state buffer
#endif
#include "SharcPath_v8.hlsli"
#include "SharcGuide_v8.hlsli"
#if !SHARC_UPDATE_PASS
#include "SharcDebug_v8.hlsli"
#include "RestirLite_v8.hlsli"
#endif


uint2 PtPixel()
{
    uint2 pixel = DispatchRaysIndex().xy;
#if SHARC_UPDATE_PASS
    // A per-tile permutation visits every pixel, including partial edge tiles.
    // Separate streams from rendering avoid training/query sample correlation.
    uint tileHash = Hash32(pixel.x ^ Hash32(pixel.y));
    uint phase = (sharc_frame + tileHash) % (sharc_updateStride * sharc_updateStride);
    pixel = pixel * sharc_updateStride + uint2(phase % sharc_updateStride, phase / sharc_updateStride);
#endif
    return pixel;
}

// Radiance estimate accumulation (a register: memory round trips at every
// event cost more than the 12 live bytes; the training pass never reads it).
#define PtAccumulate(value) (total += (value))

// Prepare a vertex before its one trace reorder. Cache termination and lobe
// roulette happen here, so only surviving lanes enter the expensive NEE work.
bool PtPrepareVertex(HitContext ctx, float3 geometricNormal, float3 rayDir,
    bool sssEntered, uint pathSeed, uint depth, uint maxBounces, float pathSpread,
    inout SamplingP spPath, out bool cacheSurface, out bool diffuseCached, inout float3 throughput,
#if SHARC_UPDATE_PASS
    inout SharcTrainingState training
#else
    bool liteX2, bool litePath, inout float3 total, inout float3 liteL,
    inout half3 liteSuffix
#endif
)
{
    diffuseCached = false;
    // Cache first, then local NEE. A record stores the BROAD share's
    // outgoing radiance at this vertex (the diffuse lobe and a broad GGX
    // lobe, their own direct light included), demodulated by Kd and by the
    // view-dependent transmission of the layers above them. A hit adds that
    // share and drops those lobes from this vertex; sheen, coat and a glossy
    // GGX lobe continue on the exact tracer with their own strategy mix, so
    // nothing narrow is ever cached. The primary vertex is never queried.
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
        // Cache resampling: once a training path has registered a vertex,
        // a confident record at the next eligible vertex supplies that
        // vertex's diffuse share exactly as it would for a rendered path.
        // Records keep being re-measured by the paths that do not resample
        // (the query's confidence, footprint and 1/32 always-trace gates).
        hit = queryable && training.count > 0u &&
            SharcQuery(surface, pathSpread, sCache, cached);
        // Last vertex before the depth cap: the loop would trace once more
        // and drop the hit, training the suffix as darkness. Any resolved
        // record for this vertex is an unbiased stand-in for that tail.
        if (!hit && training.count > 0u && (uint)depth + 1u >= maxBounces)
            hit = SharcQueryForced(surface, sCache, cached);
        if (hit)
            SharcTrainingRadiance(training, cached * layerT);
        // Register BEFORE this vertex's NEE so the new record receives its
        // own diffuse direct light. Leave at least half the path budget
        // behind every training vertex; never train a capped tail as
        // darkness. Views through a nearly opaque layer stack would need
        // huge demodulation weights; other paths cover that record.
        else if ((uint)depth < maxBounces / 2u && layerT >= 0.2f)
            SharcTrainingVertex(training, surface, layerT,
                RcBounceSeed(pathSeed, (uint)depth, 0x53504c54u));
#else
        // Share the reconstruction after the two acceptance policies.
        // Each policy consumes the same cache RNG draws as before.
        bool queryAccepted;
        if (liteX2)
        {
            // First bounce of a lite path: the tracer's footprint ramp
            // without its always-trace share (a miss only means the
            // candidate's radiance is path traced instead of read).
            const float ramp = SharcFootprintRamp(pathSpread, (uint)SharcLevel(ctx.hitPos));
            queryAccepted = pathSpread > 0.0f && ramp > 0.0f && RandomFloatSingle(sCache) < ramp;
        }
        else
            queryAccepted = pathSpread > 0.0f &&
                SharcQueryFootprintAccepted(surface.position, pathSpread, sCache);
        hit = queryAccepted && SharcQueryDraws(surface, sCache, cached);
        // Misses and small footprints do not need layer demodulation.
        // This uses only the cache RNG stream, so deferring the layer
        // test preserves all NEE/BSDF draws and accepted contributions.
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
                // The record ends the diffuse suffix here: it holds this
                // vertex's direct light already, so no NEE follows for
                // it. x1's other lobes (the throughput) take the value
                // and may carry on through this vertex's layers.
                liteL += (float3)liteSuffix * cached * layerT;
                liteSuffix = (half3)0.0f;
                if (!any(throughput > 0.0f)) return false;
            }
        }
#endif
        if (hit)
        {
            // The broad share is accounted for; continue with whatever
            // layers remain. A Lambertian or rough-metal vertex ends here.
            diffuseCached = true;
            if (!DropBroadLobes(spPath, ctx.hitLocalPr)) return false;
            // The remaining layers carry about 1 - layerT of this vertex's
            // energy (the Fresnel of the stack). Roulette them BEFORE their
            // NEE and scatter: a hit on a plain dielectric then ends here
            // ~96% of the time instead of paying a full specular vertex
            // for a 4% term. Survivors are compensated, so nothing is lost.
            const float pSpecular = clamp(1.0f - layerT, 0.02f, 1.0f);
            if (RandomFloatSingle(sCache) >= pSpecular) return false;
            throughput *= rcp(pSpecular);
#if SHARC_UPDATE_PASS
            SharcTrainingScatter(training, rcp(pSpecular));
#endif
        }
    }
    return true;
}

//====================================
//PACKED LOOP STATE
//====================================
// The sample index, the bounce depth, the diffuse-bounce count and the path
// flags ride the trace and the reorder in ONE register. As separate variables
// each was its own 32-bit live value (DXC keeps a widened copy of an int16
// depth, and every bool is an i32).
#define PT_PS_DEPTH_SHIFT   0u
#define PT_PS_DIFF_SHIFT    8u
#define PT_PS_SAMPLE_SHIFT  16u
#define PT_PS_SSS           (1u << 24u)   // the path took a subsurface walk
#define PT_PS_LITE_VERTEX   (1u << 25u)   // the primary hands its diffuse lobe to the lite reservoir
#define PT_PS_LITE_X2       (1u << 26u)   // the lite candidate has its second vertex
#define PT_PS_DIFF_CACHED   (1u << 27u)   // the cache answered the vertex's diffuse lobe
#define PT_PS_FLIP_IOR      (1u << 28u)   // the vertex carried across the reorder enters its medium
#define PT_PS_SPREAD        (1u << 29u)   // the scatter grows the path spread
uint PtPsInit(uint s)       { return (s << PT_PS_SAMPLE_SHIFT) | (1u << PT_PS_DEPTH_SHIFT); }
uint PtPsDepth(uint ps)     { return (ps >> PT_PS_DEPTH_SHIFT) & 0xFFu; }
uint PtPsDiffDepth(uint ps) { return (ps >> PT_PS_DIFF_SHIFT) & 0xFFu; }
uint PtPsSample(uint ps)    { return (ps >> PT_PS_SAMPLE_SHIFT) & 0xFFu; }
uint PtPsWith(uint ps, uint flag, bool on) { return on ? (ps | flag) : (ps & ~flag); }

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
                dctx.hitNormal, dctx.iors.x, dctx.iors.y, dctx.hitLocalKd, dctx.hitLocalPm);
            if (!SharcMaterialEligible(dctx, dsp, dGeometricNormal))
            {
                // Slate: the tracer never caches this surface (no broad lobe:
                // a glossy GGX lobe without a diffuse share, transmitting, SSS,
                // inside a medium, or a steep normal map). No record is
                // expected here, so this is not a missing cell.
                debugColor = float4(0.05f, 0.07f, 0.12f, 0.0f);
            }
            else if (SHARC_DEBUG_MODE == SHARC_DEBUG_GUIDING)
            {
                // Targets by default; the level toggle switches to the receiver view.
                debugColor = GuideDebugColor(primary.x1, dGeometricNormal, dctx.hitNormal,
                    (sharc_enabled & SHARC_DEBUG_OTHER_LEVEL_BIT) != 0u);
            }
            else
            {
                SharcSurface surface = SharcMakeSurface(dctx, dGeometricNormal);
                float lod = SharcLevel(primary.x1);
                // Show the level most rendering queries sample at this distance;
                // the toggle shows the other one. Training feeds both levels in
                // proportion to their query probability.
                uint level = (uint)(lod + 0.5f);
                if ((sharc_enabled & SHARC_DEBUG_OTHER_LEVEL_BIT) != 0u)
                    level = frac(lod) >= 0.5f ? level - 1u : level + 1u;
                level = min(level, SHARC_MAX_LEVEL - 1u);
                debugColor = SharcDebugColor(surface, level, SHARC_DEBUG_MODE);
            }
        }
        // Keep regular rendering/training running so toggling this view neither
        // changes exposure nor the samples, camera guides or cache contents.
        gScratchPing[uint3(pixel, SHARC_DEBUG_SCRATCH)] = debugColor;
    }
#endif

    //terminal primaries (sky / degenerate / direct emitter) were finalized by
    //Pass_camera; their scratch slots must stay untouched.
    if (load_flagsWord(g_sample_current, pixelIdx) & SD_FLAG_NOBOUNCE) return;

    //sky/sun observer at the camera altitude (bounce misses keep it: the
    //vertex they left is not live across the trace, and metres do not matter)
    SetSkyObserver(InitOrigin() + sceneOriginWorld);

    float3 total = float3(0, 0, 0);

    const uint N = SHARC_UPDATE_PASS ? 1u : max(pt_initialSamples, 1u);
    // Cache misses render with the ordinary budget; only training paths use the
    // cache depth cap. A miss path does not train anything, so tracing it deeper
    // than the reference only costs time.
    const uint maxBounces = SHARC_UPDATE_PASS ? sharc_trainBounces : pt_maxBounces;
    //The loops know the pixel by its swizzled index alone. DispatchRaysIndex
    //is pure, so every PtPixel() call in a loop folds into ONE value that
    //would ride the trace and the reorder next to the index; a sample rebuilds
    //the 2D pixel from the index where it needs it (the seed, the store) and
    //still reloads the primary record.
    uint ps = PtPsInit(0u);
    [loop]
    for (;;)
    {
        if (PtPsSample(ps) >= N) break;
        const uint  sampleIdx   = pixelIdx;
        const uint2 samplePixel = (uint2)UnmapPixelID(sampleIdx, imgSize);
        //same seed derivation as Pass_raygen so the two integrators sample
        //identical per-bounce streams (RcBounceSeed) at N==1
        uint seed     = initRandomData(samplePixel, uint2(8, 4), time, PtPsSample(ps) + 1u);
        uint pathSeed = Hash32(seed ^ 0x9E3779B9u);
#if SHARC_UPDATE_PASS
        pathSeed = Hash32(pathSeed ^ sharc_frame ^ 0x53484152u);
        SharcTrainingState training;
        SharcTrainingInit(training, sampleIdx);
#endif

        //primary vertex z1 from the G-buffer + HOT2 extras (raygen's rebuild)
        const SDRecord sd = load_SD(g_sample_current, sampleIdx);
        float2 pIors; uint pMedium; float3 pAbsorb;
        load_rg_primaryExtra(g_pathStateBuffer, sampleIdx, pIors, pMedium, pAbsorb);

        //rebuild the shared primary ctx per sample (fields must not stay live
        //across the bounce traces — raygen's rationale)
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
        float   prev_pdf     = 1.0f;   //solid-angle pdf of the last BSDF extension (MIS partner)
        uint    rayDirPk     = PackNormal(normalize(sd.x1 - InitOrigin()));
        float3 geometricNormal = sd.n1_s;
#if SHARC_UPDATE_PASS
        geometricNormal = gScratchPing[uint3(samplePixel, SHARC_DEBUG_SCRATCH)].xyz;
#else
        // Rendering keys guiding on the same geometric normal as training. A
        // debug view has already replaced that slot with its image by now.
        if (sharc_enabled != 0u && SHARC_DEBUG_MODE == 0u)
            geometricNormal = gScratchPing[uint3(samplePixel, SHARC_DEBUG_SCRATCH)].xyz;
#endif
        float pathSpread = 0.0f;
#if !SHARC_UPDATE_PASS
        // ReSTIR lite (RestirLite_v8.hlsli): at the primary vertex the BROAD
        // share's incident light (diffuse lobe and broad GGX lobe) is streamed
        // into the pixel's reservoir instead of the radiance sum; the lite
        // passes resample and shade it. Once the scatter ray has found x2, the
        // whole broad-share suffix (cache value or traced continuation)
        // accumulates into that candidate's radiance, while x1's other lobes
        // keep their share of the path on the tracer.
        float3 liteL      = 0.0f;        // outgoing radiance of x2 toward x1 (diffuse suffix)
        half3  liteSuffix = (half3)0.0f; // diffuse suffix throughput from x2
        float  liteRr     = 0.0f;        // luminance of x1's diffuse scatter weight (the suffix's roulette share)
#endif

        [loop]
        for (;;)
        {
            const uint depth = PtPsDepth(ps);
            if (depth >= maxBounces) break;
            const uint s          = PtPsSample(ps);
            const bool sssEntered = (ps & PT_PS_SSS) != 0u;
            float3 rayDir = UnpackNormal(rayDirPk);

            //per-bounce RNG streams (draw-count independent, same ids as raygen)
            uint sNee  = RcBounceSeed(pathSeed, (uint)depth, RC_STREAM_NEE);
            uint sBsdf = RcBounceSeed(pathSeed, (uint)depth, RC_STREAM_BSDF);
            uint sSss  = RcBounceSeed(pathSeed, (uint)depth, RC_STREAM_SSS);

            //strategy probabilities once per bounce, shared by NEE + BSDF. A cache
            //hit removes the diffuse lobe from this mix for the rest of the vertex.
            SamplingP spPath = CalculateStrategyProbabilities(ctx.matID, -rayDir, ctx.hitNormal, ctx.iors.x, ctx.iors.y, ctx.hitLocalKd, ctx.hitLocalPm);
#if !SHARC_UPDATE_PASS
            if (depth == 1u)
            {
                // A primary with a broad share that NEE can serve (no medium,
                // no SSS) hands that share to the reservoir. Every other pixel
                // marks its reservoir empty so no pass reads a stale one.
                // A free-bounce material never counts as a diffuse vertex, so
                // its scatter could never end in the cache: keep it off the
                // reservoir, or partners would hand it points it cannot produce.
                const bool primaryLite = LITE_ENABLED && !sssEntered &&
                    HasBroadShare(spPath, ctx.hitLocalPr, ctx.hitLocalPm) &&
                    !LoadIsSSS(ctx.matID) && ctx.mediumMatID == MEDIUM_INVALID &&
                    LoadKd_w(ctx.matID) >= EPSILON && !MaterialIsFreeBounce(ctx.matID);
                ps = PtPsWith(ps, PT_PS_LITE_VERTEX, primaryLite);
                if (LITE_ENABLED && s == 0u && !primaryLite) LiteMarkEmpty(sampleIdx);
            }
            // Candidates are generated at the primary; from the secondary vertex
            // on, the path is the suffix of the pending point candidate.
            const bool liteVertex = (ps & PT_PS_LITE_VERTEX) != 0u;
            const bool liteGen  = depth == 1u && liteVertex;
            const bool liteX2   = depth == 2u && liteVertex;
            const bool litePath = depth >= 2u && liteVertex;
#endif

            // Rebuild the small strategy mix after SER instead of carrying its
            // four floats. Eligibility uses the full mix, before removing the
            // broad share already accounted for at the preceding trace.
            bool cacheSurface = sharc_enabled != 0u && !sssEntered &&
                SharcMaterialEligible(ctx, spPath, geometricNormal);
            bool diffuseCached = (ps & PT_PS_DIFF_CACHED) != 0u;
#if SHARC_UPDATE_PASS
            if (depth == 1u && !PtPrepareVertex(ctx, geometricNormal, rayDir, sssEntered,
                pathSeed, 1u, maxBounces, pathSpread, spPath, cacheSurface, diffuseCached,
                throughput, training)) break;
#endif
            if (diffuseCached) DropBroadLobes(spPath, ctx.hitLocalPr);

#if !SHARC_UPDATE_PASS
            // Pre-trace candidates (NEE) resample in registers and commit once;
            // later initial samples continue the reservoir from memory.
            LiteGen liteG = LiteGenEmpty();
            if (liteGen && s != 0u) liteG = LiteGenLoad(sampleIdx);
#endif
            //------------- NEE: light-tree point light + sun (MIS vs BSDF) -------------
            const bool performNEE = !(ctx.mediumMatID != MEDIUM_INVALID || LoadKd_w(ctx.matID) < EPSILON);
            if (performNEE)
            {
                float3 directSum = float3(0, 0, 0);
#if SHARC_UPDATE_PASS
                float3 directBroad = float3(0, 0, 0);
#endif
                [loop]
                for (uint tech = 0u; tech < 2u; ++tech)
                {
                    float3 L;
                    float3 visTarget;
                    float3 visTargetN;
                    float3 radiance;
                    float  lightPdf;
                    float  cosSurf;

#if !SHARC_UPDATE_PASS
                    // light-tree sample fields the lite candidate needs after the shadow ray
                    uint   liteObj   = LITE_INF;
                    float3 litePos   = 0.0f;
                    float3 liteNrm   = 0.0f;
                    float  liteDistL = RAY_TMAX_PLANET;
                    float  liteCosL  = 1.0f;
#endif
                    if (tech == 0u)
                    {
                        // Initial sample 0 of the primary vertex: the descent ran in
                        // Pass_pt_nee_v8; continue its NEE stream from the record
                        // (Path_State_v8.hlsli). Deeper vertices, further initial
                        // samples and pixels without a record descend inline.
                        LT_Sample treeSample;
                        bool prefetched = false;
#if !SHARC_UPDATE_PASS
                        prefetched = depth == 1u && s == 0u &&
                            load_pt_neePrefetch(g_pathStateBuffer, sampleIdx, treeSample.id, treeSample.pdf, sNee);
#endif
                        if (!prefetched) treeSample = LT_SampleLight(ctx.hitPos, ctx.hitNormal, sNee);
                        LT_LightSampleResult light = LT_SamplePointOnLightTree(ctx.hitPos, treeSample, sNee);

                        const float3 toLight = light.position - ctx.hitPos;
                        const float  dist    = sqrt(dot(toLight, toLight));
                        L = toLight / dist;

                        cosSurf = dot(ctx.hitNormal, L);
                        const float cosLightS = dot(light.normal, -L);
                        if (!(cosSurf > 1e-6f && cosLightS > 1e-6f && light.pdfSolidAngle > 1e-20f))
                            continue;

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
                    else
                    {
                        const float2 rSun = float2(RandomFloatSingle(sNee), RandomFloatSingle(sNee));
                        SunSampleResult sun = SampleSun(rSun, ctx.hitPos + sceneOriginWorld);
                        L = sun.direction;

                        cosSurf = dot(ctx.hitNormal, L);
                        if (!(cosSurf > 1e-6f && sun.pdf > 1e-20f))
                            continue;

                        visTarget  = ctx.hitPos + sun.direction * RAY_TMAX_PLANET;
                        visTargetN = -sun.direction;
                        radiance   = sun.radiance;
                        lightPdf   = sun.pdf;
                    }

                    //visibility first (keeps the BSDF eval out of the occluded
                    //lanes); thin glass attenuates instead of blocking
                    const float3 visT = VisibilityTransmittance(ctx.hitPos, ctx.hitNormal, visTarget, visTargetN);
                    if (!any(visT > 0.0f))
                        continue;


                    // One traversal yields the full BSDF and the broad share
                    // (LOBE_BROAD): training labels fresh records with the share,
                    // and ReSTIR lite streams the primary's share into its reservoir.
                    float3 broadNEE; float broadNeePdf;
                    BrdfData bdataNEE = EvaluateAndPdf_COMBINED_L(spPath, LOBE_BROAD, ctx.matID, ctx.hitNormal, ctx.hitNormal, L, -rayDir,
                        ctx.hitLocalKd, ctx.hitLocalPr, ctx.hitLocalPm, ctx.iors.x, ctx.iors.y, false, broadNEE, broadNeePdf);
                    if (!(bdataNEE.pdf > 0.0f))
                        continue;

                    // MIS partner: the DECLARED continuation pdf, i.e. the plain
                    // BSDF pdf even with guiding on. Guiding only changes the
                    // diffuse lobe's actual sampling density; the emitter and sun
                    // hits below weight with the same declared pdf (prev_pdf), and
                    // any positive pdf used consistently by both techniques keeps
                    // the balance heuristic unbiased. Declaring the unguided pdf
                    // keeps the cone set out of the NEE live state and off the
                    // shadow-ray path; guided cones aim at cached surfaces, not
                    // emitters, so the weights lose almost nothing.
                    const float  misWeight  = lightPdf / (lightPdf + bdataNEE.pdf);
                    const float3 lightScale = radiance * cosSurf * visT * (misWeight / lightPdf);
                    const float3 direct     = bdataNEE.val * lightScale;
#if !SHARC_UPDATE_PASS
                    if (liteGen)
                    {
                        // The diffuse lobe's share becomes a reservoir candidate
                        // (visibility, MIS weight and light pdf ride its weight);
                        // the other lobes keep their direct light here.
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
                        // Suffix of a lite candidate: at an eligible x2 only its
                        // diffuse lobe (its layers are dropped exactly as a cache
                        // hit drops them), deeper or elsewhere the full BSDF.
                        if (litePath)
                            liteL += (float3)liteSuffix * ((liteX2 && cacheSurface) ? broadNEE : bdataNEE.val) * lightScale;
                    }
#else
                    PtAccumulate(throughput * direct);
                    directSum += direct;
#endif
#if SHARC_UPDATE_PASS
                    // A vertex registered this bounce labels its record with the
                    // diffuse lobe's share only. The MIS weight keeps the full
                    // strategy pdf, which is what the continuation samples.
                    if (training.fresh != SHARC_INVALID)
                        directBroad += broadNEE * lightScale;
#endif
                }
#if SHARC_UPDATE_PASS
                SharcTrainingRadianceSplit(training, directSum, directBroad);
#endif
            }
#if !SHARC_UPDATE_PASS
            if (liteGen) LiteGenCommit(sampleIdx, liteG);
#endif

            //------------- SSS: chance to enter the medium instead of reflecting -------------
            //Mirror of raygen's stochastic enter/walk (minus the reconnection
            //bookkeeping): the walk is analog, so the throughput just takes
            //wTotal * surface albedo, and the path re-emerges at the boundary
            //exit as a white diffuse surface.
            if (!sssEntered && LoadIsSSS(ctx.matID))
            {
                const float fT     = 1.0f - FresnelDielectric(-rayDir, ctx.hitNormal, ctx.iors.x, ctx.iors.y).x;
                const float pEnter = saturate(LoadSSSWeight(ctx.matID) * fT);
                if (RandomFloatSingle(sSss) < pEnter)
                {
                    SSSWalkResult w = SubsurfaceWalk(ctx.hitPos, ctx.hitNormal, ctx.matID, sSss);
                    if (!w.valid) break;   //escaped / absorbed / step-cap -> dead path

                    const float3 surfKd = (float3)ctx.hitLocalKd;
                    throughput *= w.wTotal * surfKd;
#if SHARC_UPDATE_PASS
                    SharcTrainingAdvance(training, w.wTotal * surfKd, w.wTotal * surfKd, 1.0f);
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
                    continue;
                }
            }

            //------------- diffuse-bounce budget (same policy as raygen) -------------
            if (!MaterialIsFreeBounce(ctx.matID))
            {
                // Rendering keeps the reference cap on cache misses too. Training
                // suffixes stay uncapped here (roulette and the depth cap bound
                // them) so cold regions are not trained as prematurely dark.
                if (!SHARC_UPDATE_PASS && PtPsDiffDepth(ps) >= (uint)pt_maxDiffuseBounces) break;
                ps += 1u << PT_PS_DIFF_SHIFT;
            }

            //------------- path guiding for the diffuse lobe -------------
            // The receiver cell's bright patches become cones; their share q of
            // this vertex's cosine samples is drawn from them. The ACTUAL pdf of
            // the scatter (the mixture) weights the throughput; MIS keeps the
            // declared BSDF pdf (see NEE). Built here, after NEE and the budget
            // checks, so nothing of it is live across a trace. Training paths
            // also keep the receiver entry for discovery after the trace.
            GuideSet guide = GuideEmpty();
#if SHARC_UPDATE_PASS
            uint guideEntry = GUIDE_INVALID;
            uint guidedSlot = GUIDE_INVALID; // patch this scatter aimed at, for the visibility report
#endif
            if (cacheSurface && GUIDE_ENABLED && HasBroadShare(spPath, ctx.hitLocalPr, ctx.hitLocalPm) &&
                (int)depth <= GUIDE_MAX_DEPTH)
            {
                uint sKey = RcBounceSeed(pathSeed, (uint)depth, 0x4b455953u);
                const GuideKey guideKey = GuideKeyOf(ctx.hitPos, geometricNormal, sKey);
#if SHARC_UPDATE_PASS
                guideEntry = GuideFindOrInsert(guideKey);
                if (GUIDE_TRAIN) guide = GuideBuild(guideEntry, ctx.hitPos, ctx.hitNormal);
#else
                guide = GuideBuild(GuideFind(guideKey), ctx.hitPos, ctx.hitNormal);
#endif
            }

            //------------- BSDF sample (the NEE MIS partner) -------------
            uint sampledStrategy;
            float3 dir = SampleBRDF(spPath, ctx.matID, -rayDir, ctx.hitNormal, ctx.hitNormal, ctx.hitLocalKd, ctx.hitLocalPr, ctx.hitLocalPm, sBsdf, ctx.iors.x, ctx.iors.y, false, sampledStrategy);
            // Guided share of the broad lobes: this cosine or rough-GGX sample
            // becomes a cone sample toward a bright patch.
            const bool broadGGX = IsBroadGGX(ctx.hitLocalPr);
            if (guide.q > 0.0f && (sampledStrategy == 0u || (sampledStrategy == 1u && broadGGX)))
            {
                uint sGuide = RcBounceSeed(pathSeed, (uint)depth, GUIDE_STREAM);
                if (RandomFloatSingle(sGuide) < guide.q)
                {
                    uint pick;
                    dir = GuideSample(guide, sGuide, pick);
#if SHARC_UPDATE_PASS
                    guidedSlot = pick;
#endif
                }
            }
            float3 broadScatter; float broadScatterPdf;
            const BrdfData bdata = EvaluateAndPdf_COMBINED_L(spPath, LOBE_BROAD, ctx.matID, ctx.hitNormal, ctx.hitNormal, dir, -rayDir,
                ctx.hitLocalKd, ctx.hitLocalPr, ctx.hitLocalPm, ctx.iors.x, ctx.iors.y, false, broadScatter, broadScatterPdf);
            // Actual density of this direction under the guided mixture (any lobe
            // may land in a cone); the throughput must use it. The guided share
            // is the broad share's strategy probability with its latched pdf.
            const float pdfTotal = GuideMixPdf(guide, spPath.Pdiff + (broadGGX ? spPath.Pspec : 0.0f),
                broadScatterPdf, dir, bdata.pdf);

            const float  cosTheta     = abs(dot(ctx.hitNormal, dir));
            const float3 updateWeight = (pdfTotal > 1e-6f)
                ? (bdata.val * (float3)ctx.absorptionTint * cosTheta / pdfTotal)
                : float3(0, 0, 0);
            if (dot(dir, dir) < 1e-12f || pdfTotal <= 1e-6f ||
                any(isnan(updateWeight)) || any(isinf(updateWeight)))
                break;
            // A patch cone may straddle the horizon. A direction the BSDF cannot
            // scatter into carries nothing from here on, so end the path instead
            // of tracing it (the same zero the trace would have produced).
            if (!any(updateWeight > 0.0f))
                break;
#if !SHARC_UPDATE_PASS
            // The primary's broad share of this scatter: the candidate weight
            // of whatever the ray finds, and the part of the throughput that
            // the tracer must NOT add for it.
            float3 liteBroad = 0.0f;
            if (liteGen)
                liteBroad = broadScatter * (float3)ctx.absorptionTint * cosTheta / pdfTotal;
#endif
#if SHARC_UPDATE_PASS
            // Broad share of the same scatter for a vertex registered this
            // bounce: those lobes' gated value over the full strategy pdf.
            float3 updateWeightBroad = updateWeight;
            if (training.fresh != SHARC_INVALID)
                updateWeightBroad = broadScatter
                    * (float3)ctx.absorptionTint * cosTheta / pdfTotal;
            // Irradiance observations after the trace divide by the ACTUAL pdf.
            const float guidePdf = pdfTotal;
#endif

            // Declared pdf for the emitter / sun MIS partner and the path spread:
            // the plain BSDF pdf, matching the NEE side above.
            prev_pdf = bdata.pdf;
            ps = PtPsWith(ps, PT_PS_SPREAD, SharcScatterHasSpread(sampledStrategy, ctx.matID, ctx.hitLocalPr));
            rayDir   = dir;
            rayDirPk = PackNormal(dir);

            const float3 offsetN = (dot(dir, ctx.hitNormal) >= 0.0f) ? ctx.hitNormal : -ctx.hitNormal;
            const float3 rayOrigin = offset_ray(ctx.hitPos, offsetN);

            throughput *= updateWeight;
#if !SHARC_UPDATE_PASS
            // The suffix scatters with x2's broad share alone where the cache
            // would have answered, with the full BSDF elsewhere.
            if (litePath)
                liteSuffix *= (half3)((liteX2 && cacheSurface)
                    ? broadScatter * (float3)ctx.absorptionTint * cosTheta / pdfTotal
                    : updateWeight);
#endif

            //Russian roulette (off when pt_rrStartDepth is high)
            float rrWeight = 1.0f;
            if (depth >= (uint)(SHARC_UPDATE_PASS ? sharc_trainRrDepth : pt_rrStartDepth))
            {
                uint sRr = RcBounceSeed(pathSeed, (uint)depth, RC_STREAM_RR);
#if SHARC_UPDATE_PASS
                // Survival follows the suffix since the last registered vertex,
                // not the camera prefix: dark prefixes keep training, and dark
                // suffixes stop instead of running to the depth cap.
                const float survivalProb = SharcTrainingSurvival(training, updateWeight);
#else
                // The throughput is x1's other lobes' share; the diffuse share
                // is the suffix weight times x1's diffuse weight, so their sum
                // is the plain tracer's throughput.
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
#endif

            prevNormalPk = PackNormal(ctx.hitNormal);
#if !SHARC_UPDATE_PASS
            // Park the broad share and the scatter pdf across the trace; the
            // reservoir itself already lives in memory (nothing lite is live).
            if (liteGen) LiteParkStore(sampleIdx, liteBroad, pdfTotal);
#endif

            //----- TRACE (BSDF technique, NEE's MIS partner) -----
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

            //Nothing of the ray is read back from the HitObject: the compiler
            //folds that to the pre-trace registers and keeps origin and
            //direction live across the reorder. The direction is its packed
            //loop copy, the hit position comes from the triangle, and the
            //previous vertex is rebuilt from the hit distance where needed.
            rayDir = UnpackNormal(rayDirPk);

            //----- MISS: sky (only technique, weight 1) + MIS'd sun disc -----
            if (!hitObj.IsHit())
            {
                SetSkyObserver(cloudEnabled > .5f ? ctx.hitPos + sceneOriginWorld : InitOrigin() + sceneOriginWorld);

                const float  sunSAPdf   = GetSunPdf(rayDir);
                const float3 sunRad     = (sunSAPdf > 0.0f) ? EvaluateSun(rayDir) : float3(0, 0, 0);
                const float  sunMisBsdf = (sunSAPdf > 0.0f)
                    ? prev_pdf / max(prev_pdf + sunSAPdf, EPSILON) : 0.0f;
                const float3 sky = EvaluateSky(rayDir);
                const float3 envL = sky + sunRad * sunMisBsdf;
#if !SHARC_UPDATE_PASS
                // Diffuse/sheen samples stay broad even on a smooth material;
                // clearcoat uses its own roughness, not the base GGX roughness.
                const float cloudRoughness = sampledStrategy==0u || sampledStrategy==3u ? 1.0f
                    : sampledStrategy==2u ? (float)LoadPcr(ctx.matID) : (float)ctx.hitLocalPr;
                const uint cloudGuide = depth==1u && cloudRoughness<.25f && dot(ctx.hitNormal,rayDir)>0 ? 1u:0u;
                if (liteGen)
                {
                    // Candidate: the direction with its full radiance; the MIS
                    // weight against sun NEE rides the candidate weight.
                    const uint litePx = sampleIdx;
                    float3 liteBroad; float litePdf;
                    LiteParkLoad(litePx, liteBroad, litePdf);
                    float3 mainWeight=max(throughput-liteBroad,0.0f);
                    if(cloudEnabled>.5f) {
                        CumulusQueueMiss(samplePixel,ctx.hitPos,rayDir,mainWeight,
                            cloudGuide,RcBounceSeed(pathSeed,depth,0x434C4F55u),cloudRoughness);
                        PtAccumulate(mainWeight*sunRad*sunMisBsdf);
                    } else PtAccumulate(mainWeight*envL);
                    const float liteCosX = max(dot(UnpackNormal(prevNormalPk), rayDir), 0.0f);
                    uint sLite = RcBounceSeed(pathSeed, (uint)depth, 0x4c495447u);
                    LiteCandidate(litePx, load_kd(g_sample_current, litePx),
                        LiteSampleDirection(rayDir, sky + sunRad), rayDir,
                        LiteLinkFrom(RAY_TMAX_PLANET, liteCosX, 1.0f, true), (float3)1.0f,
                        sunSAPdf > 0.0f ? sunMisBsdf : 1.0f, litePdf, rcp((float)N), sLite);
                }
                else
                {
                    if(cloudEnabled>.5f) {
                        CumulusQueueMiss(samplePixel,ctx.hitPos,rayDir,throughput,
                            cloudGuide,RcBounceSeed(pathSeed,depth,0x434C4F55u),cloudRoughness);
                        PtAccumulate(throughput*sunRad*sunMisBsdf);
                    } else PtAccumulate(throughput * envL);
                    if (litePath) liteL += (float3)liteSuffix * envL;
                }
#else
                PtAccumulate(throughput * envL);
#endif
#if SHARC_UPDATE_PASS
                SharcTrainingRadiance(training, envL);
                // Sky and MIS-weighted sun seen by the continuation: one
                // irradiance observation for the receiver cell.
                GuideObserve(guideEntry, Luma(envL) *
                    max(dot(UnpackNormal(prevNormalPk), rayDir), 0.0f) / guidePdf);
                GuideReport(guideEntry, guidedSlot, false, 0.0f); // aimed at a patch, reached the sky
#endif
                break;
            }

            //----- HIT: extract the next vertex -----
            const float  hitT_n   = hitObj.GetRayTCurrent();
            const uint   instID_n = hitObj.GetInstanceID();
            const uint   primID_n = FlatPrimID(instID_n, hitObj.GetGeometryIndex(), hitObj.GetPrimitiveIndex());
            const uint   matID_n  = GetMatIDFast(instID_n, primID_n);
            // The attribute struct's scope ends here: a loop-body local whose
            // scope holds every later break makes DXC route those breaks and
            // the fallthrough through one shared cleanup block, whose phis
            // keep the previous vertex live across the trace.
            float2 bary_n;
            {
                BuiltInTriangleIntersectionAttributes attrB;
                hitObj.GetAttributes(attrB);
                bary_n = attrB.barycentrics;
            }
            HitInfo hinfo_n = EvalSurfaceStateDir(instID_n, primID_n, bary_n, rayDir, (uint)depth);
            const float3 hitPos_n   = hinfo_n.hitPos;
            const float3 rayOriginR = hitPos_n - rayDir * hitT_n; // previous vertex, for the light-tree pdf

            //----- Emitter hit: BSDF-technique direct emission, MIS vs NEE -----
            const float3 emission_n = (hinfo_n.lightID != 0xFFFFFFFFu)
                ? g_EmissiveTriangles[hinfo_n.lightID].emission * GLOBAL_EMISSION_STRENGTH
                : float3(0, 0, 0);
            if (any(emission_n > 0.0f))
            {
                //the NEE pdf this hit competes against: the light tree referenced
                //from the vertex we scattered at (same call as raygen)
                const float3 prevNormalCur = UnpackNormal(prevNormalPk);
                const float  lightPdfArea  = LT_Pdf_LightTree_Area(rayOriginR, prevNormalCur, hinfo_n.lightID, instID_n);
                const float  cosLight      = max(dot(hinfo_n.hitNormal, -rayDir), 0.0f);
                const float  dist2         = max(hitT_n * hitT_n, EPSILON);
                const float  lightPdfSA    = (cosLight > EPSILON) ? (lightPdfArea * dist2 / cosLight) : 0.0f;
                const float  misWeight     = prev_pdf / max(prev_pdf + lightPdfSA, EPSILON);

#if !SHARC_UPDATE_PASS
                if (liteGen)
                {
                    // Candidate: the emitter point, MIS-weighted against NEE
                    // like the tracer's own emitter hit.
                    const uint litePx = sampleIdx;
                    float3 liteBroad; float litePdf;
                    LiteParkLoad(litePx, liteBroad, litePdf);
                    PtAccumulate((throughput - liteBroad) * emission_n * misWeight);
                    const float  liteCosX = max(dot(prevNormalCur, rayDir), 0.0f);
                    const float3 liteNy   = dot(hinfo_n.geometricNormal, -rayDir) < 0.0f
                        ? -hinfo_n.geometricNormal : hinfo_n.geometricNormal;
                    uint sLite = RcBounceSeed(pathSeed, (uint)depth, 0x4c495447u);
                    LiteCandidate(litePx, load_kd(g_sample_current, litePx),
                        LiteSampleSurface(instID_n, hitPos_n, liteNy, emission_n, LITE_KIND_LIGHT), hitPos_n,
                        LiteLinkFrom(hitT_n, liteCosX, dot(liteNy, -rayDir), false), (float3)1.0f,
                        misWeight, litePdf, rcp((float)N), sLite);
                }
                else
                {
                    PtAccumulate(throughput * emission_n * misWeight);
                    if (litePath) liteL += (float3)liteSuffix * emission_n * misWeight;
                }
#else
                PtAccumulate(throughput * emission_n * misWeight);
                SharcTrainingRadiance(training, emission_n * misWeight);
                GuideObserve(guideEntry, Luma(emission_n * misWeight) *
                    max(dot(UnpackNormal(prevNormalPk), rayDir), 0.0f) / guidePdf);
                GuideReport(guideEntry, guidedSlot, true, hitPos_n);
#endif
                break;   //emitter hits terminate the walk (raygen policy)
            }

            //----- Carry the next vertex into ctx -----
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
                // x2 found: from here the diffuse suffix belongs to the pending
                // candidate. Park its point (normal facing x1, scatter pdf);
                // x1's other lobes keep their share of the same path.
                const uint litePx = sampleIdx;
                float3 liteBroad; float litePdf;
                LiteParkLoad(litePx, liteBroad, litePdf);
                const float3 liteNy = dot(hinfo_n.geometricNormal, -rayDir) < 0.0f
                    ? -hinfo_n.geometricNormal : hinfo_n.geometricNormal;
                LiteParkPointStore(litePx, WorldToObjectPos(instID_n, hitPos_n), instID_n,
                    PackNormal(WorldToObjectNrm(instID_n, liteNy)), litePdf);
                // The throughput continues as x1's other lobes' share; the
                // diffuse share restarts as the suffix weight.
                liteRr     = Luma(liteBroad);
                throughput = max(throughput - liteBroad, 0.0f);
                liteSuffix = (half3)1.0f;
                liteL      = 0.0f;
                ps |= PT_PS_LITE_X2;
            }
#endif
            ctx.hitPos         = hitPos_n;
            geometricNormal    = hinfo_n.geometricNormal;
            // Spread in area measure (distance / sqrt(pdf * receiver cosine)).
            // Every continuous lobe can grow it; the PDF limits narrow lobes'
            // spread and the samplers' true delta branches leave it unchanged.
            if ((ps & PT_PS_SPREAD) != 0u)
                pathSpread += hitT_n * sqrt(min(16.0f, rcp(max(prev_pdf *
                    abs(dot(geometricNormal, -rayDir)), 1e-6f))));
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
            ps = PtPsWith(ps, PT_PS_FLIP_IOR, flipIOR_n);
#if SHARC_UPDATE_PASS
            // Guiding discovery. The cache's outgoing luminance at this hit
            // (any resolved record, no confidence gate) registers the coarse
            // patch around it with the receiver cell we scattered from, and the
            // same hit is one irradiance observation there: zero when the
            // surface has no record yet, which only makes q conservative.
            if (guideEntry != GUIDE_INVALID)
            {
                const float receiverCos = max(dot(UnpackNormal(prevNormalPk), rayDir), 0.0f);
                uint sSeen = RcBounceSeed(pathSeed, (uint)depth, 0x5345454eu);
                float3 seen = 0.0f;
                const bool known = SharcQueryForced(SharcMakeSurface(ctx, geometricNormal), sSeen, seen);
                GuideObserve(guideEntry, (known ? Luma(seen) : 0.0f) * receiverCos / guidePdf);
                // Did a guided ray reach its patch? Learned occlusion per patch.
                GuideReport(guideEntry, guidedSlot, true, ctx.hitPos);
                if (known) GuideDiscover(guideEntry, ctx.hitPos, geometricNormal, Luma(seen),
                    SHARC_DEBUG_MODE == SHARC_DEBUG_GUIDING);
            }
#endif
            // Complete the next vertex's cache decision while its HitObject
            // is local to this iteration. Terminating lanes leave before SER;
            // one reorder groups the remaining hits for their next NEE/scatter.
            const uint nextDepth = depth + 1u;
            if (nextDepth >= maxBounces) break;
            SamplingP spNext = CalculateStrategyProbabilities(ctx.matID, -rayDir, ctx.hitNormal,
                ctx.iors.x, ctx.iors.y, ctx.hitLocalKd, ctx.hitLocalPm);
            bool nextCacheSurface, nextDiffuseCached;
            if (!PtPrepareVertex(ctx, geometricNormal, rayDir, sssEntered,
                pathSeed, nextDepth, maxBounces, pathSpread, spNext, nextCacheSurface, nextDiffuseCached, throughput,
#if SHARC_UPDATE_PASS
                training
#else
                nextDepth == 2u && liteVertex, liteVertex, total, liteL, liteSuffix
#endif
            )) break;
            ps = PtPsWith(ps, PT_PS_DIFF_CACHED, nextDiffuseCached);
            const uint hint = 0x40u | (ctx.instID & 0x3Fu);
            // Across the reorder the vertex keeps its normals packed and its
            // medium state as the flag above: the exact normals, the IORs, the
            // medium id and the absorption (38 bytes) are rebuilt on the other
            // side from 8 bytes, the hit distance and the material record.
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
#else
        // The pending secondary-vertex candidate of this path (cache value or
        // traced suffix), resampled against the reservoir in memory.
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
