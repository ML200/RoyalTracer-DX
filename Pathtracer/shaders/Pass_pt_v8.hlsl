#include "Includes_v8.hlsli"
#include "Raygen_Common_v8.hlsli"
#ifndef SHARC_UPDATE_PASS
#define SHARC_UPDATE_PASS 0
#endif
#ifndef PT_ENTRY_NAME
#define PT_ENTRY_NAME Pass_pt_v8
#endif
#include "SharcPath_v8.hlsli"
#if !SHARC_UPDATE_PASS
#include "SharcDebug_v8.hlsli"
#endif

//====================================
//CLEAN PATH-TRACING KERNEL (RIS-FREE)
//====================================
//Plain unidirectional path tracer with NEE — the reservoir-free alternative
//to the ReSTIR pipeline (selected by integratorMode; the host skips every
//reservoir pass and dispatches this instead of Pass_raygen_v8). The transport
//math mirrors Pass_raygen exactly — same light-tree NEE + sun NEE with
//balance-heuristic MIS against the BSDF technique, same SSS enter/walk, same
//diffuse-bounce budget, RR and medium/absorption handling — but contributions
//accumulate directly into radiance instead of feeding reservoir candidates,
//so with SHaRC disabled this doubles as the reference for the ReSTIR target
//functions (same estimators, no resampling).
//SHaRC's sparse update specialization includes this file with SHARC_UPDATE_PASS.
//The render specialization keeps the primary vertex exact and may replace the
//outgoing radiance of a broad, diffuse secondary vertex (its own direct light
//included, so a terminating vertex skips NEE). See docs/SHARC.md.
//
//Frame integration: reads the primary vertex from the camera pass G-buffer
//(full-screen dispatch, SD_FLAG_NOBOUNCE early-out), writes the estimate to
//scratch slot 2 (the slot the ReSTIR spatial resolve used to write); the
//DLSS-RR guides all come from the camera/shading passes, so shading /
//clouds / postprocess run unchanged and this kernel touches no reservoir.
//
//SER: every trace goes through TraceRay_Custom (dx::HitObject +
//dx::MaybeReorderThread, instance-sorted hint). Live state across the trace is
//kept to the path itself: the ray origin/direction are read back from the
//HitObject (which carries the ray through the reorder anyway), the pixel is
//recomputed from DispatchRaysIndex, and the primary G-buffer record is reloaded
//per sample instead of surviving the whole bounce loop.

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
                // Slate: the tracer never caches this surface (glossy, metallic,
                // layered, transmitting, SSS or a steep normal map). No record
                // is expected here, so this is not a missing cell.
                debugColor = float4(0.05f, 0.07f, 0.12f, 0.0f);
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

    //sky/sun observer at the camera altitude (bounce misses re-set it)
    SetSkyObserver(InitOrigin() + sceneOriginWorld);

    float3 total = float3(0, 0, 0);

    const uint N = SHARC_UPDATE_PASS ? 1u : max(pt_initialSamples, 1u);
    // Cache misses render with the ordinary budget; only training paths use the
    // cache depth cap. A miss path does not train anything, so tracing it deeper
    // than the reference only costs time.
    const uint maxBounces = SHARC_UPDATE_PASS ? sharc_trainBounces : pt_maxBounces;
    [loop]
    for (uint s = 0u; s < N; ++s)
    {
        //Recompute the pixel and reload the primary record for every sample so
        //neither survives the bounce loop as reorder live state.
        const uint2 samplePixel = PtPixel();
        const uint  sampleIdx   = MapPixelID(imgSize, samplePixel);
        //same seed derivation as Pass_raygen so the two integrators sample
        //identical per-bounce streams (RcBounceSeed) at N==1
        uint seed     = initRandomData(samplePixel, uint2(8, 4), time, s + 1u);
        uint pathSeed = Hash32(seed ^ 0x9E3779B9u);
#if SHARC_UPDATE_PASS
        pathSeed = Hash32(pathSeed ^ sharc_frame ^ 0x53484152u);
        SharcTrainingState training;
        SharcTrainingInit(training);
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
        bool    sssEntered   = false;
        int16_t diffuseDepth = 0;
        float3 geometricNormal = sd.n1_s;
#if SHARC_UPDATE_PASS
        geometricNormal = gScratchPing[uint3(samplePixel, SHARC_DEBUG_SCRATCH)].xyz;
#endif
        float pathSpread = 0.0f;

        [loop]
        for (int16_t depth = 1; depth < (int)maxBounces; ++depth)
        {
            float3 rayDir = UnpackNormal(rayDirPk);

            //per-bounce RNG streams (draw-count independent, same ids as raygen)
            uint sNee  = RcBounceSeed(pathSeed, (uint)depth, RC_STREAM_NEE);
            uint sBsdf = RcBounceSeed(pathSeed, (uint)depth, RC_STREAM_BSDF);
            uint sSss  = RcBounceSeed(pathSeed, (uint)depth, RC_STREAM_SSS);

            //strategy probabilities once per bounce, shared by NEE + BSDF. A cache
            //hit removes the diffuse lobe from this mix for the rest of the vertex.
            SamplingP spPath = CalculateStrategyProbabilities(ctx.matID, -rayDir, ctx.hitNormal, ctx.iors.x, ctx.iors.y, ctx.hitLocalKd, ctx.hitLocalPm);

            // Cache first, then local NEE. A record stores the DIFFUSE lobe's
            // outgoing radiance at this vertex (its own direct light through that
            // lobe included), demodulated by Kd and by the view-dependent
            // transmission of the layers above it. A hit adds that share and drops
            // the diffuse lobe from this vertex; sheen, coat and GGX continue on
            // the exact tracer with their own strategy mix, so nothing specular is
            // ever cached. The primary vertex is never queried.
            if (sharc_enabled != 0u && (SHARC_UPDATE_PASS || depth > 1) && !sssEntered &&
                SharcMaterialEligible(ctx, spPath, geometricNormal))
            {
                const SharcSurface surface = SharcMakeSurface(ctx, geometricNormal);
                const float layerT = SharcLayerTransmission(spPath, ctx, -rayDir);
                const bool queryable = diffuseDepth > 0 && layerT > 1e-3f;
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
                hit = queryable && SharcQuery(surface, pathSpread, sCache, cached);
                if (hit) total += throughput * cached * layerT;
#endif
                if (hit)
                {
                    // The diffuse lobe is accounted for; continue with whatever
                    // layers remain. A plain Lambertian vertex is finished here.
                    if (!DropDiffuseLobe(spPath)) break;
                    // The remaining layers carry about 1 - layerT of this vertex's
                    // energy (the Fresnel of the stack). Roulette them BEFORE their
                    // NEE and scatter: a hit on a plain dielectric then ends here
                    // ~96% of the time instead of paying a full specular vertex
                    // for a 4% term. Survivors are compensated, so nothing is lost.
                    const float pSpecular = clamp(1.0f - layerT, 0.02f, 1.0f);
                    if (RandomFloatSingle(sCache) >= pSpecular) break;
                    throughput *= rcp(pSpecular);
#if SHARC_UPDATE_PASS
                    SharcTrainingScatter(training, rcp(pSpecular));
#endif
                }
            }
            // Lanes that just ended at the cache leave their waves sparse for the
            // most expensive part of the vertex (light-tree sample, two shadow
            // rays, cloud march). Re-pack the survivors by instance before NEE
            // rather than waiting for the next trace to do it.
            if (sharc_enabled != 0u && depth > 1)
                dx::MaybeReorderThread(0x40u | (ctx.instID & 0x3Fu), 7u);

            //------------- NEE: light-tree point light + sun (MIS vs BSDF) -------------
            const bool performNEE = !(ctx.mediumMatID != MEDIUM_INVALID || LoadKd_w(ctx.matID) < EPSILON);
            if (performNEE)
            {
                float3 directSum = float3(0, 0, 0);
#if SHARC_UPDATE_PASS
                float3 directDiffuse = float3(0, 0, 0);
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

                    if (tech == 0u)
                    {
                        LT_LightSampleResult light = LT_SamplePointOnLight(ctx.hitPos, ctx.hitNormal, sNee);

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
                    }
                    else
                    {
                        const float2 rSun = float2(RandomFloatSingle(sNee), RandomFloatSingle(sNee));
                        SunSampleResult sun = SampleSun(rSun);
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

                    //sun-only cloud shadow (same block as raygen, runtime-gated)
                    if (tech == 1u && cloud_cloudShadowOnSurfaces > 0.5f)
                    {
                        float2 rCone = float2(RandomFloatSingle(sNee), RandomFloatSingle(sNee));
                        float  cosCone = cos(SURFACE_CLOUD_SHADOW_CONE_DEG * DEG2RAD);
                        float3 Lj = SampleConeAroundDir(L, cosCone, rCone);
                        float  vis = CloudSunVisibility(ctx.hitPos + sceneOriginWorld, Lj,
                                                        RandomFloatSingle(sNee));
                        radiance *= pow(max(vis, 1e-6f), SURFACE_CLOUD_SHADOW_SOFTNESS);
                    }

                    BrdfData bdataNEE = EvaluateAndPdf_COMBINED(spPath, ctx.matID, ctx.hitNormal, ctx.hitNormal, L, -rayDir, ctx.hitLocalKd, ctx.hitLocalPr, ctx.hitLocalPm, ctx.iors.x, ctx.iors.y);
                    if (!(bdataNEE.pdf > 0.0f))
                        continue;

                    const float  misWeight  = lightPdf / (lightPdf + bdataNEE.pdf);
                    const float3 lightScale = radiance * cosSurf * visT * (misWeight / lightPdf);
                    const float3 direct     = bdataNEE.val * lightScale;
                    total += throughput * direct;
                    directSum += direct;
#if SHARC_UPDATE_PASS
                    // A vertex registered this bounce labels its record with the
                    // diffuse lobe's share only. The MIS weight keeps the full
                    // strategy pdf, which is what the continuation samples.
                    if (training.fresh != SHARC_INVALID)
                        directDiffuse += EvaluateLobePdf_COMBINED(spPath, 0u, ctx.matID, ctx.hitNormal, ctx.hitNormal, L, -rayDir,
                            ctx.hitLocalKd, ctx.hitLocalPr, ctx.hitLocalPm, ctx.iors.x, ctx.iors.y).val * lightScale;
#endif
                }
#if SHARC_UPDATE_PASS
                SharcTrainingRadianceSplit(training, directSum, directDiffuse);
#endif
            }

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
#endif
                    sssEntered  = true;

                    ctx.hitPos         = w.exitPos;
                    ctx.hitNormal      = w.exitNormal;
                    ctx.hitLocalKd     = (half3)float3(1, 1, 1);
                    ctx.hitLocalPr     = (half)1.0f;
                    ctx.hitLocalPm     = (half)0.0f;
                    ctx.iors           = (half2)float2(1.0f, 1.0f);
                    ctx.mediumMatID    = MEDIUM_INVALID;
                    ctx.absorptionTint = (half3)float3(1, 1, 1);
                    rayDirPk           = PackNormal(-w.exitNormal);
                    continue;
                }
            }

            //------------- diffuse-bounce budget (same policy as raygen) -------------
            if (!MaterialIsFreeBounce(ctx.matID))
            {
                // Rendering keeps the reference cap on cache misses too. Training
                // suffixes stay uncapped here (roulette and the depth cap bound
                // them) so cold regions are not trained as prematurely dark.
                if (!SHARC_UPDATE_PASS && diffuseDepth >= (int)pt_maxDiffuseBounces) break;
                ++diffuseDepth;
            }

            //------------- BSDF sample (the NEE MIS partner) -------------
            uint sampledStrategy;
            const float3 dir = SampleBRDF(spPath, ctx.matID, -rayDir, ctx.hitNormal, ctx.hitNormal, ctx.hitLocalKd, ctx.hitLocalPr, ctx.hitLocalPm, sBsdf, ctx.iors.x, ctx.iors.y, false, sampledStrategy);
            const BrdfData bdata = EvaluateAndPdf_COMBINED(spPath, ctx.matID, ctx.hitNormal, ctx.hitNormal, dir, -rayDir, ctx.hitLocalKd, ctx.hitLocalPr, ctx.hitLocalPm, ctx.iors.x, ctx.iors.y);

            const float  cosTheta     = abs(dot(ctx.hitNormal, dir));
            const float3 updateWeight = (bdata.pdf > 1e-6f)
                ? (bdata.val * (float3)ctx.absorptionTint * cosTheta / bdata.pdf)
                : float3(0, 0, 0);
            if (dot(dir, dir) < 1e-12f || bdata.pdf <= 1e-6f ||
                any(isnan(updateWeight)) || any(isinf(updateWeight)))
                break;
#if SHARC_UPDATE_PASS
            // Diffuse lobe's share of the same scatter for a vertex registered
            // this bounce: that lobe's gated value over the full strategy pdf.
            float3 updateWeightDiffuse = updateWeight;
            if (training.fresh != SHARC_INVALID)
                updateWeightDiffuse = EvaluateLobePdf_COMBINED(spPath, 0u, ctx.matID, ctx.hitNormal, ctx.hitNormal, dir, -rayDir,
                    ctx.hitLocalKd, ctx.hitLocalPr, ctx.hitLocalPm, ctx.iors.x, ctx.iors.y).val
                    * (float3)ctx.absorptionTint * cosTheta / bdata.pdf;
#endif

            prev_pdf = bdata.pdf;
            rayDir   = dir;
            rayDirPk = PackNormal(dir);

            const float3 offsetN = (dot(dir, ctx.hitNormal) >= 0.0f) ? ctx.hitNormal : -ctx.hitNormal;
            const float3 rayOrigin = offset_ray(ctx.hitPos, offsetN);

            throughput *= updateWeight;

            //Russian roulette (off when pt_rrStartDepth is high)
            float rrWeight = 1.0f;
            if (depth >= (int)(SHARC_UPDATE_PASS ? sharc_trainRrDepth : pt_rrStartDepth))
            {
                uint sRr = RcBounceSeed(pathSeed, (uint)depth, RC_STREAM_RR);
#if SHARC_UPDATE_PASS
                // Survival follows the suffix since the last registered vertex,
                // not the camera prefix: dark prefixes keep training, and dark
                // suffixes stop instead of running to the depth cap.
                const float survivalProb = SharcTrainingSurvival(training, updateWeight);
#else
                const float survivalProb = max(min(1.0f, Luma(throughput)), 0.05f);
#endif
                if (RandomFloatSingle(sRr) >= survivalProb) break;
                throughput /= survivalProb;
                rrWeight = rcp(survivalProb);
            }
#if SHARC_UPDATE_PASS
            SharcTrainingAdvance(training, updateWeight, updateWeightDiffuse, rrWeight);
#endif

            prevNormalPk = PackNormal(ctx.hitNormal);

            //----- TRACE (BSDF technique, NEE's MIS partner) -----
            if (!IsRayValid(rayOrigin, rayDir, 10000.0f))
                break;

            RayDesc rayB;
            rayB.Origin    = rayOrigin;
            rayB.Direction = rayDir;
            rayB.TMin      = 0.00001f;
            rayB.TMax      = RAY_TMAX_PLANET;
#if SHARC_UPDATE_PASS
            // Lowest reorder bit: lanes whose suffix is about to be rouletted
            // away sort together, so waves drain as a group instead of one lane
            // at a time while the survivors keep full waves.
            dx::HitObject hitObj = TraceRay_Custom(SceneBVH, rayB, RAY_FLAG_FORCE_OMM_2_STATE, 0xFF,
                training.suffixLuma < 0.25f ? 1u : 0u, 1u);
#else
            dx::HitObject hitObj = TraceRay_Custom(SceneBVH, rayB, RAY_FLAG_FORCE_OMM_2_STATE, 0xFF);
#endif

            //The HitObject carries the ray through the reorder: read origin and
            //direction back from it rather than keeping them live or parking
            //the origin in scratch. Exact values, no packing round trip.
            const float3 rayOriginR = hitObj.GetWorldRayOrigin();
            rayDir = hitObj.GetWorldRayDirection();

            //----- MISS: sky (only technique, weight 1) + MIS'd sun disc -----
            if (!hitObj.IsHit())
            {
                SetSkyObserver(rayOriginR + sceneOriginWorld);

                const float  sunSAPdf   = GetSunPdf(rayDir);
                const float3 sunRad     = (sunSAPdf > 0.0f) ? EvaluateSun(rayDir) : float3(0, 0, 0);
                const float  sunMisBsdf = (sunSAPdf > 0.0f)
                    ? prev_pdf / max(prev_pdf + sunSAPdf, EPSILON) : 0.0f;
                float3 cloudTr;
                const float3 sky  = EvaluateSky(rayDir, cloudTr);
                const float3 envL = sky + sunRad * sunMisBsdf * cloudTr;

                total += throughput * envL;
#if SHARC_UPDATE_PASS
                SharcTrainingRadiance(training, envL);
#endif
                break;
            }

            //----- HIT: extract the next vertex -----
            const float  hitT_n   = hitObj.GetRayTCurrent();
            const float3 hitPos_n = rayOriginR + rayDir * hitT_n;
            const uint   instID_n = hitObj.GetInstanceID();
            const uint   primID_n = FlatPrimID(instID_n, hitObj.GetGeometryIndex(), hitObj.GetPrimitiveIndex());
            const uint   matID_n  = GetMatIDFast(instID_n, primID_n);
            BuiltInTriangleIntersectionAttributes attrB;
            hitObj.GetAttributes(attrB);
            HitInfo hinfo_n = EvalSurfaceState(instID_n, primID_n, attrB.barycentrics, rayOriginR, (uint)depth);

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

                total += throughput * emission_n * misWeight;
#if SHARC_UPDATE_PASS
                SharcTrainingRadiance(training, emission_n * misWeight);
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

            ctx.hitPos         = hitPos_n;
            geometricNormal    = hinfo_n.geometricNormal;
            // Spread in area measure (distance / sqrt(pdf * receiver cosine)).
            // Only a sampled diffuse lobe increases it; delta chains stay sharp.
            if (sampledStrategy == 0u)
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
        }
#if SHARC_UPDATE_PASS
        SharcTrainingCommit(training);
#endif
    }

    total /= (float)N;
    if (any(isnan(total)) || any(isinf(total))) total = float3(0, 0, 0);

    //radiance -> the slot the ReSTIR spatial resolve used to write; shading /
    //clouds / DLSS pick it up unchanged. (No reservoir writes: the spec-hit-
    //dist guide now comes from the camera pass's slot-4 reflection probe in
    //Pass_shading, so this kernel touches no reservoir plane at all.)
#if !SHARC_UPDATE_PASS
    gScratchPing[uint3(PtPixel(), 2)] = float4(total, 0.0f);
#endif
}
