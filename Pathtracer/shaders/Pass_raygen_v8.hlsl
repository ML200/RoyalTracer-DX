#include "Includes_v8.hlsli"
#include "Raygen_Common_v8.hlsli"

//====================================
//VARIANT SPECIALIZATION
//====================================
//The library is compiled TWICE (Renderer::CreateRaytracingPipeline): the default
//entry, and a "lite" variant (-D RAYGEN_ENTRY=Pass_raygen_v8_lite
//-D RAYGEN_NO_CLOUD_SURF_SHADOW=1) with the sun-NEE cloud-shadow block compiled
//OUT. The host's indirect-dispatch template selects the lite SBT record whenever
//the editor's cloud_cloudShadowOnSurfaces toggle is off — exactly the condition
//under which the block is inert — so the render is bit-identical while the hot
//variant sheds ~1.5k instructions of I$ footprint + the block's register
//pressure. Extend with more -D axes (e.g. RAYGEN_NO_SSS for SSS-free scenes) by
//adding records the same way.
#ifndef RAYGEN_ENTRY
#define RAYGEN_ENTRY Pass_raygen_v8
#endif
#ifndef RAYGEN_NO_CLOUD_SURF_SHADOW
#define RAYGEN_NO_CLOUD_SURF_SHADOW 0
#endif


//One reservoir: emissive-tri DI (d==1), first-bounce DI (d==2), GI (d>=3).
//d==1 direct sky+sun -> scratch slot 3; cold state in g_pathStateBuffer.
//
//The camera ray now lives in Pass_camera_v8: it wrote the primary G-buffer +
//scratch (slots 1/2/4), the SPMIS hash cell, the primaryExtra (iors/medium/
//absorption) HOT2 record, and finalized terminal pixels (miss / degenerate /
//direct emitter, all flagged SD_FLAG_NOBOUNCE). This pass rebuilds the primary
//ctx from that G-buffer per sample and runs the initial-sample + bounce loops.
[shader("raygeneration")]
void RAYGEN_ENTRY()
{
    //COMPACTED 1D dispatch: Pass_camera queued only the non-terminal pixels and
    //the host launches exactly that count via ExecuteIndirect, so there is no
    //NOBOUNCE test and no dead lane rides the bounce loop. Pixel coords come
    //from the queue; the image size from the push constants (the dispatch dims
    //are the 1D queue count now). Per-pixel seeds derive from `pixel`, so the
    //output is bit-identical to the full-screen dispatch.
    const uint  packedPx = g_raygenQueue.Load(16u + DispatchRaysIndex().x * 4u);
    const uint2 pixel    = uint2(packedPx & 0xFFFFu, packedPx >> 16);
    const uint2 imgSize  = uint2(IMG_W, IMG_H);
    const uint  pixelIdx = MapPixelID(imgSize, pixel);

    //(no zero storeReservoir here: Pass_camera already zero-initialized every
    //pixel's reservoir this frame — the old re-zero was redundant and cost two
    //instanceProps[0] matrix fetches per pixel through WorldToObjectPos/Nrm.)

    float wsum = 0.0f;

    //sky/sun observer at the camera altitude (absolute = view + sceneOrigin).
    SetSkyObserver(InitOrigin() + sceneOriginWorld);

    //directAtX1 in scratch (HOT1[12..15], RGB9E5); summed across the N samples,
    // /N at finalize.
    store_rg_directX1(g_pathStateBuffer, pixelIdx, float3(0, 0, 0));

    //Primary incoming (view) direction recovered from the camera origin + primary
    //hit. DoF lens jitter is sub-pixel and dropped here, matching the temporal
    //pass's NoV convention. Seeds the depth==1 back-edge carrier (rayDirPk0).
    const uint   primInst0 = load_instID(g_sample_current, pixelIdx);
    const float3 primPos0  = load_x1_with_instID(g_sample_current, pixelIdx, primInst0);
    const uint   rayDirPk0 = PackNormal(normalize(primPos0 - InitOrigin()));

    HitContext ctx;
    uint   seed;
    uint   throughputPk;
    uint   prevNormalPk;
    float  prev_pdf;
    float  pdf_product;
    uint   rayDirPk;
    float3 rayOrigin;

    //====================================
    //INITIAL-SAMPLE LOOP (pt_initialSamples)
    //====================================
    //Re-run the candidate bounce loop N times into ONE reservoir (wsum streams
    //all N; directAtX1 sums, /N at finalize). The shared primary hit is rebuilt
    //per-sample from the G-buffer + HOT2 extras rather than kept live across the
    //inner traces (that halves the continuation live-state). Only per-path
    //carriers reset per sample.
    [loop]
    for (uint s = 0u; s < pt_initialSamples; ++s)
    {
        //rebuild the shared primary ctx from the G-buffer + HOT2 extras (same
        //quantized surface as the reuse passes, so N==1 differs imperceptibly
        //from the pre-spill build). Bundled record load (2xLoad4+Load) instead
        //of 8 field-wise loads; stays inside the sample loop on purpose — the
        //fields must NOT be live across the inner bounce traces.
        const SDRecord sd   = load_SD(g_sample_current, pixelIdx);
        ctx.instID          = sd.instID;
        ctx.hitPos          = sd.x1;
        ctx.hitNormal       = sd.n1_s;
        ctx.matID           = sd.matID;
        ctx.backface        = (sd.flags & SD_FLAG_BACKFACE) != 0u;
        ctx.hitLocalKd      = (half3)sd.Kd;
        ctx.hitLocalPr      = (half)sd.Pr;
        ctx.hitLocalPm      = (half)sd.Pm;
        float2 iors_; uint medium_; float3 absorb_;
        load_rg_primaryExtra(g_pathStateBuffer, pixelIdx, iors_, medium_, absorb_);
        ctx.iors            = (half2)iors_;
        ctx.mediumMatID     = medium_;
        ctx.absorptionTint  = (half3)absorb_;

        //per-sample carrier reset. seed is re-init every sample now that the
        //camera ray lives in Pass_camera (s==0 no longer inherits a post-camera
        //seed); the depth==1 back-edge starts from the recovered view dir.
        seed = initRandomData(pixel, uint2(8, 4), time, s + 1u);
        throughputPk = PackRGB9E5(float3(1, 1, 1));
        prevNormalPk = PackNormal(float3(0, 1, 0));
        prev_pdf     = 1.0f;
        pdf_product  = 1.0f;
        store_rg_tpost(g_pathStateBuffer, pixelIdx, float3(1, 1, 1));
        rayDirPk     = rayDirPk0;

        //SSS path state (per sample). sssEntered blocks re-entry after one walk;
        //sssActive marks that the walk relocated ctx to a suffix boundary-exit B (so
        //the depth==2 reconnection bookkeeping is rerouted). They differ for the
        //no-scatter primary case, where the exit B is itself the v2 reconnection vertex.
        bool sssActive  = false;
        bool sssEntered = false;

    //====================================
    //BOUNCE LOOP
    //====================================
    //Process v_depth (ctx), then trace to v_{depth+1}. NEE then BSDF at the top;
    //the bottom trace is the BSDF MIS partner and fires every iter. depth==1
    //primary (sky+sun -> directAtX1, emissive DI -> reservoir); ==2 first bounce
    //(stores ps); >=3 GI from PathVertexState, RR + tpost active.
    [loop]
    for (int16_t depth = 1; depth < (int)pt_maxBounces; ++depth)
    {
        float3 rayDir = UnpackNormal(rayDirPk);

        //Strategy probabilities are fixed for this bounce (matID, -rayDir, normal, iors,
        //Kd, Pm don't change within the iteration). Compute once and reuse for both NEE
        //techniques and the BSDF sample instead of three identical evals — the two NEE
        //sites sit behind separate IsVisible branches so DXC can't CSE them itself.
        const SamplingP sp = CalculateStrategyProbabilities(ctx.matID, -rayDir, ctx.hitNormal, ctx.iors.x, ctx.iors.y, ctx.hitLocalKd, ctx.hitLocalPm);

        //------------- NEE: point light + sun (MIS partner of the bottom trace) -------------
        //FUSED into one 2-iteration technique loop (0 = light-tree point light, 1 = sun):
        //the two bodies were structurally identical — sample, inline visibility,
        //EvaluateAndPdf_COMBINED, depth-branched candidate store — and inlined twice,
        //which was a large slice of the pre-trace instruction footprint. One inlined
        //instance now serves both. RNG draw order matches the old back-to-back blocks
        //exactly (light-tree draws, then rSun, then the cloud-shadow draws gated inside
        //the sun iteration), and every expression keeps its original form, so the
        //output is bit-identical. [loop] keeps DXC from unrolling it back into two copies.
        const bool performNEE = !(ctx.mediumMatID != MEDIUM_INVALID || LoadKd_w(ctx.matID) < EPSILON);
        if (performNEE)
        {
            [loop]
            for (uint tech = 0u; tech < 2u; ++tech)
            {
                float3 L;                    //direction to the light
                float3 visTarget;            //visibility endpoint
                float3 visTargetN;           //endpoint normal
                float3 radiance;             //emission / sun radiance along -L
                float  lightPdf;             //solid-angle pdf
                float  cosSurf;
                float3 lightPos = float3(0, 0, 0);   //point-light only (depth==1 payload)
                float3 lightN   = float3(0, 1, 0);
                uint   lightObjID = 0u;

                if (tech == 0u)
                {
                    LT_LightSampleResult light = LT_SamplePointOnLight(ctx.hitPos, ctx.hitNormal, seed);

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
                    lightPos   = light.position;
                    lightN     = light.normal;
                    lightObjID = light.objID;
                }
                else
                {
                    const float2 rSun = float2(RandomFloatSingle(seed), RandomFloatSingle(seed));
                    SunSampleResult sun = SampleSun(rSun);
                    L = sun.direction;

                    cosSurf = dot(ctx.hitNormal, L);   //== NdotL
                    if (!(cosSurf > 1e-6f && sun.pdf > 1e-20f))
                        continue;

                    visTarget  = ctx.hitPos + sun.direction * RAY_TMAX_PLANET;
                    visTargetN = -sun.direction;
                    radiance   = sun.radiance;
                    lightPdf   = sun.pdf;
                }

                //visibility first: keep the strategy+BSDF eval out of the
                //cross-ray live set; run it only if unoccluded. Thin glass attenuates
                //(1-F)*Tf rather than blocking, so visibility is an RGB transmittance.
                const float3 visT = VisibilityTransmittance(ctx.hitPos, ctx.hitNormal, visTarget, visTargetN);
                if (!any(visT > 0.0f))
                    continue;

                //sun-only cloud shadow (draw order preserved: after the vis test, as
                //before). Compiled out of the lite variant — the host dispatches that
                //variant only when the toggle is off, i.e. when this block is inert.
#if !RAYGEN_NO_CLOUD_SURF_SHADOW
                if (tech == 1u && cloud_cloudShadowOnSurfaces > 0.5f)
                {
                    float2 rCone = float2(RandomFloatSingle(seed), RandomFloatSingle(seed));
                    float  cosCone = cos(SURFACE_CLOUD_SHADOW_CONE_DEG * DEG2RAD);
                    float3 Lj = SampleConeAroundDir(L, cosCone, rCone);
                    float  vis = CloudSunVisibility(ctx.hitPos + sceneOriginWorld, Lj,
                                                    RandomFloatSingle(seed));
                    radiance *= pow(max(vis, 1e-6f), SURFACE_CLOUD_SHADOW_SOFTNESS);
                }
#endif

                const float3 throughput = UnpackRGB9E5(throughputPk);

                BrdfData  bdataNEE = EvaluateAndPdf_COMBINED(sp, ctx.matID, ctx.hitNormal, ctx.hitNormal, L, -rayDir, ctx.hitLocalKd, ctx.hitLocalPr, ctx.hitLocalPm, ctx.iors.x, ctx.iors.y);

                const float bsdfPdf = bdataNEE.pdf;
                if (!(bsdfPdf > 0.0f))
                    continue;

                const float  misWeight        = lightPdf / (lightPdf + bsdfPdf);
                const float3 localMeasurement = radiance * bdataNEE.val * cosSurf * visT;
                const float3 F_contrib        = throughput * localMeasurement * pdf_product;
                const float  p_hat            = GetPHat(F_contrib);
                const float  p_full           = pdf_product * lightPdf;
                const float  wi               = (p_full > 1e-20f) ? (misWeight * p_hat / p_full) : 0.0f;

                if (depth == 1)
                {
                    if (tech == 0u)
                    {
                        //primary DI, x2 is the light
                        AddInitialCandidate(wsum, g_Reservoirs_current, pixelIdx, wi,
                            lightPos, lightN,
                            radiance, diMarkerFor(pixelIdx, time),
                            float3(0,0,0), 0.0f, 0.0f,
                            MATID_LIGHT_TRI, lightObjID, 1.0f,
                            F_contrib, seed);
                    }
                    else
                    {
                        //primary direct sun -> directAtX1, out of the reservoir
                        add_rg_directX1(g_pathStateBuffer, pixelIdx, DirectContribution(wi, F_contrib));
                    }
                }
                else if (depth == 2 && !sssActive)
                {
                    //first-bounce DI, x2 is the surface. sssEntered here means a
                    //no-scatter SSS pass-through: x2 is the white-diffuse exit B,
                    //tag it MATID_SSS_EXIT_BIT so Reconnect uses the 1/pi entry
                    //coupling for x1 (matching the forward pdf_product fold).
                    AddInitialCandidate(wsum, g_Reservoirs_current, pixelIdx, wi,
                        ctx.hitPos, ctx.hitNormal,
                        radiance, -L,
                        (float3)ctx.hitLocalKd, (float)ctx.hitLocalPr, (float)ctx.hitLocalPm,
                        ctx.matID | (sssEntered ? MATID_SSS_EXIT_BIT : 0u), ctx.instID, ctx.iors.y,
                        F_contrib, seed);
                }
                else
                {
                    const PathVertexState ps = load_ps(g_pathStateBuffer, pixelIdx);
                    const float3 tpost    = load_rg_tpost(g_pathStateBuffer, pixelIdx);
                    const float3 tpostNEE = tpost * bdataNEE.val * cosSurf;
                    AddInitialCandidate(wsum, g_Reservoirs_current, pixelIdx, wi,
                        ps.x2, ps.n2_s,
                        radiance * tpostNEE, ps.v2,
                        ps.Kd, ps.Pr, ps.Pm,
                        ps.matID, ps.objID, ps.eta,
                        F_contrib, seed);
                }
            }
        }

        //------------- SSS: chance to enter the medium instead of reflecting -------------
        //The surface still reflects via the regular BRDF (NEE above + the BSDF sample
        //below). With probability pEnter = sssWeight * Fresnel-transmittance the
        //CONTINUATION instead enters the object: a random walk relocates to the boundary
        //exit B (suffix). The first in-medium scatter S1 is the reconnection vertex
        //(reconnected like a transmission segment); a no-scatter pass-through (thin area)
        //transmits diffusely with W=1 and uses B itself as the reconnection vertex. The
        //stochastic reflect/enter split energy-weights the indirect bounce, so the chosen
        //branch needs no 1/p factor.
        if (!sssEntered && LoadIsSSS(ctx.matID))
        {
            const float fT     = 1.0f - FresnelDielectric(-rayDir, ctx.hitNormal, ctx.iors.x, ctx.iors.y).x;
            const float pEnter = saturate(LoadSSSWeight(ctx.matID) * fT);
            if (RandomFloatSingle(seed) < pEnter)
            {
                SSSWalkResult w = SubsurfaceWalk(ctx.hitPos, ctx.hitNormal, ctx.matID, seed);
                if (!w.valid) break;   //escaped / absorbed / step-cap -> dead path

                const float  sigma_t = 1.0f / max(LoadSSSRadius(ctx.matID), SSS_MIN_RADIUS);
                const float3 albedo  = saturate(LoadSSSAlbedo(ctx.matID));
                const float  gPhase  = LoadPhaseG(ctx.matID);
                const float  cosA    = abs(dot(ctx.hitNormal, w.entryDir));   //entry coupling cosine (G1)
                //the ENTRY boundary is tinted by the SURFACE albedo (regular Kd), not the
                //inside SSS albedo; captured before ctx is relocated to the white exit B.
                const float3 surfKd  = (float3)ctx.hitLocalKd;

                if (depth == 1 && w.nScatters >= 1u)
                {
                    //primary scattered: first scatter S1 is the volume reconnection vertex.
                    const float distA  = length(w.firstScatterPos - ctx.hitPos);
                    const float phaseF = EvaluatePhaseHG(gPhase, dot(w.entryDir, w.firstScatterDir));
                    const float pdfFac = SSS_INV_PI * cosA * exp(-sigma_t * distA) * sigma_t * phaseF;

                    store_ps_depth1(g_pathStateBuffer, pixelIdx,
                                    w.firstScatterPos, w.firstScatterDir,
                                    ctx.matID | MATID_SSS_VOLUME_BIT, ctx.instID, 1.0f,
                                    albedo, 1.0f, 0.0f);
                    store_ps_v2(g_pathStateBuffer, pixelIdx, w.firstScatterDir);
                    store_rg_tpost(g_pathStateBuffer, pixelIdx, w.wRest);   //first-scatter albedo lives in F2
                    pdf_product = min(pdf_product * max(pdfFac, 1e-20f), 1e30f);
                    sssActive   = true;
                }
                else if (depth == 1)
                {
                    //primary no-scatter (thin area): the boundary exit B is itself the v2
                    //reconnection vertex (white diffuse re-emergence). MATID_SSS_EXIT_BIT
                    //records that x1's coupling is the 1/pi diffuse entry (not its reflective
                    //BRDF) so Reconnect's target matches this forward F. The path continues
                    //from B normally at depth==2 (sssActive stays false). W=1.
                    store_ps_depth1(g_pathStateBuffer, pixelIdx,
                                    w.exitPos, w.exitNormal,
                                    ctx.matID | MATID_SSS_EXIT_BIT, ctx.instID, 1.0f,
                                    float3(1, 1, 1), 1.0f, 0.0f);
                    pdf_product = min(pdf_product * max(SSS_INV_PI * cosA, 1e-20f), 1e30f);
                    sssActive   = false;
                }
                else if (depth == 2)
                {
                    //indirect: re-bake the SSS ENTRY SURFACE (the v2 reconnection vertex) with
                    //Kd = the surface albedo so Reconnect's regular surface branch yields the
                    //surfKd/pi entry coupling; the whole walk is suffix. V2 = into-medium dir.
                    store_ps_depth1(g_pathStateBuffer, pixelIdx,
                                    ctx.hitPos, ctx.hitNormal,
                                    ctx.matID, ctx.instID, (float)ctx.iors.y,
                                    surfKd, 1.0f, 0.0f);
                    store_ps_v2(g_pathStateBuffer, pixelIdx, w.entryDir);
                    store_rg_tpost(g_pathStateBuffer, pixelIdx, w.wTotal);
                    pdf_product = min(pdf_product * max(SSS_INV_PI * cosA, 1e-20f), 1e30f);
                    sssActive   = true;
                }
                else
                {
                    //deep: the reconnection vertex is a shallower surface; the entry + walk are
                    //pure suffix, so fold the entry coupling (surfKd/pi*cosA) into tpost (-> L2).
                    const float3 tp = load_rg_tpost(g_pathStateBuffer, pixelIdx) * w.wTotal * (SSS_INV_PI * cosA) * surfKd;
                    store_rg_tpost(g_pathStateBuffer, pixelIdx, tp);
                    pdf_product = min(pdf_product * max(SSS_INV_PI * cosA, 1e-20f), 1e30f);
                    sssActive   = true;
                }

                //analog subsurface throughput -> prefix target (W=1 for a pass-through).
                //surfKd is the surface-albedo entry tint (matched by Reconnect's F1/F2).
                throughputPk = PackRGB9E5(UnpackRGB9E5(throughputPk) * w.wTotal * surfKd);
                sssEntered   = true;

                //re-emerge at the boundary exit B as a white diffuse surface seen from outside
                ctx.hitPos         = w.exitPos;
                ctx.hitNormal      = w.exitNormal;
                ctx.hitLocalKd     = (half3)float3(1, 1, 1);
                ctx.hitLocalPr     = (half)1.0f;
                ctx.hitLocalPm     = (half)0.0f;
                ctx.iors           = (half2)float2(1.0f, 1.0f);
                ctx.mediumMatID    = MEDIUM_INVALID;
                ctx.absorptionTint = (half3)float3(1, 1, 1);
                rayDirPk           = PackNormal(-w.exitNormal);   //outgoing -rayDir = exitNormal (outward)
                continue;
            }
        }

        //------------- BSDF sample (MIS partner for NEE; bottom trace evaluates it) -------------
        //sp hoisted to the top of the bounce (reused by both NEE techniques above).
        const float3 s       = SampleBRDF        (sp, ctx.matID, -rayDir, ctx.hitNormal, ctx.hitNormal, ctx.hitLocalKd, ctx.hitLocalPr, ctx.hitLocalPm, seed, ctx.iors.x, ctx.iors.y, false);
        const BrdfData bdata = EvaluateAndPdf_COMBINED(sp, ctx.matID, ctx.hitNormal, ctx.hitNormal, s, -rayDir, ctx.hitLocalKd, ctx.hitLocalPr, ctx.hitLocalPm, ctx.iors.x, ctx.iors.y, false);

        const float  cosTheta     = abs(dot(ctx.hitNormal, s));
        const float3 updateWeight = (bdata.pdf > 1e-6f)
            ? (bdata.val * ctx.absorptionTint * cosTheta) / bdata.pdf
            : float3(0, 0, 0);

        //state update + termination. bad BSDF sample -> no valid next direction
        if (dot(s, s) < 1e-12f || bdata.pdf <= 1e-6f ||
            any(isnan(updateWeight)) || any(isinf(updateWeight)))
            break;

        prev_pdf     = bdata.pdf;
        pdf_product  = min(pdf_product * bdata.pdf, 1e30f);
        rayDir       = s;
        rayDirPk     = PackNormal(s);  //carrier for next iter back-edge

        //v_2 reconnection dir (v_3 -> v_2 = -s), stored at the depth==2 bottom so
        //the depth==2 miss/emitter candidates read the real direction. Skipped when
        //sssActive: the SSS walk already wrote V2 (first-scatter / into-medium dir).
        if (depth == 2 && !sssActive)
            store_ps_v2(g_pathStateBuffer, pixelIdx, -rayDir);

        const float3 offsetN = (dot(s, ctx.hitNormal) >= 0.0f) ? ctx.hitNormal : -ctx.hitNormal;
        rayOrigin    = offset_ray(ctx.hitPos, offsetN);
        //park rayOrigin in scratch (RAY_O, free under raygen) so it doesn't cross
        //the reorder; reloaded post-trace.
        store_ray_origin(g_pathStateBuffer, pixelIdx, rayOrigin);

        float3 throughput        = UnpackRGB9E5(throughputPk) * updateWeight;
        const float3 tpostWeight = bdata.val * ctx.absorptionTint * cosTheta;

        //Russian roulette (off when pt_rrStartDepth high). The 1/p boost cancels
        //in the RR-clean target F (via pdf_product) but survives in wi; must NOT
        //touch the tpost update.
        if (depth >= (int)pt_rrStartDepth)
        {
            const float survivalProb = max(min(1.0f, Luma(throughput)), 0.05f);
            if (RandomFloatSingle(seed) >= survivalProb) break;
            const float rrBoost = 1.0f / survivalProb;
            throughput  *= rrBoost;
            pdf_product  = min(pdf_product * survivalProb, 1e30f);
        }

        //tpost: post-v_2 suffix throughput across v_3..v_{D-1}. Every depth>=4 GI
        //reconnection candidate reads it (correctness), so it runs for depth>=3
        //regardless of RR; tpostWeight is the RR-unboosted f*cos. For a scattered primary
        //SSS path the boundary-exit vertex sits at depth==2 (already in the suffix past
        //the volume reconnection vertex), so accumulation opens one bounce earlier.
        if (depth >= 3 || (sssActive && depth >= 2))
        {
            const float3 tpost = load_rg_tpost(g_pathStateBuffer, pixelIdx) * tpostWeight;
            store_rg_tpost(g_pathStateBuffer, pixelIdx, tpost);
        }

        throughputPk = PackRGB9E5(throughput);
        prevNormalPk = PackNormal(ctx.hitNormal);

        //----- TRACE v_depth -> v_{depth+1} (BSDF technique, NEE's MIS partner) -----
        if (!IsRayValid(rayOrigin, rayDir, 10000.0f))
            break;

        RayDesc rayB;
        rayB.Origin    = rayOrigin;
        rayB.Direction = rayDir;
        rayB.TMin      = 0.00001f;
        rayB.TMax      = RAY_TMAX_PLANET;
        //secondaries force 2-state to skip alpha any-hit
        dx::HitObject hitObjB = TraceRay_Custom(SceneBVH, rayB, RAY_FLAG_FORCE_OMM_2_STATE, 0xFF);

        //reload rayOrigin post-trace (globallycoherent forces the real read,
        //keeping it off the reorder — same trick as tpost).
        const float3 rayOriginR = load_ray_origin(g_pathStateBuffer, pixelIdx);

        //----- MISS: GI miss tail (sky + MIS sun disc) -----
        if (!hitObjB.IsHit())
        {
            const float3 throughputCur = UnpackRGB9E5(throughputPk);

            //evaluate sky/sun from the BOUNCE origin (observer-dependent; the
            //camera observer baked a nadir ring into indirect light).
            SetSkyObserver(rayOriginR + sceneOriginWorld);

            const float  sunSAPdf   = GetSunPdf(rayDir);
            const float3 sunRad     = (sunSAPdf > 0.0f) ? EvaluateSun(rayDir) : float3(0, 0, 0);
            const float  sunMisBsdf = (sunSAPdf > 0.0f)
                ? prev_pdf / max(prev_pdf + sunSAPdf, EPSILON) : 0.0f;
            float3 cloudTr;
            const float3 sky        = EvaluateSky(rayDir, cloudTr);
            const float3 envL       = sky + sunRad * sunMisBsdf * cloudTr;

            const float3 F_contrib = throughputCur * envL * pdf_product;
            const float  p_hat     = GetPHat(F_contrib);
            const float  wi        = (pdf_product > 1e-20f) ? (p_hat / pdf_product) : 0.0f;

            //depth==1 miss = x1 direct sky -> directAtX1; depth>=2 uses stored v_2.
            if (depth == 1)
            {
                add_rg_directX1(g_pathStateBuffer, pixelIdx, DirectContribution(wi, F_contrib));
            }
            else
            {
                const PathVertexState ps = load_ps(g_pathStateBuffer, pixelIdx);
                const float3 tpost = load_rg_tpost(g_pathStateBuffer, pixelIdx);
                AddInitialCandidate(wsum, g_Reservoirs_current, pixelIdx, wi,
                    ps.x2, ps.n2_s,
                    envL * tpost, ps.v2,
                    ps.Kd, ps.Pr, ps.Pm,
                    ps.matID, ps.objID, ps.eta,
                    F_contrib, seed);
            }
            break;
        }

        //----- HIT: extract v_{depth+1} -----
        const float  hitT_n   = hitObjB.GetRayTCurrent();
        const float3 hitPos_n = rayOriginR + rayDir * hitT_n;
        const uint   instID_n = hitObjB.GetInstanceID();
        const uint    primID_n = FlatPrimID(instID_n, hitObjB.GetGeometryIndex(), hitObjB.GetPrimitiveIndex());
        const uint    matID_n  = GetMatIDFast(instID_n, primID_n);
        BuiltInTriangleIntersectionAttributes attrB;
        hitObjB.GetAttributes(attrB);
        HitInfo hinfo_n = EvalSurfaceState(instID_n, primID_n, attrB.barycentrics, rayOriginR, (uint)depth);

        const float  matNi_n        = LoadNi(matID_n);
        const bool   transmissive_n = LoadKd_w(matID_n) < 1.0f - EPSILON;
        //thin glass: single interface, never enters a medium (see Pass_camera for the rationale)
        const bool   flipIOR_n      = hinfo_n.backface && transmissive_n && !LoadIsThinGlass(matID_n);
        const float2 iors_n         = flipIOR_n ? float2(matNi_n, 1.0f) : float2(1.0f, matNi_n);
        const uint   mediumMatID_n  = flipIOR_n ? matID_n : MEDIUM_INVALID;

        float3 hitLocalKd_n; float hitLocalPr_n, hitLocalPm_n;
        RefetchMaterial(matID_n, hinfo_n.uv, hitLocalKd_n, hitLocalPr_n, hitLocalPm_n, (uint)depth);

        const float3 absorptionTint_n = (mediumMatID_n != MEDIUM_INVALID)
            ? CalculateAbsorptionThroughput(LoadTf(mediumMatID_n), hitT_n)
            : float3(1, 1, 1);

        //emission via the lightID EvalSurfaceState already resolved (GetEmissionFast
        //re-walked instanceProps.triToLightBase + gTriToLightId for the same value).
        //hinfo lightID is backface-nulled, which only zeroes emission in cases the
        //old `&& lightID != -1` guard rejected anyway — branch decisions unchanged.
        const float3 emission_n = (hinfo_n.lightID != 0xFFFFFFFFu)
            ? g_EmissiveTriangles[hinfo_n.lightID].emission * GLOBAL_EMISSION_STRENGTH
            : float3(0, 0, 0);

        //----- Emitter hit at v_{depth+1}: BSDF-technique direct emission -----
        if (any(emission_n > 0.0f))
        {
            const float3 throughputCur = UnpackRGB9E5(throughputPk);
            const float3 prevNormalCur = UnpackNormal(prevNormalPk);
            const float  lightPdfArea  = LT_Pdf_LightTree_Area(rayOriginR, prevNormalCur, hinfo_n.lightID, instID_n);
            const float  cosLight      = max(dot(hinfo_n.hitNormal, -rayDir), 0.0f);
            const float  dist2         = max(hitT_n * hitT_n, EPSILON);
            const float  lightPdfSA    = (cosLight > EPSILON) ? (lightPdfArea * dist2 / cosLight) : 0.0f;
            const float  misWeight     = prev_pdf / max(prev_pdf + lightPdfSA, EPSILON);

            const float3 F_contrib = throughputCur * emission_n * pdf_product;
            const float  p_hat     = GetPHat(F_contrib);
            const float  wi        = (pdf_product > 1e-20f) ? (misWeight * p_hat / pdf_product) : 0.0f;

            if (depth == 1)
            {
                AddInitialCandidate(wsum, g_Reservoirs_current, pixelIdx, wi,
                    hitPos_n, hinfo_n.hitNormal,
                    emission_n, diMarkerFor(pixelIdx, time),
                    float3(0,0,0), 0.0f, 0.0f,
                    MATID_LIGHT_TRI, instID_n, 1.0f,
                    F_contrib, seed);
            }
            else
            {
                const PathVertexState ps = load_ps(g_pathStateBuffer, pixelIdx);
                const float3 tpost = load_rg_tpost(g_pathStateBuffer, pixelIdx);
                AddInitialCandidate(wsum, g_Reservoirs_current, pixelIdx, wi,
                    ps.x2, ps.n2_s,
                    emission_n * tpost, ps.v2,
                    ps.Kd, ps.Pr, ps.Pm,
                    ps.matID, ps.objID, ps.eta,
                    F_contrib, seed);
            }
            break;
        }

        //----- Non-emitter hit: stash v_2 PathVertexState at the first bounce -----
        if (depth == 1)
        {
            store_ps_depth1(g_pathStateBuffer, pixelIdx,
                            hitPos_n, hinfo_n.hitNormal,
                            matID_n, instID_n, iors_n.y,
                            hitLocalKd_n, hitLocalPr_n, hitLocalPm_n);
        }

        //----- Carry v_{depth+1} into ctx (bounded fields -> half) -----
        ctx.hitPos         = hitPos_n;
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

    }   // end initial-sample loop (s)

    //====================================
    //FINALIZE
    //====================================
    //x1 direct sky+sun -> scratch slot 3 (mean of the N single-sample estimates).
    const float3 directAtX1    = load_rg_directX1(g_pathStateBuffer, pixelIdx);
    const float3 directAtX1Avg = directAtX1 / max(1.0f, (float)pt_initialSamples);
    gScratchPing[uint3(pixel, 3)] = float4(X1_DIRECT_OFF ? float3(0, 0, 0) : directAtX1Avg, 0);

    //RIS-over-N: wsum sums to ~N, so /N for unbiased W (selection is scale-invariant).
    wsum /= max(1.0f, (float)pt_initialSamples);
    FinalizeReservoir(pixelIdx, wsum);
}
