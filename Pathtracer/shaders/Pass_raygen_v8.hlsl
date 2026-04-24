#include "Includes_v8.hlsli"
#include "Nrc_v8.hlsli"

//Overall depth cap — only fires as a safety net; Russian roulette
//(starting at depth > 2) is the actual termination mechanism for
//both rendering and training paths now. The NRC cache further
//shortens rendering paths via area-spread termination.
#ifndef MAX_BOUNCES
#define MAX_BOUNCES 32
#endif

//====================================================================
//RAYGEN, UNIFIED DI + GI RESERVOIR
//====================================================================
//Direct-lighting samples, d = 2 (x0 -> x1 -> light), now compete with GI
//samples, d >= 3, in the same reservoir. DI candidates carry sentinel
//matIDs (MATID_ENV_MISS or MATID_LIGHT_TRI) and their DI-specific x2
//payload, GI candidates reuse the stashed depth-1 vertex state.
//
//Live register state (hot): seed, rayOrigin, rayDir, throughputPk,
//prevNormalPk, prev_pdf, pdf_product, tpost, wsum. That's it.
//
//Cold state pushed to memory:
//depth-1 vertex (x2, n2, uv, matID, objID, eta): write once at depth=1
//v2 direction: write once at depth=2
//RIS F (float3 contribution): written directly to the reservoir buffer on acceptance.
//All are compressed SoA in g_pathStateBuffer. Raygen-only, Pass_spat_gi_*
//overwrites the buffer later in the frame.

[shader("raygeneration")]
void Pass_raygen_v8()
{
    const uint2 pixel    = DispatchRaysIndex().xy;
    const uint2 imgSize  = DispatchRaysDimensions().xy;
    const uint  pixelIdx = MapPixelID(imgSize, pixel);

    //====================================================================
    //NRC PATH CLASSIFICATION + SER SORT
    //====================================================================
    //Three work classes inside raygen:
    //  0 render  — short prefix, cache-terminate after x2
    //  1 train B — prefix + ~1-vertex suffix, cache-terminate suffix
    //  2 train U — prefix + full RR tail, no cache query (1/16 of train)
    //Class 2 paths are long; sorting by class groups them into coherent
    //warps. Screen bucket keeps memory accesses coalesced within class.
    //Runtime toggles come from the push cbuffer (see NrcSettings in the
    //editor). Each flag is INDEPENDENT — "enable cache" controls only
    //cache termination, "train" only training-vertex emission. Raygen
    //no longer reads the "debug view" flag; that toggle only affects
    //the three debug-chain passes scheduled after training.
    const bool kNrcEnabled    = NrcIsEnabled();
    const bool kNrcTrainOn    = NrcIsTrainOn();
    // Dynamic tile side per paper §3.5 — renderer adjusts each frame
    // from the previous frame's vertex count. Modulo (not bitmask)
    // because the side may not be a power of 2 once the feedback loop
    // settles on a non-power scaling.
    const uint  nrcTileSide   = NrcTrainingTileSide();
    const uint2 nrcTileOffset = uint2(asuint(time) % nrcTileSide,
                                      (asuint(time) / nrcTileSide) % nrcTileSide);
    bool nrcIsTraining, nrcIsUnbiased;
    uint nrcPathClass = NrcClassifyPixel(pixel, asuint(time), nrcTileSide, nrcTileOffset,
                                          nrcIsTraining, nrcIsUnbiased);
    nrcIsTraining = nrcIsTraining && kNrcTrainOn;
    if (!nrcIsTraining) nrcPathClass = NRC_CLASS_RENDER;
    // else keep whatever NrcClassifyPixel returned — CLASS_TRAIN_BIASED
    // (cache-terminates into both the reservoir AND the training tail,
    // 15/16 of training pixels) or CLASS_TRAIN_UNBIASED (runs to natural
    // termination, 1/16 of training pixels — injects ground truth).

    {
        const uint nrcSortKey =
            (nrcPathClass << 6) |
            (((pixel.y >> 4) & 0x7u) << 3) |
            ((pixel.x >> 4) & 0x7u);
        dx::MaybeReorderThread(nrcSortKey, 8);
    }

    NrcClearPendingGI(pixelIdx);

    //Cache-term eligibility, per class:
    //  CLASS_RENDER          — cache-term (driver of rendering quality)
    //  CLASS_TRAIN_BIASED    — cache-term (terminal slot seeds self-training)
    //  CLASS_TRAIN_UNBIASED  — never (runs to natural termination)
    // Debug view does NOT gate this. The debug path is a separate
    // W×H inference pass scheduled after resolve + train, so it never
    // needs to short-circuit the main cache-termination logic.
    const bool  nrcCacheEligible = kNrcEnabled &&
        (nrcPathClass == NRC_CLASS_TRAIN_BIASED ||
         nrcPathClass == NRC_CLASS_RENDER);
    // Matches host-side nrc::AlignBatch(W*H).
    const uint  nrcInferenceCapacity = (IMG_W * IMG_H + 255u) & ~255u;
    float nrcA0 = 0.0f;   //paper eq. 4, primary visible area
    float nrcA  = 0.0f;   //paper eq. 3, running √-sum (spread = nrcA²)
    int   nrcHitIdx = 0;  //1 = x1 after first real (non-pass-through) hit
    bool  nrcCacheTerminated = false;

    // Training-path state. Allocate one path id per training pixel.
    // If the per-frame quota is exhausted, skip training for this pixel
    // and render as a normal (class-0) pixel.
    uint   nrcPathId     = NRC_INVALID_PATH;
    uint   nrcTrainVIdx  = 0u;
    uint   nrcTailKind   = NRC_TAIL_INVALID;
    uint   nrcTailRadPk  = 0u;
    uint   nrcTailInfSlot = NRC_INVALID_SLOT;   //cache-termination slot for class-1 tail
    float3 nrcLNeeAccum  = float3(0, 0, 0);     //reset per vertex, feeds training target
    //Tracks whether the BSDF sample at the PREVIOUS vertex was a
    //smooth-specular (mirror / refraction). Paper §3.4 uses the
    //angular density p(ω) in the area-spread formula; for a
    //truly specular BSDF p→∞ and the segment's contribution to
    //the running spread is zero. Our GGX pdf is finite even at
    //r=0 (α²≥10⁻³ clamp), so if we didn't filter, every
    //glass-through-to-diffuse path would trip the cache-term
    //condition immediately. This flag makes specular segments
    //"free" in the spread accumulation and defers the termination
    //check by one vertex after a specular bounce — which matches
    //the paper's intent and fixes the "one extra bounce needed
    //after a car window" case.
    bool   nrcPrevSpecular = false;
    if (nrcIsTraining)
    {
        nrcPathId = NrcAllocateTrainingPath();
        if (nrcPathId == NRC_INVALID_PATH) { nrcIsTraining = false; nrcPathClass = NRC_CLASS_RENDER; }
    }

    //Clear scratch slots 1 and 3. Slot 1 carries primary emitter/sky
    //hits, written below on depth-0 emission. Slot 3 (formerly the
    //separate direct-light buffer) is now always zero — depth=0 sun
    //NEE and depth=1 env miss both feed the unified reservoir. The
    //clear prevents shading from reading last frame's contents.
    gScratchPing[uint3(pixel, 1)] = float4(0, 0, 0, 0);
    gScratchPing[uint3(pixel, 3)] = float4(0, 0, 0, 0);

    //Zero the reservoir. RIS acceptance overwrites on demand, this ensures
    //no-candidate-accepted pixels end up as empty reservoirs.
    storeReservoir(g_Reservoirs_current, pixelIdx, (Reservoir)0);

    //Seed the PathVertexState slot with safe sentinel defaults. Without
    //this, the pass-through continue at matNi <= 1+EPSILON, and any other
    //depth=1 early-break, leaves last frame's spatial-pass scratch in the
    //slot, which load_ps then decodes into a bogus matID/objID and hangs
    //the driver at specific camera angles.
    init_ps(g_pathStateBuffer, pixelIdx);

    //RIS state, only wsum is live across iterations. F is written
    //straight to the reservoir buffer on acceptance.
    float wsum = 0.0f;

    uint   seed         = initRandomData(pixel, uint2(8, 4), time, 1u);
    float3 rayOrigin    = InitOrigin();
    float3 rayDir       = InitDirection(pixel, float2(imgSize), seed);
    uint   throughputPk = PackRGB9E5(float3(1, 1, 1));
    uint   prevNormalPk = PackNormal(float3(0, 1, 0));
    float  prev_pdf     = 1.0f;
    float  pdf_product  = 1.0f;

    //tpost, post-x2 integrand accumulator, stays in registers. Updated
    //every bounce, so buffer RMW would be strictly worse than 3 scalars.
    float3 tpost = float3(1, 1, 1);

    //Pixel-and-frame-unique unit vector used as the stored V2 for DI
    //samples (env miss, emitter hit, NEE at d=0). Reconnect's sentinel
    //branches ignore V2, so this is purely a dup-map discriminator for
    //Pass_dup_gi_v8 — without it, all DI samples across the image would
    //share the same packed V2 and flood the duplication count.
    float3 diMarker;
    {
        uint h = (pixelIdx * 0x9E3779B9u) ^ (asuint(time) * 0x85EBCA6Bu);
        h ^= h >> 16; h *= 0xC2B2AE35u;
        h ^= h >> 13; h *= 0x27D4EB2Fu;
        h ^= h >> 16;
        const float u1 = (float)(h & 0xFFFFu) * (1.0f / 65535.0f);
        const float u2 = (float)((h >> 16) & 0xFFFFu) * (1.0f / 65535.0f);
        const float z  = 2.0f * u1 - 1.0f;
        const float r  = sqrt(max(0.0f, 1.0f - z * z));
        const float phi = 6.2831853f * u2;
        diMarker = float3(r * cos(phi), r * sin(phi), z);
    }

    //====================================================================
    //PATH LOOP
    //====================================================================
    [loop]
    for (int depth = 0; depth < MAX_BOUNCES; ++depth)
    {
        if (!IsRayValid(rayOrigin, rayDir, 10000.0f))
            break;

        RayDesc ray;
        ray.Origin    = rayOrigin;
        ray.Direction = rayDir;
        ray.TMin      = 0.00001f;
        ray.TMax      = 10000.0f;
        //Depth 0 keeps 4-state OMM, unknown states fall through to the
        //alpha any-hit. Secondary bounces force 2-state: unknown is
        //resolved by the pre-baked micromap and any-hit is skipped, which
        //is where the bulk of the alpha-test cost lives.
        const uint rayFlags = (depth == 0) ? RAY_FLAG_NONE : RAY_FLAG_FORCE_OMM_2_STATE;
        dx::HitObject hitObj = TraceRay_Custom(SceneBVH, ray, rayFlags, 0xFF);

        //====================================================================
        //MISS
        //====================================================================
        if (!hitObj.IsHit())
        {
            if (depth == 0)
            {
                float3 sun   = EvaluateSun(rayDir);
                float3 skyL1 = EvalMissState(rayDir, sun);
                if (length(sun) > 0.0f) skyL1 = sun;
                gScratchPing[uint3(pixel, 1)] = float4(skyL1, 0);
                gScratchPing[uint3(pixel, 2)] = float4(skyL1, 0);
                store_sky(g_sample_current, pixelIdx);
                break;
            }

            const float3 throughput = UnpackRGB9E5(throughputPk);
            const float3 envL       = EvalMissState(rayDir, float3(0, 0, 0));

            //NRC: miss on a training path becomes the backward-fill tail.
            if (nrcIsTraining) {
                nrcTailKind  = NRC_TAIL_MISS;
                nrcTailRadPk = PackRGB9E5(NrcCleanRadiance(envL));
            }

            //pdf-free contribution f = throughput * pdf_product * envL
            const float3 F_contrib = throughput * envL * pdf_product;
            const float  p_hat     = GetPHat(F_contrib);
            const float  p_full    = pdf_product;
            const float  wi        = (p_full > 1e-20f) ? (p_hat / p_full) : 0.0f;

            if (depth == 1)
            {
                //DI env miss (d=2 path), x2 stored as a DIRECTION.
                //MATID_ENV_MISS makes Reconnect's env-miss branch
                //preserve direction under shift, so reuse is cheap
                //and Jacobian = 1. V2 = diMarker is the dup-map
                //discriminator (Reconnect's sentinel branch ignores V2).
                //Visibility is implicit: the BSDF-sampled ray already
                //reached the environment.
                AddInitialCandidate(wsum, g_Reservoirs_current, pixelIdx, wi,
                    rayDir, float3(0, 1, 0),          //x2=dir, n2=unit default
                    envL,   diMarker,                 //L2, V2=per-pixel discriminator
                    float2(0, 0),
                    MATID_ENV_MISS, MATID_ENV_MISS, 1.0f,
                    F_contrib, seed);
            }
            else //depth >= 2: GI env, x2 = stashed depth-1 vertex
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

        //====================================================================
        //HIT SETUP
        //====================================================================
        const float  hitT   = hitObj.GetRayTCurrent();
        const float3 hitPos = rayOrigin + rayDir * hitT;

        const uint instID = hitObj.GetInstanceIndex();
        const uint primID = FlatPrimID(instID, hitObj.GetGeometryIndex(), hitObj.GetPrimitiveIndex());
        const uint matID  = GetMatIDFast(instID, primID);

        BuiltInTriangleIntersectionAttributes attr;
        hitObj.GetAttributes(attr);
        HitInfo hinfo = EvalSurfaceState(instID, primID, attr.barycentrics, rayOrigin, depth);

        //Backface-derived IOR pair: entering -> (air, matNi), exiting -> (matNi, air).
        //Null-IOR boundary, matNi ~= 1, is pure pass-through, skip the hit.
        const float matNi = LoadNi(matID);
        if (matNi <= 1.0f + EPSILON)
        {
            rayOrigin = hitPos;
            continue;
        }

        //Only flip the IOR pair, and declare an interior medium for
        //absorption, when the material is actually transmissive. An
        //opaque backface, thin single-sided geometry like leaves or paper,
        //or an inverted winding, has an IOR but no traversable inside,
        //swapping would apply a phantom Tf absorption to the incoming leg.
        const bool transmissive = LoadKd_w(matID) < 1.0f - EPSILON;
        const bool flipIOR = hinfo.backface && transmissive;
        const float2 iors = flipIOR ? float2(matNi, 1.0f) : float2(1.0f, matNi);
        const uint   mediumMatID = flipIOR ? matID : MEDIUM_INVALID;

        float3 hitLocalKd; float hitLocalPr, hitLocalPm;
        RefetchMaterial(matID, hinfo.uv, hitLocalKd, hitLocalPr, hitLocalPm, (uint)depth);

        const float3 absorptionTint = (mediumMatID != MEDIUM_INVALID)
            ? CalculateAbsorptionThroughput(LoadTf(mediumMatID), hitT)
            : float3(1, 1, 1);

        const float3 emission = GetEmissionFast(instID, primID);

        //====================================================================
        //DEPTH 0, PRIMARY HIT STORAGE
        //====================================================================
        if (depth == 0)
        {
            const bool isEmitter = any(emission > 0.0f);
            store_instID(g_sample_current, pixelIdx, instID);
            store_primID(g_sample_current, pixelIdx, primID, isEmitter);
            store_bary  (g_sample_current, pixelIdx, attr.barycentrics);
            store_n1_s_world(g_sample_current, pixelIdx, hinfo.hitNormal, instID);
            store_uv    (g_sample_current, pixelIdx, hinfo.uv);
            if (isEmitter) {
                gScratchPing[uint3(pixel, 1)] = float4(emission, 0);
                gScratchPing[uint3(pixel, 2)] = float4(emission, 0);
            }

            //Specular motion vector reflection probe
            {
                const float3 reflDir    = reflect(rayDir, hinfo.hitNormal);
                const float3 reflOrigin = offset_ray(hitPos, hinfo.hitNormal);

                bool committed = false;
                float reflT = 0.0f;
                if (IsRayValid(reflOrigin, reflDir, 10000.0f))
                {
                    RayDesc reflRay;
                    reflRay.Origin    = reflOrigin;
                    reflRay.Direction = reflDir;
                    reflRay.TMin      = 0.00001f;
                    reflRay.TMax      = 10000.0f;

                    RayQuery<RAY_FLAG_NONE> q;
                    q.TraceRayInline(SceneBVH, RAY_FLAG_NONE, 0xFF, reflRay);
                    //Hard cap on alpha-test iterations, a grazing reflection
                    //ray through dense foliage can produce hundreds of
                    //non-opaque candidates, and certain driver states hit
                    //TDR before BVH traversal finishes. 128 is far more
                    //than any motion-vector probe needs.
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
                    if (committed) reflT = q.CommittedRayT();
                }

                if (committed)
                {
                    float3 reflPos    = reflOrigin + reflDir * reflT;
                    float3 virtualPos = reflPos - 2.0f * dot(reflPos - hitPos, hinfo.hitNormal) * hinfo.hitNormal;
                    gScratchPing[uint3(pixel, 4)] = float4(virtualPos, asfloat(instID));
                }
                else
                {
                    gScratchPing[uint3(pixel, 4)] = float4(0, 0, 0, asfloat(0xFFFFFFFFu));
                }
            }
        }

        //====================================================================
        //EMITTER HIT
        //====================================================================
        if (any(emission > 0.0f) && hinfo.lightID != 0xFFFFFFFFu)
        {
            const float3 throughput = UnpackRGB9E5(throughputPk);
            const float3 prevNormal = UnpackNormal(prevNormalPk);
            const float  lightPdfArea = LT_Pdf_LightTree_Area(rayOrigin, prevNormal, hinfo.lightID, instID);
            const float  cosLight     = max(dot(hinfo.hitNormal, -rayDir), 0.0f);
            const float  dist2        = max(hitT * hitT, EPSILON);
            const float  lightPdfSA   = (cosLight > EPSILON) ? (lightPdfArea * dist2 / cosLight) : 0.0f;
            const float  misWeight    = prev_pdf / max(prev_pdf + lightPdfSA, EPSILON);

            //NRC: emitter hit terminates training paths. Tail must be
            //MIS_BSDF-weighted (misWeight above) — the previous vertex's
            //L_nee is already MIS_NEE-weighted, and combining both with
            //MIS weights that sum to 1 is the only way to avoid
            //double-counting this emitter in the backward-fill target.
            if (nrcIsTraining) {
                nrcTailKind  = NRC_TAIL_EMITTER;
                nrcTailRadPk = PackRGB9E5(NrcCleanRadiance(emission * misWeight));
            }

            const float3 F_contrib = throughput * emission * pdf_product;
            const float  p_hat     = GetPHat(F_contrib);
            const float  p_full    = pdf_product;
            const float  wi        = (p_full > 1e-20f) ? (misWeight * p_hat / p_full) : 0.0f;

            bool accepted = false;
            if (depth == 1)
            {
                //DI triangle-emitter candidate, d=2 path.
                //V2 = diMarker as dup-map discriminator, unused by
                //Reconnect's MATID_LIGHT_TRI branch.
                accepted = AddInitialCandidate(wsum, g_Reservoirs_current, pixelIdx, wi,
                    hitPos, hinfo.hitNormal,
                    emission, diMarker,               //V2 = per-pixel discriminator
                    float2(0, 0),
                    MATID_LIGHT_TRI, instID, 1.0f,
                    F_contrib, seed);
            }
            else //depth >= 2: GI emitter, x2 = stashed depth-1 vertex
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

        //====================================================================
        //STASH DEPTH-1 VERTEX / V2
        //====================================================================
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

        //====================================================================
        //NRC AREA-SPREAD + CACHE TERMINATION
        //====================================================================
        //Counts only real (non-pass-through) hits; the pass-through `continue`
        //above already skipped thin-glass boundaries. nrcHitIdx maps to the
        //paper's path-vertex index (x1 at 1, x2 at 2, ...).
        //
        //Bekaert area-spread heuristic, gated at hitIdx >= 3 so ReSTIR PT's
        //reconnection at x2 always has a real BSDF to evaluate. Specular
        //bounces are skipped via nrcPrevSpecular so that glass/mirror
        //chains don't short-circuit onto the first diffuse they hit
        //(paper §3.4). Class-2 (TRAIN_UNBIASED) paths are not cache-
        //eligible and continue to natural termination.
        ++nrcHitIdx;
        //Reset per-vertex NEE accumulator — NEE blocks below add into it,
        //the training-vertex write captures the sum.
        nrcLNeeAccum = float3(0, 0, 0);

        if (nrcHitIdx == 1)
        {
            const float cosPrimary = max(abs(dot(rayDir, hinfo.hitNormal)), 1e-6f);
            nrcA0 = NrcComputeA0(hitT, cosPrimary);
        }
        else
        {
            if (!nrcPrevSpecular)
            {
                const float cosHit = max(abs(dot(-rayDir, hinfo.hitNormal)), 1e-6f);
                NrcAccumulateA(nrcA, hitT, prev_pdf, cosHit);
            }

            const bool shouldFire =
                !nrcPrevSpecular &&
                NrcShouldCacheTerminate(nrcHitIdx, nrcA0, nrcA, nrcCacheEligible, nrc_area_spread_c);

            if (shouldFire)
            {
                //Cache-terminate here. Write the inference request + the
                //PendingGI record that the resolve compute shader will turn
                //into a reservoir candidate once L̂_s is known. Skip NEE and
                //the next BSDF sample — the cache subsumes both at this
                //vertex. Raygen's final resolve will see nrcCacheTerminated
                //and defer W/M to Pass_nrc_resolve_v8.
                //
                //If the slot allocation fails (inference buffer saturated),
                //let the path continue normally — losing one cache query is
                //better than losing the whole pixel's GI contribution.
                const float3 throughput = UnpackRGB9E5(throughputPk);
                const uint nrcSlot = NrcWriteTerminationRecord(
                        pixelIdx, nrcInferenceCapacity,
                        hitPos, -rayDir, hinfo.hitNormal,
                        hitLocalPr, hitLocalKd, hitLocalPm,
                        throughput, tpost, pdf_product);
                if (nrcSlot != NRC_INVALID_SLOT)
                {
                    nrcCacheTerminated = true;
                    // Class-1 training paths use the SAME query as the
                    // rendering tail — the cache-terminal vertex's L̂_s
                    // bootstraps the backward fill (self-training).
                    //
                    // Reflectance factorisation lives in the network:
                    // MLP output = L_s / (α+β) at the queried vertex, so
                    // recovering radiance from the prediction requires
                    // multiplying by (α+β) *at that same vertex*, not
                    // at the last training vertex before it. We pack
                    // (α+β)(cache-term) into the TailRadiance slot of
                    // the path meta (unused for kTailCache otherwise)
                    // so the fill kernel can apply the right factor.
                    if (nrcPathClass == NRC_CLASS_TRAIN_BIASED) {
                        nrcTailKind    = NRC_TAIL_CACHE;
                        nrcTailInfSlot = nrcSlot;
                        const float3 alphaCT = hitLocalKd * (1.0f - hitLocalPm);
                        const float3 betaCT  = lerp(float3(0.04f, 0.04f, 0.04f),
                                                    hitLocalKd, hitLocalPm);
                        nrcTailRadPk = PackRGB9E5(NrcCleanRadiance(alphaCT + betaCT));
                    }
                    break;
                }
            }
        }

        //====================================================================
        //NEE
        //====================================================================
        const bool performNEE = !(mediumMatID != MEDIUM_INVALID || LoadKd_w(matID) < EPSILON);

        uint matKdPk, matPrPmPk, hitNormalPk;
        if (performNEE)
        {
            matKdPk     = PackRGB9E5(hitLocalKd);
            matPrPmPk   = f32tof16_custom(hitLocalPr) | (f32tof16_custom(hitLocalPm) << 16u);
            hitNormalPk = PackNormal(hinfo.hitNormal);

            //====================================================================
            //POINT LIGHT NEE
            //====================================================================
            //Visibility is DEFERRED, we add the candidate assuming it's
            //unoccluded, then if it wins RIS we stash the shadow-ray info
            //in the path-state scratch. The end-of-raygen resolve traces
            //exactly one shadow ray, for the final winning NEE sample.
            {
                LT_LightSampleResult light = LT_SamplePointOnLight(hitPos, hinfo.hitNormal, seed);

                const float3 toLight = light.position - hitPos;
                const float  distSq  = dot(toLight, toLight);
                const float  dist    = sqrt(distSq);
                const float3 L       = toLight / dist;

                const float  cosSurf  = dot(hinfo.hitNormal, L);
                const float  cosLightS = dot(light.normal, -L);

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
                        //Inline visibility. The previous design stashed the
                        //shadow ray and only traced it at raygen end for the
                        //RIS winner; that saved rays but didn't cope well
                        //with the sun's infinite-distance ray, so now every
                        //accepted NEE candidate traces its own shadow ray
                        //before landing in the reservoir.
                        const float3 shadowOrigin = offset_ray(
                            hitPos,
                            dot(L, hinfo.hitNormal) >= 0.0f ? hinfo.hitNormal : -hinfo.hitNormal);
                        if (IsVisibleOffset(shadowOrigin, L, dist * 0.999f))
                        {
                            const float  misWeight       = lightPdf / (lightPdf + bsdfPdf);
                            const float3 localMeasurement = light.emission * bdataNEE.val * cosSurf;
                            const float3 F_contrib        = throughput * localMeasurement * pdf_product;
                            const float  p_hat            = GetPHat(F_contrib);
                            const float  p_full           = pdf_product * lightPdf;
                            const float  wi               = (p_full > 1e-20f) ? (misWeight * p_hat / p_full) : 0.0f;

                            //NRC: local NEE estimator contribution at THIS vertex,
                            //stripped of upstream throughput / pdf_product. Feeds
                            //target[v] = L_nee[v] + β[v]·target[v+1] during the
                            //backward fill.
                            if (nrcIsTraining)
                                nrcLNeeAccum += localMeasurement * misWeight / lightPdf;

                            if (depth == 0)
                            {
                                //DI NEE at primary vertex, d=2 path, x2 = light.
                                //V2 = diMarker as dup-map discriminator,
                                //unused by Reconnect's MATID_LIGHT_TRI branch.
                                AddInitialCandidate(wsum, g_Reservoirs_current, pixelIdx, wi,
                                    light.position, light.normal,
                                    light.emission, diMarker,
                                    float2(0, 0),
                                    MATID_LIGHT_TRI, light.objID, 1.0f,
                                    F_contrib, seed);
                            }
                            else if (depth == 1)
                            {
                                //GI NEE at depth-1 vertex, d=3 path, x2 = current hit.
                                AddInitialCandidate(wsum, g_Reservoirs_current, pixelIdx, wi,
                                    hitPos, hinfo.hitNormal,
                                    light.emission, -L,
                                    hinfo.uv,
                                    matID, instID, iors.y,
                                    F_contrib, seed);
                            }
                            else //depth >= 2: GI NEE past depth-1 vertex.
                            {
                                const PathVertexState ps = load_ps(g_pathStateBuffer, pixelIdx);
                                //tpost so far excludes current BSDF, NEE adds it.
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

            //====================================================================
            //SUN NEE
            //====================================================================
            //depth >= 1 defers visibility via the shadow-ray scratch.
            //depth == 0 bypasses the reservoir, direct scratch write, and
            //still traces its shadow ray inline, there's no RIS winner
            //to gate against.
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
                        const float  misWeight       = lightPdf / (lightPdf + bsdfPdf);
                        const float3 localMeasurement = sun.radiance * bdataNEE.val * NdotL;
                        const float3 F_contrib        = throughput * localMeasurement * pdf_product;
                        const float  p_hat            = GetPHat(F_contrib);
                        const float  p_full           = pdf_product * lightPdf;

                        //Inline sun-shadow check before the candidate lands
                        //in the reservoir (same offset / tMax for all
                        //depths, only the stored sample shape differs).
                        const float3 shadowOrigin = offset_ray(
                            hitPos,
                            dot(sun.direction, hinfo.hitNormal) >= 0.0f ? hinfo.hitNormal : -hinfo.hitNormal);
                        if (IsVisibleOffset(shadowOrigin, sun.direction, 10000.0f))
                        {
                            const float wi = (p_full > 1e-20f) ? (misWeight * p_hat / p_full) : 0.0f;

                            //NRC: accumulate local sun-NEE estimator into this
                            //vertex's L_nee. Same shape as the point-light
                            //branch above — see comment there.
                            if (nrcIsTraining)
                                nrcLNeeAccum += localMeasurement * misWeight / lightPdf;

                            if (depth == 0)
                            {
                                //DI sun NEE at primary vertex, d=2 path.
                                //x2 = sun.direction stored as a DIRECTION,
                                //matID = MATID_ENV_MISS so Reconnect's
                                //sentinel branch preserves direction under
                                //shift (same as env miss). V2 = diMarker
                                //for dup-map discrimination only.
                                AddInitialCandidate(wsum, g_Reservoirs_current, pixelIdx, wi,
                                    sun.direction, float3(0, 1, 0),   //x2=dir, n2=unit default
                                    sun.radiance, diMarker,           //L2, V2=per-pixel discriminator
                                    float2(0, 0),
                                    MATID_ENV_MISS, MATID_ENV_MISS, 1.0f,
                                    F_contrib, seed);
                            }
                            else if (depth == 1)
                            {
                                AddInitialCandidate(wsum, g_Reservoirs_current, pixelIdx, wi,
                                    hitPos, hinfo.hitNormal,
                                    sun.radiance, -sun.direction,
                                    hinfo.uv,
                                    matID, instID, iors.y,
                                    F_contrib, seed);
                            }
                            else //depth >= 2
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

        //====================================================================
        //SAMPLE NEXT BSDF DIRECTION
        //====================================================================
        SamplingP sp = CalculateStrategyProbabilities(matID, -rayDir, hinfo.hitNormal, iors.x, iors.y, hitLocalKd, hitLocalPm);
        float3    s  = SampleBRDF(sp, matID, -rayDir, hinfo.hitNormal, hinfo.hitNormal, hitLocalKd, hitLocalPr, hitLocalPm, seed, iors.x, iors.y);
        BrdfData  bdata = EvaluateAndPdf_COMBINED(sp, matID, hinfo.hitNormal, hinfo.hitNormal, s, -rayDir, hitLocalKd, hitLocalPr, hitLocalPm, iors.x, iors.y);

        const float  cosTheta     = abs(dot(hinfo.hitNormal, s));
        const float3 updateWeight = (bdata.pdf > 1e-6f)
            ? (bdata.val * absorptionTint * cosTheta) / bdata.pdf
            : float3(0, 0, 0);

        //NRC: emit training vertex (features + local NEE + β = updateWeight
        //without RR boost). We write BEFORE the bad-sample break so a path
        //that dies on a degenerate BSDF still carries the last vertex for
        //training — its β is 0, so backward fill treats it as a terminal.
        if (nrcIsTraining && nrcTrainVIdx < NRC_MAX_VERTICES_PER_PATH)
        {
            float features[16];
            const float3 alpha = hitLocalKd * (1.0f - hitLocalPm);
            const float3 betaC = lerp(float3(0.04f, 0.04f, 0.04f), hitLocalKd, hitLocalPm);
            NrcBuildFeatures(hitPos, -rayDir, hinfo.hitNormal,
                             hitLocalPr, alpha, betaC, features);

            NrcStoreTrainingVertex(
                nrcPathId, nrcTrainVIdx, features,
                PackRGB9E5(NrcCleanRadiance(nrcLNeeAccum)),
                PackRGB9E5(NrcCleanRadiance(updateWeight)));
            ++nrcTrainVIdx;
        }

        if (dot(s, s) < 1e-12f || bdata.pdf <= 1e-6f || any(isnan(updateWeight)) || any(isinf(updateWeight)))
            break;

        prev_pdf    = bdata.pdf;
        pdf_product = min(pdf_product * bdata.pdf, 1e30f);
        // Classify THIS vertex's BSDF sample as specular for the next
        // iteration's area-spread logic. Uses the same threshold the
        // GGX sampler uses to degenerate to a perfect-specular H=N
        // branch (SMOOTH_SPECULAR_THRESHOLD = 0.06), so what we treat
        // as "specular for NRC purposes" matches what the BSDF itself
        // treats as specular for sampling.
        nrcPrevSpecular = (hitLocalPr < SMOOTH_SPECULAR_THRESHOLD);
        rayDir      = s;
        float3 offsetN = dot(s, hinfo.hitNormal) >= 0.0f ? hinfo.hitNormal : -hinfo.hitNormal;
        rayOrigin   = offset_ray(hitPos, offsetN);

        {
            float3 throughput  = UnpackRGB9E5(throughputPk) * updateWeight;
            float3 tpostWeight = bdata.val * absorptionTint * cosTheta;

            //Russian roulette — now applied to BOTH rendering and training
            //paths (paper §3.2 uses RR for class-2 unbiased training
            //termination). On survival we boost `throughput` by 1/p AND
            //patch the last-written training vertex's β with the same
            //factor. Without that β patch, paths surviving RR would carry
            //an under-weighted tail in the backward fill and systematically
            //darken the cache target — which is what we were seeing
            //before this change.
            //
            //It must NOT touch tpostWeight: tpost is pure BSDF·cos·abs
            //integrand (no 1/pdf term) cached into L2 for reuse. Boosting
            //it poisons reused contributions with an extra 1/p that can't
            //be cancelled at the neighbor pixel, producing firefly-like
            //overshoots whenever an RR-surviving sample propagates
            //through temporal or spatial reservoir chains.
            if (depth > 2)
            {
                const float survivalProb = max(min(1.0f, Luma(throughput)), 0.1f);
                if (RandomFloatSingle(seed) >= survivalProb) break;
                const float rrBoost = 1.0f / survivalProb;
                throughput  *= rrBoost;
                //RR survival is part of the path pdf, include in product.
                pdf_product = min(pdf_product * survivalProb, 1e30f);

                //Update the training vertex we just emitted for THIS hit
                //so its stored β = BSDF·cos·T/pdf · (1/p_rr). That keeps
                //E[β_stored · tail(v+1)] = β_unboosted · L_s(v+1) over
                //the survive/die randomness and the backward fill stays
                //unbiased. If the path dies here (the break above), the
                //stored β is irrelevant — tail is 0 so β · 0 = 0 either
                //way.
                if (nrcIsTraining && nrcTrainVIdx > 0u)
                {
                    NrcUpdateTrainingVertexBeta(
                        nrcPathId, nrcTrainVIdx - 1u,
                        PackRGB9E5(NrcCleanRadiance(updateWeight * rrBoost)));
                }
            }

            //Accumulate tpost locally, no reservoir roundtrip.
            if (depth >= 2)
                tpost *= tpostWeight;

            throughputPk = PackRGB9E5(throughput);
        }
        prevNormalPk = PackNormal(hinfo.hitNormal);
    }

    //====================================================================
    //NRC: COMMIT TRAINING PATH META
    //====================================================================
    //Walks through for every allocated training path regardless of length.
    //numVertices == 0 happens when the primary miss branch was taken or
    //the path died before any BSDF sample; meta.numVertices stays 0 and
    //the training kernel skips it. A path that terminated via Russian
    //roulette or MAX_BOUNCES safety cap without being tagged (degenerate
    //BSDF, etc.) gets NRC_TAIL_RR — the fill kernel treats that as
    //tail = 0, which is unbiased because each vertex's stored β already
    //includes its RR-survival 1/p factor.
    if (nrcPathId != NRC_INVALID_PATH)
    {
        const uint tailKind = (nrcTailKind != NRC_TAIL_INVALID) ? nrcTailKind : NRC_TAIL_RR;
        // kTailCache carries its slot in inferenceSlot; other tail kinds
        // carry their radiance in tailRadPk (fill kernel branches on kind).
        const uint infSlot = (tailKind == NRC_TAIL_CACHE) ? nrcTailInfSlot : NRC_INVALID_SLOT;
        NrcStorePathMeta(nrcPathId, nrcTrainVIdx, tailKind, infSlot, nrcTailRadPk);
    }

    //====================================================================
    //FINAL RESOLVE
    //====================================================================
    //Commit wsum / W / M. F was already written to the reservoir by
    //AddInitialCandidate on the last acceptance. NEE candidates are now
    //visibility-tested inline before they reach RIS, so the W formula
    //is the plain RIS unbiased weight.
    //
    //When NRC cache termination fired, we DEFER W/M/invalidation to
    //Pass_nrc_resolve_v8 — it RISes the cache-terminated GI candidate
    //(using the inferred L̂_s) against any DI/emitter candidates
    //raygen already RIS'd in, then computes W from the merged wsum.
    //The debug view runs as a separate inference chain after resolve,
    //so it has no bearing on the main finalization here.
    store_wsum(g_Reservoirs_current, pixelIdx, wsum);
    if (!nrcCacheTerminated)
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
