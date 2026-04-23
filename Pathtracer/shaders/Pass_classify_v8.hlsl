#include "Includes_v8.hlsli"

//====================================================================
//CLASSIFY STAGE (universal, primary + secondary)
//====================================================================
//Reads the hit packet produced by Pass_primary / Pass_extend and routes
//the path through one of four branches:
//miss         -> depth-dependent env candidate (or sky write at d=0),
//                TERMINATED.
//passthrough  -> null-IOR boundary (matNi <= 1+EPS). Advance ray origin,
//                leave HAS_VALID_HIT off, don't increment depth (the
//                wavefront CPU loop drives depth via repeated iterations).
//                Actually we DO increment depth in flags so non-hit
//                iterations still burn a bounce slot, matching the
//                monolith's `continue` + `++depth` semantics.
//emitter hit  -> MIS-weighted candidate at d>=1, scratch[1,2] = emission
//                at d=0. TERMINATED.
//real hit     -> depth-1 / depth-2 stash, primary-only side effects at
//                d=0 (g_sample_current primary storage + motion vector
//                reflection probe), stash CLAS1/CLAS2 for MatEvalShade,
//                set HAS_VALID_HIT + PERFORM_NEE (gated).
//
//Register pressure here is still substantial (material refetch +
//light-tree area PDF on emitter hit + EvalSurfaceState + motion vector
//RayQuery at d=0) but Scatter's BSDF sampler and the NEE light-tree
//descent live in downstream stages. SER on hit class would further
//separate the depth-0 primary-storage register set from the
//emitter-MIS register set; left for a follow-up pass.

[shader("raygeneration")]
void Pass_classify_v8()
{
    const uint2 pixel    = DispatchRaysIndex().xy;
    const uint2 imgSize  = DispatchRaysDimensions().xy;
    const uint  pixelIdx = MapPixelID(imgSize, pixel);

    uint flags = load_flags(g_pathStateBuffer, pixelIdx);
    if (flags & PS_FLAG_TERMINATED) return;

    const uint depth = ps_get_depth(flags);

    const HitPacket hp = load_hp(g_pathStateBuffer, pixelIdx);
    const float3 rayOrigin = load_ray_origin(g_pathStateBuffer, pixelIdx);
    const float3 rayDir    = load_ray_dir   (g_pathStateBuffer, pixelIdx);

    //Clear stale HAS_VALID_HIT / PERFORM_NEE bits from the previous bounce
    //so the hit-class routing below can set them fresh.
    flags &= ~(PS_FLAG_HAS_VALID_HIT | PS_FLAG_PERFORM_NEE
             | PS_FLAG_IS_BACKFACE   | PS_FLAG_FLIP_IOR
             | PS_FLAG_TRANSMISSIVE  | PS_FLAG_IS_EMITTER);

    //====================================================================
    //MISS
    //====================================================================
    if (!hp.isHit)
    {
        if (depth == 0)
        {
            //Primary miss: capture the sky (and sun disk if hit) into
            //the denoiser input scratch slots.
            const float3 sun   = EvaluateSun(rayDir);
            float3 skyL1 = EvalMissState(rayDir, sun);
            if (length(sun) > 0.0f) skyL1 = sun;
            gScratchPing[uint3(pixel, 1)] = float4(skyL1, 0);
            gScratchPing[uint3(pixel, 2)] = float4(skyL1, 0);
            store_sky(g_sample_current, pixelIdx);

            flags |= PS_FLAG_TERMINATED;
            store_flags(g_pathStateBuffer, pixelIdx, flags);
            return;
        }

        //Secondary miss: env candidate.
        const HotState h = load_hot(g_pathStateBuffer, pixelIdx);
        const float3 throughput = UnpackRGB9E5(h.throughputPk);
        const float3 envL       = EvalMissState(rayDir, float3(0, 0, 0));

        const float3 F_contrib = throughput * envL * h.pdf_product;
        const float  p_hat     = GetPHat(F_contrib);
        const float  p_full    = h.pdf_product;
        const float  wi        = (p_full > 1e-20f) ? (p_hat / p_full) : 0.0f;

        float wsum = h.wsum;
        uint  seed = h.seed;

        if (depth == 1)
        {
            const float3 diMarker = diMarkerFor(pixelIdx, time);
            AddInitialCandidate(wsum, g_Reservoirs_current, pixelIdx, wi,
                rayDir, float3(0, 1, 0),
                envL, diMarker,
                float2(0, 0),
                MATID_ENV_MISS, MATID_ENV_MISS, 1.0f,
                F_contrib, seed);
        }
        else
        {
            //depth >= 2: x2 = stashed depth-1 vertex; tpost carries the
            //post-x2 integrand so envL*tpost is the measurement at x2.
            const PathVertexState ps = load_ps(g_pathStateBuffer, pixelIdx);
            const float3 tpost = UnpackRGB9E5(h.tpostPk);
            AddInitialCandidate(wsum, g_Reservoirs_current, pixelIdx, wi,
                ps.x2, ps.n2_s,
                envL * tpost, ps.v2,
                ps.uv,
                ps.matID, ps.objID, ps.eta,
                F_contrib, seed);
        }

        store_ps_wsum(g_pathStateBuffer, pixelIdx, wsum);
        store_seed(g_pathStateBuffer, pixelIdx, seed);
        flags |= PS_FLAG_TERMINATED;
        store_flags(g_pathStateBuffer, pixelIdx, flags);
        return;
    }

    //====================================================================
    //HIT SETUP
    //====================================================================
    const uint instID = hp.instID;
    const uint primID = hp.primID;
    const uint matID  = GetMatIDFast(instID, primID);

    const float matNi = LoadNi(matID);

    //====================================================================
    //PASSTHROUGH (null-IOR)
    //====================================================================
    if (matNi <= 1.0f + EPSILON)
    {
        const float3 hitPos = rayOrigin + rayDir * hp.hitT;
        store_ray_origin(g_pathStateBuffer, pixelIdx, hitPos);
        flags = ps_set_depth(flags, depth + 1u);
        store_flags(g_pathStateBuffer, pixelIdx, flags);
        return;
    }

    //====================================================================
    //FULL SURFACE STATE
    //====================================================================
    HitInfo hinfo = EvalSurfaceState(instID, primID, hp.bary, rayOrigin, depth);
    const float3 hitPos = rayOrigin + rayDir * hp.hitT;

    const bool   transmissive = LoadKd_w(matID) < 1.0f - EPSILON;
    const bool   flipIOR      = hinfo.backface && transmissive;
    const float2 iors         = flipIOR ? float2(matNi, 1.0f) : float2(1.0f, matNi);
    const uint   mediumMatID  = flipIOR ? matID : MEDIUM_INVALID;

    float3 hitLocalKd; float hitLocalPr, hitLocalPm;
    RefetchMaterial(matID, hinfo.uv, hitLocalKd, hitLocalPr, hitLocalPm, depth);

    const float3 absorptionTint = (mediumMatID != MEDIUM_INVALID)
        ? CalculateAbsorptionThroughput(LoadTf(mediumMatID), hp.hitT)
        : float3(1, 1, 1);

    const float3 emission = GetEmissionFast(instID, primID);

    //====================================================================
    //DEPTH 0 PRIMARY STORAGE + MOTION VECTOR REFLECTION PROBE
    //====================================================================
    //Matches the monolith's depth==0 block, same ordering, same writes
    //to g_sample_current / gScratchPing.
    if (depth == 0)
    {
        const bool isEmitter = any(emission > 0.0f);
        store_instID    (g_sample_current, pixelIdx, instID);
        store_primID    (g_sample_current, pixelIdx, primID, isEmitter);
        store_bary      (g_sample_current, pixelIdx, hp.bary);
        store_n1_s_world(g_sample_current, pixelIdx, hinfo.hitNormal, instID);
        store_uv        (g_sample_current, pixelIdx, hinfo.uv);
        if (isEmitter) {
            gScratchPing[uint3(pixel, 1)] = float4(emission, 0);
            gScratchPing[uint3(pixel, 2)] = float4(emission, 0);
        }

        //Specular motion vector reflection probe. Same as monolith.
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
                const float3 reflPos    = reflOrigin + reflDir * reflT;
                const float3 virtualPos = reflPos - 2.0f * dot(reflPos - hitPos, hinfo.hitNormal) * hinfo.hitNormal;
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
    //Matches the monolith: any emissive triangle with a valid lightID
    //produces an RIS candidate (depth==1 DI-triangle, depth>=2 GI-emitter).
    //The depth==0 case falls into the depth>=2 branch on the monolith
    //with load_ps returning init_ps sentinels; we preserve that
    //semantics bit-for-bit to avoid diverging from the current image.
    if (any(emission > 0.0f) && hinfo.lightID != 0xFFFFFFFFu)
    {
        const HotState h = load_hot(g_pathStateBuffer, pixelIdx);
        const float3 throughput = UnpackRGB9E5(h.throughputPk);
        const float3 prevNormal = UnpackNormal(h.prevNormalPk);
        const float  lightPdfArea = LT_Pdf_LightTree_Area(rayOrigin, prevNormal, hinfo.lightID, instID);
        const float  cosLight     = max(dot(hinfo.hitNormal, -rayDir), 0.0f);
        const float  dist2        = max(hp.hitT * hp.hitT, EPSILON);
        const float  lightPdfSA   = (cosLight > EPSILON) ? (lightPdfArea * dist2 / cosLight) : 0.0f;
        const float  misWeight    = h.prev_pdf / max(h.prev_pdf + lightPdfSA, EPSILON);

        const float3 F_contrib = throughput * emission * h.pdf_product;
        const float  p_hat     = GetPHat(F_contrib);
        const float  p_full    = h.pdf_product;
        const float  wi        = (p_full > 1e-20f) ? (misWeight * p_hat / p_full) : 0.0f;

        float wsum = h.wsum;
        uint  seed = h.seed;

        if (depth == 1)
        {
            const float3 diMarker = diMarkerFor(pixelIdx, time);
            AddInitialCandidate(wsum, g_Reservoirs_current, pixelIdx, wi,
                hitPos, hinfo.hitNormal,
                emission, diMarker,
                float2(0, 0),
                MATID_LIGHT_TRI, instID, 1.0f,
                F_contrib, seed);
        }
        else
        {
            //depth == 0 falls here with init_ps sentinels; depth >= 2 uses
            //the real depth-1 stash. Matches the monolith line-for-line.
            const PathVertexState ps = load_ps(g_pathStateBuffer, pixelIdx);
            const float3 tpost = UnpackRGB9E5(h.tpostPk);
            AddInitialCandidate(wsum, g_Reservoirs_current, pixelIdx, wi,
                ps.x2, ps.n2_s,
                emission * tpost, ps.v2,
                ps.uv,
                ps.matID, ps.objID, ps.eta,
                F_contrib, seed);
        }

        store_ps_wsum(g_pathStateBuffer, pixelIdx, wsum);
        store_seed(g_pathStateBuffer, pixelIdx, seed);
        flags |= PS_FLAG_TERMINATED | PS_FLAG_IS_EMITTER;
        store_flags(g_pathStateBuffer, pixelIdx, flags);
        return;
    }

    //====================================================================
    //DEPTH-1 / DEPTH-2 STASH (matches monolith)
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
    //STASH CLAS + SET FLAGS
    //====================================================================
    const bool performNEE = !(mediumMatID != MEDIUM_INVALID || LoadKd_w(matID) < EPSILON);

    store_clas(g_pathStateBuffer, pixelIdx,
               PackNormal(hinfo.hitNormal), matID, iors.y, hinfo.uv,
               PackRGB9E5(absorptionTint));

    flags |= PS_FLAG_HAS_VALID_HIT;
    if (performNEE)      flags |= PS_FLAG_PERFORM_NEE;
    if (hinfo.backface)  flags |= PS_FLAG_IS_BACKFACE;
    if (flipIOR)         flags |= PS_FLAG_FLIP_IOR;
    if (transmissive)    flags |= PS_FLAG_TRANSMISSIVE;
    store_flags(g_pathStateBuffer, pixelIdx, flags);
}
