#include "Includes_v8.hlsli"

#ifndef MAX_BOUNCES
#define MAX_BOUNCES 32
#endif
#ifndef MAX_DS_BOUNCES
#define MAX_DS_BOUNCES 4
#endif

//====================================================================
//MATERIAL EVAL + SHADE STAGE (universal)
//====================================================================
//Intermediate step-2 baseline for the wavefront split. Fuses the five
//material-touching stages from the final plan, PointNEE, MatEval_Point,
//SunNEE, MatEval_Sun, Scatter, into a single shader. Does no TraceRay
//(Extend did that) and no hit classification (Classify did that). Runs
//once per loop iteration for every pixel with HAS_VALID_HIT set.
//
//The register budget here is still heavy (material fetch + two NEE
//BRDF evals + one scatter BRDF sample + BRDF eval + RR + light-tree
//descent + sun sampling). Step 3 of the migration splits this into
//five separate dispatches to shed register pressure; the final gain
//over the monolith lives there. This baseline exists to prove the
//hit-packet/PathState plumbing works before we split further.

[shader("raygeneration")]
void Pass_matevalshade_v8()
{
    const uint2 pixel    = DispatchRaysIndex().xy;
    const uint2 imgSize  = DispatchRaysDimensions().xy;
    const uint  pixelIdx = MapPixelID(imgSize, pixel);

    uint flags = load_flags(g_pathStateBuffer, pixelIdx);
    if (flags & PS_FLAG_TERMINATED) return;
    if (!(flags & PS_FLAG_HAS_VALID_HIT)) return;

    const uint depth = ps_get_depth(flags);

    //====================================================================
    //RELOAD SURFACE + MATERIAL STATE FROM PATHSTATE
    //====================================================================
    const HotState      h   = load_hot(g_pathStateBuffer, pixelIdx);
    const ClassifyState c   = load_clas(g_pathStateBuffer, pixelIdx);
    const HitPacket     hp  = load_hp(g_pathStateBuffer, pixelIdx);
    const float3 rayOrigin  = load_ray_origin(g_pathStateBuffer, pixelIdx);
    const float3 rayDir     = load_ray_dir   (g_pathStateBuffer, pixelIdx);

    const float3 hitPos    = rayOrigin + rayDir * hp.hitT;
    const float3 hitNormal = UnpackNormal(c.hitNormalPk);

    const uint  matID      = c.matID;
    const uint  instID     = hp.instID;
    const float matNi      = LoadNi(matID);
    const bool  flipIOR    = (flags & PS_FLAG_FLIP_IOR) != 0u;
    const float2 iors      = flipIOR ? float2(matNi, 1.0f) : float2(1.0f, matNi);
    const float3 absorptionTint = UnpackRGB9E5(c.absTintPk);

    float3 hitLocalKd; float hitLocalPr, hitLocalPm;
    RefetchMaterial(matID, c.uv, hitLocalKd, hitLocalPr, hitLocalPm, depth);

    uint  seed = h.seed;
    float wsum = h.wsum;

    const float3 throughput = UnpackRGB9E5(h.throughputPk);
    const float3 tpost      = UnpackRGB9E5(h.tpostPk);

    //Regenerate per-pixel per-frame DI discriminator (cheap hash).
    const float3 diMarker = diMarkerFor(pixelIdx, time);

    //====================================================================
    //NEE (point light + sun), gated by PERFORM_NEE
    //====================================================================
    if (flags & PS_FLAG_PERFORM_NEE)
    {
        //----- POINT LIGHT NEE -----
        {
            LT_LightSampleResult light = LT_SamplePointOnLight(hitPos, hitNormal, seed);

            const float3 toLight = light.position - hitPos;
            const float  distSq  = dot(toLight, toLight);
            const float  dist    = sqrt(distSq);
            const float3 L       = toLight / dist;

            const float  cosSurf   = dot(hitNormal, L);
            const float  cosLightS = dot(light.normal, -L);

            if (cosSurf > 1e-6f && cosLightS > 1e-6f)
            {
                SamplingP sp_nee = CalculateStrategyProbabilities(matID, -rayDir, hitNormal, iors.x, iors.y, hitLocalKd, hitLocalPm);
                BrdfData  bdataNEE = EvaluateAndPdf_COMBINED(sp_nee, matID, hitNormal, hitNormal, L, -rayDir, hitLocalKd, hitLocalPr, hitLocalPm, iors.x, iors.y);

                const float lightPdf = light.pdfSolidAngle;
                const float bsdfPdf  = bdataNEE.pdf;

                if (lightPdf > 1e-20f && bsdfPdf > 0.0f)
                {
                    const float3 shadowOrigin = offset_ray(
                        hitPos, dot(L, hitNormal) >= 0.0f ? hitNormal : -hitNormal);
                    if (IsVisibleOffset(shadowOrigin, L, dist * 0.999f))
                    {
                        const float  misWeight        = lightPdf / (lightPdf + bsdfPdf);
                        const float3 localMeasurement = light.emission * bdataNEE.val * cosSurf;
                        const float3 F_contrib        = throughput * localMeasurement * h.pdf_product;
                        const float  p_hat            = GetPHat(F_contrib);
                        const float  p_full           = h.pdf_product * lightPdf;
                        const float  wi               = (p_full > 1e-20f) ? (misWeight * p_hat / p_full) : 0.0f;

                        if (depth == 0)
                        {
                            AddInitialCandidate(wsum, g_Reservoirs_current, pixelIdx, wi,
                                light.position, light.normal,
                                light.emission, diMarker,
                                float2(0, 0),
                                MATID_LIGHT_TRI, light.objID, 1.0f,
                                F_contrib, seed);
                        }
                        else if (depth == 1)
                        {
                            AddInitialCandidate(wsum, g_Reservoirs_current, pixelIdx, wi,
                                hitPos, hitNormal,
                                light.emission, -L,
                                c.uv,
                                matID, instID, iors.y,
                                F_contrib, seed);
                        }
                        else
                        {
                            const PathVertexState ps = load_ps(g_pathStateBuffer, pixelIdx);
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

        //----- SUN NEE -----
        {
            float2 rSun = float2(RandomFloatSingle(seed), RandomFloatSingle(seed));
            SunSampleResult sun = SampleSun(rSun);
            const float NdotL = dot(hitNormal, sun.direction);

            if (NdotL > 1e-6f)
            {
                SamplingP sp_nee = CalculateStrategyProbabilities(matID, -rayDir, hitNormal, iors.x, iors.y, hitLocalKd, hitLocalPm);
                BrdfData  bdataNEE = EvaluateAndPdf_COMBINED(sp_nee, matID, hitNormal, hitNormal, sun.direction, -rayDir, hitLocalKd, hitLocalPr, hitLocalPm, iors.x, iors.y);

                const float lightPdf = sun.pdf;
                const float bsdfPdf  = bdataNEE.pdf;

                if (lightPdf > 1e-20f && bsdfPdf > 0.0f)
                {
                    const float  misWeight        = lightPdf / (lightPdf + bsdfPdf);
                    const float3 localMeasurement = sun.radiance * bdataNEE.val * NdotL;
                    const float3 F_contrib        = throughput * localMeasurement * h.pdf_product;
                    const float  p_hat            = GetPHat(F_contrib);
                    const float  p_full           = h.pdf_product * lightPdf;

                    const float3 shadowOrigin = offset_ray(
                        hitPos, dot(sun.direction, hitNormal) >= 0.0f ? hitNormal : -hitNormal);
                    if (IsVisibleOffset(shadowOrigin, sun.direction, 10000.0f))
                    {
                        const float wi = (p_full > 1e-20f) ? (misWeight * p_hat / p_full) : 0.0f;
                        if (depth == 0)
                        {
                            AddInitialCandidate(wsum, g_Reservoirs_current, pixelIdx, wi,
                                sun.direction, float3(0, 1, 0),
                                sun.radiance, diMarker,
                                float2(0, 0),
                                MATID_ENV_MISS, MATID_ENV_MISS, 1.0f,
                                F_contrib, seed);
                        }
                        else if (depth == 1)
                        {
                            AddInitialCandidate(wsum, g_Reservoirs_current, pixelIdx, wi,
                                hitPos, hitNormal,
                                sun.radiance, -sun.direction,
                                c.uv,
                                matID, instID, iors.y,
                                F_contrib, seed);
                        }
                        else
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
    }

    //====================================================================
    //SCATTER
    //====================================================================
    SamplingP sp = CalculateStrategyProbabilities(matID, -rayDir, hitNormal, iors.x, iors.y, hitLocalKd, hitLocalPm);
    float3    s  = SampleBRDF(sp, matID, -rayDir, hitNormal, hitNormal, hitLocalKd, hitLocalPr, hitLocalPm, seed, iors.x, iors.y);
    BrdfData  bdata = EvaluateAndPdf_COMBINED(sp, matID, hitNormal, hitNormal, s, -rayDir, hitLocalKd, hitLocalPr, hitLocalPm, iors.x, iors.y);

    const float  cosTheta     = abs(dot(hitNormal, s));
    const float3 updateWeight = (bdata.pdf > 1e-6f)
        ? (bdata.val * absorptionTint * cosTheta) / bdata.pdf
        : float3(0, 0, 0);

    if (dot(s, s) < 1e-12f || bdata.pdf <= 1e-6f || any(isnan(updateWeight)) || any(isinf(updateWeight)))
    {
        //Kill path, commit what we have.
        store_ps_wsum(g_pathStateBuffer, pixelIdx, wsum);
        store_seed(g_pathStateBuffer, pixelIdx, seed);
        flags |= PS_FLAG_TERMINATED;
        store_flags(g_pathStateBuffer, pixelIdx, flags);
        return;
    }

    //Bounce classification + dsBounces cap.
    const bool isTransmission = dot(s, hitNormal) < 0.0f;
    uint dsBounces = ps_get_dsBounces(flags);
    if (!isTransmission)
    {
        if (dsBounces >= MAX_DS_BOUNCES)
        {
            store_ps_wsum(g_pathStateBuffer, pixelIdx, wsum);
            store_seed(g_pathStateBuffer, pixelIdx, seed);
            flags |= PS_FLAG_TERMINATED;
            store_flags(g_pathStateBuffer, pixelIdx, flags);
            return;
        }
        ++dsBounces;
    }

    const float  new_prev_pdf    = bdata.pdf;
    float        new_pdf_product = min(h.pdf_product * bdata.pdf, 1e30f);
    const float3 newRayDir       = s;
    const float3 offsetN         = dot(s, hitNormal) >= 0.0f ? hitNormal : -hitNormal;
    const float3 newRayOrigin    = offset_ray(hitPos, offsetN);

    float3 newThroughput = UnpackRGB9E5(h.throughputPk) * updateWeight;
    float3 tpostWeight   = bdata.val * absorptionTint * cosTheta;

    if (depth > 2)
    {
        //RR. Floor survival at 0.1, boost 1/p_s on throughput and include
        //p_s in the path pdf. Must NOT touch tpostWeight (reused across
        //neighbor pixels, boosting poisons the reuse).
        const float survivalProb = max(min(1.0f, Luma(newThroughput)), 0.1f);
        if (RandomFloatSingle(seed) >= survivalProb)
        {
            store_ps_wsum(g_pathStateBuffer, pixelIdx, wsum);
            store_seed(g_pathStateBuffer, pixelIdx, seed);
            flags |= PS_FLAG_TERMINATED;
            store_flags(g_pathStateBuffer, pixelIdx, flags);
            return;
        }
        const float rrBoost = 1.0f / survivalProb;
        newThroughput  *= rrBoost;
        new_pdf_product = min(new_pdf_product * survivalProb, 1e30f);
    }

    float3 newTpost = tpost;
    if (depth >= 2) newTpost = tpost * tpostWeight;

    //Commit next-bounce state.
    store_ray(g_pathStateBuffer, pixelIdx, newRayOrigin, newRayDir);

    const uint newThroughputPk = PackRGB9E5(newThroughput);
    const uint newTpostPk      = PackRGB9E5(newTpost);
    const uint newPrevNormalPk = PackNormal(hitNormal);

    store_hot1(g_pathStateBuffer, pixelIdx, newThroughputPk, newPrevNormalPk, new_prev_pdf, new_pdf_product);
    store_tpost_pk(g_pathStateBuffer, pixelIdx, newTpostPk);
    store_ps_wsum(g_pathStateBuffer, pixelIdx, wsum);
    store_seed(g_pathStateBuffer, pixelIdx, seed);

    //Increment depth + dsBounces; clear per-bounce hit/NEE bits (Classify
    //resets them again at next iter, but we must not carry them forward
    //in case Extend flags TERMINATED before Classify runs).
    flags = ps_set_depth(flags, depth + 1u);
    flags = ps_set_dsBounces(flags, dsBounces);
    flags &= ~(PS_FLAG_HAS_VALID_HIT | PS_FLAG_PERFORM_NEE
             | PS_FLAG_IS_BACKFACE   | PS_FLAG_FLIP_IOR
             | PS_FLAG_TRANSMISSIVE);

    //Depth cap: if we've reached MAX_BOUNCES, terminate now.
    if ((depth + 1u) >= MAX_BOUNCES) flags |= PS_FLAG_TERMINATED;

    store_flags(g_pathStateBuffer, pixelIdx, flags);
}
