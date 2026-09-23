#include "Includes_v8.hlsli"
#include "PtDefer_v8.hlsli"

// Material evaluation at the deferred vertex, on the lobe group its sample picked: the value of
// the cache-bound scatter that weights everything traced after it, the light and sun samples the
// light pass resolved (with their visibility), MIS, and the diffuse-reuse candidates. The light
// sample belongs to the pick, evaluated on the same group and divided by the probability of the
// pick, except on the broad group, which takes it on every pick that defers the vertex; each is
// MIS-weighted against its group's own density. No rays are traced here.
//
// Three phases, so each BSDF evaluation runs with little around it: the scatter weight first,
// then the two light techniques, each reading only its own part of the light record; their reuse
// candidates stay in the register generator (a reservoir-side accumulation of these produced wrong
// reservoirs in practice); then the tail, which reloads the path fields it needs.
[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    const uint2 imgSize = uint2(IMG_W, IMG_H);
    if (any(tid.xy >= imgSize)) return;
    const uint2 pixel    = tid.xy;
    const uint  pixelIdx = MapPixelID(imgSize, pixel);

    const uint info = DvLoadInfo(pixelIdx);
    if ((info & DV_VALID) == 0u) return;

    const DvVertex  v    = DvLoadVertex(pixelIdx);
    const DvPath    path = DvLoadPath(pixelIdx);
    const DvScatter sc   = DvLoadScatter(pixelIdx);
    const uint  depth    = PtPsDepth(path.ps);
    const uint  s        = PtPsSample(path.ps);
    const float invN     = rcp((float)PtSampleCount());
    const uint  pathSeed = PtPathSeed(pixel, s);
    g_regularizeRoughness = (v.flags & DVF_REGULARIZE) != 0u ? PT_REGULARIZE_ROUGHNESS : 0.0f;

    const HitContext ctx = DvContext(v);
    const float3 rayDir  = v.dirIn;
    const uint   group   = DvLobeGroup(v.flags);
    const float  neeScale = path.pickP > 0.0f ? rcp(path.pickP) : 0.0f;   // the light sample stands for this pick
    // The continuation of a broad pick at a reuse primary is broad as a whole: the reservoir carries it.
    const bool liteGen   = depth == 1u && (path.ps & PT_PS_LITE_VERTEX) != 0u && group == LOBE_GROUP_BROAD;
    float3 total = 0.0f;

    // --- the cache-bound scatter first: afterwards only its weight stays live ---
    float3 W = 0.0f;
    bool tailAlive = false;
    if ((info & DV_KIND_MASK) == DV_KIND_BSDF)
    {
        const BrdfData lobe = EvaluateLobe(v.sp, group, v.matID, v.n, v.n, sc.dirOut, -rayDir,
            ctx.hitLocalKd, ctx.hitLocalPr, ctx.hitLocalPm, ctx.iors.x, ctx.iors.y);
        const float cosTheta = abs(dot(v.n, sc.dirOut));
        W = lobe.val * v.absorb * cosTheta / sc.pdfTotal;
        tailAlive = !(any(isnan(W)) || any(isinf(W))) && any(W > 0.0f);
        // Russian roulette, replayed with the throughput the trace pass could not know.
        if (tailAlive && depth >= (uint)pt_rrStartDepth)
        {
            uint sRr = RcBounceSeed(pathSeed, depth, RC_STREAM_RR);
            const float survivalProb = max(min(1.0f, Luma(path.T * W)), 0.05f);
            if (RandomFloatSingle(sRr) >= survivalProb) tailAlive = false;
            else W /= survivalProb;
        }
    }
    const float3 liteBroad = liteGen ? W : 0.0f;

    // --- light sample and sun sample: the light pass resolved both and their visibility. The broad
    // group takes them on every pick that defers the vertex, divided by the probability of such a
    // pick (DvDeferP), so that a glossy pick does not leave the pixel without its diffuse light. A
    // picked group other than the broad one takes them on its own pick, divided by the probability
    // of that pick. Each is MIS-weighted against its own group's density, as the BSDF side of that
    // group is. ---
    const bool broadEvery = (v.flags & DVF_BROAD_NEE) != 0u;
    const bool liteNee    = (v.flags & DVF_LITE_NEE) != 0u;
    LiteGen liteG = LiteGenEmpty();
    if (liteNee && s != 0u) liteG = LiteGenLoad(pixelIdx);
    if ((v.flags & DVF_PERFORM_NEE) != 0u)
    {
        const bool  broadPick  = group == LOBE_GROUP_BROAD;
        const float broadScale = broadEvery ? rcp(DvDeferP(v)) : neeScale;
        float reward = 0.0f;
        [loop]
        for (uint tech = 0u; tech < 2u; ++tech)
        {
            float3 L, radiance, visT;
            float  lightPdf;
            uint   objID = 0u;                         // light point: technique 0 only
            float3 lightPos = 0.0f, lightN = 0.0f;
            float  dist = RAY_TMAX_PLANET;
            if (tech == 0u)
            {
                DvLoadLightMesh(pixelIdx, objID, lightPdf, lightPos, lightN, radiance, visT);
                const float3 toLight = lightPos - v.pos;
                dist = sqrt(max(dot(toLight, toLight), 1e-20f));
                L = toLight / dist;
            }
            else
                DvLoadLightSun(pixelIdx, L, radiance, lightPdf, visT);
            if (!(lightPdf > 0.0f) || !any(visT > 0.0f) || !any(radiance > 0.0f)) continue;
            const float  cosSurf    = dot(v.n, L);
            const float3 lightScale = radiance * cosSurf * visT / lightPdf;
            // The cut learns from the broad response where there is one (LTC_TrainShare).
            float3 trained = 0.0f;
            [branch] if (broadEvery || broadPick)
            {
                const BrdfData lobeNEE = EvaluateLobe(v.sp, LOBE_GROUP_BROAD, v.matID, v.n, v.n, L, -rayDir,
                    ctx.hitLocalKd, ctx.hitLocalPr, ctx.hitLocalPm, ctx.iors.x, ctx.iors.y);
                if (lobeNEE.pdf > 0.0f)
                {
                    trained = lobeNEE.val;
                    const float misWeight = lightPdf / (lightPdf + lobeNEE.pdf);
                    if (liteNee)
                    {
                        // The reservoir carries the broad part; none of it goes to the pixel here.
                        uint sLite = RcBounceSeed(pathSeed, depth, 0x4c495445u + tech);
                        LiteSample cand;
                        LiteLink   link;
                        float3     yWorld = 0.0f;
                        if (tech == 0u)
                        {
                            cand = LiteSampleSurface(objID, lightPos, lightN, radiance, LITE_KIND_LIGHT);
                            link = LiteLinkFrom(dist, cosSurf, dot(lightN, -L), false);
                            yWorld = lightPos;
                        }
                        else
                        {
                            cand = LiteSampleDirection(L, radiance);
                            link = LiteLinkFrom(RAY_TMAX_PLANET, cosSurf, 1.0f, true);
                        }
                        LiteGenCandidate(liteG, v.Kd, cand, yWorld, link, visT, misWeight, lightPdf,
                            invN * broadScale, sLite);
                    }
                    else total += path.T * lobeNEE.val * lightScale * (misWeight * broadScale);
                }
            }
            [branch] if (!broadPick)
            {
                const BrdfData lobeNEE = EvaluateLobe(v.sp, group, v.matID, v.n, v.n, L, -rayDir,
                    ctx.hitLocalKd, ctx.hitLocalPr, ctx.hitLocalPm, ctx.iors.x, ctx.iors.y);
                if (lobeNEE.pdf > 0.0f)
                {
                    if (!broadEvery) trained = LTC_TrainShare(v.Pr, v.matID, lobeNEE.val, false);
                    const float misWeight = lightPdf / (lightPdf + lobeNEE.pdf);
                    total += path.T * lobeNEE.val * lightScale * (misWeight * neeScale);
                }
            }
            if (tech == 0u) reward = dot(lightScale * trained, float3(0.2126f, 0.7152f, 0.0722f));
        }
        LT_TrainSample(DvLoadLightToken(pixelIdx), reward);
    }
    if (liteNee) LiteGenCommit(pixelIdx, liteG);

    // --- the tail: everything gathered after the deferred scatter, weighted by its value ---
    if (tailAlive)
    {
        // A reuse primary hands its broad-lobe share to the reservoir; the rest continues down the path.
        const float3 Wtail = W - liteBroad;
        total += path.T * Wtail * DvLoadPathRelL(pixelIdx);

        const uint endKind = (info >> DV_END_SHIFT) & DV_END_MASK;
        if (endKind == DV_END_EMITTER)
        {
            const DvEmitter e = DvLoadEmitter(pixelIdx);
            const float  emitterPdfArea = DvLoadEmitterPdfArea(pixelIdx);
            const float3 emission  = g_EmissiveTriangles[e.lightID].emission * GLOBAL_EMISSION_STRENGTH;
            const bool   noPartner = (info & DV_MIS_NONE) != 0u;
            const float  cosLight  = max(dot(e.n, -sc.dirOut), 0.0f);
            const float3 span      = e.pos - v.pos;
            const float  dist2     = max(dot(span, span), EPSILON);
            const float  lightPdfSA = (cosLight > EPSILON) ? (emitterPdfArea * dist2 / cosLight) : 0.0f;
            const float  misWeight  = noPartner ? 1.0f : sc.bsdfPdf / max(sc.bsdfPdf + lightPdfSA, EPSILON);
            if (liteGen)
            {
                total += path.T * (W - liteBroad) * emission * misWeight;
                const float  liteCosX = max(dot(v.n, sc.dirOut), 0.0f);
                const float3 liteNy   = dot(e.n, -sc.dirOut) < 0.0f ? -e.n : e.n;
                const LiteSample cand = LiteSampleSurface(e.inst, e.pos, liteNy, emission, LITE_KIND_LIGHT);
                const LiteLink   link = LiteLinkFrom(sqrt(dist2), liteCosX, dot(liteNy, -sc.dirOut), false);
                uint sLite = RcBounceSeed(pathSeed, depth, 0x4c495447u);
                LiteCandidate(pixelIdx, load_kd(g_sample_current, pixelIdx), cand, e.pos, link, (float3)1.0f,
                    misWeight, sc.pdfTotal, invN, sLite);
            }
            else total += path.T * W * emission * misWeight;
        }
        else if (endKind == DV_END_MISS && (info & DV_LITE_MISS) != 0u)
        {
            const DvMiss m = DvLoadMiss(pixelIdx);
            const LiteSample cand = LiteSampleDirection(m.dir, m.skySun);
            const LiteLink   link = LiteLinkFrom(RAY_TMAX_PLANET, max(dot(v.n, m.dir), 0.0f), 1.0f, true);
            uint sLite = RcBounceSeed(pathSeed, depth, 0x4c495447u);
            LiteCandidate(pixelIdx, load_kd(g_sample_current, pixelIdx), cand, m.dir, link, (float3)1.0f,
                m.candMis, sc.pdfTotal, invN, sLite);
        }
        if ((info & DV_LITE_X2) != 0u)
        {
            uint sLite = RcBounceSeed(pathSeed, 2u, 0x4c495448u);
            LiteCandidatePoint(pixelIdx, DvLoadLiteL(pixelIdx), invN, sLite);
        }
    }

    PtAddRadiance(pixel, total);
}
