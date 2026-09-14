#include "Includes_v8.hlsli"
#include "Raygen_Common_v8.hlsli"

#define RC_PK_NONE     0u
#define RC_PK_BUNDLE   1u
#define RC_PK_TPOSTPDF 2u

[shader("raygeneration")]
// Build primary reservoirs from direct-light and continuation candidates.
void Pass_raygen_v8()
{

    const uint  packedPx = g_raygenQueue.Load(16u + DispatchRaysIndex().x * 4u);
    const uint2 pixel    = uint2(packedPx & 0xFFFFu, packedPx >> 16);
    const uint2 imgSize  = uint2(IMG_W, IMG_H);
    const uint  pixelIdx = MapPixelID(imgSize, pixel);

    float wsum = 0.0f;

    SetSkyObserver(InitOrigin() + sceneOriginWorld);

    const uint   primInst0 = load_instID(g_sample_current, pixelIdx);
    const float3 primPos0  = load_x1_with_instID(g_sample_current, pixelIdx, primInst0);
    const float3 toPrim0   = primPos0 - InitOrigin();
    const uint   rayDirPk0 = PackNormal(normalize(toPrim0));

    const float rcDistMin = rs_reconnectDistMin * length(toPrim0);
    uint rcMaxK = HYBRID_SHIFT_ON ? max(rs_rcMaxK, 2u) : 2u;

    if (LOBE_PSS_ON) rcMaxK = min(rcMaxK, 8u);
    float rcFpThresh = 0.0f;
    if (HYBRID_SHIFT_ON && RC_FOOTPRINT_ON)
    {
        const float3 n1prim  = load_n1_s_with_instID(g_sample_current, pixelIdx, primInst0);
        const float  camD2   = dot(toPrim0, toPrim0);
        const float  cosPrim = abs(dot(n1prim, normalize(-toPrim0)));
        rcFpThresh = RcFpThreshold(camD2, cosPrim);
    }

    HitContext ctx;
    uint   seed;
    uint   pathSeed;
    uint   throughputPk;
    uint   prevNormalPk;
    float  prev_pdf;
    float  prev_pdfL;

    float  rrProd;
    float  lobeProd;

    uint   rcLobes;
    uint   rayDirPk;
    float3 rayOrigin;

    [loop]
    // Generate independent candidates for the pixel reservoir.
    for (uint s = 0u; s < pt_initialSamples; ++s)
    {

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

        seed = initRandomData(pixel, uint2(8, 4), time, s + 1u);
        pathSeed = Hash32(seed ^ 0x9E3779B9u);
        throughputPk = PackRGB9E5(float3(1, 1, 1));
        prevNormalPk = PackNormal(float3(0, 1, 0));
        prev_pdf     = 1.0f;
        prev_pdfL    = 1.0f;
        rrProd       = 1.0f;
        lobeProd     = 1.0f;
        rcLobes      = 0u;
        store_rg_tpost(g_pathStateBuffer, pixelIdx, float3(1, 1, 1));
        rayDirPk     = rayDirPk0;

        bool sssActive  = false;
        bool sssEntered = false;

        uint  rcK        = 0u;
        uint  rcPinFlags = 0u;
        uint  rcPkPend   = RC_PK_NONE;
        bool  sufOpen    = false;
        float rcJacNoPk  = 0.0f;
        float rcJac      = 0.0f;
        float rcGBase    = 0.0f;

        float rcPendGinv = 0.0f;

        int16_t diffuseDepth = 0;

    [loop]
    // Continue reconnectable paths until their depth or transport budget ends.
    for (int16_t depth = 1; depth < (int)pt_maxBounces; ++depth)
    {
        float3 rayDir = UnpackNormal(rayDirPk);

        uint sNee  = RcBounceSeed(pathSeed, (uint)depth, RC_STREAM_NEE);
        uint sBsdf = RcBounceSeed(pathSeed, (uint)depth, RC_STREAM_BSDF);
        uint sSss  = RcBounceSeed(pathSeed, (uint)depth, RC_STREAM_SSS);

        const SamplingP sp = CalculateStrategyProbabilities(ctx.matID, -rayDir, ctx.hitNormal, ctx.iors.x, ctx.iors.y, ctx.hitLocalKd, ctx.hitLocalPr, ctx.hitLocalPm);

        const bool performNEE = !(ctx.mediumMatID != MEDIUM_INVALID || LoadKd_w(ctx.matID) < EPSILON);
        // Direct lighting contributes one mesh-light and one sun technique.
        if (performNEE)
        {
            [loop]
            for (uint tech = 0u; tech < 2u; ++tech)
            {
                float3 L;
                float3 visTarget;
                float3 visTargetN;
                float3 radiance;
                float  lightPdf;
                float  cosSurf;
                float  distLight = 0.0f;
                float  cosLightN = 0.0f;
                float3 lightPos = float3(0, 0, 0);
                float3 lightN   = float3(0, 1, 0);
                uint   lightObjID = 0u;

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
                    lightPos   = light.position;
                    lightN     = light.normal;
                    lightObjID = light.objID;
                    distLight  = dist;
                    cosLightN  = cosLightS;
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

                const float3 visT = VisibilityTransmittance(ctx.hitPos, ctx.hitNormal, visTarget, visTargetN);
                if (!any(visT > 0.0f))
                    continue;

                const float3 throughput = UnpackRGB9E5(throughputPk);

                BrdfData  bdataNEE = EvaluateAndPdf_COMBINED(sp, ctx.matID, ctx.hitNormal, ctx.hitNormal, L, -rayDir, ctx.hitLocalKd, ctx.hitLocalPr, ctx.hitLocalPm, ctx.iors.x, ctx.iors.y);

                const float bsdfPdf = bdataNEE.pdf;
                if (!(bsdfPdf > 0.0f))
                    continue;

                const float3 localMeasurement = radiance * bdataNEE.val * cosSurf * visT;
                const float3 Fp        = throughput * localMeasurement / lightPdf;
                const float  misWeight = lightPdf / (lightPdf + bsdfPdf);
                const float  wi        = misWeight * GetPHat(Fp);

                const float3 F_store   = Fp * rrProd * lobeProd;

                const uint   lobeW     = LOBE_PSS_ON ? RcLobesWord(rcLobes) : 0u;

                if (depth == 1)
                {
                    if (tech == 0u)
                    {

                        const float gB = max(cosLightN / max(distLight * distLight, EPSILON), EPSILON);
                        AddInitialCandidate(wsum, g_Reservoirs_current, pixelIdx, wi,
                            lightPos, lightN,
                            radiance, diMarkerFor(pixelIdx, time),
                            float3(0,0,0), 0.0f, 0.0f,
                            MATID_LIGHT_TRI, lightObjID, 1.0f,
                            F_store,
                            RcPackInfo(2u, 2u, RC_F_NOPK | RC_F_NOPPREV | RC_F_PAREA) | lobeW,
                            pathSeed, lightPdf * gB, gB,
                            seed);
                    }
                    else
                    {
                        AddInitialCandidate(wsum, g_Reservoirs_current, pixelIdx, wi,
                            L, -L,
                            radiance, diMarkerFor(pixelIdx, time),
                            float3(0,0,0), 0.0f, 0.0f,
                            MATID_ENV_MISS, 0xFFFFFFFFu, 1.0f,
                            F_store,
                            RcPackInfo(2u, 2u, RC_F_NOPK | RC_F_NOPPREV | RC_F_PAREA) | lobeW,
                            pathSeed, max(lightPdf, EPSILON), 1.0f,
                            seed);
                    }
                }
                else if (rcK != 0u && rcPkPend != RC_PK_NONE)
                {

                    const float3 L2pin = (rcPkPend == RC_PK_TPOSTPDF)
                        ? radiance * load_rg_tpost(g_pathStateBuffer, pixelIdx) / lightPdf
                        : radiance / lightPdf;
                    AddInitialCandidate(wsum, g_Reservoirs_current, pixelIdx, wi,
                        ctx.hitPos, ctx.hitNormal,
                        L2pin, -L,
                        (float3)ctx.hitLocalKd, (float)ctx.hitLocalPr, (float)ctx.hitLocalPm,
                        ctx.matID | (sssEntered ? MATID_SSS_EXIT_BIT : 0u), ctx.instID, ctx.iors.y,
                        F_store,
                        RcPackInfo(rcK, (uint)depth + 1u, rcPinFlags | RC_F_NOPK) | lobeW,
                        pathSeed, rcJacNoPk, rcGBase,
                        seed);
                }
                else if (rcK != 0u)
                {

                    const PathVertexState ps = load_ps(g_pathStateBuffer, pixelIdx);
                    const float3 tpost    = load_rg_tpost(g_pathStateBuffer, pixelIdx);
                    const float3 tpostNEE = tpost * bdataNEE.val * cosSurf / lightPdf;
                    AddInitialCandidate(wsum, g_Reservoirs_current, pixelIdx, wi,
                        ps.x2, ps.n2_s,
                        radiance * tpostNEE, ps.v2,
                        ps.Kd, ps.Pr, ps.Pm,
                        ps.matID, ps.objID, ps.eta,
                        F_store,
                        RcPackInfo(rcK, (uint)depth + 1u, rcPinFlags) | lobeW,
                        pathSeed, rcJac, rcGBase,
                        seed);
                }
                else
                {

                    const uint kEnd   = (uint)depth + 1u;
                    const bool endPin = HYBRID_SHIFT_ON && !sssEntered && kEnd <= rcMaxK;
                    const bool sunEnd = (tech != 0u);
                    const float gB = sunEnd ? 1.0f
                        : max(cosLightN / max(distLight * distLight, EPSILON), EPSILON);
                    AddInitialCandidate(wsum, g_Reservoirs_current, pixelIdx, wi,
                        endPin ? (sunEnd ? L : lightPos) : ctx.hitPos,
                        endPin ? (sunEnd ? -L : lightN)  : ctx.hitNormal,
                        radiance, endPin ? diMarkerFor(pixelIdx, time) : -L,
                        float3(0,0,0), 0.0f, 0.0f,
                        endPin ? (sunEnd ? MATID_ENV_MISS : MATID_LIGHT_TRI) : ctx.matID,
                        endPin ? (sunEnd ? 0xFFFFFFFFu : lightObjID) : ctx.instID, 1.0f,
                        F_store,
                        endPin ? (RcPackInfo(kEnd, kEnd, RC_F_NOPK | RC_F_NOPPREV | RC_F_PAREA) | lobeW) : 0u,
                        pathSeed, endPin ? max(lightPdf * gB, EPSILON) : 0.0f, endPin ? gB : 0.0f,
                        seed);
                }
            }
        }

        // Subsurface walks become reconnectable path vertices when selected.
        if (!sssEntered && LoadIsSSS(ctx.matID))
        {
            const float fT     = 1.0f - FresnelDielectric(-rayDir, ctx.hitNormal, ctx.iors.x, ctx.iors.y).x;
            const float pEnter = saturate(LoadSSSWeight(ctx.matID) * fT);
            if (RandomFloatSingle(sSss) < pEnter)
            {
                SSSWalkResult w = SubsurfaceWalk(ctx.hitPos, ctx.hitNormal, ctx.matID, sSss);
                if (!w.valid) break;

                const float  sigma_t = 1.0f / max(LoadSSSRadius(ctx.matID), SSS_MIN_RADIUS);
                const float3 albedo  = saturate(LoadSSSAlbedo(ctx.matID));
                const float  gPhase  = LoadPhaseG(ctx.matID);
                const float  cosA    = abs(dot(ctx.hitNormal, w.entryDir));

                const float3 surfKd  = (float3)ctx.hitLocalKd;

                const float  entryPdfInv = 1.0f / max(SSS_INV_PI * cosA, 1e-6f);

                if (rcK == 0u && w.nScatters >= 1u && !sssActive)
                {

                    const float distA  = length(w.firstScatterPos - ctx.hitPos);
                    const float phaseF = EvaluatePhaseHG(gPhase, dot(w.entryDir, w.firstScatterDir));
                    const float pdfFac = SSS_INV_PI * cosA * exp(-sigma_t * distA) * sigma_t * phaseF;

                    store_ps_depth1(g_pathStateBuffer, pixelIdx,
                                    w.firstScatterPos, w.firstScatterDir,
                                    ctx.matID | MATID_SSS_VOLUME_BIT, ctx.instID, 1.0f,
                                    albedo, 1.0f, 0.0f);
                    store_ps_v2(g_pathStateBuffer, pixelIdx, w.firstScatterDir);
                    store_rg_tpost(g_pathStateBuffer, pixelIdx, w.wRest);

                    rcK        = (uint)depth + 1u;
                    rcPinFlags = 0u;
                    rcGBase    = max(1.0f / max(distA * distA, EPSILON), EPSILON);
                    rcJac      = max(pdfFac, 1e-20f) * rcGBase;
                    rcJacNoPk  = rcJac;
                    rcPkPend   = RC_PK_NONE;
                    sufOpen    = true;
                    sssActive  = true;
                }
                else if (rcK == 0u && !sssActive)
                {

                    const float3 dirB  = w.exitPos - ctx.hitPos;
                    const float  distB = max(length(dirB), EPSILON);
                    store_ps_depth1(g_pathStateBuffer, pixelIdx,
                                    w.exitPos, w.exitNormal,
                                    ctx.matID | MATID_SSS_EXIT_BIT, ctx.instID, 1.0f,
                                    float3(1, 1, 1), 1.0f, 0.0f);
                    store_rg_tpost(g_pathStateBuffer, pixelIdx, (float3)entryPdfInv);

                    rcK        = (uint)depth + 1u;
                    rcPinFlags = RC_F_NOPK | RC_F_NOPPREV;
                    rcGBase    = max(abs(dot(normalize(dirB), w.exitNormal)) / (distB * distB), EPSILON);
                    rcJac      = rcGBase;
                    rcJacNoPk  = rcJac;
                    rcPkPend   = RC_PK_TPOSTPDF;
                    sufOpen    = false;
                    sssActive  = false;
                }
                else if (rcK == 0u)
                {

                    rcK = 0u;
                }
                else if (rcPkPend == RC_PK_BUNDLE)
                {

                    store_ps_depth1(g_pathStateBuffer, pixelIdx,
                                    ctx.hitPos, ctx.hitNormal,
                                    ctx.matID, ctx.instID, (float)ctx.iors.y,
                                    surfKd, 1.0f, 0.0f);
                    store_ps_v2(g_pathStateBuffer, pixelIdx, w.entryDir);
                    store_rg_tpost(g_pathStateBuffer, pixelIdx, w.wTotal * entryPdfInv);

                    rcPinFlags = RC_F_NOPK;
                    rcJac      = rcJacNoPk;
                    rcPkPend   = RC_PK_NONE;
                    sufOpen    = true;
                    sssActive  = true;
                }
                else
                {

                    const float3 tp = load_rg_tpost(g_pathStateBuffer, pixelIdx) * w.wTotal * surfKd;
                    store_rg_tpost(g_pathStateBuffer, pixelIdx, tp);
                    sssActive  = true;
                }

                throughputPk = PackRGB9E5(UnpackRGB9E5(throughputPk) * w.wTotal * surfKd);
                sssEntered   = true;

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

        if (!MaterialIsFreeBounce(ctx.matID))
        {
            if (diffuseDepth >= (int)pt_maxDiffuseBounces) break;
            ++diffuseDepth;
        }

        uint strat;
        const float3 s = SampleBRDF(sp, ctx.matID, -rayDir, ctx.hitNormal, ctx.hitNormal, ctx.hitLocalKd, ctx.hitLocalPr, ctx.hitLocalPm, sBsdf, ctx.iors.x, ctx.iors.y, false, strat);
        float3 lobeVal; float lobePdf;
        const BrdfData bdata = EvaluateAndPdf_COMBINED_L(sp, strat, ctx.matID, ctx.hitNormal, ctx.hitNormal, s, -rayDir, ctx.hitLocalKd, ctx.hitLocalPr, ctx.hitLocalPm, ctx.iors.x, ctx.iors.y, false, lobeVal, lobePdf);

        const bool   lobeMode = LOBE_PSS_ON;
        const float  pSel     = lobeMode ? StrategyP(sp, strat) : 1.0f;
        const float  pdfSel   = lobeMode ? lobePdf : bdata.pdf;
        const float3 valSel   = lobeMode ? lobeVal : bdata.val;

        const float  cosTheta     = abs(dot(ctx.hitNormal, s));
        const float3 updateWeight = (bdata.pdf > 1e-6f && pdfSel > 1e-9f && pSel > EPSILON)
            ? (valSel * ctx.absorptionTint * cosTheta) / (pdfSel * pSel)
            : float3(0, 0, 0);

        if (dot(s, s) < 1e-12f || bdata.pdf <= 1e-6f || pdfSel <= 1e-9f ||
            any(isnan(updateWeight)) || any(isinf(updateWeight)))
            break;

        bool foldedPk = false;
        if (rcK != 0u && rcPkPend != RC_PK_NONE)
        {
            if (rcPkPend == RC_PK_BUNDLE)
            {
                if (HYBRID_SHIFT_ON && RC_FOOTPRINT_ON &&
                    !RcFpDensityPass(bdata.pdf, rcPendGinv, rcFpThresh))
                {

                    rcK        = 0u;
                    rcPinFlags = 0u;
                    rcJac      = 0.0f;
                    rcJacNoPk  = 0.0f;
                    rcGBase    = 0.0f;
                    rcPkPend   = RC_PK_NONE;
                    sufOpen    = false;
                }
                else
                {

                    rcJac    = rcJacNoPk * pdfSel;
                    rcPkPend = RC_PK_NONE;
                    sufOpen  = true;
                    foldedPk = true;
                }
            }
            else
            {
                const float3 tp = load_rg_tpost(g_pathStateBuffer, pixelIdx) / pdfSel;
                store_rg_tpost(g_pathStateBuffer, pixelIdx, tp);
                rcPkPend = RC_PK_NONE;
                sufOpen  = true;
                foldedPk = true;
            }
        }

        prev_pdf     = bdata.pdf;
        prev_pdfL    = pdfSel;
        lobeProd    *= pSel;

        if (lobeMode && (uint)depth <= 8u)
            rcLobes |= (strat & 3u) << (((uint)depth - 1u) << 1);
        rayDir       = s;
        rayDirPk     = PackNormal(s);

        if (foldedPk && !sssActive)
            store_ps_v2(g_pathStateBuffer, pixelIdx, -rayDir);

        const float3 offsetN = (dot(s, ctx.hitNormal) >= 0.0f) ? ctx.hitNormal : -ctx.hitNormal;
        rayOrigin    = offset_ray(ctx.hitPos, offsetN);

        store_ray_origin(g_pathStateBuffer, pixelIdx, rayOrigin);

        float3 throughput        = UnpackRGB9E5(throughputPk) * updateWeight;

        const float3 tpostWeight = (valSel * ctx.absorptionTint * cosTheta) / pdfSel;

        if (depth >= (int)pt_rrStartDepth)
        {
            uint sRr = RcBounceSeed(pathSeed, (uint)depth, RC_STREAM_RR);
            const float survivalProb = max(min(1.0f, Luma(throughput)), 0.05f);
            if (RandomFloatSingle(sRr) >= survivalProb) break;
            const float rrBoost = 1.0f / survivalProb;
            throughput  *= rrBoost;
            rrProd      *= survivalProb;
        }

        if (sufOpen && !foldedPk)
        {
            const float3 tpost = load_rg_tpost(g_pathStateBuffer, pixelIdx) * tpostWeight;
            store_rg_tpost(g_pathStateBuffer, pixelIdx, tpost);
        }

        throughputPk = PackRGB9E5(throughput);
        prevNormalPk = PackNormal(ctx.hitNormal);

        if (!IsRayValid(rayOrigin, rayDir, 10000.0f))
            break;

        RayDesc rayB;
        rayB.Origin    = rayOrigin;
        rayB.Direction = rayDir;
        rayB.TMin      = 0.00001f;
        rayB.TMax      = RAY_TMAX_PLANET;

        dx::HitObject hitObjB = TraceRay_Custom(SceneBVH, rayB, RAY_FLAG_FORCE_OMM_2_STATE, 0xFF);

        const float3 rayOriginR = load_ray_origin(g_pathStateBuffer, pixelIdx);

        if (!hitObjB.IsHit())
        {
            const float3 throughputCur = UnpackRGB9E5(throughputPk);

            SetSkyObserver(rayOriginR + sceneOriginWorld);

            const float  sunSAPdf   = GetSunPdf(rayDir);
            const float3 sunRad     = (sunSAPdf > 0.0f) ? EvaluateSun(rayDir) : float3(0, 0, 0);
            const float  sunMisBsdf = (sunSAPdf > 0.0f)
                ? prev_pdf / max(prev_pdf + sunSAPdf, EPSILON) : 0.0f;
            const float3 sky        = EvaluateSky(rayDir);
            const float3 envL       = sky + sunRad * sunMisBsdf;

            const float3 Fp      = throughputCur * envL;
            const float  wi      = GetPHat(Fp);
            const float3 F_store = Fp * rrProd * lobeProd;
            const uint   lobeW   = LOBE_PSS_ON ? RcLobesWord(rcLobes) : 0u;

            if (rcK != 0u && depth > 1)
            {
                const PathVertexState ps = load_ps(g_pathStateBuffer, pixelIdx);
                const float3 tpost = load_rg_tpost(g_pathStateBuffer, pixelIdx);
                AddInitialCandidate(wsum, g_Reservoirs_current, pixelIdx, wi,
                    ps.x2, ps.n2_s,
                    envL * tpost, ps.v2,
                    ps.Kd, ps.Pr, ps.Pm,
                    ps.matID, ps.objID, ps.eta,
                    F_store,
                    RcPackInfo(rcK, (uint)depth, rcPinFlags) | lobeW,
                    pathSeed, rcJac, rcGBase,
                    seed);
            }
            else
            {
                const uint kEnd     = (uint)depth + 1u;
                const bool feasible = (depth == 1) ||
                                      (HYBRID_SHIFT_ON && !sssEntered && kEnd <= rcMaxK);
                bool copyGuard;
                if (!HYBRID_SHIFT_ON)      copyGuard = true;
                else if (RC_FOOTPRINT_ON)  copyGuard = RcLobeProxyPass(prev_pdf);
                else                       copyGuard = (float)ctx.hitLocalPr >= rs_reconnectRoughnessMin;

                uint  info = 0u;
                float cj   = 0.0f, gB = 0.0f;
                if (feasible && copyGuard)
                {

                    info = RcPackInfo(kEnd, kEnd, RC_F_NOPK) | lobeW;
                    cj   = max(prev_pdfL, EPSILON);
                    gB   = 1.0f;
                }
                else if (feasible && HYBRID_SHIFT_ON)
                {
                    info = RcPackInfo(kEnd, kEnd,
                                      RC_F_NOPK | RC_F_NOPPREV | RC_F_ENV_REPLAY) | lobeW;
                    cj   = 1.0f;
                    gB   = 1.0f;
                }
                AddInitialCandidate(wsum, g_Reservoirs_current, pixelIdx, wi,
                    rayDir, -rayDir,
                    envL, -rayDir,
                    float3(0,0,0), 0.0f, 0.0f,
                    MATID_ENV_MISS, 0xFFFFFFFFu, 1.0f,
                    F_store,
                    info, pathSeed, cj, gB,
                    seed);
            }
            break;
        }

        const float  hitT_n   = hitObjB.GetRayTCurrent();
        const uint   instID_n = hitObjB.GetInstanceID();
        const uint    primID_n = FlatPrimID(instID_n, hitObjB.GetGeometryIndex(), hitObjB.GetPrimitiveIndex());
        const uint    matID_n  = GetMatIDFast(instID_n, primID_n);
        BuiltInTriangleIntersectionAttributes attrB;
        hitObjB.GetAttributes(attrB);
        HitInfo hinfo_n = EvalSurfaceState(instID_n, primID_n, attrB.barycentrics, rayOriginR, (uint)depth);
        const float3 hitPos_n = hinfo_n.hitPos;

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

        const float segG_n = max(abs(dot(rayDir, hinfo_n.hitNormal)) / max(hitT_n * hitT_n, EPSILON), EPSILON);

        const float3 emission_n = (hinfo_n.lightID != 0xFFFFFFFFu)
            ? g_EmissiveTriangles[hinfo_n.lightID].emission * GLOBAL_EMISSION_STRENGTH
            : float3(0, 0, 0);

        if (any(emission_n > 0.0f))
        {
            const float3 throughputCur = UnpackRGB9E5(throughputPk);
            const float3 prevNormalCur = ctx.hitNormal;
            const float  lightPdfArea  = LT_Pdf_LightTree_Area(ctx.hitPos, prevNormalCur, hinfo_n.lightID, instID_n);
            const float  cosLight      = max(dot(hinfo_n.hitNormal, -rayDir), 0.0f);
            const float  dist2         = max(hitT_n * hitT_n, EPSILON);
            const float  lightPdfSA    = (cosLight > EPSILON) ? (lightPdfArea * dist2 / cosLight) : 0.0f;
            const float  misWeight     = prev_pdf / max(prev_pdf + lightPdfSA, EPSILON);

            const float3 Fp      = throughputCur * emission_n;
            const float  wi      = misWeight * GetPHat(Fp);
            const float3 F_store = Fp * rrProd * lobeProd;
            const uint   lobeW   = LOBE_PSS_ON ? RcLobesWord(rcLobes) : 0u;

            if (rcK != 0u && depth > 1)
            {

                const PathVertexState ps = load_ps(g_pathStateBuffer, pixelIdx);
                const float3 tpost = load_rg_tpost(g_pathStateBuffer, pixelIdx);
                AddInitialCandidate(wsum, g_Reservoirs_current, pixelIdx, wi,
                    ps.x2, ps.n2_s,
                    emission_n * tpost, ps.v2,
                    ps.Kd, ps.Pr, ps.Pm,
                    ps.matID, ps.objID, ps.eta,
                    F_store,
                    RcPackInfo(rcK, (uint)depth + 1u, rcPinFlags) | lobeW,
                    pathSeed, rcJac, rcGBase,
                    seed);
            }
            else
            {

                const uint kEnd = (uint)depth + 1u;
                const bool endGuard = RC_FOOTPRINT_ON
                    ? (RcLobeProxyPass(prev_pdf) && RcFpDensityPass(prev_pdf, segG_n, rcFpThresh))
                    : RcCritPair((float)ctx.hitLocalPr, 1.0f, true, hitT_n, rcDistMin);
                const bool endPin = (depth == 1) ||
                                    (HYBRID_SHIFT_ON && !sssEntered && kEnd <= rcMaxK && endGuard);
                AddInitialCandidate(wsum, g_Reservoirs_current, pixelIdx, wi,
                    hitPos_n, hinfo_n.hitNormal,
                    emission_n, diMarkerFor(pixelIdx, time),
                    float3(0,0,0), 0.0f, 0.0f,
                    MATID_LIGHT_TRI, instID_n, 1.0f,
                    F_store,
                    endPin ? (RcPackInfo(kEnd, kEnd, RC_F_NOPK) | lobeW) : 0u,
                    pathSeed, endPin ? (prev_pdfL * segG_n) : 0.0f, endPin ? segG_n : 0.0f,
                    seed);
            }
            break;
        }

        if (rcK == 0u && !sssEntered && (uint)depth + 1u <= rcMaxK)
        {
            bool pinHere;
            if (!HYBRID_SHIFT_ON)
                pinHere = (depth == 1);
            else if (RC_FOOTPRINT_ON)
                pinHere = RcLobeProxyPass(prev_pdf) &&
                          RcFpDensityPass(prev_pdf, segG_n, rcFpThresh);
            else
                pinHere = RcCritPair((float)ctx.hitLocalPr, hitLocalPr_n, false, hitT_n, rcDistMin);
            if (pinHere)
            {
                store_ps_depth1(g_pathStateBuffer, pixelIdx,
                                hitPos_n, hinfo_n.hitNormal,
                                matID_n, instID_n, iors_n.y,
                                hitLocalKd_n, hitLocalPr_n, hitLocalPm_n);
                rcK        = (uint)depth + 1u;
                rcPinFlags = 0u;
                rcGBase    = segG_n;

                rcJacNoPk  = prev_pdfL * segG_n;
                rcJac      = rcJacNoPk;
                rcPkPend   = RC_PK_BUNDLE;

                rcPendGinv = max(abs(dot(rayDir, UnpackNormal(prevNormalPk))) /
                                 max(hitT_n * hitT_n, EPSILON), EPSILON);
            }
        }

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

    }

    wsum /= max(1.0f, (float)pt_initialSamples);
    FinalizeReservoir(pixelIdx, wsum);
}
