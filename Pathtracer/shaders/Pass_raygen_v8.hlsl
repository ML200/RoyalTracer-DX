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
//PSS / HYBRID SHIFT: F is stored as the RR-clean PRIMARY-SAMPLE-SPACE
//contribution f/p_noRR (throughput is already the f/p estimator product; only
//the RR boost is peeled back out via rrProd and the NEE light pdf divides in).
//The resampling weight wi = mis * lum(F_pss_boosted) is ALGEBRAICALLY IDENTICAL
//to the legacy mis * p_hat / p_full, so candidate selection matches the old
//build draw-for-draw; only the stored representation changes (W becomes a
//dimensionless O(1) ratio).
//
//LOBE-INDEXED PSS (supp §1, LOBE_PSS_ON): extension dims additionally split
//per sampled lobe — throughput takes rho_l/(P(l)*p(w|l)), the pmf product
//peels out of the stored F via lobeProd (so F is lobe-clean like it is
//RR-clean), the 2-bit lobe ids of vertices 1..8 ride rcInfo bits 16-31
//(RC_F_LOBES) and the jacobian bundles below use the CONDITIONAL prev_pdfL.
//MIS vs NEE/sun and the footprint criteria keep the MARGINAL prev_pdf
//(supp §3; marginal MIS is what keeps the lobe-summed estimator unbiased). The reconnection vertex is pinned at the FIRST
//vertex pair passing RcCritPair (hybrid ON) or unconditionally at x2 (hybrid
//OFF = legacy reconnection shift); every pin records rcInfo + the base-side
//jacobian bundle (cachedJac/gBase) + the pathSeed whose per-bounce streams
//(RcBounceSeed) make the prefix bit-reproducible for random replay.
//
//RC pin bookkeeping carried across the bounce loop:
//  rcK        pin vertex index (0 = none yet); replay length = rcK-2
//  rcPinFlags RC_F_* of the pin itself (candidate sites may OR RC_F_NOPK)
//  rcJacNoPk  p(w_{k-1}) * gBase   (bundle without the continuation pdf)
//  rcJac      full bundle * gBase  (valid once rcPkPend consumed)
//  rcGBase    base-side geometric factor of the reconnection segment
//  rcPkPend   what the pin still owes at the NEXT BSDF sample:
//               PK_BUNDLE    fold p(w_k) into rcJac (surface continuation pin)
//               PK_TPOSTPDF  fold 1/pdf into tpost (invariant continuation: the
//                            SSS exit's white re-emergence bounce)
//               PK_NONE      nothing pending (bundle complete at pin)
//  sufOpen    tpost accumulation open (suffix strictly beyond the pin's own dim)
#define RC_PK_NONE     0u
#define RC_PK_BUNDLE   1u
#define RC_PK_TPOSTPDF 2u

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

    //Primary incoming (view) direction recovered from the camera origin + primary
    //hit. DoF lens jitter is sub-pixel and dropped here, matching the temporal
    //pass's NoV convention. Seeds the depth==1 back-edge carrier (rayDirPk0).
    const uint   primInst0 = load_instID(g_sample_current, pixelIdx);
    const float3 primPos0  = load_x1_with_instID(g_sample_current, pixelIdx, primInst0);
    const float3 toPrim0   = primPos0 - InitOrigin();
    const uint   rayDirPk0 = PackNormal(normalize(toPrim0));

    //hybrid pin criteria constants. Legacy (footprint OFF): minimum segment
    //length as a fraction of the primary camera distance. Enhanced §4
    //(footprint ON): the per-pixel dual-footprint threshold, Eq. 5 literal.
    const float rcDistMin = rs_reconnectDistMin * length(toPrim0);
    uint rcMaxK = HYBRID_SHIFT_ON ? max(rs_rcMaxK, 2u) : 2u;
    //supp §1: the rcInfo lobe mask covers vertices 1..8 (host clamps slot 19
    //too; this is the shader-side belt)
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
    uint   seed;          //per-pixel RIS accept stream (never replayed)
    uint   pathSeed;      //per-sample replay identity -> RcBounceSeed streams
    uint   throughputPk;
    uint   prevNormalPk;
    float  prev_pdf;      //MARGINAL pdf of the last extension (MIS + criteria, supp §3)
    float  prev_pdfL;     //bundle-side pdf of the last extension: conditional p(w|l)
                          //under LOBE_PSS, == prev_pdf otherwise
    float  rrProd;        //product of RR survival probabilities (PSS source pdf)
    float  lobeProd;      //product of sampled-lobe pmfs P(l) — peels out of the
                          //stored F exactly like rrProd (supp §1 lobe indexing)
    uint   rcLobes;       //2-bit sampled-lobe ids of vertices 1..8 (rcInfo 16-31)
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

        //per-sample carrier reset. `seed` is the RIS-accept stream; pathSeed
        //derives the per-bounce BSDF/NEE/SSS/RR streams that replay re-creates.
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

        //SSS path state (per sample). sssEntered blocks re-entry after one walk
        //AND blocks any later pin (replay cannot re-trace through a walk);
        //sssActive marks that the walk relocated ctx to a suffix boundary-exit B.
        bool sssActive  = false;
        bool sssEntered = false;

        //hybrid pin state (see header comment)
        uint  rcK        = 0u;
        uint  rcPinFlags = 0u;
        uint  rcPkPend   = RC_PK_NONE;
        bool  sufOpen    = false;
        float rcJacNoPk  = 0.0f;
        float rcJac      = 0.0f;
        float rcGBase    = 0.0f;
        //Enhanced §4 inverse footprint: G(x_k -> x_{k-1}) of a tentative pin,
        //resolved against p(w_k) at the RC_PK_BUNDLE fold (revokes on fail)
        float rcPendGinv = 0.0f;

    //====================================
    //BOUNCE LOOP
    //====================================
    //Process v_depth (ctx), then trace to v_{depth+1}. NEE then BSDF at the top;
    //the bottom trace is the BSDF MIS partner and fires every iter. depth==1
    //primary (sky+sun -> directAtX1, emissive DI -> reservoir); deeper vertices
    //feed candidates through the pin (PathVertexState) or become end-pins.
    [loop]
    for (int16_t depth = 1; depth < (int)pt_maxBounces; ++depth)
    {
        float3 rayDir = UnpackNormal(rayDirPk);

        //per-bounce RNG streams: replay re-derives sBsdf/sSss for bounces < k
        //from the stored pathSeed alone; sNee/sRr consumption can never shift
        //them (variable light-tree / cloud-shadow / RR draw counts).
        uint sNee  = RcBounceSeed(pathSeed, (uint)depth, RC_STREAM_NEE);
        uint sBsdf = RcBounceSeed(pathSeed, (uint)depth, RC_STREAM_BSDF);
        uint sSss  = RcBounceSeed(pathSeed, (uint)depth, RC_STREAM_SSS);

        //Strategy probabilities are fixed for this bounce (matID, -rayDir, normal, iors,
        //Kd, Pm don't change within the iteration). Compute once and reuse for both NEE
        //techniques and the BSDF sample instead of three identical evals — the two NEE
        //sites sit behind separate IsVisible branches so DXC can't CSE them itself.
        const SamplingP sp = CalculateStrategyProbabilities(ctx.matID, -rayDir, ctx.hitNormal, ctx.iors.x, ctx.iors.y, ctx.hitLocalKd, ctx.hitLocalPm);

        //------------- NEE: point light + sun (MIS partner of the bottom trace) -------------
        //FUSED into one 2-iteration technique loop (0 = light-tree point light, 1 = sun).
        //RNG draw order within the NEE stream matches the old back-to-back blocks.
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
                float  distLight = 0.0f;     //x -> light distance (end-pin gBase)
                float  cosLightN = 0.0f;     //|cos| at the light (end-pin gBase)
                float3 lightPos = float3(0, 0, 0);   //point-light only (depth==1 payload)
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
                    float2 rCone = float2(RandomFloatSingle(sNee), RandomFloatSingle(sNee));
                    float  cosCone = cos(SURFACE_CLOUD_SHADOW_CONE_DEG * DEG2RAD);
                    float3 Lj = SampleConeAroundDir(L, cosCone, rCone);
                    float  vis = CloudSunVisibility(ctx.hitPos + sceneOriginWorld, Lj,
                                                    RandomFloatSingle(sNee));
                    radiance *= pow(max(vis, 1e-6f), SURFACE_CLOUD_SHADOW_SOFTNESS);
                }
#endif

                const float3 throughput = UnpackRGB9E5(throughputPk);

                BrdfData  bdataNEE = EvaluateAndPdf_COMBINED(sp, ctx.matID, ctx.hitNormal, ctx.hitNormal, L, -rayDir, ctx.hitLocalKd, ctx.hitLocalPr, ctx.hitLocalPm, ctx.iors.x, ctx.iors.y);

                const float bsdfPdf = bdataNEE.pdf;
                if (!(bsdfPdf > 0.0f))
                    continue;

                //PSS candidate: Fp = f/p_noRR with the RR boost still inside the
                //throughput. wi = mis * lum(Fp) equals the legacy mis*p_hat/p_full
                //draw-for-draw; the stored F peels the boost back out via rrProd.
                const float3 localMeasurement = radiance * bdataNEE.val * cosSurf * visT;
                const float3 Fp        = throughput * localMeasurement / lightPdf;
                const float  misWeight = lightPdf / (lightPdf + bsdfPdf);
                const float  wi        = misWeight * GetPHat(Fp);
                //lobe-clean like RR-clean: the pmf boosts stay in wi, peel out of F
                const float3 F_store   = Fp * rrProd * lobeProd;
                //supp §1: candidates self-describe their lobe sequence (prefix
                //dims 1..depth-1 here — the NEE segment itself is all-lobes)
                const uint   lobeW     = LOBE_PSS_ON ? RcLobesWord(rcLobes) : 0u;

                if (depth == 1)
                {
                    if (tech == 0u)
                    {
                        //primary DI, x_k is the light itself: forced NEE end-pin
                        //(k==d==2). cachedJac = lightPdfSA*gBase is the approx.
                        //position-invariant area density; RC_F_PAREA copies it
                        //across the shift (area-measure jacobian 1).
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
                        //primary direct SUN — now a reservoir candidate (§6.1
                        //unification; directAtX1 removed). Direction payload
                        //with the globally position-INVARIANT sun cone pdf:
                        //RC_F_PAREA copy, gBase = 1. L2 carries the (frozen)
                        //cloud-shadowed radiance; visibility re-traces at reuse.
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
                    //NEE FROM the pin vertex itself (ctx == x_k, continuation dim
                    //not consumed yet): the outgoing dim at the pin is this NEE
                    //draw -> bundle drops p(w_k), the (frozen-vertex) light pdf
                    //divides into the payload L2 instead. An SSS-exit pin also
                    //carries its invariant entry compensation in tpost.
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
                    //NEE beyond the pin: full-bundle candidate, everything past
                    //x_k (including this NEE's f*cos and its frozen light pdf)
                    //rides the pdf-divided suffix payload L2.
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
                    //no pin yet (glossy prefix): forced NEE end-pin at the light /
                    //sun (k==d, replayed prefix). NEE-dim copies are UNCONDITIONAL
                    //(position-invariant density, RC_F_PAREA) — glossy receivers
                    //self-gate through the shifted BSDF magnitude. Only replay
                    //feasibility gates the pin: k <= rcMaxK and a walk-free prefix.
                    //Infeasible pins still shade canonically (rcInfo 0).
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

        //------------- SSS: chance to enter the medium instead of reflecting -------------
        //The surface still reflects via the regular BRDF (NEE above + the BSDF sample
        //below). With probability pEnter = sssWeight * Fresnel-transmittance the
        //CONTINUATION instead enters the object: a random walk relocates to the boundary
        //exit B (suffix). The first in-medium scatter S1 is the reconnection vertex
        //(reconnected like a transmission segment); a no-scatter pass-through (thin area)
        //transmits diffusely with W=1 and uses B itself as the reconnection vertex. The
        //stochastic reflect/enter split energy-weights the indirect bounce, so the chosen
        //branch needs no 1/p factor. The ENTER ROLL is the SSS stream's FIRST draw —
        //replay reproduces exactly that one draw to detect enter/reflect divergence.
        if (!sssEntered && LoadIsSSS(ctx.matID))
        {
            const float fT     = 1.0f - FresnelDielectric(-rayDir, ctx.hitNormal, ctx.iors.x, ctx.iors.y).x;
            const float pEnter = saturate(LoadSSSWeight(ctx.matID) * fT);
            if (RandomFloatSingle(sSss) < pEnter)
            {
                SSSWalkResult w = SubsurfaceWalk(ctx.hitPos, ctx.hitNormal, ctx.matID, sSss);
                if (!w.valid) break;   //escaped / absorbed / step-cap -> dead path

                const float  sigma_t = 1.0f / max(LoadSSSRadius(ctx.matID), SSS_MIN_RADIUS);
                const float3 albedo  = saturate(LoadSSSAlbedo(ctx.matID));
                const float  gPhase  = LoadPhaseG(ctx.matID);
                const float  cosA    = abs(dot(ctx.hitNormal, w.entryDir));   //entry coupling cosine (G1)
                //the ENTRY boundary is tinted by the SURFACE albedo (regular Kd), not the
                //inside SSS albedo; captured before ctx is relocated to the white exit B.
                const float3 surfKd  = (float3)ctx.hitLocalKd;
                //invariant entry pdf (cos-hemisphere about the frozen normal): its
                //division belongs in the PAYLOAD (tpost) so shifted rebuilds carry it.
                const float  entryPdfInv = 1.0f / max(SSS_INV_PI * cosA, 1e-6f);

                if (rcK == 0u && w.nScatters >= 1u && !sssActive)
                {
                    //no pin yet: the first in-medium scatter S1 is the VOLUME
                    //reconnection vertex. Its full pdf chain (cos-hemisphere entry,
                    //free flight, HG phase — raygen's pdfFac) is the position-
                    //dependent jacobian bundle; ReconnectPSS re-derives it at the
                    //receiver. k = vertex index of S1.
                    const float distA  = length(w.firstScatterPos - ctx.hitPos);
                    const float phaseF = EvaluatePhaseHG(gPhase, dot(w.entryDir, w.firstScatterDir));
                    const float pdfFac = SSS_INV_PI * cosA * exp(-sigma_t * distA) * sigma_t * phaseF;

                    store_ps_depth1(g_pathStateBuffer, pixelIdx,
                                    w.firstScatterPos, w.firstScatterDir,
                                    ctx.matID | MATID_SSS_VOLUME_BIT, ctx.instID, 1.0f,
                                    albedo, 1.0f, 0.0f);
                    store_ps_v2(g_pathStateBuffer, pixelIdx, w.firstScatterDir);
                    store_rg_tpost(g_pathStateBuffer, pixelIdx, w.wRest);   //first-scatter albedo lives in F2

                    rcK        = (uint)depth + 1u;               //S1's vertex index
                    rcPinFlags = 0u;                             //volume eval rebuilds the full bundle
                    rcGBase    = max(1.0f / max(distA * distA, EPSILON), EPSILON);
                    rcJac      = max(pdfFac, 1e-20f) * rcGBase;
                    rcJacNoPk  = rcJac;
                    rcPkPend   = RC_PK_NONE;
                    sufOpen    = true;
                    sssActive  = true;
                }
                else if (rcK == 0u && !sssActive)
                {
                    //no-scatter pass-through (thin area), no pin yet: the boundary
                    //exit B is the reconnection vertex (white diffuse re-emergence,
                    //analog walk -> bundle == 1, entry pdf is invariant). B's own
                    //re-emergence bounce pdf is invariant too -> PK_TPOSTPDF folds
                    //it into the payload at the next BSDF sample. The entry-pdf
                    //compensation seeds tpost (candidates at B multiply it in).
                    const float3 dirB  = w.exitPos - ctx.hitPos;
                    const float  distB = max(length(dirB), EPSILON);
                    store_ps_depth1(g_pathStateBuffer, pixelIdx,
                                    w.exitPos, w.exitNormal,
                                    ctx.matID | MATID_SSS_EXIT_BIT, ctx.instID, 1.0f,
                                    float3(1, 1, 1), 1.0f, 0.0f);
                    store_rg_tpost(g_pathStateBuffer, pixelIdx, (float3)entryPdfInv);

                    rcK        = (uint)depth + 1u;               //B's vertex index
                    rcPinFlags = RC_F_NOPK | RC_F_NOPPREV;
                    rcGBase    = max(abs(dot(normalize(dirB), w.exitNormal)) / (distB * distB), EPSILON);
                    rcJac      = rcGBase;                        //bundle == 1
                    rcJacNoPk  = rcJac;
                    rcPkPend   = RC_PK_TPOSTPDF;
                    sufOpen    = false;
                    sssActive  = false;                          //path continues normally from B
                }
                else if (rcK == 0u)
                {
                    //unreachable (sssActive implies a pin) — defensive: keep unreusable
                    rcK = 0u;
                }
                else if (rcPkPend == RC_PK_BUNDLE)
                {
                    //pin exists but its continuation pdf is still pending — the
                    //continuation at the pin vertex turned out to be this SSS entry:
                    //re-anchor the pin to the ENTRY SURFACE (deep entry pin). The
                    //reconnection segment INTO ctx keeps its rolling prev_pdf/segGBase;
                    //the invariant entry pdf divides the payload (tpost), the walk is
                    //analog suffix.
                    store_ps_depth1(g_pathStateBuffer, pixelIdx,
                                    ctx.hitPos, ctx.hitNormal,
                                    ctx.matID, ctx.instID, (float)ctx.iors.y,
                                    surfKd, 1.0f, 0.0f);
                    store_ps_v2(g_pathStateBuffer, pixelIdx, w.entryDir);
                    store_rg_tpost(g_pathStateBuffer, pixelIdx, w.wTotal * entryPdfInv);

                    //rcK unchanged: the entry surface IS the pin vertex (the pending
                    //state means ctx == x_k).
                    rcPinFlags = RC_F_NOPK;
                    rcJac      = rcJacNoPk;
                    rcPkPend   = RC_PK_NONE;
                    sufOpen    = true;
                    sssActive  = true;
                }
                else
                {
                    //deep: pin already complete — the entry + walk are pure suffix,
                    //fold the entry coupling f/p (= surfKd, the cos/pi parts cancel
                    //analytically) into tpost.
                    const float3 tp = load_rg_tpost(g_pathStateBuffer, pixelIdx) * w.wTotal * surfKd;
                    store_rg_tpost(g_pathStateBuffer, pixelIdx, tp);
                    sssActive  = true;
                }

                //analog subsurface throughput -> the walk (incl. entry coupling) is
                //f/p by construction, so the PSS throughput just takes wTotal*surfKd.
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
        //supp §1 lobe-indexed PSS (LOBE_PSS_ON): the extension dim splits per
        //sampled lobe — throughput takes rho_l/(P(l)*p(w|l)) with the pmf
        //product peeling out of the stored F via lobeProd (the RR pattern), the
        //suffix payload takes the lobe-clean rho_l/p(w|l). MIS and the
        //footprint criteria stay on the MARGINAL pdf (supp §3). Legacy keeps
        //the mixture estimator f_full/p_marginal (all selects collapse).
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

        //state update + termination. bad BSDF sample -> no valid next direction
        if (dot(s, s) < 1e-12f || bdata.pdf <= 1e-6f || pdfSel <= 1e-9f ||
            any(isnan(updateWeight)) || any(isinf(updateWeight)))
            break;

        //pin continuation bookkeeping: the pin owed its p(w_k) (or an invariant
        //pdf division into the payload) — this sample IS that continuation.
        //Mutually exclusive with the tpost update below (this bounce's f*cos is
        //re-evaluated at reuse; only its pdf side lands per the pin type).
        //Enhanced §4: the INVERSE footprint test resolves here too — a tentative
        //pin whose reverse density p(w_k)*G(x_k->x_{k-1}) exceeds the threshold
        //is REVOKED (the pin search continues at later pairs; candidates already
        //stored with the tentative pin keep it — their continuation was a
        //non-BSDF dim the inverse test does not govern).
        bool foldedPk = false;
        if (rcK != 0u && rcPkPend != RC_PK_NONE)
        {
            if (rcPkPend == RC_PK_BUNDLE)
            {
                if (HYBRID_SHIFT_ON && RC_FOOTPRINT_ON &&
                    !RcFpDensityPass(bdata.pdf, rcPendGinv, rcFpThresh))
                {
                    //revoke: forget the pin, keep searching
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
                    //p(w_k) fold: CONDITIONAL p(w_k|l_k) under lobe PSS — the
                    //inverse footprint test above stays marginal (supp §3)
                    rcJac    = rcJacNoPk * pdfSel;
                    rcPkPend = RC_PK_NONE;
                    sufOpen  = true;
                    foldedPk = true;
                }
            }
            else // RC_PK_TPOSTPDF: invariant continuation pdf -> payload
            {
                const float3 tp = load_rg_tpost(g_pathStateBuffer, pixelIdx) / pdfSel;
                store_rg_tpost(g_pathStateBuffer, pixelIdx, tp);
                rcPkPend = RC_PK_NONE;
                sufOpen  = true;
                foldedPk = true;
            }
        }

        prev_pdf     = bdata.pdf;   //marginal (MIS + criteria)
        prev_pdfL    = pdfSel;      //bundle side (conditional under lobe PSS)
        lobeProd    *= pSel;
        //record the sampled lobe id for this vertex (unconditional per depth —
        //a revoked tentative pin must not leave mask holes)
        if (lobeMode && (uint)depth <= 8u)
            rcLobes |= (strat & 3u) << (((uint)depth - 1u) << 1);
        rayDir       = s;
        rayDirPk     = PackNormal(s);  //carrier for next iter back-edge

        //pin-vertex reconnection direction (V2 = v_{k+1} -> v_k = -s), stored at
        //the pin's own continuation sample so end candidates read the real
        //direction. Skipped when the SSS branch already wrote V2.
        if (foldedPk && !sssActive)
            store_ps_v2(g_pathStateBuffer, pixelIdx, -rayDir);

        const float3 offsetN = (dot(s, ctx.hitNormal) >= 0.0f) ? ctx.hitNormal : -ctx.hitNormal;
        rayOrigin    = offset_ray(ctx.hitPos, offsetN);
        //park rayOrigin in scratch (RAY_O, free under raygen) so it doesn't cross
        //the reorder; reloaded post-trace.
        store_ray_origin(g_pathStateBuffer, pixelIdx, rayOrigin);

        float3 throughput        = UnpackRGB9E5(throughputPk) * updateWeight;
        //suffix accumulator entry: pdf-DIVIDED under PSS, RR-free AND lobe-pmf-
        //free (rho_l*cos/p(w|l) under lobe PSS — the payload must be lobe-clean
        //like the stored F; legacy: f*cos/p_marginal).
        const float3 tpostWeight = (valSel * ctx.absorptionTint * cosTheta) / pdfSel;

        //Russian roulette (off when pt_rrStartDepth high). The 1/p boost cancels
        //in the RR-clean stored F (via rrProd) but survives in wi; must NOT
        //touch the tpost update.
        if (depth >= (int)pt_rrStartDepth)
        {
            uint sRr = RcBounceSeed(pathSeed, (uint)depth, RC_STREAM_RR);
            const float survivalProb = max(min(1.0f, Luma(throughput)), 0.05f);
            if (RandomFloatSingle(sRr) >= survivalProb) break;
            const float rrBoost = 1.0f / survivalProb;
            throughput  *= rrBoost;
            rrProd      *= survivalProb;
        }

        //tpost: post-pin suffix throughput (pdf-divided). Runs for every bounce
        //STRICTLY beyond the pin's own continuation (sufOpen), regardless of RR;
        //tpostWeight is the RR-unboosted f*cos/pdf.
        if (sufOpen && !foldedPk)
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

            const float3 Fp      = throughputCur * envL;
            const float  wi      = GetPHat(Fp);
            const float3 F_store = Fp * rrProd * lobeProd;
            const uint   lobeW   = LOBE_PSS_ON ? RcLobesWord(rcLobes) : 0u;

            //§6.1/§6.2.3: EVERY BSDF miss is a reservoir candidate now (directAtX1
            //removed). With a pin, the env tail folds into the suffix as before.
            //Without one, the end-pin splits on the guard at x_{d-1}:
            //  guard PASS  -> dir-copy end (bundle = p(w_{d-1}), gBase = 1)
            //  guard FAIL  -> RC_F_ENV_REPLAY: the shift replays the prefix AND
            //                 re-derives the final BSDF dim (pure PSS, J = 1)
            //Replay infeasible (k cap / walk in prefix) -> canonical-only.
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
                if (!HYBRID_SHIFT_ON)      copyGuard = true;   //unconditional dir-copy (legacy)
                else if (RC_FOOTPRINT_ON)  copyGuard = RcLobeProxyPass(prev_pdf);
                else                       copyGuard = (float)ctx.hitLocalPr >= rs_reconnectRoughnessMin;

                uint  info = 0u;
                float cj   = 0.0f, gB = 0.0f;
                if (feasible && copyGuard)
                {
                    //dir-copy: bundle = CONDITIONAL p(w_{d-1}|l) under lobe PSS
                    //(ReconnectPSS re-derives the recorded lobe at the receiver)
                    info = RcPackInfo(kEnd, kEnd, RC_F_NOPK) | lobeW;
                    cj   = max(prev_pdfL, EPSILON);
                    gB   = 1.0f;
                }
                else if (feasible && HYBRID_SHIFT_ON)
                {
                    info = RcPackInfo(kEnd, kEnd,                      //full replay
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

        //geometric factor of THIS segment (rolls into segGBase for a potential
        //SSS-entry pin at the next iteration; also the pin gBase below)
        const float segG_n = max(abs(dot(rayDir, hinfo_n.hitNormal)) / max(hitT_n * hitT_n, EPSILON), EPSILON);

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

            const float3 Fp      = throughputCur * emission_n;
            const float  wi      = misWeight * GetPHat(Fp);
            const float3 F_store = Fp * rrProd * lobeProd;
            const uint   lobeW   = LOBE_PSS_ON ? RcLobesWord(rcLobes) : 0u;

            if (rcK != 0u && depth > 1)
            {
                //pin exists: the emitter tail is pure suffix
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
                //no pin (depth==1, or a glossy prefix): BSDF end-pin AT the light
                //vertex (k==d, bundle = p(w_{d-1})*gBase — CONDITIONAL under lobe
                //PSS) when the leaving vertex passes the guard; else canonical-only.
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

        //----- Non-emitter hit: establish the pin when the criteria first pass -----
        //Hybrid ON, footprint mode (Enhanced §4): pdf-proxy glossiness guard on the
        //leaving vertex + forward footprint density test; the inverse half resolves
        //at the next BSDF sample (tentative pin, see the fold above). The material
        //roughness of x_k itself is NOT tested — a sharp x_k shows up as a large
        //p(w_k) and fails the inverse test instead (lobe-aware, black-box safe).
        //Hybrid ON, legacy criteria: material roughness pair + segment distance.
        //Hybrid OFF: unconditionally at the first indirect vertex (legacy shift).
        //No pins after an SSS walk (replay cannot re-trace through it).
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
                //bundle side: conditional p(w_{k-1}|l) under lobe PSS (the
                //criteria above tested the MARGINAL prev_pdf, supp §3)
                rcJacNoPk  = prev_pdfL * segG_n;
                rcJac      = rcJacNoPk;      //completed by RC_PK_BUNDLE next sample
                rcPkPend   = RC_PK_BUNDLE;
                //inverse-footprint operand: G(x_k -> x_{k-1}) of this segment
                rcPendGinv = max(abs(dot(rayDir, UnpackNormal(prevNormalPk))) /
                                 max(hitT_n * hitT_n, EPSILON), EPSILON);
            }
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
    //directAtX1 is gone (§6.1 unification: x1 sun-NEE and BSDF-miss env are
    //reservoir candidates now). Slot 3 keeps its ABI and reads zero; the
    //X1_DIRECT_OFF diagnostic flag is inert.
    gScratchPing[uint3(pixel, 3)] = float4(0, 0, 0, 0);

    //RIS-over-N: wsum sums to ~N, so /N for unbiased W (selection is scale-invariant).
    wsum /= max(1.0f, (float)pt_initialSamples);
    FinalizeReservoir(pixelIdx, wsum);
}
