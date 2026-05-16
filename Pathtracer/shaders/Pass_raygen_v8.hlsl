#include "Includes_v8.hlsli"
#include "Nrc_v8.hlsli"

//safety net, real termination is RR at depth>2 plus NRC cache short circuit
#ifndef MAX_BOUNCES
#define MAX_BOUNCES 12
#endif

//====================================
//PACKED NRC MUTABLE STATE
//====================================
//bits 0..5   hitIdx          (0..63)
//bits 6..9   trainVIdx       (0..15, NRC_MAX_VERTICES_PER_PATH=8)
//bits 10..11 pathClass       (0..2: RENDER / TRAIN_BIASED / TRAIN_UNBIASED)
//bit  13     prevSpecular
//bit  14     cacheTerminated
//tailKind used to live at bit 12 but is now written directly to the
//path-meta buffer at each termination point; bit 12 is unused.
#define NRC_PK_HITIDX_MASK     0x3Fu
#define NRC_PK_TRAINVIDX_SHIFT 6u
#define NRC_PK_TRAINVIDX_MASK  0xFu
#define NRC_PK_PATHCLASS_SHIFT 10u
#define NRC_PK_PATHCLASS_MASK  0x3u
#define NRC_PK_PREVSPEC_BIT    (1u << 13)
#define NRC_PK_CACHETERM_BIT   (1u << 14)

inline uint pk_hit   (uint s)        { return s & NRC_PK_HITIDX_MASK; }
inline uint pk_vidx  (uint s)        { return (s >> NRC_PK_TRAINVIDX_SHIFT) & NRC_PK_TRAINVIDX_MASK; }
inline uint pk_pclass(uint s)        { return (s >> NRC_PK_PATHCLASS_SHIFT) & NRC_PK_PATHCLASS_MASK; }
inline bool pk_prev  (uint s)        { return (s & NRC_PK_PREVSPEC_BIT)   != 0u; }
inline bool pk_cterm (uint s)        { return (s & NRC_PK_CACHETERM_BIT)  != 0u; }
inline uint pk_set_hit   (uint s, uint v) { return (s & ~NRC_PK_HITIDX_MASK) | (v & NRC_PK_HITIDX_MASK); }
inline uint pk_set_vidx  (uint s, uint v) { return (s & ~(NRC_PK_TRAINVIDX_MASK << NRC_PK_TRAINVIDX_SHIFT))
                                                   | ((v & NRC_PK_TRAINVIDX_MASK) << NRC_PK_TRAINVIDX_SHIFT); }
inline uint pk_set_pclass(uint s, uint v) { return (s & ~(NRC_PK_PATHCLASS_MASK << NRC_PK_PATHCLASS_SHIFT))
                                                   | ((v & NRC_PK_PATHCLASS_MASK) << NRC_PK_PATHCLASS_SHIFT); }
inline uint pk_set_prev  (uint s, bool b) { return b ? (s | NRC_PK_PREVSPEC_BIT)  : (s & ~NRC_PK_PREVSPEC_BIT); }
inline uint pk_set_cterm (uint s)         { return s | NRC_PK_CACHETERM_BIT; }


//====================================
//PER-VERTEX SURFACE STATE (loop-carried)
//====================================
//Bundles everything the bounce loop needs to process vertex v_depth: hit
//geometry, material, IOR / medium context. Populated once by TraceCameraRay
//for v_1, then overwritten at the bottom of each bounce iter with v_{depth+1}.
struct HitContext {
    float3 hitPos;          //fp32, scene-coord position
    float  hitT;            //fp32, for NRC area-spread accumulation
    float3 hitNormal;       //fp32, used for offset_ray + dot products w/ float dirs
    float2 uv;              //fp32, used in shadow-origin offsets
    uint   matID;
    uint   instID;
    bool   backface;
    //----- bounded-range params live in half -----
    half3  hitLocalKd;      //albedo, [0,1] per channel
    half   hitLocalPr;      //roughness, [0,1]
    half   hitLocalPm;      //metalness, [0,1]
    half2  iors;            //IORs, ~1..2.5
    uint   mediumMatID;
    half3  absorptionTint;  //transmittance, [0,1] per channel
};


//====================================
//CAMERA RAY (primary-hit extraction)
//====================================
//Trace the camera ray and write everything that is primary-specific:
//  - sky/emitter scratch slots that feed the post-process composite
//  - DLSS sample buffer (instID, primID, normal, uv, hitT, ...)
//  - specular MV reflection probe -> scratch slot 4
//  - NRC vertex-1 anchor (hitIdx=1, nrcA0)
//
//Returns true and populates `ctx` with v_1 hit data if the path should
//continue. The bounce loop then processes v_1 (NEE + BSDF + training) at
//iter depth==1 and traces from there.
//
//Returns false on degenerate ray / miss / primary emitter — those cases
//are terminal at the camera vertex and need no further processing.
inline bool TraceCameraRay(
    uint2  pixel,
    uint   pixelIdx,
    inout  uint   seed,
    inout  uint   nrcStateA,
    inout  float3 rayOrigin,
    inout  float3 rayDir,
    out    HitContext ctx)
{
    ctx = (HitContext)0;

    if (!IsRayValid(rayOrigin, rayDir, 10000.0f))
        return false;

    //primary uses 4-state OMM (full alpha test); bounces force 2-state
    RayDesc ray;
    ray.Origin    = rayOrigin;
    ray.Direction = rayDir;
    ray.TMin      = 0.00001f;
    ray.TMax      = 100000.0f;
    dx::HitObject hitObj = TraceRay_Custom(SceneBVH, ray, RAY_FLAG_NONE, 0xFF);

    //----- MISS: sky + sun disk written to primary scratch slots -----
    if (!hitObj.IsHit())
    {
        const float3 sun = EvaluateSun(rayDir);
        float3 skyL1     = EvalMissState(rayDir, sun);
        if (length(sun) > 0.0f) skyL1 = sun;
        gScratchPing[uint3(pixel, 1)] = float4(skyL1, 0);
        gScratchPing[uint3(pixel, 2)] = float4(skyL1, 0);
        store_sky(g_sample_current, pixelIdx);
        return false;
    }

    //----- HIT: extract surface state -----
    const float  hitT   = hitObj.GetRayTCurrent();
    const float3 hitPos = rayOrigin + rayDir * hitT;
    const uint   instID = hitObj.GetInstanceIndex();
    const uint   primID = FlatPrimID(instID, hitObj.GetGeometryIndex(), hitObj.GetPrimitiveIndex());
    const uint   matID  = GetMatIDFast(instID, primID);

    BuiltInTriangleIntersectionAttributes attr;
    hitObj.GetAttributes(attr);
    HitInfo hinfo = EvalSurfaceState(instID, primID, attr.barycentrics, rayOrigin, 0u);

    const float  matNi        = LoadNi(matID);
    const bool   transmissive = LoadKd_w(matID) < 1.0f - EPSILON;
    const bool   flipIOR      = hinfo.backface && transmissive;
    const float2 iors         = flipIOR ? float2(matNi, 1.0f) : float2(1.0f, matNi);
    const uint   mediumMatID  = flipIOR ? matID : MEDIUM_INVALID;

    float3 hitLocalKd; float hitLocalPr, hitLocalPm;
    RefetchMaterial(matID, hinfo.uv, hitLocalKd, hitLocalPr, hitLocalPm, 0u);

    const float3 absorptionTint = (mediumMatID != MEDIUM_INVALID)
        ? CalculateAbsorptionThroughput(LoadTf(mediumMatID), hitT)
        : float3(1, 1, 1);

    const float3 emission  = GetEmissionFast(instID, primID);
    const bool   isEmitter = any(emission > 0.0f);

    //----- DLSS sample-buffer writes (drive reprojection) -----
    store_instID    (g_sample_current, pixelIdx, instID);
    store_primID    (g_sample_current, pixelIdx, primID, isEmitter, hinfo.backface);
    store_bary      (g_sample_current, pixelIdx, attr.barycentrics);
    store_n1_s_world(g_sample_current, pixelIdx, hinfo.hitNormal, instID);
    store_uv        (g_sample_current, pixelIdx, hinfo.uv);
    store_hitT      (g_sample_current, pixelIdx, hitT);
    if (isEmitter)
    {
        gScratchPing[uint3(pixel, 1)] = float4(emission, 0);
        gScratchPing[uint3(pixel, 2)] = float4(emission, 0);
    }

    //----- Specular MV probe: virtualPos -> scratch slot 4 -----
    //Heavy RayQuery state is scoped to this block only and dies at the brace.
    {
        const float3 reflDir    = reflect(rayDir, hinfo.hitNormal);
        const float3 reflOrigin = offset_ray(hitPos, hinfo.hitNormal);

        bool  committed = false;
        float reflT     = 0.0f;
        if (IsRayValid(reflOrigin, reflDir, 10000.0f))
        {
            RayDesc reflRay;
            reflRay.Origin    = reflOrigin;
            reflRay.Direction = reflDir;
            reflRay.TMin      = 0.00001f;
            reflRay.TMax      = 100000.0f;

            RayQuery<RAY_FLAG_NONE> q;
            q.TraceRayInline(SceneBVH, RAY_FLAG_NONE, 0xFF, reflRay);
            uint16_t alphaIter = 0;
            while (q.Proceed() && alphaIter < uint16_t(128))
            {
                ++alphaIter;
                if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
                {
                    const uint cInstID = q.CandidateInstanceIndex();
                    const uint cPrimID = FlatPrimID(cInstID, q.CandidateGeometryIndex(), q.CandidatePrimitiveIndex());
                    const uint cMatID  = GetMatIDFast(cInstID, cPrimID);
                    if (LoadAlphaThreshold(cMatID) < 1.0f)
                        q.CommitNonOpaqueTriangleHit();
                }
            }
            if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
            {
                committed = true;
                reflT     = q.CommittedRayT();
            }
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

    //----- Emitter direct hit terminates the path here -----
    if (isEmitter && hinfo.lightID != 0xFFFFFFFFu)
        return false;

    //----- NRC vertex-1 anchor (area-spread baseline read each bounce) -----
    nrcStateA = pk_set_hit(nrcStateA, 1u);
    const float cosPrimary = max(abs(dot(rayDir, hinfo.hitNormal)), 1e-6f);
    store_rg_nrcA0(g_pathStateBuffer, pixelIdx, NrcComputeA0(hitT, cosPrimary));

    //----- Hand v_1 surface state to the bounce loop's first iter -----
    //bounded-range fields narrow to half at the assignment so the struct
    //phi the loop carries is smaller.
    ctx.hitPos         = hitPos;
    ctx.hitT           = hitT;
    ctx.hitNormal      = hinfo.hitNormal;
    ctx.uv             = hinfo.uv;
    ctx.matID          = matID;
    ctx.instID         = instID;
    ctx.backface       = hinfo.backface;
    ctx.hitLocalKd     = (half3)hitLocalKd;
    ctx.hitLocalPr     = (half) hitLocalPr;
    ctx.hitLocalPm     = (half) hitLocalPm;
    ctx.iors           = (half2)iors;
    ctx.mediumMatID    = mediumMatID;
    ctx.absorptionTint = (half3)absorptionTint;

    return true;
}


//DI at d=2 and GI at d>=3 share one reservoir, cold state stashed in g_pathStateBuffer
[shader("raygeneration")]
void Pass_raygen_v8()
{
    const uint2 pixel    = DispatchRaysIndex().xy;
    const uint2 imgSize  = DispatchRaysDimensions().xy;
    const uint  pixelIdx = MapPixelID(imgSize, pixel);

    //adaptive tile side from renderer feedback
    const uint  nrcTileSide   = NrcTrainingTileSide();
    const uint2 nrcTileOffset = uint2(asuint(time) % nrcTileSide,
                                      (asuint(time) / nrcTileSide) % nrcTileSide);
    //ClassifyPixel returns initial (training/unbiased) but we only need the
    //class enum -- the bool outs are tossed and DCE'd.
    bool clsTraining, clsUnbiased;
    uint pathClassInit = NrcClassifyPixel(pixel, asuint(time), nrcTileSide, nrcTileOffset,
                                          clsTraining, clsUnbiased);
    if (!(clsTraining && NrcIsTrainOn())) pathClassInit = NRC_CLASS_RENDER;

    NrcClearPendingGI(pixelIdx);

    //packed mutable NRC state: hitIdx, trainVIdx, pathClass, prevSpec,
    //cacheTerm. pathClass lives in bits 10..11 so the bounce loop does not
    //carry it as a separate live register.
    uint nrcStateA = 0u;
    uint nrcPathId = NRC_INVALID_PATH;

    //Allocation may fail under the per-frame quota; demote to render in that
    //case. On success seed the RR-default tail so termination points need only
    //write when their kind differs (and stale data from a previous frame at
    //this slot is wiped).
    if (pathClassInit != NRC_CLASS_RENDER)
    {
        nrcPathId = NrcAllocateTrainingPath();
        if (nrcPathId == NRC_INVALID_PATH)
            pathClassInit = NRC_CLASS_RENDER;
        else
            NrcInitPathTail(nrcPathId);
    }
    nrcStateA = pk_set_pclass(nrcStateA, pathClassInit);
    //pathClassInit dies here; downstream reads go via pk_pclass(nrcStateA)

    //nrcA + nrcEmitMask spill to coherent scratch (HOT2 plane). nrcA is
    //RMW'd each bounce inside the cache check; nrcEmitMask is OR'd at each
    //training-vertex emit via InterlockedOr and read once at finalize.
    store_rg_nrcA(g_pathStateBuffer, pixelIdx, 0.0f);
    store_rg_nrcEmitMask(g_pathStateBuffer, pixelIdx, 0u);

    //slot 1 primary emitter/sky, slot 3 unused
    gScratchPing[uint3(pixel, 1)] = float4(0, 0, 0, 0);
    gScratchPing[uint3(pixel, 3)] = float4(0, 0, 0, 0);
    //slot 7 sharp refl control rgb=Fresnel w=NRC slot, slot 8 raw env radiance
    gScratchPing[uint3(pixel, 7)] = float4(0, 0, 0, asfloat(NRC_INVALID_SLOT));
    gScratchPing[uint3(pixel, 8)] = float4(0, 0, 0, 0);
    //slot 9 NRC debug L_s, postprocess reads even when NRC debug pass skips this pixel
    gScratchPing[uint3(pixel, 9)] = float4(0, 0, 0, 0);

    storeReservoir(g_Reservoirs_current, pixelIdx, (Reservoir)0);

    //safe defaults, otherwise pass-through continues leave stale state in the slot
    init_ps(g_pathStateBuffer, pixelIdx);

    float wsum = 0.0f;

    uint   seed = initRandomData(pixel, uint2(8, 4), time, 1u);
    float3 rayOrigin;
    float3 rayDir;
    InitCameraRayDoF(pixel, imgSize, seed, rayOrigin, rayDir);

    //sky/sun sampler reads camera altitude through this static, must be set
    //before any EvaluateSky/EvaluateSun/SampleSun call below
    SetSkyObserver(InitOrigin());
    uint   throughputPk = PackRGB9E5(float3(1, 1, 1));
    uint   prevNormalPk = PackNormal(float3(0, 1, 0));
    float  prev_pdf     = 1.0f;
    float  pdf_product  = 1.0f;

    //tpost lives in scratch (PS plane, aliases HOT1) so the multiplicative
    //throughput chain does not carry its 12B as live state across the bounce
    //loop's TraceRay reorder boundary. Initialise to (1,1,1).
    store_rg_tpost(g_pathStateBuffer, pixelIdx, float3(1, 1, 1));

    //=====================================================================
    //CAMERA RAY: trace v_1, extract surface, populate ctx. Returns false if
    //the camera ray is terminal (degenerate / miss / direct emitter hit).
    //=====================================================================
    HitContext ctx;
    if (!TraceCameraRay(pixel, pixelIdx, seed, nrcStateA, rayOrigin, rayDir, ctx))
    {
        //Camera ray was terminal -- finalize and exit.
        const bool nrcCacheTerminatedFin = pk_cterm(nrcStateA);
        if (nrcPathId != NRC_INVALID_PATH)
        {
            const uint nrcTrainVIdxFinal = pk_vidx(nrcStateA);
            const uint nrcEmitMaskFinal  = load_rg_nrcEmitMask(g_pathStateBuffer, pixelIdx);
            NrcStorePathHead(nrcPathId, nrcTrainVIdxFinal, nrcEmitMaskFinal);
        }
        store_wsum(g_Reservoirs_current, pixelIdx, wsum);
        if (!nrcCacheTerminatedFin)
        {
            const float F_mag = GetPHat(load_F(g_Reservoirs_current, pixelIdx));
            float W = (F_mag > 1e-6f && wsum > 0.0f) ? (wsum / F_mag) : 0.0f;
            if (isnan(W) || isinf(W) || W < 0.0f) W = 0.0f;
            store_W(g_Reservoirs_current, pixelIdx, W);
            store_M(g_Reservoirs_current, pixelIdx, 1u);
            if (W == 0.0f)
                InvalidateReservoir_ShadingNormal(g_Reservoirs_current, pixelIdx);
        }
        return;
    }

    //Octahedral-pack rayDir as the loop's carrier. The trace at iter bottom
    //needs float3 (so live-state-at-reorder is unchanged), but the loop
    //back-edge phi shrinks from 3 dwords to 1.
    uint rayDirPk = PackNormal(rayDir);

    //=====================================================================
    //BOUNCE LOOP: each iter processes vertex v_depth (already in ctx),
    //then traces v_depth -> v_{depth+1} and extracts the new ctx for the
    //next iter. Depth conventions:
    //  depth==1 : process v_1 (primary). DI candidate's x2 = light position.
    //  depth==2 : process v_2 (first bounce). DI's x2 = ctx.hitPos. Stores
    //             ps_depth1(v_2) at iter bottom after extracting v_2.
    //  depth>=3 : later GI. x2 from PathVertexState. RR + tpost update active.
    //NEE on v_depth at iter top, BSDF sample after NEE, training emit after
    //both, then trace at iter bottom. The trace is the BSDF MIS partner for
    //NEE -- it always fires (no validity gate) so MIS stays balanced even at
    //the last iter. Termination cuts: bad BSDF / RR fail / IsRayValid.
    //=====================================================================
    [loop]
    for (int16_t depth = 1; depth < MAX_BOUNCES; ++depth)
    {
        //Unpack the carried direction once per iter; reassigned to the BSDF
        //sample at state update and re-packed for the back-edge.
        float3 rayDir = UnpackNormal(rayDirPk);
        //------------- NRC vertex bookkeeping -------------
        nrcStateA = pk_set_hit(nrcStateA, (uint)depth);

        //Accumulate nrcA from the v_{depth-1} -> v_depth trace (skip at
        //depth==1 because TraceCameraRay seeded nrcA0 for the camera ray).
        const bool prevSpec = pk_prev(nrcStateA);
        if (depth >= 2 && !prevSpec)
        {
            const float cosHit = max(abs(dot(-rayDir, ctx.hitNormal)), 1e-6f);
            float nrcA = load_rg_nrcA(g_pathStateBuffer, pixelIdx);
            NrcAccumulateA(nrcA, ctx.hitT, prev_pdf, cosHit);
            store_rg_nrcA(g_pathStateBuffer, pixelIdx, nrcA);
        }

        //Store v_2's outgoing direction (-rayDir at depth==3 is v_3->v_2,
        //which matches the legacy ps.v2 convention).
        if (depth == 3)
            store_ps_v2(g_pathStateBuffer, pixelIdx, -rayDir);

        //------------- Cache fire check -------------
        //NrcShouldCacheTerminate guards on hitIdx >= 3 internally, so this
        //is a no-op at depth 1/2 regardless. Eligibility derives from packed
        //pathClass (bits 10..11) so no extra live register survives.
        {
            const uint  nrcPClass = pk_pclass(nrcStateA);
            const bool  cacheElig = NrcIsEnabled() && (nrcPClass != NRC_CLASS_TRAIN_UNBIASED);
            const float nrcA0     = load_rg_nrcA0(g_pathStateBuffer, pixelIdx);
            const float nrcA      = load_rg_nrcA (g_pathStateBuffer, pixelIdx);
            const bool  shouldFire = !prevSpec &&
                (ctx.hitLocalPr >= NRC_CACHE_ROUGHNESS_MIN) &&
                NrcShouldCacheTerminate((int)depth, nrcA0, nrcA, cacheElig, nrc_area_spread_c);

            if (shouldFire)
            {
                const float3 throughput = UnpackRGB9E5(throughputPk);
                const float3 tpost      = load_rg_tpost(g_pathStateBuffer, pixelIdx);
                const uint   nrcSlot = NrcWriteTerminationRecord(
                    pixelIdx, nrc_inference_capacity,
                    ctx.hitPos, -rayDir, ctx.hitNormal,
                    ctx.hitLocalPr, ctx.hitLocalKd, ctx.hitLocalPm,
                    ctx.backface,
                    throughput, tpost, pdf_product);
                if (nrcSlot != NRC_INVALID_SLOT)
                {
                    nrcStateA = pk_set_cterm(nrcStateA);
                    if (nrcPClass == NRC_CLASS_TRAIN_BIASED)
                    {
                        //half-precision ops: half3 * half = half3
                        const half3 alphaCT = ctx.hitLocalKd * ((half)1.0 - ctx.hitLocalPm);
                        const half3 betaCT  = lerp((half3)0.04, ctx.hitLocalKd, ctx.hitLocalPm);
                        NrcStorePathTail(nrcPathId, NRC_TAIL_CACHE, nrcSlot,
                                         PackRGB9E5(NrcCleanRadiance((float3)(alphaCT + betaCT))));
                    }
                    break;
                }
            }
        }

        //------------- NEE: point light + sun -------------
        //Done before the BSDF sample so its working set (light data, bdataNEE
        //temporaries) doesn't overlap with the BSDF sample's intermediates
        //(s, bdata, updateWeight, cosTheta) in registers. NEE is paired with
        //the BSDF technique by MIS; what matters for unbiasedness is that the
        //BSDF SAMPLE call below also runs at every vertex (even if it returns
        //a degenerate direction). The trace at the iter bottom is the actual
        //BSDF evaluator and fires every iter including the last.
        float3 nrcLNeeAccum = float3(0, 0, 0);
        const bool performNEE = !(ctx.mediumMatID != MEDIUM_INVALID || LoadKd_w(ctx.matID) < EPSILON);
        if (performNEE)
        {
            //NEE used to pack/unpack Kd, Pr, Pm into matKdPk/matPrPmPk for
            //"precision-trim" parity with the BSDF eval. With ctx.hitLocal*
            //already half (same bits the pack roundtripped to), NEE and BSDF
            //already see identical precision -- the roundtrip is dead weight.
            //hitNormalPk is also gone: BSDF uses ctx.hitNormal directly so
            //NEE should too.
            //point-light technique, visibility inline per candidate
            {
                LT_LightSampleResult light = LT_SamplePointOnLight(ctx.hitPos, ctx.hitNormal, seed);

                const float3 toLight = light.position - ctx.hitPos;
                const float  dist    = sqrt(dot(toLight, toLight));
                const float3 L       = toLight / dist;

                const float cosSurf   = dot(ctx.hitNormal, L);
                const float cosLightS = dot(light.normal, -L);

                if (cosSurf > 1e-6f && cosLightS > 1e-6f)
                {
                    const float3 throughput = UnpackRGB9E5(throughputPk);

                    SamplingP sp_nee   = CalculateStrategyProbabilities(ctx.matID, -rayDir, ctx.hitNormal, ctx.iors.x, ctx.iors.y, ctx.hitLocalKd, ctx.hitLocalPm);
                    BrdfData  bdataNEE = EvaluateAndPdf_COMBINED(sp_nee, ctx.matID, ctx.hitNormal, ctx.hitNormal, L, -rayDir, ctx.hitLocalKd, ctx.hitLocalPr, ctx.hitLocalPm, ctx.iors.x, ctx.iors.y);

                    const float lightPdf = light.pdfSolidAngle;
                    const float bsdfPdf  = bdataNEE.pdf;

                    if (lightPdf > 1e-20f && bsdfPdf > 0.0f)
                    {
                        const float3 shadowOrigin = offset_ray(
                            ctx.hitPos,
                            dot(L, ctx.hitNormal) >= 0.0f ? ctx.hitNormal : -ctx.hitNormal);
                        if (IsVisibleOffset(shadowOrigin, L, dist * 0.999f))
                        {
                            const float  misWeight        = lightPdf / (lightPdf + bsdfPdf);
                            const float3 localMeasurement = light.emission * bdataNEE.val * cosSurf;
                            const float3 F_contrib        = throughput * localMeasurement * pdf_product;
                            const float  p_hat            = GetPHat(F_contrib);
                            const float  p_full           = pdf_product * lightPdf;
                            const float  wi               = (p_full > 1e-20f) ? (misWeight * p_hat / p_full) : 0.0f;

                            if (nrcPathId != NRC_INVALID_PATH)
                                nrcLNeeAccum += localMeasurement * misWeight / lightPdf;

                            if (depth == 1)
                            {
                                //primary DI, x2 is the light
                                AddInitialCandidate(wsum, g_Reservoirs_current, pixelIdx, wi,
                                    light.position, light.normal,
                                    light.emission, diMarkerFor(pixelIdx, time),
                                    float2(0, 0),
                                    MATID_LIGHT_TRI, light.objID, 1.0f,
                                    F_contrib, seed);
                            }
                            else if (depth == 2)
                            {
                                //first-bounce DI, x2 is the surface
                                AddInitialCandidate(wsum, g_Reservoirs_current, pixelIdx, wi,
                                    ctx.hitPos, ctx.hitNormal,
                                    light.emission, -L,
                                    ctx.uv,
                                    ctx.matID, ctx.instID, ctx.iors.y,
                                    F_contrib, seed);
                            }
                            else
                            {
                                const PathVertexState ps = load_ps(g_pathStateBuffer, pixelIdx);
                                const float3 tpost    = load_rg_tpost(g_pathStateBuffer, pixelIdx);
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

            //sun technique
            {
                const float2 rSun = float2(RandomFloatSingle(seed), RandomFloatSingle(seed));
                SunSampleResult sun = SampleSun(rSun);
                const float  NdotL = dot(ctx.hitNormal, sun.direction);

                if (NdotL > 1e-6f)
                {
                    const float3 throughput = UnpackRGB9E5(throughputPk);

                    SamplingP sp_nee   = CalculateStrategyProbabilities(ctx.matID, -rayDir, ctx.hitNormal, ctx.iors.x, ctx.iors.y, ctx.hitLocalKd, ctx.hitLocalPm);
                    BrdfData  bdataNEE = EvaluateAndPdf_COMBINED(sp_nee, ctx.matID, ctx.hitNormal, ctx.hitNormal, sun.direction, -rayDir, ctx.hitLocalKd, ctx.hitLocalPr, ctx.hitLocalPm, ctx.iors.x, ctx.iors.y);

                    const float lightPdf = sun.pdf;
                    const float bsdfPdf  = bdataNEE.pdf;

                    if (lightPdf > 1e-20f && bsdfPdf > 0.0f)
                    {
                        const float  misWeight        = lightPdf / (lightPdf + bsdfPdf);
                        const float3 localMeasurement = sun.radiance * bdataNEE.val * NdotL;
                        const float3 F_contrib        = throughput * localMeasurement * pdf_product;
                        const float  p_hat            = GetPHat(F_contrib);
                        const float  p_full           = pdf_product * lightPdf;

                        const float3 shadowOrigin = offset_ray(
                            ctx.hitPos,
                            dot(sun.direction, ctx.hitNormal) >= 0.0f ? ctx.hitNormal : -ctx.hitNormal);
                        if (IsVisibleOffset(shadowOrigin, sun.direction, 10000.0f))
                        {
                            const float wi = (p_full > 1e-20f) ? (misWeight * p_hat / p_full) : 0.0f;

                            if (nrcPathId != NRC_INVALID_PATH)
                                nrcLNeeAccum += localMeasurement * misWeight / lightPdf;

                            if (depth == 1)
                            {
                                AddInitialCandidate(wsum, g_Reservoirs_current, pixelIdx, wi,
                                    sun.direction, float3(0, 1, 0),
                                    sun.radiance, diMarkerFor(pixelIdx, time),
                                    float2(0, 0),
                                    MATID_ENV_MISS, MATID_ENV_MISS, 1.0f,
                                    F_contrib, seed);
                            }
                            else if (depth == 2)
                            {
                                AddInitialCandidate(wsum, g_Reservoirs_current, pixelIdx, wi,
                                    ctx.hitPos, ctx.hitNormal,
                                    sun.radiance, -sun.direction,
                                    ctx.uv,
                                    ctx.matID, ctx.instID, ctx.iors.y,
                                    F_contrib, seed);
                            }
                            else
                            {
                                const float3 tpost    = load_rg_tpost(g_pathStateBuffer, pixelIdx);
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

        //------------- BSDF sample (always runs as MIS partner for NEE above) -------------
        //Even a degenerate sample (s=0, pdf~0) is a valid "sampled BSDF" for
        //MIS purposes -- the trace below is the actual BSDF evaluator.
        SamplingP sp = CalculateStrategyProbabilities(ctx.matID, -rayDir, ctx.hitNormal, ctx.iors.x, ctx.iors.y, ctx.hitLocalKd, ctx.hitLocalPm);
        const float3 s       = SampleBRDF        (sp, ctx.matID, -rayDir, ctx.hitNormal, ctx.hitNormal, ctx.hitLocalKd, ctx.hitLocalPr, ctx.hitLocalPm, seed, ctx.iors.x, ctx.iors.y, false);
        const BrdfData bdata = EvaluateAndPdf_COMBINED(sp, ctx.matID, ctx.hitNormal, ctx.hitNormal, s, -rayDir, ctx.hitLocalKd, ctx.hitLocalPr, ctx.hitLocalPm, ctx.iors.x, ctx.iors.y, false);

        const float  cosTheta     = abs(dot(ctx.hitNormal, s));
        const float3 updateWeight = (bdata.pdf > 1e-6f)
            ? (bdata.val * ctx.absorptionTint * cosTheta) / bdata.pdf
            : float3(0, 0, 0);

        //------------- NRC training vertex (nrcLNeeAccum from NEE, updateWeight from BSDF) -------------
        if (nrcPathId != NRC_INVALID_PATH && pk_vidx(nrcStateA) < NRC_MAX_VERTICES_PER_PATH)
        {
            const uint  vIdx  = pk_vidx(nrcStateA);
            //half-precision ops on bounded material params; NrcStoreTrainingVertex
            //takes float so we widen at the call boundary.
            const half3 alpha = ctx.hitLocalKd * ((half)1.0 - ctx.hitLocalPm);
            const half3 betaC = lerp((half3)0.04, ctx.hitLocalKd, ctx.hitLocalPm);

            NrcStoreTrainingVertex(
                nrcPathId, vIdx,
                ctx.hitPos, -rayDir, ctx.hitNormal,
                (float)ctx.hitLocalPr, (float3)alpha, (float3)betaC, ctx.backface,
                PackRGB9E5(NrcCleanRadiance(nrcLNeeAccum)),
                PackRGB9E5(NrcCleanRadiance(updateWeight)));

            if (ctx.hitLocalPr >= (half)NRC_TRAIN_ROUGHNESS_MIN)
                atomic_or_rg_nrcEmitMask(g_pathStateBuffer, pixelIdx, 1u << vIdx);

            nrcStateA = pk_set_vidx(nrcStateA, vIdx + 1u);
        }

        //------------- State update + termination checks -------------
        //Bad BSDF sample terminates: no valid next direction to trace.
        if (dot(s, s) < 1e-12f || bdata.pdf <= 1e-6f ||
            any(isnan(updateWeight)) || any(isinf(updateWeight)))
            break;

        prev_pdf     = bdata.pdf;
        pdf_product  = min(pdf_product * bdata.pdf, 1e30f);
        nrcStateA    = pk_set_prev(nrcStateA, ctx.hitLocalPr < SMOOTH_SPECULAR_THRESHOLD);
        rayDir       = s;
        rayDirPk     = PackNormal(s);  //carrier for next iter back-edge
        const float3 offsetN = (dot(s, ctx.hitNormal) >= 0.0f) ? ctx.hitNormal : -ctx.hitNormal;
        rayOrigin    = offset_ray(ctx.hitPos, offsetN);

        float3 throughput        = UnpackRGB9E5(throughputPk) * updateWeight;
        const float3 tpostWeight = bdata.val * ctx.absorptionTint * cosTheta;

        //Russian roulette (depth >= 3 only). Fail terminates the path; pass
        //boosts throughput and patches the stored training beta.
        if (depth >= 3)
        {
            const float survivalProb = max(min(1.0f, Luma(throughput)), 0.1f);
            if (RandomFloatSingle(seed) >= survivalProb) break;
            const float rrBoost = 1.0f / survivalProb;
            throughput  *= rrBoost;
            pdf_product  = min(pdf_product * survivalProb, 1e30f);

            const uint vIdxAfter = pk_vidx(nrcStateA);
            if (nrcPathId != NRC_INVALID_PATH && vIdxAfter > 0u)
            {
                NrcUpdateTrainingVertexBeta(
                    nrcPathId, vIdxAfter - 1u,
                    PackRGB9E5(NrcCleanRadiance(updateWeight * rrBoost)));
            }

            //tpost is multiplicative across post-v_2 vertices.
            const float3 tpost = load_rg_tpost(g_pathStateBuffer, pixelIdx) * tpostWeight;
            store_rg_tpost(g_pathStateBuffer, pixelIdx, tpost);
        }

        throughputPk = PackRGB9E5(throughput);
        prevNormalPk = PackNormal(ctx.hitNormal);

        //=================================================================
        //TRACE v_depth -> v_{depth+1}, then extract / handle terminations.
        //This trace IS the BSDF technique paired with the NEE just done.
        //=================================================================
        if (!IsRayValid(rayOrigin, rayDir, 10000.0f))
            break;

        RayDesc rayB;
        rayB.Origin    = rayOrigin;
        rayB.Direction = rayDir;
        rayB.TMin      = 0.00001f;
        rayB.TMax      = 100000.0f;
        //secondaries force 2-state to skip alpha any-hit
        dx::HitObject hitObjB = TraceRay_Custom(SceneBVH, rayB, RAY_FLAG_FORCE_OMM_2_STATE, 0xFF);

        //----- MISS: GI miss tail (sky + MIS sun disk) -----
        if (!hitObjB.IsHit())
        {
            const float3 throughputCur = UnpackRGB9E5(throughputPk);

            const float  sunSAPdf   = GetSunPdf(rayDir);
            const float3 sunRad     = (sunSAPdf > 0.0f) ? EvaluateSun(rayDir) : float3(0, 0, 0);
            const float  sunMisBsdf = (sunSAPdf > 0.0f)
                ? prev_pdf / max(prev_pdf + sunSAPdf, EPSILON) : 0.0f;
            const float3 envL       = EvaluateSky(rayDir) + sunRad * sunMisBsdf;

            if (nrcPathId != NRC_INVALID_PATH)
                NrcStorePathTail(nrcPathId, NRC_TAIL_MISS, NRC_INVALID_SLOT,
                                 PackRGB9E5(NrcCleanRadiance(envL)));

            const float3 F_contrib = throughputCur * envL * pdf_product;
            const float  p_hat     = GetPHat(F_contrib);
            const float  wi        = (pdf_product > 1e-20f) ? (p_hat / pdf_product) : 0.0f;

            //v_{depth+1} == v_2 when depth==1, so the candidate is the
            //first-bounce env miss with x2 = rayDir. depth>=2 uses the
            //stored v_2 vertex (ps.x2 / ps.v2).
            if (depth == 1)
            {
                AddInitialCandidate(wsum, g_Reservoirs_current, pixelIdx, wi,
                    rayDir, float3(0, 1, 0),
                    envL, diMarkerFor(pixelIdx, time),
                    float2(0, 0),
                    MATID_ENV_MISS, MATID_ENV_MISS, 1.0f,
                    F_contrib, seed);
            }
            else
            {
                const PathVertexState ps = load_ps(g_pathStateBuffer, pixelIdx);
                const float3 tpost = load_rg_tpost(g_pathStateBuffer, pixelIdx);
                AddInitialCandidate(wsum, g_Reservoirs_current, pixelIdx, wi,
                    ps.x2, ps.n2_s,
                    envL * tpost, ps.v2,
                    ps.uv,
                    ps.matID, ps.objID, ps.eta,
                    F_contrib, seed);
            }
            break;
        }

        //----- HIT: extract v_{depth+1} surface -----
        const float  hitT_n   = hitObjB.GetRayTCurrent();
        const float3 hitPos_n = rayOrigin + rayDir * hitT_n;
        const uint   instID_n = hitObjB.GetInstanceIndex();
        const uint   primID_n = FlatPrimID(instID_n, hitObjB.GetGeometryIndex(), hitObjB.GetPrimitiveIndex());
        const uint   matID_n  = GetMatIDFast(instID_n, primID_n);

        BuiltInTriangleIntersectionAttributes attrB;
        hitObjB.GetAttributes(attrB);
        HitInfo hinfo_n = EvalSurfaceState(instID_n, primID_n, attrB.barycentrics, rayOrigin, (uint)depth);

        const float  matNi_n        = LoadNi(matID_n);
        const bool   transmissive_n = LoadKd_w(matID_n) < 1.0f - EPSILON;
        const bool   flipIOR_n      = hinfo_n.backface && transmissive_n;
        const float2 iors_n         = flipIOR_n ? float2(matNi_n, 1.0f) : float2(1.0f, matNi_n);
        const uint   mediumMatID_n  = flipIOR_n ? matID_n : MEDIUM_INVALID;

        float3 hitLocalKd_n; float hitLocalPr_n, hitLocalPm_n;
        RefetchMaterial(matID_n, hinfo_n.uv, hitLocalKd_n, hitLocalPr_n, hitLocalPm_n, (uint)depth);

        const float3 absorptionTint_n = (mediumMatID_n != MEDIUM_INVALID)
            ? CalculateAbsorptionThroughput(LoadTf(mediumMatID_n), hitT_n)
            : float3(1, 1, 1);

        const float3 emission_n = GetEmissionFast(instID_n, primID_n);

        //----- Emitter hit at v_{depth+1}: BSDF-technique direct emission -----
        if (any(emission_n > 0.0f) && hinfo_n.lightID != 0xFFFFFFFFu)
        {
            const float3 throughputCur = UnpackRGB9E5(throughputPk);
            const float3 prevNormalCur = UnpackNormal(prevNormalPk);
            const float  lightPdfArea  = LT_Pdf_LightTree_Area(rayOrigin, prevNormalCur, hinfo_n.lightID, instID_n);
            const float  cosLight      = max(dot(hinfo_n.hitNormal, -rayDir), 0.0f);
            const float  dist2         = max(hitT_n * hitT_n, EPSILON);
            const float  lightPdfSA    = (cosLight > EPSILON) ? (lightPdfArea * dist2 / cosLight) : 0.0f;
            const float  misWeight     = prev_pdf / max(prev_pdf + lightPdfSA, EPSILON);

            if (nrcPathId != NRC_INVALID_PATH)
                NrcStorePathTail(nrcPathId, NRC_TAIL_EMITTER, NRC_INVALID_SLOT,
                                 PackRGB9E5(NrcCleanRadiance(emission_n * misWeight)));

            const float3 F_contrib = throughputCur * emission_n * pdf_product;
            const float  p_hat     = GetPHat(F_contrib);
            const float  wi        = (pdf_product > 1e-20f) ? (misWeight * p_hat / pdf_product) : 0.0f;

            if (depth == 1)
            {
                AddInitialCandidate(wsum, g_Reservoirs_current, pixelIdx, wi,
                    hitPos_n, hinfo_n.hitNormal,
                    emission_n, diMarkerFor(pixelIdx, time),
                    float2(0, 0),
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
                    ps.uv,
                    ps.matID, ps.objID, ps.eta,
                    F_contrib, seed);
            }
            break;
        }

        //----- Non-emitter hit: stash v_2 PathVertexState at first bounce -----
        if (depth == 1)
        {
            store_ps_depth1(g_pathStateBuffer, pixelIdx,
                            hitPos_n, hinfo_n.hitNormal,
                            hinfo_n.uv, matID_n, instID_n, iors_n.y);
        }

        //----- Carry v_{depth+1} into ctx for the next iter -----
        //Bounded-range fields narrow to half on assignment (smaller phi).
        ctx.hitPos         = hitPos_n;
        ctx.hitT           = hitT_n;
        ctx.hitNormal      = hinfo_n.hitNormal;
        ctx.uv             = hinfo_n.uv;
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

    //=====================================================================
    //FINALIZE
    //=====================================================================
    const uint nrcTrainVIdxFinal     = pk_vidx(nrcStateA);
    const bool nrcCacheTerminatedFin = pk_cterm(nrcStateA);

    //tail (kind/slot/rad) was written directly at the termination point. RR
    //fallback was seeded at allocation. Head publish is last so torn reads
    //are impossible -- the fill kernel gates on numVertices != 0.
    //nrcEmitMask was atomically OR'd in coherent scratch as each vertex
    //emitted, loaded here at finalize.
    if (nrcPathId != NRC_INVALID_PATH)
    {
        const uint nrcEmitMaskFinal = load_rg_nrcEmitMask(g_pathStateBuffer, pixelIdx);
        NrcStorePathHead(nrcPathId, nrcTrainVIdxFinal, nrcEmitMaskFinal);
    }

    //cache terminated pixels defer W/M/invalidation to Pass_nrc_resolve_v8
    store_wsum(g_Reservoirs_current, pixelIdx, wsum);
    if (!nrcCacheTerminatedFin)
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
