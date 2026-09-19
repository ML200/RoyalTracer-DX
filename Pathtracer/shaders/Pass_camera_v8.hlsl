#include "Includes_v8.hlsli"
#include "Raygen_Common_v8.hlsli"

inline bool TraceCameraRay(
    uint2  pixel,
    uint   pixelIdx,
    inout  uint   seed,
    inout  float3 rayOrigin,
    inout  float3 rayDir,
    out    HitContext ctx)
{
    ctx = (HitContext)0;

    if (!IsRayValid(rayOrigin, rayDir, 10000.0f))
        return false;

    RayDesc ray;
    ray.Origin    = rayOrigin;
    ray.Direction = rayDir;
    ray.TMin      = 0.00001f;
    ray.TMax      = RAY_TMAX_PLANET;
    dx::HitObject hitObj = TraceRay_Custom(SceneBVH, ray, RAY_FLAG_NONE, 0xFF);

    if (!hitObj.IsHit())
    {
        const float3 sun = EvaluateSunUnattenuated(rayDir);
        float3 skyL1     = (length(sun) > 0.0f) ? sun : float3(0, 0, 0);

        gScratchPing[uint3(pixel, 1)] = float4(skyL1, 0);
        gScratchPing[uint3(pixel, 2)] = 0.0f;
        store_sky(g_sample_current, pixelIdx);
        return false;
    }

    const float  hitT   = hitObj.GetRayTCurrent();
    const uint   instID = hitObj.GetInstanceID();
    const uint    primID   = FlatPrimID(instID, hitObj.GetGeometryIndex(), hitObj.GetPrimitiveIndex());
    const uint    matID    = GetMatIDFast(instID, primID);
    BuiltInTriangleIntersectionAttributes attr;
    hitObj.GetAttributes(attr);
    HitInfo hinfo = EvalSurfaceState(instID, primID, attr.barycentrics, rayOrigin, 0u);

    const float3 hitPos = hinfo.hitPos;
    const float3  emission = GetEmissionFast(instID, primID);

    const float  matNi        = LoadNi(matID);
    const bool   transmissive = LoadKd_w(matID) < 1.0f - EPSILON;

    const bool   flipIOR      = hinfo.backface && transmissive && !LoadIsThinGlass(matID);
    const float2 iors         = flipIOR ? float2(matNi, 1.0f) : float2(1.0f, matNi);
    const uint   mediumMatID  = flipIOR ? matID : MEDIUM_INVALID;

    float3 hitLocalKd; float hitLocalPr, hitLocalPm;
    RefetchMaterial(matID, hinfo.uv, hitLocalKd, hitLocalPr, hitLocalPm, 0u);

    const float3 absorptionTint = (mediumMatID != MEDIUM_INVALID)
        ? CalculateAbsorptionThroughput(LoadTf(mediumMatID), hitT)
        : float3(1, 1, 1);

    const bool   isEmitter = any(emission > 0.0f);

    store_instID    (g_sample_current, pixelIdx, instID);
    store_flags     (g_sample_current, pixelIdx, isEmitter, hinfo.backface);
    store_matID     (g_sample_current, pixelIdx, matID);
    store_kd        (g_sample_current, pixelIdx, hitLocalKd);
    store_prpm      (g_sample_current, pixelIdx, hitLocalPr, hitLocalPm);
    store_n1_s_world(g_sample_current, pixelIdx, hinfo.hitNormal, instID);
    store_x1        (g_sample_current, pixelIdx, hitPos, instID);

    gScratchPing[uint3(pixel, 3)] = float4(hinfo.rawNormal, 0.0f);

    if (sharc_enabled != 0u)
        gScratchPing[uint3(pixel, SHARC_DEBUG_SCRATCH)] = float4(hinfo.geometricNormal, 0.0f);
    if (isEmitter)
    {
        gScratchPing[uint3(pixel, 1)] = float4(emission, 0);
        gScratchPing[uint3(pixel, 2)] = 0.0f;
    }

    {
        // Mirror probe: the reflection's virtual point feeds specular reprojection everywhere; near-delta
        // reflectors also follow the reflection through delta surfaces for primary surface replacement.
        const float3 reflDir    = reflect(rayDir, hinfo.hitNormal);
        const float3 reflOrigin = offset_ray(hitPos, hinfo.hitNormal);
        const bool   psrCandidate = (dbg_dlssLayer & DLSS_GUIDE_OPT_NO_PSR) == 0u &&
            PsrCandidateMaterial(matID, hitLocalPr);
        const PsrChainEnd refl = PsrWalkDeltaChain(reflOrigin, reflDir, mediumMatID,
            PsrReflectionMatrix(hinfo.hitNormal), hitPos, rayDir, psrCandidate,
            psrCandidate ? DLSS_PSR_MAX_CHAIN : 1u, true);
        gScratchPing[uint3(pixel, 4)] = float4(refl.xFirst, asfloat(refl.instFirst));
        if (psrCandidate)
            gScratchPing[uint3(pixel, DLSS_PSR_CHAIN_SLOT)] = float4(refl.xVirtual, asfloat(refl.instID));
        gScratchPing[uint3(pixel, DLSS_PSR_PROBE_SLOT)] = psrCandidate ? PsrProbePack(refl) : float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    if (isEmitter && hinfo.lightID != 0xFFFFFFFFu)
        return false;

    ctx.hitPos         = hitPos;
    ctx.hitNormal      = hinfo.hitNormal;
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

[shader("raygeneration")]
// Generate primary camera hits and their persistent sample state.
void Pass_camera_v8()
{
    const uint2 pixel    = DispatchRaysIndex().xy;
    const uint2 imgSize  = DispatchRaysDimensions().xy;
    const uint  pixelIdx = MapPixelID(imgSize, pixel);
    if(cloudEnabled>.5f) g_cumulusQueries.Store4(CumulusQueryAddress(pixel),0u);

    if (all(pixel == uint2(0, 0))) {
        gAutoExpose.Store(SENT_OFFS_MASK,      0u);
        gAutoExpose.Store(SENT_OFFS_MAXLUMA,   0u);
        gAutoExpose.Store(SENT_OFFS_MAXMV,     0u);
        gAutoExpose.Store(SENT_OFFS_MAXSPECMV, 0u);
        gAutoExpose.Store(SENT_OFFS_CAPCOUNT,  0u);
        gAutoExpose.Store(SENT_OFFS_BADCOUNT,  0u);
        gAutoExpose.Store(SENT_OFFS_FIRSTBAD,  0u);
    }

    store_sky(g_sample_current, pixelIdx);

    gScratchPing[uint3(pixel, 1)] = float4(0, 0, 0, 0);

    if (!PT_ONLY_MODE)
        storeReservoir(g_Reservoirs_current, pixelIdx, (Reservoir)0);

    uint   seed = initRandomData(pixel, uint2(8, 4), time, 1u);
    float3 rayOrigin;
    float3 rayDir;
    InitCameraRayDoF(pixel, imgSize, seed, rayOrigin, rayDir);

    SetSkyObserver(InitOrigin() + sceneOriginWorld);

    HitContext ctx;
    if (!TraceCameraRay(pixel, pixelIdx, seed, rayOrigin, rayDir, ctx))
    {
        const uint f = load_flagsWord(g_sample_current, pixelIdx) | SD_FLAG_NOBOUNCE;
        store_flagsWord(g_sample_current, pixelIdx, f);
        if (!PT_ONLY_MODE)
            FinalizeReservoir(pixelIdx, 0.0f);
        return;
    }

    if (!PT_ONLY_MODE)
    {
        uint slot;
        g_raygenQueue.InterlockedAdd(0, 1u, slot);
        g_raygenQueue.Store(16u + slot * 4u, (pixel.x & 0xFFFFu) | (pixel.y << 16));
    }

    if (SPMIS_SPATIAL_MODE)
    {
        const uint ts   = max(spmis_tileSize, 1u);
        int2 tjit = int2(0, 0);
        if (SPMIS_CELL_JITTER)
        {
            const uint jh = SP_h2_xxhash32(asuint(time));
            tjit = int2((int)(jh % ts), (int)(SP_h2_xxhash32(jh) % ts));
        }

        uint sp_checksum;
        const uint sp_hash = SP_screen_hash((int)pixel.x + tjit.x, (int)pixel.y + tjit.y, ts,
                                            ctx.hitPos, ctx.hitNormal, sp_checksum);
        g_spmisBuffer.Store(SP_A(SP_HASH, pixelIdx),
                            SP_insert(sp_hash % SP_NUMCELLS(), sp_checksum, SP_NUMCELLS()));
    }

    store_rg_primaryExtra(g_pathStateBuffer, pixelIdx,
                          (float2)ctx.iors, ctx.mediumMatID, (float3)ctx.absorptionTint);
}
