#include "Includes_v8.hlsli"

// False when no path is needed: a miss or a directly seen mesh light.
inline bool TraceCameraRay(uint2 pixel, uint pixelIdx, float3 rayOrigin, float3 rayDir, bool cameraWater)
{
    RayDesc ray;
    ray.Origin    = rayOrigin;
    ray.Direction = rayDir;
    ray.TMin      = 0.00001f;
    ray.TMax      = RAY_TMAX_PLANET;
    if (!IsRayDescValid(ray))
        return false;
    dx::HitObject hitObj = TraceRay_Custom(SceneBVH, ray, RAY_FLAG_NONE, 0xFF);

    if (!hitObj.IsHit())
    {
        const float3 sun = EvaluateSunUnattenuated(rayDir);
        gScratchPing[uint3(pixel, 1)] = float4((length(sun) > 0.0f) ? sun : float3(0, 0, 0), 0);
        gScratchPing[uint3(pixel, 2)] = 0.0f;
        store_sky(g_sample_current, pixelIdx);
        return false;
    }

    const float hitT   = hitObj.GetRayTCurrent();
    const uint  instID = hitObj.GetInstanceID();
    const uint  primID = FlatPrimID(instID, hitObj.GetGeometryIndex(), hitObj.GetPrimitiveIndex());
    BuiltInTriangleIntersectionAttributes attr;
    hitObj.GetAttributes(attr);
    const HitInfo hinfo    = EvalSurfaceState(instID, primID, attr.barycentrics, rayOrigin, PixelConeAngle() * hitT);
    const uint matID = ResolveSurfaceMaterial(GetMatIDFast(instID, primID), hinfo);
    const float3  hitPos   = hinfo.hitPos;
    const float3  emission = GetEmissionFast(instID, primID);
    const bool    isEmitter = any(emission > 0.0f);

    // Correspondence only; the motion pass evaluates the previous geometry.
    if (hinfo.isOcean) {
        gScratchPing[uint3(pixel, OCEAN_PREVIOUS_POSITION_SLOT)] =
            float4(attr.barycentrics, asfloat(primID), asfloat(instID));
        gScratchPing[uint3(pixel, OCEAN_GUIDE_SLOT)] = float4(hinfo.oceanFoam, hinfo.oceanBubbles, 0.0f, 0.0f);
    }

    // Material: written out, then dead.
    uint mediumMatID;
    bool psrCandidate;
    {
        const float matNi        = LoadNi(matID);
        const bool  transmissive = LoadKd_w(matID) < 1.0f - EPSILON;
        const bool  flipIOR      = hinfo.backface && transmissive && !LoadIsThinGlass(matID);
        mediumMatID = flipIOR ? matID : MEDIUM_INVALID;

        float3 hitLocalKd; float hitLocalPr, hitLocalPm;
        RefetchMaterial(matID, hinfo, hitLocalKd, hitLocalPr, hitLocalPm);
        psrCandidate = !hinfo.isOcean && (dbg_dlssLayer & DLSS_GUIDE_OPT_NO_PSR) == 0u &&
            PsrCandidateMaterial(matID, hitLocalPr) && !cameraWater;

        store_instID    (g_sample_current, pixelIdx, instID);
        store_flags     (g_sample_current, pixelIdx, isEmitter, hinfo.backface);
        store_matID     (g_sample_current, pixelIdx, matID);
        store_kd        (g_sample_current, pixelIdx, hitLocalKd);
        store_prpm      (g_sample_current, pixelIdx, hitLocalPr, hitLocalPm);
        store_n1_s_world(g_sample_current, pixelIdx, hinfo.hitNormal, instID);
        store_x1        (g_sample_current, pixelIdx, hitPos, instID);
        store_rg_primaryExtra(pixelIdx, flipIOR ? float2(matNi, 1.0f) : float2(1.0f, matNi), mediumMatID,
            flipIOR && !LoadIsOceanMaterial(matID) ? CalculateAbsorptionThroughput(LoadTf(matID), hitT) : float3(1, 1, 1));
    }

    gScratchPing[uint3(pixel, 3)] = float4(hinfo.rawNormal, 0.0f);
    if (sharc_enabled != 0u)
        gScratchPing[uint3(pixel, SHARC_DEBUG_SCRATCH)] = float4(hinfo.geometricNormal, 0.0f);
    if (isEmitter)
    {
        gScratchPing[uint3(pixel, 1)] = float4(emission, 0);
        gScratchPing[uint3(pixel, 2)] = 0.0f;
    }
    const bool shade = !(isEmitter && hinfo.lightID != 0xFFFFFFFFu);

    // Mirror probe for specular reprojection; PSR chain for near-delta reflectors.
    if (hinfo.isOcean)
    {
        gScratchPing[uint3(pixel, 4)] = float4(hitPos, asfloat(instID));
        gScratchPing[uint3(pixel, DLSS_PSR_CHAIN_SLOT)] = float4(hitPos, asfloat(instID));
        gScratchPing[uint3(pixel, DLSS_PSR_PROBE_SLOT)] = 0.0f;
    }
    else
    {
        const float3 reflDir    = reflect(rayDir, hinfo.hitNormal);
        const float3 reflOrigin = offset_ray(hitPos, hinfo.hitNormal);
        const PsrChainEnd refl = PsrWalkDeltaChain(reflOrigin, reflDir, mediumMatID,
            PsrReflectionMatrix(hinfo.hitNormal), hitPos, rayDir, psrCandidate,
            psrCandidate ? DLSS_PSR_MAX_CHAIN : 1u, true);
        gScratchPing[uint3(pixel, 4)] = float4(refl.xFirst, asfloat(refl.instFirst));
        if (psrCandidate)
            gScratchPing[uint3(pixel, DLSS_PSR_CHAIN_SLOT)] = float4(refl.xVirtual, asfloat(refl.instID));
        gScratchPing[uint3(pixel, DLSS_PSR_PROBE_SLOT)] = psrCandidate ? PsrProbePack(refl) : float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    return shade;
}

[shader("raygeneration")]
void Pass_camera_v8()
{
    const uint2 pixel    = DispatchRaysIndex().xy;
    const uint2 imgSize  = DispatchRaysDimensions().xy;
    const uint  pixelIdx = MapPixelID(imgSize, pixel);

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
    gScratchPing[uint3(pixel, OCEAN_PREVIOUS_POSITION_SLOT)] = float4(0, 0, 0, asfloat(0xFFFFFFFFu));

    gScratchPing[uint3(pixel, 1)] = float4(0, 0, 0, 0);

    uint   seed = initRandomData(pixel, uint2(8, 4), time, 1u);
    float3 rayOrigin;
    float3 rayDir;
    InitCameraRayDoF(pixel, imgSize, seed, rayOrigin, rayDir);

    SetSkyObserver(InitOrigin() + sceneOriginWorld);

    bool cameraWater = OceanPointInside(rayOrigin);
    if (!TraceCameraRay(pixel, pixelIdx, rayOrigin, rayDir, cameraWater))
        store_flagsWord(g_sample_current, pixelIdx, load_flagsWord(g_sample_current, pixelIdx) | SD_FLAG_NOBOUNCE);
    // Visible water: the triangle side beats the height-field guess.
    if (OceanMediumEnabled() && load_instID(g_sample_current,pixelIdx) != 0xffffffffu &&
        LoadIsOceanMaterial(load_matID(g_sample_current,pixelIdx)))
        cameraWater = load_backface(g_sample_current,pixelIdx);
    if (cameraWater) {
        store_flagsWord(g_sample_current,pixelIdx,load_flagsWord(g_sample_current,pixelIdx) | SD_FLAG_CAMERA_WATER);
        // The path pass adds underwater radiance; the record is for guides only.
        gScratchPing[uint3(pixel,1)] = 0.0f;
    }
}
