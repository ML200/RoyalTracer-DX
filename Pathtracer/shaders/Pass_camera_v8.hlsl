#include "Includes_v8.hlsli"
#include "Raygen_Common_v8.hlsli"


//====================================
//CAMERA RAY (primary-hit extraction)
//====================================
//Trace v_1 and write the primary outputs: sky/emitter scratch, the resolved
//G-buffer downstream passes read, and the specular-MV probe. Returns false
//(terminal) on degenerate/miss/direct-emitter; else fills ctx with v_1.
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

    //primary: 4-state OMM (full alpha test); bounces force 2-state
    RayDesc ray;
    ray.Origin    = rayOrigin;
    ray.Direction = rayDir;
    ray.TMin      = 0.00001f;
    ray.TMax      = RAY_TMAX_PLANET;
    dx::HitObject hitObj = TraceRay_Custom(SceneBVH, ray, RAY_FLAG_NONE, 0xFF);

    //MISS: sun disc only. Pass_clouds_primary writes the full sky (slot 10) +
    //combined transmittance (slot 11); the shading composite does sun*combinedTr.
    //EvaluateSunUnattenuated (NOT EvaluateSun): the latter bakes transmittance
    //the composite already applies, double-attenuating.
    if (!hitObj.IsHit())
    {
        const float3 sun = EvaluateSunUnattenuated(rayDir);
        float3 skyL1     = (length(sun) > 0.0f) ? sun : float3(0, 0, 0);

        gScratchPing[uint3(pixel, 1)] = float4(skyL1, 0);
        gScratchPing[uint3(pixel, 2)] = float4(skyL1, 0);
        store_sky(g_sample_current, pixelIdx);
        return false;
    }

    //HIT: resolve the surface from instID + primID (closest-hit is a stub).
    const float  hitT   = hitObj.GetRayTCurrent();
    const float3 hitPos = rayOrigin + rayDir * hitT;
    const uint   instID = hitObj.GetInstanceID();   // user InstanceID == instanceProps index
    const uint    primID   = FlatPrimID(instID, hitObj.GetGeometryIndex(), hitObj.GetPrimitiveIndex());
    const uint    matID    = GetMatIDFast(instID, primID);
    BuiltInTriangleIntersectionAttributes attr;
    hitObj.GetAttributes(attr);
    HitInfo hinfo = EvalSurfaceState(instID, primID, attr.barycentrics, rayOrigin, 0u);
    const float3  emission = GetEmissionFast(instID, primID);

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

    const bool   isEmitter = any(emission > 0.0f);

    //Bake the resolved surface to the G-buffer (no later pass re-reads
    //per-triangle data or a material texture).
    store_instID    (g_sample_current, pixelIdx, instID);
    store_flags     (g_sample_current, pixelIdx, isEmitter, hinfo.backface);
    store_matID     (g_sample_current, pixelIdx, matID);
    store_kd        (g_sample_current, pixelIdx, hitLocalKd);
    store_prpm      (g_sample_current, pixelIdx, hitLocalPr, hitLocalPm);
    store_n1_s_world(g_sample_current, pixelIdx, hinfo.hitNormal, instID);
    store_x1        (g_sample_current, pixelIdx, hitPos, instID);
    if (isEmitter)
    {
        gScratchPing[uint3(pixel, 1)] = float4(emission, 0);
        gScratchPing[uint3(pixel, 2)] = float4(emission, 0);
    }

    //Specular-MV probe: virtualPos -> scratch slot 4 (RayQuery scoped here).
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
            reflRay.TMax      = RAY_TMAX_PLANET;

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

    //direct emitter hit terminates at the camera vertex
    if (isEmitter && hinfo.lightID != 0xFFFFFFFFu)
        return false;

    //hand v_1 to the bounce loop (bounded fields -> half)
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


//====================================
//CAMERA PASS
//====================================
//Primary hit only: writes the G-buffer + primary scratch (slots 1/2/4), the
//SPMIS hash cell, and the primaryExtra (iors/medium/absorption) HOT2 record.
//Terminal pixels (miss / degenerate / direct emitter) are finalized here and
//flagged SD_FLAG_NOBOUNCE so Pass_raygen skips their bounce loop. The bounce
//pass rebuilds the primary ctx from this G-buffer per sample.
[shader("raygeneration")]
void Pass_camera_v8()
{
    const uint2 pixel    = DispatchRaysIndex().xy;
    const uint2 imgSize  = DispatchRaysDimensions().xy;
    const uint  pixelIdx = MapPixelID(imgSize, pixel);

    //Default the G-buffer to a clean miss so a degenerate camera ray (which
    //TraceCameraRay leaves unwritten) still reads as sky downstream. A real hit
    //or a miss overwrites this inside TraceCameraRay.
    store_sky(g_sample_current, pixelIdx);

    //slot 1 primary emitter/sky, slot 3 x1 direct (cleared for the bounce pass)
    gScratchPing[uint3(pixel, 1)] = float4(0, 0, 0, 0);
    gScratchPing[uint3(pixel, 3)] = float4(0, 0, 0, 0);

    storeReservoir(g_Reservoirs_current, pixelIdx, (Reservoir)0);

    uint   seed = initRandomData(pixel, uint2(8, 4), time, 1u);
    float3 rayOrigin;
    float3 rayDir;
    InitCameraRayDoF(pixel, imgSize, seed, rayOrigin, rayDir);

    //sky/sun sampler reads camera altitude here; view space is floating-origin
    //shifted, so add sceneOriginWorld back for absolute.
    SetSkyObserver(InitOrigin() + sceneOriginWorld);

    //CAMERA RAY: trace v_1; terminal (miss/degenerate/direct emitter) -> finalize
    //the (empty) reservoir and mark NOBOUNCE so the bounce pass skips this pixel.
    HitContext ctx;
    if (!TraceCameraRay(pixel, pixelIdx, seed, rayOrigin, rayDir, ctx))
    {
        const uint f = load_flagsWord(g_sample_current, pixelIdx) | SD_FLAG_NOBOUNCE;
        store_flagsWord(g_sample_current, pixelIdx, f);
        FinalizeReservoir(pixelIdx, 0.0f);   // wsum==0: W=0, reservoir invalidated
        return;
    }

    //====================================
    //SPMIS HASH INSERTION
    //====================================
    //Insert this pixel into the SPMIS hash grid for the spatial-reuse pass. The
    //screen-tile origin is jittered per frame (from `time`) so cell boundaries
    //average out under accumulation. No-op when SPMIS is off.
    if (SPMIS_SPATIAL_MODE)
    {
        const uint ts   = max(spmis_tileSize, 1u);
        const uint jh   = SP_h2_xxhash32(asuint(time));
        const int2 tjit = int2((int)(jh % ts), (int)(SP_h2_xxhash32(jh) % ts));

        uint sp_checksum;
        const uint sp_hash = SP_screen_hash((int)pixel.x + tjit.x, (int)pixel.y + tjit.y, ts,
                                            ctx.hitPos, ctx.hitNormal, sp_checksum);
        g_spmisBuffer.Store(SP_A(SP_HASH, pixelIdx),
                            SP_insert(sp_hash % SP_NUMCELLS(), sp_checksum, SP_NUMCELLS()));
    }

    //Hand the primary extras to the bounce pass (it rebuilds ctx from the
    //G-buffer + these per sample).
    store_rg_primaryExtra(g_pathStateBuffer, pixelIdx,
                          (float2)ctx.iors, ctx.mediumMatID, (float3)ctx.absorptionTint);
}
