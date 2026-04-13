#include "Includes_raygen_v8.hlsli"

[shader("raygeneration")]
void Pass_camera_v8()
{
    uint2 pixel   = DispatchRaysIndex().xy;
    uint2 imgSize = DispatchRaysDimensions().xy;
    uint pixelIdx = MapPixelID(imgSize, pixel);

    // ── Clear reservoirs ───────────────────────────────────────────────
    storeReservoirDI(g_Reservoirs_current_di, pixelIdx, (Reservoir_DI)0);
    store_wsum_di(g_Reservoirs_current_di, pixelIdx, 0.0f);
    store_W_di(g_Reservoirs_current_di, pixelIdx, 0.0f);
    store_phat_di(g_Reservoirs_current_di, pixelIdx, 0.0f);

    storeReservoirGI(g_Reservoirs_current_gi, pixelIdx, (Reservoir_GI)0);
    store_wsum_gi(g_Reservoirs_current_gi, pixelIdx, 0.0f);
    store_W_gi(g_Reservoirs_current_gi, pixelIdx, 0.0f);
    store_F_gi(g_Reservoirs_current_gi, pixelIdx, 0u);
    store_F_mag_gi(g_Reservoirs_current_gi, pixelIdx, 0.0f);
    store_M_gi(g_Reservoirs_current_gi, pixelIdx, 0u);
    store_Tpost_gi(g_Reservoirs_current_gi, pixelIdx, 1.0f);

    gScratchPing[uint3(pixel, 3)] = float4(0, 0, 0, 0);

    // ── Generate & trace camera ray ────────────────────────────────────
    uint   seed      = initRandomData(pixel, uint2(8, 4), time, 1u);
    float3 rayOrigin = InitOrigin();
    float3 rayDir    = InitDirection(pixel, float2(imgSize), seed);

    RayDesc ray;
    ray.Origin    = rayOrigin;
    ray.Direction = rayDir;
    ray.TMin      = 0.00001f;
    ray.TMax      = 10000.0f;
    dx::HitObject hitObj = TraceRay_Custom(SceneBVH, ray, RAY_FLAG_NONE, 0xFF);

    // ── Miss ───────────────────────────────────────────────────────────
    if (!hitObj.IsHit())
    {
        float3 sun   = EvaluateSun(rayDir);
        float3 skyL1 = EvalMissState(rayDir, sun);
        if (length(sun) > 0.0f) skyL1 = sun;
        gScratchPing[uint3(pixel, 1)] = float4(skyL1, 0);
        gScratchPing[uint3(pixel, 2)] = float4(skyL1, 0);
        store_sky(g_sample_current, pixelIdx);
        return;
    }

    // ── Hit ────────────────────────────────────────────────────────────
    float  hitT   = hitObj.GetRayTCurrent();
    float3 hitPos = rayOrigin + rayDir * hitT;

    const uint instID = hitObj.GetInstanceIndex();
    const uint primID = FlatPrimID(instID, hitObj.GetGeometryIndex(), hitObj.GetPrimitiveIndex());

    BuiltInTriangleIntersectionAttributes attr;
    hitObj.GetAttributes(attr);
    HitInfo hinfo = EvalSurfaceState(instID, primID, attr.barycentrics, rayOrigin, 0);

    float3 emission  = GetEmissionFast(instID, primID);
    bool   isEmitter = any(emission > 0.0f);

    // Store G-buffer
    store_instID(g_sample_current, pixelIdx, instID);
    store_primID(g_sample_current, pixelIdx, primID, isEmitter);
    store_bary(g_sample_current, pixelIdx, attr.barycentrics);
    store_n1_s_world(g_sample_current, pixelIdx, hinfo.hitNormal, instID);
    store_uv(g_sample_current, pixelIdx, hinfo.uv);

    if (isEmitter)
    {
        gScratchPing[uint3(pixel, 1)] = float4(emission, 0);
        gScratchPing[uint3(pixel, 2)] = float4(emission, 0);
    }

    // ── Specular motion-vector reflection probe ────────────────────────
    {
        float3 reflDir    = reflect(rayDir, hinfo.hitNormal);
        float3 reflOrigin = offset_ray(hitPos, hinfo.hitNormal);
        RayDesc reflRay;
        reflRay.Origin    = reflOrigin;
        reflRay.Direction = reflDir;
        reflRay.TMin      = 0.00001f;
        reflRay.TMax      = 10000.0f;

        RayQuery<RAY_FLAG_NONE> q;
        q.TraceRayInline(SceneBVH, RAY_FLAG_NONE, 0xFF, reflRay);
        while (q.Proceed())
        {
            if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
            {
                uint cInstID = q.CandidateInstanceIndex();
                uint cPrimID = FlatPrimID(cInstID, q.CandidateGeometryIndex(), q.CandidatePrimitiveIndex());
                uint cMatID  = GetMatIDFast(cInstID, cPrimID);
                float alpha  = materials[cMatID].alphaThreshold;
                if (alpha < 1.0f)
                    q.CommitNonOpaqueTriangleHit();
            }
        }

        if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
        {
            float3 reflPos    = reflOrigin + reflDir * q.CommittedRayT();
            float3 virtualPos = reflPos - 2.0f * dot(reflPos - hitPos, hinfo.hitNormal) * hinfo.hitNormal;
            gScratchPing[uint3(pixel, 4)] = float4(virtualPos, asfloat(instID));
        }
        else
        {
            gScratchPing[uint3(pixel, 4)] = float4(0, 0, 0, asfloat(0xFFFFFFFFu));
        }
    }
}
