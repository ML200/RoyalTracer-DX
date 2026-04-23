#include "Includes_v8.hlsli"

//====================================================================
//PRIMARY STAGE (depth = 0)
//====================================================================
//Wavefront-pipeline entry. Performs per-pixel initialization, generates
//the camera ray, traces it, and writes the hit packet + ray to PathState
//so the subsequent (Classify, MatEvalShade, Extend) stages can process
//this bounce. Primary-only side effects, primary-hit storage to
//g_sample_current, motion-vector reflection probe, scratch[1,2] = emission
//at primary hit, scratch[1,2] = skyL1 on primary miss, are handled in
//Classify's depth==0 branches, not here, so this shader stays focused
//on pure ray-generation work.
//
//Flag state on exit:
//miss:       TERMINATED, no HAS_VALID_HIT
//hit:        HAS_VALID_HIT, PERFORM_NEE/FLIP_IOR/... set by Classify
//The hit packet stores instID = 0xFFFFFFFF on miss (see load_hp).

[shader("raygeneration")]
void Pass_primary_v8()
{
    const uint2 pixel    = DispatchRaysIndex().xy;
    const uint2 imgSize  = DispatchRaysDimensions().xy;
    const uint  pixelIdx = MapPixelID(imgSize, pixel);

    //Clear scratch slots 1 and 3, as in the monolithic Pass_raygen_v8.
    gScratchPing[uint3(pixel, 1)] = float4(0, 0, 0, 0);
    gScratchPing[uint3(pixel, 3)] = float4(0, 0, 0, 0);

    //Zero the reservoir so pixels with no accepted candidate end up empty.
    storeReservoir(g_Reservoirs_current, pixelIdx, (Reservoir)0);

    //Sentinel the depth-1 vertex stash (see init_ps).
    init_ps(g_pathStateBuffer, pixelIdx);

    //Hot-register initial state.
    const uint   throughputPk0 = PackRGB9E5(float3(1, 1, 1));
    const uint   prevNormalPk0 = PackNormal(float3(0, 1, 0));
    const float  prev_pdf0     = 1.0f;
    const float  pdf_product0  = 1.0f;
    const uint   tpostPk0      = PackRGB9E5(float3(1, 1, 1));
    const float  wsum0         = 0.0f;
    const uint   flags0        = 0u;  //depth=0, dsBounces=0, no TERMINATED, no HAS_VALID_HIT
    uint         seed          = initRandomData(pixel, uint2(8, 4), time, 1u);

    store_hot1(g_pathStateBuffer, pixelIdx, throughputPk0, prevNormalPk0, prev_pdf0, pdf_product0);
    store_hot2(g_pathStateBuffer, pixelIdx, tpostPk0, wsum0, flags0);
    //seed is written at the end, after TraceRay doesn't consume randoms.

    //Camera ray.
    const float3 rayOrigin = InitOrigin();
    const float3 rayDir    = InitDirection(pixel, float2(imgSize), seed);
    store_ray(g_pathStateBuffer, pixelIdx, rayOrigin, rayDir);

    if (!IsRayValid(rayOrigin, rayDir, 10000.0f))
    {
        store_hp_miss(g_pathStateBuffer, pixelIdx);
        store_flags(g_pathStateBuffer, pixelIdx, flags0 | PS_FLAG_TERMINATED);
        store_seed(g_pathStateBuffer, pixelIdx, seed);
        return;
    }

    RayDesc ray;
    ray.Origin    = rayOrigin;
    ray.Direction = rayDir;
    ray.TMin      = 0.00001f;
    ray.TMax      = 10000.0f;

    //Primary bounce keeps 4-state OMM (RAY_FLAG_NONE). Alpha any-hit is
    //still live so unknown-OMM candidates can commit via the alpha test.
    dx::HitObject hitObj = TraceRay_Custom(SceneBVH, ray, RAY_FLAG_NONE, 0xFF);

    if (!hitObj.IsHit())
    {
        store_hp_miss(g_pathStateBuffer, pixelIdx);
        store_seed(g_pathStateBuffer, pixelIdx, seed);
        //flags left as-is (depth=0, not terminated, not HAS_VALID_HIT).
        //Classify's miss branch at depth=0 writes skyL1 + store_sky +
        //sets TERMINATED.
        return;
    }

    const float hitT   = hitObj.GetRayTCurrent();
    const uint  instID = hitObj.GetInstanceIndex();
    const uint  primID = FlatPrimID(instID, hitObj.GetGeometryIndex(), hitObj.GetPrimitiveIndex());
    BuiltInTriangleIntersectionAttributes attr;
    hitObj.GetAttributes(attr);

    store_hp(g_pathStateBuffer, pixelIdx, hitT, instID, primID, attr.barycentrics);
    store_seed(g_pathStateBuffer, pixelIdx, seed);
    //flags left as-is; Classify runs next and does the full work.
}
