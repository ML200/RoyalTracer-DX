#include "Includes_v8.hlsli"

//====================================================================
//EXTEND STAGE (secondary bounces only)
//====================================================================
//Reads rayOrigin / rayDir from PathState and traces one secondary ray.
//Writes the hit packet for Classify to consume. Primary (depth = 0)
//uses Pass_primary instead, which does the same trace but with
//RAY_FLAG_NONE for 4-state OMM; secondaries use FORCE_OMM_2_STATE so
//unknown states resolve via the pre-baked micromap and skip the alpha
//any-hit, which is where the bulk of alpha-test cost lives.

[shader("raygeneration")]
void Pass_extend_v8()
{
    const uint2 pixel    = DispatchRaysIndex().xy;
    const uint2 imgSize  = DispatchRaysDimensions().xy;
    const uint  pixelIdx = MapPixelID(imgSize, pixel);

    const uint flags = load_flags(g_pathStateBuffer, pixelIdx);
    if (flags & PS_FLAG_TERMINATED)
    {
        //Terminated paths don't trace. Leave hit packet stale; Classify
        //short-circuits on TERMINATED too.
        return;
    }

    const float3 rayOrigin = load_ray_origin(g_pathStateBuffer, pixelIdx);
    const float3 rayDir    = load_ray_dir   (g_pathStateBuffer, pixelIdx);

    if (!IsRayValid(rayOrigin, rayDir, 10000.0f))
    {
        store_hp_miss(g_pathStateBuffer, pixelIdx);
        store_flags(g_pathStateBuffer, pixelIdx, flags | PS_FLAG_TERMINATED);
        return;
    }

    RayDesc ray;
    ray.Origin    = rayOrigin;
    ray.Direction = rayDir;
    ray.TMin      = 0.00001f;
    ray.TMax      = 10000.0f;

    dx::HitObject hitObj = TraceRay_Custom(SceneBVH, ray, RAY_FLAG_FORCE_OMM_2_STATE, 0xFF);

    if (!hitObj.IsHit())
    {
        store_hp_miss(g_pathStateBuffer, pixelIdx);
        return;
    }

    const float hitT   = hitObj.GetRayTCurrent();
    const uint  instID = hitObj.GetInstanceIndex();
    const uint  primID = FlatPrimID(instID, hitObj.GetGeometryIndex(), hitObj.GetPrimitiveIndex());
    BuiltInTriangleIntersectionAttributes attr;
    hitObj.GetAttributes(attr);

    store_hp(g_pathStateBuffer, pixelIdx, hitT, instID, primID, attr.barycentrics);
}
