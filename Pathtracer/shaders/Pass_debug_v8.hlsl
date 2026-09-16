#include "Includes_v8.hlsli"

#define DBG_COLOR_MODE 1

#define DBG_TRACE 1

float3 DbgHashColor(uint key)
{
    key ^= key >> 16; key *= 0x7feb352du;
    key ^= key >> 15; key *= 0x846ca68bu;
    key ^= key >> 16;
    return float3(uint3(key, key >> 8, key >> 16) & 0xFFu) / 255.0f;
}

[shader("raygeneration")]
// Render selected diagnostic buffers into the debug target.
void Pass_debug_v8()
{
    const uint2 pixel   = DispatchRaysIndex().xy;
    const uint2 imgSize = DispatchRaysDimensions().xy;

    uint   seed      = 0u;
    float3 rayOrigin = InitOrigin();
    float3 rayDir    = InitDirection(pixel, imgSize, seed);

#if DBG_TRACE
    float3 outColor = float3(0.0f, 0.0f, 0.0f);

    if (IsRayValid(rayOrigin, rayDir, RAY_TMAX_PLANET))
    {
        RayDesc ray;
        ray.Origin    = rayOrigin;
        ray.Direction = rayDir;
        ray.TMin      = 0.00001f;
        ray.TMax      = RAY_TMAX_PLANET;

        dx::HitObject hitObj = TraceRay_Custom(SceneBVH, ray, RAY_FLAG_NONE, 0xFF);

        if (hitObj.IsHit())
        {
            const uint instID = hitObj.GetInstanceID();
        #if DBG_COLOR_MODE == 1

            const uint primID = FlatPrimID(instID,
                                           hitObj.GetGeometryIndex(),
                                           hitObj.GetPrimitiveIndex());
            outColor = DbgHashColor(primID * 0x9E3779B9u + instID);
        #else

            outColor = DbgHashColor(instID);
        #endif
        }
    }
#else

    float3 outColor = rayDir * 0.5f + 0.5f;
#endif

    gOutput[uint3(pixel, 0)] = float4(outColor, 1.0f);
}
