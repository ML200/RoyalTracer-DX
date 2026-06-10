//====================================
//DEBUG PASS — PRIMARY-RAY CHUNK / TRIANGLE INSPECTION
//====================================
//Temporary planet-bring-up pass. One pinhole camera ray per pixel against
//SceneBVH; flat-shades each hit by a hashed colour. Writes output layer 0.
//All other rendering passes are disabled; see m_passes.Build in Renderer.cpp.

#include "Includes_v8.hlsli"

//====================================
//COLOUR MODE
//====================================
//0 = per CHUNK  — hash of InstanceID, so each terrain chunk / fallback face is
//    ONE flat colour. This is the view for inspecting the streaming system:
//    cell SIZE shows the LOD (small near the camera, large far away), and the
//    cell pattern visibly changes as chunks stream in/terrain. Use this to answer
//    "is the LOD camera-adaptive / is the terrain updating".
//1 = per TRIANGLE — hash of InstanceID+primID. Shows the raw tessellation grid,
//    but at CHUNK_GRID=128 each chunk is 128*128*2 = 32768 triangles, so a whole
//    chunk reads as random noise — useless for seeing chunk-level structure.
#define DBG_COLOR_MODE 1

//====================================
//CRASH BISECTION TOGGLE
//====================================
//1 = trace SceneBVH normally. 0 = DO NOT trace — write a flat camera-ray
//gradient instead. The planet copy/compute queues (chunk BLAS builds + the
//unified TLAS build) run every frame regardless; only ray TRAVERSAL is gated.
//  crash GONE at 0     => hang is in TRAVERSAL (a chunk BLAS / the TLAS).
//  crash PERSISTS at 0 => hang is in the planet COMPUTE work (a build).
#define DBG_TRACE 1

//integer hash -> distinct, well-spread RGB (Wellons' lowbias32 finalizer).
float3 DbgHashColor(uint key)
{
    key ^= key >> 16; key *= 0x7feb352du;
    key ^= key >> 15; key *= 0x846ca68bu;
    key ^= key >> 16;
    return float3(uint3(key, key >> 8, key >> 16) & 0xFFu) / 255.0f;
}

[shader("raygeneration")]
void Pass_debug_v8()
{
    const uint2 pixel   = DispatchRaysIndex().xy;
    const uint2 imgSize = DispatchRaysDimensions().xy;

    uint   seed      = 0u;                       // pinhole InitDirection ignores it
    float3 rayOrigin = InitOrigin();
    float3 rayDir    = InitDirection(pixel, imgSize, seed);

#if DBG_TRACE
    float3 outColor = float3(0.0f, 0.0f, 0.0f);  // miss -> black

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
            //per-triangle: mix instance + primitive
            const uint primID = FlatPrimID(instID,
                                           hitObj.GetGeometryIndex(),
                                           hitObj.GetPrimitiveIndex());
            outColor = DbgHashColor(primID * 0x9E3779B9u + instID);
        #else
            //per-chunk: one flat colour per instance (terrain chunk / fallback
            //face / scene mesh). Cell size = LOD; pattern shift = streaming.
            outColor = DbgHashColor(instID);
        #endif
        }
    }
#else
    //no trace - SceneBVH is never touched. A flat camera-ray gradient just
    //proves the pass ran and rays are being generated.
    float3 outColor = rayDir * 0.5f + 0.5f;
#endif

    gOutput[uint3(pixel, 0)] = float4(outColor, 1.0f);
}
