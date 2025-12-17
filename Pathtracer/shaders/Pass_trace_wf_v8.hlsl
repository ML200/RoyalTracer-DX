#include "Includes_v8.hlsli"

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    // Stack Management
    uint waveIdx = tid.x;
    uint activeCount = g_GlobalCounters.Load(g_InputStackIdx * 4);
    if (waveIdx >= activeCount) return;

    uint pixelIdx = PopStack(g_InputStackIdx, waveIdx);
    uint2 coord = UnmapPixelID(pixelIdx, gImageSize);

    // Load Path State data needed for trace
    RayGeometry rgeom = LoadRayGeometry(g_pathStateBuffer, pixelIdx);

    // Trace Ray
    RayDesc ray;
    ray.Origin = rgeom.origin;
    ray.Direction = rgeom.dir;
    ray.TMin = 0.0001f;
    ray.TMax = 10000.0f;
    HitState payload = (HitState)0.0f;
    TraceRayInline_HitState(SceneBVH, ray, payload, RAY_FLAG_NONE, 0xFF);

    {
        // Did we hit the sky?
        if(payload.instanceID == 0xFFFFFFFF){
            // We hit the skybox
                // Load the throughput up to this point
            float3 t = LoadPathThroughput(g_pathStateBuffer, pixelIdx);
            gScratchPing[uint3(coord, 1)] = float4(t * EvalMissState(), 0);
            return;
        }
        // Did we hit a light?
        float3 emission = GetEmissionFast(payload.instanceID, payload.primitiveID);
        if(length(emission > 0.0f)){
            float3 t = LoadPathThroughput(g_pathStateBuffer, pixelIdx);
            gScratchPing[uint3(coord, 1)] = float4(t * emission, 0);
            return;
        }
    }

    gScratchPing[uint3(coord, 1)] = (float4)0.0f;

    // Only push relevant work!
    StoreHitState(g_pathStateBuffer, pixelIdx, payload);
    // Also push the data on the new stack
    PushStack(g_OutputStackIdx, pixelIdx);
}