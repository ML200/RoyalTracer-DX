/*
Camera ray operations, optimized
*/

//Initial ray origin
float3 InitOrigin(){
    return mul(viewI, float4(0, 0, 0, 1)).xyz;
}

//Initial ray direction
float3 InitDirection(){
    float2 d = (((DispatchRaysIndex().xy) / float2(DispatchRaysDimensions().xy)) * 2.f - 1.f);
    float4 target = mul(projectionI, float4(d.x, -d.y, 1, 1));
    return normalize(mul(viewI, float4(target.xyz, 0)).xyz);
}

//Pixel idx for directly writing into the sample data -> more efficient
SampleData SampleCameraRay(uint idx){
    RayDesc ray;
    ray.Origin = InitOrigin();
    ray.Direction = InitDirection();
    ray.TMin = 0.0001;
    ray.TMax = 10000;

    store_o(-ray.Direction, g_sample_current, idx);

    // Trace the camera ray
    HitInfo payload;
    TraceRay(SceneBVH, RAY_FLAG_NONE, 0xFF, 0, 0, 0, ray, payload);

    float3 ke = materials[payload.materialID].Ke;

    // Compress and save relevant data: x1, L1, n1, mID and oID
    store_x1(payload.hitPosition, g_sample_current, idx);
    store_n1(payload.hitNormal, g_sample_current, idx);
    store_L1(ke, g_sample_current, idx);
    store_matID(payload.materialID, g_sample_current, idx);
    store_objID(payload.objID, g_sample_current, idx);

    SampleData sdata = (SampleData)0;
    sdata.x1 = payload.hitPosition;
    sdata.n1 = payload.hitNormal;
    sdata.L1 = ke;
    sdata.o = -ray.Direction;
    sdata.objID = payload.objID;
    sdata.matID = payload.materialID;

    //return the sample data
    return sdata;
}

inline float2 GetLastFramePixelCoordinates_Float(
    float3 worldPos,
    float4x4 prevView,
    float4x4 prevProjection,
    float2 resolution,
    uint objID)
{
    // 1. Convert current world-space position back into the local space of this object:
    float4 localPos = mul(instanceProps[objID].objectToWorldInverse, float4(worldPos, 1.0f));

    // 2. Transform that local position by the *previous* frame's object-to-world matrix:
    float4 prevWorldPos = mul(instanceProps[objID].prevObjectToWorld, localPos);

    // 3. Project it into clip space using the previous frame’s view and projection:
    float4 clipPos = mul(prevProjection, mul(prevView, prevWorldPos));

    // If the clip-space w is not positive, it means the position was behind the camera last frame:
    if (clipPos.w <= 0.0f)
    {
        // Return some sentinel value that indicates it's off-screen or invalid:
        return float2(-1.0f, -1.0f);
    }

    // 4. Convert clip space to normalized device coordinates:
    float2 ndc = clipPos.xy / clipPos.w;

    // 5. Transform NDC (-1..1) to screen UV (0..1):
    float2 screenUV = ndc * 0.5f + 0.5f;

    // 6. Flip Y if needed (common in many rendering APIs):
    screenUV.y = 1.0f - screenUV.y;

    // 7. Finally convert to actual pixel coordinates:
    return screenUV * resolution;
}

inline int2 GetBestReprojectedPixel_d(
    float3 worldPos,
    float4x4 prevView,
    float4x4 prevProjection,
    float2 resolution,
    uint objID
    )
{
    float2 subPixelCoord = GetLastFramePixelCoordinates_Float(worldPos, prevView, prevProjection, resolution, objID);
    int2 pixel = int2(round(subPixelCoord));
    return pixel;
}