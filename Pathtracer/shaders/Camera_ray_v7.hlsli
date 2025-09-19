/*
Camera ray operations, optimized
*/

//Initial ray origin
float3 InitOrigin(){
    return mul(viewI, float4(0, 0, 0, 1)).xyz;
}

//Initial ray direction
float3 InitDirection(uint2 pixel, uint2 imgSize){
    float2 d = (((pixel) / float2(imgSize)) * 2.f - 1.f);
    float4 target = mul(projectionI, float4(d.x, -d.y, 1, 1));
    return normalize(mul(viewI, float4(target.xyz, 0)).xyz);
}

//Pixel idx for directly writing into the sample data -> more efficient
SampleData SampleCameraRay(uint idx, uint2 pixel, uint2 imgSize){
    RayDesc ray;
    ray.Origin = InitOrigin();
    ray.Direction = InitDirection(pixel, imgSize);
    ray.TMin = 0.0001;
    ray.TMax = 10000;

    // Trace the camera ray
    HitInfo payload;
    TraceRayInline_HitInfo(SceneBVH, ray, payload, RAY_FLAG_NONE, 0xFF);

    float3 ke = materials[payload.materialID].Ke;
    if(dot(payload.hitNormal,ray.Direction) > 0.0f){
        ke = 0.0f;
        payload.hitNormal = -payload.hitNormal;
    }

    SampleData sdata = (SampleData)0;
    sdata.x1 = payload.hitPosition;
    sdata.n1 = payload.hitNormal;
    sdata.L1 = ke;
    sdata.o = -ray.Direction;
    sdata.objID = payload.objID;
    sdata.matID = payload.materialID;

    storeSampleData(g_sample_current, idx, sdata);

    //return the sample data
    return sdata;
}

static const float2  kInvalidUV     = float2(-2.0f, -2.0f);
static const int2    kInvalidPixel  = int2(-1, -1);

static const float kReprojectionProbExponent = 1.0f;

inline float2 GetLastFramePixelCoordinates_Float(
    float3 worldPos,
    float4x4 prevView,
    float4x4 prevProjection,
    float2 resolution,
    uint objID)
{
    float4 localPos     = mul(instanceProps[objID].objectToWorldInverse, float4(worldPos, 1.0f));
    float4 prevWorldPos = mul(instanceProps[objID].prevObjectToWorld, localPos);
    float4 clipPos      = mul(prevProjection, mul(prevView, prevWorldPos));

    // reject behind camera or bad w
    if (clipPos.w <= 0.0f || !isfinite(clipPos.w)) return float2(-1.0f, -1.0f);

    float2 ndc = clipPos.xy / clipPos.w;

    // reject if outside the clip volume
    if (any(ndc < -1.0f) || any(ndc > 1.0f)) return float2(-1.0f, -1.0f);

    float2 screenUV = ndc * 0.5f + 0.5f;
    screenUV.y = 1.0f - screenUV.y;

    float2 px = screenUV * resolution;

    // extra safety against tiny numeric drift
    if (any(px < 0.0f) || any(px > (resolution - 1.0f))) return float2(-1.0f, -1.0f);

    return px;
}

inline int2 GetBestReprojectedPixel_d(
    float3 worldPos,
    float4x4 prevView,
    float4x4 prevProjection,
    float2 resolution,
    uint objID)
{
    float2 subPixel = GetLastFramePixelCoordinates_Float(worldPos, prevView, prevProjection, resolution, objID);
    if (subPixel.x < 0.0f) return int2(-1, -1); // propagated reject

    int2 p = int2(round(subPixel));

    // final guard in integer space (handles 0.5-rounding to 'resolution')
    int2 resi = int2(resolution);
    if (any(p < int2(0,0)) || any(p >= resi)) return int2(-1, -1);

    return p;
}


inline void GetBestReprojectedPixel_d_Advanced(
    float3 worldPos,
    float4x4 prevView,
    float4x4 prevProjection,
    float2 resolution,
    uint objID,
    out int2 outPixels[4],
    out float outDist[4])
{
    float2 subPixelCoord = GetLastFramePixelCoordinates_Float(worldPos, prevView, prevProjection, resolution, objID);
    if (subPixelCoord.x < 0.0f || subPixelCoord.y < 0.0f ||
        subPixelCoord.x >= resolution.x || subPixelCoord.y >= resolution.y)
    {
        [unroll]
        for (int i = 0; i < 4; ++i)
        {
            outPixels[i] = int2(-1, -1);
            outDist[i] = 1e9f;
        }
        return;
    }
    int2 basePixel = int2(round(subPixelCoord));
    float2 pixelCenter = float2(basePixel) + 0.5f;
    float2 offset = subPixelCoord - pixelCenter;
    bool right = offset.x > 0.0f;
    bool top = offset.y > 0.0f;
    outPixels[0] = basePixel;

    if (right && top)
    {
        outPixels[1] = basePixel + int2(1, 0);
        outPixels[2] = basePixel + int2(0, 1);
        outPixels[3] = basePixel + int2(1, 1);
    }
    else if (!right && top)
    {
        outPixels[1] = basePixel + int2(-1, 0);
        outPixels[2] = basePixel + int2(0, 1);
        outPixels[3] = basePixel + int2(-1, 1);
    }
    else if (right && !top)
    {
        outPixels[1] = basePixel + int2(1, 0);
        outPixels[2] = basePixel + int2(0, -1);
        outPixels[3] = basePixel + int2(1, -1);
    }
    else
    {
        outPixels[1] = basePixel + int2(-1, 0);
        outPixels[2] = basePixel + int2(0, -1);
        outPixels[3] = basePixel + int2(-1, -1);
    }

    [unroll]
    for (int i = 0; i < 4; ++i)
    {
        float2 center = float2(outPixels[i]) + 0.5f;
        float2 diff = center - subPixelCoord;
        outDist[i] = dot(diff, diff);
    }
}





