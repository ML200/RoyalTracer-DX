/*
Camera ray operations, optimized
*/

//Initial ray origin
float3 InitOrigin(){
    return mul(viewI, float4(0, 0, 0, 1)).xyz;
}

//Initial ray direction with subpixel jitter
float3 InitDirection(uint2 pixel, uint2 imgSize, inout uint seed)
{
    float2 jitter = float2(RandomFloatSingle(seed), RandomFloatSingle(seed)) - 0.5f;
    float2 pixelSample = float2(pixel) + 0.5f + jitter;
    float2 d = (pixelSample / float2(imgSize)) * 2.0f - 1.0f;

    float4 target = mul(projectionI, float4(d.x, -d.y, 1, 1));
    return normalize(mul(viewI, float4(target.xyz, 0)).xyz);
}


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

    int2 resi = int2(resolution);
    if (any(p < int2(0,0)) || any(p >= resi)) return int2(-1, -1);

    return p;
}




