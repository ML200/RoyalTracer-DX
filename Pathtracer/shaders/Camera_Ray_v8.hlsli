//====================================================================
//CAMERA RAY GENERATION AND REPROJECTION
//====================================================================

//Initial ray origin
float3 InitOrigin(){
    return mul(viewI, float4(0, 0, 0, 1)).xyz;
}

//Initial ray direction with subpixel jitter
float3 InitDirection(uint2 pixel, uint2 imgSize, inout uint seed)
{
    float2 pixelSample = float2(pixel) + 0.5f + jitter;
    float2 d = (pixelSample / float2(imgSize)) * 2.0f - 1.0f;

    float4 target = mul(projectionI, float4(d.x, -d.y, 1, 1));
    return normalize(mul(viewI, float4(target.xyz, 0)).xyz);
}

//====================================================================
//PREVIOUS-FRAME REPROJECTION
//====================================================================
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

    if (clipPos.w <= 0.0f || !isfinite(clipPos.w)) return float2(-1.0f, -1.0f);

    float2 ndc = clipPos.xy / clipPos.w;

    if (any(ndc < -1.0f) || any(ndc > 1.0f)) return float2(-1.0f, -1.0f);

    float2 uv = ndc * 0.5f + 0.5f;
    uv.y = 1.0f - uv.y;

    //pixel-center coordinates
    float2 px = uv * resolution - 0.5f;

    //allow half-pixel margin
    if (any(px < -0.5f) || any(px > (resolution - 0.5f))) return float2(-1.0f, -1.0f);

    return px;
}

//Unclamped variant for motion vectors, allows off-screen previous positions.
//Only rejects behind-camera where projection is undefined.
inline float2 GetLastFramePixelCoordinates_Unclamped(
    float3 worldPos,
    float4x4 prevView,
    float4x4 prevProjection,
    float2 resolution,
    uint objID)
{
    float4 localPos     = mul(instanceProps[objID].objectToWorldInverse, float4(worldPos, 1.0f));
    float4 prevWorldPos = mul(instanceProps[objID].prevObjectToWorld, localPos);
    float4 clipPos      = mul(prevProjection, mul(prevView, prevWorldPos));

    if (clipPos.w <= 0.0f || !isfinite(clipPos.w)) return float2(-1e9f, -1e9f);

    float2 ndc = clipPos.xy / clipPos.w;
    float2 uv  = ndc * 0.5f + 0.5f;
    uv.y = 1.0f - uv.y;

    return uv * resolution - 0.5f;
}

inline int2 GetBestReprojectedPixel_d(
    float3 worldPos,
    float4x4 prevView,
    float4x4 prevProjection,
    float2 resolution,
    uint objID)
{
    float2 px = GetLastFramePixelCoordinates_Float(worldPos, prevView, prevProjection, resolution, objID);
    if (px.x < 0.0f) return int2(-1, -1);

    int2 p = int2(floor(px + 0.5f));

    int2 resi = int2(resolution);
    if (any(p < 0) || any(p >= resi)) return int2(-1, -1);

    return p;
}

