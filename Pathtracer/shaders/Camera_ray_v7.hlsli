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

//---------------------------------------------------------------
//  Common helpers / sentinels
//---------------------------------------------------------------
static const float2  kInvalidUV     = float2(-2.0f, -2.0f);    // <- outside [0,1]² by design
static const int2    kInvalidPixel  = int2(-1, -1);

//---------------------------------------------------------------
//  Global or CB‑level tweakable
//---------------------------------------------------------------
static const float kReprojectionProbExponent = 1.0f;   // γ  (1 = current behaviour)

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

    // Early out if invalid reprojection
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

    // Rounded integer pixel (nearest)
    int2 basePixel = int2(round(subPixelCoord));
    float2 pixelCenter = float2(basePixel) + 0.5f;

    // Compute the subpixel offset from pixel center
    float2 offset = subPixelCoord - pixelCenter;

    // Decide the quadrant of the subpixel
    bool right = offset.x > 0.0f;
    bool top = offset.y > 0.0f;

    // Determine pixel patch based on quadrant
    // Always include base pixel
    outPixels[0] = basePixel;

    if (right && top)
    {
        // Top-right quadrant → include base, right, top, top-right
        outPixels[1] = basePixel + int2(1, 0);  // right
        outPixels[2] = basePixel + int2(0, 1);  // top
        outPixels[3] = basePixel + int2(1, 1);  // top-right
    }
    else if (!right && top)
    {
        // Top-left quadrant → include base, left, top, top-left
        outPixels[1] = basePixel + int2(-1, 0); // left
        outPixels[2] = basePixel + int2(0, 1);  // top
        outPixels[3] = basePixel + int2(-1, 1); // top-left
    }
    else if (right && !top)
    {
        // Bottom-right quadrant → include base, right, bottom, bottom-right
        outPixels[1] = basePixel + int2(1, 0);  // right
        outPixels[2] = basePixel + int2(0, -1); // bottom
        outPixels[3] = basePixel + int2(1, -1); // bottom-right
    }
    else
    {
        // Bottom-left quadrant → include base, left, bottom, bottom-left
        outPixels[1] = basePixel + int2(-1, 0); // left
        outPixels[2] = basePixel + int2(0, -1); // bottom
        outPixels[3] = basePixel + int2(-1, -1); // bottom-left
    }

    // Compute squared distances to pixel centers (to avoid sqrt unless you really need it)
    [unroll]
    for (int i = 0; i < 4; ++i)
    {
        float2 center = float2(outPixels[i]) + 0.5f;
        float2 diff = center - subPixelCoord;
        outDist[i] = dot(diff, diff); // squared distance
    }
}





