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

    // Compress and save relevant data: x1, L1, n1, mID and oID
    /*store_x1(payload.hitPosition, g_sample_current, idx);
    store_n1(payload.hitNormal, g_sample_current, idx);
    store_L1(ke, g_sample_current, idx);
    store_matID(payload.materialID, g_sample_current, idx);
    store_objID(payload.objID, g_sample_current, idx);*/

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
//  Reprojects a world‑space point to previous‑frame *UV* coords
//  • returns kInvalidUV on failure
//  • snaps the UV to one of the 4 bilinear neighbours
//---------------------------------------------------------------
inline float2 GetLastFrameUV(
    float3     worldPos,
    float4x4   prevView,
    float4x4   prevProjection,   // already contains last‑frame jitter
    uint       objID,
    float2     resolution,
    inout uint seed)
{
    //-----------------------------------------------------------
    // 1‑5.  Same as before  (world → prev‑local → clip → UV)
    //-----------------------------------------------------------
    float4 localPos  = mul(instanceProps[objID].objectToWorldInverse,
                           float4(worldPos, 1.0f));
    float4 prevWorld = mul(instanceProps[objID].prevObjectToWorld,
                           localPos);

    float4 clipPos   = mul(prevProjection, mul(prevView, prevWorld));

    if (clipPos.w <= 0.0f)
        return kInvalidUV;

    float2 uv = clipPos.xy / clipPos.w * 0.5f + 0.5f;
    uv.y      = 1.0f - uv.y;            // API‑specific Y‑flip
    if (any(uv < 0.0f) || any(uv > 1.0f))
        return kInvalidUV;

    //-----------------------------------------------------------
    // 6.  Bilinear‑probability snap  (no extra half‑pixel!)
    //-----------------------------------------------------------
    float2 p    = uv * resolution;          // pixel‑corner coords
    int2   base = int2(floor(p));           // top‑left integer pixel

    // Clamp so base+1 is still inside the image
    base = clamp(base, int2(0, 0), int2(resolution) - 2);

    float2 f = p - base;                    // fractional part  [0,1)

    float w00 = (1.0f - f.x) * (1.0f - f.y);
    float w10 =  f.x        * (1.0f - f.y);
    float w01 = (1.0f - f.x) *  f.y;
    // w11 = 1 − (w00 + w10 + w01)

    float r = RandomFloatSingle(seed);

    int2 offset;
         if (r < w00)                     offset = int2(0, 0);
    else if (r < w00 + w10)               offset = int2(1, 0);
    else if (r < w00 + w10 + w01)         offset = int2(0, 1);
    else                                  offset = int2(1, 1);

    // 7. Back to 0‑1 space, at the *centre* of the chosen pixel
    return (float2(base + offset)) / resolution;
}

//---------------------------------------------------------------
//  Wrapper that converts the valid UV into an integer pixel
//  • multiplies by *resolution* only once, right at the end
//  • rounds with an explicit +0.5  → avoids the “-1 sentinel → 0” artefact
//---------------------------------------------------------------
inline int2 GetBestReprojectedPixel_d(
    float3     worldPos,
    float4x4   prevView,
    float4x4   prevProjection,
    float2     resolution,
    uint       objID,
    inout uint seed
)
{
    float2 uv = GetLastFrameUV(worldPos, prevView, prevProjection, objID, resolution, seed);

    if (uv.x < 0.0f)            // saw kInvalidUV
        return kInvalidPixel;

    // Convert *once* to pixel space and round to nearest integer
    return int2(uv * resolution + 0.5f);
}



//---------------------------------------------------------------
//  A) Matrix-based previous-world reconstruction
//---------------------------------------------------------------
float3 GetPrevWorldPos_MatrixSpec(
    float3 currWorldPos,
    uint   hitObjID)                     // object that was struck
{
    // Current-world → current-local
    float4 localPos = mul(instanceProps[hitObjID].objectToWorldInverse,
                          float4(currWorldPos, 1.0f));

    // Current-local → *previous* world
    float4 prevW    = mul(instanceProps[hitObjID].prevObjectToWorld, localPos);
    return prevW.xyz;
}

//---------------------------------------------------------------
//  Unified helper that chooses A or B at compile time
//---------------------------------------------------------------
float3 GetPrevWorldPosSpecular(
    float3 currWorldPos, uint hitObjID
)
{
    return GetPrevWorldPos_MatrixSpec(currWorldPos, hitObjID);
}

//---------------------------------------------------------------
//  ①  Reflection hit → previous-frame *UV*     (0-1 space)
//---------------------------------------------------------------
inline float2 GetLastFrameUV_Specular(
    float3   currWorldPos, uint hitObjID,
    float4x4 prevView,
    float4x4 prevProjection)             // already jittered
{
    //-----------------------------------------------------------
    // 1.  Put the hit where it was last frame
    //-----------------------------------------------------------
    float3 prevWorld = GetPrevWorldPosSpecular(
        currWorldPos, hitObjID
    );

    //-----------------------------------------------------------
    // 2.  Previous world → clip
    //-----------------------------------------------------------
    float4 clipPos = mul(prevProjection,
                         mul(prevView, float4(prevWorld, 1.0f)));

    if (clipPos.w <= 0.0f)
        return kInvalidUV;               // behind eye

    //-----------------------------------------------------------
    // 3.  Clip → NDC → UV  (with API-specific Y-flip)
    //-----------------------------------------------------------
    float2 uv = clipPos.xy / clipPos.w * 0.5f + 0.5f;
    uv.y      = 1.0f - uv.y;

    //-----------------------------------------------------------
    // 4.  Screen-edge cull
    //-----------------------------------------------------------
    if (any(uv < 0.0f) || any(uv > 1.0f))
        return kInvalidUV;

    return uv;                           // high-precision 0-1
}

//---------------------------------------------------------------
//  ②  Reflection hit → previous-frame *pixel*
//---------------------------------------------------------------
inline int2 GetBestReprojectedPixel_s(
    float3   currWorldPos, uint   hitObjID,
    float4x4 prevView,
    float4x4 prevProjection,
    float2   resolution)                 // (width, height)
{
    float2 uv = GetLastFrameUV_Specular(
        currWorldPos, hitObjID,
        prevView, prevProjection);

    if (uv.x < 0.0f)                     // kInvalidUV seen
        return kInvalidPixel;

    // single multiply, single round → no “-1 becomes 0” bug
    return int2(uv * resolution + 0.5f);
}