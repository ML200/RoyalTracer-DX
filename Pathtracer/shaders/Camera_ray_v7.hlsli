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

//---------------------------------------------------------------
//  Reprojects a world‑space point to previous‑frame *UV* coords
//  • returns kInvalidUV on failure
//  • snaps the UV to one of the 4 bilinear neighbours
//  • probability weights are (wᵢ)ᵞ and renormalised
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
    // 6.  Bilinear‑probability snap with exponent γ
    //-----------------------------------------------------------
    float2 p    = uv * resolution;          // pixel‑corner coords
    int2   base = int2(floor(p));           // top‑left integer pixel
    base = clamp(base, int2(0, 0), int2(resolution) - 2);

    float2 f = p - base;                    // fractional part  [0,1)

    // Classical bilinear weights (still in [0,1])
    float w00 = (1.0f - f.x) * (1.0f - f.y);
    float w10 =  f.x        * (1.0f - f.y);
    float w01 = (1.0f - f.x) *  f.y;
    float w11 = 1.0f - (w00 + w10 + w01);   // 4th corner

    // Apply exponent γ and renormalise
    float g   = kReprojectionProbExponent;
    float p00 = pow(w00, g);
    float p10 = pow(w10, g);
    float p01 = pow(w01, g);
    float p11 = pow(w11, g);

    float norm = 1.0f / max(p00 + p10 + p01 + p11, 1e-6f); // guard against div‑by‑zero
    p00 *= norm;
    p10 *= norm;
    p01 *= norm;
    p11 = 1.0f - (p00 + p10 + p01);          // exact closure

    // Resolve the random pick
    float r = RandomFloatSingle(seed);
    int2 offset;
         if (r < p00)                     offset = int2(0, 0);
    else if (r < p00 + p10)               offset = int2(1, 0);
    else if (r < p00 + p10 + p01)         offset = int2(0, 1);
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


//--------------------------------------------------------------------
//  Reproject the current sample into last-frame pixel space and
//  return the four neighbours needed for a bilinear history fetch.
//  • NO call to GetLastFrameUV – everything is done locally
//  • Multiplication by *resolution* happens once, with an explicit +0.5
//  • Each neighbour is range-checked independently
//--------------------------------------------------------------------
inline bool GetLastFramePixels4(
    float3              worldPos,
    float4x4            prevView,
    float4x4            prevProjection,
    uint                objID,
    float2              resolution,       // (width, height)
    out int2            outPixels[4])     // ← result
{
    //--------------------------------------------------------------- 1-5
    float4 localPos  = mul(instanceProps[objID].objectToWorldInverse,
                           float4(worldPos, 1.0f));
    float4 prevWorld = mul(instanceProps[objID].prevObjectToWorld, localPos);
    float4 clipPos   = mul(prevProjection, mul(prevView, prevWorld));

    // Behind the camera → hopeless, abort early
    if (clipPos.w <= 0.0f)
    {
        [unroll] for (int i = 0; i < 4; ++i)
            outPixels[i] = kInvalidPixel;
        return false;
    }

    //--------------------------------------------------------------- 6
    float2 uv = clipPos.xy / clipPos.w * 0.5f + 0.5f;
    uv.y      = 1.0f - uv.y;            // API-specific Y-flip

    // Convert **once** to pixel space and round to nearest integer.
    // (+0.5 avoids the  -1  sentinel → 0  artefact.)
    int2 centrePix = int2(uv * resolution + 0.5f);

    //--------------------------------------------------------------- 7
    static const int2 OFFS[4] = {
        int2(-1, -1),   // top-left
        int2( 0, -1),   // top-right
        int2(-1,  0),   // bottom-left
        int2( 0,  0)    // bottom-right
    };

    bool anyValid = false;

    [unroll]
    for (int i = 0; i < 4; ++i)
    {
        int2 pix = centrePix + OFFS[i];

        bool inside =
             (pix.x >= 0) && (pix.y >= 0) &&
             (pix.x <  (int)resolution.x) &&
             (pix.y <  (int)resolution.y);

        outPixels[i] = inside ? pix : kInvalidPixel;
        anyValid    |= inside;
    }

    return anyValid;      // “true”  ⇢ at least one neighbour usable
}