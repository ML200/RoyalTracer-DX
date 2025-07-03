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

//---------------------------------------------------------------
//  Common helpers / sentinels
//---------------------------------------------------------------
static const float2  kInvalidUV     = float2(-2.0f, -2.0f);    // <- outside [0,1]² by design
static const int2    kInvalidPixel  = int2(-1, -1);

//---------------------------------------------------------------
//  Reprojects a world-space point to previous-frame *UV* coordinates
//  • stays in 0-1 space  → avoids half-float quantisation at 4K+
//  • returns kInvalidUV  → caller can early-out cheaply
//---------------------------------------------------------------
inline float2 GetLastFrameUV(
    float3     worldPos,
    float4x4   prevView,
    float4x4   prevProjection, // MUST already contain last frame’s jitter
    uint       objID           // per-instance transforms
)
{
    // 1. Current world → previous local → previous world
    float4 localPos   = mul(instanceProps[objID].objectToWorldInverse, float4(worldPos, 1.0f));
    float4 prevWorld  = mul(instanceProps[objID].prevObjectToWorld,  localPos);

    // 2. Previous world → clip
    float4 clipPos    = mul(prevProjection, mul(prevView, prevWorld));

    // 3. Cull points that were behind the camera
    if (clipPos.w <= 0.0f)
        return kInvalidUV;

    // 4. Clip → NDC → UV
    float2 uv = clipPos.xy / clipPos.w * 0.5f + 0.5f;
    uv.y      = 1.0f - uv.y;                   // API-specific Y-flip

    // 5. Off-screen clamp (after the flip!)
    if (any(uv < 0.0f) || any(uv > 1.0f))
        return kInvalidUV;

    return uv;                                 // high-precision 0-1 UV
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
    uint       objID
)
{
    float2 uv = GetLastFrameUV(worldPos, prevView, prevProjection, objID);

    if (uv.x < 0.0f)            // saw kInvalidUV
        return kInvalidPixel;

    // Convert *once* to pixel space and round to nearest integer
    return int2(uv * resolution + 0.5f);
}