//====================================
//CAMERA RAY GENERATION
//====================================

//thin-lens DoF parameters live in CameraParams (dofApertureRadius, dofFocusDistance)
//aperture radius is in world units, larger = shallower depth of field
//focus distance is along view-space forward axis, surfaces at this depth stay sharp

float3 InitOrigin(){
    return mul(viewI, float4(0, 0, 0, 1)).xyz;
}

//direction with subpixel jitter (pinhole)
float3 InitDirection(uint2 pixel, uint2 imgSize, inout uint seed)
{
    float2 pixelSample = float2(pixel) + 0.5f + jitter;
    float2 d = (pixelSample / float2(imgSize)) * 2.0f - 1.0f;

    float4 target = mul(projectionI, float4(d.x, -d.y, 1, 1));
    return normalize(mul(viewI, float4(target.xyz, 0)).xyz);
}

//world position recovered from primary ray hit distance
//uses centered pinhole ray (no jitter, no DoF) so reconstruction is purely a
//function of pixel index + stored hitT. Error vs the actual jittered/DoF ray
//is at most subpixel offset projected onto the surface, well below the spatial
//reuse rejection threshold (which scales with camera distance).
float3 ReconstructPositionFromHitT(int2 pixel, float hitT)
{
    float2 pixelSample = float2(pixel) + 0.5f;
    float2 d = (pixelSample / float2(IMG_W, IMG_H)) * 2.0f - 1.0f;
    float4 target = mul(projectionI, float4(d.x, -d.y, 1, 1));
    float3 rayDir = normalize(mul(viewI, float4(target.xyz, 0)).xyz);
    return InitOrigin() + rayDir * hitT;
}

//concentric disk mapping, uniform on unit disk
float2 SampleUnitDisk(inout uint seed)
{
    float u1 = RandomFloatSingle(seed) * 2.0f - 1.0f;
    float u2 = RandomFloatSingle(seed) * 2.0f - 1.0f;
    if (u1 == 0.0f && u2 == 0.0f) return float2(0.0f, 0.0f);

    float r, theta;
    if (abs(u1) > abs(u2)) {
        r     = u1;
        theta = (PI * 0.25f) * (u2 / u1);
    } else {
        r     = u2;
        theta = (PI * 0.5f) - (PI * 0.25f) * (u1 / u2);
    }
    return float2(r * cos(theta), r * sin(theta));
}

//thin-lens primary ray, lens lives on view-space z=0 plane, focal plane at z=dofFocusDistance
//perturbs the origin uniformly inside the aperture disk and aims at the focal point that the
//pinhole ray would have hit, so the focus plane stays sharp while everything else blurs
void InitCameraRayDoF(uint2 pixel, uint2 imgSize, inout uint seed,
                      out float3 rayOrigin, out float3 rayDir)
{
    float2 pixelSample = float2(pixel) + 0.5f + jitter;
    float2 d           = (pixelSample / float2(imgSize)) * 2.0f - 1.0f;

    //view-space far-plane point along the pinhole ray, homogeneous form is fine here since
    //the focus-plane scale below cancels any leftover w divide
    //abs(viewH.z) keeps the focal point on the camera-forward side for both LH and RH
    //projections, this engine is RH (XMMatrixPerspectiveFovRH) so viewH.z is negative
    float4 viewH       = mul(projectionI, float4(d.x, -d.y, 1, 1));
    float3 viewFocus   = viewH.xyz * (dofFocusDistance / abs(viewH.z));

    float2 lensXY      = SampleUnitDisk(seed) * dofApertureRadius;
    float3 viewLensPos = float3(lensXY, 0.0f);
    float3 viewRayDir  = normalize(viewFocus - viewLensPos);

    rayOrigin = mul(viewI, float4(viewLensPos, 1)).xyz;
    rayDir    = normalize(mul(viewI, float4(viewRayDir, 0)).xyz);
}

//====================================
//PREVIOUS-FRAME REPROJECTION
//====================================
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

    float2 px = uv * resolution - 0.5f;

    //half-pixel margin
    if (any(px < -0.5f) || any(px > (resolution - 0.5f))) return float2(-1.0f, -1.0f);

    return px;
}

//current-frame pinhole projection, mirrors GetLastFramePixelCoordinates_Unclamped
//for DoF the lens-jittered curPix no longer equals the pinhole projection of the hit point,
//MV must use the pinhole projection on both ends so DLSS sees pure scene motion
//round trip through localPos matches the prev path so stationary objects get bit identical
//currWorldPos and prevWorldPos, otherwise FP32 matmul residual leaks into the MV
//view stage applies the rotation to (currWorldPos - camPos), avoiding the big-minus-big
//cancellation that mul(view, worldPos) hits at large world coords; this kills the residual
//drift visible on grazing angle surfaces
inline float2 GetCurrentFramePixelCoordinates_Unclamped(
    float3 worldPos,
    float4x4 V,
    float4x4 P,
    float2 resolution,
    uint objID)
{
    float4 localPos     = mul(instanceProps[objID].objectToWorldInverse, float4(worldPos, 1.0f));
    float4 currWorldPos = mul(instanceProps[objID].objectToWorld,        localPos);
    float3 viewPos      = mul((float3x3)V, currWorldPos.xyz - InitOrigin());
    float4 clipPos      = mul(P, float4(viewPos, 1.0f));
    if (clipPos.w <= 0.0f || !isfinite(clipPos.w)) return float2(-1e9f, -1e9f);
    float2 ndc = clipPos.xy / clipPos.w;
    float2 uv  = ndc * 0.5f + 0.5f;
    uv.y = 1.0f - uv.y;
    return uv * resolution - 0.5f;
}

//unclamped variant for MV, allows off-screen previous pos, only rejects behind-camera
//same camera relative view step as the current frame counterpart, prev camera position
//extracted from prevView since there is no prevViewI in the cbuffer
inline float2 GetLastFramePixelCoordinates_Unclamped(
    float3 worldPos,
    float4x4 prevView,
    float4x4 prevProjection,
    float2 resolution,
    uint objID)
{
    float4 localPos     = mul(instanceProps[objID].objectToWorldInverse, float4(worldPos, 1.0f));
    float4 prevWorldPos = mul(instanceProps[objID].prevObjectToWorld, localPos);
    //prevCamPos = -R^T * t where R^T is the upper 3x3 of prevView and t is its translation column
    float3 prevTransCol = mul(prevView, float4(0, 0, 0, 1)).xyz;
    float3 prevCamPos   = -mul(transpose((float3x3)prevView), prevTransCol);
    float3 viewPos      = mul((float3x3)prevView, prevWorldPos.xyz - prevCamPos);
    float4 clipPos      = mul(prevProjection, float4(viewPos, 1.0f));

    if (clipPos.w <= 0.0f || !isfinite(clipPos.w)) return float2(-1e9f, -1e9f);

    float2 ndc = clipPos.xy / clipPos.w;
    float2 uv  = ndc * 0.5f + 0.5f;
    uv.y = 1.0f - uv.y;

    return uv * resolution - 0.5f;
}

//====================================
//STATIC WORLD REPROJECTION
//====================================
//Variants of the above for points that have NO instance — e.g., volumetric
//cloud sample positions. Skips the instanceProps[objID] lookup and treats
//the position as world-static (no animation). Same camera-relative
//view-space step as the instance versions to avoid the big-minus-big
//cancellation at large world coords.
inline float2 GetLastFramePixelCoordinates_World(
    float3 worldPos,
    float4x4 prevView,
    float4x4 prevProjection,
    float2 resolution)
{
    float3 prevTransCol = mul(prevView, float4(0, 0, 0, 1)).xyz;
    float3 prevCamPos   = -mul(transpose((float3x3)prevView), prevTransCol);
    float3 viewPos      = mul((float3x3)prevView, worldPos - prevCamPos);
    float4 clipPos      = mul(prevProjection, float4(viewPos, 1.0f));

    if (clipPos.w <= 0.0f || !isfinite(clipPos.w)) return float2(-1e9f, -1e9f);

    float2 ndc = clipPos.xy / clipPos.w;
    float2 uv  = ndc * 0.5f + 0.5f;
    uv.y = 1.0f - uv.y;
    return uv * resolution - 0.5f;
}

inline float2 GetCurrentFramePixelCoordinates_World(
    float3 worldPos,
    float4x4 V,
    float4x4 P,
    float2 resolution)
{
    float3 viewPos = mul((float3x3)V, worldPos - InitOrigin());
    float4 clipPos = mul(P, float4(viewPos, 1.0f));
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

//PLANET: world-static variant of GetBestReprojectedPixel_d for points with no
//instance (terrain) - skips the instanceProps[objID] lookup. Terrain is static,
//so its reprojection is pure camera motion.
inline int2 GetBestReprojectedPixel_World(
    float3 worldPos,
    float4x4 prevView,
    float4x4 prevProjection,
    float2 resolution)
{
    float2 px = GetLastFramePixelCoordinates_World(worldPos, prevView, prevProjection, resolution);
    if (px.x < -1e8f) return int2(-1, -1);

    int2 p = int2(floor(px + 0.5f));

    int2 resi = int2(resolution);
    if (any(p < 0) || any(p >= resi)) return int2(-1, -1);

    return p;
}
