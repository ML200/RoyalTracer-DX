float3 InitOrigin(){
    return mul(viewI, float4(0, 0, 0, 1)).xyz;
}

// Concentric mapping (Shirley & Chiu 1997).
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

// Thin lens; the focus plane stays sharp.
void InitCameraRayDoF(uint2 pixel, uint2 imgSize, inout uint seed,
                      out float3 rayOrigin, out float3 rayDir)
{
    float2 pixelSample = float2(pixel) + 0.5f + jitter;
    float2 d           = (pixelSample / float2(imgSize)) * 2.0f - 1.0f;

    float4 viewH       = mul(projectionI, float4(d.x, -d.y, 1, 1));
    float3 viewFocus   = viewH.xyz * (dofFocusDistance / abs(viewH.z));

    float2 lensXY      = SampleUnitDisk(seed) * dofApertureRadius;
    float3 viewLensPos = float3(lensXY, 0.0f);
    float3 viewRayDir  = normalize(viewFocus - viewLensPos);

    rayOrigin = mul(viewI, float4(viewLensPos, 1)).xyz;
    rayDir    = normalize(mul(viewI, float4(viewRayDir, 0)).xyz);
}

inline float2 GetCurrentFramePixelCoordinates_Unclamped(
    float3 worldPos,
    float4x4 V,
    float4x4 P,
    float2 resolution,
    uint objID)
{
    float3 localPos     = mul(instanceProps[objID].objectToWorldInverse, float4(worldPos, 1.0f));
    float3 currWorldPos = mul(instanceProps[objID].objectToWorld,        float4(localPos, 1.0f));
    float3 viewPos      = mul((float3x3)V, currWorldPos - InitOrigin());
    float4 clipPos      = mul(P, float4(viewPos, 1.0f));
    if (clipPos.w <= 0.0f || !isfinite(clipPos.w)) return float2(-1e9f, -1e9f);
    float2 ndc = clipPos.xy / clipPos.w;
    float2 uv  = ndc * 0.5f + 0.5f;
    uv.y = 1.0f - uv.y;
    return uv * resolution - 0.5f;
}

inline float2 GetLastFramePixelCoordinates_Unclamped(
    float3 worldPos,
    float4x4 prevView,
    float4x4 prevProjection,
    float2 resolution,
    uint objID)
{
    float3 localPos     = mul(instanceProps[objID].objectToWorldInverse, float4(worldPos, 1.0f));
    float3 prevWorldPos = mul(instanceProps[objID].prevObjectToWorld, float4(localPos, 1.0f));

    float3 prevTransCol = mul(prevView, float4(0, 0, 0, 1)).xyz;
    float3 prevCamPos   = -mul(transpose((float3x3)prevView), prevTransCol);
    float3 viewPos      = mul((float3x3)prevView, prevWorldPos - prevCamPos);
    float4 clipPos      = mul(prevProjection, float4(viewPos, 1.0f));

    if (clipPos.w <= 0.0f || !isfinite(clipPos.w)) return float2(-1e9f, -1e9f);

    float2 ndc = clipPos.xy / clipPos.w;
    float2 uv  = ndc * 0.5f + 0.5f;
    uv.y = 1.0f - uv.y;

    return uv * resolution - 0.5f;
}

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

