inline float Luma(float3 c) { return dot(c, float3(0.2126, 0.7152, 0.0722)); }

inline uint MapPixelID(uint2 dims, int2 lIndex)
{
    if (lIndex.x < 0 || lIndex.y < 0 ||
        lIndex.x >= int(dims.x) || lIndex.y >= int(dims.y))
    {
        return 0xFFFFFFFF;
    }
    const uint tileWidth  = 8;
    const uint tileHeight = 4;

    uint2 uIndex   = uint2(lIndex);
    uint tileCountX = (dims.x + tileWidth - 1u) / tileWidth;

    uint tileX = uIndex.x / tileWidth;
    uint tileY = uIndex.y / tileHeight;

    uint localX = uIndex.x % tileWidth;
    uint localY = uIndex.y % tileHeight;

    uint tileIndex  = tileY * tileCountX + tileX;
    uint localIndex = localY * tileWidth + localX;

    return tileIndex * (tileWidth * tileHeight) + localIndex;
}

inline int2 UnmapPixelID(uint pixelID, uint2 dims)
{
    if (pixelID == 0xFFFFFFFF)
    {
        return int2(-1, -1);
    }

    const uint tileWidth  = 8;
    const uint tileHeight = 4;
    const uint tileSize   = tileWidth * tileHeight;

    uint tileIndex  = pixelID / tileSize;
    uint localIndex = pixelID % tileSize;

    uint tileCountX = (dims.x + tileWidth - 1u) / tileWidth;

    uint tileY = tileIndex / tileCountX;
    uint tileX = tileIndex % tileCountX;

    uint localY = localIndex / tileWidth;
    uint localX = localIndex % tileWidth;

    uint globalX = tileX * tileWidth + localX;
    uint globalY = tileY * tileHeight + localY;

    if (globalX >= dims.x || globalY >= dims.y)
    {
        return int2(-1, -1);
    }

    return int2(globalX, globalY);
}

// Ray Tracing Gems, chapter 32.
float3 EnvBRDFApprox2(float3 Kd, float Pr, float Pm, float NoV)
{
    float3 SpecularColor = lerp(0.04.xxx, Kd, saturate(Pm));

    float alpha = Pr * Pr;

    NoV = abs(NoV);

    float4 X;
    X.x = 1.f;
    X.y = NoV;
    X.z = NoV * NoV;
    X.w = NoV * X.z;

    float4 Y;
    Y.x = 1.f;
    Y.y = alpha;
    Y.z = alpha * alpha;
    Y.w = alpha * Y.z;

    float2x2 M1 = float2x2(0.99044f, -1.28514f,
                           1.29678f, -0.755907f);

    float3x3 M2 = float3x3(1.f,     2.92338f,  59.4188f,
                           20.3225f, -27.0302f, 222.592f,
                           121.563f, 626.13f,   316.627f);

    float2x2 M3 = float2x2(0.0365463f,  3.32707f,
                           9.0632f,    -9.04756f);

    float3x3 M4 = float3x3(1.f,      3.59685f, -1.36772f,
                           9.04401f, -16.3174f,  9.22949f,
                           5.56589f,  19.7886f, -20.2123f);

    float bias  = dot(mul(M1, X.xy),  Y.xy)  * rcp(dot(mul(M2, X.xyw), Y.xyw));
    float scale = dot(mul(M3, X.xy),  Y.xy)  * rcp(dot(mul(M4, X.xzw), Y.xyw));

    bias *= saturate(SpecularColor.g * 50);

    return mad(SpecularColor, max(0, scale), max(0, bias));
}


float DLSS_GuideDepthFromWorldPos(float3 worldPos)
{
    float3 camera=mul(viewI,float4(0,0,0,1)).xyz;
    float3 viewPos = mul((float3x3)view, worldPos-camera);
    const float z = max(-viewPos.z, DLSS_GUIDE_DEPTH_NEAR);
    const float n = DLSS_GUIDE_DEPTH_NEAR;
    const float f = DLSS_GUIDE_DEPTH_FAR;
    return saturate(n * (f - z) / ((f - n) * z));
}
