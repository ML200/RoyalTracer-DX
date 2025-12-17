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




