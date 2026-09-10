#ifndef LIGHT_TREE_SG_HLSLI
#define LIGHT_TREE_SG_HLSLI
// Original implementation of a moment-matched spherical-Gaussian proposal.
// The positive cone mixture preserves support when the fitted lobe is inaccurate.
float LT_SGIntegral(float k) {
    return k < 0.01f ? 4.0f*LT_PI*(1.0f-k+(2.0f/3.0f)*k*k)
        : 2.0f*LT_PI*(1.0f-exp(-2.0f*k))/k;
}
float LT_SGImportance(float3 x, float3 n, float3 mean, float variance,
    float3 emissionAxis, float emissionSharpness, float power)
{
    float3 d=mean-x;
    float d2=dot(d,d);
    float v=max(variance,1e-12f);
    // Bound sharpness to keep the product stable and broaden the near field.
    float k=min(d2/v,1024.0f);
    float3 axis=d2>1e-20f?d*rsqrt(d2):n;
    // Product of the spatial SG, reversed emission SG and a cosine SG.
    // exp(2*(n.w-1)) is a positive fit to the receiver cosine; visibility and
    // the exact BSDF are still evaluated on the sampled triangle.
    float3 e=-emissionAxis;
    float3 p=k*axis+emissionSharpness*e+2.0f*n;
    float kp=length(p);
    float attenuation=exp(min(0.0f,kp-k-emissionSharpness-2.0f));
    float integral=LT_SGIntegral(kp)/LT_SGIntegral(k);
    float emissionNorm=LT_PI/LT_SGIntegral(emissionSharpness);
    return max(0.0f,power/max(d2+v,1e-12f)*emissionNorm*attenuation*integral);
}
float3 LT_LocalReceiverNormal(float4x4 worldToLocal, float3 normal) {
    float3 a=worldToLocal[0].xyz, b=worldToLocal[1].xyz, c=worldToLocal[2].xyz;
    float3 cof0=cross(b,c), cof1=cross(c,a), cof2=cross(a,b);
    float3 pullback=float3(dot(cof0,normal),dot(cof1,normal),dot(cof2,normal));
    return normalize(pullback*(dot(a,cof0)<0.0f?-1.0f:1.0f));
}
#endif
