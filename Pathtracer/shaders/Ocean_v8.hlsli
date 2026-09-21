#pragma once
#include "OceanLayout.h"
#include "OceanMath.hlsli"

// Shared material-coordinate surface evaluation. Raw derivatives are summed before inversion.
// Geometry is filtered to its mesh spacing. Reflection/refraction shading uses full-resolution
// wave normals and authored roughness; only direct-light evaluation widens its highlight lobe.
OceanParamsGPU OceanParams() {
    StructuredBuffer<OceanParamsGPU> b = ResourceDescriptorHeap[OCEAN_SRV_PARAMS];
    return b[0];
}
float2 OceanCascadeUV(float2 q, OceanParamsGPU P, uint c) {
    // FFT node i represents i*L/N; texture texel i is centred at (i+0.5)/N.
    return (q + float2(P.originWrapX[c], P.originWrapZ[c])) / P.cascadeLength[c] + 0.5f / OCEAN_FFT_SIZE;
}
float OceanCascadeMip(float widthM, float L) {
    return clamp(log2(max(widthM * OCEAN_FFT_SIZE / L, 1.0f)), 0.0f, OCEAN_MIP_LEVELS - 1.0f);
}
float3 OceanDisplacementFrom(float2 q, float widthM, uint descriptor) {
    const OceanParamsGPU P = OceanParams();
    Texture2DArray<float4> disp = ResourceDescriptorHeap[descriptor];
    float3 d = 0.0f;
    [loop] for (uint c = 0; c < OCEAN_CASCADES; ++c)
        d += disp.SampleLevel(g_sampler, float3(OceanCascadeUV(q, P, c), c),
                              OceanCascadeMip(widthM, P.cascadeLength[c])).xyz;
    return d * P.waveHeightScale;
}
float3 OceanDisplacement(float2 q, float widthM) {
    return OceanDisplacementFrom(q, widthM, OCEAN_SRV_DISP);
}
void OceanDerivatives(float2 q, float widthM, out float2 gradient, out float3 stretch) {
    const OceanParamsGPU P = OceanParams();
    Texture2DArray<float4> deriv = ResourceDescriptorHeap[OCEAN_SRV_DERIV];
    Texture2DArray<float4> disp = ResourceDescriptorHeap[OCEAN_SRV_DISP];
    gradient = 0.0f;
    stretch = float3(1, 1, 0);
    [loop] for (uint c = 0; c < OCEAN_CASCADES; ++c) {
        const float3 uv = float3(OceanCascadeUV(q, P, c), c);
        const float mip = OceanCascadeMip(widthM, P.cascadeLength[c]);
        const float4 d = deriv.SampleLevel(g_sampler, uv, mip) * P.waveHeightScale;
        gradient += d.xy;
        stretch += float3(d.zw, disp.SampleLevel(g_sampler, uv, mip).w * P.waveHeightScale);
    }
}
float3 OceanNormal(float2 q, float widthM, float2 curveGradient) {
    float2 gradient; float3 stretch;
    OceanDerivatives(q, widthM, gradient, stretch);
    const float2 slope = OceanWarpedSlope(gradient - curveGradient, stretch);
    return normalize(float3(-slope.x, 1, -slope.y));
}
struct OceanSurface {
    float3 normal;
    float3 albedo;
    float roughness;
    uint materialOffset;
};
OceanSurface OceanEvalSurface(float2 q, float widthM) {
    const OceanParamsGPU P = OceanParams();
    float2 gradient; float3 A;
    OceanDerivatives(q, 0.0f, gradient, A);
    const float2 slope = OceanWarpedSlope(gradient - (q + P.curveOrigin) * P.invRadius, A);
    OceanSurface s;
    s.normal = normalize(float3(-slope.x, 1, -slope.y));
    s.roughness = 0.0f;
    s.albedo = 1.0f.xxx;
    s.materialOffset = 0u;

    // Opt-in diagnostics retain their opaque material; they never feed beauty shading.
    if (P.debugMode != 0u) {
        const float det = A.x * A.y - A.z * A.z;
        const float minStretch = 0.5f * (A.x + A.y - length(float2(A.x - A.y, 2 * A.z)));
        const float filterW = max(widthM * P.filterScale, 1e-4f) / max(minStretch, 0.15f);
        if (P.debugMode == 1u) s.albedo = s.normal * 0.5f + 0.5f;
        if (P.debugMode == 2u) s.albedo = saturate(float3(1.0f-det, minStretch, det));
        if (P.debugMode == 3u) {
            // Raw height-gradient covariance at the footprint, for inspection only.
            Texture2DArray<float4> deriv = ResourceDescriptorHeap[OCEAN_SRV_DERIV];
            Texture2DArray<float4> moments = ResourceDescriptorHeap[OCEAN_SRV_MOMENTS];
            float3 covariance = 0.0f;
            [loop] for (uint c = 0; c < OCEAN_CASCADES; ++c) {
                const float3 uv = float3(OceanCascadeUV(q, P, c), c);
                const float mip = OceanCascadeMip(filterW, P.cascadeLength[c]);
                const float2 mean = deriv.SampleLevel(g_sampler, uv, mip).xy;
                const float3 raw = moments.SampleLevel(g_sampler, uv, mip).xyz;
                covariance += raw - float3(mean.x * mean.x, mean.x * mean.y, mean.y * mean.y);
            }
            covariance *= P.waveHeightScale * P.waveHeightScale;
            s.albedo = saturate(float3(covariance.x, covariance.z, abs(covariance.y)) * 20.0f);
        }
        if (P.debugMode == 4u) s.albedo = 0.0f; // no footprint roughness on continuation
        if (P.debugMode == 5u) s.albedo = 0.0f; // removed foam
        if (P.debugMode == 6u) s.albedo = saturate(float3(OceanCascadeMip(filterW,P.cascadeLength.x),
            OceanCascadeMip(filterW,P.cascadeLength.y),OceanCascadeMip(filterW,P.cascadeLength.z)) / (OCEAN_MIP_LEVELS-1));
        s.materialOffset = OCEAN_MATERIAL_LEVELS-1;
        s.roughness = 1.0f;
    }
    return s;
}
