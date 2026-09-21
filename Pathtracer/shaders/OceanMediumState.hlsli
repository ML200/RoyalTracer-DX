#pragma once

bool OceanMediumEnabled() {
    if (!OCEAN_ENABLED || FORCE_DIFFUSE) return false;
    const OceanParamsGPU p = OceanParams();
    return p.debugMode == 0u && !LoadIsThinGlass(p.materialBase) && LoadKd_w(p.materialBase) < 1.0f-EPSILON;
}
void OceanMediumCoefficients(out float3 sigmaA, out float3 sigmaS, out float g) {
    const MatPacked m = g_mat[OceanParams().materialBase];
    sigmaA = max(UnpackRGB9E5(m.Tf_rgb), 0.0f);
    const float strength = (m.texIDs_2 & (1u << 17)) != 0u ? float(m.texIDs_2 >> 24) / 255.0f : 0.0f;
    sigmaS = max(UnpackRGB9E5(m.sss_albedo), 0.0f) * strength /
        max(f16tof32(m.sss_radius_g & 0xffffu), 0.0005f);
    g = clamp(f16tof32(m.sss_radius_g >> 16), -0.95f, 0.95f);
}
// Only initialization and lighting query the height field. Path membership thereafter
// follows actual triangle crossings, including total internal reflection.
float OceanHeight(float2 xz) {
    const OceanParamsGPU p = OceanParams();
    float2 q = xz;
    [unroll] for (uint i=0u; i<4u; ++i) {
        const float3 d = OceanDisplacement(q, 0.0f);
        float2 gradient; float3 stretch;
        OceanDerivatives(q, 0.0f, gradient, stretch);
        const float2 residual = q + d.xz - xz;
        q -= clamp(float2(stretch.y*residual.x-stretch.z*residual.y,
                          stretch.x*residual.y-stretch.z*residual.x) /
                   max(stretch.x*stretch.y-stretch.z*stretch.z, 0.0225f), -8.0f, 8.0f);
    }
    const float2 absoluteQ = q + p.curveOrigin;
    return p.surfaceY + OceanDisplacement(q, 0.0f).y - 0.5f*dot(absoluteQ,absoluteQ)*p.invRadius;
}
bool OceanPointInside(float3 pos) {
    if (!OceanMediumEnabled()) return false;
    const OceanParamsGPU p = OceanParams();
    return all(abs(pos.xz+p.curveOrigin) < p.halfExtent) && pos.y < OceanHeight(pos.xz);
}
float OceanBoundaryDistance(float3 pos, float3 dir, float distanceM) {
    const OceanParamsGPU p = OceanParams();
    const float2 xz = pos.xz+p.curveOrigin;
    if (any(abs(xz) > p.halfExtent)) return 0.0f;
    [unroll] for (uint axis=0u; axis<2u; ++axis) {
        const float d = dir[axis == 0u ? 0u : 2u];
        if (abs(d) > 1e-8f)
            distanceM = min(distanceM, max(((d>0.0f ? p.halfExtent : -p.halfExtent)-xz[axis])/d, 0.0f));
    }
    return distanceM;
}
// Straight shadow connections use a local height-plane intersection. Refracted
// environmental volume lighting explicitly splits its water and air legs instead.
float3 OceanShadowTransmittance(float3 a, float3 b) {
    if (!OceanMediumEnabled()) return 1.0f;
    const float3 span = b-a;
    const float distanceM = length(span);
    if (distanceM < 1e-5f) return 1.0f;
    const float3 dir = span/distanceM;
    const float h = OceanHeight(a.xz);
    float begin = 0.0f, end = distanceM;
    if (a.y >= h) {
        if (dir.y >= -1e-6f) return 1.0f;
        begin = (h-a.y)/dir.y;
    } else if (dir.y > 1e-6f) end = min(end, (h-a.y)/dir.y);
    if (begin >= end) return 1.0f;
    const float waterDistance = OceanBoundaryDistance(a+dir*begin, dir, end-begin);
    float3 sigmaA, sigmaS; float g;
    OceanMediumCoefficients(sigmaA, sigmaS, g);
    return OceanMediumTransmittance(sigmaA+sigmaS, waterDistance);
}
