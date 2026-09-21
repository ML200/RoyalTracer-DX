#pragma once

// Artistic direct-light filter only. Continuation, environment reflection and DLSS
// guides retain the authored roughness and full-resolution wave normal.
float OceanHighlightRoughness(float authoredRoughness) {
    return max(authoredRoughness, 0.14f);
}
bool OceanDirectLightingOwnsRay(bool directLighting, float outgoingCosine) {
    return directLighting && outgoingCosine > 0.0f;
}

// Proposal probability only. Both GGX sampling and PDF evaluation call this function;
// Fresnel/BSDF energy is unchanged. Preserve absent lobes and total internal reflection.
float OceanReflectionProbability(float reflection, float transmission) {
    if (reflection <= 0.0f) return 0.0f;
    if (transmission <= 0.0f) return 1.0f;
    return max(0.7f, reflection / (reflection + transmission));
}

float3 OceanMediumTransmittance(float3 sigmaT, float distanceM) {
    return exp(-max(sigmaT, 0.0f) * max(distanceM, 0.0f));
}
float OceanPhase(float cosTheta, float g) {
    g = clamp(g, -0.95f, 0.95f);
    const float d = max(1.0f + g*g - 2.0f*g*clamp(cosTheta, -1.0f, 1.0f), 1e-4f);
    return (1.0f-g*g) / (12.566370614359f * d * sqrt(d));
}
// Stable truncated-exponential sampling; uniform limit avoids cancellation in clear water.
float OceanDistanceMass(float rate, float distanceM) {
    float x = rate * distanceM;
    return x < 1e-3f ? x * (1.0f - 0.5f*x + x*x/6.0f) : 1.0f-exp(-x);
}
float OceanSampleDistance(float rate, float distanceM, float u) {
    return rate * distanceM < 1e-3f ? u * distanceM :
        -log(max(1.0f-u*OceanDistanceMass(rate, distanceM), 1e-7f)) / rate;
}
float OceanDistancePdf(float rate, float distanceM, float t) {
    return rate * distanceM < 1e-3f ? rcp(distanceM) :
        rate*exp(-rate*t) / OceanDistanceMass(rate, distanceM);
}
