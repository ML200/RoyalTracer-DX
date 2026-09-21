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
struct OceanFlight {
    float distance;
    float3 weight;
    bool scattered;
};
// RGB mixture free flight: event density is mean(sigma_t * Tr), and the
// discrete probability of reaching geometry is mean(Tr). Absorption is in
// sigma_t; sigma_s in the numerator accounts for absorption versus scattering.
OceanFlight OceanFreeFlight(float3 sigmaA, float3 sigmaS, float limit, bool used, float2 u) {
    OceanFlight f;
    f.distance = max(limit, 0.0f);
    f.scattered = false;
    const float3 sigmaT = max(sigmaA,0.0f) + max(sigmaS,0.0f);
    if (used || !any(sigmaS > 0.0f) || limit <= 0.0f) {
        f.weight = OceanMediumTransmittance(sigmaT,f.distance);
        return f;
    }
    const uint channel = min((uint)(saturate(u.x)*3.0f),2u);
    const float rate = sigmaT[channel];
    const float sampledDistance = rate > 0.0f ? -log(max(1.0f-u.y,1e-7f))/rate : limit;
    f.scattered = sampledDistance < limit;
    f.distance = min(sampledDistance,limit);
    const float3 tr = OceanMediumTransmittance(sigmaT,f.distance);
    const float3 density = f.scattered ? sigmaT*tr : tr;
    const float pdf = (density.x+density.y+density.z)/3.0f;
    f.weight = pdf > 0.0f ? tr*(f.scattered ? sigmaS : 1.0f.xxx)/pdf : 0.0f;
    return f;
}
float OceanSamplePhaseCosine(float g, float u) {
    g = clamp(g,-0.95f,0.95f);
    if (abs(g) < 1e-3f) return 1.0f-2.0f*u;
    const float ratio = (1.0f-g*g)/(1.0f-g+2.0f*g*u);
    return clamp((1.0f+g*g-ratio*ratio)/(2.0f*g),-1.0f,1.0f);
}
bool OceanScatterUsedAfterSurface(bool used, bool incomingWater, bool outgoingWater,
    bool waterSurface, bool thinTransmission) {
    // Internal reflection retains the budget. An object bounce or fresh entry resets it.
    return used && incomingWater && outgoingWater && (waterSurface || thinTransmission);
}
