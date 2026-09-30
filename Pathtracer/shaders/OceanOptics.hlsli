#pragma once

// Direct light only (sun fireflies); paths and guides keep the authored roughness.
float OceanHighlightRoughness(float authoredRoughness, float lobeRoughness) {
    return max(authoredRoughness, lobeRoughness);
}
bool OceanDirectLightingOwnsRay(bool directLighting, float outgoingCosine) {
    return directLighting && outgoingCosine > 0.0f;
}

// Lobe proposal probability only, shared by sampling and pdf; 50/50 avoids fireflies.
float OceanReflectionProbability(float reflection, float transmission) {
    if (reflection <= 0.0f) return 0.0f;
    if (transmission <= 0.0f) return 1.0f;
    return 0.5f;
}

float3 OceanMediumTransmittance(float3 sigmaT, float distanceM) {
    return exp(-max(sigmaT, 0.0f) * max(distanceM, 0.0f));
}
// Henyey & Greenstein 1941
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
// RGB mixture free flight; pdf is the mean over channels.
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
    // Kept through internal reflection; an object bounce or entry resets it.
    return used && incomingWater && outgoingWater && (waterSurface || thinTransmission);
}
