#pragma once
// Homogeneous single scattering, with RGB distance importance sampling. Geometry
// supplies segment length; there is no diffuse surface return or arbitrary body depth.
// Lighting connections refract at the local horizontal interface (no focused caustics).
float3 OceanVolumeEnvironment(float3 pos, float3 rayDir, float3 airDir, float3 radiance,
    float3 sigmaT, float g) {
    if (airDir.y <= 0.0f || !any(radiance > 0.0f)) return 0.0f;
    const float ni = max(LoadNi(OceanParams().materialBase), 1.0001f);
    const float3 up = float3(0,1,0);
    const float3 waterDir = -refract(-airDir, up, 1.0f/ni);
    float distanceToSurface = max(OceanHeight(pos.xz)-pos.y, 0.0f) / max(waterDir.y, 1e-4f);
    [unroll] for (uint i=0u; i<2u; ++i)
        distanceToSurface = max(OceanHeight((pos+waterDir*distanceToSurface).xz)-pos.y, 0.0f) / max(waterDir.y,1e-4f);
    const float3 entry = pos + waterDir*distanceToSurface;
    if (any(abs(entry.xz+OceanParams().curveOrigin) > OceanParams().halfExtent)) return 0.0f;
    // No water-interface Fresnel in the visibility helper: it is applied exactly once below.
    const float3 visibility = VisibilityTransmittance(pos, waterDir, entry, -waterDir, true) *
        VisibilityTransmittance(entry+up*0.01f, airDir, entry+airDir*RAY_TMAX_PLANET, -airDir, true);
    const float F = FresnelDielectricTIR(airDir, up, 1.0f, ni).x;
    // eta^2 radiance conversion cancels the refracted solid-angle Jacobian's eta^-2.
    const float entryWeight = (1.0f-F) * airDir.y / max(waterDir.y,1e-4f);
    return radiance * visibility * OceanMediumTransmittance(sigmaT, distanceToSurface) *
        (entryWeight * OceanPhase(dot(rayDir,waterDir), g));
}
void OceanIntegrateVolume(float3 origin, float3 dir, float distanceM, inout uint seed,
    out float3 transmittance, out float3 radiance) {
    transmittance = 1.0f; radiance = 0.0f;
    if (!OceanMediumEnabled()) return;
    distanceM = OceanBoundaryDistance(origin, dir, distanceM);
    if (distanceM <= 1e-5f) return;
    float3 sigmaA, sigmaS; float g;
    OceanMediumCoefficients(sigmaA, sigmaS, g);
    const float3 sigmaT = sigmaA+sigmaS;
    transmittance = OceanMediumTransmittance(sigmaT, distanceM);
    if (!any(sigmaS > 0.0f)) return;
    // Truncate only the scattering integral at 16 optical depths in the clearest
    // channel. Background attenuation above always uses the full traced distance.
    const float integralLength = min(distanceM, 16.0f/max(min(sigmaT.x,min(sigmaT.y,sigmaT.z)),1e-6f));
    [loop] for (uint i=0u; i<2u; ++i) {
        const uint channel = min((uint)(RandomFloatSingle(seed)*3.0f), 2u);
        const float t = OceanSampleDistance(sigmaT[channel], integralLength, (float(i)+RandomFloatSingle(seed))*0.5f);
        const float pdf = (OceanDistancePdf(sigmaT.x,integralLength,t) +
            OceanDistancePdf(sigmaT.y,integralLength,t) + OceanDistancePdf(sigmaT.z,integralLength,t))/3.0f;
        const float3 pos = origin+dir*t;
        const float3 surface = float3(pos.x, OceanHeight(pos.xz)+0.02f, pos.z);
        SetSkyObserver(surface+sceneOriginWorld);
        const SunSampleResult sun = SampleSun(float2(RandomFloatSingle(seed),RandomFloatSingle(seed)), surface+sceneOriginWorld);
        float3 source = sun.pdf > 1e-20f ? OceanVolumeEnvironment(pos,dir,sun.direction,sun.radiance/sun.pdf,sigmaT,g) : 0.0f;
        const float y = RandomFloatSingle(seed), phi = 6.28318530718f*RandomFloatSingle(seed);
        const float r = sqrt(max(1.0f-y*y,0.0f));
        const float3 skyDir = float3(r*cos(phi), y, r*sin(phi));
        source += OceanVolumeEnvironment(pos,dir,skyDir,EvaluateSky(skyDir)*6.28318530718f,sigmaT,g);
        radiance += OceanMediumTransmittance(sigmaT,t)*sigmaS*source / max(2.0f*pdf,1e-20f);
    }
    SetSkyObserver(InitOrigin()+sceneOriginWorld);
}
