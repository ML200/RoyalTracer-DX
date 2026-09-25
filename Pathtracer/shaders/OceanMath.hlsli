#pragma once
#include "OceanLayout.h"

// Horizontal displacement gain of a wave of wavenumber k: one, the physical first-order sea, plus
// the short waves' extra that points their crests (ocean::ShortWaveChop). `band` is (log2 k where
// the extra starts, log2 k where it is full, log2 k above which it fades over an octave, extra).
// Must match ocean::ChopGainAt.
float OceanChopGain(float k, float4 band) {
    const float l = log2(max(k, 1e-9f));
    return 1.0f + band.w * smoothstep(band.x, band.y, l) * (1.0f - smoothstep(band.z, band.z + 1.0f, l));
}

// Second-order Stokes crest sharpening.
//
// A linear sea is Gaussian and therefore symmetric: every trough is a mirror image of a crest,
// which is why an FFT ocean reads as rolling rather than as a real sea however much energy it
// carries. Real gravity waves are not - a bound second harmonic rides in phase with the crest,
// pulling it up into a narrow peak and leaving a long shallow trough behind it. For a band of
// elevation eta and variance s2 that harmonic is a*(eta^2 - s2): the offset keeps the mean where
// it was, and a is the band's skew coefficient, equal to its wavenumber for a physical Stokes
// wave and clamped on the host so the warp stays monotonic over the sea it is applied to.
//
// It never moves a sample sideways, so it lifts and narrows a crest without folding the surface;
// the horizontal chop does the rest of the sharpening (ocean::ShortWaveChop).
float OceanSkewTurn(float a) {
    // Below -1/(2a) the parabola turns and would lift the trough back up. Any coefficient the
    // host allows puts that past four standard deviations, so holding the warp flat there costs
    // nothing but keeps it C1.
    return (a > 1e-8f) ? (-0.5f / a) : -3.0e38f;
}

float OceanSkewHeight(float eta, float a, float variance) {
    const float e = max(eta, OceanSkewTurn(a));
    return e + a * (e * e - variance);
}

// d(eta')/d(eta): the factor this band's height gradient picks up. Slopes steepen towards the
// crest and flatten in the troughs, which is also what makes short waves ride up the long crests.
float OceanSkewSlope(float eta, float a) {
    return (eta > OceanSkewTurn(a)) ? (1.0f + 2.0f * a * eta) : 0.0f;
}

// Full inverse transpose of the horizontal displacement map, including its cross derivative.
float2 OceanWarpedSlope(float2 gradient, float3 stretch)
{
    const float determinant = stretch.x * stretch.y - stretch.z * stretch.z;
    const float safeDet = max(determinant, 1e-5f);
    return float2(stretch.y * gradient.x - stretch.z * gradient.y,
                        stretch.x * gradient.y - stretch.z * gradient.x) / safeDet;
}
