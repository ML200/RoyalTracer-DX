#pragma once
#include "OceanLayout.h"

// Chop gain at wavenumber k; band = log2 k (start, full, fade), extra. Must match ocean::ChopGainAt.
float OceanChopGain(float k, float4 band) {
    const float l = log2(max(k, 1e-9f));
    return 1.0f + band.w * smoothstep(band.x, band.y, l) * (1.0f - smoothstep(band.z, band.z + 1.0f, l));
}

// Second-order crest sharpening, eta + a (eta^2 - var) (Tayfun 1980, narrow-band).
float OceanSkewTurn(float a) {
    // Flat below the parabola's minimum -1/(2a).
    return (a > 1e-8f) ? (-0.5f / a) : -3.0e38f;
}

float OceanSkewHeight(float eta, float a, float variance) {
    const float e = max(eta, OceanSkewTurn(a));
    return e + a * (e * e - variance);
}

// d(eta')/d(eta)
float OceanSkewSlope(float eta, float a) {
    return (eta > OceanSkewTurn(a)) ? (1.0f + 2.0f * a * eta) : 0.0f;
}

// Inverse transpose of the horizontal displacement Jacobian.
float2 OceanWarpedSlope(float2 gradient, float3 stretch)
{
    const float determinant = stretch.x * stretch.y - stretch.z * stretch.z;
    const float safeDet = max(determinant, 1e-5f);
    return float2(stretch.y * gradient.x - stretch.z * gradient.y,
                        stretch.x * gradient.y - stretch.z * gradient.x) / safeDet;
}
