#pragma once
#include "OceanLayout.h"

// Sharp gravity-wave crests need horizontal displacement, but centimetre ripples must not
// dominate the global no-fold bound and flatten every larger crest. This rolls off horizontal
// chop below 0.5 m while preserving vertical displacement and its normal-map derivatives.
float OceanChopGain(float k) {
    const float ratio = k / 12.56637061436f;
    return rcp(1.0f + ratio * ratio * ratio * ratio);
}

// Full inverse transpose of the horizontal displacement map, including its cross derivative.
float2 OceanWarpedSlope(float2 gradient, float3 stretch)
{
    const float determinant = stretch.x * stretch.y - stretch.z * stretch.z;
    const float safeDet = max(determinant, 1e-5f);
    return float2(stretch.y * gradient.x - stretch.z * gradient.y,
                        stretch.x * gradient.y - stretch.z * gradient.x) / safeDet;
}
