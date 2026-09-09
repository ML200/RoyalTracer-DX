#ifndef TEMPORAL_REUSE_MATH_V8_HLSLI
#define TEMPORAL_REUSE_MATH_V8_HLSLI

// Legacy ReSTIR: duplicates collapse confidence toward one;
// direction samples and the diagnostic bypass keep the configured cap.
uint TemporalConfidenceCap(uint maxM, float duplication, float exponent, bool exempt)
{
    const uint cap = max(1u, maxM);
    const float effective = exempt ? (float)cap
        : lerp((float)cap, 1.0f, pow(saturate(duplication), exponent));
    return (uint)clamp(round(effective), 1.0f, (float)cap);
}

void ApplyPermutationSampling(inout int2 prevPixelPos, uint uniformRandomNumber)
{
    int2 offset = int2(uniformRandomNumber & 3, (uniformRandomNumber >> 2) & 3);
    prevPixelPos += offset;

    prevPixelPos.x ^= 3;
    prevPixelPos.y ^= 3;

    prevPixelPos -= offset;
}

#endif
