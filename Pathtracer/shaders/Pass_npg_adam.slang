/******************************************************************************
 * Pass_npg_adam.slang — Neural Path Guiding: Adam optimizer weight update
 *
 * Runs after Pass_npg_train. Each thread updates one weight parameter using
 * the Adam optimizer with the accumulated gradients.
 *
 * Dispatch: ceil(NPG_TOTAL_PARAMS / 256) = ceil(3269/256) = 13 groups
 *****************************************************************************/
#define COMPUTE_PASS
#include "Includes_v8.slang"
#include "NPG_common.slang"

RWByteAddressBuffer g_npgWeights    : register(u24);
RWByteAddressBuffer g_npgGradients  : register(u25);
RWByteAddressBuffer g_npgAdamM      : register(u26);
RWByteAddressBuffer g_npgAdamV      : register(u27);
RWByteAddressBuffer g_npgCounters   : register(u28);

static const float ADAM_LR    = 0.002f;
static const float ADAM_BETA1 = 0.9f;
static const float ADAM_BETA2 = 0.999f;
static const float ADAM_EPS   = 1e-8f;

[numthreads(256, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    uint paramIdx = DTid.x;
    if (paramIdx >= NPG_TOTAL_PARAMS) return;

    uint sampleCount = g_npgCounters.Load(0);
    if (sampleCount == 0) return;

    // Normalize gradient by sample count
    float grad = asfloat(g_npgGradients.Load(paramIdx * 4)) / float(sampleCount);
    grad = clamp(grad, -NPG_GRAD_CLIP, NPG_GRAD_CLIP);

    // Read Adam state
    float m = asfloat(g_npgAdamM.Load(paramIdx * 4));
    float v = asfloat(g_npgAdamV.Load(paramIdx * 4));

    // Adam timestep
    uint adamT;
    if (paramIdx == 0)
    {
        g_npgCounters.InterlockedAdd(4, 1, adamT);
        adamT += 1;
    }
    else
    {
        adamT = g_npgCounters.Load(4) + 1;
    }

    // Update moments
    m = ADAM_BETA1 * m + (1.0f - ADAM_BETA1) * grad;
    v = ADAM_BETA2 * v + (1.0f - ADAM_BETA2) * grad * grad;

    // Bias correction
    float mHat = m / (1.0f - pow(ADAM_BETA1, float(adamT)));
    float vHat = v / (1.0f - pow(ADAM_BETA2, float(adamT)));

    // Update weight
    float w = asfloat(g_npgWeights.Load(paramIdx * 4));
    w -= ADAM_LR * mHat / (sqrt(vHat) + ADAM_EPS);

    // Write back
    g_npgWeights.Store(paramIdx * 4, asuint(w));
    g_npgAdamM.Store(paramIdx * 4, asuint(m));
    g_npgAdamV.Store(paramIdx * 4, asuint(v));

    // Clear gradient for next frame
    g_npgGradients.Store(paramIdx * 4, 0);

    // Clear sample counter (only thread 0)
    if (paramIdx == 0)
        g_npgCounters.Store(0, 0);
}
