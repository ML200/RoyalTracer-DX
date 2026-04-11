/******************************************************************************
 * Pass_npg_init.slang — Neural Path Guiding: NASG weight initialization
 *
 * He initialization for ReLU layers, small Xavier for output weights,
 * explicit bias initialization for diverse component directions.
 *
 * Dispatch: ceil(3269 / 256) = 13 groups of 256 threads
 *****************************************************************************/
#define COMPUTE_PASS
#include "Includes_v8.slang"
#include "NPG_common.slang"

RWByteAddressBuffer g_npgWeights : register(u24);

// Simple hash-based RNG for initialization
float InitRandom(uint idx, uint seed)
{
    uint h = idx * 1664525u + seed * 1013904223u;
    h ^= h >> 16;
    h *= 0x45d9f3bu;
    h ^= h >> 16;
    h *= 0x45d9f3bu;
    h ^= h >> 16;
    return float(h) / float(0xFFFFFFFFu) * 2.0f - 1.0f; // [-1, 1]
}

[numthreads(256, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    uint idx = DTid.x;
    if (idx >= NPG_TOTAL_PARAMS) return;

    float val = 0.0f;

    if (idx < NPG_L1_W) // Layer 0 weights: He init (fan_in = NPG_INPUT_DIM)
    {
        val = InitRandom(idx, 42u) * sqrt(2.0f / float(NPG_INPUT_DIM));
    }
    else if (idx < NPG_OUT_W) // Layer 1 weights: He init (fan_in = NPG_HIDDEN_DIM)
    {
        val = InitRandom(idx, 42u) * sqrt(2.0f / float(NPG_HIDDEN_DIM));
    }
    else if (idx < NPG_OUT_B) // Output weights: small Xavier init
    {
        val = InitRandom(idx, 42u) * sqrt(1.0f / float(NPG_HIDDEN_DIM)) * 0.1f;
    }
    else // Output bias: explicit per-parameter initialization
    {
        uint biasIdx = idx - NPG_OUT_B;  // 0..36
        uint comp = biasIdx / NPG_PARAMS_PER_COMP;
        uint param = biasIdx % NPG_PARAMS_PER_COMP;

        if (comp < NPG_NUM_COMPONENTS)
        {
            if (param == 0) // zx: spread lobes across hemisphere
            {
                // 6 directions evenly spread in tangent plane
                static const float zx_init[6] = { 0.0f, 0.8f, -0.8f, 0.4f, -0.4f, 0.0f };
                val = zx_init[comp];
            }
            else if (param == 1) // zy: spread lobes across hemisphere
            {
                static const float zy_init[6] = { 0.0f, 0.0f, 0.0f, 0.7f, -0.7f, 0.0f };
                val = zy_init[comp];
            }
            else if (param == 2) // tau: different rotation angles
            {
                val = float(comp) * 1.047f;  // ~60° apart
            }
            else if (param == 3) // lambda: moderate sharpness (exp(0) = 1)
            {
                // Start with low sharpness so lobes cover wide area
                val = 0.0f;  // exp(0) = 1.0, clamped to max(0.5, 1.0) = 1.0
            }
            else if (param == 4) // a: mild anisotropy
            {
                val = 0.0f;  // softplus(0) = ln2 ≈ 0.69
            }
            else if (param == 5) // weight logit: equal weights → softmax(0) = 1/6
            {
                val = 0.0f;
            }
        }
        else // Selection probability (biasIdx == 36)
        {
            val = -2.0f;  // sigmoid(-2) ≈ 0.12 → start mostly BSDF
        }
    }

    g_npgWeights.Store(idx * 4, asuint(val));
}
