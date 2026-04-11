/******************************************************************************
 * Pass_npg_train.slang — Neural Path Guiding: batch training pass
 *
 * Collects training samples from the GI reservoir, runs forward+backward
 * through the MLP, and atomically accumulates gradients into the gradient
 * buffer. The Adam optimizer (Pass_npg_adam) applies the update afterward.
 *
 * This pass is an ALTERNATIVE to the inline Hogwild! SGD in raygen.
 * Enable one or the other, not both.
 *
 * Dispatch: ceil(NPG_SAMPLES_PER_FRAME / 64) groups of 64 threads
 *****************************************************************************/
#define COMPUTE_PASS
#include "Includes_v8.slang"
#include "NPG_common.slang"

RWByteAddressBuffer g_npgWeights    : register(u24);
RWByteAddressBuffer g_npgGradients  : register(u25);
RWByteAddressBuffer g_npgAdamM      : register(u26);
RWByteAddressBuffer g_npgAdamV      : register(u27);
RWByteAddressBuffer g_npgCounters   : register(u28);

static const uint NPG_SAMPLES_PER_FRAME = 4096;

// CAS-based atomic float add
void NPG_AtomicAddGrad(RWByteAddressBuffer buf, uint idx, float val)
{
    uint byteAddr = idx * 4;
    uint expected;
    buf.InterlockedCompareExchange(byteAddr, 0, 0, expected);
    [allow_uav_condition]
    for (uint iter = 0; iter < 64; ++iter)
    {
        uint desired = asuint(asfloat(expected) + val);
        uint actual;
        buf.InterlockedCompareExchange(byteAddr, expected, desired, actual);
        if (actual == expected) break;
        expected = actual;
    }
}

[numthreads(64, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    if (DTid.x >= NPG_SAMPLES_PER_FRAME) return;

    uint2 imgSize = uint2(gImageWidth, gImageHeight);
    uint totalPixels = imgSize.x * imgSize.y;

    // Hash-based random pixel selection
    uint seed = DTid.x * 1664525u + (uint)time * 1013904223u + 12345u;
    seed ^= seed >> 16;
    seed *= 0x45d9f3bu;
    seed ^= seed >> 16;
    uint pixelLinear = seed % totalPixels;
    uint2 pixel = uint2(pixelLinear % imgSize.x, pixelLinear / imgSize.x);
    uint pixelIdx = MapPixelID(imgSize, pixel);

    if (load_isEmitter(g_sample_current, pixelIdx)) return;
    uint instID = load_instID(g_sample_current, pixelIdx);
    if (instID == 0xFFFFFFFFu) return;

    // Reconstruct surface
    uint primID = load_primID(g_sample_current, pixelIdx);
    float2 bary = load_bary(g_sample_current, pixelIdx);
    float3 camPos = mul(viewI, float4(0, 0, 0, 1)).xyz;
    float3 surfPos = ReconstructPosition(instID, primID, bary);
    float3 surfNormal = load_n1_s_with_instID(g_sample_current, pixelIdx, instID);
    float3 wo = normalize(camPos - surfPos);

    // GI reservoir target
    Reservoir_GI rgi = loadReservoirGI(g_Reservoirs_current_gi, pixelIdx);
    float3 giDir = normalize(rgi.x2_gi - surfPos);
    float3 giRadiance = rgi.L2_gi;
    float giW = rgi.W_gi;
    float radWeight = Luma(giRadiance) * giW;
    if (radWeight < 1e-6f || any(isnan(giDir)) || any(isnan(giRadiance))) return;
    radWeight = min(radWeight, 2.0f);

    // Build tangent frame
    float3 T, B;
    NPG_BuildTBN(surfNormal, T, B);
    float3 wo_local = NPG_WorldToLocal(wo, T, B, surfNormal);
    float3 dir_local = NPG_WorldToLocal(giDir, T, B, surfNormal);
    if (dir_local.z < 0.01f) return;
    dir_local = normalize(dir_local);

    // Build input
    float npgInput[NPG_INPUT_DIM];
    NPG_BuildInput(surfPos, wo_local, 0.5f, npgInput);

    // Forward pass with stored activations
    float h0_pre[NPG_HIDDEN_DIM], h0[NPG_HIDDEN_DIM];
    float h1_pre[NPG_HIDDEN_DIM], h1[NPG_HIDDEN_DIM];
    float rawOut[NPG_OUTPUT_DIM];

    [unroll] for (uint j = 0; j < NPG_HIDDEN_DIM; ++j)
    {
        float sum = 0.0f;
        [unroll] for (uint i = 0; i < NPG_INPUT_DIM; ++i)
            sum += npgInput[i] * NPG_LoadW(g_npgWeights, NPG_L0_W + j * NPG_INPUT_DIM + i);
        h0_pre[j] = sum;
        h0[j] = max(0.0f, sum);
    }
    [unroll] for (uint j1 = 0; j1 < NPG_HIDDEN_DIM; ++j1)
    {
        float sum = 0.0f;
        [unroll] for (uint i1 = 0; i1 < NPG_HIDDEN_DIM; ++i1)
            sum += h0[i1] * NPG_LoadW(g_npgWeights, NPG_L1_W + j1 * NPG_HIDDEN_DIM + i1);
        h1_pre[j1] = sum;
        h1[j1] = max(0.0f, sum);
    }
    [unroll] for (uint jo = 0; jo < NPG_OUTPUT_DIM; ++jo)
    {
        float sum = NPG_LoadW(g_npgWeights, NPG_OUT_B + jo);  // bias
        [unroll] for (uint io = 0; io < NPG_HIDDEN_DIM; ++io)
            sum += h1[io] * NPG_LoadW(g_npgWeights, NPG_OUT_W + jo * NPG_HIDDEN_DIM + io);
        rawOut[jo] = sum;
    }

    [unroll] for (uint nc = 0; nc < NPG_OUTPUT_DIM; ++nc)
        if (isnan(rawOut[nc]) || isinf(rawOut[nc])) return;

    NASGMixture mix = NPG_DecodeOutput(rawOut);

    // Compute mixture PDF
    float pLearned = 0.0f;
    float compPdf[NPG_NUM_COMPONENTS];
    [unroll] for (uint ci = 0; ci < NPG_NUM_COMPONENTS; ++ci)
    {
        NASGComponent c = mix.comps[ci];
        compPdf[ci] = NASG_PDF(dir_local, c.z, c.x, c.lambda, c.a);
        pLearned += c.weight * compPdf[ci];
    }
    float cosHemi = max(dir_local.z, 0.0f) / NPG_PI;
    float pMix = (1.0f - NPG_UNIFORM_FLOOR) * pLearned + NPG_UNIFORM_FLOOR * cosHemi;
    if (pMix < 1e-8f) return;

    float dLdP = -radWeight / pMix;
    float sc = dLdP * (1.0f - NPG_UNIFORM_FLOOR);

    float dRaw[NPG_OUTPUT_DIM];
    [unroll] for (uint z2 = 0; z2 < NPG_OUTPUT_DIM; ++z2) dRaw[z2] = 0.0f;

    [unroll] for (uint ci2 = 0; ci2 < NPG_NUM_COMPONENTS; ++ci2)
    {
        uint base = ci2 * NPG_PARAMS_PER_COMP;
        NASGComponent c = mix.comps[ci2];
        float w_c = c.weight;
        float pdf_c = compPdf[ci2];

        dRaw[base + 5] = sc * w_c * (pdf_c - pLearned);

        NASGGradients ng = NASG_LogGradients(dir_local, c.z, c.x, c.lambda, c.a);

        float tZx = tanh(rawOut[base + 0]);
        float tZy = tanh(rawOut[base + 1]);
        float len = sqrt(tZx * tZx + tZy * tZy + 1.0f);
        float invLen = 1.0f / len;
        float dotZ_v = dot(dir_local, c.z);
        float dDotZ_dtZx = (dir_local.x - dotZ_v * c.z.x) * invLen;
        float dDotZ_dtZy = (dir_local.y - dotZ_v * c.z.y) * invLen;
        float dtZx_draw = 1.0f - tZx * tZx;
        float dtZy_draw = 1.0f - tZy * tZy;
        dRaw[base + 0] = sc * w_c * pdf_c * ng.dDotZ * dDotZ_dtZx * dtZx_draw;
        dRaw[base + 1] = sc * w_c * pdf_c * ng.dDotZ * dDotZ_dtZy * dtZy_draw;

        float dDotX_dTau = dot(dir_local, c.y);
        dRaw[base + 2] = sc * w_c * pdf_c * ng.dDotX * dDotX_dTau;

        float lambdaUnc = exp(rawOut[base + 3]);
        float lambdaMask = (lambdaUnc >= NPG_LAMBDA_MIN && lambdaUnc <= NPG_LAMBDA_MAX) ? 1.0f : 0.0f;
        dRaw[base + 3] = sc * w_c * pdf_c * ng.dLambda * c.lambda * lambdaMask;

        float sigA = 1.0f / (1.0f + exp(-rawOut[base + 4]));
        float aUnc = log(1.0f + exp(rawOut[base + 4]));
        float aMask = (aUnc >= 0.0f && aUnc <= NPG_A_MAX) ? 1.0f : 0.0f;
        dRaw[base + 4] = sc * w_c * pdf_c * ng.dA * sigA * aMask;
    }

    [unroll] for (uint gc = 0; gc < NPG_OUTPUT_DIM; ++gc)
        dRaw[gc] = clamp(dRaw[gc], -NPG_GRAD_CLIP, NPG_GRAD_CLIP);

    // Backprop through layers and accumulate gradients atomically
    float dH1[NPG_HIDDEN_DIM];
    [unroll] for (uint jj = 0; jj < NPG_HIDDEN_DIM; ++jj) dH1[jj] = 0.0f;

    [unroll] for (uint jo2 = 0; jo2 < NPG_OUTPUT_DIM; ++jo2)
    {
        float g = dRaw[jo2];
        if (abs(g) < 1e-12f) continue;
        [unroll] for (uint io2 = 0; io2 < NPG_HIDDEN_DIM; ++io2)
        {
            NPG_AtomicAddGrad(g_npgGradients, NPG_OUT_W + jo2 * NPG_HIDDEN_DIM + io2, g * h1[io2]);
            dH1[io2] += g * NPG_LoadW(g_npgWeights, NPG_OUT_W + jo2 * NPG_HIDDEN_DIM + io2);
        }
    }

    [unroll] for (uint r1 = 0; r1 < NPG_HIDDEN_DIM; ++r1)
    {
        dH1[r1] *= (h1_pre[r1] > 0.0f) ? 1.0f : 0.0f;
        dH1[r1] = clamp(dH1[r1], -NPG_GRAD_CLIP, NPG_GRAD_CLIP);
    }

    float dH0[NPG_HIDDEN_DIM];
    [unroll] for (uint jj0 = 0; jj0 < NPG_HIDDEN_DIM; ++jj0) dH0[jj0] = 0.0f;

    [unroll] for (uint j1b = 0; j1b < NPG_HIDDEN_DIM; ++j1b)
    {
        float g = dH1[j1b];
        if (abs(g) < 1e-12f) continue;
        [unroll] for (uint i1b = 0; i1b < NPG_HIDDEN_DIM; ++i1b)
        {
            NPG_AtomicAddGrad(g_npgGradients, NPG_L1_W + j1b * NPG_HIDDEN_DIM + i1b, g * h0[i1b]);
            dH0[i1b] += g * NPG_LoadW(g_npgWeights, NPG_L1_W + j1b * NPG_HIDDEN_DIM + i1b);
        }
    }

    [unroll] for (uint r0 = 0; r0 < NPG_HIDDEN_DIM; ++r0)
    {
        dH0[r0] *= (h0_pre[r0] > 0.0f) ? 1.0f : 0.0f;
        dH0[r0] = clamp(dH0[r0], -NPG_GRAD_CLIP, NPG_GRAD_CLIP);
    }

    [unroll] for (uint j0b = 0; j0b < NPG_HIDDEN_DIM; ++j0b)
    {
        float g = dH0[j0b];
        if (abs(g) < 1e-12f) continue;
        [unroll] for (uint i0b = 0; i0b < NPG_INPUT_DIM; ++i0b)
            NPG_AtomicAddGrad(g_npgGradients, NPG_L0_W + j0b * NPG_INPUT_DIM + i0b, g * npgInput[i0b]);
    }

    // Bias gradients (∂output/∂bias = 1, so gradient = dRaw directly)
    [unroll] for (uint jb = 0; jb < NPG_OUTPUT_DIM; ++jb)
    {
        if (abs(dRaw[jb]) > 1e-12f)
            NPG_AtomicAddGrad(g_npgGradients, NPG_OUT_B + jb, dRaw[jb]);
    }

    uint dummy;
    g_npgCounters.InterlockedAdd(0, 1, dummy);
}
