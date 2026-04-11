/******************************************************************************
 * NPG_common.slang — Neural Path Guiding with NASG (Normalized Anisotropic
 *                     Spherical Gaussian) mixture model
 *
 * Based on: "Online Neural Path Guiding with Normalized Anisotropic Spherical
 * Gaussians" (Huang et al., 2024, ACM Trans. Graph.)
 *
 * ── Architecture ──────────────────────────────────────────────────────────
 * Density model: NASG mixture (6 components)
 *   - Closed-form normalization, direct sampling, anisotropic lobes
 *   - Each component: 6 params (direction, rotation, sharpness, eccentricity, weight)
 *
 * Network: position(one-blob 24) + wo_local(3) + roughness(1) + pad(4) → 6 NASG + sel.prob
 *   Input:  32 floats (one-blob encoded position + tangent-space features)
 *   Hidden: 2 layers × 32 neurons, ReLU, no bias (padding=1 for implicit bias)
 *   Output: 37 floats (6 components × 6 params + 1 selection probability)
 *
 * Tangent-space design (same as before):
 *   - Normal is implicit z-axis
 *   - wo and sampled directions are in tangent space
 *   - NASG lobes are in tangent space (z-axis always upper hemisphere)
 *   - Position is world-space, one-blob encoded for spatial variation
 *
 * Weight layout (no bias, RowMajor W[out×in]):
 *   L0: 32×32 = 1024   (offsets 0–1023)
 *   L1: 32×32 = 1024   (offsets 1024–2047)
 *   Out: 37×32 = 1184   (offsets 2048–3231)
 *   Total: 3232 floats = 12928 bytes
 *****************************************************************************/
#ifndef NPG_COMMON_SLANG
#define NPG_COMMON_SLANG

// ─── Constants ──────────────────────────────────────────────────────────────
static const float NPG_PI = 3.14159265358979f;

static const uint NPG_NUM_COMPONENTS = 6;
static const uint NPG_PARAMS_PER_COMP = 6;  // zx, zy, tau, lambda, a, weight
static const uint NPG_OUTPUT_DIM = NPG_NUM_COMPONENTS * NPG_PARAMS_PER_COMP + 1;  // 37
static const uint NPG_INPUT_DIM  = 32;
static const uint NPG_HIDDEN_DIM = 32;
static const uint NPG_POS_BINS   = 8;   // one-blob encoding bins per dimension
static const uint NPG_POS_ENC_DIM = NPG_POS_BINS * 3;  // 24

// Hyperparameters
static const float NPG_LR            = 0.002f;
static const float NPG_LAMBDA_MIN    = 0.5f;
static const float NPG_LAMBDA_MAX    = 30.0f;
static const float NPG_A_MAX         = 200.0f;
static const float NPG_UNIFORM_FLOOR = 0.05f;  // blend with cosine hemisphere
static const float NPG_GRAD_CLIP     = 1.0f;

// Weight layout (float indices)
static const uint NPG_L0_W  = 0;
static const uint NPG_L1_W  = NPG_L0_W + NPG_HIDDEN_DIM * NPG_INPUT_DIM;    // 1024
static const uint NPG_OUT_W = NPG_L1_W + NPG_HIDDEN_DIM * NPG_HIDDEN_DIM;   // 2048
static const uint NPG_OUT_B = NPG_OUT_W + NPG_OUTPUT_DIM * NPG_HIDDEN_DIM;  // 3232 (output bias)
static const uint NPG_TOTAL_PARAMS = NPG_OUT_B + NPG_OUTPUT_DIM;            // 3269
static const uint NPG_TOTAL_BYTES  = NPG_TOTAL_PARAMS * 4;                   // 13076

// ─── Weight access ──────────────────────────────────────────────────────────
float NPG_LoadW(RWByteAddressBuffer buf, uint idx)
{
    return asfloat(buf.Load(idx * 4));
}
void NPG_StoreW(RWByteAddressBuffer buf, uint idx, float val)
{
    buf.Store(idx * 4, asuint(val));
}

// ─── Tangent frame construction ─────────────────────────────────────────────
void NPG_BuildTBN(float3 N, out float3 T, out float3 B)
{
    float3 up = abs(N.z) < 0.999f ? float3(0, 0, 1) : float3(1, 0, 0);
    T = normalize(cross(up, N));
    B = cross(N, T);
}

float3 NPG_WorldToLocal(float3 w, float3 T, float3 B, float3 N)
{
    return float3(dot(w, T), dot(w, B), dot(w, N));
}

float3 NPG_LocalToWorld(float3 l, float3 T, float3 B, float3 N)
{
    return T * l.x + B * l.y + N * l.z;
}

// ─── One-blob positional encoding ───────────────────────────────────────────
// Maps a scalar x ∈ [0,1] to NPG_POS_BINS Gaussian blobs.
// Kernel: exp(-(B-1)² * (x - center_k)² / 2)
void NPG_OneBlobEncode1D(float x, uint offset, inout float enc[NPG_INPUT_DIM])
{
    float invB = 1.0f / float(NPG_POS_BINS - 1);
    [unroll] for (uint k = 0; k < NPG_POS_BINS; ++k)
    {
        float center = float(k) * invB;
        float diff = (x - center) * float(NPG_POS_BINS - 1);
        enc[offset + k] = exp(-0.5f * diff * diff);
    }
}

// Normalize world position to [0,1] via atan mapping
float NPG_NormalizePos(float x)
{
    return 0.5f + atan(x * 0.1f) / NPG_PI;
}

// ─── Build input vector ─────────────────────────────────────────────────────
// pos: world-space position (one-blob encoded for spatial variation)
// wo_local: outgoing direction in tangent space
// roughness: surface roughness
void NPG_BuildInput(float3 pos, float3 wo_local, float roughness,
                    out float input[NPG_INPUT_DIM])
{
    // One-blob encode normalized position (24 dims)
    NPG_OneBlobEncode1D(NPG_NormalizePos(pos.x), 0,  input);
    NPG_OneBlobEncode1D(NPG_NormalizePos(pos.y), 8,  input);
    NPG_OneBlobEncode1D(NPG_NormalizePos(pos.z), 16, input);

    // Tangent-space wo + roughness (4 dims)
    input[24] = wo_local.x;
    input[25] = wo_local.y;
    input[26] = wo_local.z;
    input[27] = roughness;

    // Padding with 1.0 for implicit bias (4 dims)
    input[28] = 1.0f;
    input[29] = 1.0f;
    input[30] = 1.0f;
    input[31] = 1.0f;
}

// ─── NASG component ─────────────────────────────────────────────────────────
struct NASGComponent
{
    float  weight;   // mixture weight (softmax normalized)
    float3 z;        // lobe axis (tangent space, upper hemisphere)
    float3 x;        // tangent direction (perpendicular to z)
    float3 y;        // bitangent (z × x... well, cross(z, x))
    float  lambda;   // sharpness
    float  a;        // eccentricity (0 = isotropic SG)
};

struct NASGMixture
{
    NASGComponent comps[NPG_NUM_COMPONENTS];
    float selectionProb;  // blend weight: selProb * learned + (1-selProb) * BSDF
};

// ─── NASG evaluation ────────────────────────────────────────────────────────
// G(v; z, x, λ, a) from Eq. 10 of the paper (with ε=0):
//   u = (v·z + 1) / 2
//   p = a * (v·x)² / (1 - (v·z)²)
//   G = exp(2λ(u^(1+p) - 1)) · u^p
//
// In log space: log(G) = 2λ(exp((1+p)·log(u)) - 1) + p·log(u)
float NASG_Evaluate(float3 v, float3 z, float3 x, float lambda, float a)
{
    float dotZ = dot(v, z);
    float dotX = dot(v, x);
    float sinZ2 = max(1.0f - dotZ * dotZ, 1e-10f);
    float cos2phi = dotX * dotX / sinZ2;

    float u = max((dotZ + 1.0f) * 0.5f, 1e-20f);
    float logU = log(u);
    float p = a * cos2phi;

    // log(G) = 2λ·exp((1+p)·logU) - 2λ + p·logU
    float h = (1.0f + p) * logU;           // log(u^(1+p))
    float up1 = exp(h);                     // u^(1+p)
    float logG = 2.0f * lambda * (up1 - 1.0f) + p * logU;

    return exp(logG);
}

// ─── NASG normalization constant ────────────────────────────────────────────
// K = 2π(1 - e^{-2λ}) / (λ · √(1+a))     [Eq. 12]
float NASG_NormK(float lambda, float a)
{
    float eminus2l = exp(-2.0f * lambda);
    return 2.0f * NPG_PI * (1.0f - eminus2l) / (lambda * sqrt(1.0f + a));
}

// ─── NASG normalized PDF ────────────────────────────────────────────────────
float NASG_PDF(float3 v, float3 z, float3 x, float lambda, float a)
{
    return NASG_Evaluate(v, z, x, lambda, a) / NASG_NormK(lambda, a);
}

// ─── NASG sampling ──────────────────────────────────────────────────────────
// Direct sampling via inverse CDF (Appendix C of the paper, with ε=0).
// Three uniform random numbers ξ₀, ξ₁, ξ₂ ∈ [0,1].
float3 NASG_Sample(float3 z, float3 x, float3 y,
                   float lambda, float a,
                   float xi0, float xi1, float xi2)
{
    float eminus2l = exp(-2.0f * lambda);

    // Map ξ₀ → s ∈ [e^{-2λ}, 1]
    float s = eminus2l + xi0 * (1.0f - eminus2l);

    // Map ξ₁ → ρ ∈ [-π/2, π/2]
    float rho = -NPG_PI * 0.5f + xi1 * NPG_PI;
    float cosRho = cos(rho);
    float sinRho = sin(rho);

    // Intermediate: t = (cosτ + 1) / 2 = log(s)/(2λ) + 1
    float t = clamp(log(max(s, 1e-30f)) / (2.0f * lambda) + 1.0f, 0.0f, 1.0f);

    // Exponent for the change of variable
    float sin2Rho = sinRho * sinRho;
    float exponent = (1.0f + a * sin2Rho) / (1.0f + a);

    // cosθ from the inverse mapping
    float cosTheta = 2.0f * pow(max(t, 1e-20f), exponent) - 1.0f;
    cosTheta = clamp(cosTheta, -1.0f, 1.0f);
    float sinTheta = sqrt(max(0.0f, 1.0f - cosTheta * cosTheta));

    // φ from ρ: φ = atan2(√(1+a)·sinρ, cosρ) for eastern hemisphere
    float sqrtA1 = sqrt(1.0f + a);
    float phi = atan2(sqrtA1 * sinRho, cosRho);
    if (xi2 <= 0.5f) phi += NPG_PI;  // western hemisphere

    // Direction in the NASG local frame [x, y, z]
    float3 v = x * (sinTheta * cos(phi))
             + y * (sinTheta * sin(phi))
             + z * cosTheta;
    return normalize(v);
}

// ─── Mixture PDF ────────────────────────────────────────────────────────────
float NASGMixture_PDF(NASGMixture mix, float3 v)
{
    float learned = 0.0f;
    [unroll] for (uint i = 0; i < NPG_NUM_COMPONENTS; ++i)
    {
        NASGComponent c = mix.comps[i];
        learned += c.weight * NASG_PDF(v, c.z, c.x, c.lambda, c.a);
    }
    // Blend with cosine-weighted hemisphere for robustness
    float cosHemi = max(v.z, 0.0f) / NPG_PI;  // cosine-weighted hemisphere PDF
    return (1.0f - NPG_UNIFORM_FLOOR) * learned + NPG_UNIFORM_FLOOR * cosHemi;
}

// ─── Mixture sampling ───────────────────────────────────────────────────────
// Returns direction in tangent space. Caller transforms to world via TBN.
float3 NASGMixture_Sample(NASGMixture mix, inout uint seed, out float pdf)
{
    float u = RandomFloatSingle(seed);

    // Cosine-weighted hemisphere fallback
    if (u < NPG_UNIFORM_FLOOR)
    {
        float u1 = RandomFloatSingle(seed);
        float u2 = RandomFloatSingle(seed);
        float r = sqrt(u1);
        float phi = 2.0f * NPG_PI * u2;
        float3 dir = float3(r * cos(phi), r * sin(phi), sqrt(max(0.0f, 1.0f - u1)));
        pdf = NASGMixture_PDF(mix, dir);
        return dir;
    }

    // Select component by weight
    float rem = (u - NPG_UNIFORM_FLOOR) / (1.0f - NPG_UNIFORM_FLOOR);
    float cumW = 0.0f;
    uint selected = NPG_NUM_COMPONENTS - 1;
    [unroll] for (uint i = 0; i < NPG_NUM_COMPONENTS; ++i)
    {
        cumW += mix.comps[i].weight;
        if (rem < cumW) { selected = i; break; }
    }

    // Sample from selected NASG component
    float xi0 = RandomFloatSingle(seed);
    float xi1 = RandomFloatSingle(seed);
    float xi2 = RandomFloatSingle(seed);
    NASGComponent c = mix.comps[selected];
    float3 dir = NASG_Sample(c.z, c.x, c.y, c.lambda, c.a, xi0, xi1, xi2);

    // Ensure upper hemisphere (tangent space)
    if (dir.z < 0.0f) dir.z = -dir.z;

    pdf = NASGMixture_PDF(mix, dir);
    return dir;
}

// ─── Construct x,y axes from z and rotation angle τ ─────────────────────────
void NASG_BuildFrame(float3 z, float tau, out float3 x, out float3 y)
{
    float3 up = abs(z.z) < 0.999f ? float3(0, 0, 1) : float3(1, 0, 0);
    float3 t = normalize(cross(up, z));
    float3 b = cross(z, t);
    x = t * cos(tau) + b * sin(tau);
    y = cross(z, x);
}

// ─── Decode raw network output → NASGMixture ────────────────────────────────
NASGMixture NPG_DecodeOutput(float raw[NPG_OUTPUT_DIM])
{
    NASGMixture mix;

    // Softmax for component weights
    float maxLogit = raw[5];  // weight logit is at offset 5 within each component
    [unroll] for (uint l = 1; l < NPG_NUM_COMPONENTS; ++l)
        maxLogit = max(maxLogit, raw[l * NPG_PARAMS_PER_COMP + 5]);
    float sumExp = 0.0f;
    float expW[NPG_NUM_COMPONENTS];
    [unroll] for (uint l2 = 0; l2 < NPG_NUM_COMPONENTS; ++l2)
    {
        expW[l2] = exp(raw[l2 * NPG_PARAMS_PER_COMP + 5] - maxLogit);
        sumExp += expW[l2];
    }

    [unroll] for (uint i = 0; i < NPG_NUM_COMPONENTS; ++i)
    {
        uint base = i * NPG_PARAMS_PER_COMP;

        // Weight
        mix.comps[i].weight = expW[i] / sumExp;

        // z-axis: normalize(tanh(zx), tanh(zy), 1) → upper hemisphere
        float tZx = tanh(raw[base + 0]);
        float tZy = tanh(raw[base + 1]);
        mix.comps[i].z = normalize(float3(tZx, tZy, 1.0f));

        // Rotation angle τ for x-axis
        float tau = raw[base + 2];  // unbounded, wraps naturally via cos/sin

        // Build orthonormal frame
        NASG_BuildFrame(mix.comps[i].z, tau, mix.comps[i].x, mix.comps[i].y);

        // λ (sharpness): exp → clamp
        mix.comps[i].lambda = clamp(exp(raw[base + 3]), NPG_LAMBDA_MIN, NPG_LAMBDA_MAX);

        // a (eccentricity): softplus → clamp
        float aRaw = raw[base + 4];
        mix.comps[i].a = clamp(log(1.0f + exp(aRaw)), 0.0f, NPG_A_MAX);
    }

    // Selection probability
    uint selIdx = NPG_NUM_COMPONENTS * NPG_PARAMS_PER_COMP;  // index 36
    mix.selectionProb = 1.0f / (1.0f + exp(-raw[selIdx]));

    return mix;
}

// ─── Forward pass (manual matmul, no bias) ──────────────────────────────────
NASGMixture NPG_Forward(float input[NPG_INPUT_DIM], RWByteAddressBuffer weights)
{
    // Layer 0: 32 → 32, ReLU
    float h0[NPG_HIDDEN_DIM];
    [unroll] for (uint j = 0; j < NPG_HIDDEN_DIM; ++j)
    {
        float sum = 0.0f;
        [unroll] for (uint i = 0; i < NPG_INPUT_DIM; ++i)
            sum += input[i] * NPG_LoadW(weights, NPG_L0_W + j * NPG_INPUT_DIM + i);
        h0[j] = max(0.0f, sum);
    }

    // Layer 1: 32 → 32, ReLU
    float h1[NPG_HIDDEN_DIM];
    [unroll] for (uint j1 = 0; j1 < NPG_HIDDEN_DIM; ++j1)
    {
        float sum = 0.0f;
        [unroll] for (uint i1 = 0; i1 < NPG_HIDDEN_DIM; ++i1)
            sum += h0[i1] * NPG_LoadW(weights, NPG_L1_W + j1 * NPG_HIDDEN_DIM + i1);
        h1[j1] = max(0.0f, sum);
    }

    // Output: 32 → 37 + bias
    float raw[NPG_OUTPUT_DIM];
    [unroll] for (uint jo = 0; jo < NPG_OUTPUT_DIM; ++jo)
    {
        float sum = NPG_LoadW(weights, NPG_OUT_B + jo);  // bias
        [unroll] for (uint io = 0; io < NPG_HIDDEN_DIM; ++io)
            sum += h1[io] * NPG_LoadW(weights, NPG_OUT_W + jo * NPG_HIDDEN_DIM + io);
        raw[jo] = sum;
    }

    return NPG_DecodeOutput(raw);
}

// ─── NASG log-gradient helpers ──────────────────────────────────────────────
// Compute ∂log(G/K)/∂λ, ∂log(G/K)/∂a, ∂log(G)/∂(v·z), ∂log(G)/∂(v·x)
// These are the analytical derivatives from the paper's formulation.
struct NASGGradients
{
    float dLambda;   // ∂log(pdf)/∂λ
    float dA;        // ∂log(pdf)/∂a
    float dDotZ;     // ∂log(G)/∂(v·z)  [K doesn't depend on orientation]
    float dDotX;     // ∂log(G)/∂(v·x)
};

NASGGradients NASG_LogGradients(float3 v, float3 z, float3 x, float lambda, float a)
{
    NASGGradients grad;

    float dotZ = dot(v, z);
    float dotX = dot(v, x);
    float sinZ2 = max(1.0f - dotZ * dotZ, 1e-10f);
    float cos2phi = dotX * dotX / sinZ2;

    float u = max((dotZ + 1.0f) * 0.5f, 1e-20f);
    float logU = log(u);
    float p = a * cos2phi;
    float h = (1.0f + p) * logU;
    float up1 = exp(h);  // u^(1+p)

    float eminus2l = exp(-2.0f * lambda);

    // ∂log(G)/∂λ = 2(u^{1+p} - 1)
    float dLogG_dLambda = 2.0f * (up1 - 1.0f);
    // ∂log(K)/∂λ = 2e^{-2λ}/(1-e^{-2λ}) - 1/λ
    float dLogK_dLambda = 2.0f * eminus2l / max(1.0f - eminus2l, 1e-10f) - 1.0f / lambda;
    grad.dLambda = dLogG_dLambda - dLogK_dLambda;

    // ∂log(G)/∂a = (2λ·u^{1+p} + 1) · log(u) · cos²φ
    float dLogG_dA = (2.0f * lambda * up1 + 1.0f) * logU * cos2phi;
    // ∂log(K)/∂a = -0.5/(1+a)
    float dLogK_dA = -0.5f / (1.0f + a);
    grad.dA = dLogG_dA - dLogK_dA;

    // ∂log(G)/∂(v·x):
    //   = (2λ·u^{1+p} + 1) · log(u) · 2a·dotX / sinZ²
    grad.dDotX = (2.0f * lambda * up1 + 1.0f) * logU * 2.0f * a * dotX / sinZ2;

    // ∂log(G)/∂(v·z):
    float dlogU = 1.0f / (dotZ + 1.0f + 1e-10f);
    float dpddZ = 2.0f * a * dotX * dotX * dotZ / (sinZ2 * sinZ2);
    float dh_ddotZ = (1.0f + p) * dlogU + logU * dpddZ;
    grad.dDotZ = 2.0f * lambda * up1 * dh_ddotZ + logU * dpddZ + p * dlogU;

    return grad;
}

#endif // NPG_COMMON_SLANG
