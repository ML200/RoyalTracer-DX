// Sheen lobe constants (Charlie NDF)
static const float  SHEEN_R      = 0.20f;
static const float3 SHEEN_COLOR  = float3(1,1,1);

// Precomputed constants derived from SHEEN_R (all compile-time)
static const float SHEEN_INVR    = 1.0f / SHEEN_R;                          // 5.0
static const float SHEEN_D_SCALE = (2.0f + SHEEN_INVR) * (0.5f * INV_PI);  // 7/(2pi)
static const float SHEEN_SAMP_EXPO = (2.0f * SHEEN_R) / (2.0f * SHEEN_R + 1.0f); // 2/7

// Lambda fit params: P = (1-r)^2 * P0 + (1-(1-r)^2) * P1
static const float SHEEN_W0 = (1.0f - SHEEN_R) * (1.0f - SHEEN_R);         // 0.64
static const float SHEEN_W1 = 1.0f - SHEEN_W0;                             // 0.36
static const float SHEEN_FIT_A = SHEEN_W0 * 25.3245f + SHEEN_W1 * 21.5473f;
static const float SHEEN_FIT_B = SHEEN_W0 *  3.32435f + SHEEN_W1 *  3.82987f;
static const float SHEEN_FIT_C = SHEEN_W0 *  0.16801f + SHEEN_W1 *  0.19823f;
static const float SHEEN_FIT_D = SHEEN_W0 * (-1.27393f) + SHEEN_W1 * (-1.97760f);
static const float SHEEN_FIT_E = SHEEN_W0 * (-4.85967f) + SHEEN_W1 * (-4.32054f);

// D term (Eq. 2): D(m) = (2 + 1/r) * sin(theta_h)^(1/r) / (2pi)
inline float SHEEN_D_Charlie(float NdotH)
{
    float sinTh2 = saturate(1.0f - NdotH * NdotH);
    float sinTh  = sqrt(sinTh2);
    return SHEEN_D_SCALE * pow(max(1e-8f, sinTh), SHEEN_INVR);
}

inline float SHEEN_L_eval(float x)
{
    return SHEEN_FIT_A / (1.0f + SHEEN_FIT_B * pow(max(1e-4f, x), SHEEN_FIT_C))
         + SHEEN_FIT_D * x + SHEEN_FIT_E;
}

// Precomputed L(0.5) — used in every Lambda_Charlie call
static const float SHEEN_L_HALF = SHEEN_FIT_A / (1.0f + SHEEN_FIT_B * pow(0.5f, SHEEN_FIT_C))
                                + SHEEN_FIT_D * 0.5f + SHEEN_FIT_E;

inline float SHEEN_Lambda_Charlie(float cosTheta)
{
    float x = saturate(cosTheta);

    float Lx    = SHEEN_L_eval(x);
    float L_1mx = SHEEN_L_eval(1.0f - x);

    // piecewise definition
    return (x < 0.5f) ? exp(Lx)
                      : exp(2.0f * SHEEN_L_HALF - L_1mx);
}

inline float SHEEN_G_Charlie(float NdotV, float NdotL)
{
    float lambdaV = SHEEN_Lambda_Charlie(NdotV);
    float lambdaL = SHEEN_Lambda_Charlie(NdotL);
    return 1.0f / (1.0f + lambdaV + lambdaL); // correlated Smith
}

// Half-vector sampling for Charlie: sample m ~ D(m) * (N·m)
// Derivation gives sin^2(theta_h) = u^(2r/(2r+1))
inline float3 SHEEN_SampleHalfVector(uint seed, float3 N, out float NdotH, out float pdf_H)
{
    float u1  = RandomFloatSingle(seed);
    float u2  = RandomFloatSingle(seed);

    float sin2Th   = pow(saturate(u1), SHEEN_SAMP_EXPO);
    float cosTh    = sqrt(saturate(1.0f - sin2Th));
    float sinTh    = sqrt(sin2Th);
    float phi      = 2.0f * PI * u2;

    float3 T, B;
    CoordinateSystem(N, T, B);
    float3 H = sinTh * cos(phi) * T + sinTh * sin(phi) * B + cosTh * N;
    H = normalize(H);

    NdotH = saturate(dot(N, H));

    // p_H(m) = D(m) * (N·m)
    pdf_H = SHEEN_D_Charlie(NdotH) * NdotH;
    return H;
}

// BRDF eval: f = w * color * F * G * D / (4 N·V N·L); For sheen we take F~1
inline float3 EvaluateBRDF_SHEEN(
    uint   mID,
    float3 normal,
    float3 incoming,
    float3 outgoing)
{
    float3 N = normalize(normal);
    float3 V = normalize(outgoing);
    float3 L = normalize(-incoming);

    // Sheen weight
    float w = saturate(materials[mID].Pr_Pm_Ps_Pc.z);
    if (w <= 0.0f) return 0.0.xxx;

    float NdotV = max(0.0f, dot(N, V));
    float NdotL = max(0.0f, dot(N, L));
    if (NdotV <= 0.0f || NdotL <= 0.0f) return 0.0.xxx;

    // Half-vector and microfacet terms
    float3 H    = normalize(V + L);
    float  NdotH = max(0.0f, dot(N, H));

    float  D = SHEEN_D_Charlie(NdotH);
    float  G = SHEEN_G_Charlie(NdotV, NdotL);
    float  denom = max(1e-6f, 4.0f * NdotV * NdotL);

    // F~1, SHEEN_COLOR=white — both multiply to identity
    return w * (G * D / denom);
}


inline float Transmittance_SHEEN(
    uint   mID,
    float3 normal,
    float3 incoming,
    float3 outgoing)
{
    float3 N = normalize(normal);
    float3 V = normalize(outgoing);
    float NdotV = dot(N, V);

    if (NdotV <= 0.0f)
        return 1.0f;

    float w = saturate(materials[mID].Pr_Pm_Ps_Pc.z);
    if (w <= 0.0f)
        return 1.0f;

    float aV = GetSheenLUT(SHEEN_R, NdotV);
    // SHEEN_COLOR=white → Luma=1, tintLum=1
    return saturate(1.0f - w * aV);
}

inline float Sampling_Weight_SHEEN(
    uint   mID,
    float3 normal,
    float3 outgoing)
{
    float3 N = normalize(normal);
    float3 V = normalize(outgoing);
    float  NdotV = max(0.0f, dot(N, V));

    float  w = saturate(materials[mID].Pr_Pm_Ps_Pc.z);
    if (w <= 0.0f || NdotV <= 0.0f) return 0.0f;

    float  aV = GetSheenLUT(SHEEN_R, NdotV);
    // SHEEN_COLOR=white → Luma=1, tintLum=1
    return saturate(w * aV);
}


inline float3 SampleBRDF_SHEEN(
    uint    mID,
    float3  outgoing,       // wo
    float3  normal,
    float3  flatNormal,
    inout uint seed)
{
    float3 N = normalize(normal);
    float3 V = normalize(outgoing);

    float NdotV = max(0.0f, dot(N, V));
    if (NdotV <= 0.0f) { return 0.0f; }

    float NdotH, pdfH;
    float3 H = SHEEN_SampleHalfVector(seed, N, NdotH, pdfH);

    // reflect view about H to get incident direction
    float3 wi = reflect(-V, H);
    if (dot(N, wi) <= 0.0f) { return 0.0f; }

    return wi;
}


inline float BRDF_PDF_SHEEN(uint mID, float3 N, float3 wi, float3 wo)
{
    float3 V = normalize(wo);
    float3 L = normalize(-wi);
    float NdotL = max(0.0f, dot(N, L));
    if (NdotL <= 0.0f) return 0.0f;

    float3 H    = normalize(V + L);
    float NdotH = max(0.0f, dot(N, H));
    float VoH   = max(1e-6f, dot(V, H));

    // p(wi) = p(H) / (4 |V·H|), with p(H) = D(H) * (N·H)
    return SHEEN_D_Charlie(NdotH) * NdotH / (4.0f * VoH);
}