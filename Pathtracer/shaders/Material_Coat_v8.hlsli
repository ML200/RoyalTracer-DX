inline float3 EvaluateBRDF_COAT(
    uint   mID,
    float3 normal,
    float3 incoming,
    float3 outgoing,
    float  etai,
    float  etat)
{
    float3 N = normalize(normal);
    float3 V = normalize(outgoing);
    float3 L = normalize(-incoming);

    float NdotV = max(0.0f, dot(N, V));
    float NdotL = max(0.0f, dot(N, L));

    // Coat strength
    const float pc = saturate(LoadPc(mID));
    if (pc <= 0.0f) return 0.0.xxx;

    // Microfacet terms (GGX)
    float3 H    = normalize(V + L);
    float  NdotH = max(0.0f, dot(N, H));
    float  VdotH = max(0.0f, dot(V, H));

    float rough = saturate(LoadPcr(mID));
    float alpha = max(EPSILON, rough * rough);

    float  D = D_GGX(NdotH, alpha);
    float  G = G2_SmithGGX(NdotV, NdotL, alpha);
    float  denom = max(4.0f * NdotV * NdotL, EPSILON);
    float3 F = FresnelDielectricTIR(V, H, etai, etat);

    float3 spec = pc * (F * D * G) / denom;

    float Pr  = LoadPcr(mID);
    float Ess = GetEssLUT(Pr, NdotV);
    float kms = (1.0f - Ess) / max(Ess, 1e-6f);

    spec = spec * (1.0f + F * kms);

    return (any(isnan(spec)) || any(isinf(spec))) ? 0.0.xxx : spec;
}

// Pair-aware: T = (1 - pc * F(wo)) * (1 - pc * F(wi))
inline float Transmittance_COAT(
    uint   mID,
    float3 normal,
    float3 incoming,
    float3 outgoing,
    float  etai,
    float  etat)
{
    float3 N = normalize(normal);
    float3 V = normalize(outgoing);
    float3 L = normalize(-incoming);

    float NdotV = max(0.0f, dot(N, V));
    float NdotL = max(0.0f, dot(N, L));
    if (NdotV <= 0.0f || NdotL <= 0.0f) return 1.0f;

    float  Pr   = LoadPcr(mID);
    float  Fo = FresnelDielectric( V, N, etat, etai).x * (1.0f - Pr * 0.7f) * (1.0f - Pr * 0.7f); // wo going from etai -> etat
    float  Fi = FresnelDielectric( L, N, etai, etat).x; // wi coming from etai -> etat

    float pc = saturate(LoadPc(mID));

    float  T_in  = saturate(1.0f - pc * Fi);
    float  T_out = saturate(1.0f - pc * Fo);
    return T_in * T_out;
}

// Probability to choose the COAT reflection branch
inline float Sampling_Weight_COAT(
    uint   mID,
    float3 normal,
    float3 outgoing,
    float  etai,
    float  etat)
{
    float3 N = normalize(normal);
    float3 V = normalize(outgoing);
    float  NdotV = max(0.0f, dot(N, V));

    float  pc = saturate(LoadPc(mID));
    if (pc <= 0.0f || NdotV <= 0.0f) return 0.0f;

    float Fv = FresnelDielectric(V, N, etai, etat).x;
    return saturate(pc * Fv);
}

// Fused coat eval, pdf, and transmittance
struct CoatResult {
    float3 f;
    float  pdf;
    float  t;
};

inline CoatResult EvalCoatAll(
    uint matID, float3 N, float3 V, float3 L,
    float etai, float etat)
{
    CoatResult r;
    r.f = 0.0f;
    r.pdf = 0.0f;
    r.t = 1.0f;

    float NdotV = max(0.0f, dot(N, V));
    float NdotL = max(0.0f, dot(N, L));

    float pc = saturate(LoadPc(matID));
    if (pc <= 0.0f) return r;

    float Pr_coat = LoadPcr(matID);

    //Transmittance
    if (NdotV > 0.0f && NdotL > 0.0f)
    {
        float Fo = FresnelDielectric(V, N, etat, etai).x * (1.0f - Pr_coat * 0.7f) * (1.0f - Pr_coat * 0.7f);
        float Fi = FresnelDielectric(L, N, etai, etat).x;
        r.t = saturate(1.0f - pc * Fi) * saturate(1.0f - pc * Fo);
    }

    // Eval + PDF
    if (NdotV <= 0.0f || NdotL <= 0.0f) return r;

    float3 H     = normalize(V + L);
    float  NdotH = max(0.0f, dot(N, H));
    float  VdotH = max(EPSILON, dot(V, H));

    float rough = saturate(Pr_coat);
    float alpha = max(EPSILON, rough * rough);

    float D   = D_GGX(NdotH, alpha);
    float G1V = G1_SmithGGX(NdotV, alpha);

    // Eval
    {
        float  G2    = G1V * G1_SmithGGX(NdotL, alpha);
        float  denom = max(4.0f * NdotV * NdotL, EPSILON);
        float3 F     = FresnelDielectricTIR(V, H, etai, etat);

        float3 spec = pc * (F * D * G2) / denom;

        float Ess = GetEssLUT(Pr_coat, NdotV);
        float kms = (1.0f - Ess) / max(Ess, 1e-6f);
        spec = spec * (1.0f + F * kms);

        r.f = (any(isnan(spec)) || any(isinf(spec))) ? 0.0.xxx : spec;
    }

    // PDF: p(wi) = D * G1V / (4 * NdotV)  (VdotH cancels in VNDF reflection Jacobian)
    r.pdf = (D * G1V) / (4.0f * NdotV);

    return r;
}

inline float3 SampleBRDF_COAT(
    uint    mID,
    float3  outgoing,
    float3  normal,
    float3  flatNormal,
    inout uint  seed)
{
    float3 N = normalize(normal);
    float3 V = normalize(outgoing);

    float rough = saturate(LoadPcr(mID));
    float alpha = max(EPSILON, rough * rough);

    // Visible normal sampling; perfect reflection for very smooth coats
    float3 H;
    if (rough < SMOOTH_SPECULAR_THRESHOLD)
        H = N;
    else
        H = SampleVNDF_H(alpha, V, N, seed);

    // Reflect
    float3 L = reflect(-V, H);
    if (dot(N, L) <= 0.0f) { return 0.0.xxx; }

    return normalize(L);
}

inline float BRDF_PDF_COAT(
    uint   mID,
    float3 N,
    float3 wi,
    float3 wo,
    float  etai,
    float  etat)
{
    if (dot(N, -wi) < 0.0f) return 0.0f;

    float3 V = normalize(wo);
    float3 L = normalize(-wi);

    float NdotV = max(EPSILON, dot(N, V));
    float NdotL = max(EPSILON, dot(N, L));
    if (NdotV <= 0.0f || NdotL <= 0.0f) return 0.0f;

    float3 H     = normalize(V + L);
    float  VdotH = max(EPSILON, dot(V, H));
    float  NdotH = max(EPSILON, dot(N, H));

    float rough = saturate(LoadPcr(mID));
    float alpha = max(EPSILON, rough * rough);

    // p(H) for VNDF mapped to reflection: p(wi) = p(H) / (4 V·H)
    float pdf_H = (D_GGX(NdotH, alpha) * G1_SmithGGX(NdotV, alpha) * VdotH) / NdotV;
    return pdf_H / (4.0f * VdotH);
}
