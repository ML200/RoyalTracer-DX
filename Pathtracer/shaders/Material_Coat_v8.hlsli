//====================================
//COAT BRDF EVALUATION
//====================================
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

    const float pc = saturate(LoadPc(mID));
    if (pc <= 0.0f) return 0.0.xxx;

    //GGX microfacet
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

//====================================
//COAT TRANSMITTANCE
//====================================
//pair-aware, T = (1 - pc*F(wo)) * (1 - pc*F(wi))
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
    float  Fo = FresnelDielectric( V, N, etat, etai).x * (1.0f - Pr * 0.7f) * (1.0f - Pr * 0.7f);
    float  Fi = FresnelDielectric( L, N, etai, etat).x;

    float pc = saturate(LoadPc(mID));

    float  T_in  = saturate(1.0f - pc * Fi);
    float  T_out = saturate(1.0f - pc * Fo);
    return T_in * T_out;
}

//====================================
//COAT SAMPLING WEIGHT
//====================================
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

//====================================
//FUSED COAT EVAL PDF TRANSMITTANCE
//====================================
struct CoatResult {
    float3 f;
    float  pdf;
    float  t;
};

inline CoatResult EvalCoatAll(
    uint matID, float3 N, float3 V, float3 L,
    half etai, half etat)
{
    CoatResult r;
    r.f = 0.0f;
    r.pdf = 0.0f;
    r.t = 1.0f;

    //NdotV/NdotL kept float, used as denominators with EPSILON-class floors
    const float NdotV = max(0.0f, dot(N, V));
    const float NdotL = max(0.0f, dot(N, L));

    const half pc = (half)saturate(LoadPc(matID));
    if (pc <= (half)0.0) return r;

    const half Pr_coat = (half)LoadPcr(matID);

    //transmittance
    if (NdotV > 0.0f && NdotL > 0.0f)
    {
        const half PrFactor = (half)1.0 - Pr_coat * (half)0.7;
        const half Fo = (half)FresnelDielectric(V, N, etat, etai).x * PrFactor * PrFactor;
        const half Fi = (half)FresnelDielectric(L, N, etai, etat).x;
        r.t = (float)(saturate((half)1.0 - pc * Fi) * saturate((half)1.0 - pc * Fo));
    }

    //eval + pdf
    if (NdotV <= 0.0f || NdotL <= 0.0f) return r;

    float3 H     = normalize(V + L);
    const half NdotH = (half)max(0.0f, dot(N, H));
    //VdotH stays float, divides spec via 4*VdotH
    const float VdotH = max(EPSILON, dot(V, H));

    const half rough = saturate(Pr_coat);
    //fp16 normal min is ~6.1e-5; 1e-4 is the lowest safe alpha clamp here
    const half alpha = max((half)1e-4, rough * rough);

    //D and G can blow past fp16 range for smooth coat, keep float
    const float D   = D_GGX(NdotH, alpha);
    const float G1V = G1_SmithGGX(NdotV, alpha);

    {
        const float  G2    = G1V * G1_SmithGGX(NdotL, alpha);
        const float  denom = max(4.0f * NdotV * NdotL, EPSILON);
        const float3 F     = FresnelDielectricTIR(V, H, etai, etat);

        float3 spec = (float)pc * (F * D * G2) / denom;

        const float Ess = GetEssLUT((float)Pr_coat, NdotV);
        const float kms = (1.0f - Ess) / max(Ess, 1e-6f);
        spec = spec * (1.0f + F * kms);

        r.f = (any(isnan(spec)) || any(isinf(spec))) ? 0.0.xxx : spec;
    }

    //p(wi) = D*G1V/(4*NdotV), VdotH cancels in VNDF reflection Jacobian
    r.pdf = (D * G1V) / (4.0f * NdotV);

    return r;
}

//====================================
//COAT SAMPLING
//====================================
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

    //VNDF, force perfect reflection for very smooth coats
    float3 H;
    if (rough < SMOOTH_SPECULAR_THRESHOLD)
        H = N;
    else
        H = SampleVNDF_H(alpha, V, N, seed);

    float3 L = reflect(-V, H);
    if (dot(N, L) <= 0.0f) { return 0.0.xxx; }

    return normalize(L);
}

//====================================
//COAT PDF
//====================================
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

    //VNDF -> reflection, p(wi) = p(H)/(4*V.H)
    float pdf_H = (D_GGX(NdotH, alpha) * G1_SmithGGX(NdotV, alpha) * VdotH) / NdotV;
    return pdf_H / (4.0f * VdotH);
}
