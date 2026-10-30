

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

inline float CoatTransmittance(
    uint matID, float3 N, float3 V, float3 L,
    half etai, half etat)
{
    const float NdotV = max(0.0f, dot(N, V));
    const float NdotL = max(0.0f, dot(N, L));
    const half pc = (half)saturate(LoadPc(matID));
    if (pc <= (half)0.0) return 1.0f;
    float t = 1.0f;
    [branch] if (NdotV > 0.0f && NdotL > 0.0f)
    {
        t = 1.0f - (float)pc * GGXDirectionalReflectance(LoadPcr(matID), NdotV, etai, etat, true);
    }
    return t;
}
struct CoatResult {
    float3 f;
    float  pdf;
    float  t;
};

inline CoatResult EvalCoatAll(
    uint matID, float3 N, float3 V, float3 L,
    half etai, half etat, bool needTransmission = true)
{
    CoatResult r;
    r.f = 0.0f;
    r.pdf = 0.0f;
    r.t = 1.0f;

    const float NdotV = max(0.0f, dot(N, V));
    const float NdotL = max(0.0f, dot(N, L));

    const half pc = (half)saturate(LoadPc(matID));
    if (pc <= (half)0.0) return r;

    const float Pr_coat = LoadPcr(matID);

    [branch] if (needTransmission && NdotV > 0.0f && NdotL > 0.0f)
    {
        r.t = 1.0f - (float)pc * GGXDirectionalReflectance(Pr_coat, NdotV, etai, etat, true);
    }

    if (NdotV <= 0.0f || NdotL <= 0.0f) return r;

    float3 H     = normalize(V + L);

    const float rough = saturate(Pr_coat);

    const float alpha = max(EPSILON, rough * rough);

    const float D   = D_GGX(N, H, alpha);
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

    float3 H;
    if (rough < SMOOTH_SPECULAR_THRESHOLD)
        H = N;
    else
        H = SampleVNDF_H(alpha, V, N, seed);

    float3 L = reflect(-V, H);
    if (dot(N, L) <= 0.0f) { return 0.0.xxx; }

    return normalize(L);
}

