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
    const float pc = saturate(materials[mID].Pr_Pm_Ps_Pc.w);
    if (pc <= 0.0f) return 0.0.xxx;

    // Microfacet terms (GGX)
    float3 H    = normalize(V + L);
    float  NdotH = max(0.0f, dot(N, H));
    float  VdotH = max(0.0f, dot(V, H));

    float rough = saturate(materials[mID].Pcr_aniso_anisor.x);
    float alpha = max(EPSILON, rough * rough);

    float  D = D_GGX(NdotH, alpha);
    float  G = G2_SmithGGX(NdotV, NdotL, alpha);
    float  denom = max(4.0f * NdotV * NdotL, EPSILON);
    float3 F = FresnelDielectricTIR(V, H, etai, etat);

    float3 spec = pc * (F * D * G) / denom;
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

    float  Pr   = materials[mID].Pcr_aniso_anisor.x;
    float  Fo = FresnelDielectric( V, N, etat, etai).x * (1.0f - Pr * 0.7f) * (1.0f - Pr * 0.7f); // wo going from etai -> etat
    float  Fi = FresnelDielectric( L, N, etai, etat).x; // wi coming from etai -> etat

    float pc = saturate(materials[mID].Pr_Pm_Ps_Pc.w);

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

    float  pc = saturate(materials[mID].Pr_Pm_Ps_Pc.w);
    if (pc <= 0.0f || NdotV <= 0.0f) return 0.0f;

    float Fv = FresnelDielectric(V, N, etat, etai).x;
    return saturate(pc * Fv);
}

inline float3 SampleBRDF_COAT(
    uint    mID,
    float3  outgoing,
    float3  normal,
    float3  flatNormal,
    inout uint2  seed)
{
    float3 N = normalize(normal);
    float3 V = normalize(outgoing);

    float rough = saturate(materials[mID].Pcr_aniso_anisor.x);
    float alpha = max(EPSILON, rough * rough);

    // Visible normal sampling
    float3 H = SampleVNDF_H(alpha, V, N, seed);

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

    float rough = saturate(materials[mID].Pcr_aniso_anisor.x);
    float alpha = max(EPSILON, rough * rough);

    // p(H) for VNDF mapped to reflection: p(wi) = p(H) / (4 V·H)
    float pdf_H = (D_GGX(NdotH, alpha) * G1_SmithGGX(NdotV, alpha) * VdotH) / NdotV;
    return pdf_H / (4.0f * VdotH);
}
