/*inline float3 EvaluateBTDF_GGXT(uint mID, float3 normal, float3 incoming, float3 outgoing, float etai, float etat)
{
    float3 N = normalize(normal);
    float3 V = normalize(outgoing);
    float3 L = normalize(-incoming);

    float NdotV = dot(N, V);
    float NdotL = dot(N, L);

    // Transmission only if directions are on opposite sides
    if (NdotV * NdotL >= 0.0f) return 0.0.xxx;

    float r     = materials[mID].Pr_Pm_Ps_Pc.x;
    float alpha = max(EPSILON, r * r);

    // Half vector for transmission (Walter 2007)
    float3 H = normalize(etai * V + etat * L);
    if (dot(N, H) < 0.0f) H = -H;

    float NdotH = max(0.0f, dot(N, H));
    float VdotH = dot(V, H);
    float LdotH = dot(L, H);

    // Exact dielectric Fresnel at the microfacet
    float  F = FresnelDielectricTIR(V, H, etai, etat);

    // GGX terms
    float  D = D_GGX(NdotH, alpha);
    float  G = G2_SmithGGX(abs(NdotV), abs(NdotL), alpha);

    // Walter 2007 BTDF (Eq. 21 form) for rough refraction
    // ft = (1 - F) * D * G * |(V·H)(L·H)| * η_t^2 / (|N·V||N·L| * (η_i(V·H) + η_t(L·H))^2)
    float  denom = (etai * VdotH + etat * LdotH);
    float  denom2 = max(denom * denom, EPSILON);
    float  scale  = (1.0f - F) * D * G * (etat * etat)
                    * abs(VdotH * LdotH)
                    / (max(abs(NdotV * NdotL), EPSILON) * denom2);

    float3 ft = scale.xxx;
    //TODO: tint/absorptio
    return (any(isnan(ft)) || any(isinf(ft))) ? 0.0.xxx : ft;
}


// This is the top dog :D Functions here for completeness
inline float Transmittance_GGXT(uint mID, float3 normal, float3 incoming, float3 outgoing, float etai, float etat){
    return 1.0f;
}
inline float Sampling_Weight_GGXT(uint mID, float3 normal, float3 outgoing, float etai, float etat){
    return 1.0f;
}


inline void SampleBTDF_GGXT(
    uint    mID,
    float3  outgoing,
    float3  normal,
    float3  flatNormal,
    inout float3 sample,
    inout uint2 seed)
{
    float r     = materials[mID].Pr_Pm_Ps_Pc.x;
    float alpha = r * r;

    float3 V = normalize(outgoing);
    float3 N = normalize(normal);

    // 1. Sample the microfacet normal
    float3 H  = SampleVNDF_H(alpha, V, N, seed);
    if (dot(V, H) < 0.0f) H = -H; // Ensure H faces V

    // 2. Refract across it
    float ior   = materials[mID].Ni;
    float NdotV = dot(N, V);
    bool  entering = (NdotV > 0.0f);
    float etaI = entering ? 1.0f : ior;
    float etaT = entering ? ior : 1.0f;
    float eta  = etaI / etaT;

    float3 L;
    if (!RefractVector(V, H, eta, L))
    {
        // TIR
        sample = 0.0.xxx;
        return;
    }

    // Must lie on the other hemisphere
    if (dot(N, L) * NdotV >= 0.0f)
    {
        sample = 0.0.xxx;
        return;
    }

    sample = normalize(L);
}


inline float BTDF_PDF_GGXT(uint mID, float3 normal, float3 incoming, float3 outgoing)
{
    float3 V = normalize(outgoing);
    float3 L = normalize(-incoming);

    float NdotV = dot(normal, V);
    float NdotL = dot(normal, L);

    if (NdotV * NdotL >= 0.0f) return 0.0f;

    float r     = materials[mID].Pr_Pm_Ps_Pc.x;
    float alpha = max(EPSILON, r * r);

    float ior     = materials[mID].Ni;
    bool  entering = (NdotV > 0.0f);
    float etaI = entering ? 1.0f : ior;
    float etaT = entering ? ior : 1.0f;

    // Transmission half-vector
    float3 H = normalize(etaI * V + etaT * L);
    if (dot(normal, H) < 0.0f) H = -H;

    float VdotH = dot(V, H);
    float LdotH = dot(L, H);

    // If the refracted ray goes "backwards" through the microfacet,
    // the geometry is invalid for this model.
    if (VdotH * LdotH > 0.0f) return 0.0f;

    float NdotH = max(0.0f, dot(normal, H));

    // PDF of sampling the visible microfacet normal H
    float  D     = D_GGX_TRANS(NdotH, alpha);
    float  G1    = G1_SmithGGX_TRANS(abs(NdotV), alpha);
    float  pdf_m = D * G1 * abs(VdotH) / max(abs(NdotV), EPSILON);

    // Jacobian for the change of variables from H to L
    float denom_jac = (etaI * VdotH + etaT * LdotH);
    float dwh_dwi = abs((etaT * etaT * LdotH) / max(denom_jac * denom_jac, EPSILON));

    float pdf = pdf_m * dwh_dwi;
    return (isnan(pdf) || isinf(pdf)) ? 0.0f : pdf;
}*/